from xml.etree import ElementTree as ET
from pathlib import Path
import shutil
import functools
from typing import Generic, Optional, Callable, Any, TypeVar, TypedDict
import ffmpeg


BIN_DIR = Path('/opt/homebrew/bin/')
FFMPEG_BIN = str(BIN_DIR / 'ffmpeg')
FFPROBE_BIN = str(BIN_DIR / 'ffprobe')

T = TypeVar('T')


class FFMPEGObject(Generic[T]):
    def __init__(
        self,
        delay: float,
        size: tuple[int, int],
        fps: int,
        rate: float,
        log_level: Optional[str] = None
    ):
        self.delay = delay
        self.size = size
        self.fps = fps
        self.rate = rate
        self.pts: str = f'{1./rate:.2f}*PTS' if self.rate != 1. else ''
        self.dar: str = f'{size[0]/size[1]:.2f}'
        self.image_framerate: float = 1000 / delay
        self.output_kwds = {}
        self.run_args: list[str] = ['-hide_banner', '-nostats']
        if log_level is not None:
            self.run_args.extend(['-loglevel', log_level.lower()])
        self.streams: list[ffmpeg.Stream] = []
        self.streams_used: dict[ffmpeg.Stream, int] = {}
        self.total_duration: float = 0.

    @staticmethod
    def probe(path: Path | str) -> dict[str, Any]:
        path = str(path)
        return ffmpeg.probe(path, cmd=FFPROBE_BIN)

    @staticmethod
    def _auto_split_stream(
        stream: ffmpeg.Stream,
        streams_used: dict[ffmpeg.Stream, int],
    ) -> ffmpeg.Stream:
        i = streams_used.get(stream, 0)
        streams_used[stream] = i + 1
        return stream.split()[i]

    @staticmethod
    def _get_media_duration(path: str) -> float:
        try:
            return float(FFMPEGObject.probe(path)['format']['duration'])
        except ffmpeg.Error as e:
            print(e.stderr)
            raise e

    @staticmethod
    def _apply_standard_filters(
        stream: ffmpeg.Stream,
        media_type: str,
        size: tuple[int, int],
        fps: int,
        pts: str = '',
        dar: str = ''
    ) -> ffmpeg.Stream:
        stream = (
            stream.filter('scale', *size, force_original_aspect_ratio='decrease')
                  .filter('pad', *size, '(ow-iw)/2')
                  .filter('setsar', '1')
                  .filter('fps', fps)
                  .filter('setdar', dar)
        )
        if pts:
            stream = stream.filter('setpts', pts)
        # if media_type == 'image':
        #     stream = stream.filter('setdar', dar)
        return stream

    def create_stream(self, media: ET.Element) -> tuple[ffmpeg.Stream, float]:
        if media.tag == 'overlay':
            overlays = map(self.create_stream, media)
            stream, duration = next(overlays)
            for overlay_stream, _ in overlays:
                stream = stream.overlay(overlay_stream)
        else:
            media_path: str = media.get('path')
            match media_type := media.get('media_type'):
                case 'image':
                    duration = self.image_framerate
                    stream = ffmpeg.input(media_path, framerate=self.image_framerate)
                case 'animation':
                    loop = int(media.get('repeat', 1))
                    duration = self._get_media_duration(media_path) * loop
                    stream = (
                        ffmpeg.input(media_path)
                              .filter('loop', loop=loop - 1, size=32767, start=0)
                        )
                case 'video':
                    loop = int(media.get('repeat', 1))
                    duration = self._get_media_duration(media_path) * loop
                    stream = ffmpeg.input(media_path, stream_loop=loop - 1)
                case _:
                    raise IOError(f'UNKNOWN STREAM: {media_type}({media_path})')
            stream = self._apply_standard_filters(
                stream, media_type, self.size, self.fps, self.pts, self.dar
            )
            stream = self._auto_split_stream(stream, self.streams_used)
        return stream, duration

    def add_stream(self, media: ET.Element) -> None:
        stream, duration = self.create_stream(media)
        self.streams.append(stream)
        self.total_duration += duration

    def output_stream(self, fname: str) -> ffmpeg.Stream:
        stream = (
            ffmpeg.concat(*self.streams)
                  .output(fname, **self.output_kwds)
                  .global_args(*self.run_args)
        )
        return stream

    @staticmethod
    def _compile_call(
        fname: str,
        outstream: ffmpeg.Stream,
        temp_fname: Optional[str] = None,
        callback: Optional[Callable[[str], T]] = None
    ) -> T | None:
        _fname = temp_fname if temp_fname is not None else fname
        outstream.run(cmd=FFMPEG_BIN, overwrite_output=True)
        if temp_fname is not None:
            shutil.move(_fname, fname)
        if callback is not None:
            return callback(fname)

    @staticmethod
    def callback(fname: str) -> Any:
        ...

    def compile_call(
        self,
        path: Path,
        temp_path: Optional[Path] = None
    ) -> Callable[[], T | None]:
        _path = temp_path if temp_path is not None else path
        fname = str(path)
        temp_fname = str(temp_path) if temp_path is not None else None
        outstream = self.output_stream(str(_path))
        callback = self.callback
        f = functools.partial(self._compile_call, fname, outstream, temp_fname, callback)
        return f

    def run(self, path: Path, temp_path: Optional[Path] = None) -> Any:
        return self.compile_call(path, temp_path)()

    def reset(self) -> None:
        self.streams.clear()
        self.streams_used.clear()
        self.total_duration = 0


class VideoMeta(TypedDict):
    tag: str
    content: str


class FFMPEGObjectLive(FFMPEGObject):
    def __init__(
        self,
        delay: float,
        size: tuple[int, int],
        fps: int = 30,
        rate: float = 1.,
        log_level: Optional[str] = None
    ):
        super().__init__(delay, size, fps, rate, log_level)
        # self.run_args.extend(['-loglevel', 'error'])
        self.output_kwds.update(
            vcodec='libx264',
            format='mpegts',
            pix_fmt='yuv420p',
            fps_mode='auto',
            preset='veryfast',
            tune='zerolatency'
        )

    @staticmethod
    def callback(fname: str) -> VideoMeta:
        return {'tag': 'video', 'content': fname}


class FFMPEGObjectOutput(FFMPEGObject):
    def __init__(
        self,
        delay: float,
        size: tuple[int, int],
        fps: int,
        rate: float,
        log_level: Optional[str] = None
    ):
        super().__init__(delay, size, fps, rate, log_level)
        # self.run_args.extend(['-loglevel', 'info'])
        self.output_kwds.update(
            format='mp4',
            pix_fmt='yuv420p',
            fps_mode='auto',
            codec='libx264',
        )
