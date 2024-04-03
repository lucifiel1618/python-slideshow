import subprocess
from xml.etree import ElementTree as ET
from pathlib import Path
import shutil
import functools
from typing import Generic, Iterable, Literal, Optional, Callable, Any, TypeVar, TypedDict, cast
import ffmpeg

from . import utils


BIN_DIR = Path('/opt/homebrew/bin/')
FFMPEG_BIN = str(BIN_DIR / 'ffmpeg')
FFPROBE_BIN = str(BIN_DIR / 'ffprobe')

T = TypeVar('T')
LogLevel = Literal['debug', 'info', 'warning', 'error', 'quiet']


class FFMPEGObject(Generic[T]):
    DEFAULT_RUN_ARGS = ['-hide_banner', '-nostats']
    DEFAULT_LOGLEVL = 'info'

    def __init__(
        self,
        delay: float,
        size: tuple[int, int],
        fps: int,
        rate: float,
        loglevel: Optional[LogLevel] | bool = None
    ):
        self._delay = delay
        self._size = size
        self.fps = fps
        self._rate = rate
        self.pts: str = f'{1. / rate:.2f}*PTS' if rate != 1. else ''
        self.dar: str = f'{size[0] / size[1]:.2f}'
        self.image_framerate: float = 1000 / delay
        self.output_kwds = {}
        self._run_args: list[str] = self.DEFAULT_RUN_ARGS
        self._loglevel = loglevel
        self.streams: list[ffmpeg.Stream] = []
        self.streams_used: dict[ffmpeg.Stream, int] = {}
        self.total_duration: float = 0.
        self._dryrun = False
        self.logger = utils.get_logger(name=f'SlideShow.{self.__class__.__name__}')
        self.logger.info(f'{__class__.__name__} initialized.')

    @property
    def delay(self) -> float:
        return self._delay

    @delay.setter
    def delay(self, delay: float) -> None:
        self._delay = delay
        self.image_framerate = 1000 / delay

    @property
    def size(self) -> tuple[int, int]:
        return self._size
    
    @size.setter
    def size(self, size: tuple[int, int]) -> None:
        self._size = size
        self.dar = f'{size[0] / size[1]:.2f}'

    @property
    def rate(self) -> float:
        return self._rate
    
    @rate.setter
    def rate(self, rate: float):
        self._rate = rate
        self.pts = f'{1. / rate:.2f}*PTS' if rate != 1. else ''

    def get_run_args(self) -> list[str]:
        run_args = self._run_args[:]
        loglevel = self.loglevel
        if loglevel is not None:
            if loglevel is not False:
                run_args.extend(['-loglevel', loglevel])
        return run_args

    def set_run_args(self, value: Optional[Iterable[str] | bool] = None):
        if value is None or value is True:
            value = self.DEFAULT_RUN_ARGS
        elif value is False:
            value = []
        self._run_args[:] = value

    run_args = property(get_run_args, set_run_args)

    def get_loglevel(self) -> LogLevel | None:
        loglevel = self._loglevel
        if loglevel is None:
            return None
        if loglevel is False:
            return 'quiet'
        if loglevel is True:
            return self.DEFAULT_LOGLEVL
        return loglevel.lower()

    def set_loglevel(self, value: Optional[LogLevel | bool]):
        self._loglevel = value

    loglevel = property(get_loglevel, set_loglevel)

    def get_dryrun(self) -> bool:
        return self._dryrun

    def set_dryrun(self, value: bool):
        self._dryrun = value
        if self._dryrun is True:
            self.output_kwds['format'] = 'null'

    dryrun = property(get_dryrun, set_dryrun)

    @staticmethod
    def print_stream_cmd(stream: ffmpeg.Stream, logf: str | Path = Path('info.txt')):
        if isinstance(logf, str):
            logf = Path(logf)
        with logf.open('a') as logf_of:
            subprocess.check_call(stream.compile(), stderr=logf_of)

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
        return stream.split()[i]  # type: ignore

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
            stream.filter('scale', *size, force_original_aspect_ratio='decrease')  # type: ignore
                  .filter('pad', *size, '(ow-iw)/2')
                  .filter('setsar', '1')
                  .filter('fps', fps)
                  .filter('setdar', dar)
        )
        if pts:
            stream = stream.filter('setpts', pts)  # type: ignore
        # if media_type == 'image':
        #     stream = stream.filter('setdar', dar)
        return stream

    def create_stream(self, media: ET.Element) -> tuple[ffmpeg.Stream, float]:
        if media.tag == 'overlay':
            overlays = map(self.create_stream, media)
            stream, duration = next(overlays)
            for overlay_stream, _ in overlays:
                stream = stream.overlay(overlay_stream)  # type: ignore
        else:
            media_path: str = cast(str, media.get('path'))
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

    def get_stream(self) -> ffmpeg.Stream:
        stream: ffmpeg.Stream = ffmpeg.concat(*self.streams)
        return stream

    def output_stream(self, fname: str) -> ffmpeg.Stream:
        stream: ffmpeg.Stream = self.get_stream().output(fname, **self.output_kwds).global_args(*self.run_args)
        return stream

    def print_dryrun(self) -> None:
        self.print_stream_cmd(self.output_stream('-'))

    @staticmethod
    def _compile_call(
        fname: str,
        outstream: ffmpeg.Stream,
        temp_fname: Optional[str] = None,
        callback: Optional[Callable[[str], T]] = None
    ) -> T | None:
        _fname = temp_fname if temp_fname is not None else fname
        outstream.run(cmd=FFMPEG_BIN, overwrite_output=True)  # type: ignore
        if temp_fname is not None:
            shutil.move(_fname, fname)
        if callback is not None:
            return callback(fname)

    @staticmethod
    def callback(fname: str) -> T:
        ...

    def compile_call(
        self,
        path: Path,
        temp_path: Optional[Path] = None
    ) -> Callable[[], T]:
        _path = temp_path if temp_path is not None else path
        fname = str(path)
        temp_fname = str(temp_path) if temp_path is not None else None
        outstream = self.output_stream(str(_path))
        callback = self.callback
        self.logger.detail(f'Prepare {fname}')
        f: Callable[[], T] = functools.partial(
            self._compile_call, fname, outstream, temp_fname, callback
        )  # type: ignore
        return f

    def run(self, path: Path, temp_path: Optional[Path] = None, dryrun: Optional[bool] = None) -> T | None:
        if dryrun is None:
            dryrun = self.dryrun
        if self.dryrun:
            return self.print_dryrun()
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
        loglevel: Optional[LogLevel] = None
    ):
        super().__init__(delay, size, fps, rate, loglevel)
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
        meta: VideoMeta = {'tag': 'video', 'content': fname}
        return meta


class FFMPEGObjectOutput(FFMPEGObject):
    def __init__(
        self,
        delay: float,
        size: tuple[int, int],
        fps: int,
        rate: float,
        loglevel: Optional[LogLevel] = None
    ):
        super().__init__(delay, size, fps, rate, loglevel)
        # self.run_args.extend(['-loglevel', 'debug'])
        self.output_kwds.update(
            format='mp4',
            pix_fmt='yuv420p',
            fps_mode='auto',
            codec='libx264',
        )
