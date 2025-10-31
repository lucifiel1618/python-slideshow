import shlex
import subprocess
from xml.etree import ElementTree as ET
from pathlib import Path
import shutil
import functools
from typing import Iterable, Literal, Optional, Callable, Any, Sequence, TypedDict, cast
import ffmpeg

from slideshow import Subtitle

from . import utils


SUBTITLER = Subtitle.NullSubtitle


def find_executable(executable_name: str) -> str:
    executable_name, _, version = executable_name.partition('@')
    try:
        # Try `brew`
        d = subprocess.check_output(
            ['/opt/homebrew/bin/brew', '--prefix', 'ffmpeg' if not version else f'ffmpeg@{version}']
        ).strip().decode()
        return f'{d}/bin/{executable_name}'
    except subprocess.CalledProcessError:
        try:
            # Try `which`
            return subprocess.check_output(['which', executable_name]).strip().decode()
        except subprocess.CalledProcessError:
            return f'/opt/homebrew/bin/{executable_name}'


FFMPEG_BIN = find_executable('ffmpeg@5')
FFPROBE_BIN = find_executable('ffprobe@5')


LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'QUIET']


class FFMPEGObject[T]:
    DEFAULT_RUN_ARGS = ['-hide_banner', '-nostats']
    DEFAULT_LOGLEVL: LogLevel = 'INFO'

    def __init__(
        self,
        delay: float,
        size: tuple[int, int],
        fps: int,
        rate: float,
        loglevel: Optional[LogLevel | bool] = None
    ):
        self._delay = delay
        self._size = size
        self.fps = fps
        self._rate = rate
        self._start_pts: int = 0
        self.pts: str = f'{1. / rate:.2f}*PTS' if rate != 1. else ''
        self.dar: str = f'{size[0] / size[1]:.2f}'
        self.image_framerate: float = 1000 / delay
        self.output_kwds = {}
        self._run_args: list[str] = self.DEFAULT_RUN_ARGS
        self._loglevel = loglevel
        self.streams: list[ffmpeg.nodes.FilterableStream] = []
        self.streams_used: dict[ffmpeg.nodes.FilterableStream, int] = {}
        self.subtitler = SUBTITLER(*size)
        self.total_duration: float = 0.
        self.total_duration_ts: float = 0.
        self._dryrun = False
        self.logger = utils.get_logger(name=f'SlideShow.{self.__class__.__name__}')
        self.logger.info(f'{__class__.__name__} initialized.')

    @property
    def start_pts(self) -> int:
        return self._start_pts

    @start_pts.setter
    def start_pts(self, start_pts: int) -> None:
        self._start_pts = start_pts

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
        if rate == 1.:
            self.pts = ''
        else:
            self.pts = f'{1. / rate:.2f}*PTS'

    def get_run_args(self) -> list[str]:
        run_args = self._run_args[:]
        loglevel = self.loglevel
        if loglevel is not None:
            if loglevel is not False:
                run_args.extend(['-loglevel', loglevel.lower()])
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
            return 'QUIET'
        if loglevel is True:
            return self.DEFAULT_LOGLEVL
        return loglevel.upper()  # pyright: ignore[reportReturnType]

    def set_loglevel(self, value: Optional[LogLevel | bool]):
        self._loglevel: bool | None | LogLevel = value

    loglevel = property(get_loglevel, set_loglevel)

    def get_dryrun(self) -> bool:
        return self._dryrun

    def set_dryrun(self, value: bool):
        self._dryrun = value
        if self._dryrun is True:
            self.output_kwds['format'] = 'null'

    dryrun = property(get_dryrun, set_dryrun)

    @staticmethod
    def print_stream_cmd(stream: ffmpeg.nodes.FilterableStream, logf: str | Path = Path('info.txt')):
        if isinstance(logf, str):
            logf = Path(logf)
        with logf.open('a') as logf_of:
            subprocess.check_call(stream.compile(), stderr=logf_of)  # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def probe(path: Path | str) -> dict[str, Any]:
        path = str(path)
        return ffmpeg.probe(path, cmd=FFPROBE_BIN)

    @staticmethod
    def _auto_split_stream(
        stream: ffmpeg.nodes.FilterableStream,
        streams_used: dict[ffmpeg.nodes.FilterableStream, int],
    ) -> ffmpeg.nodes.FilterableStream:
        i = streams_used.get(stream, 0)
        streams_used[stream] = i + 1
        return stream.split()[i]  # type: ignore

    @staticmethod
    @functools.lru_cache(maxsize=1000)
    def __get_media_duration(
        path: str,
        scales: Sequence[Literal['sec', 'ts']] = ('sec', 'ts')
    ) -> tuple[float, ...]:
        '''
        About pts rounding method, see: https://ffmpeg.org/ffmpeg-filters.html#fps
        '''
        try:
            d = FFMPEGObject.probe(path)
            return tuple(
                float(d['format'].get('duration', 0)) if scale != 'ts' else
                next(stream.get('duration_ts', 0) for stream in d['streams'] if stream['codec_type'] == 'video')
                for scale in scales
            )
        except ffmpeg.Error as e:
            raise ValueError(e.stderr.decode())
        except Exception as e:
            raise e

    @staticmethod
    def _get_media_duration(
        path: str,
        scales: Sequence[Literal['sec', 'ts']] = ('sec', 'ts'),
        rate: float = 1.0,
        loop: int = 1
    ) -> tuple[float, ...]:
        return tuple(t * loop / rate for t in FFMPEGObject.__get_media_duration(path, scales))

    @staticmethod
    def _apply_standard_filters(
        stream: ffmpeg.nodes.FilterableStream,
        media_type: str,
        size: tuple[int, int],
        fps: int,
        pts: str = '',
        dar: str = '',
        *,
        autoscale: bool = True
    ) -> ffmpeg.nodes.FilterableStream:
        if autoscale:
            stream = (
                stream.filter('scale', *size, force_original_aspect_ratio='decrease')  # type: ignore
                      .filter('pad', *size, '(ow-iw)/2', '(oh-ih)/2')
            )
        stream = (
            stream.filter('setsar', '1')  # type: ignore
                  .filter('fps', fps)
                  .filter('setdar', dar)
        )
        if pts:
            stream = stream.filter('setpts', pts)  # type: ignore
        # if media_type == 'image':
        #     stream = stream.filter('setdar', dar)
        return stream

    def create_stream(
        self, media: ET.Element, autoscale: bool = True
    ) -> tuple[ffmpeg.nodes.FilterableStream, tuple[float, ...]]:
        if media.tag == 'overlay':
            stream, durations = self.overlay_stream(media)
            if autoscale:
                stream = (
                    stream.filter('scale', *self.size, force_original_aspect_ratio='decrease')  # type: ignore
                          .filter('pad', *self.size, '(ow-iw)/2', '(oh-ih)/2')
                )
                stream = (
                    stream.filter('setsar', '1')  # type: ignore
                          .filter('fps', self.fps)
                          .filter('setdar', self.dar)
                )
        else:
            media_path: str = cast(str, media.get('path'))
            match media_type := media.get('media_type'):
                case 'image':
                    duration = self.image_framerate
                    duration_ts = duration * 90000  # timebase = 1/90000, not sure if it's a constant in mpegts format
                    durations = (duration, duration_ts)
                    stream = ffmpeg.input(media_path, framerate=self.image_framerate)
                    if (bkg_color := media.get('bkg-color')) is not None:
                        meta = self.probe(media_path)['streams'][0]
                        shape = (meta['width'], meta['height'])
                        stream = ffmpeg.input(
                            f'color=c={bkg_color}:s={shape[0]}x{shape[1]}', f='lavfi', t=1
                        ).overlay(stream)
                        # stream = self._apply_standard_filters(
                        #     stream, media_type, self.size, self.fps, self.pts, self.dar, autoscale=autoscale
                        # )
                        # return stream, durations
                case 'animation':
                    loop = int(media.get('repeat', 1))
                    durations = self._get_media_duration(media_path, loop=loop)
                    stream = (
                        ffmpeg.input(media_path)
                              .filter('loop', loop=loop - 1, size=32767, start=0)
                    )
                case 'video':
                    loop = int(media.get('repeat', 1))
                    durations = self._get_media_duration(media_path, loop=loop)
                    stream = ffmpeg.input(media_path, stream_loop=loop - 1)
                case _:
                    raise IOError(f'UNKNOWN STREAM: {media_type}({media_path})')
            stream = self._apply_standard_filters(
                stream, media_type, self.size, self.fps, self.pts, self.dar, autoscale=autoscale
            )
            stream = self._auto_split_stream(stream, self.streams_used)
        return stream, durations

    def add_stream(self, media: ET.Element) -> None:
        stream, (duration, duration_ts) = self.create_stream(media)
        # _stream = _stream.output(
        #     'pipe:',
        #     format='mpegts',
        #     vcodec='libx264',
        #     **{'bsf:v': 'h264_metadata=sei_user_data=086f3693-b7b3-4f2c-9653-21492feee5b8+' + sei}
        # )  # TODO: this does work to store related info into the file.
        # however, I don't know how to extract info with ffprobe or any other tools yet
        # # # print(f'output bytestream {bytestream=}')
        # stream = ffmpeg.input('pipe:', format='mpegts')
        # stream.output('pipe:', format='mpegts', codec='libx264').run(input=bytestream, quiet=True)
        # proc.stdin.write(bytestream)
        # proc.stdin.close()

        self.streams.append(stream)
        start_time = self.total_duration
        self.total_duration += duration
        end_time = self.total_duration
        self.total_duration_ts += duration_ts

        if (text := media.get('path')) is not None:
            self.subtitler.add_entry(start_time, end_time, text)

    def get_stream(self) -> ffmpeg.nodes.FilterableStream:
        stream: ffmpeg.nodes.FilterableStream = ffmpeg.concat(*self.streams, v=1, a=0)
        # if self.start_pts != 0:
        #     stream = stream.filter('setpts', f'{self.start_pts}+PTS')
        return stream

    def output_stream(self, fname: str) -> ffmpeg.nodes.FilterableStream:
        output_kwds = self.output_kwds.copy()
        output_kwds['output_ts_offset'] = self.start_pts / 90000
        subtitle_path = Path(fname).with_suffix(self.subtitler.SUFFIX)
        self.subtitler.export(subtitle_path)
        stream: ffmpeg.nodes.FilterableStream = self.get_stream()
        stream = self.subtitler.injected(stream)
        stream = (
            stream
            .output(fname, **output_kwds)  # pyright: ignore[reportAttributeAccessIssue]
            .global_args(*self.run_args)
        )
        return stream

    def print_dryrun(self) -> None:
        self.print_stream_cmd(self.output_stream('-'))

    @staticmethod
    def _compile_call(
        fname: Path,
        outstream: ffmpeg.nodes.FilterableStream,
        temp_fname: Optional[str] = None,
        callback: Optional[Callable[[str], T]] = None,
        *,
        logger=None
    ) -> T | None:
        _fname = temp_fname if temp_fname is not None else fname
        if logger is not None:
            logger.detail(shlex.join(outstream.compile()))  # pyright: ignore[reportAttributeAccessIssue]
        # print(shlex.join(outstream.compile()))
        outstream.run(cmd=FFMPEG_BIN, overwrite_output=True)  # type: ignore
        if temp_fname is not None:
            shutil.move(_fname, fname)
        if callback is not None:
            return callback(str(fname))

    @staticmethod
    def callback(fname: str, *, meta: Optional[dict[str, Any]] = None) -> T:
        ...

    def compile_call(
        self,
        path: Path,
        temp_path: Optional[Path] = None,
        *,
        logger=None
    ) -> Callable[[], T]:
        _path = temp_path if temp_path is not None else path
        temp_fname = str(temp_path) if temp_path is not None else None
        outstream = self.output_stream(str(_path))
        callback = self.callback
        self.logger.detail(f'Prepare {path}')
        f: Callable[[], T] = functools.partial(
            self._compile_call, path, outstream, temp_fname, callback, logger=logger
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
        self.total_duration = 0.
        self.total_duration_ts = 0.
        self.subtitler.clear()

    def _overlay(
        self,
        m1: ffmpeg.nodes.FilterableStream,
        m2: ffmpeg.nodes.FilterableStream,
        x: str = '0',
        y: str = '0'
    ) -> ffmpeg.nodes.FilterableStream:
        oversized = False
        if x != '0':
            if not x.isalnum():
                oversized |= eval(f'({x} - W) >= 0', dict(w=1, W=1))
        if y != '0':
            if not y.isalnum():
                oversized |= eval(f'({y} - H) >= 0', dict(h=1, H=1))
        print(f'{x=}, {y=}, {oversized=}')
        if not oversized:
            return m1.overlay(m2, x=x, y=y)  # pyright: ignore[reportAttributeAccessIssue]
        n = m2.node
        while not isinstance(n, ffmpeg.nodes.InputNode):
            n = next(iter(n.incoming_edge_map.values()))[0]
        meta2 = self.probe(n.kwargs['filename'])['streams'][0]
        w: str = '0' if x is None else ('{}'.format(eval(f'str({x})+"+"+str(w)', dict(w=meta2['width'], W='iw'))))
        h: str = '0' if y is None else ('{}'.format(eval(f'str({y})+"+"+str(h)', dict(h=meta2['height'], H='ih'))))
        stream = m1.filter('pad', w=w, h=h)  # pyright: ignore[reportAttributeAccessIssue]
        return stream.overlay(m2, x='W-w', y='H-h')

    def overlay_stream(self, media: ET.Element) -> tuple[ffmpeg.nodes.FilterableStream, tuple[float, ...]]:
        streams_dict: dict[str | Literal[0], list[ET.Element]] = {}
        main_stream_id: str | Literal[0] | None = None

        for medium in media:
            id = medium.get('id')
            parent_id = medium.get('parent')
            # 當 parent 為 None
            if parent_id is None:
                if id is None:
                    # 如果 id 和 parent 都是 None，將兩者設為 0
                    id, parent_id = 0, 0
                else:
                    # 如果該 id 值第一次出現，將 parent 設為 0
                    if id not in streams_dict:
                        parent_id = 0
                    else:
                        # 如果該 id 值已經出現過，將 parent 設為 id 本身
                        parent_id = id
            else:
                # 當 parent 不為 None 且 id 為 None 時，設 id 為 parent
                if id is None:
                    id = parent_id
            if main_stream_id is None and (parent_id == id):
                main_stream_id = parent_id
            streams_dict.setdefault(parent_id, []).append(medium)

        stream_id = main_stream_id
        streams = streams_dict.pop(stream_id)  # pyright: ignore[reportArgumentType]
        stream, durations = self.create_stream(streams.pop(0), autoscale=False)

        stack = [(stream, {}, streams, stream_id)]
        while stack:
            stream, attribs, streams, stream_id = stack.pop()
            while streams:
                medium = streams.pop(0)
                _stream_id = medium.get('id')
                _attribs = {xi: v for xi in ('x', 'y') if (v := medium.get(xi)) is not None}
                if _stream_id is None:
                    _stream_id = stream_id
                if _stream_id == stream_id:
                    stream = self._overlay(
                        stream,
                        self.create_stream(medium, autoscale=False)[0],
                        **_attribs
                    )
                else:
                    _streams = streams_dict.pop(_stream_id)  # pyright: ignore[reportArgumentType]
                    _stream = self.create_stream(medium, autoscale=False)[0]
                    stack.append((stream, attribs, streams, stream_id))
                    stack.append((_stream, _attribs, _streams, _stream_id))
                    break
            else:
                if stack:
                    _stream, _attribs, _streams, _stream_id = stack[-1]
                    stack[-1] = (self._overlay(_stream, stream, **attribs), _attribs, _streams, _stream_id)

        return (stream, durations)


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
        loglevel: Optional[LogLevel | bool] = None
    ):
        super().__init__(delay, size, fps, rate, loglevel)
        # self.run_args.extend(['-loglevel', 'error'])
        self.output_kwds.update(
            codec='libx264',
            format='mpegts',
            pix_fmt='yuv420p',
            fps_mode='auto',
            preset='veryfast',
            tune='zerolatency'
        )

    @staticmethod
    def callback(fname: str, *, meta: Optional[dict[str, Any]] = None) -> VideoMeta:
        _meta: VideoMeta = {'tag': 'video', 'content': fname}
        if meta is not None:
            _meta.update(**meta)
        subtitle_file = Path(fname).with_suffix(SUBTITLER.SUFFIX)
        subtitle_file.unlink(missing_ok=True)
        return _meta


class FFMPEGObjectOutput(FFMPEGObject):
    def __init__(
        self,
        delay: float,
        size: tuple[int, int],
        fps: int,
        rate: float,
        loglevel: Optional[LogLevel | bool] = None
    ):
        super().__init__(delay, size, fps, rate, loglevel)
        # self.run_args.extend(['-loglevel', 'debug'])
        self.output_kwds.update(
            format='mp4',
            pix_fmt='yuv420p',
            fps_mode='auto',
            codec='libx264',
        )
