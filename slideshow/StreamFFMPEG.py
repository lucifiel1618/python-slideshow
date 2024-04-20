from argparse import Namespace
import asyncio
import functools
import multiprocessing
from multiprocessing.pool import AsyncResult
import queue
import shutil
import threading
from urllib.parse import unquote
from xml.etree import ElementTree as ET
from typing import Awaitable, Callable, Iterator, Literal, Optional, Iterable, Sequence, TypedDict
from pathlib import Path

from fastapi.responses import FileResponse

from . import utils
from . import AlbumReader
from . import FFMPEGObject

logger = utils.get_logger('SlideShow.StreamFFMPEG')


'''
def _ffmpeg_read_pipe(
    output: Path | str,
    iterable: Iterable[ET.Element],
    delay: float,
    size: tuple[int, int],
    segment_time: int = 10,
    parent: Optional[object] = None,
    loglevel: Optional[FFMPEGObject.LogLevel] = None
) -> Iterator[bytes]:
    output = str(output)
    logger.debug('Submmiting jobs to media processing queue...')
    if parent is None:
        ffmpeg_object = FFMPEGObject.FFMPEGObjectLive(delay, size, loglevel=loglevel)
    else:
        ffmpeg_object = parent.ffmpeg_object

    logger.debug('Iterating over files...')

    if output != 'pipe:':
        p_out = (
            ffmpeg.input('pipe:', format='mpegts')
            .output(output, codec='copy')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    else:
        p_out = None

    for media in iterable:
        if media.tag == 'meta':
            if parent is not None:
                exec(f'parent.{media.get("command")}', {'parent': parent})
            continue
        ffmpeg_object.add_stream(media)
        if ffmpeg_object.total_duration > segment_time:
            p_in = ffmpeg_object.output_stream('pipe:').run_async(pipe_stdout=True, pipe_stderr=True)
            input_stream, _ = p_in.communicate()
            if p_out is None:
                yield input_stream
            else:
                p_out.stdin.write(input_stream)
            ffmpeg_object.reset()
        logger.detail(f'End processing {media}...')
    if ffmpeg_object.streams:
        p_in = ffmpeg_object.output_stream('pipe:').run_async(pipe_stdout=True, pipe_stderr=True)
        input_stream, _ = p_in.communicate()
        if p_out is None:
            yield input_stream
        else:
            p_out.stdin.write(input_stream)
        ffmpeg_object.reset()

    if p_out is not None:
        p_out.close()

    logger.debug('Submmiting jobs to media processing queue finished. Waiting for jobs to end...')
    logger.debug('All media processing jobs finished.')
'''


def process_meta(command: str) -> FFMPEGObject.VideoMeta:
    return {'tag': 'meta', 'content': command}


def _ffmpeg_read(
    queue: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]],
    iterable: Iterable[ET.Element],
    ffmpeg_object: FFMPEGObject.FFMPEGObject,
    dpath: str,
    segment_time: float = 10,
    callback: Optional[Callable[[], None]] = None,
) -> None:
    # with multiprocessing.Pool(None, limit_cpu) as pool:
    limit_cpu = (multiprocessing.cpu_count() - 1) or 1
    segment_time = segment_time * ffmpeg_object.rate
    with multiprocessing.Pool(limit_cpu) as pool:
        logger.debug('Submmiting jobs to media processing queue...')
        index = 0
        skipped = 0
        logger.debug('Iterating over files...')
        current_pts = 0.
        for media in iterable:
            logger.detail(f'Processing {media}...')
            if media.tag == 'meta':
                queue.put_nowait(pool.apply_async(process_meta, (media.get('command'),)))
                skipped += 1
            else:
                ffmpeg_object.add_stream(media)
                if ffmpeg_object.total_duration > segment_time:
                    if skipped > 0:
                        qput = queue.put_nowait
                        skipped -= 1
                    else:
                        qput = queue.put
                        fname = f'{dpath}/{index:0>4}.ts'
                    logger.detail(f'Queuing `{fname}`...')
                    ffmpeg_object.start_pts = int(round(current_pts))
                    qput(pool.apply_async(ffmpeg_object.compile_call(fname, f'{fname}.part')))  # type: ignore
                    current_pts += ffmpeg_object.total_duration * 90000 / ffmpeg_object.rate
                    ffmpeg_object.reset()
                    index += 1
        if ffmpeg_object.streams:
            if skipped > 0:
                qput = queue.put_nowait
                skipped -= 1
            else:
                qput = queue.put
                fname = f'{dpath}/{index:0>4}.ts'
            ffmpeg_object.start_pts = round(current_pts)
            qput(pool.apply_async(ffmpeg_object.compile_call(fname)))  # type: ignore
        pool.close()
        logger.debug('Submmiting jobs to media processing queue finished. Waiting for jobs to end...')
        pool.join()
        callback()
        queue.join()
        logger.debug('All media processing jobs finished.')


class FStat(TypedDict):
    path: Path
    size: int


class App:
    def __init__(
        self,
        media_files: Iterable[str],
        delay: int = 1000,
        rate: float = 1.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect: str = '16X9',
        chapters: Optional[Sequence[str]] = None,
        loglevel: Optional[FFMPEGObject.LogLevel] = None
    ):
        self._delay: int = delay
        self._rate: float = rate
        self._size: tuple[int, int] = self.parse_size(width, height, aspect)
        self.ffmpeg_object = FFMPEGObject.FFMPEGObjectLive(delay=delay, size=self._size, rate=rate)
        self.album: AlbumReader.AlbumReader = AlbumReader.AlbumReader(*media_files, chapters=chapters)
        self.queue: queue.Queue[AsyncResult[FFMPEGObject.VideoMeta]] = queue.Queue()
        self.media_thread = None
        self.loglevel = loglevel

        for meta in self.album.preprocess_meta():
            cmd = meta.get('command')
            logger.info(f'Executing App.{cmd}')
            exec(f'self.{cmd}')

    """
    def show_slides_pipe(self, output: Path | Literal['pipe:']) -> Iterator[bytes]:
        '''If `output` is __NOT__ set to `'pipe:'`, bytestream is saved into `output`.
        An empty iterator will be returned instead. Can be simply executed by consuming it.
        '''
        yield from _ffmpeg_read_pipe(output, iter(self.album), self.delay, self.size, parent=self)
    """

    def process(self, output_dir: Path, callback: Optional[Callable[[], None]] = None) -> None:
        logger.debug('Initializing video thread...')
        if self.media_thread is None:
            self.media_thread = threading.Thread(
                target=_ffmpeg_read,
                args=(self.queue, iter(self.album), self.ffmpeg_object, str(output_dir)),
                kwargs=dict(callback=callback)
            )
        self.media_thread.start()
        logger.debug('Terminating video thread...')

    def play_next(self, first: bool = False) -> Iterator[FFMPEGObject.VideoMeta]:
        if first:
            logger.debug('Start pulling files from the media queue...')
        logger.detail('Accessing media queue for the next file...')
        if first or self.queue.unfinished_tasks > 0:
            logger.detail('Waiting for the media processing to finish...')
            try:
                timeout = 60
                entry: FFMPEGObject.VideoMeta = self.queue.get(timeout=timeout).get(timeout=timeout)
            except queue.Empty as e:
                e.args = (f'{timeout}s Timeout. Media queue is empty when it shouldn\'t be.', )
                logger.error(e)
                raise e
            logger.detail(f'Next file found `{entry}`...')
            self.queue.task_done()
        else:
            logger.debug('Empty queue. Exiting gracefully.')
            return
        content = entry['content']
        if entry['tag'] == 'meta':
            exec(f'self.{content}')
        else:
            yield entry
        yield from self.play_next()

    def show_slides(self, output_dir: Path) -> None:
        self.process(output_dir)
        for _ in self.play_next(True):
            ...
        self.media_thread.join()

    def set_ffmpeg_loglevel(self, value: Optional[FFMPEGObject.LogLevel] = None) -> None:
        self.ffmpeg_object.loglevel = value

    @property
    def delay(self) -> int:
        return self._delay

    @delay.setter
    def delay(self, delay: int) -> None:
        self._delay = delay
        self.ffmpeg_object.delay = delay

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, rate: float) -> None:
        if rate < 0:
            rate = 0.
        self._rate = rate
        self.ffmpeg_object.rate = rate

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @size.setter
    def size(self, size: tuple[int, int]) -> None:
        self._size = size
        self.ffmpeg_object.size = size

    def parse_size(self, width: Optional[int] = None, height: Optional[int] = None, aspect: str = '16X9'):
        if (width, height) == (None, None):
            match aspect:
                case '4X3':
                    width, height = 1024, 768
                case '3X4':
                    width, height = 768, 1024
                case '16X9':
                    width, height = 1280, 720
                case '9X16':
                    width, height = 720, 1280
                case _:
                    width = 1280

        if width is not None and height is not None:
            pass
        else:
            aspect = tuple(int(l) for l in aspect.split('X', 1))
            aspect_ratio = aspect[0] / aspect[1]
            if width is not None:
                height = width * aspect_ratio
            else:
                width = height / aspect_ratio
        return (width, height)

    def set_size(self, width: Optional[int] = None, height: Optional[int] = None, aspect: str = '16X9'):
        self.size = self.parse_size(width, height, aspect)

    def showFullScreen(self):
        ...

    def change_playspeed(self, speed: float) -> None:
        self.rate = speed


class Resource:
    def __init__(
            self,
            id: Optional[str] = None,
            path_from: Optional[Path] = None,
            rootdir: Optional[Path] = None
    ):
        self.path_from = path_from
        self.id: str = id if id is not None else self.get_id(path_from)
        self.path = rootdir / self.id if rootdir is not None else Path(self.id)
        self.status_file = self.path / 'STATUS'

    @staticmethod
    @functools.lru_cache(maxsize=10)
    def get_id(path_from: Path) -> str:
        return f'${path_from.stat().st_ino}R'

    def get_status(self) -> Literal['', 'created', 'finished']:
        if not self.path.exists():
            return ''
        return self.status_file.read_text()

    def initialize(self) -> None:
        status = self.get_status()
        if not status:
            self.path.mkdir()
            self.status_file.write_text('created')
            return
        return

    def set_status(self, status: Literal['', 'created', 'finished']):
        if not status:
            shutil.rmtree(self.path)
        self.status_file.write_text(status)

    @staticmethod
    def get_segment_base(index: int | str, ext: str) -> str:
        return f'{index:0>4}{ext}'

    def get_segment(self, index: int, ext: str) -> Path:
        return self.path / self.get_segment_base(index, ext)


def start_server(args: Namespace | None = None):
    import fastapi
    import uvicorn
    api = fastapi.FastAPI()
    if args is None:
        mediadir = Path('media')
        srcdir = mediadir / 'disk'
        dstdir = mediadir / '_temp'
        port = 8000
        ffmpeg_loglevel = 'DEBUG'
    else:
        srcdir = Path(args.srcdir)
        dstdir = Path(args.dstdir)
        port = args.port
        ffmpeg_loglevel = args.ffmpeg_loglevel
    assert srcdir.exists()
    assert dstdir.exists()

    # @api.middleware('http')
    # async def log_requests(
    #     request: fastapi.Request,
    #     call_next: Callable[[fastapi.Request], Awaitable[fastapi.Response]]
    # ) -> fastapi.Response:
    #     logger.info(f'\"{request.method} {request.url.path} {request.scope["type"]}\"')
    #     response = await call_next(request)
    #     return response

    @api.get('/path/{path:path}')
    async def run_server(
        path: str,
        request: fastapi.Request,
        chapters: Optional[str] = None,
        delay: Optional[int] = None,
        rate: Optional[float] = None,
        aspect: Optional[str] = None
    ) -> fastapi.Response:
        path = unquote(path)
        if path == '$PWD':
            path = '.'
        if args is not None:
            if delay is None:
                delay = args.delay
            if rate is None:
                rate = args.rate
            if aspect is None:
                aspect = args.aspect
        _chapters = chapters.split(',') if chapters is not None else None
        resource = Resource(path_from=srcdir / path, rootdir=dstdir)
        if not resource.get_status():
            resource.initialize()
            app = App([str(resource.path_from)], delay, rate=rate, aspect=aspect.upper(), chapters=_chapters)
            app.set_ffmpeg_loglevel(ffmpeg_loglevel)
            app.process(resource.path, callback=lambda: resource.set_status('finished'))
            next(app.play_next(True))  # waiting for the first segment to finish
        return (await get_playlist(resource.id, 0, request))

    @api.get('/resource/{path}/{segment}.m3u8')
    async def get_playlist(path: str, segment: int, request: fastapi.Request):
        url = request.url
        content_lines = [
            '#EXTM3U',
            '#EXT-X-VERSION:4',
            # '#EXT-X-PLAYLIST-TYPE:VOD',
            '#EXT-X-PLAYLIST-TYPE:EVENT',
            '#EXT-X-TARGETDURATION:10',
            f'#EXT-X-MEDIA-SEQUENCE:{segment}'
        ]
        _segment = segment
        resource = Resource(path, rootdir=dstdir)
        f = resource.get_segment(_segment, '.ts')
        while f.exists():
            if segment != _segment:
                # content_lines.append('#EXT-X-DISCONTINUITY')
                ...
            content_lines.extend(
                [
                    f'#EXTINF:{FFMPEGObject.FFMPEGObjectLive._get_media_duration(str(f))[0]:.2f},',
                    str(url.replace(path=f'/resource/{path}/{f.name}'))
                ]
            )
            _segment += 1
            f = resource.get_segment(_segment, '.ts')
        status = resource.get_status()
        if segment == _segment and status == 'created':
            await asyncio.sleep(1)
            return (await get_playlist(path, segment, request))
        if status != 'finished':
            content_lines.extend(
                [
                    '#EXT-X-STREAM-INF:',
                    str(url.replace(path=f'/resource/{path}/{Resource.get_segment_base(_segment, ".m3u8")}'))
                ]
            )
        else:
            content_lines.append('#EXT-X-ENDLIST')
        return fastapi.Response('\n'.join(content_lines), media_type='application/vnd.apple.mpegurl')

    @api.get('/resource/{path}/{segment}.ts')
    async def get_video(path: str, segment: str) -> FileResponse:
        f = dstdir / path / Resource.get_segment_base(segment, '.ts')
        return fastapi.responses.FileResponse(f, media_type='video/mpegts')

    uvicorn.run(api, host='0.0.0.0', port=port)


'''
def run_app_pipe(
    path: list[str], output: list[str] | Literal['pipe:'],
    delay: int, rate: float, aspect: str, chapters: Optional[list[str]]
) -> Iterator[bytes]:
    for outp, _chapters in AlbumReader.iomap(path, chapters=chapters, outputs=output, for_each=False).items():
        app = App(path, delay, rate=rate, aspect=aspect.upper(), chapters=_chapters)
        yield from app.show_slides_pipe(outp)


def start_server_pipe(args: Namespace | None = None):
    import fastapi
    import uvicorn
    api = fastapi.FastAPI()

    @api.get('/path/{path}')
    async def run_server(
        path: str,
        chapters: Optional[str] = None,
        delay: Optional[int] = None,
        rate: Optional[float] = None,
        aspect: Optional[str] = None,
        bytes_range: str = fastapi.Header('0-')
    ) -> fastapi.Response:
        print('run_server')
        if path == '$PWD':
            path = '.'
        if args is not None:
            if delay is None:
                delay = args.delay
            if rate is None:
                rate = args.rate
            if aspect is None:
                aspect = args.aspect
        _chapters = chapters.split(',') if chapters is not None else None
        print(f'{bytes_range=}')
        _start, _end = bytes_range.replace('bytes=', '').split('-')
        start = int(_start)
        end = None if not _end else int(_end)
        identifiers = dict(path=[path], output='pipe:', delay=delay, rate=rate, aspect=aspect, chapters=_chapters)
        bs = utils.ByteStreamReader.spawn(identifiers)
        print(f'{start=}, {end=}')
        if start < bs._cursor or start == 0:
            bs.set_streams(run_app_pipe(**identifiers))
        response = fastapi.Response(
            bs.read(start, end),
            status_code=206,
            media_type='video/mp4',
            headers={
                'Content-Range': f'bytes {bs._cursor}-{bs._stream_end}/{bs.total_size or "*"}',
                'Accept-Ranges': 'bytes'
            }
        )
        print(f'{response.headers=}')
        return response

    uvicorn.run(api)
'''
