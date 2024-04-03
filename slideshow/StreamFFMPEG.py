from xml.etree import ElementTree as ET
from typing import Optional, Iterable, Sequence
from pathlib import Path

import ffmpeg

from . import utils
from . import AlbumReader
from . import FFMPEGObject

logger = utils.get_logger('SlideShow.StreamFFMPEG')


def _ffmpeg_read(
    output: Path,
    iterable: Iterable[ET.Element],
    delay: float,
    size: tuple[int, int],
    segment_time: int = 10,
    parent: Optional[object] = None,
    loglevel: Optional[FFMPEGObject.LogLevel] = None
) -> None:
    # with multiprocessing.Pool(None, limit_cpu) as pool:
    output = str(output)
    logger.debug('Submmiting jobs to media processing queue...')
    if parent is None:
        ffmpeg_object = FFMPEGObject.FFMPEGObjectLive(delay, size, loglevel=loglevel)
    else:
        ffmpeg_object = parent.ffmpeg_object

    logger.debug('Iterating over files...')

    with (
        ffmpeg.input('pipe:', format='mpegts')
            .output(output, codec='copy')
            .overwrite_output()
            .run_async(pipe_stdin=True)
    ) as p_out:        
        for media in iterable:
            if media.tag == 'meta':
                if parent is not None:
                    exec(f'parent.{media.get("command")}', {'parent': parent})
                continue
            ffmpeg_object.add_stream(media)
            if ffmpeg_object.total_duration > segment_time:
                p_in = ffmpeg_object.output_stream('pipe:').run_async(pipe_stdout=True, pipe_stderr=True)
                input_stream, _ = p_in.communicate()
                p_out.stdin.write(input_stream)
                ffmpeg_object.reset()
            logger.detail(f'End processing {media}...')
        if ffmpeg_object.streams:
            p_in = ffmpeg_object.output_stream('pipe:').run_async(pipe_stdout=True, pipe_stderr=True)
            input_stream, _ = p_in.communicate()
            p_out.stdin.write(input_stream)
            ffmpeg_object.reset()

    logger.debug('Submmiting jobs to media processing queue finished. Waiting for jobs to end...')
    logger.debug('All media processing jobs finished.')


class App:
    def __init__(
        self,
        media_files: Iterable[str],
        delay: int = 1000,
        rate: float = 1.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        aspect: str = '16X9',
        chapters: Optional[Sequence[str]] = None
    ):
        self._delay: int = delay
        self._rate: float = rate
        self._size: tuple[int, int] = self.parse_size(width, height, aspect)
        self.ffmpeg_object = FFMPEGObject.FFMPEGObjectLive(delay=delay, size=self._size, rate=rate)
        self.album: AlbumReader.AlbumReader = AlbumReader.AlbumReader(*media_files, chapters=chapters)

        for meta in self.album.preprocess_meta():
            cmd = meta.get('command')
            logger.info(f'Executing App.{cmd}')
            exec(f'self.{cmd}')

    def show_slides(self, output: Path) -> None:
        _ffmpeg_read(output, iter(self.album), self.delay, self.size, parent=self)

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
