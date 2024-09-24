from __future__ import annotations
from collections import Counter
import functools
import logging
import math
import multiprocessing as mp
from pathlib import Path
import random
import re

from typing import Callable, Iterable, Iterator, Literal, Optional, Self, Sequence, TypeVar

from PIL import Image
import ffmpeg

LOGLEVEL = 'DEBUG'
FFMPEG_LOGLEVEL = 'debug'
COLOR_LOG = True

T = TypeVar('T')


def _addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError(f'{levelName} already defined in logging module')
    if hasattr(logging, methodName):
        raise AttributeError(f'{methodName} already defined in logging module')
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError(f'{methodName} already defined in logger class')

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


_addLoggingLevel('DETAIL', logging.DEBUG - 5)


def get_logger(
        name: str, color: Optional[bool] = None,
        to_stream: bool = True,
        to_file: bool | Path | str = False
) -> logging.Logger:
    if color is None:
        color = COLOR_LOG
    has_colorlogs = True
    if color:
        try:
            import coloredlogs
            from humanfriendly.terminal import terminal_supports_colors
            has_colorlogs &= terminal_supports_colors()
        except ModuleNotFoundError:
            has_colorlogs = False
    # assert False, f'{terminal_supports_colors()=}'
    # Logger Creation
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    # Formatter Setting
    fmt = {
        'fmt': '{asctime} {name} {levelname} {message}',
        'datefmt': '%H:%M:%S',
        'style': '{'
    }
    if color is True and has_colorlogs:
        formatter = coloredlogs.ColoredFormatter(**fmt)
    else:
        formatter = logging.Formatter(**fmt)
    # Handlers Setting
    handlers: list[logging.Handler] = []
    if to_stream:
        handlers.append(logging.StreamHandler())
    if to_file is not False:
        if to_file is True:
            logger_path = Path('./slideshow.log')
        else:
            logger_path = Path(to_file)
        handlers.append(logging.FileHandler(logger_path))
    for handler in handlers:
        logger.addHandler(handler)
        handler.setFormatter(formatter)
    # LoggerLevel Setting
    logger.setLevel(getattr(logging, LOGLEVEL))
    if color is True and not has_colorlogs:
        logger.info('coloredlogs not installed. uncolored logging will not be populated.')
    return logger


def rescaled(image, size, result, i=0, fmt=None):
    if not image:
        try:
            result.put((0, None))
        except AssertionError:
            pass
        return
    image_object = Image.open(image)
    width, height = image_object.width, image_object.height
    winfo_width, winfo_height = size
    ratio = min(winfo_width / width, winfo_height / height)
    try:
        image_object = image_object.resize((int(width * ratio), int(height * ratio)), 1)
        if fmt is not None:
            image_object = fmt(image_object)
    except ValueError:
        pass
    try:
        result.put((i, image_object))
    except AssertionError:
        pass


class AspectEstimator:
    def __init__(self, prop: float = 0.1, default_aspect: Optional[tuple[int, int] | Literal['auto']] = None):
        self.logger = get_logger('AspectEstimator')
        self._aspect: Optional[tuple[int, int] | Literal['auto']] = default_aspect
        self._prop: float
        self._add_sample_aspect: Callable[[mp.Queue[tuple[int, int]], str], None]
        self.set_prop(prop)

        self._pool: mp.pool.Pool
        self._manager: mp.managers.SyncManager
        self._result_queue: mp.Queue[tuple[int, int]]
        if self._prop != 0:
            self._pool = mp.Pool()
            self._manager = mp.Manager()
            self._result_queue = self._manager.Queue()

    @property
    def prop(self) -> float:
        return self._prop

    def set_prop(self, prop: float):
        self._prop = 0. if self._aspect is not None else prop
        if self._prop == 1.:
            add_sample_aspect = self._add_sample_aspect_all
        elif self._prop == 0:
            add_sample_aspect = self._add_sample_aspect_empty
        else:
            add_sample_aspect = functools.partial(self._add_sample_aspect_prop, prop=self._prop)
        self._add_sample_aspect = add_sample_aspect

    def add_sample_aspect(self, sample: str):
        self._pool.apply_async(self._add_sample_aspect, (self._result_queue, sample))

    @staticmethod
    def _add_sample_aspect_empty(queue: mp.Queue[tuple[int, int]], sample: str) -> None:
        ...

    @staticmethod
    def _add_sample_aspect_prop(queue: mp.Queue[tuple[int, int]], sample: str, prop: float) -> None:
        if random.random() > prop:
            return
        AspectEstimator._add_sample_aspect_all(queue, sample)

    @staticmethod
    def _add_sample_aspect_all(queue: mp.Queue[tuple[int, int]], sample: str) -> None:
        aspect = AspectEstimator.get_sample_aspect(sample)
        if aspect is not None:
            queue.put(aspect)

    @staticmethod
    def get_sample_aspect(sample: str) -> Optional[tuple[int, int]]:
        sample_meta = ffmpeg.probe(sample)['streams'][0]
        display_aspect_ratio = sample_meta.get('display_aspect_ratio', None)
        if display_aspect_ratio is not None:
            return tuple(map(int, display_aspect_ratio.split(':', 1)))
        else:
            w, h = map(sample_meta.get, ('width', 'height'), (None, None))
            if None not in (w, h):
                d = math.gcd(w, h)
                return (w // d, h // d)
        return None

    def get_aspect(self) -> tuple[int, int] | Literal['auto']:
        if self._aspect is None:
            self._pool.close()
            self._pool.join()
            sample_aspects = []
            while not self._result_queue.empty():
                sample_aspects.append(self._result_queue.get())

            counter = Counter(sample_aspects)
            counter_size = counter.total()
            self.logger.info(f'{counter_size} samples collected. Computing sampled aspect now...')
            if counter_size:
                self._aspect = counter.most_common(1)[0][0]
            else:
                self.logger.warn('Aspect infomation not extracted properly. "auto" mode used instead.')
                self._aspect = 'auto'
        return self._aspect

    def get_aspect_as_str(self) -> str:
        aspect = self.get_aspect()
        if isinstance(aspect, str):
            aspect = aspect
        else:
            aspect = 'X'.join(map(str, aspect))
        return aspect


def flatten(t: list[T], inplace: bool = False) -> list[T]:
    flat_list = []
    for sublist in t:
        for item in sublist:
            flat_list.append(item)
    if inplace:
        t[:] = flat_list
    else:
        t = flat_list
    return t


class _Replacement:
    groupindex = type('groupindex', (), {'__getitem__': lambda self, _: 0})()
    groups = re._parser.MAXGROUPS
    string = ''

    def __init__(self, replacements: dict[int | str, str] | Iterable[str]):
        self.groupindex: dict[int | str, int] = {}
        if not isinstance(replacements, dict):
            replacements = dict(enumerate(replacements, 1))
        else:
            count = -1
            for k, v in tuple(replacements.items()):
                if isinstance(k, str):
                    self.groupindex[k] = count
                    replacements[count] = v
                    count -= 1
        self._data: dict[int | str, str] = replacements

    def group(self, name: int | str) -> str:
        return self._data[name]


def expand_template(
        template: str, replacements: dict[int | str, str] | Iterable[str]
) -> str:
    r = _Replacement(replacements)
    template = re._parser.parse_template(template, r)
    expanded = ''.join(r.group(part) if isinstance(part, int) else part for part in template)
    # expanded = re._parser.expand_template(template, r)
    return expanded


def sampled(dataset: Sequence[T], sample_size: int = 3) -> list[T]:
    n = len(dataset)
    if sample_size >= n:
        return list(dataset)
    step_size = n / sample_size
    result = [dataset[0], *(dataset[int(i * step_size)] for i in range(1, sample_size))]
    return result


class ByteStreamReader:
    cached_streams: list[Self] = []

    def __init__(self, identifiers):
        self.identifiers = identifiers
        self.stream = bytearray()
        self.streams: Iterator[bytes] = iter([])
        self._cursor: int = 0
        self._stream_end: int = 0
        self.total_size = None
        self.cached_streams.insert(0, self)

    @classmethod
    def spawn(cls, identifiers) -> Self:
        try:
            bs = next(filter(lambda bs: bs.identifiers == identifiers, cls.cached_streams))
        except StopIteration:
            bs = cls(identifiers)
        return bs

    def set_streams(self, streams: Iterator[bytes]) -> None:
        self.streams = streams
        self.stream.clear()
        self._cursor = 0
        self._stream_end = -1

    def process_next_stream(self) -> bool:
        print('process next...')
        try:
            next_stream = next(self.streams)
            print(f'{len(next_stream)=}')
            self._stream_end += len(next_stream)
            self.stream.extend(next_stream)
            return False
        except StopIteration:
            self.total_size = self._stream_end
            return True

    def read(self, start: Optional[int] = None, end: Optional[int] = None) -> bytes:
        if start is None:
            start = self._cursor
        elif start < self._cursor:
            raise IOError()
        if start > self._stream_end:
            if self.process_next_stream():
                return b''
            return self.read(start, end)
        self.stream = self.stream[start - self._cursor:]
        self._cursor = start
        if end is None:
            return bytes(self.stream)
        if end > self._stream_end:
            return bytes(self.stream) + self.read(self._stream_end + 1, end)
        return bytes(self.stream[:end - start + 1])
