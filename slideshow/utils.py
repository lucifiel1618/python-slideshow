from __future__ import annotations
from PIL import Image
import logging
import random
from collections import Counter
import math
import ffmpeg
import multiprocessing as mp
from typing import Callable, Iterable, Optional, Sequence, TypeVar
import functools

import re

LOG_LEVEL = 'DEBUG'
COLOR_LOG = True

T = TypeVar('T')


def get_logger(name, color=None):
    if color is None:
        color = COLOR_LOG
    if color:
        try:
            import coloredlogs
            from humanfriendly.terminal import terminal_supports_colors
        except ModuleNotFoundError:
            color = False
            logger.info('coloredlogs not installed. colored logging will not be populated.')

    fmt = {
        'fmt': '{asctime} {name} {levelname} {message}',
        'datefmt': '%H:%M:%S',
        'style': '{'
    }
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    formatter = (coloredlogs.ColoredFormatter if color and terminal_supports_colors() else logging.Formatter)(**fmt)

    handler.setFormatter(formatter)
    logger.setLevel(getattr(logging, LOG_LEVEL))

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
    def __init__(self, prop: float = 0.1, default_aspect: Optional[tuple[int, int] | str] = None):
        self.logger = get_logger('AspectEstimator')
        self._aspect: Optional[tuple[int, int] | str] = default_aspect
        self.set_prop(prop)
        if self._prop != 0:
            self._pool: mp.pool.Pool = mp.Pool()
            self._manager: mp.managers.SyncManager = mp.Manager()
            self._result_queue: mp.Queue[tuple[int, int]] = self._manager.Queue()
            # self._result_queue: mp.Queue[tuple[int, int]] = mp.Queue()

    @property
    def prop(self) -> float:
        return self._prop

    def set_prop(self, prop: float):
        self._prop: float = 0. if self._aspect is not None else prop
        if self._prop == 1.:
            add_sample_aspect = self._add_sample_aspect_all
        elif self._prop == 0:
            add_sample_aspect = self._add_sample_aspect_empty
        else:
            add_sample_aspect = functools.partial(self._add_sample_aspect_prop, prop=self._prop)
        self._add_sample_aspect: Callable[[mp.Queue[tuple[int, int]], str], None] = add_sample_aspect

    def add_sample_aspect(self, sample: str):
        self._pool.apply_async(self._add_sample_aspect, (self._result_queue, sample))

    @staticmethod
    def _add_sample_aspect_empty(queue: mp.Queue[tuple[int, int]], sample: str):
        return

    @staticmethod
    def _add_sample_aspect_prop(queue: mp.Queue[tuple[int, int]], sample: str, prop: float):
        if random.random() > prop:
            return
        aspect = AspectEstimator.get_sample_aspect(sample)
        if aspect is not None:
            queue.put(aspect)

    @staticmethod
    def _add_sample_aspect_all(queue: mp.Queue[tuple[int, int]], sample: str):
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

    def get_aspect(self) -> tuple[int, int] | str:
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


def flatten(t: list, inplace: bool = False) -> list:
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
    expanded = re._parser.expand_template(template, r)
    return expanded


def sampled(dataset: Sequence[T], sample_size: int = 3) -> Sequence[T]:
    n = len(dataset)
    if sample_size >= n:
        return dataset
    step_size = n / sample_size
    result = [dataset[0], *(dataset[int(i * step_size)] for i in range(1, sample_size))]
    return result
