import collections
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
import functools
import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Self, Sequence

import yaml

from . import utils
from .Sorter import (
    Element, ElementGroup, FuturePair, GroupedSimilarImageFilter, GroupedSimilarImageSorter, ImageGroupTagSorterCoeff, Pair, RegexSorterCoeff, SimilarImageSorter, Sorter, GenericSorterCoeff, SorterChain, StrGroup, as_elements, collect_futures
)
logger = utils.get_logger('Slideshow.Config')


@dataclasses.dataclass(slots=True)
class Node:
    id: str = ''
    parent: Optional[Self] = dataclasses.field(default=None)


@dataclasses.dataclass(slots=True)
class Domain(Node):
    _sorters: list[None | Sorter | GenericSorterCoeff] = dataclasses.field(default_factory=list)
    meta: list[str] = dataclasses.field(default_factory=list)

    def get_sorter(self, i) -> Sorter:
        sorter = self._sorters[i]
        if sorter is None:
            assert self.parent is not None
            return self.parent.get_sorter(i)
        elif isinstance(sorter, GenericSorterCoeff):
            sorter_cls = sorter.__class__
            coeffs = [sorter]
            n = self
            while n.parent is not None:
                n = n.parent
                try:
                    coeffs.append(n._sorters[i])
                except IndexError:
                    pass
            kwds = functools.reduce(sorter_cls.updated_by, reversed(coeffs)).asdict()
            sorter = sorter_cls.get_sorter_cls().create(**kwds)
        return sorter

    def get_sorter_chain(self) -> SorterChain:
        n = len(self._sorters)
        return SorterChain([self.get_sorter(i) for i in range(n)])

    def separated(self, dataset: Iterable[Element]) -> Pair[ElementGroup]:
        return self.get_sorter_chain().separated(dataset)

    def separated_async(self, ft_in: Future[ElementGroup], executor: ThreadPoolExecutor) -> FuturePair[ElementGroup]:
        ft_d = ft_in
        fts_k: list[Future[ElementGroup]] = []
        for e in self.elements[1:]:
            ft_k, ft_d = e.separated_async(ft_d, executor=executor)
            fts_k.append(ft_k)
        fts_out = FuturePair(executor.submit(collect_futures, *fts_k), ft_d)
        return fts_out

    def combined(self, b: Self) -> Self:
        return self.__class__(
            self.id,
            None,
            self._sorters + b._sorters,
            self.meta + b.meta
        )


def get_processor(
    processor: Optional[ModuleType | str] = None,
    package: Optional[str] = None
) -> ModuleType | None:
    if isinstance(processor, ModuleType):
        return processor
    if package is None:
        package = '.config_processors'
    pkg = importlib.import_module(package, __package__)
    if processor is None:
        processor = pkg.processor
        if processor is None:
            return None
    processor = importlib.import_module(f'.{processor}', f'{__package__}{package}')
    return processor


@dataclasses.dataclass(slots=True, frozen=True)
class ConfigWriter:
    processor: ModuleType

    @classmethod
    def create(cls, processor: Optional[ModuleType | str] = None) -> Self:
        processor = get_processor(processor)
        if processor is None:
            raise ModuleNotFoundError('No proper module found.')
        return cls(processor)

    def write(self, config_file: Path | str, pattern: Optional[str | Path]) -> None:
        self.processor.create_config(
            to_file=config_file,
            **self.processor.parse_args(pattern)
        )


@dataclasses.dataclass(slots=True)
class ConfigReader:
    header: Domain
    _global: Domain
    chapters: list[Domain]
    ditches: list[Domain]

    def __post_init__(self):
        for ch in self.chapters:
            ch.parent = self._global

    def combined(self, b: Self) -> Self:
        header = self.header.combined(b.header)
        _global = self._global.combined(b._global)

        chapter_dict = {}
        for ch in self.chapters + b.chapters:
            if ch.id not in chapter_dict:
                chapter_dict[ch.id] = ch
            else:
                chapter_dict[ch.id] = chapter_dict[ch.id].combined(ch)

        ditches = self.ditches + b.ditches
        return type(self)(header, _global, list(chapter_dict.values()), ditches)

    @classmethod
    def from_args(
        cls,
        reg: dict[str, Sequence[str]],
        meta: Optional[Sequence[str]] = None,
        sorter: Optional[str] = None
    ) -> Self:
        header = Domain('header')
        if meta is not None:
            header.meta.extend(meta)
        _global = Domain('global')
        chapters: list[Domain] = []
        ditches: list[Domain] = []

        if sorter is not None:
            sorter = SimilarImageSorter.create(alg=sorter)

        for id, patterns in reg.items():
            match cls.domain_type({'id': id}):
                case 'header':
                    n = header
                case 'global':
                    n = _global
                case 'chapter':
                    n = Domain(id)
                    chapters.append(n)
                case 'ditch':
                    n = Domain('ditch')
                    ditches.append(n)

            n._sorters.append(GenericSorterCoeff(patterns=tuple(patterns)))

            if sorter is not None:
                n._sorters.append(sorter)

        return cls(header, _global, chapters, ditches)

    @classmethod
    def read(cls, path: Path | str, processor: Optional[str] = None) -> Self:
        with Path(path).open() as of:
            d: list[dict[str, Any]] = yaml.load(of, Loader=yaml.FullLoader)
        processor = get_processor(processor)
        if processor is not None:
            d = processor.process_config(d)
        header = Domain('header')
        _global = Domain('global')
        chapters: list[Domain] = []
        ditches: list[Domain] = []

        for en in d:
            match cls.domain_type(en):
                case 'header':
                    n = header
                case 'global':
                    n = _global
                case 'chapter':
                    n = Domain(cls.get_domain_id(en, ''))
                    chapters.append(n)
                case 'ditch':
                    n = Domain('ditch')
                    ditches.append(n)

            n.meta.extend(en.get('cmd', []))

            for p in cls.get_group(en, []):
                sorter_type: str | None = p.pop('type')
                if sorter_type in ('path', 'image_tags', 'image_group_tags'):
                    patterns = p.get('pattern', None)
                    if isinstance(patterns, str) or patterns is None:
                        patterns = (patterns,)
                    else:
                        patterns = tuple((p,) if isinstance(p, str) else p for p in patterns)

                    keep = cls.container_fmt(p.get('keep', None))
                    ditch = cls.container_fmt(p.get('ditch', None))
                    if sorter_type in ('path', 'path_group'):
                        coef_cls = RegexSorterCoeff
                    elif sorter_type in ('image_tags', 'image_group_tags'):
                        coef_cls = ImageGroupTagSorterCoeff
                    do_group = '_group' in sorter_type
                    kwds = {f: p[f] for f in coef_cls.get_field_names()[3:] if f in p}
                    kwds.setdefault('do_group', do_group)
                    n._sorters.append(coef_cls(patterns, keep, ditch, **kwds))
                else:
                    if sorter_type == 'image':
                        n._sorters.append(SimilarImageSorter.create(**p))
                    elif sorter_type == 'image_group':
                        n._sorters.append(GroupedSimilarImageSorter.create(**p))
                    elif sorter_type == 'image_group_filter':
                        p.setdefault('reference', en.get('image', ''))
                        n._sorters.append(GroupedSimilarImageFilter.create(**p))
                    elif sorter_type is None:
                        n._sorters.append(None)
                    else:
                        raise TypeError(f'Unknown sorter type "{sorter_type}"')

        return cls(header, _global, chapters, ditches)

    def process(
        self, dataset: Iterable[str], callback: Optional[Callable[[Element], str]] = None
    ) -> dict[str, StrGroup]:
        result: dict[str, StrGroup] = collections.defaultdict(list)

        if callback is None:
            callback = lambda e: e.as_str()

        # Collect data to be kept
        ditched = ElementGroup(as_elements(dataset))
        for ch in self.chapters:
            logger.info(f'Start processing Chapter("{ch.id}")...')
            kept, ditched = ch.separated(ditched)
            result[ch.id].extend(map(callback, kept))

        # Collect data to be ditched
        kept = ditched

        for ch in self.ditches:
            logger.info(f'Start processing Chapter("{ch.id}")...')
            ditched, kept = ch.separated(kept)
        result['@body'].extend(map(callback, kept))
        for en in result['@body']:
            logger.debug(f'Secured uncaptured file: {en}')

        return result

    def process_async(
        self, dataset: Iterable[str], callback: Optional[Callable[[Element], str]] = None
    ) -> dict[str, StrGroup]:
        result: dict[str, StrGroup] = collections.defaultdict(StrGroup)

        if callback is None:
            callback = lambda e: e.as_str()

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Collect data to be kept
            ft_d = Future()
            ft_d.set_result(ElementGroup(as_elements(dataset)))
            tasks: list[Future[ElementGroup]] = []
            for ch in self.chapters:
                logger.info(f'Processing Chapter("{ch.id}")')
                ft_k, ft_d = ch.separated_async(ft_d, executor=executor)
                tasks.append(ft_k)
            for ch, task in zip(self.chapters, tasks):
                logger.info(f'Collecting results in Chapter("{ch.id}")')
                k = task.result()
                result[ch.id].extend(map(callback, k))
            tasks.clear()
            # Collect data to be ditched
            ft_k = ft_d
            for ch in self.ditches:
                ft_d, ft_k = ch.separated_async(ft_k, executor=executor)
                tasks.append(ft_k)
            result['@body']
            for ch, task in zip(self.ditches, tasks):
                k = task.result()
                result['@body'].extend(map(callback, k))

        return result

    @staticmethod
    def get_group(en: dict[str, Any], default: Optional[list] = None) -> Iterator[dict[str, Any]]:
        try:
            g: list[str | list[str] | dict[str, Any] | None] = en['group']
        except KeyError as e:
            try:
                logger.warning(
                    DeprecationWarning('Use of `path` instead of `group` has been deprecated.')
                )
                g = en['path']
            except KeyError:
                if default is not None:
                    return default
                raise e
        for sorter in g:
            if isinstance(sorter, dict):
                sorter = {k: v if not isinstance(v, list) else tuple(v) for k, v in sorter.items()}
                if 'type' not in sorter:
                    sorter['type'] = 'path'
                yield sorter
            elif sorter is not None:
                yield {'pattern': sorter, 'type': 'path'}
            else:
                yield {'type': None}

    @classmethod
    def get_domain_id(cls, en: dict[str, Any], default: Optional[str] = None) -> str:
        try:
            return en['id']
        except KeyError as e:
            try:
                logger.warning(
                    DeprecationWarning(
                        'Use of `name` instead of `id` has been deprecated.'
                    )
                )
                return en['name']
            except KeyError:
                if default is None:
                    raise e
                return default

    @classmethod
    def domain_type(
        cls, en: dict[str, Any]
    ) -> Literal['header', 'global', 'chapter', 'ditch']:
        id = cls.get_domain_id(en)
        if not id.startswith('@'):
            return 'chapter'
        if id.startswith('@meta'):
            logger.warning(
                DeprecationWarning(
                    'Use of `@meta` instead of `@header` has been deprecated.'
                )
            )
            return 'header'
        if id.startswith('@header'):
            return 'header'
        if id.startswith('@ditch'):
            return 'ditch'
        if id == '@global':
            return 'global'
        return 'chapter'

    @staticmethod
    def container_fmt(
        container: Optional[Sequence[Optional[str] | Sequence[str]]]
    ) -> list[tuple[str, ...]] | None:
        if container is None:
            return None
        result = []
        for en in container:
            if en is None:
                result.append(None)
            elif isinstance(en, str):
                result.append((en,))
            else:
                result.append(tuple(en))
        return result
