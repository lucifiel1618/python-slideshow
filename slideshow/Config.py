import collections
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
import importlib
import itertools
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Self, Sequence

import yaml

from .utils import get_logger
from .Sorter import FuturePair, GroupedSimilarImageSorter, Pair, SimilarImageSorter, Sorter, RegexSorter, RegexSorterConfig, SorterChain, StrGroup, StrGroups, collect_futures, flatten_futures, wrap_future

logger = get_logger('Slideshow.Config')


@dataclasses.dataclass(slots=True)
class Node:
    id: str = ''
    parent: Optional[Self] = dataclasses.field(default=None)


@dataclasses.dataclass(slots=True)
class Element(Node):
    _meta: list[str] = dataclasses.field(default_factory=list)
    _patterns: dict[int, list[RegexSorterConfig]] = dataclasses.field(default_factory=dict)
    _extra_sorters: dict[int, Optional[Sorter]] = dataclasses.field(default_factory=dict)

    def add_pattern(self, p: RegexSorterConfig, i: int = -1) -> None:
        if not isinstance(p.pattern, str):
            assert i >= 0, 'Token pattern must be given a index.'
        self._patterns.setdefault(i, []).append(p)

    def add_sorter(self, sorter: Sorter, i: int = -1) -> None:
        self._extra_sorters[i] = sorter

    @property
    def meta(self) -> list[str]:
        if self.parent is None:
            return self._meta
        return self.parent.meta + self._meta

    def get_patterns(self) -> list[tuple[int, RegexSorterConfig]]:
        patterns: list[tuple[int, RegexSorterConfig]] = []
        for i in self._patterns:
            for p in self.iter_pattern(i):
                patterns.append((i, p))
        return patterns

    def iter_pattern(self, i: int) -> Iterator[RegexSorterConfig]:
        if i not in self._patterns:
            return
        _patterns = self._patterns[i]
        for p in _patterns:
            if not isinstance(p.pattern, str):
                try:
                    p = p.updated(next(self.parent.iter_pattern(i)))
                except AttributeError as e:
                    print(f'{self}')
                    raise e
            yield p

    def iter_regex_sorter(self, i: int) -> Iterator[RegexSorter]:
        for p in self.iter_pattern(i):
            yield RegexSorter.create(**p._asdict())

    def get_regex_sorters(self) -> list[tuple[int, RegexSorter]]:
        regex_sorters = []
        for i in self._patterns:
            for sorter in self.iter_regex_sorter(i):
                regex_sorters.append((i, sorter))
        return regex_sorters

    def iter_extra_sorter(self, i: int) -> Iterator[Sorter]:
        if i not in self._extra_sorters:
            return
        sorter = self._extra_sorters[i]
        if sorter is not None:
            yield sorter
        else:
            assert self.parent is not None
            yield from self.parent.iter_regex_sorter(i)
            yield from self.parent.iter_extra_sorter(i)

    def get_extra_sorters(self) -> list[tuple[int, Sorter]]:
        '''
        Extra sorters include fully inherited RegexSorters as well
        '''
        extra_sorters: list[tuple[int, Sorter]] = []
        for i in self._extra_sorters:
            extra_sorters.extend(((i, sorter) for sorter in self.iter_extra_sorter(i)))
        return extra_sorters

    def get_sorters(self) -> list[Sorter]:
        sorters: list[Sorter] = [
            sorter for _, sorter in sorted(self.get_regex_sorters() + self.get_extra_sorters())
        ]
        return sorters

    def get_sorterchain(self) -> SorterChain:
        return SorterChain(self.get_sorters())

    def separated(self, dataset: Iterable[str]) -> Pair[StrGroup]:
        return self.get_sorterchain().separated(dataset)

    def _separated_async(
        self,
        ft_in: Future[StrGroups],
        executor: ThreadPoolExecutor
    ) -> FuturePair[StrGroups]:
        return self.get_sorterchain()._separated_async(FuturePair.create(ft_in, StrGroups([[]])), executor=executor)

    def separated_async(
        self,
        ft_in: Future[StrGroup],
        executor: ThreadPoolExecutor
    ) -> FuturePair[StrGroup]:
        wrapped_ft_in: Future[StrGroups] = executor.submit(wrap_future, ft_in)
        fts = self._separated_async(wrapped_ft_in, executor=executor)
        return FuturePair(*(executor.submit(flatten_futures, ft) for ft in fts))


@dataclasses.dataclass(slots=True)
class Domain(Node):
    elements: list[Element] = dataclasses.field(default_factory=lambda: [Element(id='base')])
    is_linked: bool = False

    def create_element(self, *args, **kwds) -> Element:
        e = Element(*args, **kwds)
        self.elements.append(e)
        return e

    def link(self) -> None:
        if self.is_linked:
            return
        self.is_linked = True
        if self.parent is None:
            return
        self.elements[0].parent = self.parent.elements[0]
        if len(self.parent.elements) > 1:
            assert len(self.parent.elements) == 2
            for e in self.elements[1:]:
                e.parent = self.parent.elements[1]
        self.parent.link()

    def separated(self, dataset: Iterable[str]) -> Pair[StrGroup]:
        k_out = StrGroup()
        d = StrGroup(dataset)
        for e in self.elements[1:]:
            logger.debug(f'{e=}')
            k, d = e.separated(d)
            k_out.extend(k)
            logger.debug(f'{d=}')
        return Pair(k_out, d)

    def separated_async(self, ft_in: Future[StrGroup], executor: ThreadPoolExecutor) -> FuturePair[StrGroup]:
        ft_d = ft_in
        fts_k: list[Future[StrGroup]] = []
        for e in self.elements[1:]:
            ft_k, ft_d = e.separated_async(ft_d, executor=executor)
            fts_k.append(ft_k)
        fts_out = FuturePair(executor.submit(collect_futures, *fts_k), ft_d)
        return fts_out

    def combined(self, b: Self) -> Self:
        combined = Domain()
        combined.elements[0]._meta = self.elements[0]._meta + b.elements[0]._meta
        for element in self.elements[1:] + b.elements[1:]:
            combined.elements.append(element)
        return combined


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
            ch.link()

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
        header = Domain()
        if meta is not None:
            header.elements[0]._meta.extend(meta)
        _global = Domain()
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
                    n = Domain()
                    ditches.append(n)

            for p in patterns:
                e = n.create_element()
                e.add_pattern(RegexSorterConfig(pattern=p))

            if sorter is not None:
                n.create_element().add_sorter(sorter)

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

            n.elements[0].meta.extend(en.get('cmd', []))

            pattern_chains, kepts, _ditches = [], [], []

            for p in cls.get_group(en, []):
                sorter_type: str | None = p.pop('type')
                if sorter_type == 'path':
                    patterns = p['pattern']
                    if isinstance(patterns, str) or patterns is None:
                        patterns = [patterns]
                    else:
                        patterns = [(p,) if isinstance(p, str) else p for p in patterns]
                    kepts.append(cls.container_fmt(p.get('keep', None)))
                    _ditches.append(cls.container_fmt(p.get('ditch', None)))
                    pattern_chains.append(patterns)
                else:
                    if sorter_type == 'image':
                        pattern_chains.append([SimilarImageSorter.create(**p)])
                    elif sorter_type == 'image_group':
                        pattern_chains.append([GroupedSimilarImageSorter.create(**p)])
                    elif sorter_type is None:
                        pattern_chains.append([None])
                    kepts.append(None)
                    _ditches.append(None)
            logger.debug(f'{pattern_chains=}')
            if not pattern_chains:
                # itertools.product(*[]) -> [()] instead of [] for some reasons
                # Hence such special case needs to be handled specifically
                logger.debug('skipping empty chains')
                continue
            for pattern_chain in itertools.product(*pattern_chains):
                logger.debug(f'{pattern_chain=}')
                e = n.create_element()
                for i, pattern in enumerate(pattern_chain):
                    if isinstance(pattern, Sorter) or pattern is None:
                        e.add_sorter(pattern, i=i)
                    else:
                        e.add_pattern(RegexSorterConfig(pattern, kepts[i], _ditches[i]), i=i)
                print(e)

        return cls(header, _global, chapters, ditches)

    def process(
        self, dataset: Iterable[str], callback: Optional[Callable[[str], str]] = None
    ) -> dict[str, StrGroup]:
        result: dict[str, StrGroup] = collections.defaultdict(list)

        # Collect data to be kept
        ditched = StrGroup(dataset)
        for ch in self.chapters:
            logger.info(f'Start processing Chapter("{ch.id}")...')
            kept, ditched = ch.separated(ditched)
            if callback is not None:
                kept = map(callback, kept)
            result[ch.id].extend(kept)

        # Collect data to be ditched
        kept = ditched

        for ch in self.ditches:
            ditched, kept = ch.separated(kept)
        if callback is not None:
            kept = map(callback, kept)
        result['@body'].extend(kept)
        for en in result['@body']:
            logger.debug(f'Secured uncaptured file: {en}')

        return result

    def process_async(
        self, dataset: Iterable[str], callback: Optional[Callable[[str], str]] = None
    ) -> dict[str, StrGroup]:
        result: dict[str, StrGroup] = collections.defaultdict(StrGroup)

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Collect data to be kept
            ft_d = Future()
            ft_d.set_result(StrGroup(dataset))
            tasks: list[Future[StrGroup]] = []
            for ch in self.chapters:
                logger.info(f'Processing Chapter("{ch.id}")')
                ft_k, ft_d = ch.separated_async(ft_d, executor=executor)
                tasks.append(ft_k)
            for ch, task in zip(self.chapters, tasks):
                logger.info(f'Collecting results in Chapter("{ch.id}")')
                k = task.result()
                if callback is not None:
                    k = map(callback, k)
                result[ch.id].extend(k)
            tasks.clear()
            # Collect data to be ditched
            ft_k = ft_d
            for ch in self.ditches:
                ft_d, ft_k = ch.separated_async(ft_k, executor=executor)
                tasks.append(ft_k)
            result['@body']
            for ch, task in zip(self.ditches, tasks):
                k = task.result()
                if callback is not None:
                    k = map(callback, k)
                result['@body'].extend(k)

        return result

    @staticmethod
    def get_group(en: dict[str, Any], default: Optional[list] = None) -> list[dict[str, Any]]:
        try:
            g: list[str | list[str] | dict[str, Any] | None] = en['group']
        except KeyError as e:
            try:
                logger.warning(
                    DeprecationWarning(
                        'Use of `path` instead of `group` has been deprecated.'
                    )
                )
                g = en['path']
            except KeyError:
                if default is not None:
                    return default
                raise e
        return [en if isinstance(en, dict) else {'pattern': en, 'type': 'path'} if en is not None else {'type': None} for en in g]

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
