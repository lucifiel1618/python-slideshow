import abc
import collections
import dataclasses
import importlib
import itertools
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Literal, NamedTuple, Optional, Self, Sequence

import yaml

from .utils import get_logger, expand_template

logger = get_logger('SlideShow.Config')


class Sorter:
    @abc.abstractmethod
    def _separated(
        self,
        datasets: Iterable[Iterable[str]],
        ditched: Optional[Iterable[Iterable[str]]] = None
    ) -> tuple[list[list[str]], list[list[str]]]:
        ...

    def separated(self, dataset: Iterable[str]) -> tuple[list[str], list[str]]:
        _kept, _ditched = self._separated((dataset,))
        kept = []
        ditched = []
        for d in _kept:
            kept.extend(d)
        for d in _ditched:
            ditched.extend(d)
        return kept, ditched

    def sorted(self, dataset: Iterable[str]) -> list[str]:
        return self.separated(dataset)[0]


@dataclasses.dataclass(slots=True, frozen=True)
class RegexSorter(Sorter):
    pattern: re.Pattern[str]
    keep: Sequence[Optional[tuple[str]]]
    ditch: Sequence[Optional[tuple[str]]]
    _whitelist_only: bool = dataclasses.field(repr=False)
    _match_only: bool = dataclasses.field(repr=False)

    @classmethod
    def create(
        cls,
        pattern: re.Pattern[str] | str,
        keep: Optional[Sequence[Optional[tuple[str]]]] = None,
        ditch: Optional[Sequence[Optional[tuple[str]]]] = None,
        tokens: Optional[Sequence[re.Pattern[str] | str]] = None
    ) -> Self:

        if tokens is not None:
            if not isinstance(pattern, str):
                pattern = pattern.pattern
            pattern = re.compile(
                expand_template(
                    pattern, (t if isinstance(t, str) else t.pattern for t in tokens)
                )
            )
        elif isinstance(pattern, str):
            pattern = re.compile(pattern)

        if keep is None:
            keep = (None,)
        if ditch is None:
            ditch = (None,)

        keep = tuple(tuple(en) if en is not None else None for en in keep)
        ditch = tuple(tuple(en) if en is not None else None for en in ditch)

        assert all(k not in ditch for k in keep if k is not None), 'Ambiguous whether to keep or ditch'
        whitelist_only = all(en is not None for en in keep)
        match_only = any(en is None for en in ditch)

        return cls(pattern, keep, ditch, whitelist_only, match_only)

    def _key(self, item: str) -> tuple[str, ...] | Literal['ditch']:
        m = self.pattern.fullmatch(item)
        if m is not None:
            k = m.groups()
            if self._whitelist_only and k not in self.keep:
                k = 'ditch'
            elif k in self.ditch:
                k = 'ditch'
            logger.debug(f'⭕ {self.pattern}: {item} << token: {k}')
        else:

            if self._whitelist_only:
                k = 'ditch'
            elif self._match_only:
                k = 'ditch'
            else:
                k = ()
            logger.debug(f'❌ {self.pattern}: {item} << token: {k}')
        return k

    def _separated(
        self,
        datasets: Iterable[Iterable[str]],
        ditched: Optional[Iterable[Iterable[str]]] = None
    ) -> tuple[list[list[str]], list[list[str]]]:
        kept = []
        if ditched is None:
            ditched = [[]]
        for dataset in datasets:
            groups = collections.defaultdict(list)
            for data in dataset:
                groups[self._key(data)].append(data)

            if 'ditch' in groups:
                ditched[0].extend(groups.pop('ditch'))
            for k in self.keep:
                if k is not None:
                    if k in groups:
                        kept.append(groups.pop(k))
                else:
                    rest_keys = sorted(k for k in groups.keys() if k not in self.keep)
                    start_i = 1 if () in rest_keys else 0  # unmatched (i.e. `()`) always at index 0
                    for k in rest_keys[start_i:]:
                        if k in groups:
                            kept.append(groups.pop(k))
                    if start_i == 1:
                        kept.append(groups[()])
        return kept, ditched


@dataclasses.dataclass(frozen=True, slots=True)
class SorterChain:
    sorters: list[Sorter] = dataclasses.field(default_factory=list)

    def _separated(
        self,
        datasets: Iterable[Iterable[str]],
        ditched: Optional[Iterable[Iterable[str]]] = None,
        start: int = 0,
        end: Optional[int] = None
    ) -> tuple[list[list[str]], list[list[str]]]:
        if end is None:
            end = len(self.sorters)

        if ditched is None:
            ditched = [[]]
        if indices := range(start, end):
            for i in indices:
                datasets, ditched = self.sorters[i]._separated(datasets, ditched)
        else:
            datasets = list(list(dataset) for dataset in datasets)
            ditched = list(list(dataset) for dataset in ditched)
        return datasets, ditched

    def separated(
        self,
        dataset: Iterable[str],
        start: int = 0,
        end: Optional[int] = None
    ) -> tuple[list[str], list[str]]:
        _kept, _ditched = self._separated((dataset,), start=start, end=end)
        kept = []
        ditched = []
        for d in _kept:
            kept.extend(d)
        for d in _ditched:
            ditched.extend(d)
        return kept, ditched

    def sorted(
        self,
        dataset: Iterable[str],
        start: int = 0,
        end: Optional[int] = None
    ) -> list[str]:
        return self.separated(dataset, start, end)[0]


class RegexSorterConfig(NamedTuple):
    pattern: Optional[str | Sequence[str]] = None
    keep: Optional[Sequence[Optional[tuple[str]]]] = None
    ditch: Optional[Sequence[Optional[tuple[str]]]] = None

    def updated(self, p: Self) -> Self:
        pattern = self.pattern
        if pattern is None:
            pattern = p.pattern
        elif isinstance(pattern, Sequence):
            assert isinstance(p.pattern, str)
            pattern = expand_template(p.pattern, pattern)
        keep = self.keep
        if keep is None:
            keep = p.keep
        ditch = self.ditch
        if ditch is None:
            ditch = p.ditch
        return type(self)(pattern, keep, ditch)


@dataclasses.dataclass(slots=True)
class Node:
    id: str = ''
    parent: Optional[Self] = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass(slots=True)
class Element(Node):
    _meta: list[str] = dataclasses.field(default_factory=list)
    _patterns: list[RegexSorterConfig] = dataclasses.field(default_factory=list)
    _pattern_indices: list[Optional[int]] = dataclasses.field(default_factory=list, repr=False)

    def add_pattern(self, p: RegexSorterConfig, i: Optional[int] = None) -> None:
        self._patterns.append(p)
        if not isinstance(p.pattern, str):
            assert isinstance(i, int), 'Token pattern must be given a index.'
        self._pattern_indices.append(i)

    @property
    def meta(self) -> list[str]:
        if self.parent is None:
            return self._meta
        return self.parent.meta + self._meta

    def get_patterns(self) -> list[RegexSorterConfig]:
        if self.parent is None:
            return self._patterns[:]
        patterns = []
        for i, p in zip(self._pattern_indices, self._patterns):
            if not isinstance(p.pattern, str):
                p = p.updated(self.parent.get_patterns()[i])
            patterns.append(p)
        return patterns

    def get_sorterchain(self) -> SorterChain:
        sorters = [RegexSorter.create(**p._asdict()) for p in self.get_patterns()]
        return SorterChain(sorters)

    def separated(self, dataset: Iterable[str]) -> tuple[list[str], list[str]]:
        return self.get_sorterchain().separated(dataset)


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

        for e, e_parent in zip(self.elements, self.parent.elements):
            e.parent = e_parent
        self.parent.link()

    def separated(self, dataset: Iterable[str]) -> tuple[list[str], list[str]]:
        kept = []
        ditched = list(dataset)
        for e in self.elements[1:]:
            _kept, ditched = e.separated(ditched)
            kept.extend(_kept)
        return kept, ditched

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
        cls, reg: dict[str, Sequence[str]], meta: Optional[Sequence[str]] = None
    ) -> Self:
        header = Domain()
        if meta is not None:
            header.elements[0]._meta.extend(meta)
        _global = Domain()
        chapters: list[Domain] = []
        ditches: list[Domain] = []

        for id, en in reg.items():
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
                    logger.info(ditches)

            e = n.create_element()
            e.add_pattern(RegexSorterConfig(pattern=en))

        return cls(header, _global, chapters, ditches)

    @classmethod
    def read(cls, path: Path | str, processor: Optional[str] = None) -> Self:
        with Path(path).open() as of:
            d = yaml.load(of, Loader=yaml.FullLoader)
        processor = get_processor(processor)
        if processor is not None:
            d = processor.process_config(d)
        header = Domain()
        _global = Domain()
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
                    n = Domain()
                    ditches.append(n)

            n.elements[0].meta.extend(en.get('cmd', []))

            pattern_chains, kepts, _ditches = [], [], []
            for p in en.get('path', ()):
                if not isinstance(p, dict):
                    p = {'pattern': p}
                patterns = p['pattern']
                if isinstance(patterns, str) or patterns is None:
                    patterns = [patterns]
                else:
                    patterns = [(p,) if isinstance(p, str) else p for p in patterns]
                kepts.append(cls.container_fmt(p.get('keep', None)))
                _ditches.append(cls.container_fmt(p.get('ditch', None)))
                pattern_chains.append(patterns)

            for pattern_chain in itertools.product(*pattern_chains):
                e = n.create_element()
                for i, pattern in enumerate(pattern_chain):
                    e.add_pattern(RegexSorterConfig(pattern, kepts[i], _ditches[i]), i=i)

        return cls(header, _global, chapters, ditches)

    def process(
        self, dataset: Iterable[str], callback: Optional[Callable[[str], str]] = None
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = collections.defaultdict(list)

        ditched = list(dataset)
        for ch in self.chapters:
            kept, ditched = ch.separated(ditched)
            if callback is not None:
                kept = map(callback, kept)
            result[ch.id].extend(kept)
        logger.debug(self.ditches)
        kept = ditched
        for ch in self.ditches:
            ditched, kept = ch.separated(kept)
        if callback is not None:
            kept = map(callback, kept)
        result['@body'].extend(kept)

        return result

    @classmethod
    def get_domain_id(cls, en: dict[str, Any], default: Optional[str] = None) -> str:
        try:
            return en['id']
        except KeyError as e:
            try:
                logger.warning(DeprecationWarning('Use of `name` instead of `id` has been deprecated.'))
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
            logger.warning(DeprecationWarning('Use of `@meta` instead of `@header` has been deprecated.'))
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
    ):
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
