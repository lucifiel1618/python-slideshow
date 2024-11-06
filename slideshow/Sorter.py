import abc
import collections
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
import functools
import re
import textwrap
from typing import (
    Any, Callable, Generic, Iterable, Literal, NamedTuple, Optional, Protocol, Self, Sequence, Type, TypeAlias,
    TypeVar, cast
)

from . import utils

logger = utils.get_logger('Slideshow.Sorter')


T = TypeVar('T')
StrGroup: TypeAlias = list[str]
StrGroups: TypeAlias = list[StrGroup]


class Pair(NamedTuple, Generic[T]):
    k: T
    d: T


class FuturePair(Pair[Future[T]]):
    @classmethod
    def create(
        cls,
        result_k: Optional[T | Future[T]] = None,
        result_d: Optional[T | Future[T]] = None
    ) -> Self:
        if isinstance(result_k, Future):
            k = result_k
        else:
            k = Future()
            if result_k is not None:
                k.set_result(result_k)

        if isinstance(result_d, Future):
            d = result_d
        else:
            d = Future()
            if result_d is not None:
                d.set_result(result_d)
        return cls(k, d)


def collect_futures(*fts_in: Future[Sequence[T]]) -> list[T]:
    result: list[T] = []
    for ft in fts_in:
        result.extend(ft.result())
    return result


def flatten_futures(*fts_in: Future[Sequence[Sequence[T]]]) -> Sequence[T]:
    result: list[T] = []
    for ft in fts_in:
        for g in ft.result():
            result.extend(g)
    return result


def wrap_future(ft: Future[T]) -> Sequence[T]:
    result = [ft.result()]
    return result


class Sorter:
    @property
    @abc.abstractmethod
    def keep_all(self) -> bool:
        ...

    @abc.abstractmethod
    def _separated(
        self,
        datasets: Iterable[Iterable[str]],
        ditched: Optional[Iterable[Iterable[str]]] = None
    ) -> Pair[StrGroups]:
        ...

    def _separated_ft(self, fts_in: FuturePair[StrGroups], fts_out: FuturePair[StrGroups]) -> None:
        d_in = fts_in.d.result()
        if self.keep_all:
            fts_out.d.set_result(d_in)
        k_in = fts_in.k.result()
        k_out, d_out = self._separated(k_in, d_in)
        fts_out.k.set_result(k_out)
        if not self.keep_all:
            fts_out.d.set_result(d_out)

    def _separated_async(
        self,
        fts_in: FuturePair[StrGroups],
        executor: ThreadPoolExecutor
    ) -> FuturePair[StrGroups]:
        fts_out: FuturePair[StrGroups] = FuturePair.create()
        executor.submit(self._separated_ft, fts_in, fts_out)
        return fts_out

    def separated(self, dataset: Iterable[str]) -> Pair[StrGroup]:
        sgs_pair = self._separated((dataset,))
        sg_pair = Pair(StrGroup(), StrGroup())

        for sg_en, sgs_en in zip(sg_pair, sgs_pair):
            for data in sgs_en:
                sg_en.extend(data)

        return sg_pair

    def sorted(self, dataset: Iterable[str]) -> StrGroup:
        return self.separated(dataset).k


@dataclasses.dataclass(slots=True, frozen=True)
class SimilarImageSorter(Sorter):
    sorter_func: Callable[[Iterable[str]], StrGroups] = dataclasses.field(repr=False)
    keep_all: bool = dataclasses.field(default=True, repr=False)

    @classmethod
    def create(
        cls,
        alg: str = 'pixelwise',
        threshold: Optional[float] = None,
        kind: str = 'primary',
        chunk: Optional[int] = None,
        target: str = 'score'
    ) -> Self:
        from image_sorter.image_sorter import image_sorted
        sorter_func = functools.partial(
            image_sorted,
            alg=alg,
            threshold=threshold,
            kind=kind,
            ret_key=lambda im: im['path'],
            chunk=chunk,
            target=target
        )
        return cls(sorter_func)

    def _separated(
        self,
        datasets: StrGroups,
        ditched: Optional[StrGroups] = None
    ) -> Pair[StrGroups]:
        if ditched is None:
            ditched = StrGroups()
        kept = StrGroups(*map(self.sorter_func, datasets))
        return Pair(kept, ditched)


@dataclasses.dataclass(slots=True, frozen=True)
class GroupedSimilarImageSorter(Sorter):
    compare_func: Callable[[Sequence[str], *tuple[Sequence[str]]], Sequence[bool]]
    sample_size: Optional[int] = None
    keep_all: bool = dataclasses.field(default=True, repr=False)
    kind: str = 'bruteforce'

    @classmethod
    def create(
        cls,
        alg: str = 'pixelwise',
        threshold: Optional[float] = None,
        sample_size: Optional[int] = None,
        kind: str = 'bruteforce',
        target: str = 'score',
        inverted: bool = False
    ) -> Self:
        from image_sorter.image_sorter import compare_groups
        fn = functools.partial(compare_groups, alg=alg, threshold=threshold, kind=kind, target=target)
        if not inverted:
            compare_func = fn
        else:
            @functools.wraps(fn)
            def compare_func(*args, **kwds):
                return [not x for x in fn(*args, **kwds)]
        return cls(
            compare_func,
            sample_size
        )

    def _separated(
        self,
        datasets: StrGroups,
        ditched: Optional[StrGroups] = None
    ) -> Pair[StrGroups]:

        kept = StrGroups()
        if ditched is None:
            ditched = StrGroups()
        out_pair = Pair(kept, ditched)

        for i, group_a in enumerate(datasets):
            # print('>>>', group_a)
            if group_a in kept:
                continue
            kept.append(group_a)
            groups_b = datasets[i + 1:]

            if self.sample_size is not None:
                sampled_group_a = utils.sampled(group_a, sample_size=self.sample_size)
                sampled_groups_b = [utils.sampled(group_b, sample_size=self.sample_size) for group_b in groups_b]
            else:
                sampled_group_a = group_a
                sampled_groups_b = groups_b

            for i, r in enumerate(self.compare_func(sampled_group_a, *sampled_groups_b)):
                if r:
                    msg_lines = [
                        'Match found!',
                        textwrap.indent(str(sampled_group_a), ' ' * 4),
                        *(textwrap.indent(str(sampled_group_b), ' ' * 8) for sampled_group_b in sampled_groups_b)
                    ]
                    logger.debug('\n'.join(msg_lines))
                    kept.append(groups_b[i])
        logger.debug(f'result: {kept}')
        return out_pair


@dataclasses.dataclass(slots=True, frozen=True)
class GroupedSimilarImageFilter(GroupedSimilarImageSorter):
    keep_all: bool = dataclasses.field(default=False, repr=False)
    reference: str = ''
    kind: str = 'bruteforce'

    @classmethod
    def create(
        cls,
        reference: str,
        alg: str = 'ccip',
        threshold: Optional[float] = None,
        sample_size: Optional[int] = None,
        kind: str = 'bruteforce',
        target: str = 'score',
        inverted: bool = False
    ) -> Self:
        from image_sorter.image_sorter import compare_groups

        fn = functools.partial(compare_groups, alg=alg, threshold=threshold, kind=kind, target=target)
        if not inverted:
            compare_func = fn
        else:
            @functools.wraps(fn)
            def compare_func(*args, **kwds):
                return [not x for x in fn(*args, **kwds)]

        return cls(
            functools.partial(compare_func, alg=alg, threshold=threshold, kind=kind, target=target),
            sample_size,
            reference=reference
        )

    def _separated(
        self,
        datasets: StrGroups,
        ditched: Optional[StrGroups] = None
    ) -> Pair[StrGroups]:
        kept = StrGroups()
        if ditched is None:
            ditched = StrGroups()
        out_pair = Pair(kept, ditched)

        groups_b = datasets

        sampled_groups_b = [
            utils.sampled(dataset, sample_size=self.sample_size) if self.sample_size is not None else dataset for dataset in datasets
        ]

        _ditched: StrGroups = StrGroups()
        for i, r in enumerate(self.compare_func((self.reference,), *sampled_groups_b)):
            if r:
                msg_lines = [
                    'Match found!',
                    textwrap.indent(str(self.reference), ' ' * 4),
                    *(textwrap.indent(str(sampled_group_b), ' ' * 8) for sampled_group_b in sampled_groups_b)
                ]
                logger.debug('\n'.join(msg_lines))
                kept.append(groups_b[i])
            else:
                _ditched.append(groups_b[i])

        if self.keep_all:
            kept.extend(_ditched)
        else:
            ditched.extend(_ditched)
        logger.debug(f'result: {kept}')
        return out_pair


class GenericPattern(Protocol):
    class GenericMatch(Protocol):
        def groups(self) -> tuple[str]:
            raise NotImplementedError

    def fullmatch(self, target: str, *targets: str) -> GenericMatch | None:
        raise NotImplementedError


@dataclasses.dataclass(slots=True, frozen=True)
class GenericSorter(Sorter):
    patterns: tuple[GenericPattern, ...]
    keep: Sequence[Optional[tuple[str, ...]]]
    ditch: Sequence[Optional[tuple[str, ...]]]
    key_fmt: Optional[str]
    _whitelist_only: bool = dataclasses.field(repr=False)
    _match_only: bool = dataclasses.field(repr=False)
    keep_all: bool = dataclasses.field(repr=False)
    sample_size: Optional[int] = None
    do_group: bool = False

    @classmethod
    def _create_patterns(
        cls,
        patterns: Iterable[GenericPattern | str],
        tokens: Optional[Sequence[GenericPattern | str]] = None
    ) -> list[GenericPattern]:
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        patterns: Iterable[GenericPattern | str],
        keep: Optional[Sequence[Optional[tuple[str, ...]]]] = None,
        ditch: Optional[Sequence[Optional[tuple[str, ...]]]] = None,
        key_fmt: Optional[str] = None,
        tokens: Optional[Sequence[GenericPattern | str]] = None,
        sample_size: Optional[int] = None,
        do_group: bool = False
    ) -> Self:

        pattern_list = cls._create_patterns(patterns, tokens)
        keep = (None,) if keep is None else tuple(tuple(en) if en is not None else None for en in keep)
        ditch = (None,) if ditch is None else tuple(tuple(en) if en is not None else None for en in ditch)
        assert all(k not in ditch for k in keep if k is not None), 'Ambiguous whether to keep or ditch'
        whitelist_only = all(en is not None for en in keep)
        match_only = any(en is None for en in ditch)
        keep_all = not ditch

        return cls(
            tuple(pattern_list), keep, ditch, key_fmt, whitelist_only, match_only, keep_all, sample_size, do_group
        )

    def _to_keep(self, k: tuple[str, ...] | None) -> tuple[str, ...] | None | Literal[False]:
        if k in self.keep:
            return k
        return False

    def _to_ditch(self, k: tuple[str, ...] | None) -> bool:
        return (k in self.ditch)

    def _key(self, item: str, *items: str) -> tuple[str, ...] | Literal['ditch'] | None:
        k = 'ditch'
        for pattern in self.patterns:
            m = pattern.fullmatch(item, *items)
            if m is not None:
                g = m.groups()
                k = self._to_keep(g)
                if self._whitelist_only and (k is False):
                    k = 'ditch'
                elif self._to_ditch(g):
                    k = 'ditch'
                elif k is False:
                    k = g
            else:
                if self._whitelist_only:
                    k = 'ditch'
                elif self._match_only:
                    k = 'ditch'
                else:
                    k = ()
            if (self.key_fmt is not None) and (k not in ((), 'ditch')):
                k = (self.key_fmt.format(*k),)
            if k != 'ditch':
                log_str = '' if k == () else f' <<< token: {k}'
                logger.debug(f'⭕ {pattern}: {item}{log_str}')
                break
            logger.debug(f'❌ {pattern}: {item}')
        return k

    def _separated(
        self,
        datasets: StrGroups,
        ditched: Optional[StrGroups] = None
    ) -> Pair[StrGroups]:
        kept = []
        if ditched is None:
            ditched = StrGroups([StrGroup()])
        for dataset in datasets:
            groups = collections.defaultdict(StrGroup)
            if not self.do_group:
                for data in dataset:
                    groups[self._key(data)].append(data)
            else:
                groups[self._key(*utils.sampled(dataset, self.sample_size))].extend(dataset)

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
        return Pair(kept, ditched)


@dataclasses.dataclass(slots=True, frozen=True)
class GenericSorterCoeff:
    patterns: Optional[tuple[str | Sequence[str] | Sequence[Sequence[str]], ...]] = None
    keep: Optional[Sequence[Optional[tuple[str, ...]]]] = None
    ditch: Optional[Sequence[Optional[tuple[str, ...]]]] = None
    key_fmt: Optional[str] = None
    sample_size: Optional[int] = None
    do_group: Optional[bool] = None

    def updated(self, p: Self) -> Self:
        if (patterns := self.patterns) is None:
            patterns = p.patterns
        else:
            assert p.patterns is not None
            assert all(self.status(pattern) == 'independent' for pattern in p.patterns)
            pattern_list = []
            for pattern in patterns:
                for pattern_p in p.patterns:
                    match self.status(pattern):
                        case 'copy':
                            assert isinstance(pattern_p, str)
                            pattern_list.append(pattern_p)
                        case 'dependent':
                            assert isinstance(pattern_p, str)
                            pattern_list.append(utils.expand_template(pattern_p, cast(Sequence[str], pattern)))
                        case 'independent':
                            pattern_list.append(pattern)
                            break
            patterns = tuple(pattern_list)

        if (keep := self.keep) is None:
            keep = p.keep
        if (ditch := self.ditch) is None:
            ditch = p.ditch
        if (key_fmt := self.key_fmt) is None:
            key_fmt = p.key_fmt
        if (sample_size := self.sample_size) is None:
            sample_size = p.sample_size
        if (do_group := self.do_group) is None:
            do_group = p.do_group
        return type(self)(patterns, keep, ditch, key_fmt, sample_size, do_group)

    def updated_by(self, c: Self) -> Self:
        return c.updated(self)

    @classmethod
    def get_field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__)

    def asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def status(pattern: Optional[str | Sequence[str]]) -> Literal['independent', 'dependent', 'copy']:
        if pattern is None:
            return 'copy'
        elif isinstance(pattern, str):
            return 'independent'
        return 'dependent'

    @classmethod
    def get_sorter_cls(cls) -> Type[GenericSorter]:
        raise NotImplementedError


@dataclasses.dataclass(slots=True, frozen=True)
class RegexPattern:
    pattern: re.Pattern[str]

    @dataclasses.dataclass(slots=True, frozen=True)
    class RegexMatch:
        _groups: tuple[str, ...]

        def groups(self) -> tuple[str, ...]:
            return self._groups

    @classmethod
    def create(cls, pattern: re.Pattern[str]) -> Self:
        return cls(pattern)

    def fullmatch(self, target: str, *targets: str) -> RegexMatch | None:
        m = self.pattern.fullmatch(target)
        if m is None:
            return None
        groups = m.groups()
        for t in targets:
            m = self.pattern.fullmatch(t)
            if m is None:
                return None
            if groups != m.groups():
                return None
        return self.RegexMatch(groups)


@dataclasses.dataclass(slots=True, frozen=True)
class RegexSorter(GenericSorter):
    @classmethod
    def _create_patterns(
        cls,
        patterns: Iterable[RegexPattern | str],
        tokens: Optional[Sequence[re.Pattern[str] | str]] = None
    ) -> list[RegexPattern]:
        pattern_list: list[RegexPattern] = []
        for pattern in patterns:
            if tokens is not None:
                p = pattern.pattern.pattern if not isinstance(pattern, str) else pattern
                re_pattern = re.compile(
                    utils.expand_template(
                        p, (t if isinstance(t, str) else t.pattern for t in tokens)
                    )
                )
                pattern = RegexPattern.create(re_pattern)
            elif isinstance(pattern, str):
                re_pattern = re.compile(pattern)
                pattern = RegexPattern.create(re_pattern)
            pattern_list.append(pattern)
        return pattern_list


@dataclasses.dataclass(slots=True, frozen=True)
class RegexSorterCoeff(GenericSorterCoeff):
    @classmethod
    def get_sorter_cls(cls) -> Type[RegexSorter]:
        return RegexSorter


@dataclasses.dataclass(slots=True, frozen=True)
class ImageTagPattern:
    _fn: Callable[[str], set[str]]

    @dataclasses.dataclass(slots=True, frozen=True)
    class ImageTagMatch:
        _groups: tuple[str, ...]

        def groups(self) -> tuple[str, ...]:
            return self._groups

    @classmethod
    def create(cls, models: Sequence[str | None]) -> Self:
        from image_sorter.image_tagging import get_image_tagger
        return cls(get_image_tagger(models))

    def fullmatch(self, target: str, *targets: str) -> ImageTagMatch | None:
        m = self._fn(target)
        m.update(*(self._fn(t) for t in targets))
        if not m:
            return None
        return self.ImageTagMatch(tuple(m))


@dataclasses.dataclass(slots=True, frozen=True)
class ImageGroupTagSorter(GenericSorter):
    @classmethod
    def _create_patterns(
        cls,
        patterns: Iterable[ImageTagPattern | Iterable[str] | None],
        tokens: Literal[None] = None
    ) -> list[ImageTagPattern]:
        assert tokens is None
        return [p if isinstance(p, ImageTagPattern) else ImageTagPattern.create(tuple(p) if p is not None else (None,)) for p in patterns]

    def _to_keep(self, k: tuple[str, ...] | None) -> tuple[str, ...] | None | Literal[False]:
        assert k is not None
        for _k in self.keep:
            if _k is None:
                return None
            if set(_k).issubset(k):
                return _k
        return False

    def _to_ditch(self, k: tuple[str, ...] | None) -> bool:
        assert k is not None
        for _k in self.ditch:
            if _k is None:
                continue
            if set(_k).issubset(k):
                return True
        return False

    @classmethod
    def create(
        cls,
        patterns: Iterable[ImageTagPattern | Iterable[str]],
        keep: Optional[Sequence[tuple[str, ...] | None]] = None,
        ditch: Optional[Sequence[tuple[str, ...] | None]] = None,
        key_fmt: Optional[str] = None,
        tokens: Optional[Sequence[GenericPattern | str]] = None,
        sample_size: Optional[int] = None,
        do_group: bool = False
    ) -> Self:
        assert key_fmt is None
        assert tokens is None
        return super(ImageGroupTagSorter, cls).create(
            patterns, keep, ditch, key_fmt, tokens, sample_size=sample_size, do_group=do_group
        )


@dataclasses.dataclass(slots=True, frozen=True)
class ImageGroupTagSorterCoeff(GenericSorterCoeff):
    sample_size: Optional[int] = None

    @classmethod
    def get_sorter_cls(cls) -> Type[ImageGroupTagSorter]:
        return ImageGroupTagSorter


@dataclasses.dataclass(frozen=True, slots=True)
class SorterChain:
    sorters: list[Sorter] = dataclasses.field(default_factory=list)

    def _separated(
        self,
        datasets: StrGroups,
        ditched: Optional[StrGroups] = None,
        *,
        start: int = 0,
        end: Optional[int] = None
    ) -> Pair[StrGroups]:
        for sorter in self.sorters[start:end]:
            datasets, ditched = sorter._separated(datasets, ditched)
        if ditched is None:
            ditched = StrGroups()
        return Pair(datasets, ditched)

    def _separated_async(
        self,
        fts_in: FuturePair[StrGroups],
        executor: ThreadPoolExecutor,
        *,
        start: int = 0,
        end: Optional[int] = None
    ) -> FuturePair[StrGroups]:
        fts_out = fts_in
        for sorter in self.sorters[start:end]:
            fts_out = sorter._separated_async(fts_out, executor=executor)
        return fts_out

    def separated(
        self,
        dataset: Iterable[str],
        *,
        start: int = 0,
        end: Optional[int] = None
    ) -> Pair[StrGroup]:
        pair_in = self._separated(
            StrGroups([StrGroup(dataset)]), start=start, end=end
        )
        pair_out = Pair(StrGroup(), StrGroup())
        for k in pair_in.k:
            pair_out.k.extend(k)
        for d in pair_in.d:
            pair_out.d.extend(d)
        return pair_out

    def sorted(
        self,
        dataset: Iterable[str],
        *,
        start: int = 0,
        end: Optional[int] = None
    ) -> StrGroup:
        return self.separated(dataset, start=start, end=end).k
