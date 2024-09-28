import abc
import collections
from concurrent.futures import Future, ThreadPoolExecutor
import dataclasses
import functools
import re
import textwrap
from typing import Callable, Generic, Iterable, Literal, NamedTuple, Optional, Self, Sequence, TypeAlias, TypeVar, cast
from .utils import expand_template, get_logger, sampled

logger = get_logger('Slideshow.Sorter')


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
        chunk: Optional[int] = None
    ) -> Self:
        from image_sorter.image_sorter import image_sorted
        sorter_func = functools.partial(
            image_sorted,
            alg=alg,
            threshold=threshold,
            kind=kind,
            ret_key=lambda im: im['path'],
            chunk=chunk
        )
        return cls(sorter_func)

    def _separated(
        self,
        datasets: StrGroups,
        ditched: Optional[StrGroups] = None
    ) -> Pair[StrGroups]:
        if ditched is None:
            ditched = StrGroups()
        kept = StrGroups(map(self.sorter_func, datasets))
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
        kind: str = 'bruteforce'
    ) -> Self:
        from image_sorter.image_sorter import compare_groups
        return cls(functools.partial(compare_groups, alg=alg, threshold=threshold, kind=kind), sample_size)

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
                sampled_group_a = sampled(group_a, sample_size=self.sample_size)
                sampled_groups_b = [sampled(group_b, sample_size=self.sample_size) for group_b in groups_b]
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
        kind: str = 'bruteforce'
    ) -> Self:
        from image_sorter.image_sorter import compare_groups

        return cls(
            functools.partial(compare_groups, alg=alg, threshold=threshold, target='score', kind=kind),
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
            sampled(dataset, sample_size=self.sample_size) if self.sample_size is not None else dataset for dataset in datasets
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


class RegexSorterCoef(NamedTuple):
    patterns: Optional[tuple[str | Sequence[str], ...]] = None
    keep: Optional[Sequence[Optional[tuple[str, ...]]]] = None
    ditch: Optional[Sequence[Optional[tuple[str, ...]]]] = None
    key_fmt: Optional[str] = None

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
                            pattern_list.append(expand_template(pattern_p, cast(Sequence[str], pattern)))
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
        return type(self)(patterns, keep, ditch, key_fmt)

    def updated_by(self, c: Self) -> Self:
        return c.updated(self)

    @staticmethod
    def status(pattern: Optional[str | Sequence[str]]) -> Literal['independent', 'dependent', 'copy']:
        if pattern is None:
            return 'copy'
        elif isinstance(pattern, str):
            return 'independent'
        return 'dependent'


@dataclasses.dataclass(slots=True, frozen=True)
class RegexSorter(Sorter):
    patterns: tuple[re.Pattern[str], ...]
    keep: Sequence[Optional[tuple[str, ...]]]
    ditch: Sequence[Optional[tuple[str, ...]]]
    key_fmt: Optional[str]
    _whitelist_only: bool = dataclasses.field(repr=False)
    _match_only: bool = dataclasses.field(repr=False)
    keep_all: bool = dataclasses.field(repr=False)

    @classmethod
    def create(
        cls,
        patterns: Iterable[re.Pattern[str] | str],
        keep: Optional[Sequence[Optional[tuple[str, ...]]]] = None,
        ditch: Optional[Sequence[Optional[tuple[str, ...]]]] = None,
        key_fmt: Optional[str] = None,
        tokens: Optional[Sequence[re.Pattern[str] | str]] = None
    ) -> Self:

        pattern_list: list[re.Pattern[str]] = []
        for pattern in patterns:
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
            pattern_list.append(pattern)

        keep = (None,) if keep is None else tuple(tuple(en) if en is not None else None for en in keep)
        ditch = (None,) if ditch is None else tuple(tuple(en) if en is not None else None for en in ditch)
        assert all(k not in ditch for k in keep if k is not None), 'Ambiguous whether to keep or ditch'
        whitelist_only = all(en is not None for en in keep)
        match_only = any(en is None for en in ditch)
        keep_all = not ditch

        return cls(tuple(pattern_list), keep, ditch, key_fmt, whitelist_only, match_only, keep_all)

    def _key(self, item: str) -> tuple[str, ...] | Literal['ditch']:
        k = 'ditch'
        for pattern in self.patterns:
            m = pattern.fullmatch(item)
            if m is not None:
                k = m.groups()
                if self._whitelist_only and k not in self.keep:
                    k = 'ditch'
                elif k in self.ditch:
                    k = 'ditch'
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
        return Pair(kept, ditched)


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
