from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import cos, isfinite, pi
from typing import ClassVar, Protocol, Self, final, override

__all__ = [
    "At",
    "Const",
    "Cosine",
    "Linear",
    "Piecewise",
    "Resolver",
    "Schedule",
    "Struct",
]


def _validate_progress(progress: float) -> float:
    if not isfinite(progress):
        raise ValueError(f"progress must be finite, got {progress!r}")
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be in [0, 1], got {progress!r}")
    return progress


class Schedule[T](ABC):
    __slots__: ClassVar[tuple[str, ...]] = ()

    @final
    def at(self, progress: float, /) -> T:
        return self._at(_validate_progress(progress))

    @final
    def __call__(self, progress: float, /) -> T:
        return self.at(progress)

    @abstractmethod
    def _at(self, progress: float, /) -> T:
        ...


@dataclass(frozen=True, slots=True)
class Const[T](Schedule[T]):
    value: T

    @override
    def _at(self, progress: float, /) -> T:
        return self.value


@dataclass(frozen=True, slots=True)
class Linear(Schedule[float]):
    start: float
    stop: float

    @override
    def _at(self, progress: float, /) -> float:
        if progress == 0.0:
            return self.start
        if progress == 1.0:
            return self.stop
        return self.start + (self.stop - self.start) * progress


@dataclass(frozen=True, slots=True)
class Cosine(Schedule[float]):
    start: float
    stop: float

    @override
    def _at(self, progress: float, /) -> float:
        if progress == 0.0:
            return self.start
        if progress == 1.0:
            return self.stop
        weight = (1.0 - cos(pi * progress)) / 2.0
        return self.start + (self.stop - self.start) * weight


@dataclass(frozen=True, order=True, slots=True)
class At:
    progress: float

    def __post_init__(self) -> None:
        _ = _validate_progress(self.progress)

    @classmethod
    def percent(cls, percent: float, /) -> Self:
        if not isfinite(percent):
            raise ValueError(f"percent must be finite, got {percent!r}")
        if not 0.0 <= percent <= 100.0:
            raise ValueError(f"percent must be in [0, 100], got {percent!r}")
        return cls(percent / 100.0)


class Piecewise[T](Schedule[T]):
    __slots__: ClassVar[tuple[str, ...]] = ("_entries", "_starts")

    _entries: tuple[tuple[At, Schedule[T]], ...]
    _starts: tuple[float, ...]

    def __init__(self, phases: Mapping[At, Schedule[T]], /) -> None:
        entries = tuple(sorted(phases.items(), key=lambda entry: entry[0].progress))
        if not entries:
            raise ValueError("Piecewise requires at least one phase")
        if entries[0][0].progress != 0.0:
            raise ValueError("the first Piecewise phase must start at At(0)")

        self._entries = entries
        self._starts = tuple(at.progress for at, _ in entries)

    @property
    def phases(self) -> tuple[tuple[At, Schedule[T]], ...]:
        return self._entries

    @override
    def _at(self, progress: float, /) -> T:
        index = bisect_right(self._starts, progress) - 1
        start = self._starts[index]
        stop = self._starts[index + 1] if index + 1 < len(self._starts) else 1.0

        local_progress = (
            0.0 if stop == start else (progress - start) / (stop - start)
        )
        return self._entries[index][1].at(local_progress)


class Resolver(Protocol):
    def __call__[T](self, schedule: Schedule[T], /) -> T: ...


@dataclass(frozen=True, slots=True)
class _Resolver(Resolver):
    progress: float

    @override
    def __call__[T](self, schedule: Schedule[T], /) -> T:
        return schedule.at(self.progress)


class Struct[T](Schedule[T]):
    __slots__: ClassVar[tuple[str, ...]] = ("_build",)

    _build: Callable[[Resolver], T]

    def __init__(self, build: Callable[[Resolver], T], /) -> None:
        self._build = build

    @override
    def _at(self, progress: float, /) -> T:
        return self._build(_Resolver(progress))
