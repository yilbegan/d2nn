from bisect import bisect_right
from collections.abc import Sequence
from itertools import pairwise
from math import cos, pi
from typing import Annotated, Protocol, overload

from msgspec import Meta

from d2nn.config import Config

__all__ = [
    "Const",
    "Cosine",
    "Linear",
    "Progress",
    "Schedule",
    "Stage",
    "SwitchSpec",
    "ValueSpec",
    "resolve",
]

type Progress = Annotated[float, Meta(ge=0.0, le=1.0)]


class Schedule[T](Protocol):
    def at(self, progress: float, /) -> T: ...


class Const[T](Config, frozen=True, tag_field="kind", tag="const"):
    value: T

    def at(self, _progress: float, /) -> T:
        return self.value


class Linear(Config, frozen=True, tag_field="kind", tag="linear"):
    start: float
    stop: float

    def at(self, progress: float, /) -> float:
        return self.start * (1.0 - progress) + self.stop * progress


class Cosine(Config, frozen=True, tag_field="kind", tag="cosine"):
    start: float
    stop: float

    def at(self, progress: float, /) -> float:
        weight = (1.0 - cos(pi * progress)) / 2.0
        return self.start * (1.0 - weight) + self.stop * weight


class Stage[S](Config, frozen=True):
    at: Progress
    schedule: S


type Value = Const[float] | Linear | Cosine
type ValueSpec = Value | list[Stage[Value]]
type SwitchSpec[T] = Const[T] | list[Stage[Const[T]]]


@overload
def resolve[T](
    schedule: Schedule[T] | Sequence[Stage[Schedule[T]]], progress: float, /
) -> T: ...
@overload
def resolve[T](
    schedule: Schedule[T] | None | Sequence[Stage[Schedule[T] | None]],
    progress: float,
    /,
) -> T | None: ...
def resolve[T](
    schedule: Schedule[T] | None | Sequence[Stage[Schedule[T] | None]],
    progress: float,
    /,
) -> T | None:
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be in [0, 1], got {progress!r}")

    if isinstance(schedule, Sequence):
        starts = [stage.at for stage in schedule]
        if not starts or starts[0] != 0.0 or any(a >= b for a, b in pairwise(starts)):
            raise ValueError(
                "stages must start at 0 and be strictly increasing in `at`"
            )
        index = bisect_right(starts, progress) - 1
        start = starts[index]
        stop = starts[index + 1] if index + 1 < len(starts) else 1.0
        schedule = schedule[index].schedule
        progress = (progress - start) / (stop - start) if stop > start else 0.0

    return None if schedule is None else schedule.at(progress)
