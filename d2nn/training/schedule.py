from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import cos, isfinite, pi
from typing import ClassVar, Protocol, Self, cast, final, override

__all__ = [
    "At",
    "Const",
    "Cosine",
    "from_dict",
    "Linear",
    "Piecewise",
    "Resolver",
    "Schedule",
    "Struct",
    "StructRegistry",
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
    def _at(self, progress: float, /) -> T: ...


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

        local_progress = 0.0 if stop == start else (progress - start) / (stop - start)
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


type StructRegistry = Mapping[str, Callable[..., object]]


def _number(config: Mapping[str, object], key: str) -> float:
    try:
        value = config[key]
    except KeyError:
        raise ValueError(
            f"missing {key!r} for {config.get('kind')!r} schedule"
        ) from None

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key!r} must be a number, got {value!r}")
    return float(value)


def _phase_start(value: object) -> At:
    if isinstance(value, str):
        value = value.strip()
        if not value.endswith("%"):
            raise ValueError(f"phase 'at' must be a percentage, got {value!r}")
        try:
            return At.percent(float(value[:-1]))
        except ValueError as error:
            raise ValueError(f"invalid phase 'at' value {value!r}") from error

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(
            f"phase 'at' must be a percentage or progress number, got {value!r}"
        )
    return At(float(value))


def _struct(config: Mapping[str, object], registry: StructRegistry) -> Struct[object]:
    name = config.get("name")
    if not isinstance(name, str):
        raise TypeError(f"struct 'name' must be a string, got {name!r}")

    try:
        factory = registry[name]
    except KeyError:
        raise ValueError(f"unknown struct {name!r}") from None

    values = config.get("values")
    if not isinstance(values, Mapping):
        raise TypeError(f"struct 'values' must be a mapping, got {values!r}")
    values = cast(Mapping[object, object], values)

    fields: dict[str, Schedule[object]] = {}
    for field, value in values.items():
        if not isinstance(field, str):
            raise TypeError(f"struct field names must be strings, got {field!r}")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"schedule for struct field {field!r} must be a mapping, got {value!r}"
            )
        fields[field] = from_dict(cast(Mapping[str, object], value), registry)

    def build(resolve: Resolver) -> object:
        return factory(
            **{field: resolve(schedule) for field, schedule in fields.items()}
        )

    return Struct(build)


def _piecewise(
    config: Mapping[str, object], registry: StructRegistry
) -> Piecewise[object]:
    phases = config.get("phases")
    if not isinstance(phases, list):
        raise TypeError(f"piecewise 'phases' must be a list, got {phases!r}")
    phases = cast(list[object], phases)

    loaded: dict[At, Schedule[object]] = {}
    for index, phase in enumerate(phases):
        if not isinstance(phase, Mapping):
            raise TypeError(f"piecewise phase {index} must be a mapping, got {phase!r}")
        phase = cast(Mapping[str, object], phase)
        if "at" not in phase:
            raise ValueError(f"piecewise phase {index} is missing 'at'")
        if "schedule" not in phase:
            raise ValueError(f"piecewise phase {index} is missing 'schedule'")

        at = _phase_start(phase["at"])
        if at in loaded:
            raise ValueError(f"duplicate piecewise phase at {phase['at']!r}")

        schedule = phase["schedule"]
        if not isinstance(schedule, Mapping):
            message = f"schedule for piecewise phase {index} must be a mapping"
            raise TypeError(f"{message}, got {schedule!r}")
        loaded[at] = from_dict(cast(Mapping[str, object], schedule), registry)

    return Piecewise(loaded)


def from_dict(
    config: Mapping[str, object],
    registry: StructRegistry | None = None,
) -> Schedule[object]:
    kind = config.get("kind")
    structs = registry or {}

    match kind:
        case "const":
            if "value" not in config:
                raise ValueError("missing 'value' for 'const' schedule")
            return Const(config["value"])
        case "linear":
            return Linear(_number(config, "start"), _number(config, "stop"))
        case "cosine":
            return Cosine(_number(config, "start"), _number(config, "stop"))
        case "piecewise":
            return _piecewise(config, structs)
        case "struct":
            return _struct(config, structs)
        case _:
            raise ValueError(f"unknown schedule kind {kind!r}")
