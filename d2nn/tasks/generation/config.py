from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast, final, override

import yaml

from d2nn.models.generative import DecoderConfig, EncoderConfig
from d2nn.optics.diffractive import DiffractiveLayerConditions
from d2nn.optics.phase import PhaseConditions
from d2nn.optics.propagation import PropagationConditions
from d2nn.optics.stack import DiffractiveStackConditions
from d2nn.training.schedule import Schedule, Struct, from_dict

from .types import TrainingParameters

__all__ = [
    "ConfigError",
    "GenerationTaskConfig",
    "TeacherCacheConfig",
    "TrainingConfig",
    "load_config",
    "parse_config",
]


class ConfigError(ValueError): ...


@dataclass(frozen=True, slots=True)
class TeacherCacheConfig:
    size: int = 10_000
    sampling_steps: int = 500

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("cache size must be positive")
        if self.sampling_steps <= 0:
            raise ValueError("cache sampling_steps must be positive")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 200
    histogram_bins: int = 64
    histogram_weight: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2")
        if not isfinite(self.histogram_weight) or self.histogram_weight < 0:
            raise ValueError("histogram_weight must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GenerationTaskConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig
    cache: TeacherCacheConfig
    training: TrainingConfig
    schedule: Schedule[TrainingParameters]


@final
class _ParsedSchedule[T](Schedule[T]):
    def __init__(
        self,
        schedule: Schedule[object],
        parse: Callable[[object], T],
    ) -> None:
        self._schedule: Schedule[object] = schedule
        self._parse: Callable[[object], T] = parse

    @override
    def _at(self, progress: float, /) -> T:
        return self._parse(self._schedule.at(progress))


_ENCODER_KEYS = {
    "in_size",
    "out_size",
    "num_layers",
    "num_classes",
    "class_embedding_size",
}
_DECODER_KEYS = {
    "size",
    "num_layers",
    "wavelength",
    "refractive_index",
    "extinction_coefficient",
    "environment_refractive_index",
    "pixel_size",
    "distance",
    "base_thickness",
    "scale_factor",
    "quantization_levels",
}
_CACHE_KEYS = {"size", "sampling_steps"}
_TRAINING_KEYS = {"epochs", "batch_size", "histogram_bins", "histogram_weight"}
_SCHEDULE_KEYS = {
    "encoder_learning_rate",
    "decoder_learning_rate",
    "conditions",
}
_CONDITION_STRUCTS: dict[str, Callable[..., object]] = {
    "DiffractiveStackConditions": DiffractiveStackConditions,
    "DiffractiveLayerConditions": DiffractiveLayerConditions,
    "PropagationConditions": PropagationConditions,
    "PhaseConditions": PhaseConditions,
}


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")

    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ConfigError(f"{path} keys must be strings")
    return cast(Mapping[str, object], mapping)


def _check_keys(config: Mapping[str, object], allowed: set[str], path: str) -> None:
    unknown = config.keys() - allowed
    if unknown:
        names = ", ".join(sorted(repr(name) for name in unknown))
        raise ConfigError(f"unknown {path} field(s): {names}")


def _integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if value < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return value


def _number(value: object, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{path} must be a number")
    number = float(value)
    if not isfinite(number):
        raise ConfigError(f"{path} must be finite")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    return number


def _encoder_config(config: Mapping[str, object]) -> EncoderConfig:
    _check_keys(config, _ENCODER_KEYS, "model.encoder")
    default = EncoderConfig()
    return EncoderConfig(
        in_size=_integer(
            config.get("in_size", default.in_size),
            "model.encoder.in_size",
            minimum=1,
        ),
        out_size=_integer(
            config.get("out_size", default.out_size),
            "model.encoder.out_size",
            minimum=2,
        ),
        num_layers=_integer(
            config.get("num_layers", default.num_layers),
            "model.encoder.num_layers",
            minimum=1,
        ),
        num_classes=_integer(
            config.get("num_classes", default.num_classes),
            "model.encoder.num_classes",
            minimum=2,
        ),
        class_embedding_size=_integer(
            config.get("class_embedding_size", default.class_embedding_size),
            "model.encoder.class_embedding_size",
            minimum=1,
        ),
    )


def _decoder_config(config: Mapping[str, object]) -> DecoderConfig:
    _check_keys(config, _DECODER_KEYS, "model.decoder")
    default = DecoderConfig()
    quantization_value = config.get("quantization_levels", default.quantization_levels)
    quantization_levels = (
        None
        if quantization_value is None
        else _integer(
            quantization_value,
            "model.decoder.quantization_levels",
            minimum=2,
        )
    )

    decoder = DecoderConfig(
        size=_integer(
            config.get("size", default.size), "model.decoder.size", minimum=2
        ),
        num_layers=_integer(
            config.get("num_layers", default.num_layers),
            "model.decoder.num_layers",
            minimum=1,
        ),
        wavelength=_number(
            config.get("wavelength", default.wavelength),
            "model.decoder.wavelength",
            minimum=0.0,
        ),
        refractive_index=_number(
            config.get("refractive_index", default.refractive_index),
            "model.decoder.refractive_index",
            minimum=0.0,
        ),
        extinction_coefficient=_number(
            config.get("extinction_coefficient", default.extinction_coefficient),
            "model.decoder.extinction_coefficient",
            minimum=0.0,
        ),
        environment_refractive_index=_number(
            config.get(
                "environment_refractive_index",
                default.environment_refractive_index,
            ),
            "model.decoder.environment_refractive_index",
            minimum=0.0,
        ),
        pixel_size=_number(
            config.get("pixel_size", default.pixel_size),
            "model.decoder.pixel_size",
            minimum=0.0,
        ),
        distance=_number(
            config.get("distance", default.distance),
            "model.decoder.distance",
            minimum=0.0,
        ),
        base_thickness=_number(
            config.get("base_thickness", default.base_thickness),
            "model.decoder.base_thickness",
            minimum=0.0,
        ),
        scale_factor=_number(
            config.get("scale_factor", default.scale_factor),
            "model.decoder.scale_factor",
            minimum=0.0,
        ),
        quantization_levels=quantization_levels,
    )
    if decoder.wavelength == 0:
        raise ConfigError("model.decoder.wavelength must be positive")
    if decoder.pixel_size == 0:
        raise ConfigError("model.decoder.pixel_size must be positive")
    if decoder.scale_factor == 0:
        raise ConfigError("model.decoder.scale_factor must be positive")
    if decoder.refractive_index == decoder.environment_refractive_index:
        message = "model.decoder.refractive_index must differ from " + (
            "model.decoder.environment_refractive_index"
        )
        raise ConfigError(message)
    return decoder


def _cache_config(config: Mapping[str, object]) -> TeacherCacheConfig:
    _check_keys(config, _CACHE_KEYS, "cache")
    default = TeacherCacheConfig()
    return TeacherCacheConfig(
        size=_integer(config.get("size", default.size), "cache.size", minimum=1),
        sampling_steps=_integer(
            config.get("sampling_steps", default.sampling_steps),
            "cache.sampling_steps",
            minimum=1,
        ),
    )


def _training_config(config: Mapping[str, object]) -> TrainingConfig:
    _check_keys(config, _TRAINING_KEYS, "training")
    default = TrainingConfig()
    return TrainingConfig(
        epochs=_integer(
            config.get("epochs", default.epochs), "training.epochs", minimum=1
        ),
        batch_size=_integer(
            config.get("batch_size", default.batch_size),
            "training.batch_size",
            minimum=1,
        ),
        histogram_bins=_integer(
            config.get("histogram_bins", default.histogram_bins),
            "training.histogram_bins",
            minimum=2,
        ),
        histogram_weight=_number(
            config.get("histogram_weight", default.histogram_weight),
            "training.histogram_weight",
            minimum=0.0,
        ),
    )


def _learning_rate(value: object) -> float:
    learning_rate = _number(value, "scheduled learning rate")
    if learning_rate <= 0:
        raise ConfigError("scheduled learning rates must be positive")
    return learning_rate


def _conditions(value: object) -> DiffractiveStackConditions | None:
    if value is None or isinstance(value, DiffractiveStackConditions):
        return value
    raise ConfigError(
        "scheduled conditions must resolve to DiffractiveStackConditions or null"
    )


def _schedule_config(config: Mapping[str, object]) -> Schedule[TrainingParameters]:
    _check_keys(config, _SCHEDULE_KEYS, "schedule")
    for field in ("encoder_learning_rate", "decoder_learning_rate"):
        if field not in config:
            raise ConfigError(f"schedule is missing {field!r}")

    encoder_config = _mapping(
        config["encoder_learning_rate"], "schedule.encoder_learning_rate"
    )
    decoder_config = _mapping(
        config["decoder_learning_rate"], "schedule.decoder_learning_rate"
    )
    conditions_config = _mapping(
        config.get("conditions", {"kind": "const", "value": None}),
        "schedule.conditions",
    )
    try:
        encoder_learning_rate = _ParsedSchedule(
            from_dict(encoder_config), _learning_rate
        )
        decoder_learning_rate = _ParsedSchedule(
            from_dict(decoder_config), _learning_rate
        )
        conditions = _ParsedSchedule(
            from_dict(conditions_config, _CONDITION_STRUCTS), _conditions
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid schedule: {error}") from error

    return Struct(
        lambda resolve: TrainingParameters(
            encoder_learning_rate=resolve(encoder_learning_rate),
            decoder_learning_rate=resolve(decoder_learning_rate),
            conditions=resolve(conditions),
        )
    )


def parse_config(source: str) -> GenerationTaskConfig:
    try:
        document = cast(object, yaml.safe_load(source))
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML: {error}") from error

    root = _mapping(document, "configuration")
    _check_keys(root, {"model", "cache", "training", "schedule"}, "configuration")
    for section in ("model", "cache", "training", "schedule"):
        if section not in root:
            raise ConfigError(f"configuration is missing {section!r}")

    model = _mapping(root["model"], "model")
    _check_keys(model, {"encoder", "decoder"}, "model")
    encoder = _encoder_config(_mapping(model.get("encoder", {}), "model.encoder"))
    decoder = _decoder_config(_mapping(model.get("decoder", {}), "model.decoder"))
    if encoder.out_size != decoder.size:
        raise ConfigError("model.encoder.out_size must equal model.decoder.size")

    config = GenerationTaskConfig(
        encoder=encoder,
        decoder=decoder,
        cache=_cache_config(_mapping(root["cache"], "cache")),
        training=_training_config(_mapping(root["training"], "training")),
        schedule=_schedule_config(_mapping(root["schedule"], "schedule")),
    )
    try:
        _ = config.schedule.at(0.0)
        _ = config.schedule.at(1.0)
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid scheduled value: {error}") from error
    return config


def load_config(path: str | Path) -> GenerationTaskConfig:
    config_path = Path(path)
    if config_path.suffix.lower() != ".yaml":
        raise ConfigError(f"configuration file must use the .yaml suffix: {path}")
    return parse_config(config_path.read_text(encoding="utf-8"))
