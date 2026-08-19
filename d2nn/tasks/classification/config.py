from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast, final, override

import yaml

from d2nn.models.classifier import ClassifierConfig
from d2nn.optics.diffractive import DiffractiveLayerConditions
from d2nn.optics.phase import PhaseConditions
from d2nn.optics.propagation import PropagationConditions
from d2nn.optics.stack import DiffractiveStackConditions
from d2nn.training.schedule import Schedule, Struct, from_dict

from .types import ClassificationLoss, TrainingParameters

__all__ = [
    "ClassificationTaskConfig",
    "ConfigError",
    "TrainingConfig",
    "load_config",
    "parse_config",
]


class ConfigError(ValueError):
    ...


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 500
    num_workers: int = 4
    loss: ClassificationLoss = ClassificationLoss.CROSS_ENTROPY

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


@dataclass(frozen=True, slots=True)
class ClassificationTaskConfig:
    model: ClassifierConfig
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


_MODEL_KEYS = {
    "size",
    "num_layers",
    "num_classes",
    "wavelength",
    "refractive_index",
    "extinction_coefficient",
    "environment_refractive_index",
    "pixel_size",
    "distance",
    "base_thickness",
    "det_size",
    "scale_factor",
    "quantization_levels",
}
_TRAINING_KEYS = {"epochs", "batch_size", "num_workers", "loss"}
_SCHEDULE_KEYS = {"learning_rate", "conditions"}
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


def _model_config(config: Mapping[str, object]) -> ClassifierConfig:
    _check_keys(config, _MODEL_KEYS, "model")
    default = ClassifierConfig()
    quantization_value = config.get("quantization_levels", default.quantization_levels)
    quantization_levels = (
        None
        if quantization_value is None
        else _integer(quantization_value, "model.quantization_levels", minimum=2)
    )

    model = ClassifierConfig(
        size=_integer(config.get("size", default.size), "model.size", minimum=2),
        num_layers=_integer(
            config.get("num_layers", default.num_layers),
            "model.num_layers",
            minimum=1,
        ),
        num_classes=_integer(
            config.get("num_classes", default.num_classes),
            "model.num_classes",
            minimum=2,
        ),
        wavelength=_number(
            config.get("wavelength", default.wavelength),
            "model.wavelength",
            minimum=0.0,
        ),
        refractive_index=_number(
            config.get("refractive_index", default.refractive_index),
            "model.refractive_index",
            minimum=0.0,
        ),
        extinction_coefficient=_number(
            config.get("extinction_coefficient", default.extinction_coefficient),
            "model.extinction_coefficient",
            minimum=0.0,
        ),
        environment_refractive_index=_number(
            config.get(
                "environment_refractive_index",
                default.environment_refractive_index,
            ),
            "model.environment_refractive_index",
            minimum=0.0,
        ),
        pixel_size=_number(
            config.get("pixel_size", default.pixel_size),
            "model.pixel_size",
            minimum=0.0,
        ),
        distance=_number(
            config.get("distance", default.distance),
            "model.distance",
            minimum=0.0,
        ),
        base_thickness=_number(
            config.get("base_thickness", default.base_thickness),
            "model.base_thickness",
            minimum=0.0,
        ),
        det_size=_integer(
            config.get("det_size", default.det_size),
            "model.det_size",
            minimum=1,
        ),
        scale_factor=_number(
            config.get("scale_factor", default.scale_factor),
            "model.scale_factor",
            minimum=0.0,
        ),
        quantization_levels=quantization_levels,
    )
    if model.det_size > model.size:
        raise ConfigError("model.det_size cannot exceed model.size")
    if model.wavelength == 0:
        raise ConfigError("model.wavelength must be positive")
    if model.pixel_size == 0:
        raise ConfigError("model.pixel_size must be positive")
    if model.scale_factor == 0:
        raise ConfigError("model.scale_factor must be positive")
    if model.refractive_index == model.environment_refractive_index:
        raise ConfigError(
            "model.refractive_index must differ from model.environment_refractive_index"
        )
    return model


def _training_config(config: Mapping[str, object]) -> TrainingConfig:
    _check_keys(config, _TRAINING_KEYS, "training")
    default = TrainingConfig()
    loss_value = config.get("loss", default.loss.value)
    if not isinstance(loss_value, str):
        raise ConfigError("training.loss must be a string")
    try:
        loss = ClassificationLoss(loss_value)
    except ValueError:
        choices = ", ".join(loss.value for loss in ClassificationLoss)
        raise ConfigError(f"training.loss must be one of: {choices}") from None

    return TrainingConfig(
        epochs=_integer(
            config.get("epochs", default.epochs), "training.epochs", minimum=1
        ),
        batch_size=_integer(
            config.get("batch_size", default.batch_size),
            "training.batch_size",
            minimum=1,
        ),
        num_workers=_integer(
            config.get("num_workers", default.num_workers),
            "training.num_workers",
            minimum=0,
        ),
        loss=loss,
    )


def _learning_rate(value: object) -> float:
    learning_rate = _number(value, "scheduled learning_rate")
    if learning_rate <= 0:
        raise ConfigError("scheduled learning_rate must be positive")
    return learning_rate


def _conditions(value: object) -> DiffractiveStackConditions | None:
    if value is None or isinstance(value, DiffractiveStackConditions):
        return value
    raise ConfigError(
        "scheduled conditions must resolve to DiffractiveStackConditions or null"
    )


def _schedule_config(
    config: Mapping[str, object],
) -> Schedule[TrainingParameters]:
    _check_keys(config, _SCHEDULE_KEYS, "schedule")
    if "learning_rate" not in config:
        raise ConfigError("schedule is missing 'learning_rate'")

    learning_rate_config = _mapping(config["learning_rate"], "schedule.learning_rate")
    conditions_config = _mapping(
        config.get("conditions", {"kind": "const", "value": None}),
        "schedule.conditions",
    )
    try:
        learning_rate = _ParsedSchedule(from_dict(learning_rate_config), _learning_rate)
        conditions = _ParsedSchedule(
            from_dict(conditions_config, _CONDITION_STRUCTS), _conditions
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigError(f"invalid schedule: {error}") from error

    return Struct(
        lambda resolve: TrainingParameters(
            learning_rate=resolve(learning_rate),
            conditions=resolve(conditions),
        )
    )


def parse_config(source: str) -> ClassificationTaskConfig:
    try:
        document = cast(object, yaml.safe_load(source))
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML: {error}") from error

    root = _mapping(document, "configuration")
    _check_keys(root, {"model", "training", "schedule"}, "configuration")
    for section in ("model", "training", "schedule"):
        if section not in root:
            raise ConfigError(f"configuration is missing {section!r}")

    config = ClassificationTaskConfig(
        model=_model_config(_mapping(root["model"], "model")),
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


def load_config(path: str | Path) -> ClassificationTaskConfig:
    config_path = Path(path)
    if config_path.suffix.lower() != ".yaml":
        raise ConfigError(f"configuration file must use the .yaml suffix: {path}")
    return parse_config(config_path.read_text(encoding="utf-8"))
