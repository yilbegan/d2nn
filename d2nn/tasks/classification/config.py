from pathlib import Path

from d2nn.config import (
    Config,
    ConfigError,
    NonNegativeInt,
    PositiveInt,
    parse_document,
    read_document,
)
from d2nn.models.classifier import ClassifierConfig
from d2nn.optics.input import InputConfig
from d2nn.training.conditions import ConditionsSpec
from d2nn.training.schedule import ValueSpec, resolve

from .types import ClassificationLoss, TrainingParameters

__all__ = [
    "ClassificationTaskConfig",
    "ConfigError",
    "ScheduleConfig",
    "TrainingConfig",
    "load_config",
    "parse_config",
]


class TrainingConfig(Config, frozen=True):
    epochs: PositiveInt = 5
    batch_size: PositiveInt = 500
    num_workers: NonNegativeInt = 4
    loss: ClassificationLoss = ClassificationLoss.CROSS_ENTROPY


class ScheduleConfig(Config, frozen=True):
    learning_rate: ValueSpec
    conditions: ConditionsSpec = None

    def __post_init__(self) -> None:
        for progress in (0.0, 0.5, 1.0):
            _ = self.at(progress)

    def at(self, progress: float, /) -> TrainingParameters:
        return TrainingParameters(
            learning_rate=resolve(self.learning_rate, progress),
            conditions=resolve(self.conditions, progress),
        )


class ClassificationTaskConfig(Config, frozen=True):
    model: ClassifierConfig
    training: TrainingConfig
    schedule: ScheduleConfig
    input: InputConfig = InputConfig()


def parse_config(source: str) -> ClassificationTaskConfig:
    return parse_document(source, ClassificationTaskConfig)


def load_config(path: str | Path) -> ClassificationTaskConfig:
    return read_document(path, ClassificationTaskConfig)
