from pathlib import Path
from typing import Annotated

from msgspec import Meta

from d2nn.config import (
    Config,
    ConfigError,
    NonNegativeFloat,
    PositiveInt,
    parse_document,
    read_document,
)
from d2nn.models.generative import DecoderConfig, EncoderConfig
from d2nn.training.conditions import ConditionsSpec
from d2nn.training.schedule import ValueSpec, resolve

from .types import TrainingParameters

__all__ = [
    "ConfigError",
    "GenerationTaskConfig",
    "ModelConfig",
    "ScheduleConfig",
    "TeacherCacheConfig",
    "TrainingConfig",
    "load_config",
    "parse_config",
]


class TeacherCacheConfig(Config, frozen=True):
    size: PositiveInt = 10_000
    sampling_steps: PositiveInt = 500


class TrainingConfig(Config, frozen=True):
    epochs: PositiveInt = 10
    batch_size: PositiveInt = 200
    histogram_bins: Annotated[int, Meta(ge=2)] = 64
    histogram_weight: NonNegativeFloat = 1.0e-4


class ModelConfig(Config, frozen=True):
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()

    def __post_init__(self) -> None:
        if self.encoder.out_size != self.decoder.size:
            raise ValueError("encoder.out_size must equal decoder.size")


class ScheduleConfig(Config, frozen=True):
    encoder_learning_rate: ValueSpec
    decoder_learning_rate: ValueSpec
    conditions: ConditionsSpec = None

    def __post_init__(self) -> None:
        for progress in (0.0, 0.5, 1.0):
            _ = self.at(progress)

    def at(self, progress: float, /) -> TrainingParameters:
        return TrainingParameters(
            encoder_learning_rate=resolve(self.encoder_learning_rate, progress),
            decoder_learning_rate=resolve(self.decoder_learning_rate, progress),
            conditions=resolve(self.conditions, progress),
        )


class GenerationTaskConfig(Config, frozen=True):
    model: ModelConfig
    cache: TeacherCacheConfig
    training: TrainingConfig
    schedule: ScheduleConfig

    @property
    def encoder(self) -> EncoderConfig:
        return self.model.encoder

    @property
    def decoder(self) -> DecoderConfig:
        return self.model.decoder


def parse_config(source: str) -> GenerationTaskConfig:
    return parse_document(source, GenerationTaskConfig)


def load_config(path: str | Path) -> GenerationTaskConfig:
    return read_document(path, GenerationTaskConfig)
