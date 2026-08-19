from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from d2nn.optics.stack import DiffractiveStackConditions


@dataclass(frozen=True, slots=True)
class Metrics:
    accuracy: float
    correct_efficiency: float
    total_efficiency: float


class ClassificationLoss(StrEnum):
    MSE = "mse"
    CROSS_ENTROPY = "cross_entropy"


@dataclass(frozen=True, slots=True)
class TrainingParameters:
    learning_rate: float
    conditions: DiffractiveStackConditions | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    loss: float
    metrics: Metrics
    parameters: TrainingParameters


type EpochCallback = Callable[[EpochResult], None]
