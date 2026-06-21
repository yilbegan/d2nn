from enum import StrEnum
from typing import NamedTuple


class Metrics(NamedTuple):
    accuracy: float
    correct_efficiency: float
    total_efficiency: float


class ClassificationLoss(StrEnum):
    MSE = "mse"
    CROSS_ENTROPY = "cross_entropy"
