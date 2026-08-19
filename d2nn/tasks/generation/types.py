from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Self, cast

import torch

from d2nn.optics.stack import DiffractiveStackConditions

__all__ = [
    "EpochCallback",
    "EpochResult",
    "TeacherCache",
    "TrainingParameters",
]


@dataclass(frozen=True, slots=True)
class TrainingParameters:
    encoder_learning_rate: float
    decoder_learning_rate: float
    conditions: DiffractiveStackConditions | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("encoder_learning_rate", self.encoder_learning_rate),
            ("decoder_learning_rate", self.decoder_learning_rate),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class EpochResult:
    epoch: int
    loss: float
    parameters: TrainingParameters


type EpochCallback = Callable[[EpochResult], None]


@dataclass(frozen=True, slots=True)
class TeacherCache:
    """CPU tensors produced by the diffusion teacher.

    Shapes are ``(N, 1, H, W)`` for noises, ``(N,)`` for labels, and
    ``(N, H, W)`` for generated images.
    """

    noises: torch.Tensor
    labels: torch.Tensor
    images: torch.Tensor

    def __post_init__(self) -> None:
        if self.noises.ndim != 4 or self.noises.shape[1] != 1:
            raise ValueError("cache noises must have shape (N, 1, H, W)")
        if self.noises.shape[2] != self.noises.shape[3]:
            raise ValueError("cache noises must be square")
        if self.labels.ndim != 1:
            raise ValueError("cache labels must have shape (N,)")
        if self.images.ndim != 3:
            raise ValueError("cache images must have shape (N, H, W)")
        if self.images.shape[1] != self.images.shape[2]:
            raise ValueError("cache images must be square")

        sizes = {self.noises.shape[0], self.labels.shape[0], self.images.shape[0]}
        if len(sizes) != 1:
            raise ValueError("cache tensors must contain the same number of samples")
        if self.size == 0:
            raise ValueError("teacher cache must contain at least one sample")
        if self.labels.dtype != torch.long:
            raise ValueError("cache labels must use torch.long dtype")
        if not torch.is_floating_point(self.noises):
            raise ValueError("cache noises must use a floating-point dtype")
        if not torch.is_floating_point(self.images):
            raise ValueError("cache images must use a floating-point dtype")
        if self.noises.device.type != "cpu" or self.images.device.type != "cpu":
            raise ValueError("cache noises and images must be stored on the CPU")
        if self.labels.device.type != "cpu":
            raise ValueError("cache labels must be stored on the CPU")

    @property
    def size(self) -> int:
        return self.labels.shape[0]

    @property
    def noise_size(self) -> int:
        return self.noises.shape[-1]

    def validate_for(
        self,
        *,
        noise_size: int,
        num_classes: int,
        size: int | None = None,
    ) -> None:
        if size is not None and self.size != size:
            raise ValueError(f"cache contains {self.size} samples; expected {size}")
        if self.noise_size != noise_size:
            raise ValueError(
                f"cache noise size is {self.noise_size}; expected {noise_size}"
            )
        if bool((self.labels < 0).any()) or bool((self.labels >= num_classes).any()):
            raise ValueError(f"cache labels must be in [0, {num_classes})")

    def save(self, path: str | Path) -> None:
        cache_path = Path(path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"noises": self.noises, "labels": self.labels, "images": self.images},
            cache_path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        raw = cast(
            object,
            torch.load(path, map_location="cpu", weights_only=True),
        )
        if not isinstance(raw, Mapping):
            raise ValueError("teacher cache must contain a tensor mapping")

        data = cast(Mapping[object, object], raw)
        tensors: dict[str, torch.Tensor] = {}
        for name in ("noises", "labels", "images"):
            value = data.get(name)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"teacher cache is missing tensor {name!r}")
            tensors[name] = value
        return cls(tensors["noises"], tensors["labels"], tensors["images"])
