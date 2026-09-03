import pathlib
from dataclasses import asdict, dataclass, replace
from typing import NotRequired, Self, TypedDict, cast, override

import torch
import torch.nn as nn

from d2nn.optics.detector import Detector
from d2nn.optics.parameters import PhysicalParameters
from d2nn.optics.stack import DiffractiveStack, DiffractiveStackConditions


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    size: int = 200
    num_layers: int = 5
    num_classes: int = 10
    wavelength: float = 0.75e-3
    refractive_index: float = 1.7227
    extinction_coefficient: float = 0.0
    environment_refractive_index: float = 1.0
    pixel_size: float = 0.4e-3
    distance: float = 0.03
    base_thickness: float = 0.0
    det_size: int = 20
    scale_factor: float = 6.0
    quantization_levels: int | None = None

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 1 <= self.det_size <= self.size:
            raise ValueError("det_size must be at least 1 and cannot exceed size")
        if self.quantization_levels is not None and self.quantization_levels < 2:
            raise ValueError("quantization_levels must be at least 2")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        if self.distance < 0:
            raise ValueError("distance must be non-negative")
        if self.base_thickness < 0:
            raise ValueError("base_thickness must be non-negative")
        if self.extinction_coefficient < 0:
            raise ValueError("extinction_coefficient must be non-negative")
        if self.refractive_index == self.environment_refractive_index:
            raise ValueError(
                "refractive_index must differ from environment_refractive_index"
            )

    @property
    def physical_parameters(self) -> PhysicalParameters:
        return PhysicalParameters(
            wavelength=self.wavelength,
            refractive_index=self.refractive_index,
            extinction_coefficient=self.extinction_coefficient,
            environment_refractive_index=self.environment_refractive_index,
        )


class _ClassifierConfigData(TypedDict):
    size: int
    num_layers: int
    num_classes: int
    wavelength: float
    refractive_index: NotRequired[float]
    extinction_coefficient: NotRequired[float]
    environment_refractive_index: NotRequired[float]
    pixel_size: float
    distance: float
    base_thickness: NotRequired[float]
    det_size: int
    scale_factor: float
    quantization_levels: int | None


class _ClassifierCheckpoint(TypedDict):
    state_dict: dict[str, torch.Tensor]
    config: _ClassifierConfigData


class DiffractiveClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig | None = None) -> None:
        super().__init__()
        self.config: ClassifierConfig = config or ClassifierConfig()
        c = self.config

        self.stack: DiffractiveStack = DiffractiveStack(
            size=c.size,
            pixel_size=c.pixel_size,
            distance=c.distance,
            physical_parameters=c.physical_parameters,
            num_layers=c.num_layers,
            base_thickness=c.base_thickness,
            scale_factor=c.scale_factor,
            quantization_levels=c.quantization_levels,
        )
        self.conditions: DiffractiveStackConditions | None = None
        self.detector: Detector = Detector(
            size=c.size,
            num_classes=c.num_classes,
            det_size=c.det_size,
        )

    def set_conditions(self, conditions: DiffractiveStackConditions | None) -> None:
        self.conditions = conditions
        self.stack.set_conditions(conditions)

    @override
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        field = image.to(torch.complex64)
        field = cast(torch.Tensor, self.stack(field))
        return cast(torch.Tensor, self.detector(field))

    def save(self, path: str | pathlib.Path) -> None:
        checkpoint: _ClassifierCheckpoint = {
            "state_dict": self.state_dict(),
            "config": cast(_ClassifierConfigData, cast(object, asdict(self.config))),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        device: torch.device | str = "cpu",
        *,
        quantization_levels: int | None = None,
    ) -> Self:
        checkpoint = cast(
            _ClassifierCheckpoint,
            torch.load(path, map_location=device, weights_only=True),
        )
        config = ClassifierConfig(**checkpoint["config"])
        if quantization_levels is not None:
            config = replace(config, quantization_levels=quantization_levels)

        model = cls(config)
        _ = model.load_state_dict(checkpoint["state_dict"])
        return model
