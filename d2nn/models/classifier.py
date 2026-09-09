import pathlib
from typing import Annotated, Self, TypedDict, cast, override

import msgspec
import torch
import torch.nn as nn
from msgspec import Meta

from d2nn.config import Config, NonNegativeFloat, PositiveFloat, PositiveInt
from d2nn.optics.detector import Detector
from d2nn.optics.parameters import PhysicalParameters
from d2nn.optics.stack import DiffractiveStack, DiffractiveStackConditions

__all__ = ["ClassifierConfig", "DiffractiveClassifier"]


class ClassifierConfig(Config, frozen=True):
    size: Annotated[int, Meta(ge=2)] = 200
    num_layers: PositiveInt = 5
    num_classes: Annotated[int, Meta(ge=2)] = 10
    wavelength: PositiveFloat = 0.75e-3
    refractive_index: float = 1.7227
    extinction_coefficient: NonNegativeFloat = 0.0
    environment_refractive_index: float = 1.0
    pixel_size: PositiveFloat = 0.4e-3
    distance: NonNegativeFloat = 0.03
    base_thickness: NonNegativeFloat = 0.0
    det_size: PositiveInt = 20
    scale_factor: PositiveFloat = 6.0
    quantization_levels: Annotated[int, Meta(ge=2)] | None = None

    def __post_init__(self) -> None:
        if self.det_size > self.size:
            raise ValueError("det_size cannot exceed size")
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


class _Checkpoint(TypedDict):
    state_dict: dict[str, torch.Tensor]
    config: dict[str, object]


class DiffractiveClassifier(nn.Module):
    def __init__(
        self,
        config: ClassifierConfig | None = None,
    ) -> None:
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

    def propagate(self, field: torch.Tensor) -> torch.Tensor:
        """Propagate an encoded input field (B, H, W) to the detector plane."""

        return cast(torch.Tensor, self.stack(field))

    @override
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Detector scores (B, num_classes) for an encoded input field (B, H, W)."""

        return cast(torch.Tensor, self.detector(self.propagate(field)))

    def save(self, path: str | pathlib.Path) -> None:
        checkpoint: _Checkpoint = {
            "state_dict": self.state_dict(),
            "config": msgspec.to_builtins(self.config),
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
            _Checkpoint,
            torch.load(path, map_location=device, weights_only=True),
        )
        document = checkpoint["config"]
        if quantization_levels is not None:
            document = {**document, "quantization_levels": quantization_levels}

        model = cls(msgspec.convert(document, ClassifierConfig))
        _ = model.load_state_dict(checkpoint["state_dict"])
        return model
