from collections.abc import Callable
from typing import Annotated, Literal, cast, override

import torch
import torch.nn as nn
from msgspec import Meta

from d2nn.config import Config, PositiveFloat
from d2nn.optics.beam import GaussianBeam

__all__ = [
    "BeamConfig",
    "FieldEncoder",
    "GaussianBeamConfig",
    "GaussianBeamParameters",
    "InputConfig",
    "InputEncoder",
    "MaskConfig",
    "MaskEncoding",
    "PlaneBeamConfig",
]

type FieldEncoder = Callable[[torch.Tensor], torch.Tensor]
type MaskEncoding = Literal["amplitude", "phase"]


class MaskConfig(Config, frozen=True):
    size: Annotated[float, Meta(gt=0.0, le=1.0)] = 0.4
    type: MaskEncoding = "amplitude"


class PlaneBeamConfig(Config, frozen=True, tag_field="type", tag="plane"): ...


class GaussianBeamParameters(Config, frozen=True):
    distance: PositiveFloat
    waist_radius: PositiveFloat
    beam_quality: Annotated[float, Meta(ge=1.0)] = 1.0


class GaussianBeamConfig(Config, frozen=True, tag_field="type", tag="gaussian"):
    parameters: GaussianBeamParameters


type BeamConfig = PlaneBeamConfig | GaussianBeamConfig


class InputConfig(Config, frozen=True):
    mask: MaskConfig = MaskConfig()
    beam: BeamConfig = PlaneBeamConfig()


class InputEncoder(nn.Module):
    def __init__(
        self,
        config: InputConfig | None = None,
        *,
        size: int,
        pixel_size: float,
        wavelength: float,
    ) -> None:
        super().__init__()

        self.config: InputConfig = config or InputConfig()
        self.size: int = size
        self.beam: GaussianBeam | None = None

        if isinstance(self.config.beam, GaussianBeamConfig):
            beam = self.config.beam.parameters
            self.beam = GaussianBeam(
                size=size,
                pixel_size=pixel_size,
                wavelength=wavelength,
                distance=beam.distance,
                waist_radius=beam.waist_radius,
                beam_quality=beam.beam_quality,
            )

    @override
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 3 or images.shape[-2:] != (self.size, self.size):
            raise ValueError(
                f"images must have shape (batch, {self.size}, {self.size})"
            )
        match self.config.mask.type:
            case "amplitude":
                field = images.to(torch.complex64)
            case "phase":
                field = torch.exp(1j * torch.pi * images).to(torch.complex64)
        return field if self.beam is None else cast(torch.Tensor, self.beam(field))
