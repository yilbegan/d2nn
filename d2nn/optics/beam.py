import math
from typing import override

import torch
import torch.nn as nn


def gaussian_field(
    size: int,
    pixel_size: float,
    wavelength: float,
    distance: float,
    waist_radius: float,
    beam_quality: float = 1.0,
) -> torch.Tensor:
    rayleigh_distance = math.pi * waist_radius**2 / (beam_quality * wavelength)
    beam_radius = waist_radius * math.sqrt(1 + (distance / rayleigh_distance) ** 2)
    inverse_curvature = distance / (distance**2 + rayleigh_distance**2)
    wave_number = math.tau / wavelength

    side = (torch.arange(size, dtype=torch.float64) - (size - 1) / 2) * pixel_size
    radius_squared = side.square()[:, None] + side.square()[None, :]

    amplitude = torch.exp(-radius_squared / beam_radius**2)
    phase = wave_number * radius_squared * inverse_curvature / 2
    return torch.polar(amplitude, -phase)


class GaussianBeam(nn.Module):
    def __init__(
        self,
        size: int,
        pixel_size: float,
        wavelength: float,
        distance: float,
        waist_radius: float,
        beam_quality: float = 1.0,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()

        self.size: int = size
        self.pixel_size: float = pixel_size
        self.wavelength: float = wavelength
        self.distance: float = distance
        self.waist_radius: float = waist_radius
        self.beam_quality: float = beam_quality

        field = gaussian_field(
            size, pixel_size, wavelength, distance, waist_radius, beam_quality
        ).to(dtype)

        self.register_buffer("field", field, persistent=False)
        self.field: torch.Tensor = field

    @property
    def rayleigh_distance(self) -> float:
        return math.pi * self.waist_radius**2 / (self.beam_quality * self.wavelength)

    @override
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.field * mask
