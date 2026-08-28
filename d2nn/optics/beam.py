from typing import override

import torch
import torch.nn as nn


class GaussianBeam(nn.Module):
    def __init__(
        self,
        size: int,
        pixel_size: float,
        wavelength: float,
        distance: float,
        waist_radius: float,
        beam_quality: float = 1.0,
    ):
        super().__init__()

        self.size: int = size
        self.pixel_size: float = pixel_size
        self.wavelength: float = wavelength
        self.distance: float = distance
        self.waist_radius: float = waist_radius
        self.beam_quality: float = beam_quality

    @property
    def rayleigh_distance(self) -> float:
        return torch.pi * self.waist_radius**2 / (self.beam_quality * self.wavelength)

    @override
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.field() * mask

    def field(self):
        side = (
            torch.arange(self.size, dtype=torch.float64) - (self.size - 1) / 2
        ) * self.pixel_size

        y, x = torch.meshgrid(side, side, indexing="ij")
        radius = x**2 + y**2
        wave_number = 2 * torch.pi / self.wavelength

        beam_radius = self.waist_radius * torch.sqrt(
            torch.as_tensor(
                1 + (self.distance / self.rayleigh_distance) ** 2,
                dtype=torch.float64,
            )
        )

        curvature_radius = self.distance * (
            1 + (self.distance / self.rayleigh_distance) ** 2
        )

        amplitude = -radius / beam_radius**2
        phase = wave_number * radius / (2 * curvature_radius)

        return torch.exp(amplitude - 1j * phase)
