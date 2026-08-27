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
    ):
        super().__init__()

        self.size: int = size
        self.pixel_size: float = size
        self.wavelength: float = wavelength
        self.distance: float = distance
        self.waist_radius: float = waist_radius

    @override
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.field() * mask

    def field(self):
        side = (
            torch.arange(self.size, dtype=torch.float64)
            - (self.size - 1) / 2
        ) * self.pixel_size

        y, x = torch.meshgrid(side, side, indexing="ij")

        radius = x**2 + y**2

        wave_number = 2 * torch.pi / self.wavelength
        rayleigh_distance = torch.pi * self.waist_radius**2 / self.wavelength

        q = torch.complex(
            torch.as_tensor(self.distance, dtype=torch.float64),
            torch.as_tensor(rayleigh_distance, dtype=torch.float64),
        )

        return torch.exp(-1j * wave_number * radius / q / 2)
