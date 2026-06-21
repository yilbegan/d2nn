import torch
import torch.nn as nn

from .propagation import Propagation


class DiffractiveLayer(nn.Module):
    def __init__(
        self,
        size: int,
        pixel_size: float,
        wavelength: float,
        distance: float,
        scale_factor: float = 6.0,
    ) -> None:
        super().__init__()
        self.propagation = Propagation(size, pixel_size, wavelength, distance)
        self.scale_factor = scale_factor
        self.wavelength = wavelength
        self.phase = nn.Parameter(torch.zeros(size, size))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = self.propagation(field)
        return field * torch.exp(1j * self.scale_factor * self.phase)
