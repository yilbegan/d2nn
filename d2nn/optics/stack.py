import torch
import torch.nn as nn

from .diffractive import DiffractiveLayer
from .propagation import Propagation


class DiffractiveStack(nn.Module):
    def __init__(
        self,
        size: int,
        pixel_size: float,
        wavelength: float,
        distance: float,
        num_layers: int,
        scale_factor: float = 6.0,
        quantization_levels: int | None = None,
        quantization_steepness: float = 4.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DiffractiveLayer(
                size,
                pixel_size,
                wavelength,
                distance,
                scale_factor,
                quantization_levels,
                quantization_steepness,
            )
            for _ in range(num_layers)
        )
        self.output_propagation = Propagation(size, pixel_size, wavelength, distance)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            field = layer(field)
        return self.output_propagation(field)
