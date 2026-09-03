from dataclasses import dataclass
from typing import cast, override

import torch
import torch.nn as nn

from d2nn.optics.diffractive import DiffractiveLayer, DiffractiveLayerConditions
from d2nn.optics.parameters import PhysicalParameters
from d2nn.optics.propagation import Propagation, PropagationConditions


@dataclass(frozen=True, slots=True)
class DiffractiveStackConditions:
    layers: DiffractiveLayerConditions | None = None
    output_propagation: PropagationConditions | None = None


class DiffractiveStack(nn.Module):
    def __init__(
        self,
        size: int,
        pixel_size: float,
        distance: float,
        physical_parameters: PhysicalParameters,
        num_layers: int,
        base_thickness: float = 0.0,
        scale_factor: float = 6.0,
        quantization_levels: int | None = None,
    ) -> None:
        super().__init__()

        self.layers: nn.ModuleList = nn.ModuleList(
            DiffractiveLayer(
                size=size,
                pixel_size=pixel_size,
                distance=distance,
                physical_parameters=physical_parameters,
                base_thickness=base_thickness,
                scale_factor=scale_factor,
                quantization_levels=quantization_levels,
            )
            for _ in range(num_layers)
        )
        self.output_propagation: Propagation = Propagation(
            size=size,
            pixel_size=pixel_size,
            wavelength=physical_parameters.wavelength,
            distance=distance,
        )
        self.conditions: DiffractiveStackConditions | None = None

    def set_conditions(self, conditions: DiffractiveStackConditions | None) -> None:
        self.conditions = conditions
        layer_conditions = None if conditions is None else conditions.layers
        output_conditions = (
            None if conditions is None else conditions.output_propagation
        )

        for module in self.layers:
            layer = cast(DiffractiveLayer, module)
            layer.set_conditions(layer_conditions)
        self.output_propagation.set_conditions(output_conditions)

    @override
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        for module in self.layers:
            layer = cast(DiffractiveLayer, module)
            field = cast(torch.Tensor, layer(field))
        return cast(torch.Tensor, self.output_propagation(field))
