import math
from dataclasses import dataclass
from math import isfinite
from typing import cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from d2nn.optics.parameters import PhysicalParameters
from d2nn.optics.phase import Phase, PhaseConditions
from d2nn.optics.propagation import Propagation, PropagationConditions


@dataclass(frozen=True, slots=True)
class DiffractiveLayerConditions:
    fabrication_error_std: float = 0.0
    xy_drift_std: float = 0.0
    propagation: PropagationConditions | None = None
    phase: PhaseConditions | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("fabrication_error_std", self.fabrication_error_std),
            ("xy_drift_std", self.xy_drift_std),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


class DiffractiveLayer(nn.Module):
    def __init__(
        self,
        size: int,
        pixel_size: float,
        distance: float,
        physical_parameters: PhysicalParameters,
        base_thickness: float = 0,
        scale_factor: float = 6.0,
        quantization_levels: int | None = None,
    ) -> None:
        super().__init__()

        self.size: int = size
        self.pixel_size: float = pixel_size
        self.base_thickness: float = base_thickness
        self.physical_parameters: PhysicalParameters = physical_parameters
        self.conditions: DiffractiveLayerConditions | None = None
        self.phase: Phase = Phase(size, scale_factor, quantization_levels)
        self.propagation: Propagation = Propagation(
            size, pixel_size, physical_parameters.wavelength, distance
        )

    def set_conditions(self, conditions: DiffractiveLayerConditions | None) -> None:
        self.conditions = conditions

        if conditions is None:
            self.phase.set_conditions(None)
            self.propagation.set_conditions(None)
        else:
            self.phase.set_conditions(conditions.phase)
            self.propagation.set_conditions(conditions.propagation)

    @override
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = cast(torch.Tensor, self.propagation(field))
        phase = cast(torch.Tensor, self.phase())
        thickness = self._realized_thickness(phase)
        return field * self._transmission(thickness)

    def _realized_thickness(self, phase: torch.Tensor) -> torch.Tensor:
        thickness = self._thickness(phase)
        conditions = self.conditions

        if conditions is None:
            return thickness

        if conditions.fabrication_error_std > 0:
            error = torch.randn_like(thickness) * conditions.fabrication_error_std
            thickness = thickness + error

        if conditions.xy_drift_std > 0:
            drift_xy = (
                torch.randn(2, device=thickness.device, dtype=thickness.dtype)
                * conditions.xy_drift_std
            )
            thickness = self._shift_thickness(thickness, drift_xy)

        return thickness

    def _shift_thickness(
        self, thickness: torch.Tensor, drift_xy: torch.Tensor
    ) -> torch.Tensor:
        height, width = thickness.shape
        scale = thickness.new_tensor(
            (width * self.pixel_size, height * self.pixel_size)
        )

        transform = torch.eye(2, 3, device=thickness.device, dtype=thickness.dtype)
        transform[:, 2] = -2 * drift_xy / scale
        grid = F.affine_grid(
            transform.unsqueeze(0),
            size=[1, 1, height, width],
            align_corners=False,
        )

        relief = (thickness - self.base_thickness)[None, None]
        shifted = F.grid_sample(
            relief,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return shifted[0, 0] + self.base_thickness

    def _transmission(self, thickness: torch.Tensor) -> torch.Tensor:
        parameters = self.physical_parameters
        phase = thickness / parameters.thickness_per_radian
        attenuation = torch.exp(
            -math.tau
            * parameters.extinction_coefficient
            * thickness
            / parameters.wavelength
        )
        return attenuation * torch.exp(1j * phase)

    def _thickness(self, phase: torch.Tensor) -> torch.Tensor:
        modulation_thickness = phase * self.physical_parameters.thickness_per_radian
        return self.base_thickness + modulation_thickness
