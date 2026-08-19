from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast, override

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class PhaseConditions:
    quantization_steepness: float
    quantization_mode: Literal["soft", "hard"] = "soft"

    def __post_init__(self) -> None:
        steepness = cast(object, self.quantization_steepness)
        if isinstance(steepness, bool) or not isinstance(steepness, int | float):
            raise TypeError("quantization_steepness must be a number")
        if not isfinite(steepness) or steepness <= 0:
            raise ValueError("quantization_steepness must be finite and positive")
        if self.quantization_mode not in ("soft", "hard"):
            raise ValueError("quantization_mode must be 'soft' or 'hard'")


def _quantize_hard(phase: torch.Tensor, *, levels: int) -> torch.Tensor:
    step = 2 * torch.pi / levels
    normalized = phase / step
    hard_indices = torch.remainder(torch.floor(normalized + 0.5), levels)
    return hard_indices * step


def _quantize_soft(
    phase: torch.Tensor, *, levels: int, steepness: float
) -> torch.Tensor:
    step = 2 * torch.pi / levels
    normalized = phase / step
    thresholds = torch.arange(levels, device=phase.device, dtype=phase.dtype) + 0.5
    soft_indices = torch.sigmoid(
        steepness * (normalized.unsqueeze(-1) - thresholds)
    ).sum(dim=-1)
    return soft_indices * step


class Phase(nn.Module):
    def __init__(
        self,
        size: int,
        scale_factor: float = 6.0,
        quantization_levels: int | None = None,
    ) -> None:
        super().__init__()

        self.quantization_levels: int | None = quantization_levels
        self.conditions: PhaseConditions | None = None
        self.scale_factor: float = scale_factor
        self.phase_raw: nn.Parameter = nn.Parameter(torch.zeros(size, size))

    def set_conditions(self, conditions: PhaseConditions | None = None) -> None:
        self.conditions = conditions

    @override
    def forward(self) -> torch.Tensor:
        if self.training:
            return self._training()
        return self._inference()

    def _inference(self) -> torch.Tensor:
        phase = self.phase
        if self.quantization_levels is not None:
            phase = _quantize_hard(phase, levels=self.quantization_levels)
        return phase

    def _training(self) -> torch.Tensor:
        phase = self.phase
        if self.conditions is None:
            return phase

        steepness = self.conditions.quantization_steepness
        levels = self.quantization_levels

        if levels is None:
            return phase

        soft = _quantize_soft(phase, levels=levels, steepness=steepness)
        match self.conditions.quantization_mode:
            case "soft":
                return soft
            case "hard":
                hard = _quantize_hard(phase, levels=levels)
                return soft + (hard - soft).detach()

    @property
    def phase(self) -> torch.Tensor:
        scaled = self.scale_factor * self.phase_raw
        return torch.remainder(scaled, 2 * torch.pi)
