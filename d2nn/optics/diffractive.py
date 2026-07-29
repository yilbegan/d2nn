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
        quantization_levels: int | None = None,
        quantization_steepness: float = 4.0,
    ) -> None:
        super().__init__()
        if quantization_levels is not None and quantization_levels < 2:
            raise ValueError("quantization_levels must be at least 2")
        if quantization_steepness <= 0:
            raise ValueError("quantization_steepness must be positive")

        self.propagation = Propagation(size, pixel_size, wavelength, distance)
        self.scale_factor = scale_factor
        self.wavelength = wavelength
        self.quantization_levels = quantization_levels
        self.quantization_steepness = quantization_steepness
        self.phase = nn.Parameter(torch.zeros(size, size))

    def effective_phase(self) -> torch.Tensor:
        phase = self.scale_factor * self.phase
        levels = self.quantization_levels
        if levels is None:
            return phase

        step = 2 * torch.pi / levels
        normalized = torch.remainder(phase, 2 * torch.pi) / step

        hard_indices = torch.remainder(torch.floor(normalized + 0.5), levels)
        hard_phase = hard_indices * step

        if not self.is_training and not torch.is_grad_enabled():
            return hard_phase

        thresholds = torch.arange(levels, device=phase.device, dtype=phase.dtype) + 0.5
        soft_indices = torch.sigmoid(
            self.quantization_steepness * (normalized.unsqueeze(-1) - thresholds)
        ).sum(dim=-1)
        soft_phase = soft_indices * step

        return soft_phase + (hard_phase - soft_phase).detach()

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = self.propagation(field)
        return field * torch.exp(1j * self.effective_phase())
