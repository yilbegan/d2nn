from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PhysicalParameters:
    wavelength: float
    refractive_index: float
    extinction_coefficient: float = 0
    environment_refractive_index: float = 1.0

    @property
    def refractive_index_delta(self) -> float:
        return self.refractive_index - self.environment_refractive_index

    @property
    def thickness_per_radian(self) -> float:
        return self.wavelength / (2 * torch.pi * self.refractive_index_delta)
