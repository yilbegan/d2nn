import math
from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftfreq, ifft2  # pyright: ignore[reportUnknownVariableType]

type TransferFunction = Literal["exact", "fresnel"]
type Distance = float | torch.Tensor


@dataclass(frozen=True, slots=True)
class PropagationConditions:
    distance_std: float = 0.0

    def __post_init__(self) -> None:
        distance_std = cast(object, self.distance_std)
        if isinstance(distance_std, bool) or not isinstance(distance_std, int | float):
            raise TypeError("distance_std must be a number")
        if not isfinite(distance_std) or distance_std < 0:
            raise ValueError("distance_std must be finite and non-negative")


def _as_distance_tensor(distance: Distance, dtype: torch.dtype) -> torch.Tensor:
    real_dtype = (
        torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32
    )
    distance_tensor = torch.as_tensor(distance, dtype=real_dtype)
    if distance_tensor.numel() != 1:
        raise ValueError("distance must be a scalar")
    return distance_tensor


def _frequency_grid(
    size: int,
    pixel_size: float,
    distance: Distance,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if size < 2:
        raise ValueError("size must be at least 2")

    distance_tensor = _as_distance_tensor(distance, dtype)
    frequencies = cast(
        torch.Tensor,
        fftfreq(
            size,
            d=pixel_size,
            device=distance_tensor.device,
            dtype=distance_tensor.dtype,
        ),
    )
    fx, fy = cast(
        tuple[torch.Tensor, torch.Tensor],
        torch.meshgrid(frequencies, frequencies, indexing="ij"),
    )
    return distance_tensor, fx, fy, frequencies[1] - frequencies[0]


def exact_transfer_function(
    size: int,
    pixel_size: float,
    wavelength: float,
    distance: Distance,
    band_limit: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    distance_tensor, fx, fy, frequency_step = _frequency_grid(
        size, pixel_size, distance, dtype
    )

    wave_number = math.tau / wavelength
    root = (1 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2).to(dtype)
    transfer = torch.exp(1j * wave_number * distance_tensor * torch.sqrt(root))

    if band_limit:
        frequency_limit = 1.0 / (
            wavelength * torch.sqrt((2 * distance_tensor * frequency_step) ** 2 + 1)
        )
        mask = (fx.abs() <= frequency_limit) & (fy.abs() <= frequency_limit)
        transfer = transfer * mask

    return transfer.to(dtype)


def fresnel_transfer_function(
    size: int,
    pixel_size: float,
    wavelength: float,
    distance: Distance,
    band_limit: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    distance_tensor, fx, fy, frequency_step = _frequency_grid(
        size, pixel_size, distance, dtype
    )

    phase = distance_tensor * (
        math.tau / wavelength - math.pi * wavelength * (fx**2 + fy**2)
    )
    transfer = torch.exp(1j * phase)

    if band_limit:
        frequency_limit = 1.0 / (
            2 * wavelength * distance_tensor.abs() * frequency_step
        )
        mask = (fx.abs() <= frequency_limit) & (fy.abs() <= frequency_limit)
        transfer = transfer * mask

    return transfer.to(dtype)


class Propagation(nn.Module):
    def __init__(
        self,
        size: int,
        pixel_size: float,
        wavelength: float,
        distance: float,
        transfer_function: TransferFunction = "exact",
        zero_pad: bool | int = True,
    ) -> None:
        super().__init__()

        if isinstance(zero_pad, bool):
            zero_pad = size // 2 if zero_pad else 0

        self.size: int = size + 2 * zero_pad
        self.pixel_size: float = pixel_size
        self.wavelength: float = wavelength
        self.distance: float = distance
        self.transfer_function: TransferFunction = transfer_function
        self.zero_pad: int = zero_pad
        self.conditions: PropagationConditions | None = None
        h = self._build_transfer_function(distance)
        self.register_buffer("h", h)
        self.h: torch.Tensor = h

    def set_conditions(self, conditions: PropagationConditions | None) -> None:
        self.conditions = conditions

    @override
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        pad = self.zero_pad
        if pad:
            field = F.pad(field, (pad, pad, pad, pad))

        transfer = self._active_transfer_function()
        spectrum = cast(torch.Tensor, fft2(field))
        field = cast(torch.Tensor, ifft2(spectrum * transfer))

        if pad:
            field = field[..., pad:-pad, pad:-pad]

        return field

    def _active_transfer_function(self) -> torch.Tensor:
        conditions = self.conditions
        if not self.training or conditions is None or conditions.distance_std == 0:
            return self.h

        noise = torch.randn((), device=self.h.device, dtype=self.h.real.dtype)
        noisy_distance = self.h.real.new_tensor(self.distance)
        noisy_distance = noisy_distance + conditions.distance_std * noise
        return self._build_transfer_function(noisy_distance, dtype=self.h.dtype)

    def _build_transfer_function(
        self, distance: Distance, dtype: torch.dtype = torch.complex64
    ) -> torch.Tensor:
        match self.transfer_function:
            case "exact":
                return exact_transfer_function(
                    self.size,
                    self.pixel_size,
                    self.wavelength,
                    distance,
                    dtype=dtype,
                )
            case "fresnel":
                return fresnel_transfer_function(
                    self.size,
                    self.pixel_size,
                    self.wavelength,
                    distance,
                    dtype=dtype,
                )
