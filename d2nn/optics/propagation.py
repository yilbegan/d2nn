import math
from enum import StrEnum

import torch
import torch.nn as nn


class TransferFunction(StrEnum):
    EXACT = "exact"
    FRESNEL = "fresnel"


def exact_transfer_function(
    size: int,
    pixel_size: float,
    wavelength: float,
    distance: float,
    band_limit: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    freqs = torch.fft.fftfreq(size, d=pixel_size)
    fx, fy = torch.meshgrid(freqs, freqs, indexing="ij")

    k = 2 * math.pi / wavelength
    root = (1 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2).to(dtype)
    phase = k * distance * torch.sqrt(root)
    h = torch.exp(1j * phase)

    if band_limit:
        df = freqs[1] - freqs[0]
        f_lim = 1.0 / (wavelength * math.sqrt((2 * distance * df) ** 2 + 1))
        mask = (fx.abs() <= f_lim) & (fy.abs() <= f_lim)
        h = h * mask

    return h.to(dtype)


def fresnel_transfer_function(
    size: int,
    pixel_size: float,
    wavelength: float,
    distance: float,
    band_limit: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    freqs = torch.fft.fftfreq(size, d=pixel_size)
    fx, fy = torch.meshgrid(freqs, freqs, indexing="ij")

    k = 2 * math.pi / wavelength
    phase = k * distance - math.pi * wavelength * distance * (fx**2 + fy**2)
    h = torch.exp(1j * phase)

    if band_limit:
        df = freqs[1] - freqs[0]
        f_lim = 1.0 / (2 * wavelength * distance * df)
        mask = (fx.abs() <= f_lim) & (fy.abs() <= f_lim)
        h = h * mask

    return h.to(dtype)


class Propagation(nn.Module):
    def __init__(
        self,
        size: int,
        pixel_size: float,
        wavelength: float,
        distance: float,
        transfer_function: TransferFunction = TransferFunction.EXACT,
        zero_pad: bool | int = True,
    ) -> None:
        super().__init__()

        if isinstance(zero_pad, bool):
            zero_pad = size // 2 if zero_pad else 0

        self.zero_pad: int = zero_pad
        padded_size = size + 2 * zero_pad

        match transfer_function:
            case TransferFunction.EXACT:
                h = exact_transfer_function(
                    padded_size, pixel_size, wavelength, distance
                )
            case TransferFunction.FRESNEL:
                h = fresnel_transfer_function(
                    padded_size, pixel_size, wavelength, distance
                )

        self.register_buffer("h", h)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        pad = self.zero_pad
        if pad:
            field = torch.nn.functional.pad(field, (pad, pad, pad, pad))

        field = torch.fft.ifft2(torch.fft.fft2(field) * self.h)

        if pad:
            field = field[..., pad:-pad, pad:-pad]

        return field
