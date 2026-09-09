import math
from typing import override

import torch
import torch.nn as nn


def rounded_rows(num_classes: int) -> list[int]:
    n_rows = max(1, round(math.sqrt(num_classes)))
    base, extra = divmod(num_classes, n_rows)
    counts = [base] * n_rows
    center = (n_rows - 1) / 2
    for r in sorted(range(n_rows), key=lambda r: abs(r - center))[:extra]:
        counts[r] += 1
    return counts


def detector_masks(
    size: int,
    num_classes: int,
    det_size: int,
    rows: list[int] | None = None,
) -> torch.Tensor:
    if rows is None:
        rows = rounded_rows(num_classes)

    masks = torch.zeros(num_classes, size, size)
    centers_y = torch.linspace(0, size, len(rows) + 2)[1:-1]
    half = det_size // 2

    k = 0
    for r, count in enumerate(rows):
        cy = int(centers_y[r])
        centers_x = torch.linspace(0, size, count + 2)[1:-1]
        for cx in centers_x:
            cx = int(cx)
            masks[k, cy - half : cy + half, cx - half : cx + half] = 1.0
            k += 1
    return masks


class Detector(nn.Module):
    def __init__(
        self,
        size: int,
        num_classes: int = 10,
        det_size: int = 20,
        rows: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_points: int = det_size**2
        self.register_buffer("masks", detector_masks(size, num_classes, det_size, rows))

    @override
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        intensity = field.abs() ** 2  # (B, H, W)
        return torch.einsum("bhw,chw->bc", intensity, self.masks.to(intensity.dtype))
