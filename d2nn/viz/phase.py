# pyright: reportUnknownMemberType=false

import math
from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn

from ..optics import DiffractiveLayer
from .plotting import FigureGrid, OutputPath, float_matrix

__all__ = ["plot_phase_masks"]


def plot_phase_masks(modules: Iterable[nn.Module], path: OutputPath) -> None:
    layers: list[DiffractiveLayer] = []
    for index, module in enumerate(modules):
        if not isinstance(module, DiffractiveLayer):
            received = type(module).__name__
            raise TypeError(
                f"modules[{index}] must be a DiffractiveLayer, got {received}"
            )
        layers.append(module)

    grid = FigureGrid.create(len(layers))
    for idx, layer in enumerate(layers):
        axis = grid.axes[idx]
        phase = cast(torch.Tensor, layer.phase())
        wrapped = float_matrix(torch.remainder(phase, math.tau))
        image = axis.imshow(wrapped, cmap="twilight", vmin=0.0, vmax=math.tau)
        _ = axis.set_title(f"layer {idx + 1}")
        _ = axis.set_xticks([])
        _ = axis.set_yticks([])
        _ = grid.figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            label="phase (rad)",
        )

    grid.save(path)
