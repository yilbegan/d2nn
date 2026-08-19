import math
import pathlib
from collections.abc import Iterable
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ..optics import DiffractiveLayer


def plot_phase_masks(modules: Iterable[nn.Module], path: pathlib.Path | str) -> None:
    layers = cast(list[DiffractiveLayer], list(modules))
    n = len(layers)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for idx, layer in enumerate(layers):
        ax = axes[idx // cols][idx % cols]
        phase = cast(torch.Tensor, layer.phase()).detach().cpu().numpy()
        wrapped = np.mod(phase, 2 * np.pi)
        im = ax.imshow(wrapped, cmap="twilight", vmin=0, vmax=2 * np.pi)
        ax.set_title(f"layer {idx + 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phase (rad)")

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
