import math
import pathlib
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import Batch
from ..models import DiffractiveClassifier

DOWNSCALE = 5


def detector_field(model: DiffractiveClassifier, images: torch.Tensor) -> torch.Tensor:
    field = images.to(torch.complex64)
    field = model.stack(field)
    return (field.abs() ** 2).detach().cpu()


def downscale(x: torch.Tensor, factor: int = DOWNSCALE) -> torch.Tensor:
    return torch.nn.functional.avg_pool2d(x.unsqueeze(1), kernel_size=factor).squeeze(1)


def sample_one_per_digit(
    loader: DataLoader[Batch], num_classes: int = 10
) -> tuple[torch.Tensor, list[int]]:
    found: dict[int, torch.Tensor] = {}
    for images, targets in loader:
        images = images.squeeze(1)  # (B, H, W)
        perm = torch.randperm(images.size(0))
        for i in perm.tolist():
            label = int(targets[i])
            found.setdefault(label, images[i])
        if len(found) >= num_classes:
            break

    labels = sorted(k for k in found if k < num_classes)
    stacked = torch.stack([found[d] for d in labels])
    return stacked, labels


def energy_distribution(
    model: DiffractiveClassifier,
    loader: DataLoader[Batch],
    device: torch.device,
    num_classes: int = 10,
) -> torch.Tensor:
    model.eval()
    totals = torch.zeros(num_classes, num_classes)
    counts = torch.zeros(num_classes)
    with torch.no_grad():
        for images, targets in loader:
            images = images.squeeze(1).to(device)
            scores = model(images).cpu()  # (B, num_classes)
            for d in range(num_classes):
                sel = targets == d
                if sel.any():
                    totals[d] += scores[sel].sum(dim=0)
                    counts[d] += int(sel.sum())
    means = totals / counts.clamp(min=1).unsqueeze(1)
    return 100.0 * means / means.sum(dim=1, keepdim=True).clamp(min=1e-12)


def plot_energy_distribution(
    model: DiffractiveClassifier,
    loader: DataLoader[Batch],
    device: torch.device,
    path: pathlib.Path | str,
    num_classes: int = 10,
) -> None:
    dist = energy_distribution(model, loader, device, num_classes).cpu().numpy()

    cols = math.ceil(math.sqrt(num_classes))
    rows = math.ceil(num_classes / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.6 * cols, 2.2 * rows), squeeze=False, sharey=True
    )
    regions = np.arange(num_classes)
    for d in range(num_classes):
        ax = axes[d // cols][d % cols]
        colors = ["tab:orange" if k == d else "tab:blue" for k in regions]
        ax.bar(regions, dist[d], color=colors)
        ax.set_title(f"digit {d}  ({dist[d, d]:.0f}% on target)", fontsize=9)
        ax.set_xticks(regions)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 100)

    for idx in range(num_classes, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.supxlabel("detector region")
    fig.supylabel("intensity share (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix(
    model: DiffractiveClassifier,
    loader: DataLoader[Batch],
    device: torch.device,
    num_classes: int = 10,
) -> torch.Tensor:
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for images, targets in loader:
            images = images.squeeze(1).to(device)  # (B, H, W)
            preds = model(images).argmax(dim=1).cpu()
            for t, p in zip(targets.tolist(), preds.tolist()):
                matrix[t, p] += 1
    return matrix


def plot_confusion_matrix(
    model: DiffractiveClassifier,
    loader: DataLoader[Batch],
    device: torch.device,
    path: pathlib.Path | str,
    num_classes: int = 10,
) -> None:
    matrix = confusion_matrix(model, loader, device, num_classes).numpy()
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.clip(row_sums, 1, None)

    fig, ax = plt.subplots(figsize=(0.6 * num_classes + 1.5, 0.6 * num_classes + 1.0))
    im = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    regions = np.arange(num_classes)
    ax.set_xticks(regions)
    ax.set_yticks(regions)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    for i in range(num_classes):
        for j in range(num_classes):
            count = int(matrix[i, j])
            if count:
                ax.text(
                    j,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] > 0.5 else "black",
                    fontsize=7,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_input_output_table(
    model: DiffractiveClassifier,
    loader: DataLoader[Batch],
    device: torch.device,
    path: pathlib.Path | str,
    num_classes: int = 10,
) -> None:
    model.eval()
    images, labels = sample_one_per_digit(loader, num_classes)
    with torch.no_grad():
        intensity = detector_field(model, images.to(device))  # (N, H, W)

    masks = cast(torch.Tensor, model.detector.masks).detach().cpu()

    energy = torch.einsum("nhw,chw->nc", intensity, masks)
    dist = (100.0 * energy / energy.sum(dim=1, keepdim=True).clamp(min=1e-12)).numpy()

    outputs = downscale(intensity)
    masks = downscale(masks)
    any_mask = masks.max(dim=0).values > 0.5
    masked_outputs = (outputs * any_mask).numpy()
    outputs = outputs.numpy()
    masks = masks.numpy()

    inputs = images.cpu().numpy()
    n = len(labels)
    regions = np.arange(num_classes)

    fig, axes = plt.subplots(4, n, figsize=(1.8 * n, 8.0), squeeze=False)
    for col, digit in enumerate(labels):
        ax_in = axes[0][col]
        ax_in.imshow(inputs[col], cmap="gray")
        ax_in.set_title(f"digit {digit}")
        ax_in.set_xticks([])
        ax_in.set_yticks([])

        ax_out = axes[1][col]
        ax_out.imshow(outputs[col], cmap="inferno")
        for k in range(masks.shape[0]):
            ax_out.contour(
                masks[k],
                levels=[0.5],
                colors="cyan" if k == digit else "white",
                linewidths=1.2 if k == digit else 0.5,
            )
        ax_out.set_xticks([])
        ax_out.set_yticks([])

        ax_mask = axes[2][col]
        ax_mask.imshow(masked_outputs[col], cmap="inferno")
        for k in range(masks.shape[0]):
            ax_mask.contour(
                masks[k],
                levels=[0.5],
                colors="cyan" if k == digit else "white",
                linewidths=1.2 if k == digit else 0.5,
            )
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])

        ax_bar = axes[3][col]
        colors = ["tab:orange" if k == digit else "tab:blue" for k in regions]
        ax_bar.bar(regions, dist[col], color=colors)
        ax_bar.set_title(f"{dist[col, digit]:.0f}% on target", fontsize=8)
        ax_bar.set_xticks(regions)
        ax_bar.tick_params(labelsize=6)
        ax_bar.set_ylim(0, 100)
        if col > 0:
            ax_bar.set_yticklabels([])

    axes[0][0].set_ylabel("input", fontsize=11)
    axes[1][0].set_ylabel("detector\nintensity", fontsize=11)
    axes[2][0].set_ylabel("mask\nintensity", fontsize=11)
    axes[3][0].set_ylabel("energy\nshare (%)", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
