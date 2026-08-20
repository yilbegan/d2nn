# pyright: reportUnknownMemberType=false

from collections.abc import Iterable
from typing import cast

import torch

from ..data import Batch
from ..models.classifier import DiffractiveClassifier
from .plotting import (
    FigureGrid,
    OutputPath,
    float_at,
    float_images,
    float_matrix,
    image_at,
    int_at,
    int_matrix,
    row_at,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_energy_distribution",
    "plot_input_output_table",
]

DOWNSCALE = 5

type Batches = Iterable[Batch]


def _require_num_classes(num_classes: int) -> None:
    if num_classes < 1:
        raise ValueError("num_classes must be positive")


def _images_without_channel(images: torch.Tensor) -> torch.Tensor:
    images = images.squeeze(1)
    if images.ndim != 3:
        raise ValueError("images must have shape (batch, 1, height, width)")
    return images


def _targets_on_cpu(
    targets: torch.Tensor,
    *,
    batch_size: int,
    num_classes: int,
) -> torch.Tensor:
    targets = targets.detach().to(device="cpu", dtype=torch.long)
    if targets.ndim != 1 or targets.shape[0] != batch_size:
        raise ValueError("targets must have shape (batch,)")
    if torch.any((targets < 0) | (targets >= num_classes)):
        raise ValueError(f"targets must be between 0 and {num_classes - 1}")
    return targets


def detector_field(model: DiffractiveClassifier, images: torch.Tensor) -> torch.Tensor:
    field = cast(torch.Tensor, model.stack(images.to(torch.complex64)))
    return field.abs().square().detach().cpu()


def downscale(x: torch.Tensor, factor: int = DOWNSCALE) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, height, width)")
    if factor < 1:
        raise ValueError("factor must be positive")
    return torch.nn.functional.avg_pool2d(x.unsqueeze(1), factor).squeeze(1)


def sample_one_per_digit(
    batches: Batches,
    num_classes: int = 10,
) -> tuple[torch.Tensor, list[int]]:
    _require_num_classes(num_classes)
    found: dict[int, torch.Tensor] = {}
    for images, targets in batches:
        images = _images_without_channel(images)
        targets = _targets_on_cpu(
            targets,
            batch_size=images.shape[0],
            num_classes=num_classes,
        )
        for index_tensor in torch.randperm(images.shape[0]):
            index = int(index_tensor)
            label = int(targets[index])
            if label not in found:
                found[label] = images[index]
        if len(found) == num_classes:
            break

    missing = sorted(set(range(num_classes)) - found.keys())
    if missing:
        raise ValueError(f"batches contain no samples for classes {missing}")

    labels = list(range(num_classes))
    return torch.stack([found[label] for label in labels]), labels


@torch.no_grad()
def energy_distribution(
    model: DiffractiveClassifier,
    batches: Batches,
    device: torch.device,
    num_classes: int = 10,
) -> torch.Tensor:
    _require_num_classes(num_classes)
    _ = model.eval()
    totals = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.float32)

    for images, targets in batches:
        images = _images_without_channel(images).to(device)
        targets = _targets_on_cpu(
            targets,
            batch_size=images.shape[0],
            num_classes=num_classes,
        )
        scores = (
            cast(torch.Tensor, model(images))
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        expected_shape = (images.shape[0], num_classes)
        if scores.shape != expected_shape:
            raise ValueError(
                f"model scores must have shape {expected_shape}, got {tuple(scores.shape)}"
            )

        _ = totals.index_add_(0, targets, scores)
        counts += torch.bincount(targets, minlength=num_classes).to(counts.dtype)

    means = totals / counts.clamp_min(1).unsqueeze(1)
    return 100.0 * means / means.sum(dim=1, keepdim=True).clamp_min(1e-12)


def plot_energy_distribution(
    model: DiffractiveClassifier,
    batches: Batches,
    device: torch.device,
    path: OutputPath,
    num_classes: int = 10,
) -> None:
    distribution = float_matrix(
        energy_distribution(model, batches, device, num_classes)
    )
    grid = FigureGrid.create(
        num_classes,
        cell_size=(2.6, 2.2),
        sharey=True,
    )
    regions = list(range(num_classes))
    for digit, axis in enumerate(grid.axes):
        colors = ["tab:orange" if region == digit else "tab:blue" for region in regions]
        _ = axis.bar(regions, row_at(distribution, digit), color=colors)
        _ = axis.set_title(
            f"digit {digit}  ({float_at(distribution, digit, digit):.0f}% on target)",
            fontsize=9,
        )
        _ = axis.set_xticks(regions)
        axis.tick_params(labelsize=7)
        _ = axis.set_ylim(0, 100)

    _ = grid.figure.supxlabel("detector region")
    _ = grid.figure.supylabel("intensity share (%)")
    grid.save(path)


@torch.no_grad()
def confusion_matrix(
    model: DiffractiveClassifier,
    batches: Batches,
    device: torch.device,
    num_classes: int = 10,
) -> torch.Tensor:
    _require_num_classes(num_classes)
    _ = model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, targets in batches:
        images = _images_without_channel(images).to(device)
        targets = _targets_on_cpu(
            targets,
            batch_size=images.shape[0],
            num_classes=num_classes,
        )
        scores = cast(torch.Tensor, model(images))
        expected_shape = (images.shape[0], num_classes)
        if scores.shape != expected_shape:
            raise ValueError(
                f"model scores must have shape {expected_shape}, got {tuple(scores.shape)}"
            )
        predictions = scores.argmax(dim=1).detach().to(device="cpu")
        flat_indices = targets * num_classes + predictions
        matrix += torch.bincount(
            flat_indices,
            minlength=num_classes**2,
        ).reshape(num_classes, num_classes)

    return matrix


def plot_confusion_matrix(
    model: DiffractiveClassifier,
    batches: Batches,
    device: torch.device,
    path: OutputPath,
    num_classes: int = 10,
) -> None:
    matrix = confusion_matrix(model, batches, device, num_classes)
    normalized = matrix / matrix.sum(dim=1, keepdim=True).clamp_min(1)
    matrix_values = int_matrix(matrix)
    normalized_values = float_matrix(normalized)

    grid = FigureGrid.create(
        1,
        cell_size=(0.6 * num_classes + 1.5, 0.6 * num_classes + 1.0),
    )
    axis = grid.axes[0]
    image = axis.imshow(normalized_values, cmap="Blues", vmin=0.0, vmax=1.0)
    regions = list(range(num_classes))
    _ = axis.set_xticks(regions)
    _ = axis.set_yticks(regions)
    _ = axis.set_xlabel("predicted")
    _ = axis.set_ylabel("true")
    _ = axis.set_title("Confusion matrix")
    for row in range(num_classes):
        for column in range(num_classes):
            count = int_at(matrix_values, row, column)
            if count:
                _ = axis.text(
                    column,
                    row,
                    str(count),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if float_at(normalized_values, row, column) > 0.5
                        else "black"
                    ),
                    fontsize=7,
                )
    _ = grid.figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="row-normalized",
    )
    grid.save(path)


def plot_input_output_table(
    model: DiffractiveClassifier,
    batches: Batches,
    device: torch.device,
    path: OutputPath,
    num_classes: int = 10,
) -> None:
    _require_num_classes(num_classes)
    _ = model.eval()
    images, labels = sample_one_per_digit(batches, num_classes)
    with torch.no_grad():
        intensity = detector_field(model, images.to(device))

    masks = cast(torch.Tensor, model.detector.masks).detach().cpu()
    expected_mask_shape = (num_classes, *intensity.shape[-2:])
    if masks.shape != expected_mask_shape:
        raise ValueError(f"detector masks must have shape {expected_mask_shape}")

    energy = torch.einsum("nhw,chw->nc", intensity, masks)
    distribution = 100.0 * energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-12)

    outputs = downscale(intensity)
    scaled_masks = downscale(masks)
    any_mask = scaled_masks.amax(dim=0) > 0.5
    masked_outputs = outputs * any_mask

    input_values = float_images(images)
    output_values = float_images(outputs)
    mask_values = float_images(scaled_masks)
    masked_output_values = float_images(masked_outputs)
    distribution_values = float_matrix(distribution)

    sample_count = len(labels)
    grid = FigureGrid.create(
        4 * sample_count,
        columns=sample_count,
        cell_size=(1.8, 2.0),
    )
    regions = list(range(num_classes))
    for column, digit in enumerate(labels):
        input_axis = grid.axes[column]
        _ = input_axis.imshow(image_at(input_values, column), cmap="gray")
        _ = input_axis.set_title(f"digit {digit}")
        _ = input_axis.set_xticks([])
        _ = input_axis.set_yticks([])

        output_axis = grid.axes[sample_count + column]
        _ = output_axis.imshow(image_at(output_values, column), cmap="inferno")
        masked_axis = grid.axes[2 * sample_count + column]
        _ = masked_axis.imshow(image_at(masked_output_values, column), cmap="inferno")
        for region in regions:
            color = "cyan" if region == digit else "white"
            width = 1.2 if region == digit else 0.5
            _ = output_axis.contour(
                image_at(mask_values, region),
                levels=[0.5],
                colors=color,
                linewidths=width,
            )
            _ = masked_axis.contour(
                image_at(mask_values, region),
                levels=[0.5],
                colors=color,
                linewidths=width,
            )
        for axis in (output_axis, masked_axis):
            _ = axis.set_xticks([])
            _ = axis.set_yticks([])

        bar_axis = grid.axes[3 * sample_count + column]
        colors = ["tab:orange" if region == digit else "tab:blue" for region in regions]
        _ = bar_axis.bar(regions, row_at(distribution_values, column), color=colors)
        _ = bar_axis.set_title(
            f"{float_at(distribution_values, column, digit):.0f}% on target",
            fontsize=8,
        )
        _ = bar_axis.set_xticks(regions)
        bar_axis.tick_params(labelsize=6)
        _ = bar_axis.set_ylim(0, 100)
        if column > 0:
            _ = bar_axis.set_yticks([])

    _ = grid.axes[0].set_ylabel("input", fontsize=11)
    _ = grid.axes[sample_count].set_ylabel("detector\nintensity", fontsize=11)
    _ = grid.axes[2 * sample_count].set_ylabel("mask\nintensity", fontsize=11)
    _ = grid.axes[3 * sample_count].set_ylabel("energy\nshare (%)", fontsize=11)
    grid.save(path)
