# pyright: reportUnknownMemberType=false

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist

from ..models.generative import DiffractiveGenerativeModel
from ..optics import DiffractiveLayer
from .plotting import (
    FigureGrid,
    FloatImages,
    FloatMatrix,
    OutputPath,
    float_images,
    float_matrix,
    image_at,
)

__all__ = ["plot_decoder_intensity", "plot_generated_digits"]

type Frame = FloatMatrix


def plot_grid(
    images: FloatImages,
    path: OutputPath,
    num_classes: int,
    samples_per_digit: int,
) -> None:
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if samples_per_digit < 1:
        raise ValueError("samples_per_digit must be positive")
    if images.ndim != 3:
        raise ValueError("images must have shape (samples, height, width)")

    expected_samples = num_classes * samples_per_digit
    if images.shape[0] != expected_samples:
        raise ValueError(f"expected {expected_samples} images, got {images.shape[0]}")

    grid = FigureGrid.create(
        expected_samples,
        columns=samples_per_digit,
        cell_size=(1.4, 1.4),
    )
    for digit in range(num_classes):
        for sample in range(samples_per_digit):
            index = digit * samples_per_digit + sample
            axis = grid.axes[index]
            _ = axis.imshow(image_at(images, index), cmap="inferno")
            _ = axis.set_xticks([])
            _ = axis.set_yticks([])
        _ = grid.axes[digit * samples_per_digit].set_ylabel(
            str(digit),
            fontsize=12,
            rotation=0,
            labelpad=12,
            va="center",
        )

    grid.save(path)


@torch.no_grad()
def plot_generated_digits(
    model: DiffractiveGenerativeModel,
    device: torch.device,
    path: OutputPath,
    num_classes: int = 10,
    samples_per_digit: int = 8,
    output_size: int = 32,
) -> None:
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if samples_per_digit < 1:
        raise ValueError("samples_per_digit must be positive")
    if output_size < 1:
        raise ValueError("output_size must be positive")

    _ = model.eval()
    noise_size = model.encoder.config.in_size
    labels = torch.arange(num_classes, device=device).repeat_interleave(
        samples_per_digit
    )
    noise = torch.randn(len(labels), 1, noise_size, noise_size, device=device)
    images, _scale = cast(tuple[torch.Tensor, torch.Tensor], model(noise, labels))

    pooled = torch.nn.functional.adaptive_avg_pool2d(
        images.unsqueeze(1), output_size=output_size
    ).squeeze(1)
    clipped = torch.clamp(
        images,
        min=torch.quantile(images, 0.7),
        max=torch.quantile(images, 0.98),
    )

    output_path = Path(path)
    plot_grid(
        float_images(clipped),
        output_path.with_stem(f"{output_path.stem}_raw"),
        num_classes,
        samples_per_digit,
    )
    plot_grid(
        float_images(pooled),
        output_path.with_stem(f"{output_path.stem}_pooled"),
        num_classes,
        samples_per_digit,
    )


def intensity_frame(field: torch.Tensor) -> Frame:
    if field.ndim != 3 or field.shape[0] < 1:
        raise ValueError("field must have shape (batch, height, width) with batch > 0")

    intensity = field.abs().square()[0]
    low = torch.quantile(intensity, 0.05)
    high = torch.quantile(intensity, 0.95)
    return float_matrix(torch.clamp(intensity, min=low, max=high))


@torch.no_grad()
def decoder_intensities(
    model: DiffractiveGenerativeModel,
    device: torch.device,
    label: int,
    noise: torch.Tensor | None,
) -> tuple[list[Frame], list[str]]:
    num_classes = model.encoder.config.num_classes
    if not 0 <= label < num_classes:
        raise ValueError(f"label must be between 0 and {num_classes - 1}")

    _ = model.eval()
    noise_size = model.encoder.config.in_size
    if noise is None:
        noise = torch.randn(1, 1, noise_size, noise_size, device=device)
    else:
        noise = noise.to(device)

    labels = torch.tensor([label], device=device)
    field, _scale = cast(
        tuple[torch.Tensor, torch.Tensor], model.encoder(noise, labels)
    )

    intensities: list[Frame] = []
    titles: list[str] = []
    stack = model.decoder.stack
    for idx, module in enumerate(stack.layers):
        if not isinstance(module, DiffractiveLayer):
            received = type(module).__name__
            raise TypeError(
                f"decoder layer {idx} must be a DiffractiveLayer, got {received}"
            )
        field = cast(torch.Tensor, module(field))
        intensities.append(intensity_frame(field))
        titles.append(f"layer {idx + 1}")

    field = cast(torch.Tensor, stack.output_propagation(field))
    intensities.append(intensity_frame(field))
    titles.append("output plane")
    return intensities, titles


def _validate_frames(intensities: Sequence[Frame], titles: Sequence[str]) -> None:
    if not intensities:
        raise ValueError("intensities must not be empty")
    if len(intensities) != len(titles):
        raise ValueError("intensities and titles must have the same length")


def plot_intensity_chart(
    intensities: Sequence[Frame],
    titles: Sequence[str],
    path: OutputPath,
) -> None:
    _validate_frames(intensities, titles)
    grid = FigureGrid.create(len(intensities), cell_size=(3.0, 3.0))
    for axis, image, title in zip(grid.axes, intensities, titles, strict=True):
        _ = axis.imshow(image, cmap="inferno")
        _ = axis.set_title(title)
        _ = axis.set_xticks([])
        _ = axis.set_yticks([])

    grid.save(path)


def animate_intensity(
    intensities: Sequence[Frame],
    titles: Sequence[str],
    path: OutputPath,
    delay: float,
) -> None:
    _validate_frames(intensities, titles)
    if delay <= 0:
        raise ValueError("delay must be positive")

    figure, axis = plt.subplots(figsize=(4.0, 4.0))
    _ = axis.set_xticks([])
    _ = axis.set_yticks([])
    image = axis.imshow(intensities[0], cmap="inferno")

    def update(index: int) -> Sequence[Artist]:
        frame = intensities[index]
        image.set_data(frame)
        frame_max = float(cast(float, frame.max()))
        image.set_clim(vmin=0.0, vmax=frame_max if frame_max > 0.0 else 1.0)
        _ = axis.set_title(titles[index])
        return (image,)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(intensities),
        interval=delay * 1000,
        blit=False,
    )
    try:
        animation.save(
            Path(path),
            writer=PillowWriter(fps=max(1, round(1.0 / delay))),
        )
    finally:
        plt.close(figure)


def plot_decoder_intensity(
    model: DiffractiveGenerativeModel,
    device: torch.device,
    path: OutputPath,
    label: int = 0,
    noise: torch.Tensor | None = None,
    delay: float = 1.0,
) -> None:
    intensities, titles = decoder_intensities(model, device, label, noise)
    output_path = Path(path)
    plot_intensity_chart(intensities, titles, output_path)
    animate_intensity(intensities, titles, output_path.with_suffix(".gif"), delay)
