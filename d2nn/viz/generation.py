import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist

from ..models import DiffractiveGenerativeModel

type Frame = np.ndarray[tuple[int, int], np.dtype[np.float32]]


def plot_grid(
    images: np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    path: pathlib.Path | str,
    num_classes: int,
    samples_per_digit: int,
) -> None:
    fig, axes = plt.subplots(
        num_classes,
        samples_per_digit,
        figsize=(1.4 * samples_per_digit, 1.4 * num_classes),
        squeeze=False,
    )
    for d in range(num_classes):
        for s in range(samples_per_digit):
            ax = axes[d][s]
            ax.imshow(images[d * samples_per_digit + s], cmap="inferno")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[d][0].set_ylabel(str(d), fontsize=12, rotation=0, labelpad=12, va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_generated_digits(
    model: DiffractiveGenerativeModel,
    device: torch.device,
    path: pathlib.Path | str,
    num_classes: int = 10,
    samples_per_digit: int = 8,
    output_size: int = 32,
) -> None:
    model.eval()
    noise_size = model.encoder.config.in_size

    labels = torch.arange(num_classes, device=device).repeat_interleave(
        samples_per_digit
    )
    noise = torch.randn(len(labels), 1, noise_size, noise_size, device=device)
    images, _ = model(noise, labels)

    pooled = torch.nn.functional.adaptive_avg_pool2d(
        images.unsqueeze(1), output_size=output_size
    ).squeeze(1)

    path = pathlib.Path(path)
    plot_grid(
        torch.clamp(
            images,
            min=torch.quantile(images, 0.7),
            max=torch.quantile(images, 0.98),
        )
        .cpu()
        .numpy(),
        path.with_stem(f"{path.stem}_raw"),
        num_classes,
        samples_per_digit,
    )

    plot_grid(
        pooled.cpu().numpy(),
        path.with_stem(f"{path.stem}_pooled"),
        num_classes,
        samples_per_digit,
    )


def intensity_frame(
    field: torch.Tensor,
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    intensity = (field.abs() ** 2)[0].cpu().numpy()
    low, high = np.quantile(intensity, [0.05, 0.95])
    return np.clip(intensity, low, high)


@torch.no_grad()
def decoder_intensities(
    model: DiffractiveGenerativeModel,
    device: torch.device,
    label: int,
    noise: torch.Tensor | None,
) -> tuple[list[Frame], list[str]]:
    model.eval()
    noise_size = model.encoder.config.in_size
    if noise is None:
        noise = torch.randn(1, 1, noise_size, noise_size, device=device)

    labels = torch.tensor([label], device=device)
    field, _ = model.encoder(noise, labels)

    intensities: list[Frame] = []
    titles: list[str] = []
    decoder = model.decoder
    for idx, layer in enumerate(decoder.layers):
        field = layer(field)
        intensities.append(intensity_frame(field))
        titles.append(f"layer {idx + 1}")

    field = decoder.output_propagation(field)
    intensities.append(intensity_frame(field))
    titles.append("output plane")

    return intensities, titles


def plot_intensity_chart(
    intensities: list[Frame],
    titles: list[str],
    path: pathlib.Path,
) -> None:
    n = len(intensities)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for idx, (img, title) in enumerate(zip(intensities, titles)):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(img, cmap="inferno")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def animate_intensity(
    intensities: list[Frame],
    titles: list[str],
    path: pathlib.Path,
    delay: float,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(intensities[0], cmap="inferno")

    def update(i: int) -> list[Artist]:
        frame = intensities[i]
        im.set_data(frame)
        im.set_clim(vmin=0.0, vmax=float(frame.max()) or 1.0)
        ax.set_title(titles[i])
        return [im]

    anim = FuncAnimation(
        fig, update, frames=len(intensities), interval=delay * 1000, blit=False
    )
    anim.save(path, writer=PillowWriter(fps=round(1.0 / delay)))
    plt.close(fig)


def plot_decoder_intensity(
    model: DiffractiveGenerativeModel,
    device: torch.device,
    path: pathlib.Path | str,
    label: int = 0,
    noise: torch.Tensor | None = None,
    delay: float = 1.0,
) -> None:
    intensities, titles = decoder_intensities(model, device, label, noise)
    path = pathlib.Path(path)
    plot_intensity_chart(intensities, titles, path)
    animate_intensity(intensities, titles, path.with_suffix(".gif"), delay)
