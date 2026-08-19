from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import (
    DDPMScheduler,
    DDPMSchedulerOutput,
)

from d2nn.data import Batch

from .types import TeacherCache

__all__ = [
    "TeacherCacheBuildConfig",
    "TeacherEpochCallback",
    "TeacherEpochResult",
    "TeacherModelConfig",
    "TeacherTrainingConfig",
    "build_teacher",
    "build_teacher_cache",
    "load_teacher",
    "sample_teacher",
    "save_teacher",
    "train_teacher",
]


class _BatchLoader(Protocol):
    def __iter__(self) -> Iterator[Batch]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True, slots=True)
class TeacherModelConfig:
    size: int = 32
    num_classes: int = 10

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("teacher size must be positive")
        if self.num_classes < 2:
            raise ValueError("teacher num_classes must be at least 2")


@dataclass(frozen=True, slots=True)
class TeacherTrainingConfig:
    epochs: int = 30
    learning_rate: float = 1.0e-4
    batch_size: int = 256
    num_workers: int = 4
    num_train_timesteps: int = 1_000

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("teacher epochs must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("teacher learning_rate must be finite and positive")
        if self.batch_size <= 0:
            raise ValueError("teacher batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("teacher num_workers must be non-negative")
        if self.num_train_timesteps <= 0:
            raise ValueError("teacher num_train_timesteps must be positive")


@dataclass(frozen=True, slots=True)
class TeacherCacheBuildConfig:
    size: int = 10_000
    noise_size: int = 32
    batch_size: int = 500
    sampling_steps: int = 500
    num_classes: int = 10
    num_train_timesteps: int = 1_000

    def __post_init__(self) -> None:
        for name, value in (
            ("size", self.size),
            ("noise_size", self.noise_size),
            ("batch_size", self.batch_size),
            ("sampling_steps", self.sampling_steps),
            ("num_train_timesteps", self.num_train_timesteps),
        ):
            if value <= 0:
                raise ValueError(f"cache {name} must be positive")
        if self.num_classes < 2:
            raise ValueError("cache num_classes must be at least 2")
        if self.sampling_steps > self.num_train_timesteps:
            raise ValueError("cache sampling_steps cannot exceed num_train_timesteps")


@dataclass(frozen=True, slots=True)
class TeacherEpochResult:
    epoch: int
    loss: float


type TeacherEpochCallback = Callable[[TeacherEpochResult], None]


def build_teacher(config: TeacherModelConfig | None = None) -> UNet2DModel:
    resolved = config or TeacherModelConfig()
    return UNet2DModel(
        sample_size=resolved.size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=resolved.num_classes,
    )


def _state_dict(value: object) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError("teacher state_dict must be a tensor mapping")

    state: dict[str, torch.Tensor] = {}
    for key, tensor in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("teacher state_dict must map strings to tensors")
        state[key] = tensor
    return state


def _checkpoint_config(value: object) -> TeacherModelConfig:
    if not isinstance(value, Mapping):
        raise ValueError("teacher checkpoint config must be a mapping")
    config = cast(Mapping[object, object], value)
    size = config.get("size")
    num_classes = config.get("num_classes")
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("teacher checkpoint size must be an integer")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise ValueError("teacher checkpoint num_classes must be an integer")
    return TeacherModelConfig(size=size, num_classes=num_classes)


def save_teacher(
    model: UNet2DModel,
    config: TeacherModelConfig,
    path: str | Path,
) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(config)},
        path,
    )


def load_teacher(
    path: str | Path,
    device: torch.device,
    *,
    config: TeacherModelConfig | None = None,
) -> UNet2DModel:
    raw = cast(
        object,
        torch.load(path, map_location=device, weights_only=True),
    )
    if not isinstance(raw, Mapping):
        raise ValueError("teacher checkpoint must be a mapping")
    checkpoint = cast(Mapping[object, object], raw)

    if "state_dict" in checkpoint:
        stored_config = checkpoint.get("config")
        resolved_config = (
            config if stored_config is None else _checkpoint_config(stored_config)
        )
        if resolved_config is None:
            resolved_config = TeacherModelConfig()
        if config is not None and resolved_config != config:
            raise ValueError(
                f"teacher checkpoint uses {resolved_config}; expected {config}"
            )
        state = _state_dict(checkpoint["state_dict"])
    else:
        resolved_config = config or TeacherModelConfig()
        state = _state_dict(checkpoint)

    model = build_teacher(resolved_config)
    _ = model.load_state_dict(state)
    _ = model.to(device)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
    return model


def train_teacher(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    train_loader: _BatchLoader,
    device: torch.device,
    config: TeacherTrainingConfig,
    on_epoch: TeacherEpochCallback | None = None,
) -> UNet2DModel:
    if len(train_loader) == 0:
        raise ValueError("train_loader must contain at least one batch")

    _ = model.to(device)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.epochs + 1):
        _ = model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device) * 2.0 - 1.0  # (B, 1, H, W), [-1, 1]
            labels = labels.to(device)

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0,
                config.num_train_timesteps,
                (images.size(0),),
                device=device,
            ).long()
            noisy = scheduler.add_noise(
                images,
                noise,
                cast(torch.IntTensor, timesteps),
            )

            optimizer.zero_grad(set_to_none=True)
            predicted = cast(
                torch.Tensor,
                model(noisy, timesteps, class_labels=labels).sample,
            )
            loss = F.mse_loss(predicted, noise)
            _ = cast(
                object,
                loss.backward(),  # pyright: ignore[reportUnknownMemberType]
            )
            _ = cast(
                object,
                optimizer.step(),  # pyright: ignore[reportUnknownMemberType]
            )
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(f"epoch {epoch:2d}  loss {average_loss:.4f}")
        if on_epoch is not None:
            on_epoch(TeacherEpochResult(epoch=epoch, loss=average_loss))

    return model


@torch.no_grad()
def sample_teacher(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    noise: torch.Tensor,
    labels: torch.Tensor,
    *,
    sampling_steps: int,
) -> torch.Tensor:
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive")
    _ = model.eval()
    scheduler.set_timesteps(sampling_steps, device=noise.device)
    timesteps = scheduler.timesteps

    sample = noise
    for timestep_tensor in timesteps:
        timestep = int(timestep_tensor.item())
        predicted_noise = cast(
            torch.Tensor,
            model(sample, timestep, class_labels=labels).sample,
        )
        result = cast(
            DDPMSchedulerOutput,
            scheduler.step(  # pyright: ignore[reportUnknownMemberType]
                predicted_noise, timestep, sample
            ),
        )
        sample = result.prev_sample

    return ((sample + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def build_teacher_cache(
    teacher: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
    config: TeacherCacheBuildConfig,
) -> TeacherCache:
    _ = teacher.to(  # pyright: ignore[reportUnknownMemberType]
        device  # pyright: ignore[reportArgumentType]
    )
    _ = teacher.eval()
    noises: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    images: list[torch.Tensor] = []

    generated = 0
    while generated < config.size:
        batch_size = min(config.batch_size, config.size - generated)
        batch_labels = torch.randint(
            0, config.num_classes, (batch_size,), device=device
        )
        batch_noise = torch.randn(
            batch_size,
            1,
            config.noise_size,
            config.noise_size,
            device=device,
        )
        batch_images = sample_teacher(
            teacher,
            scheduler,
            batch_noise,
            batch_labels,
            sampling_steps=config.sampling_steps,
        )

        noises.append(batch_noise.cpu())
        labels.append(batch_labels.cpu())
        images.append(batch_images.squeeze(1).cpu())
        generated += batch_size
        print(f"teacher cache {generated}/{config.size}")

    return TeacherCache(
        noises=torch.cat(noises),
        labels=torch.cat(labels),
        images=torch.cat(images),
    )
