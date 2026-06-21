import pathlib

import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader

from d2nn.data import Batch


def build_teacher(size: int = 32, num_classes: int = 10) -> UNet2DModel:
    return UNet2DModel(
        sample_size=size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=num_classes,
    )


def load_teacher(
    path: pathlib.Path | str,
    device: torch.device,
    size: int = 32,
    num_classes: int = 10,
) -> UNet2DModel:
    model = build_teacher(size, num_classes).to(device)  # pyright: ignore[reportArgumentType]
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model


def train_teacher(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    train_loader: DataLoader[Batch],
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
) -> UNet2DModel:
    model.to(device)  # pyright: ignore[reportArgumentType]
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_steps = int(scheduler.config.num_train_timesteps)  # pyright: ignore[reportAttributeAccessIssue]

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device) * 2.0 - 1.0  # (B, 1, H, W), [-1, 1]
            labels = labels.to(device)

            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, num_steps, (images.size(0),), device=device
            ).long()
            noisy = scheduler.add_noise(images, noise, timesteps)  # pyright: ignore[reportArgumentType]

            optimizer.zero_grad()
            predicted = model(noisy, timesteps, class_labels=labels).sample
            loss = criterion(predicted, noise)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"epoch {epoch:2d}  loss {avg_loss:.4f}")

    return model


@torch.no_grad()
def sample_teacher(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    noise: torch.Tensor,
    labels: torch.Tensor,
    timesteps: int | None = None,
) -> torch.Tensor:
    model.eval()

    if timesteps is None:
        timesteps = int(scheduler.config.num_train_timesteps)  # pyright: ignore[reportAttributeAccessIssue]

    scheduler.set_timesteps(timesteps)
    for timestep in scheduler.timesteps:
        noise_pred = model(noise, timestep, class_labels=labels).sample
        noise = scheduler.step(noise_pred, timestep, noise).prev_sample  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

    return ((noise + 1.0) / 2.0).clamp(0.0, 1.0)
