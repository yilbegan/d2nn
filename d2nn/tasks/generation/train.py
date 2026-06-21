import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

from d2nn.models.generative import DiffractiveGenerativeModel

from .teacher import sample_teacher


def soft_histogram(
    x: torch.Tensor,
    num_bins: int = 64,
    start: float = -1.0,
    end: float = 1.0,
    sigma: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    x = (2 * x - 1).reshape(-1)
    centers = torch.linspace(start, end, num_bins, device=x.device)
    if sigma is None:
        sigma = (end - start) / num_bins

    d = x.unsqueeze(1) - centers.unsqueeze(0)
    w = torch.exp(-0.5 * (d / sigma) ** 2)
    hist = w.sum(dim=0)
    hist = hist / (hist.sum() + eps)
    return hist


def histogram_kl_loss(
    o_model: torch.Tensor,
    o_teacher: torch.Tensor,
    num_bins: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    p_model = soft_histogram(o_model, num_bins=num_bins).clamp_min(eps)
    p_teacher = soft_histogram(o_teacher, num_bins=num_bins).clamp_min(eps)

    return torch.nn.functional.kl_div(p_model.log(), p_teacher, reduction="sum")


def criterion(
    o_model: torch.Tensor,
    o_teacher: torch.Tensor,
    scale: torch.Tensor,
    gamma: float = 1e-4,
) -> torch.Tensor:
    scale = scale.view(-1, 1, 1)
    mse = torch.nn.functional.mse_loss(scale * o_model, o_teacher)
    kl = histogram_kl_loss(o_model, o_teacher)

    return mse + gamma * kl


@torch.no_grad()
def build_teacher_cache(
    teacher: UNet2DModel,
    scheduler: DDPMScheduler,
    device: torch.device,
    cache_size: int,
    noise_size: int,
    batch_size: int = 200,
    timesteps: int = 500,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher.to(device).eval()  # pyright: ignore[reportArgumentType]

    noises: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    images: list[torch.Tensor] = []

    generated = 0
    while generated < cache_size:
        n = min(batch_size, cache_size - generated)
        batch_labels = torch.randint(0, 10, (n,), device=device)
        batch_noise = torch.randn(n, 1, noise_size, noise_size, device=device)

        o_teacher = sample_teacher(
            teacher, scheduler, batch_noise, batch_labels, timesteps=timesteps
        )

        noises.append(batch_noise.cpu())
        labels.append(batch_labels.cpu())
        images.append(o_teacher.squeeze(1).cpu())

        generated += n
        print(f"teacher cache {generated}/{cache_size}")

    return torch.cat(noises), torch.cat(labels), torch.cat(images)


def train(
    model: DiffractiveGenerativeModel,
    teacher_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int = 10,
    cache_size: int = 10000,
    batch_size: int = 200,
    lr_encoder: float = 1e-4,
    lr_decoder: float = 2e-3,
) -> DiffractiveGenerativeModel:
    model.to(device)
    cache_noise, cache_labels, cache_images = teacher_cache
    loss_size = cache_images.shape[1]

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": lr_encoder},
            {"params": model.decoder.parameters(), "lr": lr_decoder},
        ]
    )

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    steps_per_epoch = cache_size // batch_size

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(cache_size)
        running_loss = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            noise = cache_noise[idx].to(device)
            labels = cache_labels[idx].to(device)
            o_teacher = cache_images[idx].to(device)

            o_model, scale = model(noise, labels)
            o_model = nn.functional.adaptive_avg_pool2d(
                o_model.unsqueeze(1), output_size=loss_size
            ).squeeze(1)

            optimizer.zero_grad()
            loss = criterion(o_model, o_teacher, scale)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        lr_scheduler.step()
        avg_loss = running_loss / steps_per_epoch
        print(f"epoch {epoch:2d}  loss {avg_loss:.4f}")

    return model
