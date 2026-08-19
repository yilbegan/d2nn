from math import ceil, isfinite
from typing import cast

import torch
import torch.nn.functional as F

from d2nn.models.generative import DiffractiveGenerativeModel
from d2nn.training.schedule import Schedule

from .config import TrainingConfig
from .types import EpochCallback, EpochResult, TeacherCache, TrainingParameters

__all__ = [
    "criterion",
    "histogram_kl_loss",
    "soft_histogram",
    "train",
]


def soft_histogram(
    values: torch.Tensor,
    num_bins: int = 64,
    start: float = -1.0,
    end: float = 1.0,
    sigma: float | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    if not isfinite(start) or not isfinite(end) or start >= end:
        raise ValueError("histogram bounds must be finite and increasing")
    if not isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if sigma is None:
        sigma = (end - start) / num_bins
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

    normalized = (2 * values - 1).reshape(-1)
    centers = torch.linspace(
        start,
        end,
        num_bins,
        device=normalized.device,
        dtype=normalized.dtype,
    )
    distances = normalized.unsqueeze(1) - centers.unsqueeze(0)
    weights = torch.exp(-0.5 * (distances / sigma).square())
    histogram = weights.sum(dim=0)
    return histogram / (histogram.sum() + eps)


def histogram_kl_loss(
    model_output: torch.Tensor,
    teacher_output: torch.Tensor,
    num_bins: int = 64,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    model_histogram = soft_histogram(
        model_output, num_bins=num_bins, eps=eps
    ).clamp_min(eps)
    teacher_histogram = soft_histogram(
        teacher_output, num_bins=num_bins, eps=eps
    ).clamp_min(eps)
    return F.kl_div(model_histogram.log(), teacher_histogram, reduction="sum")


def criterion(
    model_output: torch.Tensor,
    teacher_output: torch.Tensor,
    scale: torch.Tensor,
    gamma: float = 1.0e-4,
    *,
    num_bins: int = 64,
) -> torch.Tensor:
    if not isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    if model_output.shape != teacher_output.shape:
        raise ValueError("model and teacher outputs must have the same shape")
    if scale.ndim != 1 or scale.shape[0] != model_output.shape[0]:
        raise ValueError("scale must have shape (batch_size,)")

    scaled = scale.view(-1, 1, 1) * model_output
    mse = F.mse_loss(scaled, teacher_output)
    kl = histogram_kl_loss(model_output, teacher_output, num_bins=num_bins)
    return mse + gamma * kl


def _set_learning_rates(
    optimizer: torch.optim.Optimizer,
    parameters: TrainingParameters,
) -> None:
    encoder_group, decoder_group = optimizer.param_groups
    encoder_group["lr"] = parameters.encoder_learning_rate
    decoder_group["lr"] = parameters.decoder_learning_rate


def train(
    model: DiffractiveGenerativeModel,
    cache: TeacherCache,
    device: torch.device,
    config: TrainingConfig,
    schedule: Schedule[TrainingParameters],
    on_epoch: EpochCallback | None = None,
) -> DiffractiveGenerativeModel:
    cache.validate_for(
        noise_size=model.encoder.config.in_size,
        num_classes=model.encoder.config.num_classes,
    )
    _ = model.to(device)
    initial_parameters = schedule.at(0.0)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": initial_parameters.encoder_learning_rate,
            },
            {
                "params": model.decoder.parameters(),
                "lr": initial_parameters.decoder_learning_rate,
            },
        ]
    )

    steps_per_epoch = ceil(cache.size / config.batch_size)
    total_steps = config.epochs * steps_per_epoch
    final_step = max(total_steps - 1, 1)
    global_step = 0
    parameters = initial_parameters
    output_size = (cache.images.shape[-2], cache.images.shape[-1])

    try:
        for epoch in range(1, config.epochs + 1):
            _ = model.train()
            permutation = torch.randperm(cache.size)
            running_loss = 0.0

            for start in range(0, cache.size, config.batch_size):
                progress = global_step / final_step
                parameters = (
                    initial_parameters if global_step == 0 else schedule.at(progress)
                )
                _set_learning_rates(optimizer, parameters)
                model.set_conditions(parameters.conditions)

                indices = permutation[start : start + config.batch_size]
                noise = cache.noises[indices].to(device)
                labels = cache.labels[indices].to(device)
                teacher_output = cache.images[indices].to(device)

                model_output, scale = cast(
                    tuple[torch.Tensor, torch.Tensor], model(noise, labels)
                )
                model_output = F.adaptive_avg_pool2d(
                    model_output.unsqueeze(1), output_size=output_size
                ).squeeze(1)

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(
                    model_output,
                    teacher_output,
                    scale,
                    gamma=config.histogram_weight,
                    num_bins=config.histogram_bins,
                )
                _ = cast(
                    object,
                    loss.backward(),  # pyright: ignore[reportUnknownMemberType]
                )
                _ = cast(
                    object,
                    optimizer.step(),  # pyright: ignore[reportUnknownMemberType]
                )
                running_loss += loss.item()
                global_step += 1

            average_loss = running_loss / steps_per_epoch
            print(f"epoch {epoch:2d}  loss {average_loss:.4f}")
            if on_epoch is not None:
                on_epoch(
                    EpochResult(
                        epoch=epoch,
                        loss=average_loss,
                        parameters=parameters,
                    )
                )
    finally:
        model.set_conditions(None)

    return model
