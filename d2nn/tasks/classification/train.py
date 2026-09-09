from collections.abc import Iterator
from typing import Protocol, cast

import torch
import torch.nn as nn

from d2nn.data import Batch
from d2nn.models.classifier import DiffractiveClassifier
from d2nn.optics.input import FieldEncoder
from d2nn.training.schedule import Schedule

from .config import TrainingConfig
from .types import (
    ClassificationLoss,
    EpochCallback,
    EpochResult,
    Metrics,
    TrainingParameters,
)


class _BatchLoader(Protocol):
    def __iter__(self) -> Iterator[Batch]: ...

    def __len__(self) -> int: ...


def criterion(
    scores: torch.Tensor,
    targets: torch.Tensor,
    loss: ClassificationLoss,
    scale: int = 20 * 20,
) -> torch.Tensor:
    match loss:
        case ClassificationLoss.MSE:
            one_hot = nn.functional.one_hot(targets, scores.size(1)).to(scores.dtype)
            return nn.functional.mse_loss(scores / scale, one_hot)
        case ClassificationLoss.CROSS_ENTROPY:
            return nn.functional.cross_entropy(scores, targets)


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def evaluate(
    model: DiffractiveClassifier,
    loader: _BatchLoader,
    device: torch.device,
    *,
    encoder: FieldEncoder,
) -> Metrics:
    _ = model.eval()
    correct = total = 0
    correct_eff_sum = total_eff_sum = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.squeeze(1).to(device)  # (B, H, W)
            targets = targets.to(device)
            field = encoder(images)
            scores = cast(torch.Tensor, model(field))  # (B, num_classes)

            correct += int((scores.argmax(dim=1) == targets).sum().item())
            total += targets.size(0)

            input_intensity = field.abs().square().sum(dim=(1, 2))
            correct_intensity = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
            detector_intensity = scores.sum(dim=1)
            correct_eff_sum += (correct_intensity / input_intensity).sum().item()
            total_eff_sum += (detector_intensity / input_intensity).sum().item()

    return Metrics(
        accuracy=correct / total,
        correct_efficiency=correct_eff_sum / total,
        total_efficiency=total_eff_sum / total,
    )


def train(
    model: DiffractiveClassifier,
    train_loader: _BatchLoader,
    test_loader: _BatchLoader,
    device: torch.device,
    config: TrainingConfig,
    schedule: Schedule[TrainingParameters],
    on_epoch: EpochCallback | None = None,
    *,
    encoder: FieldEncoder,
) -> DiffractiveClassifier:
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("train_loader must contain at least one batch")

    _ = model.to(device)
    initial_parameters = schedule.at(0.0)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=initial_parameters.learning_rate
    )
    total_steps = config.epochs * steps_per_epoch
    final_step = max(total_steps - 1, 1)
    global_step = 0
    parameters = initial_parameters

    try:
        for epoch in range(1, config.epochs + 1):
            _ = model.train()
            running_loss = 0.0
            for images, targets in train_loader:
                progress = global_step / final_step
                parameters = (
                    initial_parameters if global_step == 0 else schedule.at(progress)
                )
                _set_learning_rate(optimizer, parameters.learning_rate)
                model.set_conditions(parameters.conditions)

                images = images.squeeze(1).to(device)  # (B, H, W)
                targets = targets.to(device)
                field = encoder(images)

                optimizer.zero_grad(set_to_none=True)
                loss_value = criterion(
                    cast(torch.Tensor, model(field)),
                    targets,
                    config.loss,
                    scale=model.detector.num_points,
                )
                _ = cast(
                    object,
                    loss_value.backward(),  # pyright: ignore[reportUnknownMemberType]
                )
                _ = cast(
                    object,
                    optimizer.step(),  # pyright: ignore[reportUnknownMemberType]
                )
                running_loss += loss_value.item()
                global_step += 1

            model.set_conditions(None)
            metrics = evaluate(model, test_loader, device, encoder=encoder)
            avg_loss = running_loss / steps_per_epoch
            print(
                "  ".join(
                    (
                        f"epoch {epoch:2d}",
                        f"loss {avg_loss:.4f}",
                        f"test acc {metrics.accuracy:.4f}",
                        f"eff(correct) {metrics.correct_efficiency:.4f}",
                        f"eff(total) {metrics.total_efficiency:.4f}",
                    )
                )
            )
            if on_epoch is not None:
                on_epoch(
                    EpochResult(
                        epoch=epoch,
                        loss=avg_loss,
                        metrics=metrics,
                        parameters=parameters,
                    )
                )
    finally:
        model.set_conditions(None)

    return model
