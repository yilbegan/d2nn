from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from d2nn.data import Batch
from d2nn.models.classifier import DiffractiveClassifier

from .types import ClassificationLoss, Metrics


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


def evaluate(
    model: nn.Module, loader: DataLoader[Batch], device: torch.device
) -> Metrics:
    model.eval()
    correct = total = 0
    correct_eff_sum = total_eff_sum = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.squeeze(1).to(device)  # (B, H, W)
            targets = targets.to(device)
            scores = model(images)  # (B, num_classes)

            correct += int((scores.argmax(dim=1) == targets).sum().item())
            total += targets.size(0)

            input_intensity = images.abs().pow(2).sum(dim=(1, 2))
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
    train_loader: DataLoader[Batch],
    test_loader: DataLoader[Batch],
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-2,
    loss: ClassificationLoss = ClassificationLoss.CROSS_ENTROPY,
    on_epoch: Callable[[int, float, Metrics], None] | None = None,
) -> DiffractiveClassifier:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.squeeze(1).to(device)  # (B, H, W)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss_value = criterion(
                model(images), targets, loss, scale=model.detector.num_points
            )
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()

        metrics = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)
        print(
            f"epoch {epoch:2d}  loss {avg_loss:.4f}  test acc {metrics.accuracy:.4f}"
            f"  eff(correct) {metrics.correct_efficiency:.4f}"
            f"  eff(total) {metrics.total_efficiency:.4f}"
        )
        if on_epoch is not None:
            on_epoch(epoch, avg_loss, metrics)

    return model
