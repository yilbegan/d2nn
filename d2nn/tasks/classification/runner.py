from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
import wandb
import yaml

from d2nn.data import mnist_loaders
from d2nn.models.classifier import DiffractiveClassifier
from d2nn.viz import plot_confusion_matrix, plot_input_output_table, plot_phase_masks

from .config import parse_config
from .train import evaluate, train
from .types import EpochResult

__all__ = ["run_classification"]


def run_classification(
    config_source: str,
    dataset_dir: str | Path,
    results_dir: str | Path,
) -> float:
    config = parse_config(config_source)
    run_id = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    document = cast(dict[str, object], yaml.safe_load(config_source))
    run = wandb.init(project="d2nn-classification", name=run_id, config=document)

    def log_epoch(result: EpochResult) -> None:
        run.log(
            {
                "train_loss": result.loss,
                "test_accuracy": result.metrics.accuracy,
                "correct_efficiency": result.metrics.correct_efficiency,
                "total_efficiency": result.metrics.total_efficiency,
                "learning_rate": result.parameters.learning_rate,
            },
            step=result.epoch,
        )

    try:
        result_path = Path(results_dir) / "runs" / run_id
        result_path.mkdir(parents=True)
        _ = (result_path / "config.yaml").write_text(config_source, encoding="utf-8")
        train_loader, test_loader = mnist_loaders(
            root=str(dataset_dir),
            layer_size=config.model.size,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
        )
        model = DiffractiveClassifier(config.model)
        _ = train(
            model,
            train_loader,
            test_loader,
            device,
            config.training,
            config.schedule,
            on_epoch=log_epoch,
        )
        metrics = evaluate(model, test_loader, device)

        model.save(result_path / "d2nn_mnist.pt")
        plot_phase_masks(model.stack.layers, result_path / "phase_masks.png")
        plot_input_output_table(
            model, test_loader, device, result_path / "input_output_table.png"
        )
        plot_confusion_matrix(
            model, test_loader, device, result_path / "confusion_matrix.png"
        )

        run.summary["final_test_accuracy"] = metrics.accuracy
        run.summary["final_correct_efficiency"] = metrics.correct_efficiency
        run.summary["final_total_efficiency"] = metrics.total_efficiency
        run.summary["model"] = asdict(config.model)
        run.log(
            {
                "phase_masks": wandb.Image(str(result_path / "phase_masks.png")),
                "input_output_table": wandb.Image(
                    str(result_path / "input_output_table.png")
                ),
                "confusion_matrix": wandb.Image(
                    str(result_path / "confusion_matrix.png")
                ),
            }
        )
        print(
            "  ".join(
                (
                    f"final test accuracy {metrics.accuracy:.4f}",
                    f"eff(correct) {metrics.correct_efficiency:.4f}",
                    f"eff(total) {metrics.total_efficiency:.4f}",
                )
            )
        )
        return metrics.accuracy
    finally:
        run.finish()
