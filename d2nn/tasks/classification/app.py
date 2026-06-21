import datetime
import pathlib

import modal

from .types import ClassificationLoss

app = modal.App("d2nn-classification")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "matplotlib", "diffusers", "wandb")
    .add_local_python_source("d2nn")
)

data_volume = modal.Volume.from_name("mnist", create_if_missing=True)
results_volume = modal.Volume.from_name("d2nn-classification", create_if_missing=True)

DATASET_DIR = "/dataset"
TASK_DATA_DIR = "/task_data"


@app.function(
    image=image,
    gpu="A100",
    volumes={DATASET_DIR: data_volume, TASK_DATA_DIR: results_volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=3600,
)
def train(
    epochs: int = 5,
    lr: float = 1e-2,
    batch_size: int = 500,
    size: int = 200,
    num_layers: int = 5,
    loss: str = ClassificationLoss.CROSS_ENTROPY,
) -> float:
    import torch
    import wandb

    from d2nn.data import mnist_loaders
    from d2nn.models import DiffractiveClassifier
    from d2nn.viz import plot_phase_masks
    from d2nn.viz.classification import plot_confusion_matrix, plot_input_output_table

    from .train import evaluate, train

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    run = wandb.init(
        project="d2nn-classification",
        name=run_id,
        config={
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "size": size,
            "num_layers": num_layers,
            "loss": loss,
        },
    )

    train_loader, test_loader = mnist_loaders(
        root=DATASET_DIR, size=size, batch_size=batch_size
    )

    model = DiffractiveClassifier(size=size, num_layers=num_layers, det_size=size // 10)
    train(
        model,
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        lr=lr,
        loss=ClassificationLoss(loss),
        on_epoch=lambda epoch, avg_loss, metrics: run.log(
            {
                "train_loss": avg_loss,
                "test_accuracy": metrics.accuracy,
                "correct_efficiency": metrics.correct_efficiency,
                "total_efficiency": metrics.total_efficiency,
            },
            step=epoch,
        ),
    )
    metrics = evaluate(model, test_loader, device)

    results_path = pathlib.Path(TASK_DATA_DIR) / "runs" / run_id
    results_path.mkdir(parents=True)

    torch.save(model.state_dict(), results_path / "d2nn_mnist.pt")

    plot_phase_masks(model.stack.layers, results_path / "phase_masks.png")
    plot_input_output_table(
        model, test_loader, device, results_path / "input_output_table.png"
    )
    plot_confusion_matrix(
        model, test_loader, device, results_path / "confusion_matrix.png"
    )

    run.summary["final_test_accuracy"] = metrics.accuracy
    run.summary["final_correct_efficiency"] = metrics.correct_efficiency
    run.summary["final_total_efficiency"] = metrics.total_efficiency
    run.log(
        {
            "phase_masks": wandb.Image(str(results_path / "phase_masks.png")),
            "input_output_table": wandb.Image(
                str(results_path / "input_output_table.png")
            ),
            "confusion_matrix": wandb.Image(str(results_path / "confusion_matrix.png")),
        }
    )
    run.finish()

    data_volume.commit()
    results_volume.commit()
    print(
        f"final test accuracy {metrics.accuracy:.4f}"
        f"  eff(correct) {metrics.correct_efficiency:.4f}"
        f"  eff(total) {metrics.total_efficiency:.4f}"
    )
    return metrics.accuracy
