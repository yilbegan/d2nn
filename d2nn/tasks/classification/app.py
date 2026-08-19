import pathlib
import shutil
from typing import cast

import modal

app = modal.App("d2nn-classification")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "matplotlib", "wandb", "trimesh", "pyyaml")
    .add_local_python_source("d2nn")
)

data_volume = modal.Volume.from_name("mnist", create_if_missing=True)
results_volume = modal.Volume.from_name("d2nn-classification", create_if_missing=True)

DATASET_DIR = "/dataset"
TASK_DATA_DIR = "/task_data"


@app.function(  # pyright: ignore[reportUnknownMemberType]
    image=image,
    gpu="A100",
    volumes={DATASET_DIR: data_volume, TASK_DATA_DIR: results_volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=3600,
)
def _train(config_yaml: str) -> float:
    from .runner import run_classification

    accuracy = run_classification(config_yaml, DATASET_DIR, TASK_DATA_DIR)
    data_volume.commit()
    results_volume.commit()
    return accuracy


@app.local_entrypoint(name="train")
def train(config: str) -> None:
    """Validate a local YAML file and submit it to the GPU training function."""

    from .config import load_config

    config_path = pathlib.Path(config)
    _ = load_config(config_path)
    config_yaml = config_path.read_text(encoding="utf-8")
    accuracy = _train.remote(config_yaml)
    print(f"training completed with accuracy {accuracy:.4f}")


@app.function(  # pyright: ignore[reportUnknownMemberType]
    image=image,
    gpu="A10G",
    volumes={TASK_DATA_DIR: results_volume},
    timeout=3600,
)
def export(
    run_id: str,
    base_thickness: float = 0.5e-3,
    base_padding: float = 0.01,
    delta_n: float = 0.7227,
    force: bool = True,
) -> None:
    from d2nn.export.flat import create_flat_mesh
    from d2nn.export.relief import get_relief
    from d2nn.models.classifier import DiffractiveClassifier
    from d2nn.optics.diffractive import DiffractiveLayer

    results_path = pathlib.Path(TASK_DATA_DIR) / "runs" / run_id
    if not results_path.exists():
        raise ValueError(f"Run {run_id} not found")

    model = DiffractiveClassifier.load(results_path / "d2nn_mnist.pt")
    _ = model.eval()

    layers_path = results_path / "layers"
    if layers_path.exists():
        if not force:
            raise ValueError("3D-models for this run already generated")
        shutil.rmtree(layers_path)

    layers_path.mkdir()
    for i, layer in enumerate(model.stack.layers):
        layer = cast(DiffractiveLayer, layer)
        relief = get_relief(layer, delta_n=delta_n)

        mesh = create_flat_mesh(
            relief=relief,
            pixel_size=model.config.pixel_size,
            base_thickness=base_thickness,
            base_padding=base_padding,
        )

        _ = cast(
            object,
            mesh.export(  # pyright: ignore[reportUnknownMemberType]
                layers_path / f"layer_{i:02}.stl"
            ),
        )

    results_volume.commit()
    print(f"created 3d models for {len(model.stack.layers)} layer(s)")
