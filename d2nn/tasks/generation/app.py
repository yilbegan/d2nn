from pathlib import Path

import modal

app = modal.App("d2nn-generation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "matplotlib", "diffusers", "pyyaml")
    .add_local_python_source("d2nn")
)

data_volume = modal.Volume.from_name("mnist", create_if_missing=True)
results_volume = modal.Volume.from_name("d2nn-generation", create_if_missing=True)

DATASET_DIR = "/dataset"
TASK_DATA_DIR = "/task_data"


def get_cache_path(cache_size: int, noise_size: int, timesteps: int) -> Path:
    """Return a cache path using the legacy helper interface."""

    from .runner import teacher_cache_path

    return teacher_cache_path(
        TASK_DATA_DIR,
        cache_size=cache_size,
        noise_size=noise_size,
        sampling_steps=timesteps,
    )


@app.function(  # pyright: ignore[reportUnknownMemberType]
    image=image,
    gpu="A10G",
    volumes={TASK_DATA_DIR: results_volume},
    timeout=3600,
)
def _train(config_yaml: str) -> str:
    from .runner import run_generation

    run_id = run_generation(config_yaml, TASK_DATA_DIR)
    results_volume.commit()
    return run_id


@app.local_entrypoint(name="train")
def train(config: str) -> None:
    """Validate a local YAML file and submit it to the GPU training function."""

    from .config import load_config

    config_path = Path(config)
    _ = load_config(config_path)
    config_yaml = config_path.read_text(encoding="utf-8")
    run_id = _train.remote(config_yaml)
    print(f"generation training completed for run {run_id}")


@app.function(  # pyright: ignore[reportUnknownMemberType]
    image=image,
    gpu="A100",
    volumes={TASK_DATA_DIR: results_volume},
    timeout=3600,
)
def build_cache(
    cache_size: int = 10_000,
    batch_size: int = 500,
    noise_size: int = 32,
    timesteps: int = 500,
    num_classes: int = 10,
    num_train_timesteps: int = 1_000,
    force: bool = False,
) -> None:
    from .runner import build_cache as run_cache_build
    from .teacher import TeacherCacheBuildConfig

    config = TeacherCacheBuildConfig(
        size=cache_size,
        noise_size=noise_size,
        batch_size=batch_size,
        sampling_steps=timesteps,
        num_classes=num_classes,
        num_train_timesteps=num_train_timesteps,
    )
    _ = run_cache_build(config, TASK_DATA_DIR, force=force)
    results_volume.commit()


@app.function(  # pyright: ignore[reportUnknownMemberType]
    image=image,
    gpu="A10G",
    volumes={DATASET_DIR: data_volume, TASK_DATA_DIR: results_volume},
    timeout=3600,
)
def train_teacher(
    epochs: int = 30,
    lr: float = 1.0e-4,
    batch_size: int = 256,
    num_workers: int = 4,
    size: int = 32,
    num_classes: int = 10,
    num_train_timesteps: int = 1_000,
) -> None:
    from .runner import run_teacher_training
    from .teacher import (
        TeacherModelConfig,
        TeacherTrainingConfig,
    )

    model_config = TeacherModelConfig(size=size, num_classes=num_classes)
    training_config = TeacherTrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_timesteps=num_train_timesteps,
    )
    _ = run_teacher_training(
        model_config,
        training_config,
        DATASET_DIR,
        TASK_DATA_DIR,
    )
    data_volume.commit()
    results_volume.commit()
