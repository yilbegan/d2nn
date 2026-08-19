from datetime import UTC, datetime
from pathlib import Path

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from d2nn.data import mnist_loaders
from d2nn.models.generative import DiffractiveGenerativeModel
from d2nn.viz import plot_phase_masks
from d2nn.viz.generation import plot_decoder_intensity, plot_generated_digits

from .config import parse_config
from .teacher import (
    TeacherCacheBuildConfig,
    TeacherModelConfig,
    TeacherTrainingConfig,
    build_teacher,
    build_teacher_cache,
    load_teacher,
    save_teacher,
    train_teacher,
)
from .train import train
from .types import TeacherCache

__all__ = [
    "build_cache",
    "run_generation",
    "run_teacher_training",
    "teacher_cache_path",
]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def teacher_cache_path(
    results_dir: str | Path,
    *,
    cache_size: int,
    noise_size: int,
    sampling_steps: int,
) -> Path:
    filename = f"teacher_cache_n{cache_size}_s{noise_size}_t{sampling_steps}.pt"
    return Path(results_dir) / "cache" / filename


def run_generation(config_source: str, results_dir: str | Path) -> str:
    config = parse_config(config_source)
    run_id = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
    device = _device()
    print(f"training on {device}")

    cache_path = teacher_cache_path(
        results_dir,
        cache_size=config.cache.size,
        noise_size=config.encoder.in_size,
        sampling_steps=config.cache.sampling_steps,
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"teacher cache not found at {cache_path}; build it before training"
        )

    cache = TeacherCache.load(cache_path)
    cache.validate_for(
        size=config.cache.size,
        noise_size=config.encoder.in_size,
        num_classes=config.encoder.num_classes,
    )
    model = DiffractiveGenerativeModel(config.encoder, config.decoder)

    result_path = Path(results_dir) / "runs" / run_id
    result_path.mkdir(parents=True)
    _ = (result_path / "config.yaml").write_text(config_source, encoding="utf-8")

    _ = train(
        model,
        cache,
        device,
        config.training,
        config.schedule,
    )
    model.save(result_path / "d2nn_gen_mnist.pt")
    plot_phase_masks(model.decoder.stack.layers, result_path / "phase_masks.png")
    plot_generated_digits(
        model,
        device,
        result_path / "generated_digits.png",
        num_classes=config.encoder.num_classes,
        output_size=cache.images.shape[-1],
    )
    for label in range(min(2, config.encoder.num_classes)):
        plot_decoder_intensity(
            model,
            device,
            result_path / f"decoder_intensity_{label}.png",
            label=label,
        )

    print(f"model weights and artifacts saved for run {run_id}")
    return run_id


def build_cache(
    config: TeacherCacheBuildConfig,
    results_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    cache_path = teacher_cache_path(
        results_dir,
        cache_size=config.size,
        noise_size=config.noise_size,
        sampling_steps=config.sampling_steps,
    )
    if cache_path.exists() and not force:
        print(f"teacher cache already exists at {cache_path}, skipping")
        return cache_path

    teacher_path = Path(results_dir) / "extra" / "unet_mnist.pt"
    if not teacher_path.exists():
        raise FileNotFoundError(
            f"teacher checkpoint not found at {teacher_path}; train it first"
        )

    device = _device()
    print(f"building teacher cache on {device}")
    teacher = load_teacher(
        teacher_path,
        device,
        config=TeacherModelConfig(
            size=config.noise_size,
            num_classes=config.num_classes,
        ),
    )
    scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
    cache = build_teacher_cache(teacher, scheduler, device, config)
    cache.save(cache_path)
    print(f"saved teacher cache ({cache.size} samples) to {cache_path}")
    return cache_path


def run_teacher_training(
    model_config: TeacherModelConfig,
    training_config: TeacherTrainingConfig,
    dataset_dir: str | Path,
    results_dir: str | Path,
) -> Path:
    device = _device()
    print(f"training teacher on {device}")
    train_loader, _ = mnist_loaders(
        root=str(dataset_dir),
        size=model_config.size,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
    )
    model = build_teacher(model_config)
    scheduler = DDPMScheduler(num_train_timesteps=training_config.num_train_timesteps)
    _ = train_teacher(
        model,
        scheduler,
        train_loader,
        device,
        training_config,
    )

    weights_path = Path(results_dir) / "extra" / "unet_mnist.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_teacher(model, model_config, weights_path)
    print(f"saved teacher checkpoint to {weights_path}")
    return weights_path
