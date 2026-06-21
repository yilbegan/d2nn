import datetime
import pathlib

import modal

from d2nn.tasks.generation.teacher import load_teacher
from d2nn.tasks.generation.train import build_teacher_cache

app = modal.App("d2nn-generation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "matplotlib", "diffusers")
    .add_local_python_source("d2nn")
)

data_volume = modal.Volume.from_name("mnist", create_if_missing=True)
results_volume = modal.Volume.from_name("d2nn-generation", create_if_missing=True)

DATASET_DIR = "/dataset"
TASK_DATA_DIR = "/task_data"


def get_cache_path(cache_size: int, noise_size: int, timesteps: int) -> pathlib.Path:
    name = f"teacher_cache_n{cache_size}_s{noise_size}_t{timesteps}.pt"
    return pathlib.Path(TASK_DATA_DIR) / "cache" / name


@app.function(
    image=image,
    gpu="A100",
    volumes={TASK_DATA_DIR: results_volume},
    timeout=3600,
)
def build_cache(
    cache_size: int = 10000,
    batch_size: int = 500,
    noise_size: int = 32,
    timesteps: int = 500,
    force: bool = False,
) -> None:
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    cache_path = get_cache_path(cache_size, noise_size, timesteps)
    if cache_path.exists() and not force:
        print(f"teacher cache already exists at {cache_path}, skipping")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"building teacher cache on {device}")

    teacher_path = pathlib.Path(TASK_DATA_DIR) / "extra" / "unet_mnist.pt"
    if not teacher_path.exists():
        raise RuntimeError("Please train teacher first")

    teacher = load_teacher(teacher_path, device)
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    noises, labels, images = build_teacher_cache(
        teacher, scheduler, device, cache_size, noise_size, batch_size, timesteps
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"noises": noises, "labels": labels, "images": images}, cache_path)
    results_volume.commit()
    print(f"saved teacher cache ({cache_size} samples) to {cache_path}")


@app.function(
    image=image,
    gpu="A10G",
    volumes={TASK_DATA_DIR: results_volume},
    timeout=3600,
)
def train(
    epochs: int = 10,
    cache_size: int = 10000,
    batch_size: int = 200,
    noise_size: int = 32,
    timesteps: int = 500,
) -> None:
    import torch

    from d2nn.models import (
        DecoderConfig,
        DiffractiveGenerativeModel,
        EncoderConfig,
    )
    from d2nn.viz import plot_phase_masks
    from d2nn.viz.generation import plot_decoder_intensity, plot_generated_digits

    from .train import train

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    cache_path = get_cache_path(cache_size, noise_size, timesteps)
    if not cache_path.exists():
        raise RuntimeError("Please build cache first")
    cache = torch.load(cache_path, map_location="cpu")
    teacher_cache = (cache["noises"], cache["labels"], cache["images"])

    encoder_config = EncoderConfig(in_size=noise_size)
    decoder_config = DecoderConfig()

    model = DiffractiveGenerativeModel(encoder_config, decoder_config)

    train(model, teacher_cache, device, epochs, cache_size, batch_size)

    results_path = pathlib.Path(TASK_DATA_DIR) / "runs" / run_id
    results_path.mkdir(parents=True)

    model.save(results_path / "d2nn_gen_mnist.pt")

    plot_phase_masks(model.decoder.layers, results_path / "phase_masks.png")
    plot_generated_digits(model, device, results_path / "generated_digits.png")

    plot_decoder_intensity(
        model, device, results_path / "decoder_intensity_4.png", label=4
    )
    plot_decoder_intensity(
        model, device, results_path / "decoder_intensity_5.png", label=5
    )

    results_volume.commit()
    print(f"model weights results saved for run {run_id}")


@app.function(
    image=image,
    gpu="A10G",
    volumes={DATASET_DIR: data_volume, TASK_DATA_DIR: results_volume},
    timeout=3600,
)
def train_teacher(
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 256,
    size: int = 32,
    num_train_timesteps: int = 1000,
) -> None:
    import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    from d2nn.data import mnist_loaders

    from .teacher import build_teacher, train_teacher

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_loader, _ = mnist_loaders(root=DATASET_DIR, size=size, batch_size=batch_size)

    model = build_teacher(size=size)
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    train_teacher(model, scheduler, train_loader, device, epochs=epochs, lr=lr)

    weights_dir = pathlib.Path(TASK_DATA_DIR) / "extra"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_dir / "unet_mnist.pt")

    data_volume.commit()
    results_volume.commit()
    print(f"saved teacher weights to {weights_dir / 'unet_mnist.pt'}")
