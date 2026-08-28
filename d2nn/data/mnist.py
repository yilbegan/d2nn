from typing import cast

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # pyright: ignore[reportMissingTypeStubs]

Batch = tuple[torch.Tensor, torch.Tensor]


def _transform(
    layer_size: int, mask_size: float = 0.4, binarize: bool = True
) -> transforms.Compose:
    digit_size = round(layer_size * mask_size)
    padding = layer_size - digit_size

    def _binarize(x: torch.Tensor) -> torch.Tensor:
        if not binarize:
            return x.to(torch.float64)
        return (x > 0.5).to(torch.float64)

    return transforms.Compose(
        [
            transforms.Resize((digit_size, digit_size)),
            transforms.ToTensor(),
            transforms.Lambda(_binarize),
            transforms.Pad(
                [
                    padding // 2,  # left
                    padding // 2,  # top
                    padding - padding // 2,  # right
                    padding - padding // 2,  # bottom
                ]
            ),
        ]
    )


def mnist_loaders(
    root: str = "./data",
    layer_size: int = 200,
    mask_size: float = 0.4,
    binarize: bool = True,
    batch_size: int = 500,
    num_workers: int = 4,
    download: bool = True,
) -> tuple[DataLoader[Batch], DataLoader[Batch]]:
    transform = _transform(layer_size, mask_size, binarize)
    train = datasets.MNIST(root, train=True, download=download, transform=transform)
    test = datasets.MNIST(root, train=False, download=download, transform=transform)

    train_loader = cast(
        DataLoader[Batch],
        DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
    )
    test_loader = cast(
        DataLoader[Batch],
        DataLoader(
            test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )
    return train_loader, test_loader
