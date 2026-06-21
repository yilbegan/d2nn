import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

Batch = tuple[torch.Tensor, torch.Tensor]


def mnist_loaders(
    root: str = "./data",
    size: int = 200,
    batch_size: int = 500,
    num_workers: int = 4,
    download: bool = True,
) -> tuple[DataLoader[Batch], DataLoader[Batch]]:
    transform = transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor()]
    )
    train = datasets.MNIST(root, train=True, download=download, transform=transform)
    test = datasets.MNIST(root, train=False, download=download, transform=transform)

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
