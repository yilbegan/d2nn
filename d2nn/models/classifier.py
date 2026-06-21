import pathlib
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from d2nn.optics import Detector, DiffractiveStack


@dataclass(frozen=True)
class ClassifierConfig:
    size: int = 200
    num_layers: int = 5
    num_classes: int = 10
    wavelength: float = 0.75e-3
    pixel_size: float = 0.4e-3
    distance: float = 0.03
    det_size: int = 20
    scale_factor: float = 6.0


class DiffractiveClassifier(nn.Module):
    def __init__(self, config: ClassifierConfig | None = None) -> None:
        super().__init__()
        self.config = config or ClassifierConfig()
        c = self.config

        self.stack = DiffractiveStack(
            c.size,
            c.pixel_size,
            c.wavelength,
            c.distance,
            c.num_layers,
            c.scale_factor,
        )
        self.detector = Detector(
            c.size,
            c.num_classes,
            c.det_size,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        field = image.to(torch.complex64)
        field = self.stack(field)
        return self.detector(field)

    def save(self, path: str | pathlib.Path) -> None:
        checkpoint = {"state_dict": self.state_dict(), "config": asdict(self.config)}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | pathlib.Path, device: torch.device = torch.device("cpu")):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(ClassifierConfig(**checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"])
        return model
