import torch
import torch.nn as nn

from d2nn.optics import Detector, DiffractiveStack


class DiffractiveClassifier(nn.Module):
    def __init__(
        self,
        size: int = 200,
        num_layers: int = 5,
        num_classes: int = 10,
        wavelength: float = 0.75e-3,
        pixel_size: float = 0.4e-3,
        distance: float = 0.03,
        det_size: int = 20,
        scale_factor: float = 6.0,
    ) -> None:
        super().__init__()
        self.size = size
        self.stack = DiffractiveStack(
            size, pixel_size, wavelength, distance, num_layers, scale_factor
        )
        self.detector = Detector(size, num_classes, det_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        field = image.to(torch.complex64)
        field = self.stack(field)
        return self.detector(field)
