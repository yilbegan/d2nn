import numpy as np
import torch
from numpy.typing import NDArray

from d2nn.optics.diffractive import DiffractiveLayer


def get_relief(layer: DiffractiveLayer, delta_n: float) -> NDArray[np.float64]:
    phase = layer.scale_factor * layer.phase
    df = torch.fmod(phase, 2 * torch.pi) / (2 * torch.pi)
    return (layer.wavelength * df / delta_n).cpu().numpy()
