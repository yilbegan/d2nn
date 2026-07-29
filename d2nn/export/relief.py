import numpy as np
import torch
from numpy.typing import NDArray

from d2nn.optics.diffractive import DiffractiveLayer


def get_relief(layer: DiffractiveLayer, delta_n: float) -> NDArray[np.float64]:
    phase = torch.remainder(layer.effective_phase(), 2 * torch.pi)
    df = phase / (2 * torch.pi)
    return (layer.wavelength * df / delta_n).detach().cpu().numpy()
