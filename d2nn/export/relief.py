import numpy as np
import torch
from numpy.typing import NDArray
from typing import cast

from d2nn.optics.diffractive import DiffractiveLayer


def get_relief(layer: DiffractiveLayer, delta_n: float) -> NDArray[np.float32]:
    phase = torch.remainder(cast(torch.Tensor, layer.phase()), 2 * torch.pi)
    df = phase / (2 * torch.pi)
    wavelength = layer.physical_parameters.wavelength
    return (wavelength * df / delta_n).detach().cpu().numpy()
