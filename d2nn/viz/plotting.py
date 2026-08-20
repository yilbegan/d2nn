# pyright: reportUnknownMemberType=false

import math
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

type OutputPath = str | PathLike[str]
type FloatArray = NDArray[np.float32]
type FloatVector = np.ndarray[tuple[int], np.dtype[np.float32]]
type FloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type FloatImages = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type IntMatrix = np.ndarray[tuple[int, int], np.dtype[np.int64]]


def _float_array(tensor: torch.Tensor) -> FloatArray:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    return cast(FloatArray, array)


def float_matrix(tensor: torch.Tensor) -> FloatMatrix:
    if tensor.ndim != 2:
        raise ValueError("tensor must have rank 2")
    return cast(FloatMatrix, _float_array(tensor))


def float_images(tensor: torch.Tensor) -> FloatImages:
    if tensor.ndim != 3:
        raise ValueError("tensor must have rank 3")
    return cast(FloatImages, _float_array(tensor))


def int_matrix(tensor: torch.Tensor) -> IntMatrix:
    if tensor.ndim != 2:
        raise ValueError("tensor must have rank 2")
    array = tensor.detach().to(device="cpu", dtype=torch.int64).numpy()
    return cast(IntMatrix, array)


def image_at(images: FloatImages, index: int) -> FloatMatrix:
    return cast(FloatMatrix, images[index])


def row_at(matrix: FloatMatrix, index: int) -> FloatVector:
    return cast(FloatVector, matrix[index])


def float_at(matrix: FloatMatrix, row: int, column: int) -> float:
    return float(cast(np.float32, matrix[row, column]))


def int_at(matrix: IntMatrix, row: int, column: int) -> int:
    return int(cast(np.int64, matrix[row, column]))


@dataclass(frozen=True, slots=True)
class FigureGrid:
    figure: Figure
    axes: tuple[Axes, ...]
    columns: int

    @classmethod
    def create(
        cls,
        item_count: int,
        *,
        columns: int | None = None,
        cell_size: tuple[float, float] = (4.0, 4.0),
        sharey: bool = False,
    ) -> "FigureGrid":
        if item_count < 1:
            raise ValueError("item_count must be positive")
        if columns is not None and columns < 1:
            raise ValueError("columns must be positive")

        column_count = min(columns or math.ceil(math.sqrt(item_count)), item_count)
        row_count = math.ceil(item_count / column_count)
        figure, axes_array = plt.subplots(
            row_count,
            column_count,
            figsize=(cell_size[0] * column_count, cell_size[1] * row_count),
            squeeze=False,
            sharey=sharey,
        )
        all_axes = tuple(cast(Iterable[Axes], axes_array.flat))
        for axis in all_axes[item_count:]:
            _ = axis.axis("off")

        return cls(figure, all_axes[:item_count], column_count)

    def save(self, path: OutputPath, *, dpi: int = 150) -> None:
        try:
            self.figure.tight_layout()
            self.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            plt.close(self.figure)
