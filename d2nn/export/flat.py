from typing import cast

import numpy as np
import trimesh


def create_flat_mesh(
    relief: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    pixel_size: float,
    base_thickness: float,
    base_padding: float,
) -> trimesh.Trimesh:
    w, h = relief.shape
    parts: list[trimesh.Trimesh] = []

    base_x = w * pixel_size + 2 * base_padding
    base_y = h * pixel_size + 2 * base_padding

    base = trimesh.creation.box(extents=(base_x, base_y, base_thickness))
    base.apply_translation((0, 0, base_thickness / 2))
    parts.append(base)

    center_x = -w * pixel_size / 2
    center_y = -h * pixel_size / 2

    for i, j in np.argwhere(relief > 1e-6):
        height = cast(float, relief[i, j])
        pixel = trimesh.creation.box(extents=(pixel_size, pixel_size, height))

        pixel.apply_translation(
            (
                center_x + (i + 0.5) * pixel_size,
                center_y + (j + 0.5) * pixel_size,
                base_thickness + height / 2,
            )
        )

        parts.append(pixel)

    return trimesh.util.concatenate(parts)
