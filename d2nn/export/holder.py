from typing import cast

import trimesh

_Bounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def create_holder_mesh(
    num_layers: int,
    layer_size: float,
    layer_thickness: float,
    layer_distance: float,
    slot_depth: float = 5e-3,
    rib_width: float = 4e-3,
    wall_thickness: float = 3e-3,
    floor_thickness: float = 3e-3,
    clearance: float = 2e-4,
    rib_height: float = 1 / 3,
) -> trimesh.Trimesh:
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")

    for name, value in (
        ("layer_size", layer_size),
        ("layer_thickness", layer_thickness),
        ("layer_distance", layer_distance),
        ("slot_depth", slot_depth),
        ("rib_width", rib_width),
        ("wall_thickness", wall_thickness),
        ("floor_thickness", floor_thickness),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if clearance < 0:
        raise ValueError("clearance must be non-negative")
    if not 0 < rib_height <= 1:
        raise ValueError("rib_height must lie in (0, 1]")

    slot_width = layer_thickness + clearance
    if layer_distance - slot_width <= 0:
        raise ValueError(
            "layer_distance must exceed the slot width layer_thickness + clearance"
        )
    if slot_depth <= clearance / 2:
        raise ValueError("slot_depth must exceed half of clearance")
    if 2 * slot_depth >= layer_size:
        raise ValueError("slot_depth must be smaller than half of layer_size")

    num_slots = num_layers + 2
    half_span = (layer_size + clearance) / 2
    grip = layer_size / 2 - slot_depth
    outer = half_span + wall_thickness
    floor_bottom = -half_span - floor_thickness
    rib_top = -half_span + rib_height * 2 * half_span
    if rib_top <= -grip:
        raise ValueError("rib_height must leave side posts taller than slot_depth")

    slot_centers = [i * layer_distance for i in range(num_slots)]
    first = slot_centers[0] - slot_width / 2
    last = slot_centers[-1] + slot_width / 2

    ribs: list[tuple[float, float]] = [(first - rib_width, first)]
    for before, after in zip(slot_centers, slot_centers[1:]):
        start = before + slot_width / 2
        end = after - slot_width / 2
        if end - start <= 2 * rib_width:
            ribs.append((start, end))
        else:
            ribs.append((start, start + rib_width))
            ribs.append((end - rib_width, end))
    ribs.append((last, last + rib_width))

    span = (ribs[0][0], ribs[-1][1])
    parts = [
        _box(((-outer, outer), (floor_bottom, -half_span), span)),
        _box(((-outer, -half_span), (-half_span, rib_top), span)),
        _box(((half_span, outer), (-half_span, rib_top), span)),
    ]
    for start, end in ribs:
        parts.append(_box(((-half_span, -grip), (-half_span, rib_top), (start, end))))
        parts.append(_box(((grip, half_span), (-half_span, rib_top), (start, end))))
        parts.append(_box(((-half_span, half_span), (-half_span, -grip), (start, end))))

    return cast(
        trimesh.Trimesh,
        trimesh.util.concatenate(parts),  # pyright: ignore[reportUnknownMemberType]
    )


def _box(bounds: _Bounds) -> trimesh.Trimesh:
    (x0, x1), (y0, y1), (z0, z1) = bounds

    box = trimesh.creation.box(  # pyright: ignore[reportUnknownMemberType]
        extents=(x1 - x0, y1 - y0, z1 - z0)
    )
    box.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return box
