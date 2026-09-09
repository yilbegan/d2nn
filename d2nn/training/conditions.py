from typing import Literal

from d2nn.config import Config
from d2nn.optics.diffractive import DiffractiveLayerConditions
from d2nn.optics.phase import PhaseConditions
from d2nn.optics.propagation import PropagationConditions
from d2nn.optics.stack import DiffractiveStackConditions
from d2nn.training.schedule import Const, Stage, SwitchSpec, ValueSpec, resolve

__all__ = [
    "ConditionsSpec",
    "LayerConditionsSpec",
    "PhaseConditionsSpec",
    "PropagationConditionsSpec",
    "StackConditionsSpec",
]

type QuantizationMode = Literal["soft", "hard"]


class PropagationConditionsSpec(Config, frozen=True):
    distance_std: ValueSpec = Const(0.0)

    def at(self, progress: float, /) -> PropagationConditions:
        return PropagationConditions(distance_std=resolve(self.distance_std, progress))


class PhaseConditionsSpec(Config, frozen=True):
    quantization_steepness: ValueSpec
    quantization_mode: SwitchSpec[QuantizationMode] = Const("soft")

    def at(self, progress: float, /) -> PhaseConditions:
        return PhaseConditions(
            quantization_steepness=resolve(self.quantization_steepness, progress),
            quantization_mode=resolve(self.quantization_mode, progress),
        )


class LayerConditionsSpec(Config, frozen=True):
    fabrication_error_std: ValueSpec = Const(0.0)
    xy_drift_std: ValueSpec = Const(0.0)
    propagation: PropagationConditionsSpec | None = None
    phase: PhaseConditionsSpec | None = None

    def at(self, progress: float, /) -> DiffractiveLayerConditions:
        return DiffractiveLayerConditions(
            fabrication_error_std=resolve(self.fabrication_error_std, progress),
            xy_drift_std=resolve(self.xy_drift_std, progress),
            propagation=resolve(self.propagation, progress),
            phase=resolve(self.phase, progress),
        )


class StackConditionsSpec(Config, frozen=True):
    layers: LayerConditionsSpec | None = None
    output_propagation: PropagationConditionsSpec | None = None

    def at(self, progress: float, /) -> DiffractiveStackConditions:
        return DiffractiveStackConditions(
            layers=resolve(self.layers, progress),
            output_propagation=resolve(self.output_propagation, progress),
        )


type ConditionsSpec = (
    StackConditionsSpec | None | list[Stage[StackConditionsSpec | None]]
)
