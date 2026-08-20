import matplotlib

matplotlib.use("Agg")

from .classification import (
    plot_confusion_matrix,
    plot_energy_distribution,
    plot_input_output_table,
)
from .generation import plot_decoder_intensity, plot_generated_digits
from .phase import plot_phase_masks

__all__ = [
    "plot_confusion_matrix",
    "plot_decoder_intensity",
    "plot_energy_distribution",
    "plot_generated_digits",
    "plot_input_output_table",
    "plot_phase_masks",
]
