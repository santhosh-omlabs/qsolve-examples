"""Quantum kernels for classification: ZZ feature map, exact and sampled kernel matrices, readout
correction, depolarizing calibration from mirror circuits, and the statistics to judge a classifier.
"""

from .circuits import feature_map, mirror_circuit, overlap_circuit
from .kernel import Measured, Runner, SampledGram, exact_kernel, p_all_zero, sampled_kernel
from .mitigation import (
    correct_readout,
    counts_to_probs,
    depolarizing_strength,
    rescale_overlap,
    undo_depolarizing,
)
from .stats import mcnemar_exact, wilson_interval

__all__ = [
    "Measured",
    "Runner",
    "SampledGram",
    "correct_readout",
    "counts_to_probs",
    "depolarizing_strength",
    "exact_kernel",
    "feature_map",
    "mcnemar_exact",
    "mirror_circuit",
    "overlap_circuit",
    "p_all_zero",
    "rescale_overlap",
    "sampled_kernel",
    "undo_depolarizing",
    "wilson_interval",
]
