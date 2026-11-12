"""Quantum kernel matrices: exact by statevector, or estimated from measured circuits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .circuits import feature_map, overlap_circuit
from .mitigation import MAX_QUBITS, correct_readout, counts_to_probs

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Measured:
    """One circuit's result from whatever ran it: counts, the device's per-bit readout errors
    `[P(1|0), P(0|1)]` when it reports them, and an id the caller can cite (dispatch index)."""

    counts: dict[str, int]
    readout: list[list[float]] | None = None
    dispatch: int | None = None


# Runs circuits (same shots for all, in one batch where it can) and returns one `Measured` per
# circuit, in order. The library does not care whether that is a QPU, Aer or a Qiskit primitive.
Runner = Callable[[Sequence[QuantumCircuit]], list[Measured]]


def exact_kernel(A: FloatArray, B: FloatArray, reps: int = 2) -> FloatArray:
    """k(a, b) = |<phi(a)|phi(b)>|^2 by statevector simulation: the noise-free reference."""
    sa = np.array([Statevector(feature_map(a, reps)).data for a in A])
    sb = np.array([Statevector(feature_map(b, reps)).data for b in B])
    return np.asarray(np.abs(sa.conj() @ sb.T) ** 2, dtype=np.float64)


def p_all_zero(m: Measured, n: int, *, correct: bool = True) -> float:
    """P(0...0) from counts, with readout errors undone when the device reported them."""
    shots = max(sum(m.counts.values()), 1)
    if correct and m.readout is not None and len(m.readout) == n and n <= MAX_QUBITS:
        return float(np.clip(correct_readout(counts_to_probs(m.counts, n), m.readout)[0], 0, 1))
    return m.counts.get("0" * n, 0) / shots


@dataclass(frozen=True)
class SampledGram:
    """A measured kernel matrix. `K` has readout correction, `raw` has none; `pairs[k]` is the
    (row, column) entry that `measured[k]` belongs to (one triangle only when symmetric)."""

    K: FloatArray
    raw: FloatArray
    pairs: list[tuple[int, int]]
    measured: list[Measured]


def sampled_kernel(
    run: Runner, A: FloatArray, B: FloatArray, reps: int = 2, *, symmetric: bool = False
) -> SampledGram:
    """Compute-uncompute estimate of every kernel entry. With `symmetric` (B is A) only the upper
    triangle is measured and the diagonal is 1. All circuits go to `run` in one call."""
    pairs = [(i, j) for i in range(len(A)) for j in range(len(B)) if not symmetric or j > i]
    measured = run([overlap_circuit(A[i], B[j], reps) for i, j in pairs])
    n = A.shape[1]
    K, raw = np.zeros((len(A), len(B))), np.zeros((len(A), len(B)))
    for (i, j), m in zip(pairs, measured, strict=True):
        K[i, j], raw[i, j] = p_all_zero(m, n), p_all_zero(m, n, correct=False)
        if symmetric:
            K[j, i], raw[j, i] = K[i, j], raw[i, j]
    if symmetric:
        np.fill_diagonal(K, 1.0)
        np.fill_diagonal(raw, 1.0)
    return SampledGram(K, raw, pairs, measured)
