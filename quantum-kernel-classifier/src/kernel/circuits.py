"""Circuits of a quantum kernel: the ZZ feature map and the compute-uncompute overlap test."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

FloatArray = NDArray[np.float64]


def feature_map(x: FloatArray, reps: int = 2) -> QuantumCircuit:
    """ZZ-product feature map, `reps` times [H, P(2 x_i), RZZ(2 x_i x_j)] on one qubit per feature.
    Scale `x` by a bandwidth first: the kernel is only useful in a narrow range of scales."""
    d = len(x)
    qc = QuantumCircuit(d)
    for _ in range(reps):
        qc.h(range(d))
        for i in range(d):
            qc.p(2.0 * float(x[i]), i)
        for i in range(d):
            for j in range(i + 1, d):
                qc.rzz(2.0 * float(x[i]) * float(x[j]), i, j)
    return qc


def overlap_circuit(a: FloatArray, b: FloatArray, reps: int = 2) -> QuantumCircuit:
    """U(a) then U(b)^dagger, measured: P(all zeros) = |<phi(b)|phi(a)>|^2 = k(a, b)."""
    qc = feature_map(a, reps).compose(feature_map(b, reps).inverse())
    qc.measure_all()
    return qc


def mirror_circuit(a: FloatArray, reps: int = 2) -> QuantumCircuit:
    """An overlap circuit with b = a, so the ideal P(all zeros) is exactly 1 and whatever is
    missing is the circuit's own noise. The barrier stops the transpiler cancelling the halves."""
    ua = feature_map(a, reps)
    qc = ua.copy()
    qc.barrier()
    qc.compose(ua.inverse(), inplace=True)
    qc.measure_all()
    return qc
