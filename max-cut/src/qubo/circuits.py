"""QAOA circuits for a QUBO."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, XXPlusYYGate

from .problem import Ising


def _scs(qc: QuantumCircuit, n: int, k: int) -> None:
    """Split-and-cyclic-shift unitary SCS_{n,k} (Baertschi and Eidenbenz, 2019) on the last k+1
    qubits of the first n; its rotation angles depend on n."""
    last = n - 1
    qc.cx(last - 1, last)
    qc.append(
        RYGate(2 * math.acos(math.sqrt(1.0 / n))).control(1, annotated=False), [last, last - 1]
    )
    qc.cx(last - 1, last)
    for m in range(2, k + 1):
        qc.cx(last - m, last)
        qc.append(
            RYGate(2 * math.acos(math.sqrt(m / n))).control(2, annotated=False),
            [last, last - m + 1, last - m],
        )
        qc.cx(last - m, last)


def dicke_state(n: int, k: int) -> QuantumCircuit:
    """The uniform superposition of all n-bit strings with k ones, in O(nk) gates and no ancillas
    (generic state preparation would be far deeper)."""
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    qc = QuantumCircuit(n)
    for i in range(n - k, n):
        qc.x(i)
    if 0 < k < n:
        for m in range(n, k, -1):
            _scs(qc, m, k)
        for m in range(k, 1, -1):
            _scs(qc, m, m - 1)
    return qc


def xy_ring_mixer(qc: QuantumCircuit, beta: Any) -> None:
    """exp(-i beta (XX + YY) / 2) on every neighbouring pair of a ring. It swaps a 1 and a 0, so the
    number of ones never changes."""
    n = qc.num_qubits
    for i in range(n if n > 2 else 1):
        qc.append(XXPlusYYGate(2.0 * beta), [i, (i + 1) % n])


def _rounds(edges: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Greedy edge colouring: edges that share no qubit form a round, so one round is one layer of
    simultaneous two-qubit gates. Depth falls from about #edges to about the maximum degree."""
    rounds: list[tuple[set[int], list[tuple[int, int]]]] = []
    for i, j in sorted(edges):
        for used, members in rounds:
            if i not in used and j not in used:
                used.update((i, j))
                members.append((i, j))
                break
        else:
            rounds.append(({i, j}, [(i, j)]))
    return [members for _, members in rounds]


def qaoa_circuit(
    ising: Ising,
    gammas: Sequence[Any],
    betas: Sequence[Any],
    *,
    measure: bool = True,
    cardinality: int | None = None,
    init: str = "dicke",
) -> QuantumCircuit:
    """Alternate the cost layer and a mixer, once per (gamma, beta) pair.

    Unconstrained: start in |+...+> and mix with RX. With a `cardinality` the start state and mixer
    keep exactly that many ones, so no sample is invalid and no penalty term is needed. `init`
    picks that start: "dicke" is every valid state at once (about 5nk CNOTs); "product" is one
    evenly spread valid state that the mixer then spreads out (no CNOTs, needs more layers).
    The cost angles are scaled by the largest coefficient so one set of angles fits any problem.
    Measurements go to a register named "c" (variable i on bit i).
    """
    n = ising.n
    scale = max(float(np.abs(ising.h).max(initial=0.0)), *(abs(v) for v in ising.j.values()), 1e-9)
    qc = QuantumCircuit(n, n if measure else 0)
    if cardinality is None:
        qc.h(range(n))
    elif init == "product":
        for i in range(cardinality):
            qc.x(int(i * n / cardinality))
    else:
        qc.compose(dicke_state(n, cardinality), inplace=True)
    for gamma, beta in zip(gammas, betas, strict=True):
        for i in range(n):
            if ising.h[i] != 0:
                qc.rz(2.0 * gamma * ising.h[i] / scale, i)
        for layer in _rounds(list(ising.j)):
            for i, j in layer:
                qc.rzz(2.0 * gamma * ising.j[(i, j)] / scale, i, j)
        if cardinality is None:
            qc.rx(2.0 * beta, range(n))
        else:
            xy_ring_mixer(qc, beta)
    if measure:
        qc.measure(range(n), range(n))
    return qc
