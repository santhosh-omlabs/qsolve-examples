"""Energy measurement: qubit-wise commuting groups, optional error mitigation, error bars.

The Hamiltonian is split into groups of Pauli terms that can be read from one measurement basis,
so each group is one circuit: the state, basis-change gates, and a measurement of every qubit. All
circuits of one energy evaluation are handed to the runner together, so on a quantum device they
are one job (one queue wait) rather than one per group.

The library does not know how circuits are run. A `Runner` is any function that takes circuits and
a shot count and returns their counts; `sampler_runner` adapts a Qiskit `SamplerV2`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .mitigation import (
    DENSE_MAX_QUBITS,
    SpinSectors,
    allocate_copies,
    bit_table,
    bootstrap_stderr,
    group_spreads,
    group_weights,
    mitigated_group,
)

SHOT_GRAIN = 4  # copies per group on average: finer slices follow the optimal split closely


@dataclass(frozen=True)
class Measurement:
    """What came back for one circuit."""

    counts: dict[str, int]  # bit strings, qubit 0 on the right
    readout: list[list[float]] | None = None  # per measured bit: [P(1|0), P(0|1)], when known
    dispatch: int | None = None  # the runner's index for this circuit, to cite in a report
    two_qubit_gates: int | None = None  # as run on the device, when reported


Runner = Callable[[Sequence[QuantumCircuit], int], list[Measurement]]


def sampler_runner(sampler: Any) -> Runner:
    """A `Runner` over a Qiskit `SamplerV2`. Every circuit goes out in one `run` call, so a device
    that batches (Qsolve's does) executes them as one job. Device details (readout errors, gate
    counts, the dispatch index) are taken from `metadata["qsolve"]` when the sampler provides it;
    a plain sampler just gives counts."""

    def run(circuits: Sequence[QuantumCircuit], shots: int) -> list[Measurement]:
        result = sampler.run([(qc, None, shots) for qc in circuits]).result()
        out = []
        for pub in result:
            info = (pub.metadata.get("qsolve") or [{}])[0]
            out.append(
                Measurement(
                    dict(pub.data.c.get_counts()),
                    info.get("readout"),
                    info.get("dispatch"),
                    info.get("two_qubit_gates"),
                )  # fmt: skip
            )
        return out

    return run


# ---- grouping ----------------------------------------------------------------------------------


def group_observable(op: SparsePauliOp) -> tuple[float, list[SparsePauliOp]]:
    """Split off the identity (constant) part and group the rest by qubit-wise commutation."""
    op = op.simplify()
    ident = "I" * op.num_qubits
    const = 0.0
    rest = []
    for label, c in zip(op.paulis.to_labels(), op.coeffs, strict=True):
        if label == ident:
            const += float(np.real(c))
        else:
            rest.append((label, float(np.real(c))))
    if not rest:
        return const, []
    return const, list(SparsePauliOp.from_list(rest).group_commuting(qubit_wise=True))


def group_basis(group: SparsePauliOp) -> list[str]:
    """Per-qubit measurement basis letter (qubit 0 first) of a qubit-wise commuting group."""
    n = group.num_qubits
    basis = ["I"] * n
    for label in group.paulis.to_labels():
        for q in range(n):
            if label[n - 1 - q] != "I":
                basis[q] = label[n - 1 - q]
    return basis


def measurement_circuit(state: QuantumCircuit, basis: Sequence[str]) -> QuantumCircuit:
    """`state`, then the rotations that read each qubit in its basis letter, then measure all."""
    n = state.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(state, inplace=True)
    for q, b in enumerate(basis):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
    qc.measure(range(n), range(n))
    return qc


# ---- plain measurement (simulators) ------------------------------------------------------------


@dataclass
class EnergyEstimate:
    energy: float
    stderr: float
    n_groups: int


def estimate_energy(
    run: Runner, state: QuantumCircuit, op: SparsePauliOp, shots: int
) -> EnergyEstimate:
    """<state|op|state> from `shots` shots per group, no mitigation. The error bar is the shot
    noise propagated through the groups."""
    const, groups = group_observable(op)
    n = op.num_qubits
    results = run([measurement_circuit(state, group_basis(g)) for g in groups], shots)
    total, var = const, 0.0
    for group, res in zip(groups, results, strict=True):
        keys = list(res.counts)
        bits = np.array([[int(c) for c in reversed(k)] for k in keys], dtype=np.int8)  # [m, n]
        w = np.array([res.counts[k] for k in keys], dtype=np.float64)
        value = np.zeros(len(keys))
        for label, coef in zip(group.paulis.to_labels(), np.real(group.coeffs), strict=True):
            support = np.array([label[n - 1 - q] != "I" for q in range(n)])
            value += coef * (1 - 2 * (bits[:, support].sum(axis=1) % 2))
        mean = float(np.average(value, weights=w))
        total += mean
        var += float(np.average((value - mean) ** 2, weights=w)) / w.sum()
    return EnergyEstimate(float(total), float(np.sqrt(var)), len(groups))


# ---- mitigated measurement (quantum devices) ---------------------------------------------------


@dataclass(frozen=True)
class ShotPlan:
    """Circuit copies per measurement group and shots per circuit. The total is fixed by the
    Hamiltonian and the requested shots (SHOT_GRAIN x groups circuits of shots / SHOT_GRAIN each:
    the same shots as one circuit per group), so a quote counts it without the state."""

    copies: list[int]
    shots_per_circuit: int

    @property
    def circuits(self) -> int:
        return sum(self.copies)


def shot_plan(op: SparsePauliOp, shots: int, psi: NDArray[np.complex128] | None = None) -> ShotPlan:
    """Split shots across groups by their spread in `psi` when it is known, else by the
    Hamiltonian's coefficients."""
    groups = group_observable(op)[1]
    weights = group_spreads(groups, psi) if psi is not None else group_weights(groups)
    return ShotPlan(allocate_copies(weights, SHOT_GRAIN * len(groups)), max(1, shots // SHOT_GRAIN))


@dataclass
class GroupReport:
    """One measurement circuit's output after mitigation, for a report: the circuit is the state
    rotated into the group's basis, so every group is its own circuit with its own ideal output."""

    dispatch: list[int]  # the runner's indices of the copies of that circuit
    distribution: dict[str, float]  # after readout correction (and electron-count post-selection)
    kept: float  # share of shots that passed the electron-count check


@dataclass
class MitigatedEnergy:
    energy: float  # after readout correction and symmetry post-selection
    stderr: float  # bootstrap over the measured counts
    energy_raw: float  # the same shots without any correction, always reported next to it
    n_groups: int
    n_circuits: int
    shots_per_circuit: int
    kept_fraction: float  # shot-weighted, over groups that could be checked
    readout_corrected: bool
    two_qubit_gates: int | None  # as run on the device (max over circuits), when reported
    reports: list[GroupReport] = field(default_factory=list)  # one per measured group


def estimate_energies_mitigated(
    run: Runner,
    states: Sequence[QuantumCircuit],
    op: SparsePauliOp,
    plan: ShotPlan,
    seed: int,
    sectors: SpinSectors | None,
    *,
    readout: bool = True,
) -> list[MitigatedEnergy]:
    """Energy of each state on a device: weighted shot allocation, readout correction, symmetry
    post-selection, bootstrap error bars. The circuits of every group of every state go out as one
    batch (one job, one queue wait): a target state and its noise-folded copy, for example."""
    const, groups = group_observable(op)
    n = op.num_qubits
    if n > DENSE_MAX_QUBITS:
        raise ValueError(f"mitigation supports up to {DENSE_MAX_QUBITS} qubits")
    if len(plan.copies) != len(groups):
        raise ValueError("the shot plan does not match the observable's groups")
    bases = [group_basis(g) for g in groups]
    circuits: list[QuantumCircuit] = []
    owner: list[tuple[int, int]] = []
    for si, state in enumerate(states):
        for gi, basis in enumerate(bases):
            circuits += [measurement_circuit(state, basis)] * plan.copies[gi]
            owner += [(si, gi)] * plan.copies[gi]
    results = run(circuits, plan.shots_per_circuit)
    bits = bit_table(n)
    rng = np.random.default_rng(seed)
    have_readout = readout and all(r.readout is not None for r in results)
    out: list[MitigatedEnergy] = []
    for si in range(len(states)):
        energy, raw, var, kept_num, kept_den = const, const, 0.0, 0.0, 0.0
        reports: list[GroupReport] = []
        mine_all = [r for r, o in zip(results, owner, strict=True) if o[0] == si]
        for gi, group in enumerate(groups):
            mine = [r for r, o in zip(results, owner, strict=True) if o == (si, gi)]
            counts = [r.counts for r in mine]
            errors = [r.readout if have_readout else None for r in mine]
            g = mitigated_group(
                counts, errors, group, bases[gi], sectors, bits, readout=have_readout
            )
            energy += g.mitigated
            raw += g.raw
            dispatch = [r.dispatch for r in mine if r.dispatch is not None]
            if g.mit_p is not None and dispatch:
                probs = np.clip(g.mit_p, 0.0, None)  # readout inversion can dip below zero
                probs /= probs.sum()
                reports.append(
                    GroupReport(
                        dispatch,
                        {format(i, f"0{n}b"): float(p) for i, p in enumerate(probs) if p > 1e-12},
                        g.kept,
                    )
                )
            var += bootstrap_stderr(counts, errors, group, bases[gi], sectors, bits,
                                    readout=have_readout, rng=rng) ** 2  # fmt: skip
            if g.checked:
                group_shots = sum(sum(c.values()) for c in counts)
                kept_num += g.kept * group_shots
                kept_den += group_shots
        gates = [r.two_qubit_gates for r in mine_all if r.two_qubit_gates is not None]
        out.append(
            MitigatedEnergy(
                energy=float(energy), stderr=float(np.sqrt(var)), energy_raw=float(raw),
                n_groups=len(groups), n_circuits=len(mine_all),
                shots_per_circuit=plan.shots_per_circuit,
                kept_fraction=kept_num / kept_den if kept_den else 1.0,
                readout_corrected=have_readout,
                two_qubit_gates=max(gates, default=None), reports=reports,
            )
        )  # fmt: skip
    return out
