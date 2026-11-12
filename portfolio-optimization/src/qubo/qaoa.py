"""Sampling the trained QAOA circuit, cleaning the samples, and costing a run beforehand."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit.primitives import BaseSamplerV2
from qiskit.quantum_info import Statevector

from .circuits import qaoa_circuit
from .problem import Qubo
from .train import EXACT_MAX_QUBITS

# On ibm_fez noise models (x1 and x1.6) a third layer only added noise: QUBO-5 hit the optimum 7.2x
# as often as random with 2 layers, ring-6 max-cut 7.1x with 1-2, and constrained circuits of 400+
# CZ lost the signal entirely. Anything that is not an ideal simulator gets at most this many.
HARDWARE_MAX_LAYERS = 2
READOUT_MAX_QUBITS = 16  # readout correction works on the full 2^n distribution
STEPS_MAX_QUBITS = 16  # so does the ideal output the mitigation steps are scored against
TOP_SAMPLES = 100


def layers_for(requested: int, target: str) -> int:
    """The QAOA depth to use on `target` (simulators: as requested)."""
    return requested if target.startswith("sim") else min(requested, HARDWARE_MAX_LAYERS)


def correct_readout(
    probs: NDArray[np.float64], errors: Sequence[Sequence[float]]
) -> NDArray[np.float64]:
    """Undo independent readout flips. errors[q] = (P(1|0), P(0|1)) of bit q. The result may have
    small negative entries; sums of it are unbiased, which is why callers clip and renormalise."""
    n = len(errors)
    t = probs.reshape([2] * n)  # axis 0 is the most significant bit
    for q, (p10, p01) in enumerate(errors):
        a = np.array([[1.0 - p10, p01], [p10, 1.0 - p01]])
        axis = n - 1 - q
        t = np.moveaxis(np.tensordot(np.linalg.inv(a), t, axes=([1], [axis])), 0, axis)
    return t.reshape(-1)


def readout_corrected(counts: dict[str, int], errors: Any, n: int) -> dict[str, float] | None:
    """The sampled distribution with each bit's calibrated readout error inverted. None when the
    device reported no calibration (a simulator) or n is too large to correct."""
    if not isinstance(errors, list) or len(errors) != n or n > READOUT_MAX_QUBITS:
        return None
    probs = np.zeros(2**n)
    for key, c in counts.items():
        probs[int(key.replace(" ", ""), 2)] += c
    q = np.clip(correct_readout(probs / probs.sum(), errors), 0.0, None)
    if q.sum() <= 0:
        return None
    q /= q.sum()
    return {format(int(i), f"0{n}b"): float(q[i]) for i in np.flatnonzero(q > 1e-12)}


def fidelity(p: dict[str, float], q: dict[str, float]) -> float:
    """Classical (Bhattacharyya) fidelity of two distributions over bit strings."""
    sp, sq = sum(p.values()), sum(q.values())
    return float(sum(np.sqrt(v / sp * q[k] / sq) for k, v in p.items() if k in q) ** 2)


@dataclass
class QaoaRun:
    """What sampling the trained circuit produced.

    Bit strings are in Qiskit order (variable 0 on the right); `solution` and `samples` are lists
    of 0/1 with variable 0 first. `report` holds the arguments for declaring the mitigation applied
    (None when none was: an ideal simulator and no constraint), minus the dispatch to cite.
    """

    solution: list[int]
    energy: float
    samples: list[dict[str, Any]]
    dispatch: int | None
    report: dict[str, Any] | None
    details: dict[str, Any] = field(default_factory=dict)


def sample_qaoa(
    qubo: Qubo,
    sampler: BaseSamplerV2,
    theta: NDArray[np.float64],
    shots: int,
    *,
    init: str = "dicke",
) -> QaoaRun:
    """Run the circuit once and clean up what comes back.

    The answer is the lowest-energy valid state among the raw samples. Readout correction (where
    the device reports its errors) and post-selection on the number of ones only change the
    distribution that is reported, never which sampled state is picked.
    """
    n, p, need = qubo.n, len(theta) // 2, qubo.ones
    circuit = qaoa_circuit(
        qubo.ising(), theta[:p], theta[p:], cardinality=qubo.cardinality, init=init
    )
    result = sampler.run([(circuit, None, shots)]).result()[0]
    counts: dict[str, int] = result.data.c.get_counts()
    meta = (result.metadata.get("qsolve") or [{}])[0]  # per-circuit device metadata, when reported

    total = sum(counts.values())
    raw = {k: v / total for k, v in counts.items()}
    corrected = readout_corrected(counts, meta.get("readout"), n)
    dist = corrected or raw

    def valid(key: str) -> bool:
        return need is None or key.count("1") == need

    ok = sorted(k for k in counts if valid(k))
    if not ok:
        raise RuntimeError(
            "no sample had the required number of ones: noise swamped the constraint"
        )
    x = np.array([[int(c) for c in reversed(k)] for k in ok], dtype=np.int8)
    e = qubo.energies(x)
    w = np.array([counts[k] for k in ok], dtype=np.float64)
    best = int(np.argmin(e))
    mean = float(np.average(e, weights=w))
    stderr = float(np.sqrt(np.average((e - mean) ** 2, weights=w) / w.sum()))
    kept = float(w.sum() / total)

    report = None
    methods = (["readout correction"] if corrected else []) + (
        ["post-selection (exactly k ones)"] if need is not None else []
    )
    if methods:
        final = {k: v for k, v in dist.items() if valid(k)} or dist
        steps: list[dict[str, Any]] = []
        if n <= STEPS_MAX_QUBITS:
            bare = circuit.remove_final_measurements(inplace=False)
            ideal = {k: v for k, v in Statevector(bare).probabilities_dict().items() if v > 1e-12}
            steps = [{"name": "raw counts", "fidelity": fidelity(raw, ideal)}]
            if corrected:
                steps.append({"name": "readout correction", "fidelity": fidelity(dist, ideal)})
            if need is not None:
                steps.append({"name": "post-selection", "fidelity": fidelity(final, ideal)})
        report = {"distribution": final, "methods": methods, "kept": kept, "steps": steps}

    top = sorted(dist.items(), key=lambda kv: -kv[1])[:TOP_SAMPLES]
    return QaoaRun(
        solution=x[best].tolist(),
        energy=float(e[best]),
        samples=[{"solution": [int(c) for c in reversed(k)], "p": v} for k, v in top],
        dispatch=meta.get("dispatch"),
        report=report,
        details={
            "layers": p,
            "theta": [float(t) for t in theta],
            "mixer": "xy-ring" if qubo.cardinality is not None else "x",
            "mean_energy": mean,
            "mean_energy_stderr": stderr,
            "valid_fraction": kept,
            "readout_corrected": corrected is not None,
        },
    )


@dataclass(frozen=True)
class QaoaPlan:
    """What a QAOA run will send to its target, and the work it does around that."""

    circuits: int
    shots: int
    qubits: int
    gates: int
    evals: int
    final_shots: int
    train_shots: int
    exact_training: bool
    notes: list[str]

    def seconds(self, statevector_seconds: Callable[..., float]) -> float:
        """Simulation time under a cost model `statevector_seconds(qubits, gates, circuits, shots)`:
        every optimiser step simulates the circuit; exact training also scores all 2^n states."""
        n, g = self.qubits, self.gates
        if self.exact_training:
            train = statevector_seconds(n, g, self.evals)
            train += self.evals * 1.3e-7 * 2.0**n + 5e-9 * n * 2.0**n  # scoring, energies
        else:
            train = statevector_seconds(n, g, self.evals, self.train_shots)
        return train + statevector_seconds(n, g, shots=self.final_shots)


def plan_qaoa(qubo: Qubo, params: dict[str, Any], target: str, init: str = "dicke") -> QaoaPlan:
    """The circuits and shots a run on `target` sends (one final circuit; with `train_on: backend`
    also one per optimiser step) and the size of the simulation around them."""
    p = layers_for(int(params["layers"]), target)
    final, train_shots = int(params["shots"]), int(params["train_shots"])
    evals = max(int(params["optimizer_iters"]), 2 * p + 2)  # COBYLA's minimum
    on_target = params["train_on"] == "backend"
    exact = params["train_on"] == "sim" and qubo.n <= EXACT_MAX_QUBITS
    gates = qaoa_circuit(
        qubo.ising(), [0.0] * p, [0.0] * p, cardinality=qubo.cardinality, init=init
    ).size()
    notes = ["training runs on the target backend"] if on_target else []
    if params["train_on"] == "sim" and not exact:
        notes.append("training uses sampled simulation (slower)")
    return QaoaPlan(
        circuits=1 + (evals if on_target else 0),
        shots=final + (evals * train_shots if on_target else 0),
        qubits=qubo.n, gates=gates, evals=evals, final_shots=final, train_shots=train_shots,
        exact_training=exact, notes=notes,
    )  # fmt: skip
