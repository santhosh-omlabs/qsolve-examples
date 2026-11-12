"""Training QAOA angles with a CVaR objective and COBYLA, noise-free so a device only measures."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import BaseSamplerV2
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .circuits import qaoa_circuit
from .problem import Qubo, all_states

EXACT_MAX_QUBITS = 20  # up to here the whole distribution is scored from the exact statevector
Counts = Callable[[QuantumCircuit], dict[str, int]]


def cvar(
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
    alpha: float,
    order: NDArray[np.intp] | None = None,
) -> float:
    """Mean energy over the best `alpha` share of the probability (alpha=1 is the plain mean).
    Optimising it rewards putting weight on good answers instead of a good average, which is what
    sampling needs. `order` is argsort(energies), passed when the same energies are scored often."""
    if order is None:
        order = np.argsort(energies)
    e, w = energies[order], weights[order] / weights.sum()
    cum = np.cumsum(w)
    keep = np.clip(alpha - (cum - w), 0.0, w)
    return float((e * keep).sum() / keep.sum())


def local_counts(shots: int, seed: int) -> Counts:
    """Sample a circuit on a local noise-free simulator (not a dispatch, never metered)."""
    sim = AerSimulator(method="statevector")

    def run(circuit: QuantumCircuit) -> dict[str, int]:
        compiled = transpile(circuit, sim, seed_transpiler=seed)
        counts: dict[str, int] = (
            sim.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
        )
        return counts

    return run


def sampler_counts(sampler: BaseSamplerV2, shots: int) -> Counts:
    """Sample a circuit through a Qiskit sampler (on a Qsolve run: on the target, and metered)."""

    def run(circuit: QuantumCircuit) -> dict[str, int]:
        result = sampler.run([(circuit, None, shots)]).result()[0]
        counts: dict[str, int] = result.data.c.get_counts()
        return counts

    return run


def training_counts(
    n: int, train_on: str, shots: int, seed: int, sampler: BaseSamplerV2
) -> Counts | None:
    """How to score a circuit while training. None means the exact statevector, used when training
    is local and small; local sampling when it is local and large; the target when `train_on` is
    "backend"."""
    if train_on == "backend":
        return sampler_counts(sampler, shots)
    return None if n <= EXACT_MAX_QUBITS else local_counts(shots, seed)


def train_qaoa(
    qubo: Qubo,
    layers: int,
    *,
    init: str = "dicke",
    alpha: float = 0.25,
    maxiter: int = 80,
    seed: int = 0,
    counts: Counts | None = None,
    on_step: Callable[[int, float, float], None] | None = None,
) -> NDArray[np.float64]:
    """Angles (gammas then betas) that minimise the CVaR of the QAOA energy.

    Starts from a linear ramp (gamma rising, beta falling), which avoids poor local minima. With
    `counts=None` each step scores the exact distribution; otherwise it scores the samples that
    `counts` returns for the circuit. `on_step(step, value, best)` is called after every step.
    """
    n, p, card = qubo.n, layers, qubo.cardinality
    ising = qubo.ising()
    rng = np.random.default_rng(seed)
    ramp = (np.arange(1, p + 1) - 0.5) / p
    x0 = np.concatenate([0.7 * ramp, 0.7 * (1.0 - ramp)]) + rng.normal(0, 0.02, 2 * p)
    history: list[float] = []

    if counts is None:
        states = all_states(n)
        energies = qubo.energies(states)
        order = np.argsort(energies)
        sim = AerSimulator(method="statevector")
        gam, bet = ParameterVector("g", p), ParameterVector("b", p)
        template = qaoa_circuit(ising, gam, bet, measure=False, cardinality=card, init=init)
        template.save_probabilities()
        compiled = transpile(template, sim, optimization_level=1, seed_transpiler=seed)

        def score(theta: NDArray[np.float64]) -> float:
            bound = compiled.assign_parameters(dict(zip([*gam, *bet], theta, strict=True)))
            probs = np.asarray(sim.run(bound).result().data()["probabilities"])
            return cvar(energies, probs, alpha, order)

    else:

        def score(theta: NDArray[np.float64]) -> float:
            circuit = qaoa_circuit(ising, theta[:p], theta[p:], cardinality=card, init=init)
            sampled = counts(circuit)
            x = np.array([[int(c) for c in reversed(k)] for k in sampled], dtype=np.int8)
            w = np.array(list(sampled.values()), dtype=np.float64)
            ok = x.sum(axis=1) == card if card is not None else np.ones(len(x), dtype=bool)
            return cvar(qubo.energies(x[ok]), w[ok], alpha) if ok.any() else 1e9

    def objective(theta: NDArray[np.float64]) -> float:
        value = score(theta)
        history.append(value)
        if on_step:
            on_step(len(history), value, min(history))
        return value

    result = minimize(
        objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3},
        tol=1e-6 if counts is None else 1e-3,
    )  # fmt: skip
    angles: NDArray[np.float64] = result.x
    return angles
