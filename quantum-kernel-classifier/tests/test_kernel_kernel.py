from collections.abc import Sequence

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.kernel import (
    Measured,
    correct_readout,
    counts_to_probs,
    depolarizing_strength,
    exact_kernel,
    mcnemar_exact,
    mirror_circuit,
    rescale_overlap,
    sampled_kernel,
    undo_depolarizing,
    wilson_interval,
)

RNG = np.random.default_rng(0)
X = RNG.uniform(0, 1, size=(5, 3))


def ideal_runner(shots: int, readout: list[list[float]] | None = None):
    """Samples each circuit from its exact distribution (what a perfect device would return)."""
    rng = np.random.default_rng(1)

    def run(circuits: Sequence[QuantumCircuit]) -> list[Measured]:
        out = []
        for qc in circuits:
            body = qc.remove_final_measurements(inplace=False)
            probs = Statevector(body).probabilities()
            n = qc.num_qubits
            drawn = rng.multinomial(shots, probs / probs.sum())
            counts = {format(i, f"0{n}b"): int(c) for i, c in enumerate(drawn) if c}
            out.append(Measured(counts, readout))
        return out

    return run


def test_exact_kernel_is_a_valid_kernel() -> None:
    K = exact_kernel(X, X, reps=2)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1.0)
    assert (K >= -1e-12).all() and (K <= 1 + 1e-12).all()
    assert np.linalg.eigvalsh(K).min() > -1e-9  # positive semi-definite


def test_sampled_kernel_converges_to_exact_and_measures_one_triangle() -> None:
    gram = sampled_kernel(ideal_runner(20000), X, X, reps=2, symmetric=True)
    assert len(gram.pairs) == 5 * 4 // 2 == len(gram.measured)
    assert np.abs(gram.K - exact_kernel(X, X, 2)).max() < 0.03
    assert np.allclose(gram.K, gram.K.T) and np.allclose(np.diag(gram.K), 1.0)


def test_sampled_kernel_rectangular_measures_every_entry() -> None:
    gram = sampled_kernel(ideal_runner(500), X[:2], X, reps=1)
    assert gram.K.shape == (2, 5) and len(gram.pairs) == 10


def test_readout_correction_inverts_the_assignment_matrix() -> None:
    errors = [[0.02, 0.05], [0.01, 0.08], [0.03, 0.02]]
    ideal = RNG.dirichlet(np.ones(8))
    noisy = ideal.copy().reshape(2, 2, 2)
    for q, (p10, p01) in enumerate(errors):
        a = np.array([[1 - p10, p01], [p10, 1 - p01]])
        axis = 2 - q
        noisy = np.moveaxis(np.tensordot(a, noisy, axes=([1], [axis])), 0, axis)
    assert np.allclose(correct_readout(noisy.reshape(-1), errors), ideal)


def test_counts_use_qiskit_bit_order() -> None:
    assert counts_to_probs({"001": 3, "100": 1}, 3)[1] == 0.75


def test_depolarizing_rescale_recovers_the_ideal_distribution() -> None:
    n, lam = 3, 0.2
    ideal = np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0])
    measured = (1 - lam) * ideal + lam / 2**n
    fitted = depolarizing_strength(1 - lam * (1 - 2.0**-n), n)  # a mirror circuit under that noise
    assert fitted == pytest.approx(lam)
    assert np.allclose(undo_depolarizing(measured, fitted, n), ideal)
    assert rescale_overlap(np.array([measured[0]]), fitted, n)[0] == pytest.approx(0.5)


def test_mirror_circuit_returns_to_zero() -> None:
    qc = mirror_circuit(X[0], reps=2).remove_final_measurements(inplace=False)
    assert Statevector(qc).probabilities()[0] == pytest.approx(1.0)


def test_wilson_interval_is_inside_zero_one_and_contains_the_share() -> None:
    lo, hi = wilson_interval(30, 30)
    assert 0.88 < lo < 1.0 and hi == 1.0
    lo, hi = wilson_interval(15, 30)
    assert lo < 0.5 < hi
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
