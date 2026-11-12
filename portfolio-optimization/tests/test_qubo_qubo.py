import itertools

import numpy as np
import pytest
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from src.qubo import (
    Qubo,
    all_states,
    correct_readout,
    cvar,
    dicke_state,
    exact_search,
    layers_for,
    plan_qaoa,
    qaoa_circuit,
    readout_corrected,
    reference_search,
    sample_qaoa,
    simulated_annealing,
    train_qaoa,
)


def random_qubo(n: int, seed: int = 0, **kwargs) -> Qubo:
    return Qubo(np.random.default_rng(seed).normal(size=(n, n)), **kwargs)


def test_ising_reproduces_the_qubo_energy() -> None:
    qubo = random_qubo(5, offset=0.7)
    ising = qubo.ising()
    for x in all_states(5):
        z = 1 - 2 * x  # a 1 is z = -1
        e = ising.offset + ising.h @ z + sum(v * z[i] * z[j] for (i, j), v in ising.j.items())
        assert e == pytest.approx(qubo.energy(x))


def test_evaluate_rejects_invalid_answers() -> None:
    qubo = random_qubo(4, ones=2)
    assert qubo.evaluate([1, 1, 0, 0]) == pytest.approx(qubo.energy([1, 1, 0, 0]))
    for bad in ([1, 0, 0, 0], [1, 1, 0], [2, 0, 0, 0]):
        with pytest.raises(ValueError):
            qubo.evaluate(bad)


def test_dicke_state_is_uniform_over_the_right_weight() -> None:
    probs = Statevector(dicke_state(5, 2)).probabilities_dict(decimals=9)
    assert len(probs) == 10
    assert all(k.count("1") == 2 and v == pytest.approx(0.1) for k, v in probs.items())


@pytest.mark.parametrize("init", ["dicke", "product"])
def test_constrained_circuit_never_leaves_the_subspace(init: str) -> None:
    qubo = random_qubo(6, ones=3)
    qc = qaoa_circuit(qubo.ising(), [0.4, 0.3], [0.5, 0.2], measure=False, cardinality=3, init=init)
    probs = Statevector(qc).probabilities_dict()
    assert all(k.count("1") == 3 for k, v in probs.items() if v > 1e-9)


def test_exact_search_matches_brute_force_and_respects_ones() -> None:
    qubo = random_qubo(7, seed=3)
    x, e = exact_search(qubo)
    assert e == pytest.approx(min(qubo.energy(b) for b in itertools.product([0, 1], repeat=7)))
    constrained = random_qubo(7, seed=3, ones=3)
    x, e = exact_search(constrained)
    assert sum(x) == 3
    brute = [constrained.energy(b) for b in itertools.product([0, 1], repeat=7) if sum(b) == 3]
    assert e == pytest.approx(min(brute))


def test_annealing_finds_the_optimum_of_a_small_problem_and_is_seeded() -> None:
    qubo = random_qubo(8, seed=1, ones=4)
    x, e = simulated_annealing(qubo, seed=2, sweeps=200, restarts=3)
    assert sum(x) == 4 and e == pytest.approx(exact_search(qubo)[1])
    assert simulated_annealing(qubo, 2, 200, 3) == (x, e)
    assert reference_search(qubo, 0)[1] == pytest.approx(e)


def test_cvar() -> None:
    e, w = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.25] * 4)
    assert cvar(e, w, 1.0) == pytest.approx(2.5)
    assert cvar(e, w, 0.5) == pytest.approx(1.5)
    assert cvar(e, w, 0.25) == pytest.approx(1.0)


def test_readout_correction_recovers_the_true_distribution() -> None:
    errors = [[0.02, 0.05], [0.01, 0.08], [0.03, 0.04]]
    truth = np.array([0.5, 0, 0, 0.25, 0, 0, 0, 0.25])
    t = truth.reshape(2, 2, 2)
    for q, (p10, p01) in enumerate(errors):  # apply the same flips the correction undoes
        a = np.array([[1 - p10, p01], [p10, 1 - p01]])
        t = np.moveaxis(np.tensordot(a, t, axes=([1], [2 - q])), 0, 2 - q)
    assert np.allclose(correct_readout(t.reshape(-1), errors), truth)
    counts = {format(i, "03b"): int(round(v * 1_000_000)) for i, v in enumerate(t.reshape(-1))}
    out = readout_corrected(counts, errors, 3)
    assert out is not None and out["000"] == pytest.approx(0.5, abs=1e-4)
    assert readout_corrected(counts, None, 3) is None


def test_layers_are_capped_off_the_ideal_simulator() -> None:
    assert layers_for(5, "sim-cpu") == 5
    assert layers_for(5, "ibm:ibm_fez") == 2 and layers_for(5, "fake:fez@1.6") == 2
    assert layers_for(1, "ibm:ibm_fez") == 1


@pytest.mark.parametrize("ones", [None, 3])
def test_train_then_sample_finds_the_optimum(ones: int | None) -> None:
    qubo = random_qubo(6, seed=4, ones=ones)
    best = exact_search(qubo)[1]
    theta = train_qaoa(qubo, 2, maxiter=60, seed=1)
    assert np.array_equal(theta, train_qaoa(qubo, 2, maxiter=60, seed=1))
    run = sample_qaoa(qubo, StatevectorSampler(seed=1), theta, 2000)
    assert run.energy == pytest.approx(best)
    assert qubo.evaluate(run.solution) == pytest.approx(run.energy)
    assert (run.report is None) == (ones is None)  # no mitigation to declare when nothing applies
    if run.report:
        assert run.report["kept"] == pytest.approx(1.0)
        assert [s["name"] for s in run.report["steps"]] == ["raw counts", "post-selection"]


def test_penalty_form_is_post_selected_too() -> None:
    qubo = random_qubo(5, seed=2, ones=2, soft=True)
    assert qubo.cardinality is None
    run = sample_qaoa(qubo, StatevectorSampler(seed=1), np.array([0.3, 0.4]), 500)
    assert sum(run.solution) == 2 and run.report is not None and run.report["kept"] < 1.0


def test_plan_counts_what_is_sent() -> None:
    qubo = random_qubo(6, ones=3)
    base = {
        "layers": 3,
        "shots": 1000,
        "train_shots": 200,
        "optimizer_iters": 50,
        "train_on": "sim",
    }
    plan = plan_qaoa(qubo, base, "sim-cpu")
    assert (plan.circuits, plan.shots, plan.exact_training) == (1, 1000, True)
    on_device = plan_qaoa(qubo, {**base, "train_on": "backend"}, "ibm:ibm_fez")
    assert (on_device.circuits, on_device.shots) == (51, 1000 + 50 * 200)
    assert plan.seconds(lambda q, g, c=1, shots=0: c * g * 2.0**q * 1e-9 + shots * 1e-6) > 0
