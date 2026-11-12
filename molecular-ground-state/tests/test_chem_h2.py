"""H2 in STO-3G end to end: small enough for seconds, with a known answer (-1.1373 Ha)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

pytest.importorskip("pyscf", reason="needs PySCF (Linux, macOS or WSL)")

from src.chem import (  # noqa: E402
    ActiveSpace,
    ChemSet,
    Measurement,
    MoleculeSpec,
    build_hamiltonian,
    energy_answer,
    plan_cost,
    run_vqe,
    sampler_runner,
)
from src.chem.compact import adapt_state  # noqa: E402

H2 = MoleculeSpec.from_xyz("H 0 0 0\nH 0 0 0.74")


@pytest.fixture(scope="module")
def h2():
    return build_hamiltonian(H2, ActiveSpace(2, 2))


def test_h2_references_and_a_cached_build(h2) -> None:
    assert h2.e_casci == pytest.approx(-1.13728, abs=1e-4)
    assert h2.e_hf == pytest.approx(-1.11675, abs=1e-4)
    assert h2.e_mp2 is not None and h2.e_hf > h2.e_mp2 > h2.e_casci - 0.01
    assert h2.n_qubits == 4 and h2.op.num_qubits == 4
    assert build_hamiltonian(H2, ActiveSpace(2, 2)) is h2  # same arguments, same object


@pytest.mark.parametrize(
    "text",
    ["", "Xx 0 0 0", "H 0 0 nan"],
)
def test_bad_geometries_are_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        MoleculeSpec.from_xyz(text)


def test_open_shell_and_oversized_active_spaces_are_rejected() -> None:
    with pytest.raises(ValueError, match="closed-shell"):
        build_hamiltonian(MoleculeSpec.from_xyz("H 0 0 0"))
    with pytest.raises(ValueError, match="exceeds"):
        build_hamiltonian(H2, ActiveSpace(2, 5))


def test_adapt_reaches_the_exact_energy_with_a_three_cnot_circuit(h2) -> None:
    st = adapt_state(h2.op.to_matrix(sparse=True), 2, 1, 1, h2.e_casci)
    assert st.energy == pytest.approx(h2.e_casci, abs=1e-4)
    assert len(st.excitations) == 1 and st.two_qubit_gates() <= 3
    psi = Statevector(st.circuit())
    assert psi.expectation_value(h2.op).real == pytest.approx(st.energy, abs=1e-8)


def exact_runner(readout: list[list[float]] | None = None, seed: int = 0):
    """Runs circuits exactly (statevector), optionally through a readout-error channel."""
    rng = np.random.default_rng(seed)
    counter = iter(range(10**6))

    def run(circuits, shots):
        out = []
        for qc in circuits:
            body = qc.remove_final_measurements(inplace=False)
            p = Statevector(body).probabilities()
            n = qc.num_qubits
            if readout:
                t = p.reshape([2] * n)
                for q, (p10, p01) in enumerate(readout):
                    a = np.array([[1 - p10, p01], [p10, 1 - p01]])
                    t = np.moveaxis(np.tensordot(a, t, axes=([1], [n - 1 - q])), 0, n - 1 - q)
                p = t.reshape(-1)
            draw = rng.multinomial(shots, p / p.sum())
            counts = {format(i, f"0{n}b"): int(c) for i, c in enumerate(draw) if c}
            out.append(Measurement(counts, readout, next(counter), 3))
        return out

    return run


def test_plain_measurement_matches_the_exact_energy_within_its_error_bar(h2) -> None:
    res = run_vqe(
        ChemSet({"h2": h2}, {"h2": 1.0}), exact_runner(), hardware=False, shots=40000, seed=1
    )
    assert abs(res.energy - h2.e_casci) < 4 * res.stderr + 1e-4
    assert res.noise_free == pytest.approx(h2.e_casci, abs=1e-6)


def test_mitigation_removes_readout_bias_and_reports_every_measured_circuit(h2) -> None:
    readout = [[0.03, 0.08]] * 4
    cs = ChemSet({"h2": h2}, {"h2": 1.0})
    res = run_vqe(cs, exact_runner(readout, 3), hardware=True, shots=40000, seed=1)
    s = res.systems["h2"]
    assert s.energy_raw is not None and s.energy_standard is not None
    assert abs(s.energy - h2.e_casci) < abs(s.energy_raw - h2.e_casci)
    assert s.readout_corrected and 0 < s.kept_fraction <= 1
    assert s.reports and all(abs(sum(r.distribution.values()) - 1) < 1e-9 for r in s.reports)
    assert {x["name"] for x in s.energy_steps()} >= {"raw counts"}
    assert res.circuits > 0


def test_the_quote_counts_what_the_run_sends(h2) -> None:
    cs = ChemSet({"h2": h2}, {"h2": 1.0})
    for hardware in (False, True):
        res = run_vqe(cs, exact_runner(), hardware=hardware, shots=4000, seed=0)
        cost = plan_cost(cs, hardware=hardware, ansatz="uccsd", shots=4000, maxiter=200)
        assert 0.5 <= cost.circuits / res.circuits <= 2.0
        assert 0.5 <= cost.shots / res.shots <= 2.0


def test_sampler_runner_gives_counts_from_a_plain_qiskit_sampler() -> None:
    from qiskit_aer.primitives import SamplerV2

    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure([0, 1], [0, 1])
    (m,) = sampler_runner(SamplerV2(seed=1))([qc], 100)
    assert m.counts == {"01": 100} and m.readout is None


def test_answer_reports_the_error_in_mha(h2) -> None:
    cs = ChemSet({"h2": h2}, {"h2": 1.0}, label="H2")
    res = run_vqe(cs, exact_runner(), hardware=False, shots=40000, seed=2)
    a = energy_answer(
        cs,
        res.details(),
        energy=res.energy,
        stderr=res.stderr,
        exact=h2.e_casci,
        hf=h2.e_hf,
        ccsd_t=h2.e_ccsd_t,
    )
    assert a["within_chemical_accuracy"] and abs(a["error_vs_exact_mha"]) < 1.6
    assert a["correlation_recovered_pct"] == pytest.approx(100, abs=0.1)
