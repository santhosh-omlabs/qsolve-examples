import json
import sys
from pathlib import Path

import jsonschema
import pytest
from qsolve import InvalidSolution, MethodUnavailable, WorkflowError, run_workflow

pytest.importorskip("pyscf", reason="chemistry workflows need PySCF (Linux, macOS or WSL)")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT_SCHEMA = json.loads((ROOT / "schemas" / "input.json").read_text(encoding="utf-8"))
OUTPUT_SCHEMA = json.loads((ROOT / "schemas" / "output.json").read_text(encoding="utf-8"))
from src.workflow import MolecularGroundState  # noqa: E402

H2 = json.loads((ROOT / "examples" / "h2.json").read_text(encoding="utf-8"))


def test_examples_validate() -> None:
    for p in (ROOT / "examples").glob("*.json"):
        jsonschema.validate(json.loads(p.read_text(encoding="utf-8")), INPUT_SCHEMA)


def test_every_declared_answer_field_has_a_title_for_the_platform_to_show() -> None:
    props = OUTPUT_SCHEMA["properties"]
    assert props and all("title" in p for p in props.values())


def test_h2_matches_the_exact_energy_and_every_number_is_measured_by_the_platform() -> None:
    r = run_workflow(str(ROOT), H2, seed=1).result
    assert r.baseline_objective == pytest.approx(-1.13728, abs=1e-4)  # the known H2/STO-3G answer
    assert abs(r.objective - r.baseline_objective) < 0.0016  # chemical accuracy
    assert r.metrics is not None and r.metrics.accuracy > 0.999 and r.metrics.matches
    assert r.answer["within_chemical_accuracy"] is True
    by_id = {m.id: m for m in r.methods}
    assert set(by_id) == {"vqe", "casci", "hartree-fock", "mp2", "ccsd-t"}
    assert all(m.verified == "reported" and m.claimed is None for m in r.methods)  # energies
    assert by_id["vqe"].stderr and by_id["vqe"].dispatches > 0
    assert by_id["hartree-fock"].value > by_id["mp2"].value > by_id["ccsd-t"].value - 0.01


def test_the_same_seed_gives_the_same_result() -> None:
    a = run_workflow(str(ROOT), H2, seed=3)
    b = run_workflow(str(ROOT), H2, seed=3)
    assert a.provenance.result_hash == b.provenance.result_hash


def test_the_accuracy_level_defaults_to_the_quick_one() -> None:
    without_basis = {"name": "H2", "active_space": {"n_electrons": 2, "n_orbitals": 2}}
    jsonschema.validate(without_basis, INPUT_SCHEMA)
    assert run_workflow(str(ROOT), without_basis, seed=1).result.metrics is not None


def test_a_noisy_device_run_reports_its_mitigation_next_to_the_raw_counts() -> None:
    out = run_workflow(str(ROOT), H2, target="fake:fez@1.6", params={"shots": 4000}, seed=1)
    m = out.result.quality.mitigation
    assert m is not None and m.reported and m.fidelity_mitigated >= m.fidelity_raw
    assert "electron-count post-selection" in m.methods
    assert [s["name"] for s in m.details["steps"]][0] == "raw counts"  # one system: error steps
    assert out.result.quality.circuit.two_qubit_gates <= 20  # the compact circuit, not UCCSD


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "Unobtainium", "basis": "sto-3g"},
        {"name": "H2", "basis": "sto-3g", "active_space": {"n_electrons": 2, "n_orbitals": 6}},
        {"xyz": "Xx 0 0 0", "basis": "sto-3g"},
        {"xyz": "H 0 0 0", "basis": "sto-3g"},  # an open-shell atom
        {"basis": "sto-3g"},
    ],
)
def test_bad_input_is_a_clear_user_error(bad: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(WorkflowError):
        run_workflow(str(ROOT), bad, seed=0)


def test_a_reference_that_was_not_computed_is_unavailable_and_energies_are_not_re_evaluated() -> (
    None
):
    wf = MolecularGroundState(H2, {"ansatz": "uccsd", "shots": 1000, "optimizer_iters": 5})
    wf.chemset.systems["H2"]._computed = {"mp2": None, "ccsd_t": None}
    with pytest.raises(MethodUnavailable):
        wf.classical_mp2(None)  # type: ignore[arg-type]
    with pytest.raises(InvalidSolution):
        wf.evaluate([1.0])
