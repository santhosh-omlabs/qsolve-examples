import json
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml
from qsolve import InvalidSolution, MethodUnavailable, WorkflowError, run_workflow

pytest.importorskip("pyscf", reason="chemistry workflows need PySCF (Linux, macOS or WSL)")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT_SCHEMA = json.loads((ROOT / "schemas" / "input.json").read_text(encoding="utf-8"))
OUTPUT_SCHEMA = json.loads((ROOT / "schemas" / "output.json").read_text(encoding="utf-8"))
CASES = {
    c["name"]: c
    for c in yaml.safe_load((ROOT / "benchmarks" / "cases.yaml").read_text(encoding="utf-8"))[
        "cases"
    ]
}
STRETCH = CASES["h2-stretch-energy"]["input"]  # the cheapest case: H2 at two bond lengths

from src.workflow import InteractionEnergy  # noqa: E402


def test_examples_validate() -> None:
    for p in (ROOT / "examples").glob("*.json"):
        jsonschema.validate(json.loads(p.read_text(encoding="utf-8")), INPUT_SCHEMA)


def test_every_declared_answer_field_has_a_title_for_the_platform_to_show() -> None:
    props = OUTPUT_SCHEMA["properties"]
    assert props and all("title" in p for p in props.values())


def test_a_bond_stretch_energy_is_combined_from_its_systems_and_measured_by_the_platform() -> None:
    r = run_workflow(str(ROOT), STRETCH, seed=1).result
    quantum = next(m for m in r.methods if m.role == "quantum")
    reference = next(m for m in r.methods if m.role == "reference")
    # the workflow's own declared tolerance for this case; the energy is a small difference, so the
    # relative accuracy is lower than for a total energy
    assert (
        abs(quantum.value - reference.value)
        <= CASES["h2-stretch-energy"]["expect"]["max_abs_error"]
    )
    assert reference.value == pytest.approx(0.1391, abs=2e-4)  # E(1.5 A) - E(0.74 A), exact
    assert r.metrics is not None and r.metrics.accuracy > 0.95
    systems = r.answer["systems"]
    assert len(systems) == 2 and sorted(s["coefficient"] for s in systems.values()) == [-1, 1]
    assert all(m.verified == "reported" and m.claimed is None for m in r.methods)
    assert {m.role for m in r.methods} == {"quantum", "reference", "classical"}


def test_the_same_seed_gives_the_same_result() -> None:
    a = run_workflow(str(ROOT), STRETCH, seed=2)
    b = run_workflow(str(ROOT), STRETCH, seed=2)
    assert a.provenance.result_hash == b.provenance.result_hash


def test_a_noisy_device_run_reports_mitigation_for_every_system() -> None:
    out = run_workflow(str(ROOT), STRETCH, target="fake:fez@1.6", params={"shots": 4000}, seed=1)
    m = out.result.quality.mitigation
    assert m is not None and m.reported and m.fidelity_mitigated >= m.fidelity_raw
    assert m.details["steps"] == []  # error steps are only drawn for a single system


def test_explicit_terms_weight_the_systems() -> None:
    wf = InteractionEnergy({**STRETCH, "terms": {"stretched": 2, "equilibrium": -1}}, {})
    assert wf.chemset.coefficients == {"stretched": 2.0, "equilibrium": -1.0}


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"basis": "sto-3g", "systems": {}},
        {"basis": "nope", "systems": {"a": {"name": "H2"}}},
        {"basis": "sto-3g", "systems": {"a": {"name": "Unobtainium"}}},
        {"basis": "sto-3g", "systems": {"a": {"name": "H2"}}, "terms": {"b": 1}},
    ],
)
def test_bad_input_is_a_clear_user_error(bad: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(WorkflowError):
        run_workflow(str(ROOT), bad, seed=0)


def test_a_reference_that_was_not_computed_is_unavailable_and_energies_are_not_re_evaluated() -> (
    None
):
    wf = InteractionEnergy(STRETCH, {})
    for s in wf.chemset.systems.values():
        s._computed = {"mp2": None, "ccsd_t": None}
    with pytest.raises(MethodUnavailable):
        wf.classical_ccsd_t(None)  # type: ignore[arg-type]
    with pytest.raises(InvalidSolution):
        wf.evaluate(None)
