import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest
import yaml
from src.chem import HA_TO_KCAL, MoleculeSpec
from qsolve import InvalidSolution, MethodUnavailable, Outcome, WorkflowError, run_workflow

pytest.importorskip("pyscf", reason="chemistry workflows need PySCF (Linux, macOS or WSL)")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.library import ION, SOLVENTS, SPECIES, pairs  # noqa: E402
from src.workflow import MAX_SOLVENTS, ElectrolyteSolventScreen, binding_site  # noqa: E402

INPUT_SCHEMA = json.loads((ROOT / "schemas" / "input.json").read_text(encoding="utf-8"))
OUTPUT_SCHEMA = json.loads((ROOT / "schemas" / "output.json").read_text(encoding="utf-8"))
KCAL = 1 / HA_TO_KCAL
ONE_ATOM = {"solvent_xyz": "H 0 0 0", "complex_xyz": "H 0 0 0"}


def test_library_matches_the_form_and_every_geometry_is_a_minimum() -> None:
    assert INPUT_SCHEMA["properties"]["solvents"]["items"]["enum"] == list(SOLVENTS)
    for name in SOLVENTS:
        for key, charge in ((name, 0), (f"{ION}{name}", 1)):
            g = SPECIES[key]
            assert g["charge"] == charge
            assert g["imaginary_frequencies_cm1"] == [], key
        assert SPECIES[f"{ION}{name}"]["xyz"].count("\n") == SPECIES[name]["xyz"].count("\n") + 1


def test_examples_validate() -> None:
    for p in (ROOT / "examples").glob("*.json"):
        jsonschema.validate(json.loads(p.read_text(encoding="utf-8")), INPUT_SCHEMA)


def test_every_declared_answer_field_has_a_title_for_the_platform_to_show() -> None:
    props = OUTPUT_SCHEMA["properties"]
    assert props and all("title" in p for p in props.values())


@pytest.mark.parametrize(
    "bad",
    [
        {"solvents": []},
        {"solvents": ["EC"], "custom_solvents": [{"name": "EC", **ONE_ATOM}]},
        {"custom_solvents": [{"name": f"S{i}", **ONE_ATOM} for i in range(MAX_SOLVENTS + 1)]},
        {"custom_solvents": [{"name": "Li+ PC", **ONE_ATOM}]},
    ],
)
def test_bad_selections_are_rejected_before_any_chemistry(bad: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(WorkflowError):
        ElectrolyteSolventScreen(bad, {})


def test_binding_site_follows_the_same_atoms_in_complex_and_solvent() -> None:
    water = MoleculeSpec.from_xyz("O 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0")
    li_last = MoleculeSpec.from_xyz(
        "O 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0\nLi 0 -1.84 0", charge=1
    )
    assert binding_site(water, li_last) == ([0], 3)
    li_first = MoleculeSpec.from_xyz(
        "Li 0 -1.84 0\nO 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0", charge=1
    )
    assert binding_site(water, li_first) == ([0], 0)  # site index is the solvent's own
    for bad in (
        "H 0.76 0.59 0\nO 0 0 0\nH -0.76 0.59 0\nLi 0 -1.84 0",  # atoms reordered
        "O 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0",  # no Li
        "O 0 0 0\nH 0.76 0.59 0\nH -0.76 0.59 0\nLi 0 -4 0",  # Li not bound
    ):
        with pytest.raises(WorkflowError):
            binding_site(water, MoleculeSpec.from_xyz(bad, charge=1))


def test_every_library_complex_has_its_binding_site() -> None:
    expected = {"EC": 1, "FEC": 1, "DMC": 1, "THF": 1, "ACN": 1, "water": 1, "DME": 2}
    for name, n_donors in expected.items():
        site, _li = binding_site(
            MoleculeSpec.from_xyz(SPECIES[name]["xyz"]),
            MoleculeSpec.from_xyz(SPECIES[f"{ION}{name}"]["xyz"], charge=1),
        )
        assert len(site) == n_donors, name  # DME chelates with both oxygens


def _screen(binding: dict[str, float], exact: dict[str, float], stderr: float = 0.0):
    """A workflow and a quantum outcome whose binding energies (kcal/mol) are the given values,
    without running any chemistry."""
    names = list(binding)
    ion, solvent = -7.0, -100.0

    def cplx(b: float) -> float:
        return -107.0 + b * KCAL

    def system(casci: float, hf: float, mp2: float) -> SimpleNamespace:
        meta = {"basis": "6-31g*", "active": {"n_electrons": 4, "n_orbitals": 4}}
        return SimpleNamespace(e_casci=casci, e_hf=hf, e_mp2=mp2, metadata=meta)

    systems = {ION: system(ion, ion, ion)}
    measured = {ION: ion}
    for n in names:
        key = f"{ION}·{n}"
        systems[n] = system(solvent, solvent, solvent)
        systems[key] = system(cplx(exact[n]), cplx(exact[n] + 0.5), cplx(exact[n] - 3.0))
        measured[n], measured[key] = solvent, cplx(binding[n])
    wf = object.__new__(ElectrolyteSolventScreen)
    wf.chemset = SimpleNamespace(systems=systems, label="test")
    wf.chosen = pairs({"solvents": names})
    details = {
        "systems": {
            k: {"energy_measured": v, "energy_noise_free": v, "stderr": stderr}
            for k, v in measured.items()
        }
    }
    return wf, Outcome(solution=None, objective=0.0, stderr=stderr, details=details)


def test_ranking_and_breakdown() -> None:
    wf, out = _screen(
        {"EC": -48.0, "DMC": -44.0, "water": -34.0}, {"EC": -48.2, "DMC": -44.1, "water": -34.0}
    )
    a = wf.answer(out)
    jsonschema.validate(a, OUTPUT_SCHEMA)
    assert [r["solvent"] for r in a["ranking"]] == ["EC", "DMC", "water"]
    assert a["strongest_binder"] == "EC" and a["weakest_binder"] == "water"
    assert a["ranking_matches_exact"] and a["within_chemical_accuracy"]
    ec = a["ranking"][0]
    assert ec["binding_energy_kcal_mol"] == pytest.approx(-48.0)
    assert ec["error_kcal_mol"] == pytest.approx(0.2)
    assert ec["hartree_fock_kcal_mol"] == pytest.approx(-47.7)
    assert ec["active_space_correlation_kcal_mol"] == pytest.approx(-0.5)
    assert ec["mp2_full_space_kcal_mol"] == pytest.approx(-51.2)
    assert a["active_space"] == {"n_electrons": 4, "n_orbitals": 4, "centred_on": "binding site"}
    assert "Ethylene carbonate" in wf.summary(out, out)


def test_an_order_inside_the_error_bars_is_reported_as_unresolved() -> None:
    # DMC and DME 0.3 kcal/mol apart with ~0.4 kcal/mol error bars each: not a real ordering
    wf, out = _screen(
        {"DMC": -44.0, "DME": -44.3, "water": -34.0},
        {"DMC": -44.4, "DME": -44.2, "water": -34.0},
        stderr=0.25 * KCAL,
    )
    a = wf.answer(out)
    assert not a["ranking_matches_exact"]
    assert a["unresolved_neighbours"] == [["DME", "DMC"]]


def test_a_quick_preview_ranks_a_solvent_and_reports_mitigation_on_a_noisy_device() -> None:
    case = {"solvents": ["water"], "basis": "sto-3g"}
    out = run_workflow(str(ROOT), case, target="fake:fez@1.6", params={"shots": 4000}, seed=1)
    r = out.result
    assert [s["solvent"] for s in r.answer["ranking"]] == ["water"]
    m = r.quality.mitigation
    assert m is not None and m.reported and m.fidelity_mitigated >= m.fidelity_raw
    assert {x.id for x in r.methods if x.unavailable} == {"ccsd-t"}  # out of reach, said so
    assert all(x.claimed is None for x in r.methods)


def test_energies_are_not_re_evaluated_and_missing_references_are_unavailable() -> None:
    wf = object.__new__(ElectrolyteSolventScreen)
    wf.chemset = SimpleNamespace(reference=lambda kind: None)
    with pytest.raises(InvalidSolution):
        wf.evaluate(None)
    with pytest.raises(MethodUnavailable):
        wf.classical_ccsd_t(None)  # type: ignore[arg-type]


def test_the_noise_model_case_is_declared() -> None:
    cases = yaml.safe_load((ROOT / "benchmarks" / "cases.yaml").read_text(encoding="utf-8"))[
        "cases"
    ]
    assert any(c.get("target", "").startswith("fake:") for c in cases)
