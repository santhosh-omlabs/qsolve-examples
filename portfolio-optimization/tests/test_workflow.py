import json
from pathlib import Path

import jsonschema
import pytest
from qsolve import InvalidSolution, WorkflowError, run_workflow
from qsolve.manifest import load_workflow
from qsolve.runner import _instantiate

ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = json.loads((ROOT / "schemas" / "input.json").read_text(encoding="utf-8"))
INPUT = json.loads((ROOT / "examples" / "input_small.json").read_text(encoding="utf-8"))
FAST = {"layers": 1, "optimizer_iters": 30, "shots": 1000}


def _workflow(input: dict, params: dict | None = None):
    wf = load_workflow(str(ROOT))
    return _instantiate(wf, input, wf.manifest.resolve_params(params))


def test_examples_validate_against_the_input_schema() -> None:
    for path in (ROOT / "examples").glob("*.json"):
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), INPUT_SCHEMA)


def test_runs_end_to_end_and_matches_the_reference() -> None:
    out = run_workflow(str(ROOT), INPUT, params=FAST, seed=1)
    r = out.result
    assert r.answer["constraint_satisfied"]
    assert len(r.answer["selected_assets"]) == INPUT["num_to_select"]
    assert r.gap >= -1e-9  # quantum can never beat a proven-optimal reference
    assert r.metrics.matches
    assert out.provenance.seed == 1 and out.provenance.target == "sim-cpu"


@pytest.mark.parametrize("params", [{"init": "product"}, {"constraint_mode": "penalty"}])
def test_other_start_states_and_the_penalty_form(params: dict) -> None:
    out = run_workflow(str(ROOT), INPUT, params={**FAST, "layers": 2, **params}, seed=3)
    assert len(out.result.answer["selected_assets"]) == INPUT["num_to_select"]


def test_post_selection_is_declared_with_its_steps() -> None:
    out = run_workflow(str(ROOT), INPUT, params={**FAST, "constraint_mode": "penalty"}, seed=3)
    mitigation = out.result.quality.mitigation
    assert mitigation is not None and mitigation.reported
    assert mitigation.methods == ["post-selection (exactly k ones)"]
    assert mitigation.postselection_rate < 1.0  # the penalty form samples invalid portfolios


def test_same_seed_gives_the_same_hash() -> None:
    a = run_workflow(str(ROOT), INPUT, params=FAST, seed=5)
    b = run_workflow(str(ROOT), INPUT, params=FAST, seed=5)
    assert a.provenance.result_hash == b.provenance.result_hash


def test_evaluate_rejects_a_portfolio_of_the_wrong_size() -> None:
    wf = _workflow(INPUT)
    n, k = len(INPUT["assets"]), INPUT["num_to_select"]
    wf.evaluate([1] * k + [0] * (n - k))
    with pytest.raises(InvalidSolution):
        wf.evaluate([1] * n)


def test_estimate_caps_layers_on_a_device() -> None:
    deep = {**FAST, "layers": 6}
    sim = _workflow(INPUT, deep).estimate("sim-cpu")
    dev = _workflow(INPUT, deep).estimate("ibm:ibm_fez")
    assert dev.runtime_s < sim.runtime_s and dev.circuits_on_target == 1


def test_accepts_returns_history_instead_of_moments() -> None:
    hist = [[0.01, 0.02, -0.01], [0.02, -0.01, 0.01], [-0.01, 0.03, 0.0], [0.03, 0.0, 0.02]]
    inp = {"assets": ["A", "B", "C"], "returns_history": hist, "num_to_select": 2}
    out = run_workflow(str(ROOT), inp, params=FAST, seed=0)
    assert len(out.result.answer["selected_assets"]) == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"assets": ["A"], "num_to_select": 1},
        {
            "assets": ["A", "B"],
            "num_to_select": 3,
            "expected_returns": [0.1, 0.2],
            "covariance": [[1, 0], [0, 1]],
        },
        {"assets": ["A", "B"], "num_to_select": 1},
        {
            "assets": ["A", "A"],
            "num_to_select": 1,
            "expected_returns": [0.1, 0.2],
            "covariance": [[1, 0], [0, 1]],
        },
    ],
)
def test_bad_inputs_raise_clear_errors(bad: dict) -> None:
    with pytest.raises(WorkflowError):
        run_workflow(str(ROOT), bad, params=FAST)


def test_out_of_envelope_rejected_before_solving() -> None:
    n = 25
    inp = {
        "assets": [f"S{i}" for i in range(n)],
        "num_to_select": 5,
        "expected_returns": [0.1] * n,
        "covariance": [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)],
    }
    with pytest.raises(WorkflowError):
        run_workflow(str(ROOT), inp, params=FAST)
