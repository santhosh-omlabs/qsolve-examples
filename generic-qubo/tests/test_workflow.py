import itertools
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from qsolve import InvalidSolution, WorkflowError, run_workflow
from qsolve.manifest import load_workflow
from qsolve.runner import _instantiate

ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = json.loads((ROOT / "schemas" / "input.json").read_text(encoding="utf-8"))
FAST = {"layers": 2, "optimizer_iters": 40, "shots": 2000}


def _workflow(input: dict, params: dict | None = None):
    wf = load_workflow(str(ROOT))
    return _instantiate(wf, input, wf.manifest.resolve_params(params))


def test_examples_validate_against_the_input_schema() -> None:
    for path in (ROOT / "examples").glob("*.json"):
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), INPUT_SCHEMA)


def test_matches_brute_force_on_random_instances() -> None:
    for seed in range(3):
        q = np.random.default_rng(seed).normal(size=(6, 6))
        out = run_workflow(str(ROOT), {"Q": q.tolist()}, params=FAST, seed=seed)
        best = min(np.array(b) @ q @ np.array(b) for b in itertools.product([0, 1], repeat=6))
        assert out.result.baseline_objective == pytest.approx(best)
        assert out.result.objective >= best - 1e-9
        assert out.result.metrics.matches


def test_cardinality_is_enforced_and_post_selection_declared() -> None:
    q = np.random.default_rng(9).normal(size=(7, 7))
    out = run_workflow(str(ROOT), {"Q": q.tolist(), "cardinality": 3}, params=FAST, seed=2)
    assert len(out.result.answer["ones"]) == 3
    mitigation = out.result.quality.mitigation
    assert mitigation is not None and mitigation.reported
    assert "post-selection (exactly k ones)" in mitigation.methods


def test_same_seed_gives_the_same_hash() -> None:
    inp = {"Q": np.random.default_rng(1).normal(size=(5, 5)).tolist()}
    a = run_workflow(str(ROOT), inp, params=FAST, seed=5)
    b = run_workflow(str(ROOT), inp, params=FAST, seed=5)
    assert a.provenance.result_hash == b.provenance.result_hash


def test_evaluate_rejects_invalid_solutions() -> None:
    wf = _workflow({"Q": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "cardinality": 2, "offset": 1.5})
    assert wf.evaluate([1, 1, 0]) == 3.5
    for bad in ([1, 0, 0], [1, 1], [1, 1, 2]):
        with pytest.raises(InvalidSolution):
            wf.evaluate(bad)


def test_estimate_counts_training_on_the_device_only_when_asked() -> None:
    inp = {"Q": np.eye(4).tolist()}
    assert _workflow(inp, FAST).estimate("sim-cpu").circuits_on_target == 1
    est = _workflow(inp, {**FAST, "train_on": "backend", "train_shots": 100}).estimate("sim-cpu")
    assert (est.circuits_on_target, est.shots_on_target) == (41, 2000 + 40 * 100)


@pytest.mark.parametrize(
    "bad",
    [
        {"Q": [[1, 2], [3]]},
        {"Q": [[1, 2], [3, 4]], "labels": ["a", "a"]},
        {"Q": [[1, 2], [3, 4]], "cardinality": 5},
        {"Q": [[1]]},
    ],
)
def test_bad_inputs_rejected(bad: dict) -> None:
    with pytest.raises(WorkflowError):
        run_workflow(str(ROOT), bad, params=FAST)
