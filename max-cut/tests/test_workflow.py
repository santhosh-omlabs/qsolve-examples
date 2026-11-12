import json
from pathlib import Path

import jsonschema
import pytest
from qsolve import InvalidSolution, WorkflowError, run_workflow
from qsolve.manifest import load_workflow
from qsolve.runner import _instantiate

ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = json.loads((ROOT / "schemas" / "input.json").read_text(encoding="utf-8"))
FAST = {"layers": 2, "optimizer_iters": 40, "shots": 2000}
TRIANGLE = {"nodes": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"], ["a", "c"]]}


def _workflow(input: dict, params: dict | None = None):
    wf = load_workflow(str(ROOT))
    return _instantiate(wf, input, wf.manifest.resolve_params(params))


def test_examples_validate_against_the_input_schema() -> None:
    for path in (ROOT / "examples").glob("*.json"):
        jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), INPUT_SCHEMA)


def test_triangle_best_cut_is_two_and_quantum_matches_the_reference() -> None:
    out = run_workflow(str(ROOT), TRIANGLE, params=FAST, seed=1)
    assert out.result.answer["cut_weight"] == 2.0
    assert out.result.gap == pytest.approx(0.0)
    assert out.result.metrics.matches
    assert not out.result.quality.mitigation.reported  # nothing to mitigate on an ideal simulator


def test_weights_respected() -> None:
    inp = {"nodes": ["a", "b", "c"], "edges": [["a", "b", 10], ["b", "c", 1], ["a", "c", 1]]}
    out = run_workflow(str(ROOT), inp, params=FAST, seed=1)
    a, b = out.result.answer["group_a"], out.result.answer["group_b"]
    assert {"a", "b"} & set(a) and {"a", "b"} & set(b)  # heavy edge a-b must be cut
    assert out.result.answer["cut_weight"] == 11.0


def test_same_seed_gives_the_same_hash() -> None:
    a = run_workflow(str(ROOT), TRIANGLE, params=FAST, seed=5)
    b = run_workflow(str(ROOT), TRIANGLE, params=FAST, seed=5)
    assert a.provenance.result_hash == b.provenance.result_hash


def test_evaluate_rejects_invalid_solutions() -> None:
    wf = _workflow(TRIANGLE)
    assert wf.evaluate([1, 0, 0]) == -2.0
    for bad in ([1, 0], [1, 0, 2]):
        with pytest.raises(InvalidSolution):
            wf.evaluate(bad)


def test_estimate_counts_one_circuit_and_the_final_shots() -> None:
    est = _workflow(TRIANGLE, FAST).estimate("sim-cpu")
    assert (est.qubits, est.circuits_on_target, est.shots_on_target) == (3, 1, 2000)


@pytest.mark.parametrize(
    "bad",
    [
        {"nodes": ["a", "b"], "edges": [["a", "z"]]},
        {"nodes": ["a", "b"], "edges": [["a", "a"]]},
        {"nodes": ["a", "b"], "edges": [["a", "b"], ["b", "a"]]},
        {"nodes": ["a", "a"], "edges": [["a", "a"]]},
    ],
)
def test_bad_graphs_rejected(bad: dict) -> None:
    with pytest.raises(WorkflowError):
        run_workflow(str(ROOT), bad, params=FAST)
