import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from qsolve import InvalidSolution, WorkflowError, run_workflow

ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = json.loads((ROOT / "schemas" / "input.json").read_text(encoding="utf-8"))
SMALL = {"dataset": {"kind": "moons", "n_samples": 24, "noise": 0.1, "seed": 0}}
TINY = {"dataset": {"kind": "moons", "n_samples": 14, "noise": 0.1, "seed": 0}}
FAST = {"kernel_mode": "exact", "reps": 1, "bandwidth": 0.5, "shots": 500}


def _workflow(input: dict, params: dict | None = None):
    from qsolve.manifest import load_workflow
    from qsolve.runner import _instantiate

    wf = load_workflow(str(ROOT))
    return _instantiate(wf, input, wf.manifest.resolve_params(params))


@pytest.mark.parametrize("name", ["breast_cancer", "moons"])
def test_examples_validate_against_the_input_schema(name: str) -> None:
    example = json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))
    jsonschema.validate(example, INPUT_SCHEMA)


def test_real_dataset_sample_is_stratified_reduced_and_within_limits() -> None:
    wf = _workflow(
        {"dataset": {"kind": "breast_cancer", "n_samples": 90, "features": 4, "seed": 3}}
    )
    assert wf.Xtr.shape == (60, 4) and wf.Xte.shape == (30, 4)
    assert set(np.unique(wf.ytr)) == {0, 1} and set(np.unique(wf.yte)) == {0, 1}


def test_seed_picks_a_different_sample() -> None:
    a = _workflow({"dataset": {"kind": "breast_cancer", "n_samples": 40, "seed": 0}})
    b = _workflow({"dataset": {"kind": "breast_cancer", "n_samples": 40, "seed": 1}})
    assert not np.allclose(a.Xtr, b.Xtr)


@pytest.mark.parametrize(
    "bad",
    [
        {"dataset": {"kind": "breast_cancer", "n_samples": 100}, "test_fraction": 0.1},  # 90 train
        {"train": {"X": [[0.0], [1.0]], "y": [0, 1]}},  # no test table
        {"data": {"X": [[0.0], [1.0], [2.0], [3.0]], "y": [1, 1, 1, 1]}},  # one class
        {"train": {"X": [[0.0], [1.0]], "y": [0, 1]}, "test": {"X": [[0.0, 1.0]], "y": [0]}},
    ],
)
def test_bad_input_raises_workflow_error(bad: dict) -> None:
    with pytest.raises(WorkflowError):
        _workflow(bad)


def test_evaluate_is_the_error_rate_and_rejects_invalid_solutions() -> None:
    wf = _workflow(SMALL)
    truth = [int(y) for y in wf.yte]
    assert wf.evaluate(truth) == 0.0
    assert wf.evaluate([1 - t for t in truth]) == 1.0
    for bad in (truth[:-1], [2] * len(truth)):
        with pytest.raises(InvalidSolution):
            wf.evaluate(bad)


def test_same_seed_gives_the_same_hash() -> None:
    a = run_workflow(str(ROOT), SMALL, params=FAST, seed=3)
    b = run_workflow(str(ROOT), SMALL, params=FAST, seed=3)
    assert a.provenance.result_hash == b.provenance.result_hash


def test_quantum_kernel_matches_the_tuned_rbf_svm_on_real_data() -> None:
    out = run_workflow(
        str(ROOT),
        {"dataset": {"kind": "breast_cancer", "n_samples": 90, "features": 4, "seed": 0}},
        params={"kernel_mode": "exact"},
        seed=0,
    )
    a = out.result.answer
    assert (
        a["test_accuracy"] >= 0.85 and abs(a["test_accuracy"] - a["baseline_test_accuracy"]) <= 0.1
    )
    assert all(m.verified == "recomputed" for m in out.result.methods)  # every method re-scored


def test_a_users_own_table_runs_end_to_end() -> None:
    # the 'my data' input format (was a benchmark case; with 3 test points it measured nothing)
    data = {
        "X": [[0.1, 0.2], [0.2, 0.1], [0.15, 0.3], [0.3, 0.2], [0.9, 0.8], [0.8, 0.9], [0.85, 0.7],
              [0.7, 0.85], [0.25, 0.25], [0.8, 0.75]],
        "y": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    }  # fmt: skip
    out = run_workflow(str(ROOT), {"data": data, "test_fraction": 0.3}, params=FAST, seed=4)
    assert out.result.answer["n_test"] == 3 and out.result.answer["test_accuracy"] == 1.0


def test_exact_mode_spot_checks_the_kernel_on_the_backend_and_reports_mitigation() -> None:
    out = run_workflow(str(ROOT), SMALL, params=FAST, seed=1)
    q = out.result.quality
    assert q is not None and q.circuit is not None and q.circuit.circuits == 8
    assert q.mitigation is not None and q.mitigation.reported


def test_sampled_mode_sends_one_circuit_per_kernel_entry_and_tracks_the_exact_kernel() -> None:
    out = run_workflow(
        str(ROOT), SMALL, params={**FAST, "kernel_mode": "sampled", "shots": 4000}, seed=1
    )
    m = out.result.answer
    train, test = 16, 8  # 24 points, 33% held out
    assert out.result.quality.circuit.circuits == train * (train - 1) // 2 + test * train
    assert m["kernel_check"]["mean_abs_error"] < 0.03  # shot noise only on a noise-free simulator


def test_noisy_target_reports_readout_and_depolarizing_steps() -> None:
    out = run_workflow(
        str(ROOT), TINY, params={**FAST, "kernel_mode": "sampled", "shots": 500},
        target="fake:fez@1.6", seed=1,
    )  # fmt: skip
    mit = out.result.quality.mitigation
    names = [s["name"] for s in mit.details["steps"]]
    assert names == ["raw counts", "readout correction", "depolarizing rescale (mirror circuits)"]
    assert mit.fidelity_mitigated >= mit.fidelity_raw - 0.02 and mit.postselection_rate == 1.0
