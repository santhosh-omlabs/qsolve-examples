"""Quantum-kernel SVM against tuned classical classifiers, in plain Qiskit.

The quantum method measures a kernel: for two data points a and b it runs U(a) then U(b)^dagger and
reads the probability of all zeros, which equals |<phi(b)|phi(a)>|^2. A support vector machine
learns from that matrix. Feature-map depth and bandwidth are chosen on a simulator (cross-validated
on noise-free kernels, no QPU cost). Every method predicts the test labels; the SDK re-scores them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from .kernel import (
    Measured,
    Runner,
    correct_readout,
    counts_to_probs,
    depolarizing_strength,
    exact_kernel,
    mcnemar_exact,
    mirror_circuit,
    overlap_circuit,
    p_all_zero,
    rescale_overlap,
    sampled_kernel,
    undo_depolarizing,
    wilson_interval,
)
from qiskit import QuantumCircuit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from qsolve import Context, Estimate, InvalidSolution, Outcome, Workflow, WorkflowError
from qsolve.metering import fidelity, ideal_distribution
from qsolve.sim import S_PER_GATE_AMPLITUDE, S_PER_SHOT

from .data import build_split

FloatArray = NDArray[np.float64]

_C_GRID = [0.1, 1.0, 10.0, 100.0]
_REPS_GRID = [1, 2]
# Quantum kernels depend sharply on feature bandwidth (Shaydulin & Wild, PRA 106, 042407, 2022): too
# wide and the kernel tends to the identity. A 3-point grid (0.25, 0.5, 1.0) missed the optimum on
# 2-feature data (XOR: 0.35 gives 95% test accuracy, the grid's pick 88%).
_BANDWIDTH_GRID = [0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.7, 1.0]
_PROBES = 8  # circuits measured to report mitigation (exact mode) and to calibrate noise (QPU)
# Runtime model (seconds): constants measured on one core, see qsolve.sim.
_S_PER_STATEVECTOR = 0.004  # one exact feature-map state, plus the SVM fits around them
_S_PER_KERNEL_CIRCUIT = 0.001  # the backend call around one kernel circuit
_S_PER_GATE_BUILT = 3.5e-5  # building, composing and inverting the feature maps in Python


def _noisy(target: str) -> bool:
    """Targets whose circuits suffer device noise, so calibration and rescaling make sense."""
    return target.startswith(("ibm:", "fake:"))


def _cv(seed: int, y: NDArray[np.int_]) -> StratifiedKFold:
    """Folds for model selection, the same for every model: five when each class allows it (with
    60 training points, three-fold choices were noisy enough to pick a poor bandwidth)."""
    folds = max(2, min(5, int(np.bincount(y).min())))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def _dist(probs: FloatArray, n: int) -> dict[str, float]:
    return {format(i, f"0{n}b"): float(p) for i, p in enumerate(probs) if p > 0}


class QuantumKernelClassifier(Workflow):
    """Binary classification of a held-out test set."""

    def __init__(self, input: dict[str, Any], params: dict[str, Any]) -> None:
        super().__init__(input, params)
        split = build_split(input)
        self.Xtr, self.ytr, self.Xte, self.yte = (
            split.X_train, split.y_train, split.X_test, split.y_test,
        )  # fmt: skip
        self.n = int(self.Xtr.shape[1])
        self.shots = int(params["shots"])
        self.mode = str(params["kernel_mode"])
        if self.mode not in ("sampled", "exact"):
            raise WorkflowError("kernel_mode must be 'sampled' or 'exact'")
        self._rbf_outcome: Outcome | None = None

    # -- the contract -----

    def size(self) -> int:
        return self.n

    def evaluate(self, solution: Sequence[int]) -> float:
        """Error rate of predicted labels for the test points, against the true labels."""
        pred = np.asarray(solution)
        if pred.shape != self.yte.shape or not np.isin(pred, (0, 1)).all():
            raise InvalidSolution(f"a solution is one 0 or 1 per test point ({len(self.yte)})")
        return float(np.mean(pred != self.yte))

    def answer(self, outcome: Outcome) -> dict[str, Any]:
        y, pred = self.yte, np.asarray(outcome.solution)
        n = len(y)
        lo, hi = wilson_interval(int((pred == y).sum()), n)
        base = np.asarray(self._baseline(0).solution)
        q_only = int(((pred == y) & (base != y)).sum())
        c_only = int(((pred != y) & (base == y)).sum())
        out: dict[str, Any] = {
            "test_accuracy": 1.0 - outcome.objective,
            "test_accuracy_ci95": [lo, hi],
            "baseline_test_accuracy": 1.0 - self._baseline(0).objective,
            "confusion_matrix": {
                "tp": int(((pred == 1) & (y == 1)).sum()),
                "tn": int(((pred == 0) & (y == 0)).sum()),
                "fp": int(((pred == 1) & (y == 0)).sum()),
                "fn": int(((pred == 0) & (y == 1)).sum()),
            },
            "predictions": [int(p) for p in pred],
            "baseline_predictions": [int(p) for p in base],
            "paired_comparison": {
                "test": "McNemar exact (two-sided)",
                "quantum_only_correct": q_only,
                "classical_only_correct": c_only,
                "p_value": mcnemar_exact(q_only, c_only),
            },
            "n_train": len(self.ytr),
            "n_test": n,
            "n_features": self.n,
            "kernel_check": outcome.details.get("kernel_check"),
        }
        if "cv_accuracy" in outcome.details:
            out["cv_accuracy"] = outcome.details["cv_accuracy"]
        return out

    def summary(self, quantum: Outcome, reference: Outcome) -> str:
        lo, hi = wilson_interval(round((1 - quantum.objective) * len(self.yte)), len(self.yte))
        return (
            f"Quantum kernel SVM test accuracy {1 - quantum.objective:.1%} (95% CI "
            f"{lo:.1%}-{hi:.1%}, n={len(self.yte)}) vs classical RBF-SVM "
            f"{1 - reference.objective:.1%}. Objective is test error rate."
        )

    def estimate(self, target: str) -> Estimate:
        m, t, d = len(self.ytr), len(self.yte), self.n
        reps_req, bw_req = int(self.params["reps"]), float(self.params["bandwidth"])
        combos = (1 if reps_req > 0 else len(_REPS_GRID)) * (
            1 if bw_req > 0 else len(_BANDWIDTH_GRID)
        )
        reps = reps_req or max(_REPS_GRID)
        tuning = combos * m * _S_PER_STATEVECTOR
        pairs = m * (m - 1) // 2
        if self.mode == "exact":
            circuits = min(_PROBES, pairs)  # a spot check of the exact kernel on the target
        else:
            circuits = pairs + t * m + (min(_PROBES, m) if _noisy(target) else 0)
        gates = 2 * reps * (2 * d + d * (d - 1) // 2) + d  # compute, uncompute, measure
        per_circuit = (
            _S_PER_KERNEL_CIRCUIT
            + (_S_PER_GATE_BUILT + S_PER_GATE_AMPLITUDE * 2.0**d) * gates
            + S_PER_SHOT * self.shots
        )
        exact_states = (m + t) * _S_PER_STATEVECTOR if self.mode == "exact" else 0.0
        return Estimate(
            qubits=d,
            circuits_on_target=circuits,
            shots_on_target=circuits * self.shots,
            runtime_s=tuning + exact_states + circuits * per_circuit,
            notes=[f"{circuits} kernel circuits ({m} train, {t} test points)"],
        )

    # -- the quantum method -----

    def _tune(self, seed: int) -> tuple[int, float, float]:
        """Feature-map repetitions and bandwidth by cross-validation on the training set, using
        noise-free kernels (classical simulation, no QPU cost). 0 in the parameters = auto-tune."""
        reps_req, bw_req = int(self.params["reps"]), float(self.params["bandwidth"])
        reps_grid = [reps_req] if reps_req > 0 else _REPS_GRID
        bw_grid = [bw_req] if bw_req > 0 else _BANDWIDTH_GRID
        best = (-1.0, reps_grid[0], bw_grid[0])
        cv = _cv(seed, self.ytr)
        for reps in reps_grid:
            for bw in bw_grid:
                X = self.Xtr * bw
                gs = GridSearchCV(SVC(kernel="precomputed"), {"C": _C_GRID}, cv=cv)
                gs.fit(exact_kernel(X, X, reps), self.ytr)
                if gs.best_score_ > best[0] + 1e-12:
                    best = (float(gs.best_score_), reps, bw)
        return best[1], best[2], best[0]

    def _runner(self, ctx: Context) -> Runner:
        """Circuits through the context's sampler as one `run` call: every circuit with the same
        shots goes to the device as one batch (one QPU job), split only at the batch limit."""

        def run(circuits: Sequence[QuantumCircuit]) -> list[Measured]:
            if not circuits:
                return []
            result = ctx.sampler.run([(qc, None, self.shots) for qc in circuits]).result()
            out = []
            for pub in result:
                meta = pub.metadata["qsolve"][0]
                out.append(
                    Measured(pub.data.meas.get_counts(), meta.get("readout"), meta["dispatch"])
                )
            return out

        return run

    def quantum(self, ctx: Context) -> Outcome:
        reps, bw, tuned = self._tune(ctx.seed)
        Xa, Xb = self.Xtr * bw, self.Xte * bw
        run = self._runner(ctx)
        E_tr, E_te = exact_kernel(Xa, Xa, reps), exact_kernel(Xb, Xa, reps)
        ctx.emit("stage", stage="kernel", mode=self.mode, train=len(Xa), test=len(Xb), reps=reps)
        lam: float | None = None
        check: dict[str, float] | None = None
        mirror: float | None = None
        order = np.argsort([E_tr[i, j] for i, j in _upper(len(Xa))], kind="stable")
        pairs = _upper(len(Xa))
        if self.mode == "exact":
            K_tr, K_te = E_tr, E_te
            # Spot-check the exact kernel on the target: a few entries spread over its value range
            ranked = [pairs[k] for k in order]
            picks = [
                ranked[int(r)] for r in np.linspace(0, len(ranked) - 1, min(_PROBES, len(ranked)))
            ]
            circuits = [overlap_circuit(Xa[i], Xa[j], reps) for i, j in picks]
            reported = list(zip(circuits, run(circuits), strict=True))
        else:
            train = sampled_kernel(run, Xa, Xa, reps, symmetric=True)
            test = sampled_kernel(run, Xb, Xa, reps)
            K_tr, K_te = train.K, test.K
            readout_only = _kernel_check(K_tr, K_te, E_tr, E_te, self.shots)["mean_abs_error"]
            if _noisy(ctx.target):  # circuit noise is not just readout: calibrate it out
                mirror = self._mirror_p0(run, Xa, reps, ctx.seed)
                lam = depolarizing_strength(mirror, self.n)
                off = ~np.eye(len(Xa), dtype=bool)
                K_tr = np.where(off, rescale_overlap(K_tr, lam, self.n), K_tr)
                K_te = rescale_overlap(K_te, lam, self.n)
            check = _kernel_check(K_tr, K_te, E_tr, E_te, self.shots)
            check["mean_abs_error_raw"] = _kernel_check(
                train.raw, test.raw, E_tr, E_te, self.shots
            )["mean_abs_error"]
            check["mean_abs_error_readout"] = readout_only
            reported = [
                (overlap_circuit(Xa[i], Xa[j], reps), m)
                for (i, j), m in zip(train.pairs, train.measured, strict=True)
            ] + [
                (overlap_circuit(Xb[i], Xa[j], reps), m)
                for (i, j), m in zip(test.pairs, test.measured, strict=True)
            ]
        for circuit, m in reported:  # the correction is applied to every kernel entry: report each
            self._report(ctx, circuit, m, lam)

        # Shot noise can make K slightly non-PSD; project to the PSD cone for a valid kernel.
        w, v = np.linalg.eigh((K_tr + K_tr.T) / 2)
        K_psd = (v * np.clip(w, 0.0, None)) @ v.T
        gs = GridSearchCV(SVC(kernel="precomputed"), {"C": _C_GRID}, cv=_cv(ctx.seed, self.ytr))
        gs.fit(K_psd, self.ytr)
        return self._outcome(
            gs.predict(K_te),
            cv_accuracy=float(gs.best_score_), best_params=gs.best_params_, kernel_mode=self.mode,
            reps=reps, bandwidth=bw, tuning_cv_accuracy=tuned, kernel_check=check,
            min_eigenvalue_before_psd=float(w.min()), mirror_p0=mirror, depolarizing=lam,
        )  # fmt: skip

    def _mirror_p0(self, run: Runner, Xa: FloatArray, reps: int, seed: int) -> float:
        """Mean P(0...0) of circuits whose exact value is 1; the shortfall is noise."""
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(Xa), size=min(_PROBES, len(Xa)), replace=False)
        measured = run([mirror_circuit(Xa[i], reps) for i in pick])
        return float(np.mean([p_all_zero(m, self.n) for m in measured]))

    def _report(
        self, ctx: Context, circuit: QuantumCircuit, m: Measured, lam: float | None
    ) -> None:
        """Declare the mitigated output of one kernel circuit: its full distribution after readout
        correction and, on noisy targets, the depolarizing rescale fitted on mirror circuits. The
        correction is applied to every kernel entry, so every kernel circuit is reported; the SDK
        scores each against its own ideal output and measures the raw fidelity itself."""
        n, ideal = self.n, ideal_distribution(circuit)
        if ideal is None or m.dispatch is None:
            return
        probs = counts_to_probs(m.counts, n)
        steps = [{"name": "raw counts", "fidelity": fidelity(_dist(probs, n), ideal)}]
        methods: list[str] = []
        if m.readout is not None and len(m.readout) == n:
            probs = np.clip(correct_readout(probs, m.readout), 0.0, None)
            probs = probs / probs.sum()
            steps.append(
                {"name": "readout correction", "fidelity": fidelity(_dist(probs, n), ideal)}
            )
            methods.append("readout-correction")
        if lam is not None:
            probs = undo_depolarizing(probs, lam, n)
            steps.append(
                {
                    "name": "depolarizing rescale (mirror circuits)",
                    "fidelity": fidelity(_dist(probs, n), ideal),
                }
            )
            methods.append("depolarizing-rescale (mirror circuits)")
        ctx.report_mitigation(
            circuits=[m.dispatch], distribution=_dist(probs, n), methods=methods, kept=1.0,
            steps=steps,
        )  # fmt: skip

    # -- classical methods -----

    def _outcome(self, pred: Any, **details: Any) -> Outcome:
        solution = [int(p) for p in pred]
        err = self.evaluate(solution)
        n = len(solution)
        return Outcome(
            solution=solution, objective=err, stderr=math.sqrt(err * (1 - err) / n),
            details={"n_test": n, **details},
        )  # fmt: skip

    def _baseline(self, seed: int) -> Outcome:
        """The tuned RBF-SVM, fitted once per run: the reference, and the comparison in `answer`."""
        if self._rbf_outcome is None:
            d = self.n
            grid = {"C": _C_GRID, "gamma": [0.1 / d, 0.5 / d, 1.0 / d, 2.0 / d, 5.0 / d]}
            gs = GridSearchCV(SVC(kernel="rbf"), grid, cv=_cv(seed, self.ytr))
            gs.fit(self.Xtr, self.ytr)
            self._rbf_outcome = self._outcome(
                gs.predict(self.Xte), cv_accuracy=float(gs.best_score_), best_params=gs.best_params_
            )
        return self._rbf_outcome

    def reference(self, ctx: Context) -> Outcome:
        return self._baseline(ctx.seed)

    def _fit_predict(self, model: Any) -> Outcome:
        model.fit(self.Xtr, self.ytr)
        return self._outcome(model.predict(self.Xte))

    def classical_logistic_regression(self, ctx: Context) -> Outcome:
        return self._fit_predict(LogisticRegression(max_iter=1000))

    def classical_random_forest(self, ctx: Context) -> Outcome:
        return self._fit_predict(RandomForestClassifier(n_estimators=200, random_state=ctx.seed))

    def classical_gradient_boosting(self, ctx: Context) -> Outcome:
        return self._fit_predict(GradientBoostingClassifier(random_state=ctx.seed))


def _upper(m: int) -> list[tuple[int, int]]:
    """Entries above the diagonal of an m x m matrix, in the order `sampled_kernel` takes them."""
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def _kernel_check(
    K_tr: FloatArray, K_te: FloatArray, E_tr: FloatArray, E_te: FloatArray, shots: int
) -> dict[str, float]:
    """How far the measured kernel is from the exact one (classically computable at these sizes),
    next to what shot noise alone would explain: on hardware, the device's own contribution."""
    off = ~np.eye(len(K_tr), dtype=bool)
    measured = np.concatenate([K_tr[off], K_te.ravel()])
    exact = np.concatenate([E_tr[off], E_te.ravel()])
    shot_sd = np.sqrt(np.clip(exact * (1 - exact), 0, None) / shots)
    return {
        "entries": int(len(exact)),
        "mean_abs_error": float(np.mean(np.abs(measured - exact))),
        "max_abs_error": float(np.max(np.abs(measured - exact))),
        # E|N(0, s^2)| = s sqrt(2/pi): the mean absolute error shot noise alone would give
        "shot_noise_expected": float(np.mean(shot_sd) * math.sqrt(2 / math.pi)),
    }
