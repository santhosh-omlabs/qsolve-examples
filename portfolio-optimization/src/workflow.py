"""Pick k assets: cardinality-constrained Markowitz with QAOA, in plain Qiskit.

Minimise  risk_aversion * x'Cx - mu'x  over x in {0,1}^n with exactly k ones (equal weights on the
chosen assets). Two ways to enforce "exactly k":
  * mixer (default): the circuit starts in a valid portfolio and an XY mixer moves only between
    valid ones, so every sample is valid and the objective is not drowned by a penalty;
  * penalty: the usual soft constraint P (sum x - k)^2 inside Q; the circuit then explores every
    portfolio and invalid samples are post-selected away.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from .qubo import (
    Qubo,
    layers_for,
    plan_qaoa,
    reference_search,
    sample_qaoa,
    simulated_annealing,
    train_qaoa,
    training_counts,
)
from qsolve import Context, Estimate, InvalidSolution, Outcome, Workflow, WorkflowError
from qsolve.sim import S_PER_TRANSPILE, simulator_threads, statevector_seconds


def _moments(inp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Expected returns and covariance, given directly or estimated from a returns history."""
    n = len(inp["assets"])
    if "returns_history" in inp:
        r = np.asarray(inp["returns_history"], dtype=float)
        if r.shape[1] != n:
            raise WorkflowError(f"returns_history has {r.shape[1]} columns, expected {n}")
        return r.mean(axis=0), np.cov(r, rowvar=False)
    if "expected_returns" not in inp or "covariance" not in inp:
        raise WorkflowError("Provide returns_history, or both expected_returns and covariance")
    mu = np.asarray(inp["expected_returns"], dtype=float)
    cov = np.asarray(inp["covariance"], dtype=float)
    if mu.shape != (n,) or cov.shape != (n, n):
        raise WorkflowError(f"expected_returns/covariance must match {n} assets")
    if not np.allclose(cov, cov.T, atol=1e-9):
        raise WorkflowError("covariance must be symmetric")
    return mu, cov


class Portfolio(Workflow):
    def __init__(self, input: dict[str, Any], params: dict[str, Any]) -> None:
        super().__init__(input, params)
        self.assets: list[str] = list(input["assets"])
        n, self.k = len(self.assets), int(input["num_to_select"])
        if len(set(self.assets)) != n:
            raise WorkflowError("asset names must be unique")
        if self.k > n:
            raise WorkflowError(f"num_to_select ({self.k}) exceeds number of assets ({n})")
        risk = float(input.get("risk_aversion", 0.5))
        self.mu, self.cov = _moments(input)
        q = risk * self.cov - np.diag(self.mu)
        offset, soft = 0.0, params["constraint_mode"] == "penalty"
        if soft:  # P large enough that no gain can pay for breaking the constraint
            penalty = float(params["penalty_scale"]) * (
                np.abs(self.mu).max() + 2.0 * risk * np.abs(self.cov).sum(axis=1).max()
            )
            q = q + penalty * (np.ones((n, n)) - np.eye(n)) + np.eye(n) * penalty * (1 - 2 * self.k)
            offset = penalty * self.k * self.k
        self.qubo = Qubo(q, offset=offset, ones=self.k, soft=soft)
        self.init = str(params["init"])

    # -- the contract -----

    def size(self) -> int:
        return self.qubo.n

    def evaluate(self, solution: list[int]) -> float:
        """risk_aversion * variance - return of the chosen assets, for exactly k of them."""
        try:
            return self.qubo.evaluate(solution)
        except ValueError as e:
            raise InvalidSolution(str(e)) from e

    def answer(self, outcome: Outcome) -> dict[str, Any]:
        x = np.asarray(outcome.solution, dtype=float)
        picked = [a for a, bit in zip(self.assets, x, strict=True) if bit]
        w = x / x.sum() if x.sum() else x
        return {
            "selected_assets": picked,
            "weights": {a: 1.0 / len(picked) for a in picked},
            "expected_return": float(w @ self.mu),
            "risk": float(np.sqrt(max(w @ self.cov @ w, 0.0))),
            "constraint_satisfied": len(picked) == self.k,
        }

    def summary(self, quantum: Outcome, reference: Outcome) -> str:
        a, b = self.answer(quantum), self.answer(reference)
        same = set(a["selected_assets"]) == set(b["selected_assets"])
        return (
            f"Selected {', '.join(a['selected_assets'])} "
            f"(return {a['expected_return']:.2%}, risk {a['risk']:.2%}). "
            f"{'Matches' if same else 'Differs from'} the classical optimum "
            f"({', '.join(b['selected_assets'])}; "
            f"return {b['expected_return']:.2%}, risk {b['risk']:.2%})."
        )

    def estimate(self, target: str) -> Estimate:
        plan = plan_qaoa(self.qubo, self.params, target, self.init)
        return Estimate(
            qubits=plan.qubits,
            circuits_on_target=plan.circuits,
            shots_on_target=plan.shots,
            classical_cost_class="low" if plan.exact_training else "high",
            runtime_s=plan.seconds(statevector_seconds) + S_PER_TRANSPILE,
            notes=plan.notes,
        )

    # -- quantum: QAOA, trained noise-free, sampled on the target -----

    def quantum(self, ctx: Context) -> Outcome:
        p = layers_for(int(self.params["layers"]), ctx.target)
        counts = training_counts(
            self.qubo.n,
            self.params["train_on"],
            int(self.params["train_shots"]),
            ctx.seed,
            ctx.sampler,
        )
        with simulator_threads():
            theta = train_qaoa(
                self.qubo,
                p,
                init=self.init,
                maxiter=int(self.params["optimizer_iters"]),
                seed=ctx.seed,
                counts=counts,
                on_step=lambda s, v, b: ctx.emit("train_step", step=s, value=v, best=b),
            )
        run = sample_qaoa(self.qubo, ctx.sampler, theta, int(self.params["shots"]), init=self.init)
        if run.report and run.dispatch is not None:
            ctx.report_mitigation(circuits=[run.dispatch], **run.report)
        run.details["layers_requested"] = int(self.params["layers"])
        return Outcome(run.solution, run.energy, samples=run.samples, details=run.details)

    # -- classical -----

    def reference(self, ctx: Context) -> Outcome:
        x, e = reference_search(self.qubo, ctx.seed)
        return Outcome(x, e)

    def classical_annealing(self, ctx: Context) -> Outcome:
        x, e = simulated_annealing(self.qubo, ctx.seed, sweeps=300, restarts=5)
        return Outcome(x, e)
