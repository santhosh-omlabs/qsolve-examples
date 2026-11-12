"""Any QUBO with QAOA, in plain Qiskit.

Minimise x'Qx + offset over binary variables, optionally with exactly k of them set to 1. With that
constraint the circuit starts in the uniform superposition of all valid states and mixes only
within them, so no sample breaks it and no penalty term is needed.
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


class GenericQubo(Workflow):
    def __init__(self, input: dict[str, Any], params: dict[str, Any]) -> None:
        super().__init__(input, params)
        q = np.asarray(input["Q"], dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise WorkflowError("Q must be a square matrix")
        n = q.shape[0]
        self.labels: list[str] = list(input.get("labels") or [f"x{i}" for i in range(n)])
        if len(self.labels) != n or len(set(self.labels)) != n:
            raise WorkflowError(f"labels must be {n} unique names")
        k = input.get("cardinality")
        if k is not None and k > n:
            raise WorkflowError(f"cardinality {k} exceeds number of variables {n}")
        self.qubo = Qubo(q, offset=float(input.get("offset", 0.0)), ones=k)

    # -- the contract -----

    def size(self) -> int:
        return self.qubo.n

    def evaluate(self, solution: list[int]) -> float:
        try:
            return self.qubo.evaluate(solution)
        except ValueError as e:
            raise InvalidSolution(str(e)) from e

    def answer(self, outcome: Outcome) -> dict[str, Any]:
        x = [int(b) for b in outcome.solution]
        return {
            "assignment": dict(zip(self.labels, x, strict=True)),
            "ones": [v for v, bit in zip(self.labels, x, strict=True) if bit],
            "objective": self.evaluate(x),
        }

    def estimate(self, target: str) -> Estimate:
        plan = plan_qaoa(self.qubo, self.params, target)
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
                maxiter=int(self.params["optimizer_iters"]),
                seed=ctx.seed,
                counts=counts,
                on_step=lambda s, v, b: ctx.emit("train_step", step=s, value=v, best=b),
            )
        run = sample_qaoa(self.qubo, ctx.sampler, theta, int(self.params["shots"]))
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
