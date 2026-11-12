"""Max-cut with QAOA, in plain Qiskit.

Split a weighted graph into two groups so the edges between the groups weigh as much as possible.
With x_i = 1 for one group and 0 for the other, an edge (i, j) of weight w is cut exactly when
x_i + x_j - 2 x_i x_j = 1, so maximising the cut is minimising the QUBO x'Qx built below.
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


class MaxCut(Workflow):
    def __init__(self, input: dict[str, Any], params: dict[str, Any]) -> None:
        super().__init__(input, params)
        self.nodes: list[str] = [str(v) for v in input["nodes"]]
        if len(set(self.nodes)) != len(self.nodes):
            raise WorkflowError("node names must be unique")
        index = {name: i for i, name in enumerate(self.nodes)}
        q = np.zeros((len(self.nodes), len(self.nodes)))
        self.total_weight = 0.0
        seen: set[frozenset[str]] = set()
        for edge in input["edges"]:
            u, v = str(edge[0]), str(edge[1])
            w = float(edge[2]) if len(edge) == 3 else 1.0
            if u not in index or v not in index:
                raise WorkflowError(f"edge ({u}, {v}) references an unknown node")
            if u == v:
                raise WorkflowError(f"self-loop on {u} is not allowed")
            if frozenset((u, v)) in seen:
                raise WorkflowError(f"duplicate edge ({u}, {v})")
            seen.add(frozenset((u, v)))
            i, j = index[u], index[v]
            q[i, i] -= w
            q[j, j] -= w
            q[i, j] += w
            q[j, i] += w
            self.total_weight += w
        self.qubo = Qubo(q)

    # -- the contract -----

    def size(self) -> int:
        return self.qubo.n

    def evaluate(self, solution: list[int]) -> float:
        """The negative cut weight of an assignment of the nodes to the two groups."""
        try:
            return self.qubo.evaluate(solution)
        except ValueError as e:
            raise InvalidSolution(str(e)) from e

    def answer(self, outcome: Outcome) -> dict[str, Any]:
        x = outcome.solution
        return {
            "group_a": [v for v, bit in zip(self.nodes, x, strict=True) if not bit],
            "group_b": [v for v, bit in zip(self.nodes, x, strict=True) if bit],
            "cut_weight": -self.evaluate(x),
            "total_weight": self.total_weight,
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
