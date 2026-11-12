"""Binding, solvation, adsorption or reaction energy as a signed sum of VQE energies.

Each named system (a complex, its parts, a reactant, ...) gets its own small active space and is
solved with VQE; the answer is `sum(weight * energy)`. Without explicit `terms` the first system is
the complex (+1) and every other one a separated fragment (-1): a binding energy. Qsolve also runs
the exact active-space, Hartree-Fock, MP2 and CCSD(T) energies of the same combination. The
algorithm (Hamiltonians, circuits, mitigation) is in `src/chem/` (start with `src/chem/__init__.py`).
"""

from __future__ import annotations

from typing import Any

from .chem import (
    ActiveSpace,
    ChemSet,
    MoleculeSpec,
    build_hamiltonian,
    energy_answer,
    energy_summary,
    energy_units,
    on_hardware,
    plan_cost,
    run_vqe,
    sampler_runner,
)
from qsolve import (
    Context,
    Estimate,
    InvalidSolution,
    MethodUnavailable,
    Outcome,
    Workflow,
    WorkflowError,
)
from qsolve.sim import S_PER_CIRCUIT, S_PER_SHOT, S_PER_TRANSPILE

from .library import CHARGES, LIBRARY


def _molecule(spec: dict[str, Any], basis: str) -> MoleculeSpec:
    name = spec.get("name")
    if name is None:
        return MoleculeSpec.from_xyz(spec["xyz"], charge=spec.get("charge", 0), basis=basis)
    if name not in LIBRARY:
        raise ValueError(f"unknown species {name!r}; built-ins: {sorted(LIBRARY)}")
    return MoleculeSpec.from_xyz(
        LIBRARY[name], charge=spec.get("charge", CHARGES.get(name, 0)), basis=basis
    )


class InteractionEnergy(Workflow):
    """A signed combination of system energies, in Hartree."""

    def __init__(self, input: dict[str, Any], params: dict[str, Any]) -> None:
        super().__init__(input, params)
        systems: dict[str, Any] = input["systems"]
        names = list(systems)
        terms = input.get("terms") or {n: (1.0 if i == 0 else -1.0) for i, n in enumerate(names)}
        if set(terms) != set(names):
            raise WorkflowError("terms must name exactly the systems provided")
        default_active = input.get("active_space")
        built = {}
        for name, spec in systems.items():
            try:
                act = spec.get("active_space") or default_active
                built[name] = build_hamiltonian(
                    _molecule(spec, input["basis"]), ActiveSpace(**act) if act else None
                )
            except ValueError as e:
                raise WorkflowError(f"system {name!r}: {e}") from e
        self.chemset = ChemSet(
            built, {k: float(v) for k, v in terms.items()},
            label=input.get("label", "interaction energy"),
        )  # fmt: skip

    # -- the contract -----

    def size(self) -> int:
        return self.chemset.n_qubits

    def evaluate(self, solution: Any) -> float:
        raise InvalidSolution("energies are measured, not re-evaluated from a solution")

    def answer(self, outcome: Outcome) -> dict[str, Any]:
        cs = self.chemset
        if "systems" not in outcome.details:  # a classical method: its energy, in every unit
            return {"label": cs.label, **energy_units(outcome.objective)}
        return energy_answer(
            cs, outcome.details, energy=outcome.objective, stderr=outcome.stderr or 0.0,
            exact=cs.reference("casci"), hf=cs.reference("hf"), ccsd_t=cs.reference("ccsd_t"),
        )  # fmt: skip

    def summary(self, quantum: Outcome, reference: Outcome) -> str:
        return energy_summary(
            self.chemset.label,
            quantum.objective,
            quantum.stderr or 0.0,
            reference.objective,
            self.chemset.reference("hf"),
        )

    def estimate(self, target: str) -> Estimate:
        p = self.params
        cost = plan_cost(
            self.chemset, hardware=on_hardware(target), ansatz=p["ansatz"], shots=int(p["shots"]),
            maxiter=int(p["optimizer_iters"]),
        )  # fmt: skip
        return Estimate(
            qubits=self.size(),
            circuits_on_target=cost.circuits,
            shots_on_target=cost.shots,
            classical_cost_class="high" if self.size() > 14 else "medium",
            runtime_s=cost.runtime(S_PER_TRANSPILE, S_PER_CIRCUIT, S_PER_SHOT),
            notes=[f"{cost.circuits} measurement circuits"],
        )

    # -- the quantum method: VQE -----

    def quantum(self, ctx: Context) -> Outcome:
        """Train noise-free, then measure the energy of the trained state through `ctx`. On a
        quantum device the compact circuit is used and the counts are mitigated (see `src/chem/measure.py`)."""
        p = self.params
        vqe = run_vqe(
            self.chemset, sampler_runner(ctx.sampler), hardware=on_hardware(ctx.target),
            ansatz=p["ansatz"], shots=int(p["shots"]), maxiter=int(p["optimizer_iters"]),
            seed=ctx.seed, progress=ctx.emit,
        )  # fmt: skip
        for s in vqe.systems.values():  # every measured circuit gets the same correction
            for r in s.reports:
                ctx.report_mitigation(
                    circuits=r.dispatch, distribution=r.distribution,
                    methods=s.mitigation_methods, kept=r.kept,
                    steps=s.energy_steps() if len(vqe.systems) == 1 else [],
                )  # fmt: skip
        return Outcome(
            solution=None, objective=vqe.energy, stderr=vqe.stderr, details=vqe.details()
        )

    # -- classical references -----

    def _reference_energy(self, kind: str, name: str) -> Outcome:
        energy = self.chemset.reference(kind)
        if energy is None:
            raise MethodUnavailable(f"{name} was not computed for these systems")
        return Outcome(solution=None, objective=energy)

    def reference(self, ctx: Context) -> Outcome:
        return self._reference_energy("casci", "the exact active-space energy")

    def classical_hartree_fock(self, ctx: Context) -> Outcome:
        return self._reference_energy("hf", "Hartree-Fock")

    def classical_mp2(self, ctx: Context) -> Outcome:
        return self._reference_energy("mp2", "MP2")

    def classical_ccsd_t(self, ctx: Context) -> Outcome:
        return self._reference_energy("ccsd_t", "CCSD(T)")
