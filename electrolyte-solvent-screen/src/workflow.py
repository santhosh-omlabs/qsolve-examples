"""Rank battery-electrolyte solvents by how strongly they bind Li+, each energy from VQE.

For each solvent S the workflow computes the gas-phase binding energy
`dE(S) = E(Li+ . S) - E(S) - E(Li+)` (more negative = binds more strongly). All systems go into
one `ChemSet`; its combined objective is the sum of the binding energies (each complex +1, each
solvent -1, the shared ion -N), and `answer` reads the per-system energies to report each solvent
and the ranking. Qsolve also runs the exact active-space, Hartree-Fock and MP2 energies of the same
combination. The algorithm (Hamiltonians, circuits, mitigation) is in `src/chem/` (start with `src/chem/__init__.py`).

Active spaces are centred on the binding site, identically in each complex and its free solvent:
the donor atoms' valence orbitals, never the bound ion's empty ones. Orbitals picked by energy
alone would differ between the two (the ion's empty orbitals become the complex's lowest), and the
mismatch would show up as fake binding energy.
"""

from __future__ import annotations

import math
from typing import Any

from .chem import (
    HA_TO_KCAL,
    ActiveSpace,
    ChemProblem,
    ChemSet,
    MoleculeSpec,
    build_hamiltonian,
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

from .library import GEOMETRY_METHOD, ION, SOLVENTS, pairs

MAX_SOLVENTS = 8
BOUND_WITHIN = 2.4  # Angstrom: atoms this close to Li form its binding site (Li-O/N are 1.7-2.0)
CHEMICAL_ACCURACY_KCAL = 1.0


def binding_site(solvent: MoleculeSpec, complex_: MoleculeSpec) -> tuple[list[int], int]:
    """The solvent atoms bound to Li (indices valid in both molecules' solvent part) and the Li
    index in the complex. The complex must list the solvent's atoms in the same order plus one Li,
    so the same atoms (and therefore the same orbitals) can be followed from one to the other."""
    li = [i for i, (symbol, _) in enumerate(complex_.atoms) if symbol == "Li"]
    if len(li) != 1:
        raise WorkflowError("the complex must contain exactly one Li atom")
    others = [a for i, a in enumerate(complex_.atoms) if i != li[0]]
    if [s for s, _ in others] != [s for s, _ in solvent.atoms]:
        raise WorkflowError(
            "the complex must contain the solvent's atoms in the same order, plus one Li atom"
        )
    li_xyz = complex_.atoms[li[0]][1]
    site = [i for i, (_, xyz) in enumerate(others) if math.dist(xyz, li_xyz) < BOUND_WITHIN]
    if not site:
        raise WorkflowError(f"no solvent atom is within {BOUND_WITHIN} Å of the Li atom")
    return site, li[0]


def _molecule(name: str, xyz: str, charge: int, basis: str) -> MoleculeSpec:
    try:
        return MoleculeSpec.from_xyz(xyz, charge=charge, basis=basis)
    except ValueError as e:
        raise WorkflowError(f"{name}: {e}") from e


def _hamiltonian(name: str, mol: MoleculeSpec, active: ActiveSpace) -> ChemProblem:
    """MP2 is the full-space reference (CCSD(T) is out of reach at ~110 basis functions); density
    fitting halves the build and moves binding energies by ~0.01 kcal/mol."""
    try:
        return build_hamiltonian(mol, active, with_ccsd_t=False, density_fit=True)
    except ValueError as e:
        raise WorkflowError(f"{name}: {e}") from e


class ElectrolyteSolventScreen(Workflow):
    """Li+ binding energies of a set of solvents, summed into one objective (Hartree)."""

    def __init__(self, input: dict[str, Any], params: dict[str, Any]) -> None:
        super().__init__(input, params)
        chosen = pairs(input)
        if not chosen:
            raise WorkflowError(f"choose at least one solvent; built-ins: {sorted(SOLVENTS)}")
        if len(chosen) > MAX_SOLVENTS:
            raise WorkflowError(f"at most {MAX_SOLVENTS} solvents per run")
        names = [p.solvent for p in chosen]
        if len(set(names)) != len(names):
            raise WorkflowError("each solvent may appear only once")
        for n in names:  # checked here: the schema avoids regular expressions (they can hang)
            if not n.isascii() or not all(ch.isalnum() or ch in "-_" for ch in n):
                raise WorkflowError(f"solvent name {n!r}: use letters, digits, dash or underscore")

        basis = input.get("basis", "6-31g*")
        size = input.get("active_space") or {"n_electrons": 4, "n_orbitals": 4}
        self.chosen = chosen
        # Li 1s is a frozen core orbital inside every complex; the bare ion's own correlation in a
        # 2-orbital space is 0.03 kcal/mol, kept so every system is computed the same way
        ion = _molecule(ION, "Li 0 0 0", 1, basis)
        systems = {ION: _hamiltonian(ION, ion, ActiveSpace(2, 2))}
        coefficients = {ION: -float(len(chosen))}
        for p in chosen:
            solvent = _molecule(p.solvent, p.solvent_xyz, 0, basis)
            complex_ = _molecule(p.complex_key, p.complex_xyz, 1, basis)
            try:
                site, li = binding_site(solvent, complex_)
            except WorkflowError as e:
                raise WorkflowError(f"{p.solvent}: {e}") from e
            # the complex's active space sits on the same atoms (their indices shift past Li) and
            # leaves out the ion's own empty orbitals
            in_complex = tuple(i if i < li else i + 1 for i in site)
            systems[p.solvent_key] = _hamiltonian(
                p.solvent, solvent, ActiveSpace(**size, center_atoms=tuple(site))
            )
            systems[p.complex_key] = _hamiltonian(
                p.complex_key, complex_,
                ActiveSpace(**size, center_atoms=in_complex, exclude_atoms=(li,)),
            )  # fmt: skip
            coefficients[p.solvent_key] = -1.0
            coefficients[p.complex_key] = 1.0
        self.chemset = ChemSet(
            systems, coefficients, f"{ION} binding across {len(chosen)} solvents"
        )

    # -- the contract -----

    def size(self) -> int:
        return self.chemset.n_qubits

    def evaluate(self, solution: Any) -> float:
        raise InvalidSolution("energies are measured, not re-evaluated from a solution")

    def answer(self, outcome: Outcome) -> dict[str, Any]:
        """Per-solvent Li+ binding energies, ranked, each split into its parts: mean-field
        (Hartree-Fock) binding + active-space correlation (the part VQE computes), checked against
        the exact active-space value and a full-space MP2 reference."""
        cs = self.chemset
        if "systems" not in outcome.details:  # a classical method: its energy, in every unit
            return {"label": cs.label, "ion": ION, **energy_units(outcome.objective)}
        q = outcome.details["systems"]
        measured = {k: v["energy_measured"] for k, v in q.items()}
        noise_free = {k: v["energy_noise_free"] for k, v in q.items()}
        stderr = {k: v["stderr"] for k, v in q.items()}
        exact = {k: s.e_casci for k, s in cs.systems.items()}
        hf = {k: s.e_hf for k, s in cs.systems.items()}
        mp2 = {k: s.e_mp2 for k, s in cs.systems.items()}

        rows = []
        for p in self.chosen:
            keys = (p.complex_key, p.solvent_key, ION)

            def de(e: dict[str, Any], keys: tuple[str, str, str] = keys) -> float | None:
                if any(e.get(k) is None for k in keys):
                    return None
                return (e[keys[0]] - e[keys[1]] - e[keys[2]]) * HA_TO_KCAL

            value, ref, mean_field = de(measured), de(exact), de(hf)
            assert value is not None and ref is not None and mean_field is not None
            rows.append(
                {
                    "solvent": p.solvent,
                    "name": p.title,
                    "binding_energy_kcal_mol": value,
                    "stderr_kcal_mol": math.sqrt(sum(stderr[k] ** 2 for k in keys)) * HA_TO_KCAL,
                    "noise_free_kcal_mol": de(noise_free),
                    "exact_active_space_kcal_mol": ref,
                    "error_kcal_mol": value - ref,
                    "hartree_fock_kcal_mol": mean_field,
                    "active_space_correlation_kcal_mol": ref - mean_field,
                    "mp2_full_space_kcal_mol": de(mp2),
                }
            )

        ranked = sorted(rows, key=lambda r: r["binding_energy_kcal_mol"])  # most negative first
        exact_order = [
            r["solvent"] for r in sorted(rows, key=lambda r: r["exact_active_space_kcal_mol"])
        ]
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
        # neighbours whose difference is within two combined standard errors are not resolved
        unresolved = [
            [a["solvent"], b["solvent"]]
            for a, b in zip(ranked, ranked[1:], strict=False)
            if b["binding_energy_kcal_mol"] - a["binding_energy_kcal_mol"]
            < 2 * math.hypot(a["stderr_kcal_mol"], b["stderr_kcal_mol"])
        ]
        worst = max(abs(r["error_kcal_mol"]) for r in rows)
        act = cs.systems[self.chosen[0].solvent_key].metadata  # a solvent's space, not the ion's
        method = f"{GEOMETRY_METHOD['xc'].upper()}/{GEOMETRY_METHOD['basis'].upper()}"
        return {
            "label": cs.label,
            "ion": ION,
            "basis": act["basis"],
            "active_space": {**act["active"], "centred_on": "binding site"},
            "geometries": f"{method} minima",
            "ranking": ranked,
            "strongest_binder": ranked[0]["solvent"],
            "weakest_binder": ranked[-1]["solvent"],
            "ranking_matches_exact": [r["solvent"] for r in ranked] == exact_order,
            "unresolved_neighbours": unresolved,
            "max_abs_error_kcal_mol": worst,
            "within_chemical_accuracy": worst <= CHEMICAL_ACCURACY_KCAL,
        }

    def summary(self, quantum: Outcome, reference: Outcome) -> str:
        a = self.answer(quantum)
        top = a["ranking"][0]
        order = " > ".join(r["solvent"] for r in a["ranking"])
        return (
            f"Li+ binds {top['name']} most strongly ({top['binding_energy_kcal_mol']:.1f} "
            f"+/- {top['stderr_kcal_mol']:.1f} kcal/mol). Order: {order}. "
            f"{'Matches' if a['ranking_matches_exact'] else 'Differs from'} the exact active-space "
            f"ranking; largest error {a['max_abs_error_kcal_mol']:.2f} kcal/mol."
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
            notes=[f"{len(self.chemset.systems)} systems; {cost.circuits} measurement circuits"],
        )

    # -- the quantum method: VQE -----

    def quantum(self, ctx: Context) -> Outcome:
        """Train each system noise-free, then measure its energy through `ctx`. On a quantum device
        the compact circuit is used and the counts are mitigated (see `src/chem/measure.py`)."""
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
