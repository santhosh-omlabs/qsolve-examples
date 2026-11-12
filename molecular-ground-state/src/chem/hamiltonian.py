"""Molecule to qubit Hamiltonian: PySCF integrals, an active space, the Jordan-Wigner mapping.

Needs PySCF (the `chem` extra), which is Linux and macOS only. Spin orbitals are blocked: qubit
`p` is the alpha spin of spatial orbital `p`, qubit `n + p` its beta spin.

The classical reference energies (Hartree-Fock, the exact energy inside the active space, and the
full-space MP2 and CCSD(T)) are computed on the way, so a workflow can show what the quantum
calculation is judged against.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

CHEMICAL_ACCURACY_HA = 1.6e-3
MAX_ATOMS = 40
MAX_BASIS_FUNCTIONS = 120  # the SCF of the whole molecule stays well under a minute
CCSD_T_MAX_BASIS_FUNCTIONS = 60  # the full-molecule CCSD(T) reference grows as N^7

# Nuclear charge of the supported elements.
ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11,
    "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30,
}  # fmt: skip

Atom = tuple[str, tuple[float, float, float]]  # element symbol, position in Angstrom


@dataclass(frozen=True)
class MoleculeSpec:
    """A molecule or ion. Hashable, so a built Hamiltonian can be cached on it."""

    atoms: tuple[Atom, ...]
    charge: int = 0
    basis: str = "sto-3g"

    def __post_init__(self) -> None:
        if not 1 <= len(self.atoms) <= MAX_ATOMS:
            raise ValueError(f"a molecule has 1 to {MAX_ATOMS} atoms, not {len(self.atoms)}")
        for symbol, xyz in self.atoms:
            if symbol not in ATOMIC_NUMBER:
                raise ValueError(f"unsupported element {symbol!r}")
            if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
                raise ValueError(f"atom {symbol} needs three finite coordinates")

    @classmethod
    def from_atoms(cls, atoms: list[tuple[str, list[float]]], **kwargs: Any) -> MoleculeSpec:
        """From (symbol, [x, y, z]) pairs; the symbol's case is normalised."""
        return cls(
            tuple(
                (s.strip().capitalize(), (float(p[0]), float(p[1]), float(p[2]))) for s, p in atoms
            ),
            **kwargs,
        )

    @classmethod
    def from_xyz(cls, text: str, **kwargs: Any) -> MoleculeSpec:
        """From text with one `symbol x y z` line per atom (lines of another shape are skipped)."""
        atoms = [
            (parts[0], [float(v) for v in parts[1:]])
            for parts in (line.split() for line in text.strip().splitlines())
            if len(parts) == 4
        ]
        return cls.from_atoms(atoms, **kwargs)

    @property
    def n_electrons(self) -> int:
        return sum(ATOMIC_NUMBER[s] for s, _ in self.atoms) - self.charge


@dataclass(frozen=True)
class ActiveSpace:
    """`n_electrons` electrons in `n_orbitals` spatial orbitals (so 2 x n_orbitals qubits).

    By default the orbitals nearest the HOMO-LUMO gap. With `center_atoms` (0-based atom indices)
    the valence orbitals with the most weight on those atoms instead, so that related systems such
    as a complex and its fragments correlate the same orbitals; core orbitals are never chosen.
    `exclude_atoms`: never take virtual orbitals that sit mainly on these atoms (the empty
    orbitals of a bound metal ion would otherwise crowd out the ligand's).
    """

    n_electrons: int
    n_orbitals: int
    center_atoms: tuple[int, ...] | None = None
    exclude_atoms: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.n_electrons < 0 or not 1 <= self.n_orbitals <= 12:
            raise ValueError("an active space needs 0 or more electrons and 1 to 12 orbitals")
        if self.center_atoms is not None and not self.center_atoms:
            raise ValueError("center_atoms must name at least one atom")

    def as_dict(self) -> dict[str, int]:
        return {"n_electrons": self.n_electrons, "n_orbitals": self.n_orbitals}


@dataclass
class ChemProblem:
    """One system's qubit Hamiltonian with the classical energies computed while building it.

    The full-space references (MP2, CCSD(T)) are computed on first use: a price quote does not need
    them, and CCSD(T) can cost more than everything else put together."""

    op: SparsePauliOp  # real coefficients, Qiskit order (rightmost character is qubit 0)
    n_spatial: int
    n_alpha: int
    n_beta: int
    e_hf: float
    e_casci: float  # exact (FCI) energy inside the active space
    metadata: dict[str, Any]  # basis, active space, number of molecular orbitals
    _references: Callable[[], dict[str, float | None]] | None = field(default=None, repr=False)
    _computed: dict[str, float | None] | None = field(default=None, repr=False)

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_spatial

    def _reference(self, key: str) -> float | None:
        if self._computed is None:
            self._computed = self._references() if self._references else {}
            self._references = None  # releases the SCF object it holds
        return self._computed.get(key)

    @property
    def e_mp2(self) -> float | None:
        """Full-space MP2: a cheap check on what the active space leaves out."""
        return self._reference("mp2")

    @property
    def e_ccsd_t(self) -> float | None:
        """Full-space CCSD(T), when the molecule is small enough."""
        return self._reference("ccsd_t")


@dataclass
class ChemSet:
    """Systems whose energies are combined linearly: result = sum(coefficients[k] * E[k]).

    A single molecule is `{"mol": 1}`; a binding energy is
    `{"complex": 1, "host": -1, "guest": -1}`; a reaction energy is products (+1) minus
    reactants (-1)."""

    systems: dict[str, ChemProblem]
    coefficients: dict[str, float]
    label: str = "energy"

    def __post_init__(self) -> None:
        if set(self.systems) != set(self.coefficients):
            raise ValueError("systems and coefficients must have the same names")

    @property
    def n_qubits(self) -> int:
        return max(s.n_qubits for s in self.systems.values())

    def combine(self, energies: dict[str, float]) -> float:
        return float(sum(self.coefficients[k] * energies[k] for k in self.systems))

    def reference(self, kind: str) -> float | None:
        """The combined classical energy `kind` ("hf", "casci", "mp2", "ccsd_t"), or None when it
        was not computed for every system."""
        values = {k: getattr(s, f"e_{kind}") for k, s in self.systems.items()}
        return None if any(v is None for v in values.values()) else self.combine(values)


# ---- Jordan-Wigner ---------------------------------------------------------------------------


def ladder_operators(n_modes: int) -> tuple[list[SparsePauliOp], list[SparsePauliOp]]:
    """Jordan-Wigner annihilation and creation operators of `n_modes` spin orbitals."""
    ann, cre = [], []
    for j in range(n_modes):
        idx = [*range(j), j]
        zs = "Z" * j
        ann.append(
            SparsePauliOp.from_sparse_list([(zs + "X", idx, 0.5), (zs + "Y", idx, 0.5j)], n_modes)
        )
        cre.append(
            SparsePauliOp.from_sparse_list([(zs + "X", idx, 0.5), (zs + "Y", idx, -0.5j)], n_modes)
        )
    return ann, cre


def jordan_wigner_hamiltonian(
    h1: np.ndarray, eri: np.ndarray, e_const: float, tol: float = 1e-10
) -> SparsePauliOp:
    """H = e_const + sum_pq h_pq a+_p a_q + 1/2 sum (pq|rs) a+_p a+_r a_s a_q.

    `eri` is the full four-index array in chemists' notation (pq|rs)."""
    n = h1.shape[0]
    m = 2 * n
    ann, cre = ladder_operators(m)
    pair_cc: dict[tuple[int, int], SparsePauliOp] = {}
    pair_aa: dict[tuple[int, int], SparsePauliOp] = {}

    def cc(a: int, b: int) -> SparsePauliOp:
        if (a, b) not in pair_cc:
            pair_cc[(a, b)] = cre[a].dot(cre[b]).simplify()
        return pair_cc[(a, b)]

    def aa(a: int, b: int) -> SparsePauliOp:
        if (a, b) not in pair_aa:
            pair_aa[(a, b)] = ann[a].dot(ann[b]).simplify()
        return pair_aa[(a, b)]

    terms: list[SparsePauliOp] = [SparsePauliOp.from_list([("I" * m, e_const)])]
    for p in range(n):
        for q in range(n):
            if abs(h1[p, q]) < tol:
                continue
            for s in (0, n):
                terms.append(h1[p, q] * cre[p + s].dot(ann[q + s]))
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for t in range(n):
                    g = eri[p, q, r, t]
                    if abs(g) < tol:
                        continue
                    for s1 in (0, n):
                        for s2 in (0, n):
                            if s1 == s2 and (p == r or q == t):
                                continue  # Pauli exclusion: the operator vanishes
                            terms.append(0.5 * g * cc(p + s1, r + s2).dot(aa(t + s2, q + s1)))
        if len(terms) > 4000:
            terms = [SparsePauliOp.sum(terms).simplify(atol=tol)]
    return _canonical(SparsePauliOp.sum(terms).simplify(atol=tol))


def _canonical(op: SparsePauliOp, decimals: int = 10, drop: float = 5e-10) -> SparsePauliOp:
    """A bit-reproducible form: real coefficients on a fixed grid, tiny terms dropped, terms
    sorted. Floating-point noise of about 1e-13 (multithreaded BLAS) would otherwise change which
    terms are pruned, how they are grouped and so the sampled results between runs."""
    items = sorted(
        (label, round(float(np.real(c)), decimals))
        for label, c in zip(op.paulis.to_labels(), op.coeffs, strict=True)
        if abs(float(np.real(c))) > drop
    )
    return SparsePauliOp.from_list(items)


# ---- from a molecule -------------------------------------------------------------------------


def pyscf_memory_mb() -> int:
    """Memory PySCF may plan for: 60% of the sandbox limit (QSOLVE_MEMORY_MB), so it switches to
    its slower out-of-core algorithms instead of being killed; PySCF's own default otherwise."""
    raw = os.environ.get("QSOLVE_MEMORY_MB", "")
    return max(256, int(int(raw) * 0.6)) if raw.isdigit() else 4000


@lru_cache(maxsize=16)
def build_hamiltonian(
    mol: MoleculeSpec,
    active: ActiveSpace | None = None,
    *,
    with_ccsd_t: bool = True,
    density_fit: bool = False,
) -> ChemProblem:
    """SCF, active-space selection and the qubit mapping of a closed-shell system.

    `density_fit` approximates the two-electron integrals with an auxiliary basis. Near 100 basis
    functions it halves the build time and moves energy differences by about 0.01 kcal/mol (the
    fitting error cancels); the Hamiltonian and every reference use the same integrals, so the
    comparisons stay exact. Cached: the arguments are hashable and the build is deterministic."""
    try:
        from pyscf import ao2mo, cc, gto, mcscf, scf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PySCF is needed (Linux, macOS or WSL): pip install pyscf") from e

    n_total = mol.n_electrons
    if n_total < 0 or n_total % 2:
        raise ValueError("only closed-shell systems are supported (an even electron count)")
    pm = gto.M(
        atom=[(s, p) for s, p in mol.atoms], basis=mol.basis, charge=mol.charge, spin=0,
        unit="Angstrom", verbose=0, max_memory=pyscf_memory_mb(),
    )  # fmt: skip
    if pm.nao > MAX_BASIS_FUNCTIONS:
        raise ValueError(
            f"this molecule has {pm.nao} basis functions in {mol.basis}; at most "
            f"{MAX_BASIS_FUNCTIONS} are supported. Use a smaller basis set or a smaller molecule"
        )
    mf = scf.RHF(pm)
    if density_fit:
        mf = mf.density_fit()
    mf.kernel()
    if not mf.converged:
        raise ValueError("SCF did not converge for this geometry and basis")

    n_mo = pm.nao
    if active is None:
        active = ActiveSpace(n_electrons=min(n_total, 4), n_orbitals=min(n_mo, 4))
    if active.n_electrons > n_total or active.n_electrons % 2:
        raise ValueError("active n_electrons must be even and no more than the total")
    if active.n_orbitals > n_mo:
        raise ValueError(
            f"active n_orbitals {active.n_orbitals} exceeds the {n_mo} orbitals in this basis"
        )
    if active.n_electrons > 2 * active.n_orbitals:
        raise ValueError("the active space cannot hold that many electrons")
    n_core = (n_total - active.n_electrons) // 2
    if n_core + active.n_orbitals > n_mo:
        raise ValueError(
            f"{n_core} core orbitals plus {active.n_orbitals} active orbitals exceed the {n_mo} "
            f"orbitals in this basis; use fewer active orbitals or more active electrons"
        )

    mc = mcscf.CASCI(mf, active.n_orbitals, active.n_electrons)
    if active.center_atoms is not None:
        mc.mo_coeff = mc.sort_mo(_site_orbitals(mf, active), base=0)
    e_casci = float(mc.kernel(mc.mo_coeff)[0])
    h1eff, e_core = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), active.n_orbitals)
    op = jordan_wigner_hamiltonian(np.asarray(h1eff), np.asarray(eri), float(e_core))

    # The references keep the SCF object but not its stored integrals (N^4 memory each, which
    # several systems would not fit); MP2 and CCSD transform what they need again.
    mf._eri = None
    want_cc = with_ccsd_t and n_mo <= CCSD_T_MAX_BASIS_FUNCTIONS

    def references() -> dict[str, float | None]:
        from pyscf import mp

        out: dict[str, float | None] = {"mp2": None, "ccsd_t": None}
        if n_total == 0:
            return out
        with contextlib.suppress(Exception):  # a reference only: never fail a run over it
            out["mp2"] = float(mp.MP2(mf).run().e_tot)
        if want_cc:
            with contextlib.suppress(Exception):
                ccs = cc.CCSD(mf).run()
                out["ccsd_t"] = float(ccs.e_tot + ccs.ccsd_t()) if ccs.converged else None
        return out

    return ChemProblem(
        op=SparsePauliOp(op.paulis, np.real(op.coeffs)),
        n_spatial=active.n_orbitals,
        n_alpha=active.n_electrons // 2,
        n_beta=active.n_electrons // 2,
        e_hf=float(mf.e_tot),
        e_casci=e_casci,
        metadata={"basis": mol.basis, "active": active.as_dict(), "n_mo": n_mo},
        _references=references,
    )


def _site_orbitals(mf: Any, active: ActiveSpace) -> list[int]:
    """0-based indices of the active orbitals centred on `active.center_atoms` (Mulliken weight):
    the valence occupied and the virtual orbitals that sit most on those atoms."""
    from pyscf.data import elements

    mol = mf.mol
    centre, exclude = list(active.center_atoms or ()), list(active.exclude_atoms)
    for a in [*centre, *exclude]:
        if not 0 <= a < mol.natm:
            raise ValueError(f"active-space atom index {a} is outside this molecule's {mol.natm}")
    c = np.asarray(mf.mo_coeff)
    sc = np.asarray(mf.get_ovlp()) @ c
    slices = mol.aoslice_by_atom()

    def weight(atoms: list[int]) -> np.ndarray:
        if not atoms:
            return np.zeros(c.shape[1])
        idx = np.concatenate([np.arange(slices[a][2], slices[a][3]) for a in atoms])
        return np.asarray(np.einsum("ik,ik->k", c[idx], sc[idx]))

    w, w_excluded = weight(centre), weight(exclude)
    n_occ, n_occ_active = mol.nelectron // 2, active.n_electrons // 2
    valence = range(elements.chemcore(mol), n_occ)
    if len(valence) < n_occ_active:
        raise ValueError(
            f"only {len(valence)} valence orbitals are occupied; use fewer active electrons"
        )
    occupied = sorted(valence, key=lambda k: -w[k])[:n_occ_active]
    empty = [k for k in range(n_occ, c.shape[1]) if w_excluded[k] <= 0.5]
    virtual = sorted(empty, key=lambda k: -w[k])[: active.n_orbitals - n_occ_active]
    return sorted(occupied + virtual)
