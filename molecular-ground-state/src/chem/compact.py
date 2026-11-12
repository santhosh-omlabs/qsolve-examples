"""Compact state preparation for quantum hardware.

Parameters are found noise-free before any hardware call, so the device only has to *prepare a
known state*. This module builds that state with as few two-qubit gates as it can:

1. ADAPT selection (Grimsley et al. 2019, with qubit excitations as in Yordanov et al. 2021): grow
   the ansatz one excitation at a time, always the one with the largest energy gradient, until the
   energy is within `tol` of the exact active-space energy. Only the excitations the state needs
   are compiled (H2: 1; the electrolyte default (4e,4o): a handful), instead of every UCCSD term.
2. Each qubit excitation is a Givens rotation between two basis states: CNOTs bring the two states
   to differ in a single qubit, a controlled Ry rotates it, the CNOTs are undone. A double costs
   6 CNOTs plus an Ry with three controls (8 CNOTs), against ~44 for a Pauli-gadget circuit.
3. Known-state simplification: the circuit starts from a computational basis state, so every gate
   whose controls are still classical is resolved at compile time (a CNOT with a known control is
   an X or nothing; a controlled Ry with known controls is an Ry or nothing). The first excitation
   from Hartree-Fock costs 3 CNOTs instead of 14.

Qubit excitations differ from fermionic ones only by signs from the Jordan-Wigner strings; the
angles are optimised against the true qubit Hamiltonian, so the state they prepare is exact for it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

Excitation = tuple[tuple[int, ...], tuple[int, ...]]  # (occupied qubits, virtual qubits)
Gate = tuple  # ("x", q) | ("cx", c, t) | ("mcry", ((control, value), ...), target, angle)


def hf_qubits(n_spatial: int, n_alpha: int, n_beta: int) -> list[int]:
    """Occupied qubits of the Hartree-Fock state (alpha block first, as the Hamiltonian maps)."""
    return list(range(n_alpha)) + [n_spatial + i for i in range(n_beta)]


def excitation_pool(n_spatial: int, n_alpha: int, n_beta: int) -> list[Excitation]:
    """Spin-conserving single and double qubit excitations from occupied to virtual orbitals."""
    occ_a, vir_a = list(range(n_alpha)), list(range(n_alpha, n_spatial))
    occ_b = [n_spatial + i for i in range(n_beta)]
    vir_b = [n_spatial + i for i in range(n_beta, n_spatial)]
    pool: list[Excitation] = []
    for occ, vir in ((occ_a, vir_a), (occ_b, vir_b)):
        pool += [((i,), (a,)) for i in occ for a in vir]
    for occ, vir in ((occ_a, vir_a), (occ_b, vir_b)):
        pool += [(o, v) for o in combinations(occ, 2) for v in combinations(vir, 2)]
    pool += [((i, j), (a, b)) for i in occ_a for j in occ_b for a in vir_a for b in vir_b]
    return pool


class _Rotations:
    """Statevector action of qubit excitations: a rotation between pairs of basis states."""

    def __init__(self, n_qubits: int) -> None:
        self.idx = np.arange(2**n_qubits)
        self._cache: dict[Excitation, tuple[NDArray[np.int64], NDArray[np.int64]]] = {}

    def pairs(self, ex: Excitation) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(s, t): every occupied qubit set and every virtual one empty; and their images."""
        if ex not in self._cache:
            m_occ = sum(1 << q for q in ex[0])
            m_vir = sum(1 << q for q in ex[1])
            s = self.idx[((self.idx & m_occ) == m_occ) & ((self.idx & m_vir) == 0)]
            self._cache[ex] = (s, s ^ m_occ ^ m_vir)
        return self._cache[ex]

    def apply(self, psi: NDArray[np.float64], ex: Excitation, theta: float) -> NDArray[np.float64]:
        """exp(theta G): |occ> -> cos|occ> + sin|vir>, |vir> -> -sin|occ> + cos|vir>."""
        s, t = self.pairs(ex)
        out = psi.copy()
        c, sn = np.cos(theta), np.sin(theta)
        out[s] = c * psi[s] - sn * psi[t]
        out[t] = sn * psi[s] + c * psi[t]
        return out

    def generator(self, psi: NDArray[np.float64], ex: Excitation) -> NDArray[np.float64]:
        """G psi, the derivative of `apply` at theta = 0."""
        s, t = self.pairs(ex)
        out = np.zeros_like(psi)
        out[s] = -psi[t]
        out[t] = psi[s]
        return out


@dataclass
class CompactState:
    """The excitations chosen, their angles, and the energy they reach (noise-free)."""

    excitations: list[Excitation]
    angles: list[float]
    energy: float
    exact: float
    hf: list[int]
    n_qubits: int
    history: list[float] = field(default_factory=list)  # energy after each step (0 = HF)
    step_angles: list[list[float]] = field(default_factory=list)  # optimal angles at each step

    def truncated(self, k: int) -> CompactState:
        """The state after the first k ADAPT steps (each step re-optimises all its angles, so
        these are that step's angles, not a prefix of the final ones)."""
        if not 0 <= k < len(self.history):
            raise ValueError(f"no step {k}")
        return CompactState(
            self.excitations[:k], self.step_angles[k], self.history[k], self.exact, self.hf,
            self.n_qubits, self.history[: k + 1], self.step_angles[: k + 1],
        )  # fmt: skip

    def two_qubit_gates(self) -> int:
        ops = self.circuit().count_ops()
        return int(ops.get("cx", 0) + 2 * ops.get("cry", 0))

    def choose_for_noise(
        self,
        noise_per_gate: float,
        energy_scale: float,
        *,
        corrected: Callable[[CompactState], bool] | None = None,
        corrected_residual: float = 1.0,
    ) -> tuple[int, float]:
        """The step with the smallest predicted error on a noisy device: what stopping there gives
        up (exact, from ADAPT's history) plus noise_per_gate x two-qubit gates x energy_scale.

        energy_scale is where noise drives the state: the sector-average energy minus the HF
        energy (Ha). Each two-qubit gate then shifts the energy by about its error rate times that
        scale (fitted on ibm_fez noise models: 0.2-0.4% per gate, about the median CZ error).
        States for which `corrected` holds get a further correction (zero-noise extrapolation)
        that leaves `corrected_residual` of that noise. Returns (steps, predicted error in Ha)."""
        best: tuple[float, int] = (float("inf"), 0)
        for k in range(len(self.history)):
            t = self.truncated(k)
            noise = noise_per_gate * t.two_qubit_gates() * energy_scale
            if corrected is not None and corrected(t):
                noise *= corrected_residual
            best = min(best, (self.history[k] - self.exact + noise, k))
        return best[1], best[0]

    def circuit(self) -> QuantumCircuit:
        return state_circuit(self.n_qubits, self.hf, self.excitations, self.angles)


def adapt_state(
    hamiltonian: csr_matrix,
    n_spatial: int,
    n_alpha: int,
    n_beta: int,
    exact: float,
    *,
    tol: float = 1e-4,
    max_excitations: int = 40,
    min_gradient: float = 1e-6,
) -> CompactState:
    """Grow an excitation ansatz from Hartree-Fock until it is within `tol` (Ha) of `exact`.

    `hamiltonian` is the sparse matrix of the qubit Hamiltonian (2^n x 2^n, real). Deterministic:
    the selection and the optimiser use no randomness."""
    n = 2 * n_spatial
    rot = _Rotations(n)
    hf = hf_qubits(n_spatial, n_alpha, n_beta)
    psi0 = np.zeros(2**n)
    psi0[sum(1 << q for q in hf)] = 1.0
    h = hamiltonian.real.tocsr() if np.iscomplexobj(hamiltonian.data) else hamiltonian.tocsr()
    pool = excitation_pool(n_spatial, n_alpha, n_beta)

    def state(ops: list[Excitation], th: NDArray[np.float64]) -> NDArray[np.float64]:
        psi = psi0
        for ex, t in zip(ops, th, strict=True):
            psi = rot.apply(psi, ex, float(t))
        return psi

    def energy_and_grad(
        th: NDArray[np.float64], ops: list[Excitation]
    ) -> tuple[float, NDArray[np.float64]]:
        psi = state(ops, th)
        lam = h @ psi
        e = float(psi @ lam)
        grad = np.zeros(len(ops))
        for k in range(len(ops) - 1, -1, -1):  # adjoint sweep: one pass back through the ansatz
            grad[k] = 2.0 * float(lam @ rot.generator(psi, ops[k]))
            psi = rot.apply(psi, ops[k], -float(th[k]))
            lam = rot.apply(lam, ops[k], -float(th[k]))
        return e, grad

    ops: list[Excitation] = []
    theta = np.zeros(0)
    e = float(psi0 @ (h @ psi0))
    history = [e]
    steps: list[list[float]] = [[]]
    while e - exact > tol and len(ops) < max_excitations:
        psi = state(ops, theta)
        hpsi = h @ psi
        grads = np.array([2.0 * float(hpsi @ rot.generator(psi, ex)) for ex in pool])
        k = int(np.argmax(np.abs(grads)))
        if abs(grads[k]) < min_gradient:
            break
        ops.append(pool[k])
        res = minimize(
            energy_and_grad, np.append(theta, 0.0), args=(ops,), jac=True, method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-14, "gtol": 1e-9},
        )  # fmt: skip
        theta, e = res.x, float(res.fun)
        history.append(e)
        steps.append([float(t) for t in theta])
    return CompactState(ops, [float(t) for t in theta], e, exact, hf, n, history, steps)


# ---- circuit construction ---------------------------------------------------------------------


def _givens_gates(ex: Excitation, theta: float) -> list[Gate]:
    """CNOTs map |occ> and |vir> onto two states that differ only in the first virtual qubit;
    an Ry on it, controlled on the values the other involved qubits then hold, rotates between
    them; the CNOTs are undone. Exact on every input: the control pattern matches only those two
    states."""
    occ, vir = ex
    pivot = vir[0]
    fwd: list[tuple[int, int]] = []
    # inside each group, chain the qubits so all but the first read 0 on both states
    for group in (occ, vir):
        fwd += list(zip(group, group[1:], strict=False))
    fwd.append((pivot, occ[0]))  # |occ> now has occ[0]=1, pivot=0; |vir> has occ[0]=1, pivot=1
    controls = ((occ[0], 1),) + tuple((q, 0) for q in (*occ[1:], *vir[1:]))
    gates: list[Gate] = [("cx", c, t) for c, t in fwd]
    gates.append(("mcry", controls, pivot, 2.0 * theta))
    gates += [("cx", c, t) for c, t in reversed(fwd)]
    return gates


def _simplify(n: int, hf: list[int], gates: list[Gate]) -> list[Gate]:
    """Replace gates whose controls are classical at that point by their cheaper equivalents.

    The circuit starts with X on the Hartree-Fock qubits, so every qubit physically holds the value
    tracked here until a quantum gate touches it: a CNOT with a known control becomes an X on its
    target or nothing, and a controlled Ry drops its known controls (or the whole gate, when one
    of them has the wrong value)."""
    known: dict[int, int] = {q: 0 for q in range(n)}
    out: list[Gate] = []
    for q in hf:
        known[q] = 1
        out.append(("x", q))
    for g in gates:
        if g[0] == "cx":
            _, c, t = g
            if c in known:
                if known[c] == 1:
                    out.append(("x", t))
                    if t in known:
                        known[t] ^= 1
                continue
            known.pop(t, None)
            out.append(g)
        else:  # mcry
            _, controls, t, phi = g
            if any(q in known and known[q] != v for q, v in controls) or abs(phi) < 1e-12:
                continue
            known.pop(t, None)
            out.append(("mcry", tuple((q, v) for q, v in controls if q not in known), t, phi))
    return out


def state_circuit(
    n_qubits: int, hf: list[int], excitations: list[Excitation], angles: list[float]
) -> QuantumCircuit:
    """The compact circuit preparing exp(theta_k G_k) ... exp(theta_1 G_1)|HF>."""
    gates: list[Gate] = []
    for ex, th in zip(excitations, angles, strict=True):
        gates += _givens_gates(ex, th)
    qc = QuantumCircuit(n_qubits)
    for g in _simplify(n_qubits, hf, gates):
        if g[0] == "x":
            qc.x(g[1])
        elif g[0] == "cx":
            qc.cx(g[1], g[2])
        else:
            _, controls, t, phi = g
            if not controls:
                qc.ry(phi, t)
            elif len(controls) == 1:
                ((c, v),) = controls
                if v == 0:
                    qc.x(c)
                qc.cry(phi, c, t)
                if v == 0:
                    qc.x(c)
            else:
                # an Ry on the target applied only for the matching control pattern
                pattern = sum(v << i for i, (_q, v) in enumerate(controls))
                alphas = [0.0] * (2 ** len(controls))
                alphas[pattern] = phi
                _multiplexed_ry(qc, alphas, t, [q for q, _v in controls])
    return qc


def _multiplexed_ry(
    qc: QuantumCircuit, alphas: list[float], target: int, controls: list[int]
) -> None:
    """Ry(alphas[j]) on `target` when the controls read j (bit b = controls[b]): 2^k Ry and 2^k
    CNOTs in Gray-code order (Mottonen et al., PRL 93, 130502, 2004), standard gates only."""
    size = 2 ** len(controls)

    def gray(i: int) -> int:
        return i ^ (i >> 1)

    for i in range(size):
        theta = sum(
            a * (-1.0) ** bin(j & gray(i)).count("1") for j, a in enumerate(alphas)
        ) / size  # fmt: skip
        if abs(theta) > 1e-15:
            qc.ry(theta, target)
        changed = gray(i) ^ gray((i + 1) % size)
        qc.cx(controls[changed.bit_length() - 1], target)
