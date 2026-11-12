"""Error mitigation for energy measurements. Every step is reported, none is hidden.

- Shot allocation: groups whose terms carry more weight get more circuit copies (each copy is a
  full set of shots), so the error bar shrinks for the same total. Weights come from the
  Hamiltonian alone, so a quote and the run agree exactly on what is sent.
- Readout correction: invert each measured bit's calibrated assignment matrix (per-qubit
  P(1|0), P(0|1), reported by the backend for the physical qubit that bit came from).
- Symmetry post-selection: the Hamiltonian conserves the electron count of each spin. In a group
  measured in the Z basis on all of one spin's qubits, a shot with the wrong count is known to be
  wrong and is dropped (Bonet-Monroig et al. 2018). The kept fraction is reported.
- Zero-noise extrapolation: run the state again with every two-qubit gate tripled (it does the
  same thing with three times the gate noise) and extrapolate the energy back to no noise.
- Error bars by bootstrap over the measured counts, so they include what mitigation does to the
  variance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

DENSE_MAX_QUBITS = 16  # readout correction works on the full 2^n distribution


@dataclass(frozen=True)
class SpinSectors:
    """Which qubits hold each spin's orbitals, and how many electrons each must hold."""

    alpha: tuple[int, ...]
    n_alpha: int
    beta: tuple[int, ...]
    n_beta: int

    @classmethod
    def jordan_wigner(cls, n_spatial: int, n_alpha: int, n_beta: int) -> SpinSectors:
        return cls(tuple(range(n_spatial)), n_alpha, tuple(range(n_spatial, 2 * n_spatial)), n_beta)


# ---- shot allocation ---------------------------------------------------------------------------


def group_weights(groups: Sequence[SparsePauliOp]) -> list[float]:
    """sqrt(sum of squared coefficients): a group's standard deviation per shot when its terms
    are uncorrelated. A Hamiltonian-only stand-in for the state-dependent spread."""
    return [float(np.sqrt(np.sum(np.abs(g.coeffs) ** 2))) for g in groups]


def group_spreads(groups: Sequence[SparsePauliOp], psi: NDArray[np.complex128]) -> list[float]:
    """Per-shot standard deviation of each group's value in state `psi` (known classically: the
    parameters were found noise-free). Splitting shots by it changes the error bar, not what is
    estimated, so it adds no bias."""
    out = []
    for g in groups:
        v = g.to_matrix(sparse=True) @ psi
        mean = float(np.real(np.vdot(psi, v)))
        out.append(float(np.sqrt(max(float(np.real(np.vdot(v, v))) - mean**2, 0.0))))
    return out


def allocate_copies(weights: Sequence[float], total: int) -> list[int]:
    """Exactly `total` circuit copies, at least one per group, otherwise in proportion to weight
    (largest-remainder rounding). The optimal split of shots is proportional to each group's
    spread; a fixed total lets a quote count the circuits without knowing the weights."""
    w = np.asarray(weights, dtype=float)
    g = len(w)
    if g == 0:
        return []
    if total < g:
        raise ValueError("need at least one copy per group")
    if w.sum() <= 0:
        w = np.ones(g)
    spare = total - g  # every group has its one copy; the rest follow the weights
    ideal = spare * w / w.sum()
    extra = np.floor(ideal).astype(int)
    for k in np.argsort(-(ideal - extra), kind="stable")[: spare - int(extra.sum())]:
        extra[k] += 1
    return [1 + int(e) for e in extra]


# ---- readout correction and post-selection -----------------------------------------------------


def bit_table(n: int) -> NDArray[np.int8]:
    """Row `s` holds the bits of basis state `s`; column `q` is qubit `q`."""
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def counts_to_probs(counts: dict[str, int], n: int) -> NDArray[np.float64]:
    p = np.zeros(2**n)
    total = 0
    for k, c in counts.items():
        p[int(k.replace(" ", ""), 2)] += c
        total += c
    return p / max(total, 1)


def correct_readout(
    probs: NDArray[np.float64], errors: Sequence[Sequence[float]]
) -> NDArray[np.float64]:
    """Apply the inverse of the tensor-product assignment matrix. errors[q] = (P(1|0), P(0|1)) for
    bit q. The result is a quasi-distribution (it may have small negative entries); expectation
    values computed from it are unbiased."""
    n = len(errors)
    t = probs.reshape([2] * n)  # axis 0 is the most significant bit: qubit n-1
    for q, (p10, p01) in enumerate(errors):
        a = np.array([[1.0 - p10, p01], [p10, 1.0 - p01]])
        axis = n - 1 - q
        t = np.moveaxis(np.tensordot(np.linalg.inv(a), t, axes=([1], [axis])), 0, axis)
    return t.reshape(-1)


def symmetry_mask(
    basis: Sequence[str], sectors: SpinSectors | None, bits: NDArray[np.int8]
) -> NDArray[np.bool_] | None:
    """States allowed by the spin sectors this group's basis can check; None if it checks none.
    A sector is checkable when none of its qubits is measured in the X or Y basis."""
    if sectors is None:
        return None
    ok = np.ones(len(bits), dtype=bool)
    checked = False
    for qubits, count in ((sectors.alpha, sectors.n_alpha), (sectors.beta, sectors.n_beta)):
        if all(basis[q] in "IZ" for q in qubits):
            ok &= bits[:, list(qubits)].sum(axis=1) == count
            checked = True
    return ok if checked else None


def group_value(probs: NDArray[np.float64], group: SparsePauliOp, bits: NDArray[np.int8]) -> float:
    """Expectation of a qubit-wise commuting group in the distribution `probs`."""
    n = group.num_qubits
    value = 0.0
    for label, coef in zip(group.paulis.to_labels(), np.real(group.coeffs), strict=True):
        support = [q for q in range(n) if label[n - 1 - q] != "I"]
        parity = bits[:, support].sum(axis=1) % 2
        value += float(coef) * float(np.dot(probs, 1 - 2 * parity))
    return value


@dataclass
class GroupOutcome:
    raw: float
    mitigated: float
    kept: float  # share of shots kept by post-selection (1.0 when nothing was checked)
    checked: bool  # whether the group's basis allowed an electron-count check
    raw_p: NDArray[np.float64] | None = None  # measured and mitigated distributions
    mit_p: NDArray[np.float64] | None = None  # (kept for every group, for a mitigation report)
    shots: int = 0


def mitigated_group(
    counts_per_copy: Sequence[dict[str, int]],
    readout_per_copy: Sequence[Sequence[Sequence[float]] | None],
    group: SparsePauliOp,
    basis: Sequence[str],
    sectors: SpinSectors | None,
    bits: NDArray[np.int8],
    *,
    readout: bool,
) -> GroupOutcome:
    """One group's value from the counts of all its copies, with and without mitigation."""
    n = group.num_qubits
    raw_p = np.zeros(2**n)
    mit_p = np.zeros(2**n)
    total = 0
    for counts, errors in zip(counts_per_copy, readout_per_copy, strict=True):
        shots = sum(counts.values())
        p = counts_to_probs(counts, n)
        raw_p += shots * p
        if readout and errors is not None:
            p = correct_readout(p, errors)
        mit_p += shots * p
        total += shots
    raw_p /= max(total, 1)
    mit_p /= max(total, 1)
    kept = 1.0
    mask = symmetry_mask(basis, sectors, bits)
    if mask is not None:
        kept_mass = float(mit_p[mask].sum())
        if kept_mass <= 0:
            raise ValueError("no measured shot satisfied the electron-count check")
        mit_p = np.where(mask, mit_p, 0.0) / kept_mass
        kept = float(raw_p[mask].sum())
    return GroupOutcome(
        group_value(raw_p, group, bits), group_value(mit_p, group, bits), kept, mask is not None,
        raw_p, mit_p, total,
    )  # fmt: skip


def bootstrap_stderr(
    counts_per_copy: Sequence[dict[str, int]],
    readout_per_copy: Sequence[Sequence[Sequence[float]] | None],
    group: SparsePauliOp,
    basis: Sequence[str],
    sectors: SpinSectors | None,
    bits: NDArray[np.int8],
    *,
    readout: bool,
    rng: np.random.Generator,
    samples: int = 200,
) -> float:
    """Standard error of the mitigated group value: resample each copy's shots (multinomial) and
    repeat the whole mitigation."""
    keys = [list(c) for c in counts_per_copy]
    probs = [
        np.array([c[k] for k in ks], dtype=float)
        for c, ks in zip(counts_per_copy, keys, strict=True)
    ]
    shots = [int(p.sum()) for p in probs]
    vals = []
    for _ in range(samples):
        resampled = []
        for ks, p, s in zip(keys, probs, shots, strict=True):
            draw = rng.multinomial(s, p / p.sum())
            resampled.append(dict(zip(ks, (int(x) for x in draw), strict=True)))
        try:
            out = mitigated_group(
                resampled, readout_per_copy, group, basis, sectors, bits, readout=readout
            )
        except ValueError:
            continue
        vals.append(out.mitigated)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


# ---- what noise does to an energy --------------------------------------------------------------


def diagonal_energy(op: SparsePauliOp, index: int) -> float:
    """<b|H|b> for computational basis state `index`: only I/Z terms contribute."""
    n = op.num_qubits
    e = 0.0
    for label, coef in zip(op.paulis.to_labels(), np.real(op.coeffs), strict=True):
        if any(ch in "XY" for ch in label):
            continue
        mask = sum(1 << q for q in range(n) if label[n - 1 - q] == "Z")
        e += float(coef) * (-1.0) ** bin(index & mask).count("1")
    return e


def sector_average(op: SparsePauliOp, sectors: SpinSectors) -> float:
    """Tr(P H) / Tr(P) over the determinants with the right electron count in each spin: the
    energy that noise keeping the electron count drives any state towards."""
    total, count = 0.0, 0
    for a in combinations(sectors.alpha, sectors.n_alpha):
        for b in combinations(sectors.beta, sectors.n_beta):
            total += diagonal_energy(op, sum(1 << q for q in (*a, *b)))
            count += 1
    return total / count


# ---- zero-noise extrapolation ------------------------------------------------------------------


def fold_two_qubit_gates(circuit: QuantumCircuit, factor: int) -> QuantumCircuit:
    """The same circuit with each two-qubit gate G replaced by G (G^-1 G)^((factor-1)/2): it does
    the same thing with `factor` times the two-qubit noise. Barriers keep the transpiler from
    cancelling the repeats."""
    if factor < 1 or factor % 2 == 0:
        raise ValueError("the fold factor must be odd")
    out = circuit.copy_empty_like()
    for inst in circuit.data:
        out.append(inst)
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier":
            for _ in range((factor - 1) // 2):
                out.barrier(inst.qubits)
                out.append(inst.operation.inverse(), inst.qubits)
                out.barrier(inst.qubits)
                out.append(inst)
    return out


def extrapolate_to_zero_noise(e1: float, s1: float, e3: float, s3: float) -> tuple[float, float]:
    """Linear (Richardson) extrapolation from noise factors 1 and 3 to 0: E = 1.5 E1 - 0.5 E3."""
    return 1.5 * e1 - 0.5 * e3, float(np.hypot(1.5 * s1, 0.5 * s3))
