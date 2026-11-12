"""A QUBO and its Ising form."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Ising:
    """E(z) = offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j with z_i = +1 or -1."""

    h: FloatArray
    j: dict[tuple[int, int], float]
    offset: float

    @property
    def n(self) -> int:
        return int(self.h.shape[0])


@dataclass
class Qubo:
    """Minimise x'Qx + offset over x in {0,1}^n (x_0 is the first variable).

    `ones` says a valid answer has exactly that many ones. By default the solvers enforce it
    structurally (the circuit starts in and never leaves that subspace, classical search only visits
    it). With `soft=True` the penalty already inside `q` is what enforces it: the circuit explores
    every state and only the answer is checked.
    """

    q: FloatArray
    offset: float = 0.0
    ones: int | None = None
    soft: bool = False

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=np.float64)
        if self.q.ndim != 2 or self.q.shape[0] != self.q.shape[1]:
            raise ValueError("Q must be a square matrix")
        if not np.all(np.isfinite(self.q)):
            raise ValueError("Q contains non-finite values")
        if self.ones is not None and not 0 <= self.ones <= self.n:
            raise ValueError("the number of ones must be between 0 and the number of variables")

    @property
    def n(self) -> int:
        return int(self.q.shape[0])

    @property
    def cardinality(self) -> int | None:
        """The number of ones the circuit itself enforces (None: it explores every state)."""
        return None if self.soft else self.ones

    def energy(self, x: Any) -> float:
        v = np.asarray(x, dtype=np.float64)
        return float(v @ self.q @ v + self.offset)

    def energies(self, states: Any) -> FloatArray:
        """Energies of many states at once, one per row."""
        m = np.asarray(states, dtype=np.float64)
        out: FloatArray = ((m @ self.q) * m).sum(axis=1) + self.offset
        return out

    def evaluate(self, x: Any) -> float:
        """The energy of an answer; ValueError when it is not a valid answer."""
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self.n,) or not np.isin(v, (0.0, 1.0)).all():
            raise ValueError(f"an answer is {self.n} values, each 0 or 1")
        if self.ones is not None and int(v.sum()) != self.ones:
            raise ValueError(f"exactly {self.ones} variables must be 1, got {int(v.sum())}")
        return self.energy(v)

    def ising(self) -> Ising:
        """Substitute x_i = (1 - z_i) / 2, so a measured 1 means z = -1."""
        diag = np.diag(self.q).copy()
        a = self.q + self.q.T
        np.fill_diagonal(a, 0.0)
        h = -diag / 2.0 - a.sum(axis=1) / 4.0
        j = {
            (i, k): float(a[i, k] / 4.0)
            for i in range(self.n)
            for k in range(i + 1, self.n)
            if a[i, k] != 0
        }
        offset = self.offset + diag.sum() / 2.0 + np.triu(a, 1).sum() / 4.0
        return Ising(h=h, j=j, offset=float(offset))


def all_states(n: int) -> NDArray[np.int8]:
    """Every bit string as a row; bit i of the row index is variable i."""
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
