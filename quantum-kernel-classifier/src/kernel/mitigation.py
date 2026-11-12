"""Error mitigation for kernel circuits: readout correction and a depolarizing rescale.

Both work on the whole output distribution (2^n entries), so they apply while n is small.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
MAX_QUBITS = 16  # beyond this the 2^n distribution is too large to correct


def counts_to_probs(counts: dict[str, int], n: int) -> FloatArray:
    """Counts (bit strings, qubit 0 rightmost) as a probability vector indexed by the integer."""
    p = np.zeros(2**n)
    for bits, c in counts.items():
        p[int(bits.replace(" ", ""), 2)] += c
    return p / max(p.sum(), 1.0)


def correct_readout(probs: FloatArray, errors: Sequence[Sequence[float]]) -> FloatArray:
    """Undo independent readout errors. `errors[q]` is `[P(1|0), P(0|1)]` for bit q. The inverse
    of the assignment matrix can make small entries negative; expectation values taken from the
    result are unbiased, which is what a kernel entry is."""
    n = len(errors)
    t = probs.reshape([2] * n)  # axis 0 is the most significant bit, qubit n-1
    for q, (p10, p01) in enumerate(errors):
        assign = np.array([[1.0 - p10, p01], [p10, 1.0 - p01]])
        axis = n - 1 - q
        t = np.moveaxis(np.tensordot(np.linalg.inv(assign), t, axes=([1], [axis])), 0, axis)
    return t.reshape(-1)


def depolarizing_strength(mirror_p0: float, n: int) -> float:
    """Fit lambda in  measured = (1 - lambda) ideal + lambda / 2^n  from a circuit whose ideal
    P(all zeros) is 1 and which was measured as `mirror_p0`. Capped at 0.95: beyond that the
    rescale only amplifies noise."""
    c = 2.0**-n
    return float(np.clip((1.0 - mirror_p0) / (1.0 - c), 0.0, 0.95))


def undo_depolarizing(probs: FloatArray, lam: float, n: int) -> FloatArray:
    """Invert the global depolarizing channel on a distribution; clipped to a distribution."""
    out = np.clip((probs - lam * 2.0**-n) / (1.0 - lam), 0.0, None)
    return np.asarray(out / out.sum(), dtype=np.float64)


def rescale_overlap(K: FloatArray, lam: float, n: int) -> FloatArray:
    """The same inversion for kernel entries (P(all zeros) values), clipped to [0, 1]."""
    return np.asarray(np.clip((K - lam * 2.0**-n) / (1.0 - lam), 0.0, 1.0), dtype=np.float64)
