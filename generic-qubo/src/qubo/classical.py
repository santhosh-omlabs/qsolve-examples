"""Classical solvers for a QUBO: the reference a quantum result is judged against.

Both only visit valid answers (`Qubo.ones`), so a baseline never wins by breaking a constraint.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .problem import Qubo

EXACT_MAX_VARIABLES = 22  # beyond this the reference switches to annealing
_CHUNK = 1 << 18


def _states(n: int, ones: int | None) -> Iterator[NDArray[np.float64]]:
    """Valid states in blocks: all 2^n, or only the C(n, ones) that have `ones` ones."""
    if ones is None:
        shifts = np.arange(n, dtype=np.uint64)
        for start in range(0, 1 << n, min(1 << n, _CHUNK)):
            idx = np.arange(start, min(start + _CHUNK, 1 << n), dtype=np.uint64)
            yield ((idx[:, None] >> shifts[None, :]) & 1).astype(np.float64)
        return
    combos = itertools.combinations(range(n), ones)
    while block := list(itertools.islice(combos, _CHUNK)):
        x = np.zeros((len(block), n))
        x[
            np.arange(len(block))[:, None], np.array(block, dtype=np.intp).reshape(len(block), ones)
        ] = 1
        yield x


def exact_search(qubo: Qubo, max_variables: int = 26) -> tuple[list[int], float]:
    """Try every valid state: provably optimal, but 2^n work."""
    if qubo.n > max_variables:
        raise ValueError(f"exact search supports at most {max_variables} variables, got {qubo.n}")
    best_e, best_x = np.inf, np.zeros(qubo.n)
    for x in _states(qubo.n, qubo.ones):
        e = qubo.energies(x)
        i = int(np.argmin(e))
        if e[i] < best_e:
            best_e, best_x = float(e[i]), x[i]
    return [int(b) for b in best_x], best_e


def simulated_annealing(
    qubo: Qubo, seed: int, sweeps: int = 2000, restarts: int = 20
) -> tuple[list[int], float]:
    """Random restarts of single flips (or one-for-one swaps when the number of ones is fixed),
    accepting a worse state with a probability that shrinks as the temperature falls."""
    rng = np.random.default_rng(seed)
    n, k = qubo.n, qubo.ones
    scale = max(float(np.abs(qubo.q).max()), 1e-9)
    best_e, best_x = np.inf, np.zeros(n)
    for _ in range(restarts):
        if k is None:
            x = rng.integers(0, 2, n).astype(np.float64)
        else:
            x = np.zeros(n)
            x[rng.choice(n, k, replace=False)] = 1.0
        e = qubo.energy(x)
        for s in range(sweeps):
            temp = scale * max(1e-3, 2.0 * (1.0 - s / sweeps))
            for _ in range(n):
                y = x.copy()
                if k is None:
                    i = int(rng.integers(n))
                    y[i] = 1 - y[i]
                else:
                    on, off = np.flatnonzero(x == 1), np.flatnonzero(x == 0)
                    if len(on) == 0 or len(off) == 0:
                        break
                    y[rng.choice(on)] = 0
                    y[rng.choice(off)] = 1
                ey = qubo.energy(y)
                if ey <= e or rng.random() < np.exp(-(ey - e) / temp):
                    x, e = y, ey
                    if e < best_e - 1e-12:
                        best_e, best_x = e, x.copy()
    return [int(b) for b in best_x], float(qubo.energy(best_x))


def reference_search(qubo: Qubo, seed: int) -> tuple[list[int], float]:
    """Exact search while it is affordable, annealing beyond `EXACT_MAX_VARIABLES`."""
    if qubo.n <= EXACT_MAX_VARIABLES:
        return exact_search(qubo)
    return simulated_annealing(qubo, seed)
