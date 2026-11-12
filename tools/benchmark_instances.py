"""Seeded optimisation benchmark instances at 16-20 variables (65k to 1M candidate answers).

Small enough that the exact optimum is still cheap to compute for the comparison, large enough that
the probability the circuit puts on the optimum (reported by every run) is a real measure.

    python tools/benchmark_instances.py      # writes <workflow>/benchmarks/inputs/*.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def regular_graph(n: int, seed: int) -> dict:
    """Random 3-regular graph (configuration model, retried until simple), weights 1-3."""
    rng = np.random.default_rng(seed)
    while True:
        stubs = np.repeat(np.arange(n), 3)
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        if all(a != b for a, b in pairs) and len({tuple(sorted(p)) for p in pairs}) == len(pairs):
            break
    nodes = [f"v{i}" for i in range(n)]
    return {"nodes": nodes, "edges": [[nodes[a], nodes[b], int(rng.integers(1, 4))] for a, b in pairs]}


def dense_qubo(n: int, seed: int) -> dict:
    """Symmetric Gaussian couplings rounded to 3 decimals: no structure for a heuristic to exploit."""
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1, (n, n))
    return {"Q": np.round((q + q.T) / 2, 3).tolist()}


def portfolio(n: int, k: int, seed: int) -> dict:
    """Three-factor covariance plus idiosyncratic risk, annual returns 3-20%."""
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.1, (n, 3))
    cov = f @ f.T + np.diag(rng.uniform(0.01, 0.04, n))
    return {
        "assets": [f"A{i}" for i in range(n)],
        "expected_returns": np.round(rng.uniform(0.03, 0.2, n), 4).tolist(),
        "covariance": np.round(cov, 5).tolist(),
        "num_to_select": k,
        "risk_aversion": 0.5,
    }


INSTANCES = {
    "max-cut": {"regular-3-weighted-18": regular_graph(18, 2), "regular-3-weighted-20": regular_graph(20, 3)},
    "generic-qubo": {"dense-random-16": dense_qubo(16, 4), "dense-random-18": dense_qubo(18, 5)},
    "portfolio-optimization": {"synthetic-16-choose-6": portfolio(16, 6, 6), "synthetic-18-choose-7": portfolio(18, 7, 7)},
}

if __name__ == "__main__":
    for workflow, instances in INSTANCES.items():
        out = ROOT / workflow / "benchmarks" / "inputs"
        out.mkdir(parents=True, exist_ok=True)
        for name, data in instances.items():
            (out / f"{name}.json").write_text(json.dumps(data) + "\n", encoding="utf-8")
            print(f"{workflow}/benchmarks/inputs/{name}.json")
