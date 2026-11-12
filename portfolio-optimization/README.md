# Portfolio optimization

Choose exactly **k assets** from a universe of up to 24, trading expected return against risk
(a cardinality-constrained Markowitz problem), solved with QAOA and compared with the exact
classical optimum.

## Input
- `assets`: list of names
- either `expected_returns` + `covariance`, **or** `returns_history` (periods x assets)
- `num_to_select` (k), `risk_aversion` (default 0.5)

## Real-data example
`examples/us_industries_12.json` holds annualised mean returns and the covariance of the 12 US
industry portfolios (value-weighted, monthly, January 2015 to December 2024), pick 4. With risk
aversion 1.0 the optimum is tech, utilities, retail and finance (return 14.3%, risk 14.8%): utilities
earn their place by diversifying, not by return. The platform checks the answer against the exact
solver. Rebuilt offline by `tools/portfolio-optimization/make_industry_example.py` in the examples repository.

Data: Kenneth R. French Data Library, "12 Industry Portfolios", Copyright Eugene F. Fama and
Kenneth R. French. Only derived statistics are included, not the return series.

## Output
Selected assets, equal weights, expected return, risk, whether the constraint holds, the gap to the
best classical answer (the exact solver), and the quality of the quantum run: fidelity against the ideal
output, the share of shots kept by the constraint check, and circuit depth. Fidelity is only meaningful
on small instances: it is 0.97 at 5 assets and falls with size, as it does for every noisy circuit.

## How it works
`src/workflow.py` is a plain Qiskit program. The problem is `risk_aversion * x'Cx - mu'x` with exactly k
ones. The quantum method is QAOA from `src/qubo/`, angles trained noise-free on a local statevector
simulator (CVaR objective, COBYLA); only the final circuit goes to the target. In `mixer` mode the
circuit starts in a valid portfolio and an XY-ring mixer moves only between valid ones (`init: dicke`
starts from all of them at once, `init: product` from one spread-out portfolio with no CNOTs). In
`penalty` mode the constraint is a soft term and invalid samples are post-selected away. The reference
is exhaustive search over the C(n, k) valid portfolios (annealing above 22 assets).

On `ibm:` and `fake:` targets at most 2 layers are used. Readout correction (where the device reports
its errors, up to 16 assets) and post-selection on k are declared with `ctx.report_mitigation`, with
the fidelity after each step.

## Run and test
```bash
cd qsolve-sdk
uv run qsolve test ../examples/portfolio-optimization
cd ../examples/portfolio-optimization && uv run --project ../../qsolve-sdk pytest tests
```

## Limits
Up to 24 assets (24 qubits). Equal weighting only. Treat results as research output, not advice.
