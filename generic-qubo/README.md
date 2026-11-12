# Generic QUBO solver

Minimise `x'Qx` over binary variables for any square matrix `Q`, with an optional "exactly k ones"
constraint. Use it when your problem is already in QUBO form (routing, scheduling, assignment...).

## Input
`Q` (square matrix, up to 24x24), optional `labels`, `offset`, and `cardinality`.

## Output
The assignment, the variables set to 1, the objective value, and the comparison with the exact optimum.

## How it works
`src/workflow.py` is a plain Qiskit program using QAOA from `src/qubo/`: cost rotations (RZ, RZZ) and
a mixer, repeated `layers` times, with angles trained noise-free on a local statevector simulator (CVaR
objective, COBYLA). Only the final circuit goes to the target. With `cardinality` the circuit starts in
the uniform superposition of all states with k ones (a Dicke state) and mixes with an XY ring, so every
sample is valid and no penalty term is needed. The reference is exhaustive search (annealing above 22
variables); simulated annealing is shown beside it.

On `ibm:` and `fake:` targets at most 2 layers are used. Readout correction (where the device reports
its errors, up to 16 variables) and post-selection on k are applied to the reported distribution and
declared with `ctx.report_mitigation`. `train_on: backend` trains on the target instead.

## Run and test
```bash
cd qsolve-sdk
uv run qsolve test ../examples/generic-qubo
cd ../examples/generic-qubo && uv run --project ../../qsolve-sdk pytest tests
```

## Limits
24 variables per run. For larger problems decompose first.
