# Max-cut

Split a weighted graph into two groups to maximise the total weight of edges between them. Used
for network partitioning, clustering and circuit layout.

## Input
`nodes` (names) and `edges` as `[u, v]` or `[u, v, weight]`. Duplicate edges and self-loops are rejected.

## Output
The two groups, the cut weight, the total edge weight, and the comparison with the exact optimum.

## How it works
`src/workflow.py` is a plain Qiskit program. An edge of weight w is cut when `x_i + x_j - 2 x_i x_j = 1`,
so maximising the cut is minimising a QUBO. The quantum method is QAOA from `src/qubo/`: a layer of
cost rotations (RZ, RZZ) and an X mixer, repeated `layers` times. The angles are trained noise-free on
a local statevector simulator (CVaR objective, COBYLA), then only the final circuit is sent to the
target and measured. The answer is the best sampled cut. The reference is exhaustive search (annealing
above 22 nodes); simulated annealing is shown beside it.

On `ibm:` and `fake:` targets at most 2 layers are used (a third only added noise on real runs), and
readout correction is applied where the device reports its errors (up to 16 nodes) and declared with
`ctx.report_mitigation`. Set `train_on: backend` to train on the target instead (many more circuits).

## Run and test
```bash
cd qsolve-sdk
uv run qsolve test ../examples/max-cut          # all gates, including a noise-model case
uv run qsolve run ../examples/max-cut -i ../examples/max-cut/examples/ring_6.json
cd ../examples/max-cut && uv run --project ../../qsolve-sdk pytest tests
```

## Limits
Up to 24 nodes. Low-depth QAOA is known to struggle on highly symmetric graphs such as complete
graphs; the reported probability on the optimum and the gap to the exact answer show when that happens.
