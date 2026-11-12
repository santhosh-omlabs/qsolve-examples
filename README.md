# Qsolve example workflows

Reference workflows that ship with the Qsolve platform, written to be read. Each folder is one
complete, self-contained workflow: a plain Qiskit program plus a thin bridge to the platform
([`qsolve-sdk`](https://github.com/santhosh-omlabs/qsolve-sdk)). Nothing is shared between folders and nothing is hidden in a
library, so you can open one, read it top to bottom, copy it and change it into your own.

| Workflow | Family | What it does | Inputs |
|---|---|---|---|
| `electrolyte-solvent-screen` | Electronic structure / VQE | Rank battery-electrolyte solvents by Li+ binding, each energy split into mean-field and VQE correlation and checked against exact and MP2 | 7 curated solvents (B3LYP/6-31G* minima) or your own geometries |
| `portfolio-optimization` | QUBO / QAOA | Pick exactly k assets, trading return against risk | real data: 12 US industries 2015-2024; any universe, returns table or covariance, constraint as mixer or penalty |
| `max-cut` | QUBO / QAOA | Split a weighted graph into two groups | any graph |
| `generic-qubo` | QUBO / QAOA | Solve any QUBO matrix, optional "choose k" | any binary optimisation problem |
| `molecular-ground-state` | Electronic structure / VQE | Ground-state energy of a molecule or ion vs HF, exact and CCSD(T) | any closed-shell molecule, built-in names or XYZ, chosen active space |
| `interaction-energy` | Electronic structure / VQE | Binding, solvation, adsorption or reaction energy as a signed sum of system energies | any set of species; same workflow, different input |
| `quantum-kernel-classifier` | Quantum ML | Quantum-kernel SVM vs tuned RBF-SVM with a confidence interval | real data: breast-cancer diagnosis (UCI); synthetic shapes or your own table |

## Read one (start here)

Every workflow has the same shape:

| File | What to look for |
|---|---|
| `README.md` | the problem, the quantum method in plain words, how it is judged |
| `qsolve.yaml` | the contract: name, inputs, the quantum method and the classical methods it is compared with |
| `src/workflow.py` | the workflow itself: one class with `quantum()` (your circuits, run through `ctx.sampler`), `reference()` and `classical_*()` (the best classical answer), `evaluate()` (re-scores a solution) |
| `src/qubo/`, `src/chem/`, `src/kernel/` | the algorithm code (QAOA, VQE, quantum kernels) in small plain-Qiskit modules; read `__init__.py` first |
| `benchmarks/cases.yaml` | the known-answer cases the platform runs and publishes |
| `tests/` | unit tests you can run with `pytest` |

Easiest first read: `max-cut` (a few circuits, one idea), then `generic-qubo`, then the
chemistry and kernel workflows.

## Run one

Install the SDK (`pip install "qsolve-sdk[ibm]"`) plus `scikit-learn`, and `pyscf`
for the chemistry workflows (Linux, macOS or WSL). Then:

```bash
qsolve test examples/portfolio-optimization
qsolve run examples/max-cut -i examples/max-cut/examples/ring_6.json
qsolve estimate examples/interaction-energy -i examples/interaction-energy/examples/li_h2o.json
```

## Make your own

Copy the closest workflow, change `qsolve.yaml` (inputs and methods) and `quantum()`, keep the
classical reference honest, and run `qsolve test`. The platform measures everything it publishes
(fidelity, circuit size, cost, the gap to the classical answer); the workflow never reports a
number it cannot be checked on.

## Offline data preparation

`tools/<workflow>/` holds the scripts that produced each workflow's curated data (molecular
geometries, market statistics). They are kept outside the workflows on purpose: they need packages
the platform runtime does not have (geomeTRIC, RDKit), never run on the platform, and so are not
part of any workflow snapshot. The data files they write record how they were made.

## Scientific notes

- Every result is compared with a classical baseline and states its uncertainty. Quantum does not
  always win, and these workflows show that rather than hide it (for example low-depth QAOA on a
  complete-graph max-cut is barely better than random).
- Chemistry workflows use small basis sets and active spaces so they run on simulators and today's
  QPUs. Treat binding energies as indicative; the output shows HF, exact active-space and CCSD(T)
  values so the reader can judge the approximation.
- Benchmarks with small test sets (the QML cases) carry wide confidence intervals; read the interval,
  not just the accuracy.
