# Molecular ground-state energy

Ground-state energy of a closed-shell molecule or ion with VQE (UCCSD) in a chosen active space,
reported next to the exact active-space energy (CASCI), Hartree-Fock, MP2 and full-space CCSD(T).

## Input
A built-in `name` (H2, LiH, HF, N2, CO, BeH2, H2O, NH3, CH4, Li+, Na+), or `xyz` text / `atoms`;
`charge`, `basis` (sto-3g, sto-6g, 3-21g, 6-31g) and an optional `active_space`
(`n_electrons`, `n_orbitals` up to 6 orbitals = 12 qubits).

## Output
Energy in Hartree, kcal/mol, kJ/mol and eV with its shot-noise error bar; error versus the exact
reference; percentage of correlation energy recovered; whether chemical accuracy (1.6 mHa) is met.

## How the quantum method works
PySCF solves the molecule classically and hands the active space to the quantum computer as a
qubit Hamiltonian (Jordan-Wigner, 2 qubits per orbital). The code is in
`src/chem/`; this workflow is `src/workflow.py`.

1. The trial state is trained **noise-free** on a statevector (UCCSD, or the experimental
   hardware-efficient `hea`). Only the final energy is measured, so the device never pays for
   optimisation.
2. The energy is read in groups of commuting Pauli terms, one circuit per group, all sent as **one
   batch** (one QPU job).
3. On a **simulator** that is the plain UCCSD circuit and plain sampling. On a **quantum device**
   (`ibm:...`, `fake:...`) the workflow instead prepares a *compact* circuit (ADAPT-selected qubit
   excitations, a few two-qubit gates; H2 needs 3) and mitigates the counts: shots split by each
   group's spread, readout correction, electron-count post-selection, and zero-noise extrapolation
   by gate folding. Noise-aware stopping drops excitations whose gates cost more noise than they
   gain. The Z-basis output after mitigation is reported to Qsolve next to the measured raw
   figures; for one system the energy error after each step is shown (mHa).

The reference methods (CASCI, Hartree-Fock, MP2, CCSD(T)) are classical and never touch the
quantum backend; one that was not computed (CCSD(T) above 60 basis functions) is reported as
unavailable.

## Limits
Closed-shell systems, small active spaces. Small basis sets are qualitative. The hardware-efficient
ansatz is experimental; UCCSD is the default. Mitigation needs at most 16 qubits. The price quote
assumes zero-noise extrapolation on every device system, so it can overstate by up to 2x.

## Run and test
Needs PySCF (Linux, macOS or WSL):

```bash
qsolve run . -i examples/h2.json
qsolve test .          # benchmark cases, the fake:fez noise-model case, reproducibility, estimate
python -m pytest tests
```
