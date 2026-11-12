# Interaction energy

Binding, solvation, adsorption or reaction energy as a **signed sum of system energies**, each
computed with VQE. One workflow serves many chemistries: for example Li+ binding to different
solvent molecules in battery electrolytes.

## Input
`systems`: named species (each a built-in `name` or `xyz`, with `charge`), `basis`, an `active_space`,
and optional `terms` (coefficient per system). Without `terms`, the first system is `+1` and the
others `-1` (a binding energy).

## Output
The combined energy (Hartree, kcal/mol, kJ/mol, eV) with error bars, per-system energies, and the
Hartree-Fock, exact active-space and CCSD(T) references.

## How the quantum method works
Each system gets its own qubit Hamiltonian (PySCF, active space, Jordan-Wigner) and is solved with
VQE exactly as in [molecular-ground-state](../molecular-ground-state): the trial state is trained
noise-free, then its energy is measured in commuting groups, all circuits of a system in one
batch. On a simulator that is the plain UCCSD circuit; on a quantum device (`ibm:...`, `fake:...`)
it is the compact ADAPT circuit with readout correction, electron-count post-selection and
zero-noise extrapolation, reported next to the raw counts. The answer is the weighted sum of the
system energies, with the error bars combined in quadrature. The code is in
`src/chem/`; this workflow is `src/workflow.py`.

The reference methods (exact active space, Hartree-Fock, MP2, CCSD(T)) are classical and combine the
systems the same way. The error bar of a difference is the sum in quadrature of the parts, so a
small interaction energy needs more shots than a total energy.

## Limits
Fragment-scale: the quantum calculation covers a small active space of each species while the rest
is treated at mean-field level. Indicative, not a replacement for a converged DFT or coupled-cluster
study of a full material. Closed-shell systems only, at most 6 systems of 12 qubits each.

## Run and test
Needs PySCF (Linux, macOS or WSL):

```bash
qsolve run . -i examples/li_h2o.json
qsolve test .
python -m pytest tests
```
