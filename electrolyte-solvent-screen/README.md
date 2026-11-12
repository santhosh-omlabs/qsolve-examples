# Electrolyte solvent screen

Which solvent holds on to a lithium ion most tightly? That binding energy is a first descriptor in
electrolyte design: it shapes the ion's solvation shell, how hard it is to strip the shell off at the
electrode (desolvation, a limit on fast charging), and which molecules reach the electrode surface
to form the protective interphase. This workflow ranks candidate solvents by it.

For each solvent S it computes the gas-phase binding energy of one Li+ ion

    dE(S) = E(Li+ . S) - E(S) - E(Li+)          (more negative = binds more strongly)

with every energy from VQE, and reports the ranking with error bars.

## Built-in solvents

| Name | Molecule | Role in Li-ion electrolytes |
|---|---|---|
| EC | ethylene carbonate | high-permittivity co-solvent in almost every commercial cell |
| FEC | fluoroethylene carbonate | additive that improves the interphase on silicon and lithium anodes |
| DMC | dimethyl carbonate | low-viscosity linear co-solvent |
| DME | 1,2-dimethoxyethane (glyme) | ether for lithium-metal and lithium-sulfur cells; binds with both oxygens |
| THF | tetrahydrofuran | cyclic ether |
| ACN | acetonitrile | high-conductivity nitrile solvent |
| water | water | reference point, and the contaminant every cell keeps out |

Each solvent and its Li+ complex is a B3LYP/6-31G* minimum, verified to have no imaginary
frequencies (built offline by `tools/electrolyte-solvent-screen/make_geometries.py` in the examples repository; the data records energies and frequencies). You can
add up to four of your own solvents as XYZ geometries of the molecule and of its Li+ complex.

## What the answer contains

For every solvent, the binding energy split into its parts, so you can see what the quantum
computer contributed and how far to trust it:

| Column | Meaning |
|---|---|
| Quantum (VQE) | binding energy from measured VQE energies, with its 1-sigma shot-noise error |
| Exact, active space | the same quantity solved exactly (CASCI): what a perfect quantum computer would give |
| Error | VQE minus exact; chemical accuracy is 1 kcal/mol |
| Mean-field | Hartree-Fock binding energy: the electrostatics that dominate ion-solvent binding |
| + Correlation | what the active-space correlation adds, the part VQE computes |
| MP2, full space | second-order correlation over every orbital: what a small active space leaves out |

Plus the strongest and weakest binder, whether the VQE ranking matches the exact ranking, and any
neighbouring pair whose order is inside the error bars (reported as unresolved rather than ranked).

## How it works

`src/workflow.py` builds one problem holding Li+, every solvent and every complex, and the
code in `src/chem/` does the quantum work. The trial state of each system is
trained noise-free (UCCSD) and its energy is then measured with shots through the Qsolve backend,
all circuits of a system as one batch. On a quantum device (`ibm:...`, `fake:...`) the workflow
prepares a compact ADAPT circuit instead and mitigates the counts (readout correction,
electron-count post-selection, zero-noise extrapolation), reporting the mitigated output next to
the measured raw figures. The exact active-space, Hartree-Fock and MP2 energies are classical
methods that give the binding energies the VQE result is checked against; CCSD(T) is out of reach
at this size and is reported as unavailable.

**The active space is centred on the binding site.** The atoms within 2.4 Å of Li (the carbonyl
O, the ether or water O, the nitrile N, both DME oxygens) define it: the valence orbitals with the
most weight on those atoms, chosen the same way in the complex and in the free solvent. Choosing
orbitals by energy instead fails here: in every complex the three lowest empty orbitals belong to
Li+, so the complex and the free solvent would correlate different orbitals, and the mismatch
appears as fake binding energy (it made acetonitrile look 16 kcal/mol weaker than it is).

## Results (seven solvents, 6-31G*, 4 electrons in 4 orbitals, 50,000 shots, simulator)

| Solvent | VQE | Exact, active space | Hartree-Fock | MP2, full space |
|---|---|---|---|---|
| DME | -66.7 ± 0.6 | -66.7 | -66.8 | -71.2 |
| EC | -53.3 ± 0.5 | -52.9 | -54.6 | -52.0 |
| FEC | -48.7 ± 0.4 | -48.3 | -49.6 | -47.2 |
| DMC | -47.0 ± 0.3 | -47.1 | -47.1 | -45.6 |
| THF | -46.3 ± 0.4 | -45.7 | -46.2 | -48.5 |
| ACN | -45.9 ± 0.6 | -45.7 | -45.3 | -44.8 |
| water | -41.3 ± 0.5 | -40.3 | -39.7 | -41.7 |

kcal/mol. VQE reproduces the exact ranking, every value within 1 kcal/mol of exact; DMC, THF and
ACN lie within each other's error bars and are reported as unresolved. The trends match the
chemistry: the chelating glyme binds most strongly, the cyclic carbonate more than the linear one,
fluorination weakens EC, water binds least.

## Limits, stated plainly

- **Gas phase, one solvent molecule.** Real Li+ carries four or so solvent molecules and sits in a
  liquid. One-to-one binding energies rank solvents' intrinsic affinity, which is what screening
  needs; they are not solvation free energies.
- **Small active space.** Four electrons in four orbitals (8 qubits) at the binding site. It adds
  the lone-pair correlation (for carbonates it moves the energy the same way as MP2), but not the
  dispersion and polarisation spread over the whole molecule, which is why MP2 binds ethers and
  water 2-4 kcal/mol more. Enlarge `active_space` (up to 6 orbitals, 12 qubits) to capture more.
- **No counterpoise correction.** 6-31G* overbinds slightly through basis-set superposition error,
  by a similar amount for similar solvents, so rankings are more reliable than absolute values.
- **Electronic energies only**: no zero-point or thermal corrections, so values are not enthalpies.

## Run and test

Needs PySCF (Linux, macOS or WSL):

```bash
qsolve run . -i examples/quick_preview.json
qsolve test .          # takes minutes: seven-system builds at 6-31G*
python -m pytest tests
```

## Further reading

Competitive lithium solvation of linear and cyclic carbonates from quantum chemistry, Phys. Chem.
Chem. Phys. 2016, doi:10.1039/C5CP05121E.
