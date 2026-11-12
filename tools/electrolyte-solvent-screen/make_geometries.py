"""Offline: build the curated geometries in electrolyte-solvent-screen/src/geometries.json.

Kept outside the workflow so it is not part of its snapshot: it needs geomeTRIC and RDKit, which
the platform runtime does not have, and it never runs on the platform.

Each solvent and its Li+ complex is optimised at B3LYP/6-31G* (density fitting) and checked to be
a minimum (no imaginary harmonic frequencies). The free solvent is the lower of two optimised
starting conformers: the lowest RDKit/MMFF one and, for chains, the fully extended one (binding is
measured from the free molecule's most stable form; for DME, all-trans is 1.9 kcal/mol below the
gauche form it adopts when chelating). The complex starts from the conformer that suits the site,
with Li+ placed on it (carbonyl O along C=O, nitrile N along C#N, ether/water O on the lone-pair
bisector, both DME oxygens for the chelate).

    python tools/electrolyte-solvent-screen/make_geometries.py [name ...]   (pyscf, geometric, rdkit)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

OUT = Path(__file__).resolve().parents[2] / "electrolyte-solvent-screen" / "src" / "geometries.json"
METHOD = {"xc": "b3lyp", "basis": "6-31g*", "density_fit": True}

# name: (SMILES, coordination site, Li-X start distance in Angstrom)
SOLVENTS: dict[str, tuple[str, str, float]] = {
    "water": ("O", "ether", 1.90),
    "EC": ("O=C1OCCO1", "carbonyl", 1.85),
    "DMC": ("COC(=O)OC", "carbonyl", 1.85),
    "FEC": ("O=C1OCC(F)O1", "carbonyl", 1.85),
    "ACN": ("CC#N", "nitrile", 1.95),
    "THF": ("C1CCOC1", "ether", 1.90),
    "DME": ("COCCOC", "chelate", 1.95),
}


def conformers(smiles: str, site: str) -> tuple[list[str], Chem.Mol, list[np.ndarray], np.ndarray]:
    """Atom symbols, the molecule, starting points for the free solvent, and for the complex."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=60, randomSeed=7))
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    energies = [e for _, e in res]
    lowest = min(ids, key=lambda c: energies[c])
    starts = [lowest]
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "H"]
    if not mol.GetRingInfo().NumRings() and len(heavy) >= 4:  # chains: also the extended form

        def bend(cid: int) -> float:
            c = mol.GetConformer(cid)
            return sum(
                180 - abs(rdMolTransforms.GetDihedralDeg(c, *heavy[i : i + 4]))
                for i in range(len(heavy) - 3)
            )

        extended = min(ids, key=bend)
        if extended != lowest:
            starts.append(extended)
    if site == "chelate":  # needs the gauche O-C-C-O conformer with both oxygens facing the ion
        o = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O"]

        def oo(cid: int) -> float:
            p = mol.GetConformer(cid).GetPositions()
            return float(np.linalg.norm(p[o[0]] - p[o[1]]))

        for_complex = min(ids, key=lambda c: (oo(c) > 3.0, energies[c]))
    else:
        for_complex = lowest
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = [mol.GetConformer(c).GetPositions() for c in starts]
    return syms, mol, pos, mol.GetConformer(for_complex).GetPositions()


def place_ion(mol: Chem.Mol, pos: np.ndarray, site: str, d: float) -> np.ndarray:
    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    def nbrs(i: int) -> list[int]:
        return [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()]

    if site == "carbonyl":
        o = next(
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetSymbol() == "O"
            and any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())
        )
        return pos[o] + d * unit(pos[o] - pos[nbrs(o)[0]])
    if site == "nitrile":
        n = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
        return pos[n] + d * unit(pos[n] - pos[nbrs(n)[0]])
    if site == "ether":
        o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
        return pos[o] + d * unit(2 * pos[o] - sum(pos[k] for k in nbrs(o)))
    if site == "chelate":
        os_ = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O"]
        mid = pos[os_].mean(axis=0)
        away = unit(mid - np.mean([pos[k] for o in os_ for k in nbrs(o)], axis=0))
        half = np.linalg.norm(pos[os_[0]] - pos[os_[1]]) / 2
        return mid + np.sqrt(max(d**2 - half**2, 0.8)) * away
    raise ValueError(site)


def xyz(syms: list[str], pos: np.ndarray) -> str:
    return "\n".join(f"{s} {x:.5f} {y:.5f} {z:.5f}" for s, (x, y, z) in zip(syms, pos, strict=True))


def relax(syms: list[str], pos: np.ndarray, charge: int) -> dict:
    t = time.time()
    mol = gto.M(atom=xyz(syms, pos), basis=METHOD["basis"], charge=charge, spin=0, verbose=0)
    mf = dft.RKS(mol, xc=METHOD["xc"]).density_fit()
    eq = optimize(mf, maxsteps=200)
    mf = dft.RKS(eq, xc=METHOD["xc"]).density_fit().run()
    freqs = thermo.harmonic_analysis(eq, mf.Hessian().kernel())["freq_wavenumber"]
    imag = [float(abs(f)) for f in np.atleast_1d(freqs) if np.iscomplexobj(f) and abs(f.imag) > 1e-6]
    imag += [float(-f) for f in np.atleast_1d(freqs).real if f < -1e-6]
    pos_eq = eq.atom_coords(unit="Angstrom")
    return {
        "xyz": xyz(syms, pos_eq),
        "charge": charge,
        "energy_b3lyp_hartree": float(mf.e_tot),
        "imaginary_frequencies_cm1": imag,
        "lowest_frequency_cm1": float(np.min(np.abs(np.atleast_1d(freqs)))),
        "seconds": round(time.time() - t, 1),
    }


def main(names: list[str]) -> None:
    data = json.loads(OUT.read_text()) if OUT.exists() else {"method": METHOD, "species": {}}
    for name in names or list(SOLVENTS):
        smiles, site, d = SOLVENTS[name]
        syms, mol, starts, pos = conformers(smiles, site)
        free = min((relax(syms, p, 0) for p in starts), key=lambda r: r["energy_b3lyp_hartree"])
        data["species"][name] = {"smiles": smiles, **free}
        print(name, free["seconds"], "s", free["imaginary_frequencies_cm1"], flush=True)
        li = place_ion(mol, pos, site, d)
        data["species"][f"Li+{name}"] = {
            "site": site,
            **relax([*syms, "Li"], np.vstack([pos, li]), 1),
        }
        c = data["species"][f"Li+{name}"]
        print(f"Li+{name}", c["seconds"], "s", c["imaginary_frequencies_cm1"], flush=True)
        OUT.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
