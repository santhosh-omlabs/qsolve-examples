"""Curated electrolyte solvents and their Li+ complexes.

B3LYP/6-31G* minima, built offline by tools/electrolyte-solvent-screen in the examples repo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DATA = json.loads((Path(__file__).parent / "geometries.json").read_text(encoding="utf-8"))
SPECIES: dict[str, dict[str, Any]] = _DATA["species"]
GEOMETRY_METHOD: dict[str, Any] = _DATA["method"]

SOLVENTS: dict[str, str] = {
    "EC": "Ethylene carbonate",
    "FEC": "Fluoroethylene carbonate",
    "DMC": "Dimethyl carbonate",
    "DME": "1,2-Dimethoxyethane (glyme)",
    "THF": "Tetrahydrofuran",
    "ACN": "Acetonitrile",
    "water": "Water",
}
ION = "Li+"


@dataclass(frozen=True)
class Pair:
    """One solvent in the screen: its own system and the system of its ion complex."""

    solvent: str
    title: str
    solvent_xyz: str
    complex_xyz: str

    @property
    def solvent_key(self) -> str:
        return self.solvent

    @property
    def complex_key(self) -> str:
        return f"{ION}·{self.solvent}"


def pairs(input: dict[str, Any]) -> list[Pair]:
    """The solvents a run screens, in input order: built-ins first, then the user's own."""
    out = [
        Pair(name, SOLVENTS[name], SPECIES[name]["xyz"], SPECIES[f"{ION}{name}"]["xyz"])
        for name in input.get("solvents", [])
    ]
    for c in input.get("custom_solvents", []):
        out.append(Pair(c["name"], c.get("title") or c["name"], c["solvent_xyz"], c["complex_xyz"]))
    return out
