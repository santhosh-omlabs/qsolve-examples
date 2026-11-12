"""Quantum chemistry with VQE: molecular Hamiltonians, compact circuits, mitigated energies.

Needs PySCF (`pip install pyscf`). Nothing here knows how circuits are run: functions take a
`Runner` (circuits and shots in, counts out), and `sampler_runner` adapts a Qiskit `SamplerV2`.
"""

from .hamiltonian import (
    CHEMICAL_ACCURACY_HA,
    ActiveSpace,
    ChemProblem,
    ChemSet,
    MoleculeSpec,
    build_hamiltonian,
)
from .interpret import HA_TO_KCAL, energy_answer, energy_summary, energy_units
from .measure import Measurement, Runner, sampler_runner
from .vqe import Cost, SystemEnergy, VQEResult, on_hardware, plan_cost, run_vqe

__all__ = [
    "CHEMICAL_ACCURACY_HA",
    "HA_TO_KCAL",
    "ActiveSpace",
    "ChemProblem",
    "ChemSet",
    "Cost",
    "Measurement",
    "MoleculeSpec",
    "Runner",
    "SystemEnergy",
    "VQEResult",
    "build_hamiltonian",
    "energy_answer",
    "energy_summary",
    "energy_units",
    "on_hardware",
    "plan_cost",
    "run_vqe",
    "sampler_runner",
]
