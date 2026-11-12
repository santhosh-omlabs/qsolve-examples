"""Turning VQE energies into a chemist-friendly answer: units, error against the references,
chemical-accuracy flag."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .hamiltonian import CHEMICAL_ACCURACY_HA, ChemSet

HA_TO_KCAL = 627.5094740631
HA_TO_KJ = 2625.4996394799
HA_TO_EV = 27.211386245988


def energy_units(e: float) -> dict[str, float]:
    """An energy in Hartree in the other units chemists use."""
    return {
        "energy_hartree": e, "energy_kcal_mol": e * HA_TO_KCAL, "energy_kj_mol": e * HA_TO_KJ,
        "energy_ev": e * HA_TO_EV,
    }  # fmt: skip


def energy_answer(
    chemset: ChemSet,
    details: Mapping[str, Any],
    *,
    energy: float,
    stderr: float,
    exact: float,
    hf: float,
    ccsd_t: float | None,
) -> dict[str, Any]:
    """The answer for a single molecule or a combination of systems (a binding energy, ...).

    `details` is `VQEResult.details()`; `energy` and `stderr` are the combined VQE values, and
    `exact`, `hf` and `ccsd_t` the references combined over the systems the same way."""
    err = energy - exact
    noise_free = details["objective_noise_free"]
    recovered = 100.0 * (noise_free - hf) / (exact - hf) if abs(exact - hf) > 1e-9 else None
    return {
        "label": chemset.label,
        **energy_units(energy),
        "energy_stderr_hartree": stderr,
        "reference_active_space_exact_hartree": exact,
        "hartree_fock_hartree": hf,
        "ccsd_t_full_space_hartree": ccsd_t,
        "error_vs_exact_mha": err * 1000,
        "noise_free_error_vs_exact_mha": (noise_free - exact) * 1000,
        "correlation_recovered_pct": recovered,
        "within_chemical_accuracy": abs(err) <= CHEMICAL_ACCURACY_HA,
        "systems": {
            k: {
                "energy_hartree": v["energy_measured"], "stderr_hartree": v["stderr"],
                "hartree_fock": v["hf"], "exact_active_space": v["casci"],
                "qubits": chemset.systems[k].n_qubits, "coefficient": chemset.coefficients[k],
            }
            for k, v in details["systems"].items()
        },
    }  # fmt: skip


def energy_summary(label: str, vqe_energy: float, stderr: float, exact: float, hf: float) -> str:
    return (
        f"{label}: {vqe_energy:.6f} Ha ({vqe_energy * HA_TO_KCAL:.2f} kcal/mol) "
        f"+/- {stderr * 1000:.2f} mHa. Exact active-space reference {exact:.6f} Ha "
        f"(error {(vqe_energy - exact) * 1000:+.2f} mHa); HF {hf:.6f} Ha."
    )
