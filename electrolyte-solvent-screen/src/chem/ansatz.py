"""Trial states for VQE: the Hartree-Fock reference, UCCSD, and a hardware-efficient alternative.

These are the circuits that are *trained* (noise-free, on a statevector). The shallow circuit that
is sent to a device is built from the trained state in `compact`.
"""

from __future__ import annotations

from functools import reduce
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

from .hamiltonian import ladder_operators

FermionicExcitation = tuple[list[int], list[int]]  # (occupied, virtual) spin orbitals


def hf_circuit(n_spatial: int, n_alpha: int, n_beta: int) -> QuantumCircuit:
    """The Hartree-Fock determinant: each spin's lowest orbitals filled."""
    qc = QuantumCircuit(2 * n_spatial)
    for i in range(n_alpha):
        qc.x(i)
    for i in range(n_beta):
        qc.x(n_spatial + i)
    return qc


def uccsd_excitations(n_spatial: int, n_alpha: int, n_beta: int) -> list[FermionicExcitation]:
    """Every spin-conserving single and double excitation out of the Hartree-Fock determinant."""
    occ_a, vir_a = list(range(n_alpha)), list(range(n_alpha, n_spatial))
    occ_b = [n_spatial + i for i in range(n_beta)]
    vir_b = [n_spatial + i for i in range(n_beta, n_spatial)]
    ex: list[FermionicExcitation] = []
    for occ, vir in ((occ_a, vir_a), (occ_b, vir_b)):
        ex += [([i], [a]) for i in occ for a in vir]
    for occ, vir in ((occ_a, vir_a), (occ_b, vir_b)):
        ex += [([i, j], [a, b]) for i, j in combinations(occ, 2) for a, b in combinations(vir, 2)]
    ex += [([i, j], [a, b]) for i in occ_a for j in occ_b for a in vir_a for b in vir_b]
    return ex


def uccsd_circuit(
    n_spatial: int, n_alpha: int, n_beta: int
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Hartree-Fock followed by exp(theta_k (T_k - T_k^dagger)) for each excitation k.

    The Pauli terms inside one excitation commute, so one Trotter step per excitation is exact."""
    m = 2 * n_spatial
    ann, cre = ladder_operators(m)
    excitations = uccsd_excitations(n_spatial, n_alpha, n_beta)
    theta = ParameterVector("t", len(excitations))
    qc = hf_circuit(n_spatial, n_alpha, n_beta)
    for k, (occ, vir) in enumerate(excitations):
        factors = [cre[a] for a in vir] + [ann[i] for i in reversed(occ)]
        t_op = reduce(lambda x, y: x.dot(y), factors)  # a+_a ... a_j a_i
        herm = (1j * (t_op - t_op.adjoint())).simplify()  # i (T - T+) is Hermitian
        qc.append(
            PauliEvolutionGate(SparsePauliOp(herm.paulis, herm.coeffs.real), time=theta[k]),
            range(m),
        )
    return qc, list(theta)


def hea_circuit(
    n_spatial: int, n_alpha: int, n_beta: int, reps: int = 2
) -> tuple[QuantumCircuit, list[Parameter]]:
    """A shallow hardware-efficient ansatz on top of the Hartree-Fock state: gentler on noise,
    but it can miss the ground state."""
    body = RealAmplitudes(2 * n_spatial, reps=reps, entanglement="linear")
    qc = hf_circuit(n_spatial, n_alpha, n_beta)
    qc.compose(body, inplace=True)
    return qc, list(body.parameters)
