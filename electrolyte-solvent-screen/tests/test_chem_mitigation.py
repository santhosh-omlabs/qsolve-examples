import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from src.chem.mitigation import (
    SpinSectors,
    allocate_copies,
    bit_table,
    correct_readout,
    extrapolate_to_zero_noise,
    fold_two_qubit_gates,
    symmetry_mask,
)


def test_allocate_copies_gives_exactly_the_total_with_one_per_group_at_least() -> None:
    copies = allocate_copies([5.0, 1.0, 0.0, 2.0], 16)
    assert sum(copies) == 16 and min(copies) >= 1
    assert copies[0] > copies[3] > copies[1]
    assert allocate_copies([0.0, 0.0], 5) in ([3, 2], [2, 3])  # no weights: an even split


def test_readout_correction_inverts_the_assignment_matrix() -> None:
    errors = [(0.03, 0.08), (0.01, 0.05), (0.02, 0.02)]  # P(1|0), P(0|1) per bit
    ideal = np.zeros(8)
    ideal[0b101] = 1.0
    noisy = ideal.reshape([2] * 3)
    for q, (p10, p01) in enumerate(errors):
        a = np.array([[1 - p10, p01], [p10, 1 - p01]])
        axis = 2 - q
        noisy = np.moveaxis(np.tensordot(a, noisy, axes=([1], [axis])), 0, axis)
    assert np.allclose(correct_readout(noisy.reshape(-1), errors), ideal)


def test_symmetry_mask_keeps_only_the_right_electron_counts_and_skips_rotated_sectors() -> None:
    sectors = SpinSectors.jordan_wigner(2, 1, 1)  # qubits 0-1 alpha, 2-3 beta, one electron each
    bits = bit_table(4)
    mask = symmetry_mask(["Z", "Z", "I", "Z"], sectors, bits)
    assert mask is not None and mask.sum() == 4  # 2 ways to place each spin's electron
    assert symmetry_mask(["X", "Z", "X", "Z"], sectors, bits) is None  # nothing checkable
    only_beta = symmetry_mask(["X", "Z", "Z", "Z"], sectors, bits)
    assert only_beta is not None and only_beta.sum() == 8


def test_folding_keeps_the_circuit_and_triples_the_two_qubit_gates() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.3, 1)
    folded = fold_two_qubit_gates(qc, 3)
    assert folded.count_ops()["cx"] == 3
    unfolded = folded.copy()
    unfolded.data = [i for i in unfolded.data if i.operation.name != "barrier"]
    assert Operator(unfolded).equiv(Operator(qc))


def test_zero_noise_extrapolation_is_linear() -> None:
    e, s = extrapolate_to_zero_noise(-1.10, 0.01, -1.00, 0.01)
    assert abs(e - (-1.15)) < 1e-12 and s > 0.01
