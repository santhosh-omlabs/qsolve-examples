"""VQE for a set of systems whose energies are combined (a molecule, a binding energy, ...).

Parameters are always trained noise-free on a statevector simulator, so the quantum run only has
to *prepare a known state and measure its energy*. What is sent to the device depends on where it
runs:

- simulator: the trained UCCSD (or hardware-efficient) circuit, plain sampling;
- quantum device: the compact ADAPT circuit (few two-qubit gates), stopped where one more gate
  would cost more noise than it gains accuracy, with readout correction, electron-count
  post-selection and zero-noise extrapolation, all reported next to the raw counts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .ansatz import hea_circuit, uccsd_circuit
from .compact import CompactState, adapt_state, excitation_pool
from .hamiltonian import ChemProblem, ChemSet
from .measure import (
    GroupReport,
    Runner,
    estimate_energies_mitigated,
    estimate_energy,
    group_observable,
    shot_plan,
)
from .mitigation import (
    DENSE_MAX_QUBITS,
    SpinSectors,
    extrapolate_to_zero_noise,
    fold_two_qubit_gates,
    sector_average,
)

Progress = Callable[..., None]

# Energy error per two-qubit gate, as a fraction of the system's energy scale (sector average
# minus HF). Fitted on ibm_fez noise models (0.2-0.4%), close to Heron's median CZ error (0.32-0.41%
# on fez, marrakesh, torino). 0 turns noise-aware stopping off.
NOISE_PER_GATE = 0.004
# Zero-noise extrapolation triples the two-qubit gates of the folded circuit; beyond this many the
# folded circuit is mostly noise. Noise-model fits: it left 0.0-0.14 mHa of H2's 2.3-3.7 mHa bias.
ZNE_MAX_GATES = 20
ZNE_RESIDUAL = 0.15  # share of the gate noise left after the extrapolation


def on_hardware(target: str) -> bool:
    """Whether a Qsolve target is a quantum device (`ibm:<device>`) or its noise model
    (`fake:<device>`), as opposed to a noise-free simulator."""
    return target.startswith(("ibm:", "fake:"))


@dataclass
class SystemEnergy:
    """One system's measured energy and how it was obtained."""

    energy: float  # the number reported: mitigated on a device, plain on a simulator
    stderr: float
    noise_free: float  # the trained state's exact energy: what a perfect device would measure
    hf: float
    casci: float
    n_params: int
    optimizer_evals: int
    groups: int  # measurement groups of the Hamiltonian
    circuits: int  # circuits sent for this system
    shots: int
    extra: dict[str, Any] = field(default_factory=dict)
    energy_raw: float | None = None  # device only: the same shots with no correction
    energy_standard: float | None = None  # device only: after readout and post-selection
    kept_fraction: float = 1.0
    readout_corrected: bool = False
    zne_applied: bool = False
    reports: list[GroupReport] = field(default_factory=list)  # device only: for a mitigation report

    @property
    def mitigation_methods(self) -> list[str]:
        if self.energy_raw is None:
            return []
        methods = ["readout-correction"] if self.readout_corrected else []
        methods.append("electron-count post-selection")
        if self.zne_applied:
            methods.append("zero-noise extrapolation")
        return methods

    def energy_steps(self) -> list[dict[str, Any]]:
        """Energy error against the exact active-space energy after each mitigation step (mHa)."""
        if self.energy_raw is None or self.energy_standard is None:
            return []
        steps = [
            ("raw counts", self.energy_raw),
            ("readout correction + electron-count post-selection", self.energy_standard),
        ]
        if self.zne_applied:
            steps.append(("zero-noise extrapolation", self.energy))
        return [{"name": n, "error": (e - self.casci) * 1e3, "error_unit": "mHa"} for n, e in steps]

    def as_dict(self) -> dict[str, Any]:
        return {
            "energy_measured": self.energy, "stderr": self.stderr,
            "energy_noise_free": self.noise_free, "hf": self.hf, "casci": self.casci,
            "n_params": self.n_params, "optimizer_evals": self.optimizer_evals,
            "groups": self.groups, "circuits": self.circuits, "kept_fraction": self.kept_fraction,
            "readout_corrected": self.readout_corrected, **self.extra,
        }  # fmt: skip


@dataclass
class VQEResult:
    energy: float  # combined over the systems
    stderr: float
    noise_free: float
    systems: dict[str, SystemEnergy]
    ansatz: str  # "compact" on a device, else the trained ansatz
    mitigated: bool
    circuits: int
    shots: int

    def details(self) -> dict[str, Any]:
        return {
            "ansatz": self.ansatz,
            "mitigation": "full" if self.mitigated else "none",
            "systems": {k: s.as_dict() for k, s in self.systems.items()},
            "objective_noise_free": self.noise_free,
            "measurement_circuits": self.circuits,
        }


def build_ansatz(
    s: ChemProblem, kind: str, hea_reps: int = 2
) -> tuple[QuantumCircuit, list[Parameter]]:
    if kind == "uccsd":
        return uccsd_circuit(s.n_spatial, s.n_alpha, s.n_beta)
    if kind == "hea":
        return hea_circuit(s.n_spatial, s.n_alpha, s.n_beta, hea_reps)
    raise ValueError(f"unknown ansatz {kind!r}")


def train(
    ansatz: QuantumCircuit,
    params: list[Parameter],
    op: SparsePauliOp,
    maxiter: int,
    seed: int,
    *,
    name: str = "",
    progress: Progress | None = None,
) -> tuple[NDArray[np.float64], float, int]:
    """Minimise <ansatz|op|ansatz> with L-BFGS-B on an exact statevector (no shot noise, no device).
    Returns (angles, energy, energy evaluations). Local classical work: nothing is dispatched."""
    n = ansatz.num_qubits
    template = ansatz.copy()
    template.save_expectation_value(op, list(range(n)))
    sim = AerSimulator(method="statevector")
    compiled = transpile(template, sim, optimization_level=1, seed_transpiler=seed)
    history: list[float] = []

    def energy(theta: NDArray[np.float64]) -> float:
        bound = compiled.assign_parameters(dict(zip(params, theta, strict=True)))
        val = float(sim.run(bound).result().data()["expectation_value"])
        history.append(val)
        if progress:
            progress("train_step", system=name, step=len(history), value=val, best=min(history))
        return val

    if not params:
        x0 = np.zeros(0)
        return x0, energy(x0), 1
    x0 = np.random.default_rng(seed).normal(0, 0.01, len(params))
    res = minimize(
        energy, x0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8}
    )
    return res.x, float(res.fun), len(history)


def run_vqe(
    chemset: ChemSet,
    run: Runner,
    *,
    hardware: bool,
    ansatz: str = "uccsd",
    shots: int = 4000,
    maxiter: int = 200,
    seed: int = 0,
    hea_reps: int = 2,
    compact_tol: float = 1e-4,
    noise_per_gate: float = NOISE_PER_GATE,
    progress: Progress | None = None,
) -> VQEResult:
    """Train every system noise-free, then measure each trained state's energy with `run`.

    `hardware` selects the quantum-device path (compact circuit, full mitigation); `ansatz` only
    matters on simulators. Systems are measured one after another, each as one batch of circuits."""
    if hardware and chemset.n_qubits > DENSE_MAX_QUBITS:
        raise ValueError(f"mitigation supports up to {DENSE_MAX_QUBITS} qubits")
    say = progress or (lambda *_a, **_k: None)
    out: dict[str, SystemEnergy] = {}
    for si, (name, s) in enumerate(chemset.systems.items()):
        say("stage", stage="vqe", system=name, qubits=s.n_qubits)
        if hardware:
            out[name] = _measure_on_device(
                name, s, run, shots, seed, si, compact_tol, noise_per_gate, say
            )
        else:
            out[name] = _measure_on_simulator(
                name, s, run, ansatz, hea_reps, shots, maxiter, seed + si, say
            )
        say("system_done", system=name, energy=out[name].energy, noise_free=out[name].noise_free)
    var = sum((chemset.coefficients[k] * s.stderr) ** 2 for k, s in out.items())
    return VQEResult(
        energy=chemset.combine({k: s.energy for k, s in out.items()}),
        stderr=float(np.sqrt(var)),
        noise_free=chemset.combine({k: s.noise_free for k, s in out.items()}),
        systems=out,
        ansatz="compact" if hardware else ansatz,
        mitigated=hardware,
        circuits=sum(s.circuits for s in out.values()),
        shots=sum(s.shots for s in out.values()),
    )


def _measure_on_simulator(
    name: str, s: ChemProblem, run: Runner, kind: str, hea_reps: int, shots: int, maxiter: int,
    seed: int, say: Progress,
) -> SystemEnergy:  # fmt: skip
    circuit, params = build_ansatz(s, kind, hea_reps)
    theta, e_trained, evals = train(circuit, params, s.op, maxiter, seed, name=name, progress=say)
    bound = circuit.assign_parameters(dict(zip(params, theta, strict=True)))
    est = estimate_energy(run, bound, s.op, shots)
    return SystemEnergy(
        energy=est.energy, stderr=est.stderr, noise_free=e_trained, hf=s.e_hf, casci=s.e_casci,
        n_params=len(params), optimizer_evals=evals, groups=est.n_groups, circuits=est.n_groups,
        shots=est.n_groups * shots,
    )  # fmt: skip


def _zne_fits(state: CompactState) -> bool:
    return 0 < state.two_qubit_gates() <= ZNE_MAX_GATES


def _measure_on_device(
    name: str, s: ChemProblem, run: Runner, shots: int, seed: int, si: int, tol: float,
    noise_per_gate: float, say: Progress,
) -> SystemEnergy:  # fmt: skip
    op = s.op
    state = adapt_state(
        op.to_matrix(sparse=True), s.n_spatial, s.n_alpha, s.n_beta, s.e_casci, tol=tol
    )
    sectors = SpinSectors.jordan_wigner(s.n_spatial, s.n_alpha, s.n_beta)
    extra: dict[str, Any] = {}
    if noise_per_gate > 0:
        # stop where one more excitation would cost more in device noise than it gains in
        # accuracy (measured: on Heron, past one or two excitations)
        scale = max(sector_average(op, sectors) - state.history[0], 0.0)
        keep, predicted = state.choose_for_noise(
            noise_per_gate, scale, corrected=_zne_fits, corrected_residual=ZNE_RESIDUAL
        )
        extra["noise_aware"] = {
            "steps_kept": keep, "steps_available": len(state.history) - 1,
            "predicted_error": predicted, "noise_per_gate": noise_per_gate,
            "energy_scale": scale, "full_accuracy_energy": state.energy,
        }  # fmt: skip
        state = state.truncated(keep)
    bound, gates = state.circuit(), state.two_qubit_gates()
    say("train_step", system=name, step=len(state.history), value=state.energy, best=state.energy)
    extra |= {"excitations": [[list(o), list(v)] for o, v in state.excitations],
              "two_qubit_gates_logical": gates}  # fmt: skip

    use_zne = _zne_fits(state)
    # the compact state is known exactly, so shots follow each group's true spread
    plan = shot_plan(op, shots, Statevector(bound).data)
    states = [bound] + ([fold_two_qubit_gates(bound, 3)] if use_zne else [])
    measured = estimate_energies_mitigated(run, states, op, plan, seed + 1000 * si, sectors)
    m = measured[0]
    energy, stderr = m.energy, m.stderr
    extra["two_qubit_gates_device"] = m.two_qubit_gates
    if use_zne:
        folded = measured[1]
        energy, stderr = extrapolate_to_zero_noise(m.energy, m.stderr, folded.energy, folded.stderr)
        extra["zne"] = {
            "applied": True, "fold_factors": [1, 3], "energy_folded": folded.energy,
            "two_qubit_gates_device_folded": folded.two_qubit_gates,
        }  # fmt: skip
    else:
        reason = (
            "no two-qubit gates to correct" if gates == 0
            else f"more than {ZNE_MAX_GATES} two-qubit gates to fold"
        )  # fmt: skip
        extra["zne"] = {"applied": False, "reason": reason}
    n_circuits = sum(x.n_circuits for x in measured)
    return SystemEnergy(
        energy=energy, stderr=stderr, noise_free=state.energy, hf=s.e_hf, casci=s.e_casci,
        n_params=len(state.angles), optimizer_evals=len(state.history), groups=m.n_groups,
        circuits=n_circuits, shots=n_circuits * m.shots_per_circuit, extra=extra,
        energy_raw=m.energy_raw, energy_standard=m.energy, kept_fraction=m.kept_fraction,
        readout_corrected=m.readout_corrected, zne_applied=use_zne,
        reports=m.reports + (measured[1].reports if use_zne else []),
    )  # fmt: skip


# ---- what a run will cost ----------------------------------------------------------------------

# Measured (sto-3g, UCCSD): H2O 8 qubits 26 parameters 14 steps 23 ms per energy; N2 12 qubits
# 117 parameters 23 steps 132 ms per energy. Steps are capped at a typical count, not the maximum:
# L-BFGS-B converges long before `optimizer_iters` on these problems.
_TYPICAL_LBFGS_STEPS = 30
_S_PER_ENERGY = 0.022
_S_PER_PARAM_AMPLITUDE = 1.9e-8
# Compact (ADAPT) state: measured 5.6 s for 12 qubits, 117-excitation pool, 35 steps; budgeted at
# 40 steps with headroom.
_S_PER_POOL_AMPLITUDE = 5e-7
_ADAPT_STEPS = 40
HARDWARE_TRANSPILE_TRIES = 4  # the device path transpiles each circuit with several seeds


@dataclass
class Cost:
    """What a run will send and the classical work around it."""

    circuits: int
    shots: int
    train_seconds: float  # local statevector training, no device involved
    transpile_tries: int  # transpiles per circuit sent

    def runtime(self, s_per_transpile: float, s_per_circuit: float, s_per_shot: float) -> float:
        """Predicted seconds, given the platform's per-transpile, per-circuit and per-shot costs."""
        sent = self.circuits * (self.transpile_tries * s_per_transpile + s_per_circuit)
        return self.train_seconds + sent + s_per_shot * self.shots


def plan_cost(
    chemset: ChemSet, *, hardware: bool, ansatz: str, shots: int, maxiter: int, hea_reps: int = 2
) -> Cost:
    """Circuits, shots and training time of `run_vqe`, from the Hamiltonians alone (the plan's size
    depends on them, not on the trained state). On a device it assumes zero-noise extrapolation
    doubles the circuits, an upper bound: it is skipped for states with more than ZNE_MAX_GATES."""
    steps = min(maxiter, _TYPICAL_LBFGS_STEPS)
    circuits = total_shots = 0
    seconds = 0.0
    for s in chemset.systems.values():
        if hardware:
            pool = len(excitation_pool(s.n_spatial, s.n_alpha, s.n_beta))
            seconds += 1.0 + _S_PER_POOL_AMPLITUDE * pool * 2.0**s.n_qubits * _ADAPT_STEPS
            plan = shot_plan(s.op, shots)
            n, per_circuit = 2 * plan.circuits, plan.shots_per_circuit
        else:
            n_params = len(build_ansatz(s, ansatz, hea_reps)[1])
            per_eval = (
                _S_PER_ENERGY + _S_PER_PARAM_AMPLITUDE * n_params * s.n_qubits * 2.0**s.n_qubits
            )
            # L-BFGS-B: one energy and one finite-difference gradient (a call per parameter)
            seconds += steps * (n_params + 1) * per_eval
            n, per_circuit = len(group_observable(s.op)[1]), shots
        circuits += n
        total_shots += n * per_circuit
    return Cost(circuits, total_shots, seconds, HARDWARE_TRANSPILE_TRIES if hardware else 1)
