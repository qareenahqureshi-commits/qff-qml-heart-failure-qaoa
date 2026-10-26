"""QAOA solver for the feature-selection QUBO, built on Qiskit primitives.

Settings follow the literature:
  * Turati et al. 2022 (arXiv 2211.02861): COBYLA, depth p swept, high run-to-run variance -> we
    report mean ± std over independent seeds rather than a single run.
  * arXiv 2609.30256 (2026): p = 2, COBYLA, thousands of shots, 10 independent runs.
The optimiser minimises the expectation of the cost Hamiltonian estimated from shots
(Sampler primitive), i.e. the same noisy-objective setting as hardware.
"""
from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from scipy.optimize import minimize

from .qubo import QUBO


def qubo_to_ising(Q: np.ndarray) -> tuple[SparsePauliOp, float]:
    """x^T Q x with x in {0,1} -> sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset, via x = (1 - Z)/2."""
    n = Q.shape[0]
    Qs = 0.5 * (Q + Q.T)
    h = np.zeros(n)
    J = {}
    offset = 0.0
    for i in range(n):
        offset += Qs[i, i] / 2
        h[i] -= Qs[i, i] / 2
        for j in range(i + 1, n):
            q = 2 * Qs[i, j]  # x_i x_j coefficient (counted once)
            offset += q / 4
            h[i] -= q / 4
            h[j] -= q / 4
            J[(i, j)] = J.get((i, j), 0.0) + q / 4
    terms = []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            terms.append(("I" * (n - 1 - i) + "Z" + "I" * i, h[i]))
    for (i, j), v in J.items():
        if abs(v) > 1e-12:
            s = ["I"] * n
            s[n - 1 - i] = "Z"
            s[n - 1 - j] = "Z"
            terms.append(("".join(s), v))
    return SparsePauliOp.from_list(terms), offset


def _unroute(counts: dict[str, int], routed, n: int) -> dict[str, int]:
    """After transpiling to a device, measure_all's classical bit i still holds logical qubit i
    (the final layout is applied to the measurement map), so classical register bits are already in
    logical order; only the register width can exceed n. Keep the low n bits."""
    out: dict[str, int] = {}
    for k, c in counts.items():
        kk = k.replace(" ", "")[-n:]
        out[kk] = out.get(kk, 0) + c
    return out


def top_m_contains_optimum(samples: np.ndarray, energies: np.ndarray, x_opt: np.ndarray, m: int = 5) -> bool:
    """Practical selector protocol: rank the *distinct* sampled subsets by energy and ask whether the
    exhaustive optimum is among the best m (the user then evaluates those m subsets classically)."""
    seen, order = {}, np.argsort(energies)
    for i in order:
        key = tuple(samples[i])
        if key not in seen:
            seen[key] = energies[i]
        if len(seen) >= m:
            break
    return tuple(x_opt) in seen


def _bits(key: str, n: int) -> np.ndarray:
    # Qiskit bitstrings are little-endian: rightmost char = qubit 0
    return np.array([int(key[n - 1 - i]) for i in range(n)])


class QAOASolver:
    """QAOA on an ideal or noisy simulator.

    noisy=None  -> ideal AerSimulator (shot noise only)
    noisy="fake_torino" (or any qiskit_ibm_runtime fake backend name) -> AerSimulator built from that
    backend's calibration noise model, with the circuit transpiled to its heavy-hex coupling map
    (optimization_level=3, so SWAP overhead is counted in the resource metrics).
    """

    def __init__(self, q: QUBO, reps: int = 2, shots: int = 2000, maxiter: int = 150, seed: int = 0,
                 noisy: str | None = None):
        self.q, self.reps, self.shots, self.maxiter, self.seed, self.noisy = q, reps, shots, maxiter, seed, noisy
        self.H, self.offset = qubo_to_ising(q.Q)
        self.ansatz = QAOAAnsatz(self.H, reps=reps)
        self.ansatz.measure_all()
        if noisy:
            from qiskit_ibm_runtime import fake_provider
            fake = getattr(fake_provider, "".join(w.capitalize() for w in noisy.split("_")))()
            # density-matrix method: exact noise channels, no per-shot trajectory sampling (4x faster at 11 qubits)
            self.backend = AerSimulator.from_backend(fake, method="density_matrix", seed_simulator=seed)
            self.circuit = transpile(self.ansatz, fake, optimization_level=3, seed_transpiler=seed)
        else:
            self.backend = AerSimulator(seed_simulator=seed)
            self.circuit = transpile(self.ansatz, self.backend, optimization_level=1, seed_transpiler=seed)
        self.sampler = SamplerV2.from_backend(self.backend, default_shots=shots, seed=seed)
        self.n_evals = 0

    # --- resource metrics -------------------------------------------------------------
    def resources(self) -> dict:
        if self.noisy:
            tq = self.circuit  # already routed to the device
            two_q = sum(v for k, v in tq.count_ops().items() if k in ("cz", "cx", "ecr"))
        else:
            tq = transpile(self.ansatz, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=self.seed)
            two_q = int(tq.count_ops().get("cx", 0))
        ops = tq.count_ops()
        return {
            "backend": self.noisy or "ideal",
            "qubits": self.q.n,
            "physical_qubits": tq.num_qubits if self.noisy else self.q.n,
            "qaoa_depth_p": self.reps,
            "n_parameters": self.ansatz.num_parameters,
            "circuit_depth": tq.depth(),
            "two_qubit_gates": int(two_q),
            "total_gates": int(sum(v for k, v in ops.items() if k != "measure")),
            "shots_per_eval": self.shots,
        }

    # --- optimisation --------------------------------------------------------------------
    def _sample(self, params) -> dict[str, int]:
        res = self.sampler.run([(self.circuit, params)]).result()[0]
        counts = res.data.meas.get_counts()
        if self.noisy:  # routed circuit: keep only the logical-qubit bits (measure_all maps 1:1 via layout)
            counts = _unroute(counts, self.circuit, self.q.n)
        return counts

    def _expectation(self, params) -> float:
        counts = self._sample(params)
        self.n_evals += 1
        tot = sum(counts.values())
        return sum(self.q.energy(_bits(k, self.q.n)) * c for k, c in counts.items()) / tot

    def run(self, rng: np.random.Generator) -> dict:
        x0 = rng.uniform(0, np.pi, size=self.ansatz.num_parameters)
        opt = minimize(self._expectation, x0, method="COBYLA", options={"maxiter": self.maxiter, "rhobeg": 0.5})
        counts = self._sample(opt.x)
        samples, energies = [], []
        for k, c in counts.items():
            b = _bits(k, self.q.n)
            samples += [b] * c
            energies += [self.q.energy(b)] * c
        return {
            "samples": np.array(samples), "energies": np.array(energies),
            "opt_params": opt.x, "final_expectation": float(opt.fun), "n_evals": self.n_evals,
        }
