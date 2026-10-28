"""QAOA solver for the feature-selection QUBO, built on Qiskit primitives.

Settings follow the literature:
  * Turati et al. 2022 (arXiv 2211.02861): COBYLA, depth p swept, high run-to-run variance -> we
    report mean ± std over independent seeds rather than a single run.
  * arXiv 2609.30256 (2026): p = 2, COBYLA, thousands of shots, 10 independent runs.
The optimiser minimises an objective estimated from shots (Sampler primitive), i.e. the same
noisy-objective setting as hardware.

Options added for the improvement study (scripts/08_qaoa_variants.py):
  * objective="cvar": minimise the mean energy of the best cvar_alpha fraction of shots instead of the
    mean over all shots (Barkoutsos et al. 2020, Quantum 4:256). Matches how the sampler is used here:
    only the best samples matter.
  * interp_params(): INTERP initialisation (Zhou et al. 2020, PRX 10:021067), i.e. start depth p+1 from
    the optimised depth p angles instead of random angles.
  * mixer="xy": start from the Dicke state |D^n_k> and use a ring XY mixer, so every shot has exactly k
    features (Hadfield et al. 2019, Algorithms 12:34; Wang et al. 2020, PRA 101:012320). The Dicke state
    is prepared deterministically with the Bärtschi and Eidenbenz (2019) circuit.
  * backend="statevector": exact numpy simulation of the same circuits (src/statevector.py), statistically
    equivalent to ideal AerSimulator sampling. Default backend is "aer"; override with QFF_BACKEND.

Qiskit is imported lazily so the statevector backend (and everything classical) works without it.
"""
from __future__ import annotations

import os
from math import comb

import numpy as np
from scipy.optimize import minimize

from .qubo import QUBO
from .statevector import StatevectorQAOA, all_energies, basis_bits, dicke_gate_list, ring_xy_edges

BACKEND_ENV = "QFF_BACKEND"


# ----------------------------------------------------------------------------- Ising mapping
def qubo_to_ising(Q: np.ndarray):
    """x^T Q x with x in {0,1} -> sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset, via x = (1 - Z)/2."""
    from qiskit.quantum_info import SparsePauliOp

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


# ----------------------------------------------------------------------------- circuits
def dicke_circuit(n: int, k: int):
    """Qiskit circuit preparing |D^n_k> from |0^n> (same gate list the numpy tests verify)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RYGate

    qc = QuantumCircuit(n, name=f"Dicke({n},{k})")
    for g in dicke_gate_list(n, k):
        if g[0] == "x":
            qc.x(g[1])
        elif g[0] == "cx":
            qc.cx(g[1], g[2])
        elif g[0] == "cry":
            qc.cry(g[1], g[2], g[3])
        elif g[0] == "ccry":
            qc.append(RYGate(g[1]).control(2), [g[2], g[3], g[4]])
    return qc


def xy_qaoa_circuit(H, n: int, k: int, reps: int):
    """Constrained QAOA: Dicke(n,k) start, then p x [exp(-i gamma H), ring XY mixer exp(-i beta (XX+YY)/2)].
    Parameters are named "β" and "γ" like QAOAAnsatz, so the circuit's parameter order is [β..., γ...]."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.circuit.library import PauliEvolutionGate, XXPlusYYGate

    beta, gamma = ParameterVector("β", reps), ParameterVector("γ", reps)
    qc = QuantumCircuit(n)
    qc.compose(dicke_circuit(n, k), inplace=True)
    for layer in range(reps):
        qc.append(PauliEvolutionGate(H, time=gamma[layer]), range(n))
        for a, b in ring_xy_edges(n):
            qc.append(XXPlusYYGate(2 * beta[layer]), [a, b])
    return qc


def build_ansatz(q: QUBO, reps: int, mixer: str = "x"):
    """Unmeasured Qiskit ansatz for the QUBO (X mixer: QAOAAnsatz; XY mixer: xy_qaoa_circuit)."""
    from qiskit.circuit.library import QAOAAnsatz

    H, _ = qubo_to_ising(q.Q)
    if mixer == "x":
        return QAOAAnsatz(H, reps=reps)
    return xy_qaoa_circuit(H, q.n, q.k, reps)


def logical_resources(q: QUBO, reps: int, mixer: str = "x", seed: int = 0) -> dict:
    """Gate counts of the logical circuit in the {rz, sx, x, cx} basis, all to all connectivity."""
    from qiskit import transpile

    tq = transpile(build_ansatz(q, reps, mixer), basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                   seed_transpiler=seed)
    ops = tq.count_ops()
    return {"mixer": mixer, "qaoa_depth_p": reps, "circuit_depth": tq.depth(), "two_qubit_gates": int(ops.get("cx", 0)),
            "total_gates": int(sum(ops.values()))}


# ----------------------------------------------------------------------------- helpers
def interp_params(params, p: int) -> np.ndarray:
    """INTERP (Zhou et al. 2020): depth p optimum [β(p), γ(p)] -> depth p+1 starting point.
    new_i = (i-1)/p * old_{i-1} + (p-i+1)/p * old_i, i = 1..p+1, with old_0 = old_{p+1} = 0."""
    params = np.asarray(params, dtype=float)
    beta, gamma = params[:p], params[p:]

    def up(v):
        v = np.concatenate([[0.0], v, [0.0]])
        return np.array([(i - 1) / p * v[i - 1] + (p - i + 1) / p * v[i] for i in range(1, p + 2)])

    return np.concatenate([up(beta), up(gamma)])


def cvar_from_counts(energies: np.ndarray, counts: np.ndarray, alpha: float) -> float:
    """Mean energy of the lowest-energy ceil(alpha * shots) shots."""
    order = np.argsort(energies, kind="stable")
    e, c = energies[order], counts[order]
    n_keep = int(np.ceil(alpha * c.sum()))
    before = np.cumsum(c) - c
    take = np.clip(n_keep - before, 0, c)
    return float((take * e).sum() / n_keep)


def top_m_contains_optimum(samples: np.ndarray, energies: np.ndarray, x_opt: np.ndarray, m: int = 5) -> bool:
    """Rank the *distinct* sampled subsets by energy and ask whether the exhaustive optimum is among the best m.

    Note: the optimum has the lowest energy of all 2^n bitstrings, so if it was sampled at all it ranks first.
    This is therefore identical to "the best sample is the optimum"; it is kept for backward compatibility.
    top_m_by_count_contains_optimum below is the informative version."""
    seen, order = {}, np.argsort(energies)
    for i in order:
        key = tuple(samples[i])
        if key not in seen:
            seen[key] = energies[i]
        if len(seen) >= m:
            break
    return tuple(x_opt) in seen


def top_m_by_count_contains_optimum(samples: np.ndarray, x_opt: np.ndarray, m: int = 5) -> bool:
    """Is the optimum among the m most frequently sampled distinct subsets? (What a user reading the
    histogram, without computing any energies, would pick.)"""
    keys, counts = np.unique(samples, axis=0, return_counts=True)
    top = keys[np.argsort(-counts, kind="stable")[:m]]
    return bool(np.any(np.all(top == np.asarray(x_opt), axis=1)))


def _unroute(counts: dict[str, int], routed, n: int) -> dict[str, int]:
    """After transpiling to a device, measure_all's classical bit i still holds logical qubit i
    (the final layout is applied to the measurement map), so classical register bits are already in
    logical order; only the register width can exceed n. Keep the low n bits."""
    out: dict[str, int] = {}
    for k, c in counts.items():
        kk = k.replace(" ", "")[-n:]
        out[kk] = out.get(kk, 0) + c
    return out


def _bits(key: str, n: int) -> np.ndarray:
    # Qiskit bitstrings are little-endian: rightmost char = qubit 0
    return np.array([int(key[n - 1 - i]) for i in range(n)])


def counts_to_vector(counts: dict[str, int], n: int) -> np.ndarray:
    vec = np.zeros(2 ** n, dtype=np.int64)
    for k, c in counts.items():
        vec[int(k.replace(" ", ""), 2)] += c
    return vec


def vector_to_counts(vec: np.ndarray, n: int) -> dict[str, int]:
    return {format(int(b), f"0{n}b"): int(vec[b]) for b in np.flatnonzero(vec)}


# ----------------------------------------------------------------------------- solver
class QAOASolver:
    """QAOA on an ideal or noisy simulator.

    noisy=None  -> ideal simulation (shot noise only), with backend "aer" (Qiskit AerSimulator, default)
                   or "statevector" (exact numpy simulation of the same circuit, src/statevector.py)
    noisy="fake_torino" (or any qiskit_ibm_runtime fake backend name) -> AerSimulator built from that
    backend's calibration noise model, with the circuit transpiled to its heavy-hex coupling map
    (optimization_level=3, so SWAP overhead is counted in the resource metrics).
    objective: "mean" (expected energy, the original setting) or "cvar" (mean of best cvar_alpha shots)
    mixer:     "x" (QAOAAnsatz defaults) or "xy" (Dicke start + ring XY mixer, exactly k features per shot)
    """

    def __init__(self, q: QUBO, reps: int = 2, shots: int = 2000, maxiter: int = 150, seed: int = 0,
                 noisy: str | None = None, objective: str = "mean", cvar_alpha: float = 0.1, mixer: str = "x",
                 backend: str | None = None):
        if objective not in ("mean", "cvar"):
            raise ValueError(f"unknown objective {objective!r}")
        if mixer not in ("x", "xy"):
            raise ValueError(f"unknown mixer {mixer!r}")
        self.q, self.reps, self.shots, self.maxiter, self.seed, self.noisy = q, reps, shots, maxiter, seed, noisy
        self.objective, self.cvar_alpha, self.mixer = objective, cvar_alpha, mixer
        self.backend_name = "aer" if noisy else (backend or os.environ.get(BACKEND_ENV, "aer"))
        self.E_all = all_energies(q.Q)
        self.num_parameters = 2 * reps
        self.n_evals = 0

        if self.backend_name == "statevector":
            self._sv = StatevectorQAOA(q.Q, reps, mixer, k=q.k)
            return
        if self.backend_name != "aer":
            raise ValueError(f"unknown backend {self.backend_name!r}")

        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2

        self.H, self.offset = qubo_to_ising(q.Q)
        self.ansatz = build_ansatz(q, reps, mixer)
        self.ansatz.measure_all()
        self.num_parameters = self.ansatz.num_parameters
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

    # --- resource metrics -------------------------------------------------------------
    def resources(self) -> dict:
        from qiskit import transpile

        if self.backend_name == "statevector":
            return {"backend": "statevector", "qubits": self.q.n, **logical_resources(self.q, self.reps, self.mixer, self.seed)}
        if self.noisy:
            tq = self.circuit  # already routed to the device
            two_q = sum(v for k, v in tq.count_ops().items() if k in ("cz", "cx", "ecr"))
        else:
            tq = transpile(self.ansatz, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1, seed_transpiler=self.seed)
            two_q = int(tq.count_ops().get("cx", 0))
        ops = tq.count_ops()
        return {
            "backend": self.noisy or "ideal",
            "mixer": self.mixer,
            "qubits": self.q.n,
            "physical_qubits": tq.num_qubits if self.noisy else self.q.n,
            "qaoa_depth_p": self.reps,
            "n_parameters": self.num_parameters,
            "circuit_depth": tq.depth(),
            "two_qubit_gates": int(two_q),
            "total_gates": int(sum(v for k, v in ops.items() if k != "measure")),
            "shots_per_eval": self.shots,
        }

    # --- sampling --------------------------------------------------------------------
    def _counts_vector(self, params) -> np.ndarray:
        if self.backend_name == "statevector":
            # Aer style sampling with a fixed seed per run (see StatevectorQAOA.sample_counts_fixed_seed);
            # this reproduces the committed Aer results (scripts/validate_statevector.py)
            return self._sv.sample_counts_fixed_seed(params, self.shots, self.seed)
        res = self.sampler.run([(self.circuit, params)]).result()[0]
        counts = res.data.meas.get_counts()
        if self.noisy:  # routed circuit: keep only the logical-qubit bits (measure_all maps 1:1 via layout)
            counts = _unroute(counts, self.circuit, self.q.n)
        return counts_to_vector(counts, self.q.n)

    def _sample(self, params) -> dict[str, int]:
        """Counts dict in Qiskit bitstring format (rightmost character = qubit 0)."""
        return vector_to_counts(self._counts_vector(params), self.q.n)

    def _objective(self, params) -> float:
        vec = self._counts_vector(params)
        self.n_evals += 1
        if self.objective == "cvar":
            return cvar_from_counts(self.E_all, vec, self.cvar_alpha)
        return float(vec @ self.E_all / vec.sum())

    _expectation = _objective  # backward compatible name

    def run(self, rng: np.random.Generator, x0=None) -> dict:
        """Optimise the angles with COBYLA from x0 (default: uniform random in [0, pi), as in the original
        runs), then draw one final batch of shots at the optimum."""
        if x0 is None:
            x0 = rng.uniform(0, np.pi, size=self.num_parameters)
        opt = minimize(self._objective, np.asarray(x0, dtype=float), method="COBYLA",
                       options={"maxiter": self.maxiter, "rhobeg": 0.5})
        vec = self._counts_vector(opt.x)
        idx = np.repeat(np.arange(vec.size), vec)
        samples = basis_bits(self.q.n)[idx].astype(int)
        return {
            "samples": samples, "energies": self.E_all[idx],
            "opt_params": opt.x, "final_expectation": float(opt.fun), "n_evals": self.n_evals,
        }


def random_feasible_probability(n: int, k: int) -> float:
    """P(optimum) of uniform sampling over the C(n, k) panels of size k."""
    return 1.0 / comb(n, k)
