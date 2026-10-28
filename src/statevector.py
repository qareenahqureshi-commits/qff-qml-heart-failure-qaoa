"""Exact statevector QAOA in plain numpy, mirroring the Qiskit circuits built in src/qaoa.py.

Why it exists: an 11 qubit QAOA state has only 2048 amplitudes, so the ideal circuit can be simulated
exactly and then sampled with the same number of shots as SamplerV2. This samples from the same
distribution as an ideal AerSimulator run of the same circuit, so results are statistically equivalent
(different random streams), and it runs without Qiskit installed and roughly 10x faster. Select it with
QAOASolver(..., backend="statevector") or the environment variable QFF_BACKEND=statevector.
It cannot model hardware noise: noisy runs always use Qiskit Aer.

Conventions, identical to Qiskit:
  * basis index b = sum_i x_i 2^i (qubit 0 is the least significant bit, Qiskit little endian)
  * parameters are ordered [beta_0 .. beta_{p-1}, gamma_0 .. gamma_{p-1}] (Qiskit sorts the
    ParameterVectors "β" before "γ")
  * cost layer exp(-i gamma H_C) (QAOAAnsatz / PauliEvolutionGate; the Ising offset is a global phase)
  * X mixer exp(-i beta sum_i X_i), initial state |+>^n (QAOAAnsatz defaults)
  * XY mixer: XXPlusYYGate(2 beta) on each edge of a ring in parity order, i.e. exp(-i beta (XX+YY)/2)
    per edge; initial state the Dicke state |D^n_k> (uniform superposition of all k feature panels)

tests/test_core.py checks these conventions against qiskit.quantum_info.Statevector when Qiskit is
installed, and checks the Dicke preparation circuit (shared gate list below) against the exact state.
"""
from __future__ import annotations

from math import comb

import numpy as np


# ----------------------------------------------------------------------------- basis helpers
def basis_bits(n: int) -> np.ndarray:
    """(2^n, n) int8 array; row b holds the bits of basis state b, column i = qubit i."""
    b = np.arange(2 ** n)
    return ((b[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def all_energies(Q: np.ndarray) -> np.ndarray:
    """QUBO energy x^T Q x of every basis state, indexed like the statevector."""
    B = basis_bits(Q.shape[0]).astype(float)
    return np.einsum("bi,ij,bj->b", B, Q, B)


def dicke_state(n: int, k: int) -> np.ndarray:
    w = basis_bits(n).sum(1)
    psi = (w == k).astype(complex)
    return psi / np.sqrt(comb(n, k))


# ------------------------------------------------------------------- Dicke state preparation
def dicke_gate_list(n: int, k: int) -> list[tuple]:
    """Deterministic Dicke state preparation, Bärtschi & Eidenbenz (2019), "Deterministic preparation of
    Dicke states", FCT 2019, arXiv:1904.07358. O(nk) gates, no ancillas.

    Paper labels qubits 1..n and starts from |0^{n-k} 1^k>; we map label j to qubit j-1.
    U_{n,k} = prod_{l=n}^{k+1} SCS_{l,k} * prod_{l=k}^{2} SCS_{l,l-1}, where the split and cyclic shift
    block SCS_{l,m} is one two qubit gate on (l-1, l) and m-1 three qubit gates on (l-j-1, l-j, l).
    Gate tuples: ("x", t), ("cx", c, t), ("cry", theta, c, t), ("ccry", theta, c1, c2, t).
    """
    q = lambda label: label - 1  # noqa: E731
    gates: list[tuple] = [("x", q(j)) for j in range(n - k + 1, n + 1)]

    def scs(l: int, m: int):
        # (i) two qubit block on labels (l-1, l)
        gates.append(("cx", q(l - 1), q(l)))
        gates.append(("cry", 2 * np.arccos(np.sqrt(1 / l)), q(l), q(l - 1)))
        gates.append(("cx", q(l - 1), q(l)))
        # (ii)_j three qubit blocks on labels (l-j, l-j+1, l) for j = 2..m
        for j in range(2, m + 1):
            gates.append(("cx", q(l - j), q(l)))
            gates.append(("ccry", 2 * np.arccos(np.sqrt(j / l)), q(l), q(l - j + 1), q(l - j)))
            gates.append(("cx", q(l - j), q(l)))

    for l in range(n, k, -1):
        scs(l, k)
    for l in range(k, 1, -1):
        scs(l, l - 1)
    return gates


def apply_gates(psi: np.ndarray, gates: list[tuple], n: int) -> np.ndarray:
    """Apply a gate list (format of dicke_gate_list) to a statevector, Qiskit bit order."""
    psi = psi.astype(complex).copy()
    idx = np.arange(2 ** n)
    for g in gates:
        kind = g[0]
        if kind == "x":
            t = g[1]
            psi = psi[idx ^ (1 << t)]
            continue
        if kind == "cx":
            c, t = g[1], g[2]
            sel = ((idx >> c) & 1) == 1
            new = psi.copy()
            new[sel] = psi[idx[sel] ^ (1 << t)]
            psi = new
            continue
        theta, *ctrl, t = g[1:]
        mask = np.ones(2 ** n, dtype=bool)
        for c in ctrl:
            mask &= ((idx >> c) & 1) == 1
        i0 = idx[mask & (((idx >> t) & 1) == 0)]
        i1 = i0 | (1 << t)
        cth, sth = np.cos(theta / 2), np.sin(theta / 2)
        a0, a1 = psi[i0].copy(), psi[i1].copy()
        psi[i0] = cth * a0 - sth * a1
        psi[i1] = sth * a0 + cth * a1
    return psi


# ------------------------------------------------------------------------------ XY mixer
def ring_xy_edges(n: int) -> list[tuple[int, int]]:
    """Ring edges in parity order: even pairs, odd pairs, then the closing edge for odd n."""
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    closing = [(n - 1, 0)] if n > 2 else []
    return even + odd + closing


class StatevectorQAOA:
    """Exact QAOA state for a QUBO, with the X mixer (|+>^n start) or the XY ring mixer (Dicke start)."""

    def __init__(self, Q: np.ndarray, reps: int, mixer: str = "x", k: int | None = None):
        self.n = Q.shape[0]
        self.reps = reps
        self.mixer = mixer
        self.E = all_energies(Q)
        idx = np.arange(2 ** self.n)
        if mixer == "x":
            self.psi0 = np.full(2 ** self.n, 2 ** (-self.n / 2), dtype=complex)
        elif mixer == "xy":
            if k is None:
                raise ValueError("the XY mixer needs the panel size k for its Dicke initial state")
            self.psi0 = dicke_state(self.n, k)
            self.pairs = []
            for a, b in ring_xy_edges(self.n):
                i01 = idx[(((idx >> a) & 1) == 0) & (((idx >> b) & 1) == 1)]
                self.pairs.append((i01, i01 ^ (1 << a) ^ (1 << b)))
        else:
            raise ValueError(f"unknown mixer {mixer!r}")

    def _x_mixer(self, psi, beta):
        c, s = np.cos(beta), np.sin(beta)
        t = psi.reshape((2,) * self.n)
        for ax in range(self.n):
            t = c * t - 1j * s * np.flip(t, axis=ax)
        return t.reshape(-1)

    def _xy_mixer(self, psi, beta):
        c, s = np.cos(beta), np.sin(beta)
        psi = psi.copy()
        for i01, i10 in self.pairs:
            a, b = psi[i01].copy(), psi[i10].copy()
            psi[i01] = c * a - 1j * s * b
            psi[i10] = c * b - 1j * s * a
        return psi

    def state(self, params) -> np.ndarray:
        p = self.reps
        beta, gamma = np.asarray(params[:p]), np.asarray(params[p:])
        psi = self.psi0.copy()
        for layer in range(p):
            psi = psi * np.exp(-1j * gamma[layer] * self.E)
            psi = self._x_mixer(psi, beta[layer]) if self.mixer == "x" else self._xy_mixer(psi, beta[layer])
        return psi

    def probabilities(self, params) -> np.ndarray:
        pr = np.abs(self.state(params)) ** 2
        return pr / pr.sum()

    def sample_counts(self, params, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Shot counts per basis state (length 2^n) with fresh randomness from rng."""
        return rng.multinomial(shots, self.probabilities(params))

    def sample_counts_fixed_seed(self, params, shots: int, seed: int) -> np.ndarray:
        """Shot counts the way AerSimulator(seed_simulator=seed) samples: the same sorted uniform random
        numbers every run, walked along the cumulative distribution. Counts therefore change smoothly with
        the angles (common random numbers), which is what COBYLA saw in the committed Aer runs. This choice
        reproduces those runs (results/statevector_validation.csv); fresh noise per call or a seeded
        multinomial do not."""
        pr = self.probabilities(params)
        u = np.sort(np.random.default_rng(seed).random(shots))
        idx = np.minimum(np.searchsorted(np.cumsum(pr), u, side="right"), pr.size - 1)
        return np.bincount(idx, minlength=pr.size)
