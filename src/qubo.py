"""QUBO formulations for feature selection.

Mücke, Heese, Müller, Wolter, Piatkowski (2023), "Feature selection on quantum computers",
Quantum Machine Intelligence 5:11 — the "QFS" objective:

    Q(x, a) = -a * sum_i I_i x_i + (1 - a) * sum_{i != j} R_ij x_i x_j,   x in {0,1}^n

  I_i  = MI(feature_i ; label)      relevance  (maximised)
  R_ij = MI(feature_i ; feature_j)  redundancy (minimised), R_ii = 0
  a in [0, 1] interpolates. Proposition 1 of the paper: for every k there is an a whose
  minimiser has exactly k ones, so the number of features is controlled by a binary search
  on a instead of an ill-conditioned penalty lambda*(sum x - k)^2.

Variant "fisher" (arXiv 2609.30256, 2026): relevance = Fisher score, redundancy = |covariance|,
both min-max normalised — no discretisation needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

N_BINS = 20  # quantile bins, as in Mücke et al.


def _discretise(X: pd.DataFrame, n_bins: int = N_BINS) -> np.ndarray:
    out = np.zeros(X.shape, dtype=int)
    for j, c in enumerate(X.columns):
        col = X[c].to_numpy()
        if len(np.unique(col)) <= n_bins:
            _, out[:, j] = np.unique(col, return_inverse=True)
        else:
            edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
            out[:, j] = np.digitize(col, edges)
    return out


def mi_terms(X: pd.DataFrame, y: np.ndarray):
    """Relevance vector I (n,) and redundancy matrix R (n,n) from mutual information."""
    D = _discretise(X)
    n = D.shape[1]
    I = np.array([mutual_info_score(D[:, i], y) for i in range(n)])
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            R[i, j] = R[j, i] = mutual_info_score(D[:, i], D[:, j])
    return I, R


def fisher_terms(X: pd.DataFrame, y: np.ndarray):
    """Relevance = Fisher score, redundancy = |cov|, each min-max normalised to [0,1]."""
    A = X.to_numpy()
    mu, n = A.mean(0), A.shape[1]
    num = np.zeros(n)
    den = np.zeros(n)
    for c in np.unique(y):
        Ac = A[y == c]
        num += len(Ac) * (Ac.mean(0) - mu) ** 2
        den += len(Ac) * Ac.var(0)
    I = num / np.where(den == 0, 1, den)
    I = (I - I.min()) / (I.max() - I.min() + 1e-12)
    Z = (A - mu) / (A.std(0) + 1e-12)
    R = np.abs(np.cov(Z, rowvar=False))
    np.fill_diagonal(R, 0)
    R = (R - R.min()) / (R.max() - R.min() + 1e-12)
    np.fill_diagonal(R, 0)
    return I, R


@dataclass
class QUBO:
    Q: np.ndarray          # symmetric (n,n); objective x^T Q x with x in {0,1}^n
    alpha: float
    k: int
    variant: str
    features: list[str]

    @property
    def n(self):
        return self.Q.shape[0]

    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x)

    def exhaustive(self):
        """Global optimum by brute force (2^n, fine for n <= ~20). Returns (x*, E*, all_E)."""
        best, bx = np.inf, None
        energies = {}
        for bits in product((0, 1), repeat=self.n):
            e = self.energy(bits)
            energies[bits] = e
            if e < best:
                best, bx = e, bits
        return np.array(bx), best, energies


def build_qubo(I: np.ndarray, R: np.ndarray, alpha: float) -> np.ndarray:
    """Q_ij(a) = (1-a) R_ij  (i != j),  Q_ii(a) = -a I_i."""
    Q = (1 - alpha) * R.copy()
    np.fill_diagonal(Q, -alpha * I)
    return Q


def alpha_for_k(I, R, k, tol=1e-6, max_iter=60):
    """Binary search on alpha so the exhaustive minimiser has exactly k features
    (Mücke et al., Prop. 1). Returns (alpha, x*)."""
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        Q = build_qubo(I, R, mid)
        x, _, _ = QUBO(Q, mid, k, "", []).exhaustive()
        m = int(x.sum())
        if m == k:
            return mid, x
        if m < k:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return mid, x


def make_qubo(X: pd.DataFrame, y: np.ndarray, k: int, variant: str = "mi") -> QUBO:
    I, R = mi_terms(X, y) if variant == "mi" else fisher_terms(X, y)
    alpha, _ = alpha_for_k(I, R, k)
    return QUBO(build_qubo(I, R, alpha), alpha, k, variant, list(X.columns))
