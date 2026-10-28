"""Classical solvers / baselines for the feature-selection QUBO and classical FS methods.

All QUBO solvers return a dict with the sampled bitstrings so the same optimiser metrics
(approximation ratio, P(optimum), feasibility, Jaccard) can be computed for every solver.

Two random baselines:
  * random_sampling: uniform over all 2^n bitstrings (the original lower bound)
  * random_feasible: uniform over the C(n, k) panels of exactly k features. This is the fair baseline for
    a sampler, because with k fixed there are only 165 panels at k = 3. "Found the optimum somewhere in
    2000 samples" is nearly guaranteed for it, which is why the enrichment columns below are reported.
"""
from __future__ import annotations

import time
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .qubo import QUBO, mi_terms


# ----------------------------------------------------------------------------- QUBO solvers
def random_sampling(q: QUBO, n_samples: int, rng: np.random.Generator):
    xs = rng.integers(0, 2, size=(n_samples, q.n))
    return {"samples": xs, "energies": np.array([q.energy(x) for x in xs])}


def random_feasible(q: QUBO, n_samples: int, rng: np.random.Generator):
    """Uniform over panels with exactly q.k features."""
    pick = np.argsort(rng.random((n_samples, q.n)), axis=1)[:, :q.k]
    xs = np.zeros((n_samples, q.n), dtype=int)
    np.put_along_axis(xs, pick, 1, axis=1)
    return {"samples": xs, "energies": np.array([q.energy(x) for x in xs])}


def simulated_annealing(q: QUBO, n_samples: int, rng: np.random.Generator,
                        n_sweeps: int = 200, T0: float = None, T1: float = None):
    """Single-flip Metropolis SA on the QUBO. Tuned with the same care as QAOA:
    geometric schedule from T0 (accept most flips) to T1 (accept almost none)."""
    Q = q.Q
    dQ = np.abs(Q).sum(1).max()
    T0 = T0 or dQ
    T1 = T1 or dQ / 1000
    temps = T0 * (T1 / T0) ** (np.arange(n_sweeps) / max(1, n_sweeps - 1))
    xs, es = [], []
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=q.n)
        for T in temps:
            for i in rng.permutation(q.n):
                # energy change for flipping bit i
                delta = (1 - 2 * x[i]) * (Q[i, i] + 2 * np.dot(Q[i], x) - 2 * Q[i, i] * x[i])
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    x[i] ^= 1
        xs.append(x.copy())
        es.append(q.energy(x))
    return {"samples": np.array(xs), "energies": np.array(es)}


def tabu_search(q: QUBO, n_samples: int, rng: np.random.Generator, n_iters: int = 200, tenure: int = 4):
    Q = q.Q
    xs, es = [], []
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=q.n)
        best_x, best_e = x.copy(), q.energy(x)
        tabu = np.zeros(q.n, dtype=int)
        for it in range(n_iters):
            deltas = np.array([(1 - 2 * x[i]) * (Q[i, i] + 2 * np.dot(Q[i], x) - 2 * Q[i, i] * x[i]) for i in range(q.n)])
            cur = q.energy(x)
            allowed = (tabu <= it) | (cur + deltas < best_e)  # aspiration
            cand = np.where(allowed, deltas, np.inf)
            i = int(np.argmin(cand))
            x[i] ^= 1
            tabu[i] = it + tenure
            e = cur + deltas[i]
            if e < best_e:
                best_e, best_x = e, x.copy()
        xs.append(best_x)
        es.append(best_e)
    return {"samples": np.array(xs), "energies": np.array(es)}


# ------------------------------------------------------------------- classical FS baselines
def mrmr(X: pd.DataFrame, y: np.ndarray, k: int) -> list[str]:
    """Greedy mRMR (MID form) on the same MI quantities as the QUBO."""
    I, R = mi_terms(X, y)
    sel = [int(np.argmax(I))]
    while len(sel) < k:
        rest = [j for j in range(len(I)) if j not in sel]
        score = [I[j] - np.mean([R[j, s] for s in sel]) for j in rest]
        sel.append(rest[int(np.argmax(score))])
    return [X.columns[i] for i in sel]


def lasso_topk(X: pd.DataFrame, y: np.ndarray, k: int) -> list[str]:
    Z = StandardScaler().fit_transform(X)
    for C in np.logspace(-3, 1, 200):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000).fit(Z, y)
        nz = np.flatnonzero(m.coef_[0])
        if len(nz) >= k:
            order = nz[np.argsort(-np.abs(m.coef_[0][nz]))]
            return [X.columns[i] for i in order[:k]]
    return list(X.columns[:k])


def rfe_topk(X: pd.DataFrame, y: np.ndarray, k: int, seed: int = 0) -> list[str]:
    sel = RFE(RandomForestClassifier(n_estimators=200, random_state=seed), n_features_to_select=k).fit(X, y)
    return [c for c, s in zip(X.columns, sel.support_) if s]


def rf_importance_topk(X: pd.DataFrame, y: np.ndarray, k: int, seed: int = 0) -> list[str]:
    rf = RandomForestClassifier(n_estimators=500, random_state=seed).fit(X, y)
    order = np.argsort(-rf.feature_importances_)[:k]
    return [X.columns[i] for i in order]


# ---------------------------------------------------------------------------- metrics
def optimiser_metrics(q: QUBO, samples: np.ndarray, energies: np.ndarray, x_opt: np.ndarray, e_opt: float,
                      e_max: float) -> dict:
    """Approximation ratio (normalised so optimum -> 1, worst -> 0), P(optimum), feasibility
    (fraction of samples with exactly k ones), Jaccard overlap of best sample with x_opt.

    enrichment_uniform  = P(optimum) / (1 / 2^n): how much more often than a uniform random bitstring
    enrichment_feasible = P(optimum) / (1 / C(n,k)): how much more often than a uniform random k panel
    top5_by_count       = optimum among the 5 most frequently sampled distinct subsets
    approx_ratio_feasible_mean = mean approximation ratio over the shots that are valid k panels, normalised
        on the k panel set (worst k panel -> 0, optimum -> 1). Unlike approx_ratio_mean it gives no credit for
        near optimal panels of the wrong size, so it compares the X mixer and the XY mixer fairly.
    """
    ar = (e_max - energies) / (e_max - e_opt)
    opt_hits = np.all(samples == x_opt, axis=1)
    feasible = samples.sum(1) == q.k
    best = samples[int(np.argmin(energies))]
    inter = np.sum((best == 1) & (x_opt == 1))
    union = np.sum((best == 1) | (x_opt == 1))
    keys, counts = np.unique(samples, axis=0, return_counts=True)
    top = keys[np.argsort(-counts, kind="stable")[:5]]
    p_opt = float(opt_hits.mean())
    ef_max = _worst_panel_energy(q)
    fe = energies[feasible]
    ar_feas = float(((ef_max - fe) / (ef_max - e_opt)).mean()) if feasible.any() else float("nan")
    return {
        "approx_ratio_mean": float(ar.mean()),
        "approx_ratio_feasible_mean": ar_feas,
        "approx_ratio_best": float(ar.max()),
        "p_optimum": p_opt,
        "enrichment_uniform": p_opt * 2 ** q.n,
        "enrichment_feasible": p_opt * comb(q.n, q.k),
        "feasibility": float(feasible.mean()),
        "jaccard_best_vs_opt": float(inter / union) if union else 1.0,
        "top5_by_count": bool(np.any(np.all(top == x_opt, axis=1))),
        "best_energy": float(energies.min()),
        "best_features": "+".join(f for f, b in zip(q.features, best) if b),
    }


def _worst_panel_energy(q: QUBO) -> float:
    """Highest QUBO energy among panels of exactly q.k features (cached on the QUBO object)."""
    cache = getattr(q, "_worst_panel_energy", None)
    if cache is None or cache[0] is not q.Q:
        worst = -np.inf
        for idx in combinations(range(q.n), q.k):
            x = np.zeros(q.n)
            x[list(idx)] = 1
            worst = max(worst, q.energy(x))
        q._worst_panel_energy = cache = (q.Q, worst)
    return cache[1]


def summarise_optimiser_runs(df: pd.DataFrame, group_cols: list[str], n: int = 11) -> pd.DataFrame:
    """Mean over seeds per configuration. Enrichment is computed from the mean P(optimum) so that it reads
    as "this solver samples the optimum E times more often than the random baseline"."""
    agg = dict(AR_mean=("approx_ratio_mean", "mean"), AR_std=("approx_ratio_mean", "std"),
               AR_best=("approx_ratio_best", "mean"), P_opt=("p_optimum", "mean"), P_opt_std=("p_optimum", "std"),
               feasibility=("feasibility", "mean"), jaccard=("jaccard_best_vs_opt", "mean"),
               found_opt_frac=("jaccard_best_vs_opt", lambda s: (s == 1).mean()))
    if "approx_ratio_feasible_mean" in df:
        agg["AR_feasible"] = ("approx_ratio_feasible_mean", "mean")
    if "top5_contains_opt" in df:
        agg["top5_frac"] = ("top5_contains_opt", "mean")
    if "top5_by_count" in df:
        agg["top5_by_count_frac"] = ("top5_by_count", "mean")
    if "n_evals" in df:
        agg["n_evals"] = ("n_evals", "mean")
    agg["wall_s"] = ("wall_s", "mean")
    out = df.groupby(group_cols).agg(**agg).reset_index()
    out["enrich_vs_uniform"] = out.P_opt * 2 ** n
    out["enrich_vs_random_panel"] = out.P_opt * out.k.map(lambda k: comb(n, int(k)))
    return out.round(4)


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t
