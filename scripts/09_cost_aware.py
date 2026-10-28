"""Cost aware panel selection: add what the measurements cost to the QUBO.

  minimise  Q_cost(x) = Q_fisher(x; alpha_k) + lambda * cost(x) / cost(all tests)

cost(x) is the dollar cost of the distinct tests the panel needs (src/costs.py); it is exactly quadratic, so it
is just more QUBO terms. lambda = 0 is the original problem; large lambda prefers cheap panels.

Two things change once cost enters:
  * the alpha-for-k trick (Mücke et al. Prop. 1) no longer guarantees exactly k features, because the cost
    terms are not part of that argument. This script reports the size of the unconstrained optimum.
  * so the panel size is enforced by the XY mixer instead: QAOA starts in the Dicke state of all k panels
    and the XY mixer never leaves that subspace.

Steps:
  1. for k = 2, 3, 4, sweep lambda from 0 to 1.5 and record the optimal k panel (exhaustive over k panels);
  2. evaluate every distinct panel with the random forest protocol (100 splits; reused from
     results/downstream.csv when available) and compare it with the k = 3 panel (corrected paired test);
  3. run XY mixer QAOA (p = 2, INTERP start from p = 1, 10 seeds) on the cost QUBO at lambda = 0 and 1.

Backend for QAOA: QFF_BACKEND=aer (default) or statevector.
Outputs: results/cost_path.csv, results/cost_panels.csv, results/cost_qaoa.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.costs import cost_matrix, panel_cost, tests_for, total_cost
from src.data import load
from src.evaluate import N_PATIENTS, TEST_SIZE, canonical, evaluate_subset
from src.qaoa import QAOASolver, interp_params
from src.qubo import QUBO, make_qubo
from src.solvers import optimiser_metrics
from src.stats import ci_halfwidth, paired_test, split_sizes

RES = Path(__file__).resolve().parents[1] / "results"
LAMBDAS = np.round(np.arange(0, 1.5001, 0.01), 2)
QAOA_LAMBDAS, SEEDS = [0.0, 1.0], list(range(10))
REFERENCE = "age+ejection_fraction+serum_creatinine"

X, y, feats = load()
Cn = cost_matrix(feats) / total_cost()
n_train, n_test = split_sizes(N_PATIENTS, TEST_SIZE)


def best_k_panel(Q, k):
    best, bx = np.inf, None
    for idx in itertools.combinations(range(len(feats)), k):
        x = np.zeros(len(feats))
        x[list(idx)] = 1
        e = x @ Q @ x
        if e < best:
            best, bx = e, x
    return bx.astype(int), best


# ---------------------------------------------------------------- 1. lambda path
path, cost_qubos = [], {}
for k in [2, 3, 4]:
    q = make_qubo(X, y, k, "fisher")
    prev = None
    for lam in LAMBDAS:
        Q = q.Q + lam * Cn
        x, e = best_k_panel(Q, k)
        panel = [f for f, b in zip(feats, x) if b]
        x_free, _, _ = QUBO(Q, q.alpha, k, "fisher_cost", feats).exhaustive()
        key = (tuple(panel), int(x_free.sum()))
        if key != prev:
            path.append(dict(k=k, lambda_from=lam, panel="+".join(panel), cost_usd=panel_cost(panel),
                             tests="+".join(tests_for(panel)), unconstrained_optimum_size=int(x_free.sum())))
            prev = key
        cost_qubos[(k, float(lam))] = (q, Q)
path = pd.DataFrame(path)
path["lambda_to"] = path.groupby("k").lambda_from.shift(-1).fillna(LAMBDAS[-1])
path.to_csv(RES / "cost_path.csv", index=False)
print(path.to_string(index=False))

# ---------------------------------------------------------------- 2. clinical performance of each panel
long = pd.read_csv(RES / "downstream.csv")
rf_long = long[long.model == "random_forest"].drop_duplicates(["features", "run"])
by_panel = {f: g.sort_values("run").MCC.to_numpy() for f, g in rf_long.groupby("features")}
rows = []
for panel in sorted(set(path.panel) | {REFERENCE}):
    fs = canonical(panel.split("+"))
    key = "+".join(fs)
    if key not in by_panel:
        print("evaluating", key, flush=True)
        by_panel[key] = evaluate_subset(X, y, fs, models=["random_forest"]).sort_values("run").MCC.to_numpy()
    mcc = by_panel[key]
    row = dict(panel=key, k=len(fs), cost_usd=panel_cost(fs), needs_echo="ejection_fraction" in fs,
               MCC=mcc.mean(), MCC_ci95_corrected=ci_halfwidth(mcc, n_train, n_test))
    if key != REFERENCE:
        t = paired_test(mcc, by_panel[REFERENCE], n_train, n_test)
        row.update(delta_vs_k3_panel=t["delta"], p_corrected_vs_k3_panel=t["p_corrected"])
    rows.append(row)
panels = pd.DataFrame(rows).sort_values(["k", "cost_usd"]).round(4)
panels.to_csv(RES / "cost_panels.csv", index=False)
print(panels.to_string(index=False))

# ---------------------------------------------------------------- 3. XY mixer QAOA on the cost QUBO
qrows = []
for k in [2, 3, 4]:
    for lam in QAOA_LAMBDAS:
        q0, Q = cost_qubos[(k, lam)]
        qc = QUBO(Q, q0.alpha, k, "fisher_cost", feats)
        x_opt, e_opt = best_k_panel(Q, k)
        e_max = max(qc.energy(b) for b in itertools.product((0, 1), repeat=len(feats)))
        for seed in SEEDS:
            rng = np.random.default_rng(30_000 + 100 * k + seed)
            prev = None
            for p in [1, 2]:
                x0 = rng.uniform(0, np.pi, size=2) if p == 1 else interp_params(prev, 1)
                out = QAOASolver(qc, reps=p, shots=2000, maxiter=150, seed=seed, mixer="xy").run(rng, x0=x0)
                prev = out["opt_params"]
            m = optimiser_metrics(qc, out["samples"], out["energies"], x_opt, e_opt, e_max)
            qrows.append(dict(k=k, lam=lam, seed=seed, optimum="+".join(f for f, b in zip(feats, x_opt) if b), **m))
qdf = pd.DataFrame(qrows)
qdf.to_csv(RES / "cost_qaoa.csv", index=False)
qs = (qdf.groupby(["k", "lam", "optimum"])
      .agg(P_opt=("p_optimum", "mean"), P_opt_std=("p_optimum", "std"), enrich_vs_random_panel=("enrichment_feasible", "mean"),
           feasibility=("feasibility", "mean"), found_opt_frac=("jaccard_best_vs_opt", lambda s: (s == 1).mean()),
           top5_by_count_frac=("top5_by_count", "mean"))
      .reset_index().round(4))
print(qs.to_string(index=False))
