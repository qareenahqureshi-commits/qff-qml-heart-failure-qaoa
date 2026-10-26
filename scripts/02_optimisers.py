"""Step 3/4: build the feature-selection QUBO for k in {2,3,4}, two formulations (MI, Fisher/cov),
and compare solvers under identical optimiser metrics:
  exhaustive (ground truth), random sampling, simulated annealing, tabu search, QAOA p=1,2,3.
Each stochastic solver: 10 independent seeds. QAOA: 2000 shots, COBYLA <= 150 evaluations.

Outputs: results/optimisers.csv (one row per solver x seed), results/qubo_optima.csv,
         results/qaoa_resources.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.qaoa import QAOASolver, top_m_contains_optimum
from src.qubo import make_qubo
from src.solvers import (optimiser_metrics, random_sampling, simulated_annealing, tabu_search)

RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)
KS = [2, 3, 4]
VARIANTS = ["mi", "fisher"]
SEEDS = list(range(10))
N_SAMPLES = 2000          # same sample budget for every solver
QAOA_P = [1, 2, 3]
SHOTS, MAXITER = 2000, 150

X, y, feats = load()
rows, optima, resources = [], [], []

for variant in VARIANTS:
    for k in KS:
        q = make_qubo(X, y, k, variant)
        x_opt, e_opt, allE = q.exhaustive()
        e_max = max(allE.values())
        # rank of the optimum gap: how many bitstrings are within 1% of the optimum (degeneracy)
        near = sum(1 for e in allE.values() if (e - e_opt) <= 0.01 * (e_max - e_opt))
        opt_feats = [f for f, b in zip(feats, x_opt) if b]
        optima.append(dict(variant=variant, k=k, alpha=round(q.alpha, 5), E_opt=e_opt, E_max=e_max,
                           n_within_1pct=near, optimum="+".join(opt_feats)))
        print(f"[{variant} k={k}] alpha={q.alpha:.4f} optimum={opt_feats} E*={e_opt:.4f} (near-degenerate: {near})")

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            for name, fn in [("random", random_sampling), ("simulated_annealing", simulated_annealing),
                             ("tabu", tabu_search)]:
                t = time.perf_counter()
                out = fn(q, N_SAMPLES if name == "random" else 50, rng)  # SA/tabu: 50 restarts each
                dt = time.perf_counter() - t
                m = optimiser_metrics(q, out["samples"], out["energies"], x_opt, e_opt, e_max)
                m["top5_contains_opt"] = top_m_contains_optimum(out["samples"], out["energies"], x_opt, 5)
                rows.append(dict(variant=variant, k=k, solver=name, p=0, seed=seed, wall_s=dt, n_evals=len(out["energies"]), **m))

            for p in QAOA_P:
                t = time.perf_counter()
                s = QAOASolver(q, reps=p, shots=SHOTS, maxiter=MAXITER, seed=seed)
                out = s.run(rng)
                dt = time.perf_counter() - t
                m = optimiser_metrics(q, out["samples"], out["energies"], x_opt, e_opt, e_max)
                m["top5_contains_opt"] = top_m_contains_optimum(out["samples"], out["energies"], x_opt, 5)
                rows.append(dict(variant=variant, k=k, solver="qaoa", p=p, seed=seed, wall_s=dt, n_evals=out["n_evals"], **m))
                if seed == 0:
                    resources.append(dict(variant=variant, k=k, **s.resources()))
            print(f"   seed {seed} done ({time.strftime('%H:%M:%S')})", flush=True)

pd.DataFrame(rows).to_csv(RES / "optimisers.csv", index=False)
pd.DataFrame(optima).to_csv(RES / "qubo_optima.csv", index=False)
pd.DataFrame(resources).to_csv(RES / "qaoa_resources.csv", index=False)

df = pd.DataFrame(rows)
summ = (df.groupby(["variant", "k", "solver", "p"])
          .agg(AR_mean=("approx_ratio_mean", "mean"), AR_std=("approx_ratio_mean", "std"),
               AR_best=("approx_ratio_best", "mean"), P_opt=("p_optimum", "mean"),
               feasibility=("feasibility", "mean"), jaccard=("jaccard_best_vs_opt", "mean"),
               found_opt_frac=("jaccard_best_vs_opt", lambda s: (s == 1).mean()),
               top5_frac=("top5_contains_opt", "mean"),
               wall_s=("wall_s", "mean")).round(3).reset_index())
summ.to_csv(RES / "optimisers_summary.csv", index=False)
print(summ.to_string(index=False))
