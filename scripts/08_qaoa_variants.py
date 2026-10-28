"""QAOA improvement study on the Fisher QUBO (the formulation the README takes forward).

k = 2, 3, 4; depth p = 1, 2, 3; 10 seeds; 2000 shots; COBYLA <= 150 evaluations, as in 02_optimisers.py.

Variants (for a given k and seed, every variant that starts from random angles starts from the SAME angles,
so the comparison is paired):
  baseline        X mixer, mean energy objective, random initial angles (the original setting)
  baseline_2starts  baseline run from two random starting points, keeping the run with the lower final
                  objective: an equal evaluation budget control for INTERP, whose p = 3 chain also runs p = 1, 2
  cvar            X mixer, CVaR objective = mean of the best 10% of shots (Barkoutsos et al. 2020)
  interp          X mixer, mean objective; p = 1 from the random angles, p >= 2 from INTERP of the
                  optimised p - 1 angles (Zhou et al. 2020)
  xy              Dicke state + ring XY mixer (every shot has exactly k features), mean objective, random angles
  xy_interp       XY mixer with INTERP initialisation
  xy_cvar_interp  all three changes together
Classical references on the same QUBOs: uniform random bitstrings and uniform random k panels (2000 each).

Backend: QFF_BACKEND=statevector (exact numpy simulation, no Qiskit needed) or aer (default).
Outputs: results/qaoa_variants.csv (per run), results/qaoa_variants_summary.csv,
         results/qaoa_variant_resources.csv (logical gate counts; written only when Qiskit is installed)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.qaoa import BACKEND_ENV, QAOASolver, interp_params
from src.qubo import make_qubo
from src.solvers import optimiser_metrics, random_feasible, random_sampling, summarise_optimiser_runs

RES = Path(__file__).resolve().parents[1] / "results"
KS, PS, SEEDS = [2, 3, 4], [1, 2, 3], list(range(10))
SHOTS, MAXITER, CVAR_ALPHA = 2000, 150, 0.1
BACKEND = os.environ.get(BACKEND_ENV, "aer")
VARIANTS = {  # name: (mixer, objective, init)
    "baseline": ("x", "mean", "random"),
    "baseline_2starts": ("x", "mean", "random2"),
    "cvar": ("x", "cvar", "random"),
    "interp": ("x", "mean", "interp"),
    "xy": ("xy", "mean", "random"),
    "xy_interp": ("xy", "mean", "interp"),
    "xy_cvar_interp": ("xy", "cvar", "interp"),
}

X, y, feats = load()
rows = []
t_start = time.perf_counter()
for k in KS:
    q = make_qubo(X, y, k, "fisher")
    x_opt, e_opt, allE = q.exhaustive()
    e_max = max(allE.values())
    for seed in SEEDS:
        rng = np.random.default_rng(10_000 + 100 * k + seed)
        x0 = {p: rng.uniform(0, np.pi, size=2 * p) for p in PS}
        x0b = {p: rng.uniform(0, np.pi, size=2 * p) for p in PS}  # second start for baseline_2starts
        for name, fn in [("random", random_sampling), ("random_feasible", random_feasible)]:
            out = fn(q, SHOTS, np.random.default_rng(20_000 + 100 * k + seed))
            m = optimiser_metrics(q, out["samples"], out["energies"], x_opt, e_opt, e_max)
            rows.append(dict(k=k, variant=name, p=0, seed=seed, backend="classical", n_evals=SHOTS, n_evals_chain=SHOTS,
                             wall_s=0.0, **m))
        for name, (mixer, objective, init) in VARIANTS.items():
            prev, chain = None, 0
            for p in PS:
                start = interp_params(prev, p - 1) if (init == "interp" and p > 1) else x0[p]
                t = time.perf_counter()
                runs = []
                for x_start in ([start, x0b[p]] if init == "random2" else [start]):
                    s = QAOASolver(q, reps=p, shots=SHOTS, maxiter=MAXITER, seed=seed, objective=objective,
                                   cvar_alpha=CVAR_ALPHA, mixer=mixer, backend=BACKEND)
                    runs.append(s.run(rng, x0=x_start))
                out = min(runs, key=lambda r: r["final_expectation"])
                out["n_evals"] = sum(r["n_evals"] for r in runs)
                dt = time.perf_counter() - t
                prev, chain = out["opt_params"], (chain + out["n_evals"]) if init == "interp" else out["n_evals"]
                m = optimiser_metrics(q, out["samples"], out["energies"], x_opt, e_opt, e_max)
                rows.append(dict(k=k, variant=name, p=p, seed=seed, backend=BACKEND, n_evals=out["n_evals"],
                                 n_evals_chain=chain, wall_s=dt, **m))
    print(f"k={k} done ({time.perf_counter() - t_start:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "qaoa_variants.csv", index=False)
summ = summarise_optimiser_runs(df, ["k", "variant", "p"])
summ = summ.merge(df.groupby(["k", "variant", "p"]).n_evals_chain.mean().rename("n_evals_chain").reset_index(),
                  on=["k", "variant", "p"])
order = {v: i for i, v in enumerate(["random", "random_feasible", *VARIANTS])}
summ = summ.sort_values(["k", "p", "variant"], key=lambda c: c.map(order) if c.name == "variant" else c)
summ.to_csv(RES / "qaoa_variants_summary.csv", index=False)
pd.set_option("display.width", 250)
print(summ[["k", "variant", "p", "AR_mean", "AR_std", "AR_feasible", "P_opt", "enrich_vs_uniform", "enrich_vs_random_panel",
            "feasibility", "found_opt_frac", "top5_by_count_frac", "n_evals", "n_evals_chain"]].to_string(index=False))

try:
    from src.qaoa import logical_resources
    q3 = make_qubo(X, y, 3, "fisher")
    res = [dict(k=3, **logical_resources(q3, p, mixer)) for mixer in ("x", "xy") for p in PS]
    pd.DataFrame(res).to_csv(RES / "qaoa_variant_resources.csv", index=False)
    print(pd.DataFrame(res).to_string(index=False))
except ImportError:
    print("Qiskit not installed: skipped logical gate counts (results/qaoa_variant_resources.csv)")
