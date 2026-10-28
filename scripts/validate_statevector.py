"""Check that the numpy statevector backend reproduces the committed Qiskit Aer QAOA results.

Replays scripts/02_optimisers.py exactly (same QUBOs, same random stream, so every QAOA run starts from the
same initial angles as the committed Aer run), but simulates QAOA with backend="statevector". Only the shot
noise realisation differs. Compares the per configuration means with results/optimisers_summary.csv.

Output: results/statevector_validation.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.qaoa import QAOASolver
from src.qubo import make_qubo
from src.solvers import optimiser_metrics, random_sampling, simulated_annealing, tabu_search

RES = Path(__file__).resolve().parents[1] / "results"
X, y, feats = load()
rows = []
for variant in ["mi", "fisher"]:
    for k in [2, 3, 4]:
        q = make_qubo(X, y, k, variant)
        x_opt, e_opt, allE = q.exhaustive()
        e_max = max(allE.values())
        for seed in range(10):
            rng = np.random.default_rng(seed)
            random_sampling(q, 2000, rng)        # consume the stream exactly as 02_optimisers.py does
            simulated_annealing(q, 50, rng)
            tabu_search(q, 50, rng)
            for p in [1, 2, 3]:
                out = QAOASolver(q, reps=p, shots=2000, maxiter=150, seed=seed, backend="statevector").run(rng)
                m = optimiser_metrics(q, out["samples"], out["energies"], x_opt, e_opt, e_max)
                rows.append(dict(variant=variant, k=k, p=p, seed=seed, **m))
        print(variant, k, "done", flush=True)

sv = pd.DataFrame(rows).groupby(["variant", "k", "p"]).agg(
    AR_mean_sv=("approx_ratio_mean", "mean"), P_opt_sv=("p_optimum", "mean"),
    found_sv=("jaccard_best_vs_opt", lambda s: (s == 1).mean())).reset_index()
aer = pd.read_csv(RES / "optimisers_summary.csv")
aer = aer[aer.solver == "qaoa"][["variant", "k", "p", "AR_mean", "AR_std", "P_opt", "found_opt_frac"]].rename(
    columns={"AR_mean": "AR_mean_aer", "AR_std": "AR_std_aer", "P_opt": "P_opt_aer", "found_opt_frac": "found_aer"})
cmp = aer.merge(sv, on=["variant", "k", "p"])
cmp["AR_diff"] = cmp.AR_mean_sv - cmp.AR_mean_aer
cmp["AR_diff_in_std_errors"] = cmp.AR_diff / (cmp.AR_std_aer / np.sqrt(10))
cmp = cmp.round(4)
cmp.to_csv(RES / "statevector_validation.csv", index=False)
print(cmp.to_string(index=False))
print("max |AR diff| =", cmp.AR_diff.abs().max(), " max |P_opt diff| =", (cmp.P_opt_sv - cmp.P_opt_aer).abs().max())
