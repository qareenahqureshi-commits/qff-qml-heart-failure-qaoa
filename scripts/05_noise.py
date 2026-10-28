"""Noise study: what happens to the QAOA selector on a realistic device?

Backend: qiskit-ibm-runtime `fake_torino` (IBM Heron r1, 133 qubits, heavy-hex), i.e. a simulator with that
device's calibration noise model; circuits routed to its coupling map at optimization_level=3 (so SWAP
overhead is real and counted).

Two protocols, both on the Fisher QUBO (the better-conditioned one):
  A. Parameter transfer (p = 1, 2, 3; 10 seeds): optimise angles on the ideal simulator, then sample the
     noisy device once with those angles. Standard practice when hardware time is scarce.
  B. Fully noisy optimisation (p = 1, k = 3; 3 seeds; 1000 shots; <=25 COBYLA evals): COBYLA sees the noisy objective.

Outputs results/noise.csv, results/noise_summary.csv, results/noise_resources.csv
`python scripts/05_noise.py --summarise-only` rebuilds noise_summary.csv (with enrichment columns) from noise.csv.
Note: top5_contains_opt equals "found" by construction (the optimum has the lowest energy of all bitstrings).
"""
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.qubo import make_qubo
from src.solvers import optimiser_metrics

RES = Path(__file__).resolve().parents[1] / "results"
BACKEND = "fake_torino"


def counts_to_arrays(q, counts):
    from src.qaoa import _bits
    samples, energies = [], []
    for k, c in counts.items():
        b = _bits(k, q.n)
        samples += [b] * c
        energies += [q.energy(b)] * c
    return np.array(samples), np.array(energies)


if "--summarise-only" in sys.argv:
    df = pd.read_csv(RES / "noise.csv")
else:
    from src.qaoa import QAOASolver, top_m_contains_optimum

    X, y, feats = load()
    rows, resources = [], []
    for k in [2, 3, 4]:
        q = make_qubo(X, y, k, "fisher")
        x_opt, e_opt, allE = q.exhaustive()
        e_max = max(allE.values())

        # --- A: parameter transfer ---------------------------------------------------------
        for p in [1, 2, 3]:
            for seed in range(10):
                rng = np.random.default_rng(seed)
                ideal = QAOASolver(q, reps=p, shots=2000, maxiter=150, seed=seed)
                out = ideal.run(rng)
                noisy = QAOASolver(q, reps=p, shots=2000, maxiter=0, seed=seed, noisy=BACKEND)
                if seed == 0:
                    resources.append(dict(k=k, **noisy.resources()))
                t = time.perf_counter()
                counts = noisy._sample(out["opt_params"])
                dt = time.perf_counter() - t
                s, e = counts_to_arrays(q, counts)
                m = optimiser_metrics(q, s, e, x_opt, e_opt, e_max)
                m_ideal = optimiser_metrics(q, out["samples"], out["energies"], x_opt, e_opt, e_max)
                rows.append(dict(k=k, protocol="transfer", p=p, seed=seed, wall_s=dt,
                                 top5_contains_opt=top_m_contains_optimum(s, e, x_opt, 5),
                                 ideal_AR_mean=m_ideal["approx_ratio_mean"], ideal_p_opt=m_ideal["p_optimum"], **m))
                print(f"k={k} p={p} seed={seed} transfer: AR {m['approx_ratio_mean']:.3f} (ideal {m_ideal['approx_ratio_mean']:.3f}) "
                      f"P(opt) {m['p_optimum']:.3f} top5 {rows[-1]['top5_contains_opt']}  [{dt:.0f}s]", flush=True)

        # --- B: fully noisy optimisation, p = 1, k = 3 only (each evaluation ~12 s) ---------------
        for seed in range(3 if k == 3 else 0):
            rng = np.random.default_rng(100 + seed)
            t = time.perf_counter()
            noisy = QAOASolver(q, reps=1, shots=1000, maxiter=25, seed=seed, noisy=BACKEND)
            out = noisy.run(rng)
            dt = time.perf_counter() - t
            m = optimiser_metrics(q, out["samples"], out["energies"], x_opt, e_opt, e_max)
            rows.append(dict(k=k, protocol="noisy_opt", p=1, seed=seed, wall_s=dt,
                             top5_contains_opt=top_m_contains_optimum(out["samples"], out["energies"], x_opt, 5),
                             ideal_AR_mean=np.nan, ideal_p_opt=np.nan, **m))
            print(f"k={k} p=1 seed={seed} noisy-opt: AR {m['approx_ratio_mean']:.3f} P(opt) {m['p_optimum']:.3f} "
                  f"top5 {rows[-1]['top5_contains_opt']} evals {out['n_evals']} [{dt:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "noise.csv", index=False)
    pd.DataFrame(resources).to_csv(RES / "noise_resources.csv", index=False)

summ = (df.groupby(["k", "protocol", "p"])
          .agg(AR_mean=("approx_ratio_mean", "mean"), AR_std=("approx_ratio_mean", "std"),
               ideal_AR_mean=("ideal_AR_mean", "mean"), AR_best=("approx_ratio_best", "mean"),
               P_opt=("p_optimum", "mean"), ideal_P_opt=("ideal_p_opt", "mean"),
               feasibility=("feasibility", "mean"),
               found_opt_frac=("jaccard_best_vs_opt", lambda s: (s == 1).mean()),
               top5_frac=("top5_contains_opt", "mean"), wall_s=("wall_s", "mean")).reset_index())
summ["enrich_vs_uniform"] = summ.P_opt * 2 ** 11
summ["enrich_vs_random_panel"] = summ.P_opt * summ.k.map(lambda k: comb(11, int(k)))
summ["ideal_enrich_vs_random_panel"] = summ.ideal_P_opt * summ.k.map(lambda k: comb(11, int(k)))
summ = summ.round(3)
summ.to_csv(RES / "noise_summary.csv", index=False)
print(summ.to_string(index=False))
