"""Step 4: downstream clinical performance of every selected feature subset under the
*same* 100-split protocol as the baseline (random forest + gradient boosting + LR + SVM).

Subsets compared for k in {2,3,4}:
  * QUBO optimum (MI and Fisher/cov formulations) -- the target every solver tries to find
  * QAOA best-found subset (per p, most frequent best subset across seeds)
  * classical FS: mRMR (same MI quantities), Lasso-LR, RFE(RF), RF importance
  * reference: all 11 features; Chicco & Jurman two-feature panel
Outputs results/downstream.csv (long) and results/downstream_summary.csv

`python scripts/03_downstream.py --summarise-only` rebuilds downstream_summary.csv from an existing
downstream.csv without refitting any classifier (used to add the corrected CI column).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.evaluate import evaluate_subset, summarise
from src.solvers import lasso_topk, mrmr, rfe_topk, rf_importance_topk

RES = Path(__file__).resolve().parents[1] / "results"
X, y, feats = load()

if "--summarise-only" in sys.argv:
    long = pd.read_csv(RES / "downstream.csv")
else:
    opt = pd.read_csv(RES / "qubo_optima.csv")
    optim = pd.read_csv(RES / "optimisers.csv")

    subsets = {("reference", "all_features", 11): feats,
               ("reference", "chicco_two_feature", 2): ["serum_creatinine", "ejection_fraction"]}
    for _, r in opt.iterrows():
        subsets[("qubo_optimum", r.variant, int(r.k))] = r.optimum.split("+")
    # QAOA: the modal best-found subset over seeds for each (variant, k, p)
    qa = optim[optim.solver == "qaoa"]
    for (v, k, p), g in qa.groupby(["variant", "k", "p"]):
        modal = g.best_features.mode().iloc[0]
        subsets[(f"qaoa_p{p}", v, int(k))] = modal.split("+")
    for k in [2, 3, 4]:
        subsets[("classical_fs", "mrmr", k)] = mrmr(X, y, k)
        subsets[("classical_fs", "lasso", k)] = lasso_topk(X, y, k)
        subsets[("classical_fs", "rfe_rf", k)] = rfe_topk(X, y, k)
        subsets[("classical_fs", "rf_importance", k)] = rf_importance_topk(X, y, k)

    # de-duplicate identical subsets to save compute, but keep the mapping
    cache, long_rows = {}, []
    for (method, variant, k), fs in subsets.items():
        key = tuple(sorted(fs))
        print(f"{method:14s} {variant:18s} k={k}: {fs}", flush=True)
        if key not in cache:
            cache[key] = evaluate_subset(X, y, list(key))
        df = cache[key].copy()
        df["method"], df["variant"] = method, variant
        long_rows.append(df)

    long = pd.concat(long_rows, ignore_index=True)
    long.to_csv(RES / "downstream.csv", index=False)

summ = summarise(long.assign(features=long.method + "|" + long.variant + "|" + long.features))
summ.to_csv(RES / "downstream_summary.csv", index=False)
print(summ[summ.model == "random_forest"][["features", "k", "MCC", "MCC_ci95", "MCC_ci95_corrected", "F1",
                                           "ROC_AUC", "PR_AUC"]].to_string(index=False))
