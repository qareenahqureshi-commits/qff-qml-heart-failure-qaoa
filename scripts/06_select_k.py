"""Nested selection of the panel size k, so that k is chosen WITHOUT looking at test data.

For each of the 100 outer stratified 80/20 splits:
  * on the training 80 % only: build the Fisher QUBO, take the exhaustive optimum for k in {2,3,4}
    (identical to what QAOA finds as its best sample at p>=2 in 10/10 seeds, see results/optimisers_summary.csv),
    and score each panel by inner 5-fold CV MCC of a random forest;
  * pick the k with the best inner MCC; refit on the full training split; evaluate once on the test 20 %.
This answers "would a practitioner using this pipeline end up with the k=3 panel?" without test leakage.
Outputs results/select_k.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.evaluate import N_RUNS, TEST_SIZE, _metrics
from src.qubo import make_qubo

RES = Path(__file__).resolve().parents[1] / "results"
X, y, feats = load()
sss = StratifiedShuffleSplit(n_splits=N_RUNS, test_size=TEST_SIZE, random_state=0)  # same splits as evaluate.py
rows = []
for run, (tr, te) in enumerate(sss.split(X, y)):
    Xtr, ytr = X.iloc[tr], y[tr]
    inner = StratifiedKFold(5, shuffle=True, random_state=run)
    best = None
    panels = {}
    for k in [2, 3, 4]:
        q = make_qubo(Xtr, ytr, k, "fisher")
        x, _, _ = q.exhaustive()
        panel = [f for f, b in zip(feats, x) if b]
        panels[k] = panel
        rf = RandomForestClassifier(n_estimators=200, random_state=run, n_jobs=-1)
        pred = cross_val_predict(rf, Xtr[panel], ytr, cv=inner)
        mcc = matthews_corrcoef(ytr, pred)
        if best is None or mcc > best[0]:
            best = (mcc, k, panel)
    inner_mcc, k_sel, panel = best
    rf = RandomForestClassifier(n_estimators=500, random_state=run, n_jobs=-1).fit(Xtr[panel], ytr)
    m = _metrics(y[te], rf.predict(X.iloc[te][panel]), rf.predict_proba(X.iloc[te][panel])[:, 1])
    rows.append(dict(run=run, k_selected=k_sel, panel="+".join(panel), inner_MCC=inner_mcc,
                     panel_k2="+".join(panels[2]), panel_k3="+".join(panels[3]), panel_k4="+".join(panels[4]), **m))
    if run % 20 == 0:
        print(f"run {run}: k={k_sel} {panel} test MCC {m['MCC']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "select_k.csv", index=False)
print("\nk selected (counts):", df.k_selected.value_counts().to_dict())
print("panel selected (counts):", df.panel.value_counts().to_dict())
print("k=3 panel on training data == age+EF+SC in", (df.panel_k3 == "age+ejection_fraction+serum_creatinine").mean(), "of runs")
print("nested-CV test performance (mean of 100):")
print(df[["MCC", "F1", "accuracy", "TPR", "TNR", "PR_AUC", "ROC_AUC"]].mean().round(3).to_dict(),
      "MCC 95% CI ±", round(1.96 * df.MCC.std() / 10, 3))
