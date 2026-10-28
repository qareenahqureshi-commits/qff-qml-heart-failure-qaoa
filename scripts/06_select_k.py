"""Nested selection of the panel size k, so that k is chosen WITHOUT looking at test data, with the panel
found by QAOA itself (not assumed).

For each of the 100 outer stratified 80/20 splits:
  * on the training 80 % only: build the Fisher QUBO for k in {2,3,4};
  * find each panel two ways: the exhaustive optimum, and QAOA's best sample (X mixer, p = 2, 2000 shots,
    COBYLA <= 150 evaluations, random initial angles, seed = split index);
  * score each panel by inner 5-fold CV MCC of a random forest and pick the k with the best inner MCC;
  * refit on the full training split; evaluate once on the test 20 %.
The exhaustive columns reproduce the original results exactly; the *_qaoa columns are the pipeline a user
without brute force would actually run. Confidence intervals use the corrected resampled variance
(src/stats.py), because the 100 splits overlap.

Backend for QAOA: QFF_BACKEND=aer (default) or statevector (exact numpy simulation, no Qiskit needed).
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
from src.evaluate import N_PATIENTS, N_RUNS, TEST_SIZE, _metrics
from src.qaoa import QAOASolver
from src.qubo import make_qubo
from src.stats import ci_halfwidth, split_sizes

RES = Path(__file__).resolve().parents[1] / "results"
QAOA_P, SHOTS, MAXITER = 2, 2000, 150
METRICS = ["MCC", "F1", "accuracy", "TPR", "TNR", "PR_AUC", "ROC_AUC"]
X, y, feats = load()
sss = StratifiedShuffleSplit(n_splits=N_RUNS, test_size=TEST_SIZE, random_state=0)  # same splits as evaluate.py
rows = []
for run, (tr, te) in enumerate(sss.split(X, y)):
    Xtr, ytr = X.iloc[tr], y[tr]
    inner = StratifiedKFold(5, shuffle=True, random_state=run)
    inner_cache, test_cache = {}, {}

    def inner_mcc(panel):
        key = tuple(panel)
        if key not in inner_cache:
            rf = RandomForestClassifier(n_estimators=200, random_state=run, n_jobs=-1)
            inner_cache[key] = matthews_corrcoef(ytr, cross_val_predict(rf, Xtr[panel], ytr, cv=inner))
        return inner_cache[key]

    def test_metrics(panel):
        key = tuple(panel)
        if key not in test_cache:
            rf = RandomForestClassifier(n_estimators=500, random_state=run, n_jobs=-1).fit(Xtr[panel], ytr)
            test_cache[key] = _metrics(y[te], rf.predict(X.iloc[te][panel]), rf.predict_proba(X.iloc[te][panel])[:, 1])
        return test_cache[key]

    panels, panels_q, found, p_opt = {}, {}, {}, {}
    for k in [2, 3, 4]:
        q = make_qubo(Xtr, ytr, k, "fisher")
        x, _, _ = q.exhaustive()
        panels[k] = [f for f, b in zip(feats, x) if b]
        out = QAOASolver(q, reps=QAOA_P, shots=SHOTS, maxiter=MAXITER, seed=run).run(np.random.default_rng(run))
        best = out["samples"][int(np.argmin(out["energies"]))]
        panels_q[k] = [f for f, b in zip(feats, best) if b]
        found[k] = bool(np.array_equal(best, x))
        p_opt[k] = float(np.all(out["samples"] == x, axis=1).mean())

    def choose(pan):
        best = None
        for k in [2, 3, 4]:
            mcc = inner_mcc(pan[k])
            if best is None or mcc > best[0]:
                best = (mcc, k, pan[k])
        return best

    inner_e, k_e, panel_e = choose(panels)
    inner_q, k_q, panel_q = choose(panels_q)
    m_e, m_q = test_metrics(panel_e), test_metrics(panel_q)
    row = dict(run=run, k_selected=k_e, panel="+".join(panel_e), inner_MCC=inner_e,
               panel_k2="+".join(panels[2]), panel_k3="+".join(panels[3]), panel_k4="+".join(panels[4]), **m_e,
               k_selected_qaoa=k_q, panel_qaoa="+".join(panel_q), inner_MCC_qaoa=inner_q)
    for k in [2, 3, 4]:
        row[f"qaoa_panel_k{k}"] = "+".join(panels_q[k])
        row[f"qaoa_found_k{k}"] = found[k]
        row[f"qaoa_p_opt_k{k}"] = p_opt[k]
    row.update({f"{m}_qaoa": v for m, v in m_q.items()})
    rows.append(row)
    if run % 20 == 0:
        print(f"run {run}: exhaustive k={k_e} {panel_e} MCC {m_e['MCC']:.3f} | QAOA k={k_q} {panel_q} "
              f"MCC {m_q['MCC']:.3f} | QAOA found optimum {found}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RES / "select_k.csv", index=False)
n_train, n_test = split_sizes(N_PATIENTS, TEST_SIZE)
for label, suffix in [("exhaustive optimum", ""), ("QAOA best sample", "_qaoa")]:
    kcol = "k_selected" + suffix
    pcol = "panel" + suffix
    print(f"\n== nested pipeline, panels from the {label} ==")
    print("k selected (counts):", df[kcol].value_counts().to_dict())
    print("panel selected (counts):", df[pcol].value_counts().to_dict())
    print("test performance (mean of 100):", {m: round(df[m + suffix].mean(), 3) for m in METRICS})
    s = df["MCC" + suffix]
    print(f"MCC 95% CI ± {ci_halfwidth(s, n_train, n_test, corrected=False):.3f} (naive), "
          f"± {ci_halfwidth(s, n_train, n_test, corrected=True):.3f} (corrected)")
print("\nk=3 panel on training data == age+EF+SC in", (df.panel_k3 == "age+ejection_fraction+serum_creatinine").mean(), "of runs")
print("QAOA best sample == exhaustive optimum:", {k: int(df[f'qaoa_found_k{k}'].sum()) for k in [2, 3, 4]}, "of 100 splits")
print("QAOA P(optimum) per shot, mean:", {k: round(df[f'qaoa_p_opt_k{k}'].mean(), 3) for k in [2, 3, 4]})
print("QAOA pipeline chose the same panel as the exhaustive pipeline in", int((df.panel == df.panel_qaoa).sum()), "of 100 splits")
