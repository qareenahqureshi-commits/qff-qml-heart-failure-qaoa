"""Paired comparisons between panels, with the corrected resampled t test (src/stats.py).

Reads results/downstream.csv (per split, per classifier scores of every panel on the same 100 splits) and
results/select_k.csv (the leakage free nested pipeline on the same splits), and writes
results/paired_tests.csv with, for each comparison and classifier:
  delta MCC, naive paired t test p value (what the original README reported), corrected p value and the
  corrected 95% CI of the difference.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import N_PATIENTS, TEST_SIZE
from src.stats import paired_test, split_sizes

RES = Path(__file__).resolve().parents[1] / "results"
n_train, n_test = split_sizes(N_PATIENTS, TEST_SIZE)

K3 = "age+ejection_fraction+serum_creatinine"
CJ = "ejection_fraction+serum_creatinine"
ALL = ("age+anaemia+creatinine_phosphokinase+diabetes+ejection_fraction+high_blood_pressure+platelets+"
       "serum_creatinine+serum_sodium+sex+smoking")
MRMR3 = "anaemia+high_blood_pressure+serum_creatinine"
RFIMP4 = "age+creatinine_phosphokinase+ejection_fraction+serum_creatinine"

COMPARISONS = [
    ("k=3 QUBO panel vs Chicco & Jurman 2 feature panel", K3, CJ),
    ("k=3 QUBO panel vs all 11 features", K3, ALL),
    ("k=3 QUBO panel vs greedy mRMR k=3", K3, MRMR3),
    ("k=3 QUBO panel vs RF importance k=4", K3, RFIMP4),
    ("Chicco & Jurman 2 feature panel vs all 11 features", CJ, ALL),
]

long = pd.read_csv(RES / "downstream.csv")
per_panel = long.drop_duplicates(["features", "model", "run"])
rows = []
for model, g in per_panel.groupby("model"):
    by = {f: s.sort_values("run").MCC.to_numpy() for f, s in g.groupby("features")}
    for name, a, b in COMPARISONS:
        if a in by and b in by:
            rows.append(dict(model=model, comparison=name, **paired_test(by[a], by[b], n_train, n_test)))

# leakage free nested pipeline vs fixed reference panels, random forest, same splits (run index = split)
sk_path = RES / "select_k.csv"
if sk_path.exists():
    sk = pd.read_csv(sk_path).sort_values("run")
    rf = {f: s.sort_values("run").MCC.to_numpy() for f, s in per_panel[per_panel.model == "random_forest"].groupby("features")}
    for col, label in [("MCC", "exhaustive"), ("MCC_qaoa", "QAOA")]:
        if col not in sk:
            continue
        for ref_name, ref in [("all 11 features", ALL), ("Chicco & Jurman 2 feature panel", CJ)]:
            rows.append(dict(model="random_forest", comparison=f"nested pipeline ({label}) vs {ref_name}",
                             **paired_test(sk[col].to_numpy(), rf[ref], n_train, n_test)))

out = pd.DataFrame(rows)
out.to_csv(RES / "paired_tests.csv", index=False)
pd.set_option("display.width", 200)
print(f"n_train={n_train}, n_test={n_test}, variance factor corrected/naive = "
      f"{(1/100 + n_test/n_train) / (1/100):.1f}x")
print(out[out.model == "random_forest"][["comparison", "delta", "p_naive", "p_corrected",
                                          "delta_ci95_low", "delta_ci95_high"]].round(4).to_string(index=False))
