"""Step 2 of the challenge: reproduce the classical baseline of Chicco & Jurman (2020).

Outputs results/baseline_all_features.csv and results/baseline_two_features.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load
from src.evaluate import evaluate_subset, summarise

RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)

X, y, feats = load()
print(f"Dataset: {X.shape[0]} patients, {len(feats)} clinical features, deaths={y.sum()}")

print("Baseline A: all 11 clinical features (cf. Chicco & Jurman Table 4)")
long_all = evaluate_subset(X, y, feats)
summ_all = summarise(long_all)
summ_all.to_csv(RES / "baseline_all_features.csv", index=False)
print(summ_all[["model", "MCC", "MCC_ci95", "F1", "accuracy", "TPR", "TNR", "PR_AUC", "ROC_AUC"]].to_string(index=False))

print("\nBaseline B: serum_creatinine + ejection_fraction (cf. Chicco & Jurman Table 9)")
long_two = evaluate_subset(X, y, ["serum_creatinine", "ejection_fraction"])
summ_two = summarise(long_two)
summ_two.to_csv(RES / "baseline_two_features.csv", index=False)
print(summ_two[["model", "MCC", "MCC_ci95", "F1", "accuracy", "TPR", "TNR", "PR_AUC", "ROC_AUC"]].to_string(index=False))
