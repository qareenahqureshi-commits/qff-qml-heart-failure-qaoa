"""Out-of-fold risk for every patient, for the HeartPanel worklist (docs/).

Each patient's risk is the mean random forest probability over the repeats in which that patient was in the
held-out fold, so no score comes from a model that saw the patient. Panels: the QUBO optimum with an echo, the
cost-aware optimum without one, and all 11 features for reference.

Writes results/patient_risk.csv, results/patient_risk_summary.csv and docs/data/patients.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.costs import panel_cost  # noqa: E402

PANELS = {
    "echo": ["age", "ejection_fraction", "serum_creatinine"],
    "no_echo": ["age", "serum_creatinine", "serum_sodium"],
    "all11": ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
              "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking"],
}
BANDS = (0.25, 0.5)  # low < 0.25 <= medium < 0.5 <= high, the same cut-offs as the digital twin
N_SPLITS, N_REPEATS = 5, 10


def oof_risk(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    total, count = np.zeros(len(y)), np.zeros(len(y))
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=0)
    for i, (tr, te) in enumerate(cv.split(X, y)):
        rf = RandomForestClassifier(n_estimators=500, random_state=i, n_jobs=-1).fit(X.iloc[tr], y[tr])
        total[te] += rf.predict_proba(X.iloc[te])[:, 1]
        count[te] += 1
    return total / count


def band(r: float) -> str:
    return "high" if r >= BANDS[1] else "medium" if r >= BANDS[0] else "low"


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "heart_failure_clinical_records_dataset.csv")
    y = df["DEATH_EVENT"].to_numpy()
    out = pd.DataFrame({"patient": np.arange(1, len(df) + 1)})
    rows = []
    for name, feats in PANELS.items():
        r = oof_risk(df[feats], y)
        out[f"risk_{name}"] = r.round(4)
        flagged = r >= BANDS[0]
        rows.append({
            "panel": name, "features": "+".join(feats), "cost_usd": panel_cost(feats),
            "roc_auc": roc_auc_score(y, r), "mcc_at_0.5": matthews_corrcoef(y, r >= 0.5),
            "flagged_medium_or_high": int(flagged.sum()), "deaths": int(y.sum()),
            "deaths_flagged": int((flagged & (y == 1)).sum()),
            "survivors_flagged": int((flagged & (y == 0)).sum()),
            "high": int((r >= BANDS[1]).sum()),
        })
    out["died"] = y
    out = pd.concat([df.drop(columns=["DEATH_EVENT"]), out], axis=1)
    out.to_csv(ROOT / "results" / "patient_risk.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results" / "patient_risk_summary.csv", index=False)
    print(summary.to_string(index=False))

    keep = ["patient", "age", "sex", "ejection_fraction", "serum_creatinine", "serum_sodium", "anaemia",
            "diabetes", "high_blood_pressure", "smoking", "time", "died", "risk_echo", "risk_no_echo", "risk_all11"]
    payload = {"bands": BANDS, "summary": rows, "patients": out[keep].to_dict(orient="records")}
    (ROOT / "docs" / "data").mkdir(exist_ok=True)
    (ROOT / "docs" / "data" / "patients.json").write_text(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
