"""Dataset loading for the UCI Heart Failure Clinical Records dataset (Chicco & Jurman, 2020).

299 patients, 12 clinical features, binary target DEATH_EVENT (96 deaths / 203 survivors).
Source: https://archive.ics.uci.edu/dataset/519/heart-failure-clinical-records (CC BY 4.0)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart_failure_clinical_records_dataset.csv"
TARGET = "DEATH_EVENT"

# Chicco & Jurman exclude `time` (follow-up period) from the clinical-feature analysis
# in their main tables because it is not a measurement available at admission; we follow
# that convention so the baseline is comparable with their Table 4 / Table 9.
CLINICAL_FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking",
]
ALL_FEATURES = CLINICAL_FEATURES + ["time"]


def load(include_time: bool = False):
    """Return (X: DataFrame, y: ndarray, feature_names: list)."""
    df = pd.read_csv(DATA_PATH)
    feats = ALL_FEATURES if include_time else CLINICAL_FEATURES
    X = df[feats].astype(float)
    y = df[TARGET].to_numpy().astype(int)
    return X, y, list(feats)


if __name__ == "__main__":
    X, y, f = load()
    print(X.shape, np.bincount(y), f)
