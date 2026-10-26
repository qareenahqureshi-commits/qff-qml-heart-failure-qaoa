"""Chicco & Jurman (2020)-style repeated evaluation.

Protocol: 100 random stratified 80/20 train/test splits; every metric averaged over the
100 runs (mean ± std). MCC is the primary metric because the classes are imbalanced
(96 deaths / 203 survivors). Hyper-parameters are fixed a priori (no tuning on test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_RUNS = 100
TEST_SIZE = 0.2


def classifiers(seed: int):
    """The classifier zoo of Chicco & Jurman, with scikit-learn defaults made explicit."""
    return {
        "random_forest": RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=seed)),
    }


def _metrics(y_true, y_pred, y_score):
    return {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "accuracy": float(np.mean(y_true == y_pred)),
        "TPR": recall_score(y_true, y_pred),
        "TNR": recall_score(y_true, y_pred, pos_label=0),
        "PR_AUC": average_precision_score(y_true, y_score),
        "ROC_AUC": roc_auc_score(y_true, y_score),
    }


def evaluate_subset(X: pd.DataFrame, y: np.ndarray, features: list[str], n_runs: int = N_RUNS,
                    models: list[str] | None = None, base_seed: int = 0) -> pd.DataFrame:
    """Evaluate every classifier on `features` over n_runs stratified splits.

    Returns a long DataFrame: one row per (run, model) with all metrics.
    """
    Xs = X[features].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=TEST_SIZE, random_state=base_seed)
    rows = []
    for run, (tr, te) in enumerate(sss.split(Xs, y)):
        for name, clf in classifiers(base_seed + run).items():
            if models and name not in models:
                continue
            clf.fit(Xs[tr], y[tr])
            pred = clf.predict(Xs[te])
            score = clf.predict_proba(Xs[te])[:, 1]
            m = _metrics(y[te], pred, score)
            m.update(run=run, model=name, k=len(features), features="+".join(features))
            rows.append(m)
    return pd.DataFrame(rows)


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per model, plus a 95% CI half-width for MCC."""
    metrics = ["MCC", "F1", "accuracy", "TPR", "TNR", "PR_AUC", "ROC_AUC"]
    g = long.groupby(["features", "k", "model"])
    out = g[metrics].mean().round(3)
    out["MCC_std"] = g["MCC"].std().round(3)
    out["MCC_ci95"] = (1.96 * g["MCC"].std() / np.sqrt(g["MCC"].count())).round(3)
    return out.reset_index()
