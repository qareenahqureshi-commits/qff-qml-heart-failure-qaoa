"""Measurement costs for cost aware panel selection.

Each clinical feature comes from a test. Features from the same test cost that test once: serum creatinine and
serum sodium both come from one basic metabolic panel; anaemia and platelets both come from one blood count.
Age, sex, smoking, diabetes and high blood pressure come from the history and exam and are treated as free.

Illustrative 2026 US prices (what matters is the ratio: the echocardiogram costs 43 to 56 times any lab test):
  echocardiogram, CPT 93306 (transthoracic, complete):  $197.85  2026 Medicare physician fee schedule, global
                                                                 (American Society of Echocardiography summary)
  basic metabolic panel, CPT 80048:                      $4.57   Florida Medicaid 2026 independent lab fees
  complete blood count with differential, CPT 85025:     $4.19   same schedule
  creatine kinase (CPK), CPT 82550:                      $3.52   same schedule
Lab and imaging prices come from different payer schedules and vary by payer and region; swap in local prices
by editing TESTS.

Because every paid test covers at most two features, the panel cost is exactly quadratic in the selection bits:
  cost(x) = sum_t c_t * OR(x_i : i in t),  OR(a) = a,  OR(a, b) = a + b - a b,
so it can be added to the QUBO without approximation.
"""
from __future__ import annotations

import numpy as np

TESTS = {
    "history_and_exam": {"cost": 0.0, "features": ["age", "diabetes", "high_blood_pressure", "sex", "smoking"]},
    "echocardiogram_93306": {"cost": 197.85, "features": ["ejection_fraction"]},
    "basic_metabolic_panel_80048": {"cost": 4.57, "features": ["serum_creatinine", "serum_sodium"]},
    "complete_blood_count_85025": {"cost": 4.19, "features": ["anaemia", "platelets"]},
    "creatine_kinase_82550": {"cost": 3.52, "features": ["creatinine_phosphokinase"]},
}


def panel_cost(panel: list[str], tests: dict = TESTS) -> float:
    """Dollar cost of the distinct tests needed for a panel."""
    s = set(panel)
    return float(sum(t["cost"] for t in tests.values() if s & set(t["features"])))


def tests_for(panel: list[str], tests: dict = TESTS) -> list[str]:
    s = set(panel)
    return [name for name, t in tests.items() if t["cost"] > 0 and s & set(t["features"])]


def cost_matrix(features: list[str], tests: dict = TESTS) -> np.ndarray:
    """Symmetric C with x^T C x = cost(x) for every x in {0,1}^n (exact; see module docstring)."""
    idx = {f: i for i, f in enumerate(features)}
    C = np.zeros((len(features), len(features)))
    for name, t in tests.items():
        members = [idx[f] for f in t["features"] if f in idx]
        if t["cost"] == 0 or not members:
            continue
        if len(members) > 2:
            raise ValueError(f"{name}: a paid test covering more than two features is not quadratic")
        for i in members:
            C[i, i] += t["cost"]
        if len(members) == 2:
            a, b = members
            C[a, b] -= t["cost"] / 2
            C[b, a] -= t["cost"] / 2
    return C


def total_cost(tests: dict = TESTS) -> float:
    return float(sum(t["cost"] for t in tests.values()))
