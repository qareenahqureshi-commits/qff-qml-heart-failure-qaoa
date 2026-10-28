"""Statistics for scores averaged over repeated random train/test splits.

The evaluation protocol (Chicco & Jurman 2020) uses 100 stratified 80/20 splits of the same 299 patients.
Any two splits share most of their training patients, so the 100 per-split scores are strongly positively
correlated. A plain paired t test, or a confidence interval of 1.96 * std / sqrt(100), treats them as 100
independent experiments and understates the uncertainty by a large factor.

Nadeau & Bengio (2003), "Inference for the generalization error", Machine Learning 52:239-281, correct this
by replacing the variance factor 1/J with (1/J + n_test/n_train) (the "corrected resampled t test"). With
J = 100, n_test = 60 and n_train = 239 that factor is 0.261 instead of 0.010, i.e. standard errors about
5.1x wider. Bouckaert & Frank (2004) recommend this test for exactly this kind of repeated hold out design.

The Wilcoxon signed rank test has the same independence assumption and no standard correction, so it is not
reported as evidence here.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def split_sizes(n_samples: int, test_size: float) -> tuple[int, int]:
    """(n_train, n_test) exactly as sklearn's StratifiedShuffleSplit computes them (test = ceil)."""
    n_test = int(np.ceil(test_size * n_samples))
    return n_samples - n_test, n_test


def variance_factor(J: int, n_train: int, n_test: int, corrected: bool = True) -> float:
    return 1.0 / J + (n_test / n_train if corrected else 0.0)


def ci_halfwidth(scores, n_train: int, n_test: int, level: float = 0.95, corrected: bool = True) -> float:
    """Half width of a CI for the mean score over J resampled splits.

    corrected=False reproduces the original README numbers (normal quantile, std/sqrt(J)).
    corrected=True uses the Nadeau and Bengio variance factor with a Student t quantile (J - 1 df).
    """
    s = np.asarray(scores, dtype=float)
    J = len(s)
    sd = s.std(ddof=1)
    if not corrected:
        return float(stats.norm.ppf(0.5 + level / 2) * sd / np.sqrt(J))
    q = stats.t.ppf(0.5 + level / 2, df=J - 1)
    return float(q * sd * np.sqrt(variance_factor(J, n_train, n_test)))


def paired_test(a, b, n_train: int, n_test: int) -> dict:
    """Compare two methods evaluated on the same J splits (a[j], b[j] from split j).

    Returns the mean difference a - b, the naive paired t test p value, the corrected resampled t test
    p value and the corrected 95% CI of the mean difference.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    J = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    t_naive = mean / (sd * np.sqrt(variance_factor(J, n_train, n_test, corrected=False)))
    t_corr = mean / (sd * np.sqrt(variance_factor(J, n_train, n_test, corrected=True)))
    p_naive = 2 * stats.t.sf(abs(t_naive), df=J - 1)
    p_corr = 2 * stats.t.sf(abs(t_corr), df=J - 1)
    half = ci_halfwidth(d, n_train, n_test, corrected=True)
    return {
        "delta": float(mean),
        "t_naive": float(t_naive), "p_naive": float(p_naive),
        "t_corrected": float(t_corr), "p_corrected": float(p_corr),
        "delta_ci95_low": float(mean - half), "delta_ci95_high": float(mean + half),
        "n_splits": J,
    }
