"""Figures for the README: approximation ratio by solver, P(optimum) vs QAOA depth,
downstream MCC by subset, and the alpha-vs-k curve."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

opt = pd.read_csv(RES / "optimisers.csv")
opt["label"] = np.where(opt.solver == "qaoa", "QAOA p=" + opt.p.astype(str), opt.solver)
order = ["random", "simulated_annealing", "tabu", "QAOA p=1", "QAOA p=2", "QAOA p=3"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, variant in zip(axes, ["mi", "fisher"]):
    d = opt[opt.variant == variant]
    for k, mk in zip([2, 3, 4], ["o", "s", "^"]):
        g = d[d.k == k].groupby("label").approx_ratio_mean.agg(["mean", "std"]).reindex(order)
        ax.errorbar(range(len(order)), g["mean"], yerr=g["std"], marker=mk, capsize=3, label=f"k={k}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Mean approximation ratio — {variant.upper()} QUBO")
    ax.set_ylabel("approximation ratio (1 = optimum)")
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "approx_ratio.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 4))
q = opt[opt.solver == "qaoa"].groupby(["variant", "k", "p"]).p_optimum.mean().reset_index()
for (v, k), g in q.groupby(["variant", "k"]):
    ax.plot(g.p, g.p_optimum, marker="o", label=f"{v} k={k}")
ax.set_xlabel("QAOA depth p")
ax.set_ylabel("P(sample = exhaustive optimum)")
ax.set_xticks([1, 2, 3])
ax.legend(fontsize=8)
ax.set_title("Probability of sampling the optimum")
plt.tight_layout()
plt.savefig(FIG / "p_optimum_vs_depth.png", dpi=150)

ds = pd.read_csv(RES / "downstream_summary.csv")
rf = ds[ds.model == "random_forest"].copy()
rf[["method", "variant", "feats"]] = rf.features.str.split("|", expand=True)
def _who(g):
    ms = sorted(set(g.method.str.replace("qaoa_p[123]", "qaoa", regex=True)))
    return ", ".join(ms)
agg = rf.groupby(["feats", "k"]).agg(MCC=("MCC", "first"), MCC_ci95=("MCC_ci95", "first"),
                                     found_by=("method", lambda s: _who(rf.loc[s.index]))).reset_index()
agg["label"] = np.where(agg.k == 11, "all 11 clinical features",
                        agg.feats.str.replace("_", " ").str.replace("+", " + ")) + "  [" + agg.found_by + "]"
agg["quantum"] = agg.found_by.str.contains("qaoa")
rf = agg.sort_values("MCC")
fig, ax = plt.subplots(figsize=(12, 0.45 * len(rf) + 1))
ax.barh(rf.label, rf.MCC, xerr=rf.MCC_ci95, color=np.where(rf.quantum, "#6a5acd", "#999"))
ax.tick_params(axis="y", labelsize=8)
ax.set_xlim(0, 0.5)
ax.set_xlabel("MCC (random forest, mean of 100 splits, 95% CI)")
ax.set_title("Downstream survival prediction by feature subset")
plt.tight_layout()
plt.savefig(FIG / "downstream_mcc.png", dpi=150)
print("figures written to", FIG)

# --- noise study: ideal vs fake_torino, Fisher QUBO -----------------------------------------
nz = RES / "noise_summary.csv"
if nz.exists():
    ns = pd.read_csv(nz)
    tr = ns[ns.protocol == "transfer"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, mk in zip([2, 3, 4], ["o", "s", "^"]):
        g = tr[tr.k == k].sort_values("p")
        axes[0].plot(g.p, g.ideal_AR_mean, marker=mk, ls="--", color="gray", label=f"ideal k={k}" if k == 2 else None)
        axes[0].errorbar(g.p, g.AR_mean, yerr=g.AR_std, marker=mk, capsize=3, label=f"fake_torino k={k}")
        axes[1].plot(g.p, g.ideal_P_opt, marker=mk, ls="--", color="gray")
        axes[1].plot(g.p, g.P_opt, marker=mk, label=f"fake_torino k={k}")
    axes[0].set_ylim(0.6, 1.0); axes[0].set_xticks([1, 2, 3]); axes[0].set_xlabel("QAOA depth p")
    axes[0].set_ylabel("mean approximation ratio"); axes[0].set_title("Noise: ideal (dashed) vs Heron-class device model"); axes[0].legend(fontsize=8)
    axes[1].set_xticks([1, 2, 3]); axes[1].set_xlabel("QAOA depth p"); axes[1].set_ylabel("P(sample = optimum)")
    axes[1].set_title("Per-shot optimum probability (dashed = ideal)"); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "noise_fake_torino.png", dpi=150)

# --- nested k selection ------------------------------------------------------------------
sk = RES / "select_k.csv"
if sk.exists():
    d = pd.read_csv(sk)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    vc = d.panel.value_counts()
    ax.barh([p.replace("_", " ").replace("+", " + ") for p in vc.index], vc.values, color="#6a5acd")
    ax.set_xlabel("times selected by inner 5-fold CV (of 100 outer splits)"); ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Panel chosen without test data (nested CV)")
    plt.tight_layout(); plt.savefig(FIG / "nested_k_selection.png", dpi=150)
