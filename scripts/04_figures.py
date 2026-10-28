"""Figures for the README: approximation ratio by solver, P(optimum) vs QAOA depth, downstream MCC by subset
(with CIs corrected for overlapping splits), noise study, nested k selection, QAOA variants and cost aware panels.
Every figure is skipped quietly if its input CSV does not exist yet."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

# validated categorical palette, fixed order (blue, orange, aqua, yellow); text in text tokens, recessive grid
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9a9993"
plt.rcParams.update({
    "axes.prop_cycle": cycler(color=SERIES), "lines.linewidth": 2, "lines.markersize": 6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

opt = pd.read_csv(RES / "optimisers.csv")
opt["label"] = np.where(opt.solver == "qaoa", "QAOA p=" + opt.p.astype(str), opt.solver)
order = ["random", "random_feasible", "simulated_annealing", "tabu", "QAOA p=1", "QAOA p=2", "QAOA p=3"]
order = [o for o in order if o in set(opt.label)]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, variant in zip(axes, ["mi", "fisher"]):
    d = opt[opt.variant == variant]
    for k, mk in zip([2, 3, 4], ["o", "s", "^"]):
        g = d[d.k == k].groupby("label").approx_ratio_mean.agg(["mean", "std"]).reindex(order)
        ax.errorbar(range(len(order)), g["mean"], yerr=g["std"], marker=mk, capsize=3, label=f"k={k}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Mean approximation ratio, {variant.upper()} QUBO")
    ax.set_ylabel("approximation ratio (1 = optimum)")
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "approx_ratio.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4))
q = opt[opt.solver == "qaoa"].groupby(["variant", "k", "p"]).p_optimum.mean().reset_index()
for (v, k), g in q.groupby(["variant", "k"]):
    ax.plot(g.p, g.p_optimum, marker="o" if v == "fisher" else "s", ls="-" if v == "fisher" else ":",
            color=SERIES[[2, 3, 4].index(k)], label=f"{v} k={k}")
ax.set_xlabel("QAOA depth p")
ax.set_ylabel("P(sample = exhaustive optimum)")
ax.set_xticks([1, 2, 3])
ax.legend(fontsize=8)
ax.set_title("Probability of sampling the optimum")
plt.tight_layout()
plt.savefig(FIG / "p_optimum_vs_depth.png", dpi=150)
plt.close()

ds = pd.read_csv(RES / "downstream_summary.csv")
rf = ds[ds.model == "random_forest"].copy()
rf[["method", "variant", "feats"]] = rf.features.str.split("|", expand=True)
def _who(g):
    ms = sorted(set(g.method.str.replace("qaoa_p[123]", "qaoa", regex=True)))
    return ", ".join(ms)
ci_col = "MCC_ci95_corrected" if "MCC_ci95_corrected" in rf else "MCC_ci95"
agg = rf.groupby(["feats", "k"]).agg(MCC=("MCC", "first"), CI=(ci_col, "first"),
                                     found_by=("method", lambda s: _who(rf.loc[s.index]))).reset_index()
agg["label"] = np.where(agg.k == 11, "all 11 clinical features",
                        agg.feats.str.replace("_", " ").str.replace("+", " + ")) + "  [" + agg.found_by + "]"
agg["quantum"] = agg.found_by.str.contains("qaoa")
rf = agg.sort_values("MCC")
fig, ax = plt.subplots(figsize=(12, 0.45 * len(rf) + 1.2))
ax.barh(rf.label, rf.MCC, height=0.55, xerr=rf.CI, color=np.where(rf.quantum, SERIES[0], MUTED),
        error_kw=dict(ecolor=INK2, lw=1, capsize=3))
ax.tick_params(axis="y", labelsize=8)
ax.set_xlim(0, 0.6)
ax.grid(axis="y", visible=False)
ax.set_xlabel("MCC (random forest, mean of 100 splits); bars: 95% CI corrected for overlapping splits")
ax.set_title("Downstream survival prediction by feature subset (blue = also selected by QAOA)")
plt.tight_layout()
plt.savefig(FIG / "downstream_mcc.png", dpi=150)
plt.close()

# --- noise study: ideal vs fake_torino, Fisher QUBO -----------------------------------------
nz = RES / "noise_summary.csv"
if nz.exists():
    ns = pd.read_csv(nz)
    tr = ns[ns.protocol == "transfer"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (k, mk) in enumerate(zip([2, 3, 4], ["o", "s", "^"])):
        g = tr[tr.k == k].sort_values("p")
        axes[0].plot(g.p, g.ideal_AR_mean, marker=mk, ls="--", color=MUTED, label="ideal" if k == 2 else None)
        axes[0].errorbar(g.p, g.AR_mean, yerr=g.AR_std, marker=mk, capsize=3, color=SERIES[i], label=f"fake_torino k={k}")
        axes[1].plot(g.p, g.ideal_P_opt, marker=mk, ls="--", color=MUTED)
        axes[1].plot(g.p, g.P_opt, marker=mk, color=SERIES[i], label=f"fake_torino k={k}")
    axes[0].set_ylim(0.6, 1.0); axes[0].set_xticks([1, 2, 3]); axes[0].set_xlabel("QAOA depth p")
    axes[0].set_ylabel("mean approximation ratio"); axes[0].set_title("Noise: ideal (dashed) vs Heron class device model"); axes[0].legend(fontsize=8)
    axes[1].set_xticks([1, 2, 3]); axes[1].set_xlabel("QAOA depth p"); axes[1].set_ylabel("P(sample = optimum)")
    axes[1].set_title("Per shot optimum probability (dashed = ideal)"); axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "noise_fake_torino.png", dpi=150); plt.close()

# --- nested k selection ------------------------------------------------------------------
sk = RES / "select_k.csv"
if sk.exists():
    d = pd.read_csv(sk)
    cols = [("panel", "exhaustive optimum")] + ([("panel_qaoa", "QAOA best sample")] if "panel_qaoa" in d else [])
    panels = d.panel.value_counts().index.tolist()
    for c, _ in cols[1:]:
        panels += [p for p in d[c].value_counts().index if p not in panels]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(panels) + 1.4))
    h = 0.38 if len(cols) == 2 else 0.6
    ypos = np.arange(len(panels))
    for i, (c, lab) in enumerate(cols):
        vc = d[c].value_counts().reindex(panels).fillna(0)
        ax.barh(ypos + (i - (len(cols) - 1) / 2) * (h + 0.04), vc.values, height=h, color=SERIES[i], label=lab)
    ax.set_yticks(ypos, [p.replace("_", " ").replace("+", " + ") for p in panels])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("times selected by inner 5-fold CV (of 100 outer splits)"); ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Panel chosen without test data (nested CV)")
    if len(cols) > 1:
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / "nested_k_selection.png", dpi=150); plt.close()

# --- QAOA variants: P(optimum) per shot vs depth -------------------------------------------
vs = RES / "qaoa_variants_summary.csv"
if vs.exists():
    v = pd.read_csv(vs)
    show = [("baseline", "baseline (original)", "o"), ("cvar", "CVaR objective", "s"),
            ("interp", "INTERP angles", "^"), ("xy_interp", "XY mixer + INTERP", "D")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, k in zip(axes, [2, 3, 4]):
        d = v[v.k == k]
        ref = d[d.variant == "random_feasible"].P_opt.item()
        ax.axhline(ref, color=MUTED, lw=1)
        ax.text(3.05, ref, "random\nvalid panel", va="center", fontsize=7, color=INK2)
        for (name, lab, mk), col in zip(show, SERIES):
            g = d[d.variant == name].sort_values("p")
            se = g.P_opt_std / np.sqrt(10)
            ax.plot(g.p, g.P_opt, marker=mk, color=col, label=lab)
            ax.fill_between(g.p, g.P_opt - se, g.P_opt + se, color=col, alpha=0.1, lw=0)
        ax.set_xticks([1, 2, 3]); ax.set_xlim(0.8, 3.5); ax.set_xlabel("QAOA depth p"); ax.set_title(f"k = {k}")
    axes[0].set_ylabel("P(shot = optimal panel), mean of 10 seeds")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("QAOA variants on the Fisher QUBO (band = ± 1 standard error)", color=INK)
    plt.tight_layout(); plt.savefig(FIG / "qaoa_variants.png", dpi=150); plt.close()

# --- cost aware panels ----------------------------------------------------------------------
cp = RES / "cost_panels.csv"
if cp.exists():
    c = pd.read_csv(cp).sort_values("MCC")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(c) + 1.4))
    cols = np.where(c.needs_echo, SERIES[0], SERIES[1])
    ax.barh(range(len(c)), c.MCC, height=0.55, xerr=c.MCC_ci95_corrected, color=cols,
            error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    ax.set_yticks(range(len(c)), [f"{p.replace('_', ' ').replace('+', ' + ')}   (${cost:,.2f})"
                                  for p, cost in zip(c.panel, c.cost_usd)], fontsize=8)
    ax.grid(axis="y", visible=False); ax.set_xlim(0, 0.6)
    ax.set_xlabel("MCC (random forest, 100 splits); bars: corrected 95% CI")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="needs an echocardiogram"),
                       Patch(color=SERIES[1], label="blood tests and history only")], fontsize=8, loc="lower right")
    fig.suptitle("Cost aware panels: accuracy with and without the echocardiogram", color=INK)
    plt.tight_layout(); plt.savefig(FIG / "cost_tradeoff.png", dpi=150); plt.close()

print("figures written to", FIG)
