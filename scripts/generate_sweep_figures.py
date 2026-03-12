#!/usr/bin/env python3
"""
Generate publication figures from the strategy sweep experiment.

Reads:  paper_figures/sweep_summary.csv
Writes: paper_figures/fig_sweep_*.pdf/.png
        paper_figures/sweep_findings.tex
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

DATA_DIR = "paper_figures/"
OUT_DIR  = "paper_figures/"

STRATEGY_ORDER = [
    "Base", "OT", "DRO(inj)",
    "SH", "OT+SH", "DRO(inj)+SH",
    "OT+DRO(inj)+SH", "DRO(q*)+SH", "OT+DRO(q*)+SH",
]
STRATEGY_LABELS = {
    "Base":             "Base",
    "OT":               "OT",
    "DRO(inj)":         "DRO(inj)",
    "SH":               "SH",
    "OT+SH":            "OT+SH",
    "DRO(inj)+SH":      "DRO(inj)+SH",
    "OT+DRO(inj)+SH":   "OT+DRO(inj)+SH",
    "DRO(q*)+SH":       r"DRO($q^*$)+SH",
    "OT+DRO(q*)+SH":    r"OT+DRO($q^*$)+SH",
}
STRATEGY_COLORS = {
    "Base":             "#7f7f7f",
    "OT":               "#17becf",
    "DRO(inj)":         "#bcbd22",
    "SH":               "#1f77b4",
    "OT+SH":            "#2ca02c",
    "DRO(inj)+SH":      "#ff7f0e",
    "OT+DRO(inj)+SH":   "#9467bd",
    "DRO(q*)+SH":       "#d62728",
    "OT+DRO(q*)+SH":    "#8c564b",
}
STRATEGY_MARKERS = {
    "Base": "v", "OT": "<", "DRO(inj)": ">",
    "SH": "o", "OT+SH": "^",
    "DRO(inj)+SH": "s", "OT+DRO(inj)+SH": "P",
    "DRO(q*)+SH": "D", "OT+DRO(q*)+SH": "X",
}


def load():
    path = os.path.join(DATA_DIR, "sweep_summary.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found"); sys.exit(1)
    return pd.read_csv(path)


def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}")


# ============================================================================
# Fig A: Collision rate vs switch_prob (line + whiskers)
# ============================================================================

def fig_switch_sweep_collision(df):
    sub = df[(df["rare_prob"].between(0.04, 0.06)) & (df["obs_config"] == "1obs")]
    if sub.empty:
        print("  SKIP fig_switch_sweep_collision (no matching data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for s in STRATEGY_ORDER:
        d = sub[sub["strategy"] == s].sort_values("switch_prob")
        if d.empty: continue
        c = STRATEGY_COLORS[s]
        m = STRATEGY_MARKERS[s]
        l = STRATEGY_LABELS[s]
        x = d["switch_prob"].values

        ax1.errorbar(x, d["collision_rate"]*100,
                     yerr=[(d["collision_rate"]-d["collision_ci_lo"])*100,
                           (d["collision_ci_hi"]-d["collision_rate"])*100],
                     label=l, color=c, marker=m, capsize=3, linewidth=1.5, markersize=6)

        ax2.errorbar(x, d["mode_coverage"]*100,
                     yerr=[(d["mode_coverage"]-d["mode_cov_ci_lo"])*100,
                           (d["mode_cov_ci_hi"]-d["mode_coverage"])*100],
                     label=l, color=c, marker=m, capsize=3, linewidth=1.5, markersize=6)

    ax1.set_xlabel("Switch Probability"); ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("(a) Collision Rate vs Switch Prob")
    ax1.legend(loc="best"); ax1.grid(True, alpha=0.3); ax1.set_ylim(bottom=0)

    ax2.set_xlabel("Switch Probability"); ax2.set_ylabel("Mode Coverage (%)")
    ax2.set_title("(b) Mode Coverage vs Switch Prob")
    ax2.legend(loc="best"); ax2.grid(True, alpha=0.3); ax2.set_ylim(bottom=0)

    fig.tight_layout()
    save_fig(fig, "fig_sweep_switch_prob")


# ============================================================================
# Fig B: Collision rate vs rare_prob
# ============================================================================

def fig_rare_sweep(df):
    sub = df[(df["switch_prob"].between(0.09, 0.11)) & (df["obs_config"] == "1obs")]
    if sub.empty:
        print("  SKIP fig_rare_sweep (no matching data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for s in STRATEGY_ORDER:
        d = sub[sub["strategy"] == s].sort_values("rare_prob")
        if d.empty: continue
        c, m, l = STRATEGY_COLORS[s], STRATEGY_MARKERS[s], STRATEGY_LABELS[s]
        x = d["rare_prob"].values

        axes[0].errorbar(x, d["collision_rate"]*100,
                         yerr=[(d["collision_rate"]-d["collision_ci_lo"])*100,
                               (d["collision_ci_hi"]-d["collision_rate"])*100],
                         label=l, color=c, marker=m, capsize=3, linewidth=1.5, markersize=6)

        axes[1].errorbar(x, d["mode_coverage"]*100,
                         yerr=[(d["mode_coverage"]-d["mode_cov_ci_lo"])*100,
                               (d["mode_cov_ci_hi"]-d["mode_coverage"])*100],
                         label=l, color=c, marker=m, capsize=3, linewidth=1.5, markersize=6)

        axes[2].errorbar(x, d["rare_mode_coverage"]*100,
                         yerr=[(d["rare_mode_coverage"]-d["rare_cov_ci_lo"])*100,
                               (d["rare_cov_ci_hi"]-d["rare_mode_coverage"])*100],
                         label=l, color=c, marker=m, capsize=3, linewidth=1.5, markersize=6)

    axes[0].set_xlabel("Rare Mode Probability"); axes[0].set_ylabel("Collision Rate (%)")
    axes[0].set_title("(a) Collision Rate"); axes[0].legend(loc="best"); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Rare Mode Probability"); axes[1].set_ylabel("Mode Coverage (%)")
    axes[1].set_title("(b) Mode Coverage"); axes[1].legend(loc="best"); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel("Rare Mode Probability"); axes[2].set_ylabel("Rare Mode Coverage (%)")
    axes[2].set_title("(c) Rare Mode Coverage"); axes[2].legend(loc="best"); axes[2].grid(True, alpha=0.3)

    for ax in axes: ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, "fig_sweep_rare_prob")


# ============================================================================
# Fig C: Obstacle config comparison (grouped bars)
# ============================================================================

def fig_obs_config(df):
    sub = df[(df["switch_prob"].between(0.09, 0.11)) & (df["rare_prob"].between(0.04, 0.06))]
    if sub.empty:
        print("  SKIP fig_obs_config (no matching data)")
        return

    obs_configs = ["1obs", "4obs_ind", "4obs_shared"]
    obs_labels = ["1 obstacle", "4 obs (independent)", "4 obs (shared class)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(obs_configs))
    n_strat = len(STRATEGY_ORDER)
    width = 0.8 / n_strat

    for i, s in enumerate(STRATEGY_ORDER):
        cr_vals, mc_vals, cr_err_lo, cr_err_hi = [], [], [], []
        for oc in obs_configs:
            row = sub[(sub["strategy"] == s) & (sub["obs_config"] == oc)]
            if row.empty:
                cr_vals.append(0); mc_vals.append(0)
                cr_err_lo.append(0); cr_err_hi.append(0)
            else:
                r = row.iloc[0]
                cr_vals.append(r["collision_rate"]*100)
                mc_vals.append(r["mode_coverage"]*100)
                cr_err_lo.append((r["collision_rate"]-r["collision_ci_lo"])*100)
                cr_err_hi.append((r["collision_ci_hi"]-r["collision_rate"])*100)

        offset = (i - n_strat/2 + 0.5) * width
        c = STRATEGY_COLORS[s]
        l = STRATEGY_LABELS[s]

        ax1.bar(x + offset, cr_vals, width, yerr=[cr_err_lo, cr_err_hi],
                label=l, color=c, capsize=2, edgecolor="black", linewidth=0.4, alpha=0.85)
        ax2.bar(x + offset, mc_vals, width,
                label=l, color=c, edgecolor="black", linewidth=0.4, alpha=0.85)

    ax1.set_xticks(x); ax1.set_xticklabels(obs_labels)
    ax1.set_ylabel("Collision Rate (%)"); ax1.set_title("(a) Collision Rate by Obstacle Config")
    ax1.legend(loc="upper left", fontsize=7); ax1.grid(True, alpha=0.3, axis="y")

    ax2.set_xticks(x); ax2.set_xticklabels(obs_labels)
    ax2.set_ylabel("Mode Coverage (%)"); ax2.set_title("(b) Mode Coverage by Obstacle Config")
    ax2.legend(loc="upper left", fontsize=7); ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig_sweep_obs_config")


# ============================================================================
# Fig D: Safety vs coverage across all conditions (multi-panel scatter)
# ============================================================================

def fig_safety_coverage_all(df):
    obs_configs = df["obs_config"].unique()
    n_panels = len(obs_configs)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, min(n_panels, 3), figsize=(5*min(n_panels, 3), 4.5),
                              squeeze=False)
    axes = axes.flatten()

    for pi, oc in enumerate(sorted(obs_configs)):
        if pi >= 3: break
        ax = axes[pi]
        sub = df[df["obs_config"] == oc]

        for s in STRATEGY_ORDER:
            d = sub[sub["strategy"] == s]
            if d.empty: continue
            ax.scatter(d["mode_coverage"]*100, d["collision_rate"]*100,
                      c=STRATEGY_COLORS[s], marker=STRATEGY_MARKERS[s],
                      s=40, label=STRATEGY_LABELS[s], alpha=0.7,
                      edgecolors="black", linewidth=0.3)

        ax.set_xlabel("Mode Coverage (%)")
        ax.set_ylabel("Collision Rate (%)")
        ax.set_title(f"obs={oc}")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Safety vs Coverage Across All Conditions", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_sweep_safety_vs_coverage")


# ============================================================================
# Fig E: Solve time overhead heatmap
# ============================================================================

def fig_solve_overhead(df):
    sub = df[df["obs_config"] == "1obs"]
    if sub.empty:
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    switch_vals = sorted(sub["switch_prob"].unique())
    rare_vals = sorted(sub["rare_prob"].unique())

    for si, s in enumerate(STRATEGY_ORDER):
        ax = axes[si]
        mat = np.full((len(rare_vals), len(switch_vals)), np.nan)
        for ri, rp in enumerate(rare_vals):
            for ci, sp in enumerate(switch_vals):
                row = sub[(sub["strategy"] == s) &
                          (sub["switch_prob"].between(sp-0.001, sp+0.001)) &
                          (sub["rare_prob"].between(rp-0.001, rp+0.001))]
                if not row.empty:
                    mat[ri, ci] = row.iloc[0]["collision_rate"] * 100

        im = ax.imshow(mat, aspect="auto", origin="lower",
                       vmin=0, vmax=np.nanmax(mat) if not np.all(np.isnan(mat)) else 20,
                       cmap="YlOrRd")
        ax.set_xticks(range(len(switch_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in switch_vals], fontsize=8)
        ax.set_yticks(range(len(rare_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in rare_vals], fontsize=8)
        ax.set_xlabel("Switch Prob")
        ax.set_ylabel("Rare Prob")
        ax.set_title(STRATEGY_LABELS[s])

        # Annotate cells
        for ri in range(len(rare_vals)):
            for ci in range(len(switch_vals)):
                if not np.isnan(mat[ri, ci]):
                    ax.text(ci, ri, f"{mat[ri,ci]:.1f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if mat[ri,ci] > 10 else "black")

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Collision Rate (%) Across Parameter Grid (1 obstacle)", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_sweep_heatmap")


# ============================================================================
# LaTeX summary table
# ============================================================================

def generate_latex_table(df):
    path = os.path.join(OUT_DIR, "sweep_findings.tex")

    # Aggregate across all conditions per strategy
    agg = df.groupby("strategy").agg(
        mean_collision=("collision_rate", "mean"),
        mean_mode_cov=("mode_coverage", "mean"),
        mean_rare_cov=("rare_mode_coverage", "mean"),
        mean_solve=("mean_solve_ms", "mean"),
        mean_completion=("completion_rate", "mean"),
    ).reset_index()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Strategy performance aggregated across all sweep conditions.}",
        r"\label{tab:sweep_aggregate}",
        r"\small",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"Strategy & Avg Coll.\ (\%) & Avg Mode Cov.\ (\%) & Avg Rare Cov.\ (\%) & Solve (ms) & Compl.\ (\%) \\",
        r"\midrule",
    ]

    for s in STRATEGY_ORDER:
        row = agg[agg["strategy"] == s]
        if row.empty: continue
        r = row.iloc[0]
        label = STRATEGY_LABELS[s]
        lines.append(
            f"{label} & {r['mean_collision']*100:.1f} & "
            f"{r['mean_mode_cov']*100:.1f} & "
            f"{r['mean_rare_cov']*100:.1f} & "
            f"{r['mean_solve']:.2f} & "
            f"{r['mean_completion']*100:.1f} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading sweep data...")
    df = load()
    n_cond = df["condition"].nunique()
    n_strat = df["strategy"].nunique()
    print(f"  {n_cond} conditions x {n_strat} strategies\n")

    print("Generating sweep figures...")
    fig_switch_sweep_collision(df)
    fig_rare_sweep(df)
    fig_obs_config(df)
    fig_safety_coverage_all(df)
    fig_solve_overhead(df)

    print("\nGenerating LaTeX table...")
    generate_latex_table(df)

    print("\nDone.")
