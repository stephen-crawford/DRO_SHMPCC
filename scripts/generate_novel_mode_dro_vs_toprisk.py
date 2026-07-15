#!/usr/bin/env python3
"""
Generate focused figures: DRO vs TopRisk for novel (unseen) obstacle modes.

Uses:
  - R1 data (r1_novel_mode.csv): DRO vs TopRisk across novel mode types
  - R5 data (r5_novel_multi_obstacle.csv): novel mode + multi-obstacle scaling

Output: figures/robustness/fig_dro_vs_toprisk_novel_*.{pdf,png}
"""

import pathlib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

DATA_DIR = pathlib.Path("figures/robustness")
OUT = DATA_DIR
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

METHOD_STYLE = {
    "WDRO-inject-K1": {"color": "#2ca02c", "label": "WDRO-inject (K=1)", "marker": "o", "ls": "-"},
    "WDRO-sampling":  {"color": "#1f77b4", "label": "WDRO-sampling (q*)", "marker": "s", "ls": "--"},
    "TopRisk-K1":     {"color": "#ff7f0e", "label": "TopRisk (K=1)",      "marker": "^", "ls": "-"},
    "TopRisk-K2":     {"color": "#d62728", "label": "TopRisk (K=2)",      "marker": "v", "ls": "--"},
    "WDRO-inject-K2": {"color": "#17becf", "label": "WDRO-inject (K=2)", "marker": "D", "ls": ":"},
    "Base":           {"color": "#888888", "label": "Base (no DRO)",      "marker": "x", "ls": ":"},
}


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}")


# ── R1: Novel mode types ─────────────────────────────────────────────────

def fig_r1(df):
    """Grouped bar chart of collision rates across novel mode types."""
    modes = list(df["novel_mode"].unique())
    methods = ["WDRO-inject-K1", "WDRO-sampling", "TopRisk-K1", "TopRisk-K2"]
    x = np.arange(len(modes))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel (a): Collision rate bars (zoomed, no Base) ──
    ax = axes[0]
    for i, m in enumerate(methods):
        sty = METHOD_STYLE[m]
        sub = df[df["method"] == m]
        vals, err_lo, err_hi = [], [], []
        for nm in modes:
            row = sub[sub["novel_mode"] == nm]
            if row.empty:
                vals.append(0); err_lo.append(0); err_hi.append(0)
                continue
            row = row.iloc[0]
            v = row["collision_rate"]
            vals.append(v)
            err_lo.append(v - row["coll_ci_lo"])
            err_hi.append(row["coll_ci_hi"] - v)
        ax.bar(x + (i - 1.5) * w, vals, w,
               yerr=[err_lo, err_hi],
               label=sty["label"], color=sty["color"],
               capsize=3, alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Collision Rate")
    ymax = max(0.18, df[df["method"] != "Base"]["collision_rate"].max() + 0.04)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("(a) Collision Rate: DRO vs TopRisk (Novel Modes)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (b): DRO advantage delta ──
    ax = axes[1]
    tr = df[df["method"] == "TopRisk-K1"].set_index("novel_mode")

    for method, color, marker, label in [
        ("WDRO-inject-K1", "#2ca02c", "o", "WDRO-inject (K=1)"),
        ("WDRO-sampling",  "#1f77b4", "s", "WDRO-sampling (q*)"),
    ]:
        dro = df[df["method"] == method].set_index("novel_mode")
        deltas, delta_err = [], []
        for nm in modes:
            if nm not in tr.index or nm not in dro.index:
                deltas.append(0); delta_err.append(0)
                continue
            tr_rate = tr.loc[nm, "collision_rate"]
            dro_rate = dro.loc[nm, "collision_rate"]
            tr_ci_w = (tr.loc[nm, "coll_ci_hi"] - tr.loc[nm, "coll_ci_lo"]) / 2
            dro_ci_w = (dro.loc[nm, "coll_ci_hi"] - dro.loc[nm, "coll_ci_lo"]) / 2
            delta_ci = np.sqrt(tr_ci_w**2 + dro_ci_w**2)
            deltas.append(tr_rate - dro_rate)
            delta_err.append(delta_ci)

        ax.errorbar(x, [d * 100 for d in deltas],
                    yerr=[e * 100 for e in delta_err],
                    fmt=f"{marker}-", color=color, label=label,
                    capsize=4, linewidth=2, markersize=7, capthick=1.5)

    ax.axhline(0, color="#888888", linewidth=1.2, zorder=0)
    ax.text(0.98, 0.96, r"$\uparrow$ DRO better", transform=ax.transAxes,
            fontsize=8.5, color="#2ca02c", fontstyle="italic", va="top", ha="right")
    ax.text(0.98, 0.04, r"$\downarrow$ TopRisk better", transform=ax.transAxes,
            fontsize=8.5, color="#ff7f0e", fontstyle="italic", va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Collision Rate Advantage (pp)")
    ax.set_title("(b) DRO Advantage Over TopRisk-K1")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_xlim(-0.4, len(modes) - 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("R1: DRO vs TopRisk Under Novel (Unseen) Obstacle Modes",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "fig_dro_vs_toprisk_novel_collision")


# ── R5: Novel mode + multi-obstacle ──────────────────────────────────────

def fig_r5(df):
    """Show collision rates across 1/2/3 obstacles with sharp swerve on obs #0.
    Two panels: (a) grouped bars, (b) DRO advantage delta vs obstacle count."""

    obs_counts = sorted(df["num_obstacles"].unique())
    x = np.arange(len(obs_counts))
    obs_labels = [f"{n} obstacle{'s' if n > 1 else ''}" for n in obs_counts]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel (a): Collision rate bars ──
    ax = axes[0]
    all_show = ["Base", "WDRO-inject-K1", "WDRO-sampling", "TopRisk-K1", "TopRisk-K2"]
    w_a = 0.15
    for i, m in enumerate(all_show):
        sty = METHOD_STYLE[m]
        sub = df[df["method"] == m]
        vals, err_lo, err_hi = [], [], []
        for nobs in obs_counts:
            row = sub[sub["num_obstacles"] == nobs]
            if row.empty:
                vals.append(0); err_lo.append(0); err_hi.append(0)
                continue
            row = row.iloc[0]
            v = row["collision_rate"]
            vals.append(v)
            err_lo.append(v - row["coll_ci_lo"])
            err_hi.append(row["coll_ci_hi"] - v)
        ax.bar(x + (i - 2) * w_a, vals, w_a,
               yerr=[err_lo, err_hi],
               label=sty["label"], color=sty["color"],
               capsize=3, alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(obs_labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Collision Rate")
    ax.set_title("(a) Collision Rate: Sharp Swerve + Multi-Obstacle")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel (b): DRO advantage over TopRisk-K1 vs obstacle count ──
    ax = axes[1]
    tr = df[df["method"] == "TopRisk-K1"].set_index("num_obstacles")

    for method, color, marker, label in [
        ("WDRO-inject-K1", "#2ca02c", "o", "WDRO-inject (K=1)"),
        ("WDRO-sampling",  "#1f77b4", "s", "WDRO-sampling (q*)"),
        ("WDRO-inject-K2", "#17becf", "D", "WDRO-inject (K=2)"),
    ]:
        dro = df[df["method"] == method].set_index("num_obstacles")
        deltas, delta_err = [], []
        for nobs in obs_counts:
            if nobs not in tr.index or nobs not in dro.index:
                deltas.append(0); delta_err.append(0)
                continue
            tr_rate = tr.loc[nobs, "collision_rate"]
            dro_rate = dro.loc[nobs, "collision_rate"]
            tr_ci_w = (tr.loc[nobs, "coll_ci_hi"] - tr.loc[nobs, "coll_ci_lo"]) / 2
            dro_ci_w = (dro.loc[nobs, "coll_ci_hi"] - dro.loc[nobs, "coll_ci_lo"]) / 2
            delta_ci = np.sqrt(tr_ci_w**2 + dro_ci_w**2)
            deltas.append(tr_rate - dro_rate)
            delta_err.append(delta_ci)

        ax.errorbar(x, [d * 100 for d in deltas],
                    yerr=[e * 100 for e in delta_err],
                    fmt=f"{marker}-", color=color, label=label,
                    capsize=4, linewidth=2, markersize=7, capthick=1.5)

    ax.axhline(0, color="#888888", linewidth=1.2, zorder=0)
    ax.text(0.98, 0.96, r"$\uparrow$ DRO better", transform=ax.transAxes,
            fontsize=8.5, color="#2ca02c", fontstyle="italic", va="top", ha="right")
    ax.text(0.98, 0.04, r"$\downarrow$ TopRisk better", transform=ax.transAxes,
            fontsize=8.5, color="#ff7f0e", fontstyle="italic", va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(obs_labels)
    ax.set_ylabel("Collision Rate Advantage (pp)")
    ax.set_title("(b) DRO Advantage Over TopRisk-K1")
    ax.legend(fontsize=7.5, loc="best")
    ax.set_xlim(-0.4, len(obs_counts) - 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("R5: Novel Mode (Sharp Swerve) + Multiple Obstacles",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, "fig_r5_novel_multi_obstacle")


# ── Main ─────────────────────────────────────────────────────────────────

def print_table(df, group_col, methods):
    """Print a summary table for a dataframe."""
    for val in sorted(df[group_col].unique()):
        print(f"\n    {group_col}={val}:")
        for m in methods:
            row = df[(df[group_col] == val) & (df["method"] == m)]
            if row.empty:
                continue
            row = row.iloc[0]
            print(f"      {m:20s}: {row['collision_rate']*100:5.1f}%  "
                  f"[{row['coll_ci_lo']*100:.1f}, {row['coll_ci_hi']*100:.1f}]")


if __name__ == "__main__":
    print("Generating DRO vs TopRisk novel-mode figures...")
    methods_to_show = ["WDRO-inject-K1", "WDRO-sampling", "TopRisk-K1", "TopRisk-K2"]

    # R1
    r1_path = DATA_DIR / "r1_novel_mode.csv"
    if r1_path.exists():
        df_r1 = pd.read_csv(r1_path)
        print("\n  R1: Novel mode types:")
        print_table(df_r1, "novel_mode", methods_to_show)
        print()
        fig_r1(df_r1)
    else:
        print(f"  SKIP R1: {r1_path} not found")

    # R5
    r5_path = DATA_DIR / "r5_novel_multi_obstacle.csv"
    if r5_path.exists():
        df_r5 = pd.read_csv(r5_path)
        print("\n  R5: Novel mode + multi-obstacle:")
        print_table(df_r5, "num_obstacles", methods_to_show)
        print()
        fig_r5(df_r5)
    else:
        print(f"  SKIP R5: {r5_path} not found")

    print(f"\nDone. Figures in {OUT}/")
