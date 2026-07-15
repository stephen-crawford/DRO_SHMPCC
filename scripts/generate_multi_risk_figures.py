#!/usr/bin/env python3
"""Generate figures for multi-risk injection experiment.

Key question: does TopRisk degrade at high K because it greedily selects
redundant modes from the same risk cluster, while WDRO/DiverseRisk
spread across geometrically diverse threats?
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = pathlib.Path("figures/multi_risk")

COLORS = {
    "WDRO-inject":  "#2166AC",
    "TopRisk":      "#B2182B",
    "DiverseRisk":  "#4DAF4A",
}
BL_COLORS = {"Base": "#999999", "WDRO-sampling": "#FF7F00"}


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(DIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}")


def main():
    df = pd.read_csv(DIR / "multi_risk.csv")
    conditions = df["condition"].unique()

    # Parse method into base method + K
    def parse_method(m):
        if m in ("Base", "WDRO-sampling"):
            return m, 0
        parts = m.rsplit("-K", 1)
        return parts[0], int(parts[1])

    df["base_method"] = df["method"].apply(lambda m: parse_method(m)[0])
    df["K_val"] = df["method"].apply(lambda m: parse_method(m)[1])

    inj_methods = ["WDRO-inject", "TopRisk", "DiverseRisk"]

    # --- Main figure: collision rate vs K, per condition ---
    n_cond = len(conditions)
    cols = min(3, n_cond)
    rows = (n_cond + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]

        # Baselines
        for bl, color in BL_COLORS.items():
            row = sub[sub["method"] == bl]
            if len(row) > 0:
                v = row["collision_rate"].values[0]
                lo = row["coll_ci_lo"].values[0]
                hi = row["coll_ci_hi"].values[0]
                ax.axhline(v, color=color, ls="--", lw=1.5, alpha=0.7)
                ax.axhspan(lo, hi, color=color, alpha=0.06)
                ax.text(5.15, v, f"{bl}\n{v*100:.1f}%", fontsize=6.5,
                        va="center", color=color)

        # Injection methods
        for method in inj_methods:
            msub = sub[sub["base_method"] == method].sort_values("K_val")
            msub = msub[msub["K_val"] > 0]
            if len(msub) == 0:
                continue
            color = COLORS[method]
            K = msub["K_val"].values
            cr = msub["collision_rate"].values
            lo = msub["coll_ci_lo"].values
            hi = msub["coll_ci_hi"].values
            ax.plot(K, cr, "o-", color=color, lw=2.5, ms=7, label=method)
            ax.fill_between(K, lo, hi, color=color, alpha=0.1)

        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_ylabel("Collision Rate")
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Injection Count K")
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=8, loc="best")
    # Hide unused axes
    for i in range(len(conditions), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Multi-Risk Injection: Collision Rate vs K\n"
                 "10 modes with dual high-risk clusters (N=1000 per point)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig_multi_risk_collision")

    # --- Delta plot: TopRisk minus WDRO-inject ---
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        wdro = sub[sub["base_method"] == "WDRO-inject"].sort_values("K_val")
        wdro = wdro[wdro["K_val"] > 0]

        for comp, color, marker, ls in [
            ("TopRisk", "#B2182B", "o", "-"),
            ("DiverseRisk", "#4DAF4A", "s", "--"),
        ]:
            other = sub[sub["base_method"] == comp].sort_values("K_val")
            other = other[other["K_val"] > 0]
            if len(other) > 0 and len(wdro) > 0:
                K = wdro["K_val"].values[:len(other)]
                delta = (other["collision_rate"].values[:len(K)] -
                         wdro["collision_rate"].values[:len(K)]) * 100
                ax.plot(K, delta, f"{marker}{ls}", color=color, lw=2.5, ms=7,
                        label=f"{comp} minus WDRO-inject")

        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_ylabel("Collision Delta (pp)")
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Injection Count K")
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=8)
    for i in range(len(conditions), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("WDRO-inject Advantage Over Baselines\n"
                 "Positive = WDRO better (pp, N=1000)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig_multi_risk_delta")

    # --- Clearance vs K ---
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                             sharex=True, squeeze=False)
    axes = axes.flatten()

    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        for bl, color in BL_COLORS.items():
            row = sub[sub["method"] == bl]
            if len(row) > 0:
                ax.axhline(row["mean_clearance"].values[0], color=color,
                           ls="--", lw=1.5, alpha=0.7, label=bl)
        for method in inj_methods:
            msub = sub[sub["base_method"] == method].sort_values("K_val")
            msub = msub[msub["K_val"] > 0]
            if len(msub) == 0:
                continue
            color = COLORS[method]
            ax.plot(msub["K_val"], msub["mean_clearance"], "s-", color=color,
                    lw=2, ms=6, label=method)

        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Clearance")
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Injection Count K")
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=8, loc="best")
    for i in range(len(conditions), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Multi-Risk: Mean Clearance vs K (N=1000)", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "fig_multi_risk_clearance")

    # --- Summary table to stdout ---
    print("\n=== Key Comparisons at K=3 ===")
    for cond in conditions:
        sub = df[df["condition"] == cond]
        print(f"\n  {cond}:")
        base_row = sub[sub["method"] == "Base"]
        if len(base_row) > 0:
            print(f"    Base:           {base_row['collision_rate'].values[0]*100:.1f}%")
        for method in inj_methods:
            row = sub[(sub["base_method"] == method) & (sub["K_val"] == 3)]
            if len(row) > 0:
                cr = row["collision_rate"].values[0] * 100
                lo = row["coll_ci_lo"].values[0] * 100
                hi = row["coll_ci_hi"].values[0] * 100
                print(f"    {method}-K3: {cr:.1f}% [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    print("Generating multi-risk figures...")
    main()
    print("\nDone.")
