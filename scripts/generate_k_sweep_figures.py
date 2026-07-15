#!/usr/bin/env python3
"""Generate figures for injection K-sweep experiment."""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = pathlib.Path("figures/k_sweep")

COLORS = {
    "WDRO-inject":  "#2166AC",
    "TopRisk":      "#B2182B",
    "DiverseRisk":  "#4DAF4A",
}
BASELINES = {"Base": "#999999", "WDRO-sampling": "#FF7F00"}


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(DIR / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}")


def main():
    df = pd.read_csv(DIR / "k_sweep.csv")
    conditions = df["condition"].unique()
    methods = [m for m in df["method"].unique() if m not in ("Base", "WDRO-sampling")]

    # --- Per-condition: collision rate vs K ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, cond in zip(axes, conditions):
        # Baselines as horizontal lines
        for bl, color in BASELINES.items():
            row = df[(df["condition"] == cond) & (df["method"] == bl)]
            if len(row) > 0:
                val = row["collision_rate"].values[0]
                lo = row["coll_ci_lo"].values[0]
                hi = row["coll_ci_hi"].values[0]
                ax.axhline(val, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
                ax.axhspan(lo, hi, color=color, alpha=0.08)
                ax.text(6.1, val, f"{bl}\n{val*100:.1f}%", fontsize=7,
                        va="center", color=color)

        # K-sweep lines
        for method in methods:
            sub = df[(df["condition"] == cond) & (df["method"] == method)].sort_values("K")
            if len(sub) == 0:
                continue
            K = sub["K"].values
            cr = sub["collision_rate"].values
            lo = sub["coll_ci_lo"].values
            hi = sub["coll_ci_hi"].values
            color = COLORS.get(method, "#333333")
            ax.plot(K, cr, "o-", color=color, linewidth=2, markersize=6, label=method)
            ax.fill_between(K, lo, hi, color=color, alpha=0.12)

        ax.set_title(cond, fontsize=11)
        ax.set_ylabel("Collision Rate")
        ax.set_xticks(range(1, 7))
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc="best")

    axes[2].set_xlabel("Injection Count K")
    axes[3].set_xlabel("Injection Count K")
    fig.suptitle("Collision Rate vs Injection Count K (N=1000 per point)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "fig_k_sweep_collision")

    # --- Per-condition: clearance vs K ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, cond in zip(axes, conditions):
        for bl, color in BASELINES.items():
            row = df[(df["condition"] == cond) & (df["method"] == bl)]
            if len(row) > 0:
                ax.axhline(row["mean_clearance"].values[0], color=color,
                           linestyle="--", linewidth=1.5, alpha=0.7, label=bl)

        for method in methods:
            sub = df[(df["condition"] == cond) & (df["method"] == method)].sort_values("K")
            if len(sub) == 0:
                continue
            color = COLORS.get(method, "#333333")
            ax.plot(sub["K"], sub["mean_clearance"], "s-", color=color,
                    linewidth=2, markersize=6, label=method)

        ax.set_title(cond, fontsize=11)
        ax.set_ylabel("Mean Clearance")
        ax.set_xticks(range(1, 7))
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc="best")

    axes[2].set_xlabel("Injection Count K")
    axes[3].set_xlabel("Injection Count K")
    fig.suptitle("Mean Clearance vs Injection Count K (N=1000 per point)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "fig_k_sweep_clearance")

    # --- Delta plot: WDRO-inject minus TopRisk at each K ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, cond in zip(axes, conditions):
        wdro = df[(df["condition"] == cond) & (df["method"] == "WDRO-inject")].sort_values("K")
        tr = df[(df["condition"] == cond) & (df["method"] == "TopRisk")].sort_values("K")
        dr = df[(df["condition"] == cond) & (df["method"] == "DiverseRisk")].sort_values("K")

        if len(wdro) > 0 and len(tr) > 0:
            K = wdro["K"].values
            delta = (tr["collision_rate"].values - wdro["collision_rate"].values) * 100
            ax.plot(K, delta, "o-", color="#2166AC", linewidth=2.5, markersize=8,
                    label="TopRisk minus WDRO-inject")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="-")

        if len(dr) > 0 and len(wdro) > 0:
            K = dr["K"].values[:len(wdro)]
            delta2 = (dr["collision_rate"].values[:len(wdro)] - wdro["collision_rate"].values[:len(K)]) * 100
            ax.plot(K, delta2, "s--", color="#4DAF4A", linewidth=2, markersize=7,
                    label="DiverseRisk minus WDRO-inject")

        ax.set_title(cond, fontsize=11)
        ax.set_ylabel("Collision Rate Delta (pp)")
        ax.set_xticks(range(1, 7))
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    axes[2].set_xlabel("Injection Count K")
    axes[3].set_xlabel("Injection Count K")
    fig.suptitle("WDRO-inject Advantage: Positive = WDRO Better (pp, N=1000)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    save(fig, "fig_k_sweep_delta")

    # --- Solve time vs K ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        sub = df[(df["condition"] == "S-curve") & (df["method"] == method)].sort_values("K")
        if len(sub) == 0:
            continue
        color = COLORS.get(method, "#333333")
        ax.plot(sub["K"], sub["mean_solve_ms"], "o-", color=color,
                linewidth=2, markersize=6, label=method)
    ax.set_xlabel("Injection Count K")
    ax.set_ylabel("Mean Solve Time (ms)")
    ax.set_title("Solve Time vs K (S-curve condition)")
    ax.legend()
    ax.grid(True, alpha=0.2)
    save(fig, "fig_k_sweep_runtime")


if __name__ == "__main__":
    print("Generating K-sweep figures...")
    main()
    print("Done.")
