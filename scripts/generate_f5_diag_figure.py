#!/usr/bin/env python3
"""Generate figure summarizing F5 obstacle-setup diagnostic findings."""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = pathlib.Path("figures/focused")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Base": "#888888",
    "WDRO-sampling": "#1f77b4",
    "WDRO-inject-K1": "#2ca02c",
    "WDRO-inject-K2": "#17becf",
    "TopRisk-K1": "#ff7f0e",
    "TopRisk-K2": "#d62728",
}

METHOD_ORDER = ["Base", "WDRO-sampling", "WDRO-inject-K1",
                "WDRO-inject-K2", "TopRisk-K1", "TopRisk-K2"]

# Nice display labels for setups (ordered by difficulty)
SETUP_ORDER = [
    "no_switch", "closer_0.35", "no_switch+closer",
    "tight_path", "tight_no_switch",
    "two_obstacles", "tight+two+no_sw",
]
SETUP_LABELS = {
    "no_switch":        "No mode\nswitching",
    "closer_0.35":      "Closer\n(35% arc)",
    "no_switch+closer": "No switch\n+ closer",
    "tight_path":       "Tight\nS-curve",
    "tight_no_switch":  "Tight +\nno switch",
    "two_obstacles":    "Two\nobstacles",
    "tight+two+no_sw":  "Tight + two\n+ no switch",
}


def wilson_clearance_ci(mean_clr, p5_clr, n):
    std_est = max((mean_clr - p5_clr) / 1.645, 1e-6)
    margin = 1.96 * std_est / np.sqrt(n)
    return mean_clr - margin, mean_clr + margin


def main():
    df = pd.read_csv(OUT / "f5_diag_obstacle_setup.csv")

    setups = [s for s in SETUP_ORDER if s in df["setup"].unique()]
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    n_methods = len(methods)
    x = np.arange(len(setups))
    w = 0.8 / n_methods

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Collision rate ---
    ax = axes[0]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals, err_lo, err_hi = [], [], []
        for s in setups:
            row = sub[sub["setup"] == s]
            if row.empty:
                vals.append(0); err_lo.append(0); err_hi.append(0)
                continue
            r = row.iloc[0]
            vals.append(r["collision_rate"] * 100)
            err_lo.append((r["collision_rate"] - r["coll_ci_lo"]) * 100)
            err_hi.append((r["coll_ci_hi"] - r["collision_rate"]) * 100)
        ax.bar(x + i * w, vals, w, yerr=[err_lo, err_hi],
               label=m, color=COLORS.get(m, "#1f77b4"), capsize=2, alpha=0.85)
    ax.set_xticks(x + w * (n_methods - 1) / 2)
    ax.set_xticklabels([SETUP_LABELS[s] for s in setups], fontsize=8)
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Collision Rate by Obstacle Setup (2.0 m/s, N=500)")
    ax.legend(fontsize=7, loc="upper left")

    # --- Mean clearance with Wilson CI ---
    ax = axes[1]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals, err_lo, err_hi = [], [], []
        for s in setups:
            row = sub[sub["setup"] == s]
            if row.empty:
                vals.append(0); err_lo.append(0); err_hi.append(0)
                continue
            r = row.iloc[0]
            mc = r["mean_clearance"]
            p5 = r["p5_clearance"]
            n = r["n_rollouts"]
            clo, chi = wilson_clearance_ci(mc, p5, n)
            vals.append(mc)
            err_lo.append(mc - clo)
            err_hi.append(chi - mc)
        ax.bar(x + i * w, vals, w, yerr=[err_lo, err_hi],
               label=m, color=COLORS.get(m, "#1f77b4"), capsize=2, alpha=0.85)
    ax.set_xticks(x + w * (n_methods - 1) / 2)
    ax.set_xticklabels([SETUP_LABELS[s] for s in setups], fontsize=8)
    ax.set_ylabel("Mean Clearance (m)")
    ax.set_title("Mean Clearance by Obstacle Setup (2.0 m/s, N=500)")
    ax.legend(fontsize=7, loc="upper left")

    # Add a vertical dashed line separating single-obstacle from multi-obstacle setups
    for ax in axes:
        ax.axvline(x=4.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle("Obstacle Setup Diagnostic: Single Oncoming Obstacle at 2.0 m/s is Trivially Dodgeable\n"
                 "Multi-obstacle setups required for meaningful collision rates",
                 fontsize=12, y=1.04)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_f5_diag_obstacle_setup.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_f5_diag_obstacle_setup")


if __name__ == "__main__":
    main()
