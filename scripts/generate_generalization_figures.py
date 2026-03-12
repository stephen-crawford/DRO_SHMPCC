#!/usr/bin/env python3
"""Generate figures from G1–G5 generalization experiments.

Fig G1: Path geometry comparison (grouped bars per path type)
Fig G2: Multi-obstacle scaling (grouped bars by obstacle count)
Fig G3: Switching dynamics sweep (line plots over switch_prob)
Fig G4: Mode diversity (grouped bars by mode count)
Fig G5: Path × obstacle interaction (heatmaps)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import sys

DIR = "generalization_figures"
os.makedirs(DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# Color scheme
COLORS = {
    "Base":             "#999999",
    "WDRO-sampling":    "#FF7F00",
    "WDRO-inject-K1":   "#2166AC",
    "WDRO-inject-K2":   "#4393C3",
    "TopRisk-K1":       "#B2182B",
    "DiverseRisk-K1":   "#4DAF4A",
    "Softmax-tau5":     "#984EA3",
}
MARKERS = {
    "Base":             "s",
    "WDRO-sampling":    "D",
    "WDRO-inject-K1":   "o",
    "WDRO-inject-K2":   "v",
    "TopRisk-K1":       "^",
    "DiverseRisk-K1":   "P",
    "Softmax-tau5":     "X",
}

def _color(method):
    return COLORS.get(method, "#666666")

def _marker(method):
    return MARKERS.get(method, "o")


# ============================================================================
# Fig G1: Path Geometry Comparison
# ============================================================================
def fig_g1():
    fpath = os.path.join(DIR, "g1_path_geometry.csv")
    if not os.path.exists(fpath):
        print("  Fig G1: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    paths = df["path"].unique()
    methods = df["method"].unique()
    n_methods = len(methods)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    x = np.arange(len(paths))
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        # Reorder to match paths array
        cr = [sub[sub["path"] == p]["collision_rate"].values[0] * 100 if len(sub[sub["path"] == p]) else 0 for p in paths]
        cr_lo = [sub[sub["path"] == p]["coll_ci_lo"].values[0] * 100 if len(sub[sub["path"] == p]) else 0 for p in paths]
        cr_hi = [sub[sub["path"] == p]["coll_ci_hi"].values[0] * 100 if len(sub[sub["path"] == p]) else 0 for p in paths]
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        mm = [sub[sub["path"] == p]["missed_mode_rate"].values[0] * 100 if len(sub[sub["path"] == p]) else 0 for p in paths]
        mm_lo = [sub[sub["path"] == p]["mm_ci_lo"].values[0] * 100 if len(sub[sub["path"] == p]) else 0 for p in paths]
        mm_hi = [sub[sub["path"] == p]["mm_ci_hi"].values[0] * 100 if len(sub[sub["path"] == p]) else 0 for p in paths]
        mm = np.array(mm)
        mm_err = np.array([mm - np.array(mm_lo), np.array(mm_hi) - mm])

        prog = [sub[sub["path"] == p]["mean_progress"].values[0] if len(sub[sub["path"] == p]) else 0 for p in paths]

        offset = (i - n_methods / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.9, yerr=err, label=method,
                color=_color(method), capsize=2, error_kw={"linewidth": 0.7})
        ax2.bar(x + offset, mm, width * 0.9, yerr=mm_err,
                color=_color(method), capsize=2, error_kw={"linewidth": 0.7})
        ax3.bar(x + offset, prog, width * 0.9, color=_color(method))

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate by Path Geometry")
    ax2.set_ylabel("Missed Mode Rate (%)")
    ax2.set_title("Missed Mode Rate by Path Geometry")
    ax3.set_ylabel("Mean Progress (fraction)")
    ax3.set_title("Path Progress by Geometry")
    ax3.set_xticks(x)
    ax3.set_xticklabels(paths, fontsize=9)

    ax1.legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("G1: Path Geometry Comparison", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_g1_path_geometry.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig G1 saved")


# ============================================================================
# Fig G2: Multi-Obstacle Scaling
# ============================================================================
def fig_g2():
    fpath = os.path.join(DIR, "g2_multi_obstacle.csv")
    if not os.path.exists(fpath):
        print("  Fig G2: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    obs_counts = sorted(df["num_obstacles"].unique())
    methods = df["method"].unique()
    n_methods = len(methods)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    x = np.arange(len(obs_counts))
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr = [sub[sub["num_obstacles"] == n]["collision_rate"].values[0] * 100 if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr_lo = [sub[sub["num_obstacles"] == n]["coll_ci_lo"].values[0] * 100 if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr_hi = [sub[sub["num_obstacles"] == n]["coll_ci_hi"].values[0] * 100 if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        mm = [sub[sub["num_obstacles"] == n]["missed_mode_rate"].values[0] * 100 if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        mm = np.array(mm)

        offset = (i - n_methods / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.9, yerr=err, label=method,
                color=_color(method), capsize=2, error_kw={"linewidth": 0.7})
        ax2.bar(x + offset, mm, width * 0.9, color=_color(method))

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs Obstacle Count")
    ax2.set_ylabel("Missed Mode Rate (%)")
    ax2.set_title("Missed Mode Rate vs Obstacle Count")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in obs_counts])
    ax2.set_xlabel("Number of Obstacles")

    ax1.legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle("G2: Multi-Obstacle Scaling", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_g2_multi_obstacle.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig G2 saved")


# ============================================================================
# Fig G3: Switching Dynamics Sweep (line plots)
# ============================================================================
def fig_g3():
    fpath = os.path.join(DIR, "g3_switch_dynamics.csv")
    if not os.path.exists(fpath):
        print("  Fig G3: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("switch_prob")
        sp = sub["switch_prob"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        mm = sub["missed_mode_rate"].values * 100
        mlo = sub["mm_ci_lo"].values * 100
        mhi = sub["mm_ci_hi"].values * 100

        style = dict(color=_color(method), marker=_marker(method), markersize=5, linewidth=1.3)
        ax1.plot(sp, cr, label=method, **style)
        ax1.fill_between(sp, clo, chi, alpha=0.1, color=_color(method))
        ax2.plot(sp, mm, label=method, **style)
        ax2.fill_between(sp, mlo, mhi, alpha=0.1, color=_color(method))

    ax1.set_xlabel("Switch Probability")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs Switch Probability")
    ax1.legend(fontsize=7)

    ax2.set_xlabel("Switch Probability")
    ax2.set_ylabel("Missed Mode Rate (%)")
    ax2.set_title("Missed Mode Rate vs Switch Probability")

    fig.suptitle("G3: Switching Dynamics Sweep", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_g3_switch_dynamics.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig G3 saved")


# ============================================================================
# Fig G4: Mode Diversity
# ============================================================================
def fig_g4():
    fpath = os.path.join(DIR, "g4_mode_diversity.csv")
    if not os.path.exists(fpath):
        print("  Fig G4: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    mode_sets = df["mode_set"].unique()
    methods = df["method"].unique()
    n_methods = len(methods)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    x = np.arange(len(mode_sets))
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr = [sub[sub["mode_set"] == ms]["collision_rate"].values[0] * 100 if len(sub[sub["mode_set"] == ms]) else 0 for ms in mode_sets]
        cr_lo = [sub[sub["mode_set"] == ms]["coll_ci_lo"].values[0] * 100 if len(sub[sub["mode_set"] == ms]) else 0 for ms in mode_sets]
        cr_hi = [sub[sub["mode_set"] == ms]["coll_ci_hi"].values[0] * 100 if len(sub[sub["mode_set"] == ms]) else 0 for ms in mode_sets]
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        mm = [sub[sub["mode_set"] == ms]["missed_mode_rate"].values[0] * 100 if len(sub[sub["mode_set"] == ms]) else 0 for ms in mode_sets]
        mm = np.array(mm)

        offset = (i - n_methods / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.9, yerr=err, label=method,
                color=_color(method), capsize=2, error_kw={"linewidth": 0.7})
        ax2.bar(x + offset, mm, width * 0.9, color=_color(method))

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate by Mode Diversity")
    ax2.set_ylabel("Missed Mode Rate (%)")
    ax2.set_title("Missed Mode Rate by Mode Diversity")
    ax2.set_xticks(x)
    ax2.set_xticklabels(mode_sets, fontsize=9)
    ax2.set_xlabel("Mode Set")

    ax1.legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle("G4: Mode Diversity", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_g4_mode_diversity.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig G4 saved")


# ============================================================================
# Fig G5: Path × Obstacle Interaction (heatmaps: WDRO improvement over Base)
# ============================================================================
def fig_g5():
    fpath = os.path.join(DIR, "g5_path_obstacle_grid.csv")
    if not os.path.exists(fpath):
        print("  Fig G5: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    paths = df["path"].unique()
    obs_counts = sorted(df["num_obstacles"].unique())

    # Compare WDRO-inject-K1 improvement over Base
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, title, fmt in [
        (ax1, "collision_rate", "Collision Rate Reduction (%)", ".1f"),
        (ax2, "missed_mode_rate", "Missed Mode Rate Reduction (%)", ".1f"),
    ]:
        matrix = np.zeros((len(paths), len(obs_counts)))
        for i, p in enumerate(paths):
            for j, n in enumerate(obs_counts):
                base_val = df[(df["path"] == p) & (df["num_obstacles"] == n) & (df["method"] == "Base")]
                wdro_val = df[(df["path"] == p) & (df["num_obstacles"] == n) & (df["method"] == "WDRO-inject-K1")]
                if not base_val.empty and not wdro_val.empty:
                    b = base_val.iloc[0][metric] * 100
                    w = wdro_val.iloc[0][metric] * 100
                    matrix[i, j] = b - w  # Positive = improvement

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(obs_counts)))
        ax.set_xticklabels([str(n) for n in obs_counts])
        ax.set_yticks(range(len(paths)))
        ax.set_yticklabels(paths, fontsize=8)
        ax.set_xlabel("Num Obstacles")
        ax.set_title(title)
        for i in range(len(paths)):
            for j in range(len(obs_counts)):
                val = matrix[i, j]
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > 3 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("G5: WDRO-inject-K1 Improvement over Base", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_g5_path_obstacle.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig G5 saved")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating generalization figures...")
    fig_g1()
    fig_g2()
    fig_g3()
    fig_g4()
    fig_g5()
    print("Done.")
