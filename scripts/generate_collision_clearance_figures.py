#!/usr/bin/env python3
"""
Generate figures from collision clearance test CSVs (c1-c6).

Usage:
    cd build
    python3 ../scripts/generate_collision_clearance_figures.py

Reads: figures/paper/c1_*.csv through c6_*.csv
Writes: figures/paper/fig_c1_*.png through fig_c6_*.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})

OUTPUT_DIR = "figures/paper"

METHOD_COLORS = {
    "Base":         "#1f77b4",
    "Base+SH":      "#aec7e8",
    "DRO(inj)":     "#2ca02c",
    "DRO(inj)+SH":  "#98df8a",
    "DRO(q*)+SH":   "#ff7f0e",
    "TopRisk+SH":   "#d62728",
}

METHOD_MARKERS = {
    "Base":         "o",
    "Base+SH":      "s",
    "DRO(inj)":     "^",
    "DRO(inj)+SH":  "D",
    "DRO(q*)+SH":   "v",
    "TopRisk+SH":   "X",
}


def load_csv(name):
    for d in [OUTPUT_DIR, ".", ".."]:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return pd.read_csv(p)
    print(f"  WARNING: {name} not found, skipping")
    return None


def get_color(method):
    return METHOD_COLORS.get(method, "#333333")


def get_marker(method):
    return METHOD_MARKERS.get(method, "o")


# ============================================================================
# C1: Clearance vs Switching Probability
# ============================================================================

def fig_c1():
    df = load_csv("c1_clearance_vs_switching.csv")
    if df is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    methods = df["method"].unique()

    # (a) Collision rate vs switch_prob
    ax = axes[0, 0]
    for m in methods:
        sub = df[df["method"] == m].sort_values("switch_prob")
        ax.errorbar(sub["switch_prob"], sub["collision_rate"] * 100,
                    yerr=[
                        (sub["collision_rate"] - sub["coll_ci_lo"]) * 100,
                        (sub["coll_ci_hi"] - sub["collision_rate"]) * 100,
                    ],
                    label=m, color=get_color(m), marker=get_marker(m),
                    markersize=5, capsize=3, linewidth=1.2)
    ax.set_xlabel("Switch Probability")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("(a) Collision Rate vs Switching")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Mean min clearance vs switch_prob
    ax = axes[0, 1]
    for m in methods:
        sub = df[df["method"] == m].sort_values("switch_prob")
        ax.plot(sub["switch_prob"], sub["mean_min_clearance"],
                label=m, color=get_color(m), marker=get_marker(m),
                markersize=5, linewidth=1.2)
    ax.set_xlabel("Switch Probability")
    ax.set_ylabel("Mean Min Clearance (m)")
    ax.set_title("(b) Mean Clearance vs Switching")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) 5th percentile clearance vs switch_prob
    ax = axes[1, 0]
    for m in methods:
        sub = df[df["method"] == m].sort_values("switch_prob")
        ax.plot(sub["switch_prob"], sub["p5_min_clearance"],
                label=m, color=get_color(m), marker=get_marker(m),
                markersize=5, linewidth=1.2)
    ax.set_xlabel("Switch Probability")
    ax.set_ylabel("5th Percentile Clearance (m)")
    ax.set_title("(c) Tail Clearance (p5) vs Switching")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Clearance std dev vs switch_prob
    ax = axes[1, 1]
    for m in methods:
        sub = df[df["method"] == m].sort_values("switch_prob")
        ax.plot(sub["switch_prob"], sub["stddev_min_clearance"],
                label=m, color=get_color(m), marker=get_marker(m),
                markersize=5, linewidth=1.2)
    ax.set_xlabel("Switch Probability")
    ax.set_ylabel("Clearance Std Dev (m)")
    ax.set_title("(d) Clearance Variability vs Switching")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("C1: Collision Clearance vs Switching Probability", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_c1_clearance_vs_switching.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")


# ============================================================================
# C2: Clearance vs Obstacle Count
# ============================================================================

def fig_c2():
    df = load_csv("c2_clearance_vs_obstacles.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = df["method"].unique()
    configs = df["obs_config"].unique()

    x = np.arange(len(configs))
    width = 0.8 / len(methods)

    # (a) Collision rate
    ax = axes[0]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["obs_config"] == c]["collision_rate"].values[0] * 100
                if len(sub[sub["obs_config"] == c]) > 0 else 0 for c in configs]
        lo = [sub[sub["obs_config"] == c]["coll_ci_lo"].values[0] * 100
              if len(sub[sub["obs_config"] == c]) > 0 else 0 for c in configs]
        hi = [sub[sub["obs_config"] == c]["coll_ci_hi"].values[0] * 100
              if len(sub[sub["obs_config"] == c]) > 0 else 0 for c in configs]
        yerr_lo = [v - l for v, l in zip(vals, lo)]
        yerr_hi = [h - v for v, h in zip(vals, hi)]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               yerr=[yerr_lo, yerr_hi], label=m, color=get_color(m),
               capsize=2, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("(a) Collision Rate")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) Mean clearance
    ax = axes[1]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["obs_config"] == c]["mean_min_clearance"].values[0]
                if len(sub[sub["obs_config"] == c]) > 0 else 0 for c in configs]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=m, color=get_color(m), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("Mean Min Clearance (m)")
    ax.set_title("(b) Mean Clearance")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # (c) p5 clearance
    ax = axes[2]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["obs_config"] == c]["p5_min_clearance"].values[0]
                if len(sub[sub["obs_config"] == c]) > 0 else 0 for c in configs]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=m, color=get_color(m), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("5th Percentile Clearance (m)")
    ax.set_title("(c) Tail Clearance (p5)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("C2: Collision Clearance vs Obstacle Count", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_c2_clearance_vs_obstacles.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")


# ============================================================================
# C3: Clearance Across Environment Types
# ============================================================================

def fig_c3():
    df = load_csv("c3_clearance_vs_environment.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = df["method"].unique()
    envs = df["environment"].unique()

    x = np.arange(len(envs))
    width = 0.8 / len(methods)

    # (a) Collision rate per environment
    ax = axes[0]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["environment"] == e]["collision_rate"].values[0] * 100
                if len(sub[sub["environment"] == e]) > 0 else 0 for e in envs]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=m, color=get_color(m), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("(a) Collision Rate by Environment")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) Mean clearance per environment
    ax = axes[1]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["environment"] == e]["mean_min_clearance"].values[0]
                if len(sub[sub["environment"] == e]) > 0 else 0 for e in envs]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=m, color=get_color(m), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("Mean Min Clearance (m)")
    ax.set_title("(b) Mean Clearance by Environment")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # (c) p5 clearance per environment
    ax = axes[2]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["environment"] == e]["p5_min_clearance"].values[0]
                if len(sub[sub["environment"] == e]) > 0 else 0 for e in envs]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=m, color=get_color(m), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_ylabel("5th Percentile Clearance (m)")
    ax.set_title("(c) Tail Clearance by Environment")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("C3: Collision Clearance Across Environment Types", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_c3_clearance_vs_environment.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")


# ============================================================================
# C4: Multi-Seed Robustness
# ============================================================================

def fig_c4():
    df = load_csv("c4_multi_seed_robustness.csv")
    if df is None:
        return

    methods = df["method"].unique()
    envs = df["environment"].unique()
    seeds = sorted(df["master_seed"].unique())

    # Figure 1: collision rate per seed per environment
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ei, env in enumerate(envs):
        ax = axes[ei // 2, ei % 2]
        env_df = df[df["environment"] == env]

        x = np.arange(len(seeds))
        width = 0.8 / len(methods)
        for mi, m in enumerate(methods):
            sub = env_df[env_df["method"] == m].sort_values("master_seed")
            vals = sub["collision_rate"].values * 100
            lo = sub["coll_ci_lo"].values * 100
            hi = sub["coll_ci_hi"].values * 100
            yerr = [vals - lo, hi - vals]
            offset = (mi - len(methods) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   yerr=yerr, label=m, color=get_color(m), capsize=2,
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_xlabel("Master Seed")
        ax.set_ylabel("Collision Rate (%)")
        ax.set_title(f"{env}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("C4: Multi-Seed Collision Rate Robustness", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_c4_multi_seed_robustness.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")

    # Figure 2: clearance robustness
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for ei, env in enumerate(envs):
        ax = axes[ei // 2, ei % 2]
        env_df = df[df["environment"] == env]

        for m in methods:
            sub = env_df[env_df["method"] == m].sort_values("master_seed")
            seed_labels = sub["master_seed"].astype(str).values
            ax.plot(seed_labels, sub["mean_min_clearance"].values,
                    marker=get_marker(m), color=get_color(m), label=m,
                    linewidth=1.2, markersize=6)
            ax.fill_between(seed_labels,
                            sub["p5_min_clearance"].values,
                            sub["p25_min_clearance"].values,
                            alpha=0.15, color=get_color(m))

        ax.set_xlabel("Master Seed")
        ax.set_ylabel("Clearance (m)")
        ax.set_title(f"{env} (line=mean, band=p5-p25)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("C4: Multi-Seed Clearance Robustness", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_c4_multi_seed_clearance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")


# ============================================================================
# C5: Clearance Under Rare-Mode Stress
# ============================================================================

def fig_c5():
    df = load_csv("c5_clearance_rare_stress.csv")
    if df is None:
        return

    methods = df["method"].unique()
    switch_probs = sorted(df["switch_prob"].unique())

    fig, axes = plt.subplots(len(switch_probs), 2,
                             figsize=(12, 4 * len(switch_probs)))
    if len(switch_probs) == 1:
        axes = axes.reshape(1, -1)

    for si, sp in enumerate(switch_probs):
        sp_df = df[df["switch_prob"] == sp]

        # Collision rate vs rare_prob
        ax = axes[si, 0]
        for m in methods:
            sub = sp_df[sp_df["method"] == m].sort_values("rare_prob")
            ax.plot(sub["rare_prob"], sub["collision_rate"] * 100,
                    marker=get_marker(m), color=get_color(m), label=m,
                    linewidth=1.2, markersize=5)
        ax.set_xlabel("Rare Mode Probability")
        ax.set_ylabel("Collision Rate (%)")
        ax.set_title(f"Collision Rate (switch_prob={sp:.2f})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # p5 clearance vs rare_prob
        ax = axes[si, 1]
        for m in methods:
            sub = sp_df[sp_df["method"] == m].sort_values("rare_prob")
            ax.plot(sub["rare_prob"], sub["p5_min_clearance"],
                    marker=get_marker(m), color=get_color(m), label=m,
                    linewidth=1.2, markersize=5)
        ax.set_xlabel("Rare Mode Probability")
        ax.set_ylabel("5th Percentile Clearance (m)")
        ax.set_title(f"Tail Clearance (switch_prob={sp:.2f})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("C5: Clearance Under Rare-Mode Stress", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_c5_rare_stress.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")


# ============================================================================
# C6: Combined Stress Test
# ============================================================================

def fig_c6():
    df = load_csv("c6_combined_stress.csv")
    if df is None:
        return

    methods = df["method"].unique()
    envs = df["environment"].unique()
    obs_counts = sorted(df["num_obstacles"].unique())
    switch_probs = sorted(df["switch_prob"].unique())

    # Heatmap: collision rate for each method
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for mi, m in enumerate(methods):
        ax = axes[mi]
        m_df = df[df["method"] == m]

        row_labels = []
        matrix = []
        for env in envs:
            for n_obs in obs_counts:
                row_labels.append(f"{env}/{n_obs}obs")
                row = []
                for sp in switch_probs:
                    sub = m_df[(m_df["environment"] == env) &
                               (m_df["num_obstacles"] == n_obs) &
                               (m_df["switch_prob"] == sp)]
                    if len(sub) > 0:
                        row.append(sub["collision_rate"].values[0] * 100)
                    else:
                        row.append(np.nan)
                matrix.append(row)

        matrix = np.array(matrix)
        vmax = max(np.nanmax(matrix), 1)
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(switch_probs)))
        ax.set_xticklabels([f"{sp:.2f}" for sp in switch_probs])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xlabel("Switch Prob")
        ax.set_title(m, fontsize=10)

        for i in range(len(row_labels)):
            for j in range(len(switch_probs)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Collision %")

    fig.suptitle("C6: Combined Stress -- Collision Rate (%)", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_c6_combined_stress_collision.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")

    # Second figure: clearance heatmap
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(5 * len(methods), 5))
    if len(methods) == 1:
        axes = [axes]

    for mi, m in enumerate(methods):
        ax = axes[mi]
        m_df = df[df["method"] == m]

        row_labels = []
        matrix = []
        for env in envs:
            for n_obs in obs_counts:
                row_labels.append(f"{env}/{n_obs}obs")
                row = []
                for sp in switch_probs:
                    sub = m_df[(m_df["environment"] == env) &
                               (m_df["num_obstacles"] == n_obs) &
                               (m_df["switch_prob"] == sp)]
                    if len(sub) > 0:
                        row.append(sub["p5_min_clearance"].values[0])
                    else:
                        row.append(np.nan)
                matrix.append(row)

        matrix = np.array(matrix)
        vmax = max(np.nanmax(matrix), 0.1)
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(switch_probs)))
        ax.set_xticklabels([f"{sp:.2f}" for sp in switch_probs])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_xlabel("Switch Prob")
        ax.set_title(m, fontsize=10)

        for i in range(len(row_labels)):
            for j in range(len(switch_probs)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < vmax * 0.3 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="p5 Clearance (m)")

    fig.suptitle("C6: Combined Stress -- 5th Percentile Clearance (m)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_c6_combined_stress_clearance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Written: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating collision clearance figures...")
    fig_c1()
    fig_c2()
    fig_c3()
    fig_c4()
    fig_c5()
    fig_c6()
    print("Done.")


if __name__ == "__main__":
    main()
