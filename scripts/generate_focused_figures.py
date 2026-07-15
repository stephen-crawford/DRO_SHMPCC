#!/usr/bin/env python3
"""Generate figures from F1–F5 focused experiments.

Fig F1: High-N variant differentiation on Oncoming (horizontal bar chart with CIs)
Fig F2: Speed x Path interaction heatmaps (collision rate + clearance)
Fig F3: K=1 vs K=2 injection scaling with obstacles (grouped bars)
Fig F4: Clearance CDFs and distributions (raw data)
Fig F5: Traj-DRO vs Mode-DRO head-to-head (paired comparison)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

DIR = "focused_figures"
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

COLORS = {
    "Base":             "#999999",
    "WDRO-sampling":    "#FF7F00",
    "WDRO-inject-K1":   "#2166AC",
    "WDRO-inject-K2":   "#4393C3",
    "TopRisk-K1":       "#B2182B",
    "TopRisk-K2":       "#E08080",
    "DiverseRisk-K1":   "#4DAF4A",
    "Softmax-tau5":     "#984EA3",
    "Mode-DRO(inj)":    "#2166AC",
    "Mode-DRO(q*)":     "#FF7F00",
    "Traj-DRO(inj)":    "#E41A1C",
    "Traj-DRO(comb)":   "#4DAF4A",
}

def _c(m): return COLORS.get(m, "#666666")


# ============================================================================
# Fig F1: High-N Variant Differentiation
# ============================================================================
def fig_f1():
    fpath = os.path.join(DIR, "f1_high_n_oncoming.csv")
    if not os.path.exists(fpath):
        print("  Fig F1: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    # Keep specific methods in order
    F1_METHODS = ["Base", "WDRO-sampling", "WDRO-inject-K1", "WDRO-inject-K2",
                  "TopRisk-K1", "TopRisk-K2"]
    df = df[df["method"].isin(F1_METHODS)].reset_index(drop=True)
    # Reorder to match
    df["method"] = pd.Categorical(df["method"], categories=F1_METHODS, ordered=True)
    df = df.sort_values("method").reset_index(drop=True)

    methods = df["method"].values
    cr = df["collision_rate"].values * 100
    cr_lo = df["coll_ci_lo"].values * 100
    cr_hi = df["coll_ci_hi"].values * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Collision rate with CIs
    y = np.arange(len(methods))
    colors = [_c(m) for m in methods]
    xerr = np.array([cr - cr_lo, cr_hi - cr])

    ax1.barh(y, cr, xerr=xerr, color=colors, capsize=4, error_kw={"linewidth": 1.0})
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=9)
    ax1.set_xlabel("Collision Rate (%)")
    ax1.set_title("Collision Rate (N=2000, Oncoming)")
    ax1.invert_yaxis()

    # Add rate text annotations (no CI brackets)
    for i, (m, r, lo, hi) in enumerate(zip(methods, cr, cr_lo, cr_hi)):
        if r < 50:
            ax1.text(hi + 0.5, i, f"{r:.1f}%", va="center", fontsize=7.5)
        else:
            ax1.text(lo - 0.5, i, f"{r:.1f}%", va="center", ha="right", fontsize=7.5, color="white")

    # Panel 2: Clearance with Wilson-style CI whiskers
    # Estimate std from p5 (p5 ≈ mean - 1.645*std for normal), then CI = mean ± z*std/sqrt(n)
    mean_clr = df["mean_clearance"].values
    p5_clr = df["p5_clearance"].values
    n_rollouts = df["n_rollouts"].values.astype(float)
    std_est = np.maximum((mean_clr - p5_clr) / 1.645, 1e-6)
    ci_margin = 1.96 * std_est / np.sqrt(n_rollouts)
    clr_err = np.array([ci_margin, ci_margin])

    x = np.arange(len(methods))
    colors_clr = [_c(m) for m in methods]
    ax2.bar(x, mean_clr, 0.6, yerr=clr_err, color=colors_clr, alpha=0.85,
            capsize=4, error_kw={"linewidth": 1.0}, edgecolor="white", linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Clearance (m)")
    ax2.set_title("Clearance (N=2000, Oncoming)")
    ax2.set_ylim(bottom=0)

    fig.suptitle("High-N Variant Differentiation on Oncoming Environment", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f1_high_n_variants.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F1 saved")


# ============================================================================
# Fig F2: Speed x Path Interaction
# ============================================================================
def fig_f2():
    fpath = os.path.join(DIR, "f2_speed_path_interaction.csv")
    if not os.path.exists(fpath):
        print("  Fig F2: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    speeds = sorted(df["obstacle_speed"].unique())
    paths = df["path"].unique()
    methods = df["method"].unique()

    # Panel layout: 3 speeds x 2 metrics (collision, clearance)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")

    for col, speed in enumerate(speeds):
        sdf = df[df["obstacle_speed"] == speed]
        x = np.arange(len(paths))
        n_m = len(methods)
        width = 0.8 / n_m

        for i, method in enumerate(methods):
            sub = sdf[sdf["method"] == method]
            cr = [sub[sub["path"] == p]["collision_rate"].values[0] * 100
                  if len(sub[sub["path"] == p]) else 0 for p in paths]
            clr = [sub[sub["path"] == p]["mean_clearance"].values[0]
                   if len(sub[sub["path"] == p]) else 0 for p in paths]

            offset = (i - n_m / 2 + 0.5) * width
            axes[0, col].bar(x + offset, cr, width * 0.9, label=method if col == 0 else "",
                            color=_c(method))
            axes[1, col].bar(x + offset, clr, width * 0.9, color=_c(method))

        axes[0, col].set_title(f"Speed = {speed} m/s")
        axes[1, col].set_xticks(x)
        axes[1, col].set_xticklabels(paths, rotation=20, ha="right")

    axes[0, 0].set_ylabel("Collision Rate (%)")
    axes[1, 0].set_ylabel("Mean Clearance (m)")
    axes[0, 0].legend(fontsize=7, loc="upper left")

    fig.suptitle("F2: Speed x Path Interaction (N=1000)", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f2_speed_path.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F2 saved")

    # Also make a DRO benefit heatmap
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, metric, title, fmt in [
        (ax1, "collision_rate", "Collision Rate Reduction (%)\nWDRO-inject vs Base", ".1f"),
        (ax2, "mean_clearance", "Clearance Improvement (m)\nWDRO-inject vs Base", ".2f"),
    ]:
        matrix = np.zeros((len(paths), len(speeds)))
        for i, p in enumerate(paths):
            for j, s in enumerate(speeds):
                base = df[(df["path"] == p) & (df["obstacle_speed"] == s) & (df["method"] == "Base")]
                wdro = df[(df["path"] == p) & (df["obstacle_speed"] == s) & (df["method"] == "WDRO-inject-K1")]
                if not base.empty and not wdro.empty:
                    b = base.iloc[0][metric]
                    w = wdro.iloc[0][metric]
                    if "collision" in metric:
                        matrix[i, j] = (b - w) * 100
                    else:
                        matrix[i, j] = w - b

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([f"{s}" for s in speeds])
        ax.set_xlabel("Obstacle Speed (m/s)")
        ax.set_yticks(range(len(paths)))
        ax.set_yticklabels(paths)
        ax.set_title(title, fontsize=9)
        for i_r in range(len(paths)):
            for j_c in range(len(speeds)):
                val = matrix[i_r, j_c]
                ax.text(j_c, i_r, f"{val:{fmt}}", ha="center", va="center", fontsize=10,
                        color="white" if abs(val) > 10 else "black")
        fig2.colorbar(im, ax=ax, shrink=0.8)

    fig2.suptitle("F2: DRO Benefit Heatmap (Speed x Path)", fontsize=12, y=1.02)
    fig2.tight_layout()
    for ext in ["pdf", "png"]:
        fig2.savefig(os.path.join(DIR, f"fig_f2_benefit_heatmap.{ext}"), bbox_inches="tight")
    plt.close(fig2)
    print("  Fig F2 heatmap saved")


# ============================================================================
# Fig F3: K=1 vs K=2 Multi-Obstacle
# ============================================================================
def fig_f3():
    fpath = os.path.join(DIR, "f3_k1_vs_k2_multi_obs.csv")
    if not os.path.exists(fpath):
        print("  Fig F3: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    obs_counts = sorted(df["num_obstacles"].unique())
    methods = df["method"].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(obs_counts))
    n_m = len(methods)
    width = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr = [sub[sub["num_obstacles"] == n]["collision_rate"].values[0] * 100
              if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr_lo = [sub[sub["num_obstacles"] == n]["coll_ci_lo"].values[0] * 100
                 if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr_hi = [sub[sub["num_obstacles"] == n]["coll_ci_hi"].values[0] * 100
                 if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        clr = [sub[sub["num_obstacles"] == n]["mean_clearance"].values[0]
               if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]

        offset = (i - n_m / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.9, yerr=err, label=method,
                color=_c(method), capsize=3, error_kw={"linewidth": 0.8})
        ax2.bar(x + offset, clr, width * 0.9, color=_c(method), label=method)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in obs_counts])
    ax1.set_xlabel("Number of Obstacles")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate: K=1 vs K=2")
    ax1.legend(fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in obs_counts])
    ax2.set_xlabel("Number of Obstacles")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance: K=1 vs K=2")
    ax2.legend(fontsize=7)

    fig.suptitle("F3: K=1 vs K=2 Injection Count (N=1000)", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f3_k1_vs_k2.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F3 saved")


# ============================================================================
# Fig F4: Clearance CDFs and Distributions
# ============================================================================
def fig_f4():
    raw_fpath = os.path.join(DIR, "f4_clearance_raw.csv")
    sum_fpath = os.path.join(DIR, "f4_clearance_summary.csv")

    if not os.path.exists(raw_fpath):
        print("  Fig F4: no raw data found, skipping")
        return

    df = pd.read_csv(raw_fpath)

    envs = df["environment"].unique()
    methods_all = ["Base", "WDRO-inject-K1", "TopRisk-K1"]

    # CDFs
    fig, axes = plt.subplots(1, len(envs), figsize=(12, 5))
    if len(envs) == 1:
        axes = [axes]

    for ax_i, env in enumerate(envs):
        ax = axes[ax_i]
        edf = df[df["environment"] == env]

        for method in methods_all:
            sub = edf[edf["method"] == method]
            if sub.empty:
                continue
            clr = np.sort(sub["min_clearance"].values)
            cdf = np.arange(1, len(clr) + 1) / len(clr)
            ax.plot(clr, cdf, label=method, color=_c(method), linewidth=1.5)

        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Safety r")
        ax.set_xlabel("Min Clearance (m)")
        ax.set_ylabel("CDF" if ax_i == 0 else "")
        ax.set_title(f"{env}: Clearance CDF (N=2000)")
        ax.legend(fontsize=7)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("F4: Clearance Distribution (N=2000)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f4_clearance_cdf.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F4 CDF saved")

    # Conditional clearance: collision vs non-collision
    fig2, axes2 = plt.subplots(1, len(envs), figsize=(12, 5))
    if len(envs) == 1:
        axes2 = [axes2]

    for ax_i, env in enumerate(envs):
        ax = axes2[ax_i]
        edf = df[df["environment"] == env]
        methods = [m for m in methods_all if m in edf["method"].values]
        x = np.arange(len(methods))

        coll_means = []
        nocoll_means = []
        for method in methods:
            sub = edf[edf["method"] == method]
            coll_clr = sub[sub["collision"] == 1]["min_clearance"].values
            nocoll_clr = sub[sub["collision"] == 0]["min_clearance"].values
            coll_means.append(np.mean(coll_clr) if len(coll_clr) > 0 else 0)
            nocoll_means.append(np.mean(nocoll_clr) if len(nocoll_clr) > 0 else 0)

        width = 0.35
        ax.bar(x - width/2, nocoll_means, width, label="No collision", color="#4DAF4A", alpha=0.85)
        ax.bar(x + width/2, coll_means, width, label="Collision", color="#E41A1C", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Mean Clearance (m)" if ax_i == 0 else "")
        ax.set_title(f"{env}: Conditional Clearance")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)

    fig2.suptitle("F4: Conditional Clearance (N=2000)", fontsize=12, y=1.02)
    fig2.tight_layout()
    for ext in ["pdf", "png"]:
        fig2.savefig(os.path.join(DIR, f"fig_f4_conditional_clearance.{ext}"), bbox_inches="tight")
    plt.close(fig2)
    print("  Fig F4 conditional saved")

    # Violin/box plots
    fig3, axes3 = plt.subplots(1, len(envs), figsize=(12, 5))
    if len(envs) == 1:
        axes3 = [axes3]

    for ax_i, env in enumerate(envs):
        ax = axes3[ax_i]
        edf = df[df["environment"] == env]
        methods = [m for m in methods_all if m in edf["method"].values]

        data = [edf[edf["method"] == m]["min_clearance"].values for m in methods]
        bp = ax.boxplot(data, labels=methods, patch_artist=True, widths=0.6)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(_c(method))
            patch.set_alpha(0.7)

        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("Min Clearance (m)" if ax_i == 0 else "")
        ax.set_title(f"{env}: Clearance Distribution")
        ax.grid(True, alpha=0.2, axis="y")

    fig3.suptitle("F4: Clearance Box Plots (N=2000)", fontsize=12, y=1.02)
    fig3.tight_layout()
    for ext in ["pdf", "png"]:
        fig3.savefig(os.path.join(DIR, f"fig_f4_clearance_boxplot.{ext}"), bbox_inches="tight")
    plt.close(fig3)
    print("  Fig F4 boxplot saved")


# ============================================================================
# Fig F5: Mode-DRO Speed Comparison
# ============================================================================
def fig_f5():
    fpath = os.path.join(DIR, "f5_traj_vs_mode_dro.csv")
    if not os.path.exists(fpath):
        print("  Fig F5: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    # Drop trajectory-based methods
    traj_methods = {m for m in df["method"].unique() if "Traj" in m}
    df = df[~df["method"].isin(traj_methods)]

    speeds = sorted(df["obstacle_speed"].unique())
    methods = df["method"].unique()

    fig, axes = plt.subplots(1, len(speeds), figsize=(5.5 * len(speeds), 5), sharey=True)
    if len(speeds) == 1:
        axes = [axes]

    for ax_i, speed in enumerate(speeds):
        ax = axes[ax_i]
        sdf = df[df["obstacle_speed"] == speed]

        y = np.arange(len(sdf))
        cr = sdf["collision_rate"].values * 100
        cr_lo = sdf["coll_ci_lo"].values * 100
        cr_hi = sdf["coll_ci_hi"].values * 100
        xerr = np.array([cr - cr_lo, cr_hi - cr])
        colors = [_c(m) for m in sdf["method"].values]

        ax.barh(y, cr, xerr=xerr, color=colors, capsize=3, error_kw={"linewidth": 0.8})
        ax.set_yticks(y)
        ax.set_yticklabels(sdf["method"].values, fontsize=9)
        ax.set_xlabel("Collision Rate (%)")
        ax.set_title(f"Speed = {speed} m/s")
        ax.invert_yaxis()

        # Add rate text (no brackets)
        for i, (r, lo, hi) in enumerate(zip(cr, cr_lo, cr_hi)):
            ax.text(hi + 0.3, i, f"{r:.1f}%", va="center", fontsize=7)

    fig.suptitle("Mode-DRO Speed Comparison (N=2000, S-curve)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f5_traj_vs_mode.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F5 saved")


# ============================================================================
# Fig F6: Traj-DRO Transition Speed
# ============================================================================
def fig_f6():
    fpath = os.path.join(DIR, "f6_transition_speed.csv")
    if not os.path.exists(fpath):
        print("  Fig F6: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    methods = ["Base", "Mode-DRO(inj)", "Traj-DRO(comb)"]
    markers = {"Base": "s", "Mode-DRO(inj)": "o", "Traj-DRO(comb)": "D"}

    for method in methods:
        sub = df[df["method"] == method].sort_values("obstacle_speed")
        sp = sub["obstacle_speed"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        clr = sub["mean_clearance"].values

        style = dict(color=_c(method), marker=markers.get(method, "o"),
                     markersize=7, linewidth=2)
        ax1.plot(sp, cr, label=method, **style)
        ax1.fill_between(sp, clo, chi, alpha=0.15, color=_c(method))
        ax2.plot(sp, clr, label=method, **style)

    # Mark crossover zone
    ax1.axvspan(1.05, 1.45, alpha=0.08, color="green", label="Traj-DRO advantage")
    ax1.axvline(x=1.1, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax1.set_xlabel("Obstacle Speed (m/s)")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate: Mode-DRO vs Traj-DRO")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.set_xlabel("Obstacle Speed (m/s)")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance: Mode-DRO vs Traj-DRO")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("F6: Traj-DRO Transition Speed (N=1500, S-curve)", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f6_transition_speed.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F6 saved")


# ============================================================================
# Fig F7: Traj-DRO on Tight-S (speed sweep)
# ============================================================================
def fig_f7():
    fpath = os.path.join(DIR, "f7_traj_dro_tight_s.csv")
    if not os.path.exists(fpath):
        print("  Fig F7: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    methods = df["method"].unique()
    speeds = sorted(df["obstacle_speed"].unique())
    markers = {"Base": "s", "Mode-DRO(inj)": "o", "Mode-DRO(q*)": "D", "Traj-DRO(comb)": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for method in methods:
        sub = df[df["method"] == method].sort_values("obstacle_speed")
        sp = sub["obstacle_speed"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        clr = sub["mean_clearance"].values

        style = dict(color=_c(method), marker=markers.get(method, "o"),
                     markersize=7, linewidth=2)
        ax1.plot(sp, cr, label=method, **style)
        ax1.fill_between(sp, clo, chi, alpha=0.15, color=_c(method))
        ax2.plot(sp, clr, label=method, **style)

    ax1.set_xlabel("Obstacle Speed (m/s)")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate on Tight-S Path")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.set_xlabel("Obstacle Speed (m/s)")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance on Tight-S Path")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("F7: Traj-DRO on Tight-S at Medium Speeds (N=1500)", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f7_traj_dro_tight_s.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F7 saved")

    # Also generate grouped bar chart per speed
    fig2, axes = plt.subplots(1, len(speeds), figsize=(5 * len(speeds), 5))
    if len(speeds) == 1:
        axes = [axes]

    for ax_i, speed in enumerate(speeds):
        ax = axes[ax_i]
        sdf = df[df["obstacle_speed"] == speed]

        y = np.arange(len(sdf))
        cr = sdf["collision_rate"].values * 100
        cr_lo = sdf["coll_ci_lo"].values * 100
        cr_hi = sdf["coll_ci_hi"].values * 100
        xerr = np.array([cr - cr_lo, cr_hi - cr])
        colors = [_c(m) for m in sdf["method"].values]

        ax.barh(y, cr, xerr=xerr, color=colors, capsize=3, error_kw={"linewidth": 0.8})
        ax.set_yticks(y)
        ax.set_yticklabels(sdf["method"].values, fontsize=9)
        ax.set_xlabel("Collision Rate (%)")
        ax.set_title(f"v = {speed} m/s")
        ax.invert_yaxis()

        for i, (r, lo, hi) in enumerate(zip(cr, cr_lo, cr_hi)):
            ax.text(hi + 0.3, i, f"{r:.1f}% [{lo:.1f}, {hi:.1f}]", va="center", fontsize=7)

    fig2.suptitle("F7: Tight-S per Speed (N=1500)", fontsize=12, y=1.02)
    fig2.tight_layout()
    for ext in ["pdf", "png"]:
        fig2.savefig(os.path.join(DIR, f"fig_f7_per_speed.{ext}"), bbox_inches="tight")
    plt.close(fig2)
    print("  Fig F7 per-speed bars saved")


# ============================================================================
# Fig F8: Best DRO Showcase
# ============================================================================
def fig_f8():
    fpath = os.path.join(DIR, "f8_best_dro_showcase.csv")
    if not os.path.exists(fpath):
        print("  Fig F8: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    scenarios = df["scenario"].unique()
    methods = df["method"].unique()
    n_m = len(methods)

    # Panel 1: Grouped bars by scenario
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    x = np.arange(len(scenarios))
    width = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr = [sub[sub["scenario"] == s]["collision_rate"].values[0] * 100
              if len(sub[sub["scenario"] == s]) else 0 for s in scenarios]
        cr_lo = [sub[sub["scenario"] == s]["coll_ci_lo"].values[0] * 100
                 if len(sub[sub["scenario"] == s]) else 0 for s in scenarios]
        cr_hi = [sub[sub["scenario"] == s]["coll_ci_hi"].values[0] * 100
                 if len(sub[sub["scenario"] == s]) else 0 for s in scenarios]
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        clr = [sub[sub["scenario"] == s]["mean_clearance"].values[0]
               if len(sub[sub["scenario"] == s]) else 0 for s in scenarios]

        offset = (i - n_m / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.9, yerr=err, label=method,
                color=_c(method), capsize=3, error_kw={"linewidth": 0.8})
        ax2.bar(x + offset, clr, width * 0.9, color=_c(method), label=method)

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate by Best Scenario (N=2000)")
    ax1.legend(fontsize=7.5, ncol=4, loc="upper center")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance by Best Scenario (N=2000)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=15, ha="right")
    ax2.legend(fontsize=7.5, ncol=4, loc="upper right")

    fig.suptitle("F8: Best DRO Showcase (N=2000)", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f8_best_showcase.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F8 grouped bars saved")

    # DRO benefit table: compute reduction vs Base
    fig3, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    rows = []
    for sc in scenarios:
        base = df[(df["scenario"] == sc) & (df["method"] == "Base")]
        if base.empty:
            continue
        base_cr = base.iloc[0]["collision_rate"] * 100
        base_clr = base.iloc[0]["mean_clearance"]

        row = [sc, f"{base_cr:.1f}%"]
        for method in [m for m in methods if m != "Base"]:
            sub = df[(df["scenario"] == sc) & (df["method"] == method)]
            if sub.empty:
                row.extend(["—", "—"])
                continue
            m_cr = sub.iloc[0]["collision_rate"] * 100
            m_clr = sub.iloc[0]["mean_clearance"]
            cr_red = base_cr - m_cr
            clr_imp = m_clr - base_clr
            row.append(f"{m_cr:.1f}% ({cr_red:+.1f}pp)")
            row.append(f"{m_clr:.2f} ({clr_imp:+.2f})")
        rows.append(row)

    dro_methods = [m for m in methods if m != "Base"]
    col_labels = ["Scenario", "Base CR"]
    for m in dro_methods:
        col_labels.extend([f"{m} CR", f"{m} Clr"])

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color cells by improvement magnitude
    for i, row in enumerate(rows):
        for j in range(2, len(row)):
            cell = table[i + 1, j]
            if "+" in row[j] and "pp" in row[j]:
                val = float(row[j].split("(")[1].split("pp")[0])
                if val > 0:
                    cell.set_facecolor("#d4edda")  # green
                elif val < -2:
                    cell.set_facecolor("#f8d7da")  # red

    fig3.suptitle("F8: DRO Benefit Summary Table", fontsize=12)
    fig3.tight_layout()
    for ext in ["pdf", "png"]:
        fig3.savefig(os.path.join(DIR, f"fig_f8_benefit_table.{ext}"), bbox_inches="tight")
    plt.close(fig3)
    print("  Fig F8 benefit table saved")

    # Clearance CDFs from raw data
    raw_fpath = os.path.join(DIR, "f8_showcase_clearance_raw.csv")
    if not os.path.exists(raw_fpath):
        print("  Fig F8: no raw clearance data, skipping CDF")
        return

    rdf = pd.read_csv(raw_fpath)
    scenarios_raw = rdf["scenario"].unique()

    fig4, axes = plt.subplots(1, len(scenarios_raw), figsize=(5 * len(scenarios_raw), 5))
    if len(scenarios_raw) == 1:
        axes = [axes]

    for ax_i, sc in enumerate(scenarios_raw):
        ax = axes[ax_i]
        sdf = rdf[rdf["scenario"] == sc]

        for method in methods:
            sub = sdf[sdf["method"] == method]
            if sub.empty:
                continue
            clr = np.sort(sub["min_clearance"].values)
            cdf = np.arange(1, len(clr) + 1) / len(clr)
            ax.plot(clr, cdf, label=method, color=_c(method), linewidth=1.5)

        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Safety r")
        ax.set_xlabel("Min Clearance (m)")
        ax.set_ylabel("CDF" if ax_i == 0 else "")
        ax.set_title(f"{sc}")
        ax.legend(fontsize=6.5)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig4.suptitle("F8: Clearance CDFs (N=2000)", fontsize=12, y=1.02)
    fig4.tight_layout()
    for ext in ["pdf", "png"]:
        fig4.savefig(os.path.join(DIR, f"fig_f8_clearance_cdfs.{ext}"), bbox_inches="tight")
    plt.close(fig4)
    print("  Fig F8 CDFs saved")


# ============================================================================
# Fig F9: Multi-Obstacle at Sweet Spot
# ============================================================================
def fig_f9():
    fpath = os.path.join(DIR, "f9_multi_obs_sweet_spot.csv")
    if not os.path.exists(fpath):
        print("  Fig F9: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    obs_counts = sorted(df["num_obstacles"].unique())
    methods = df["method"].unique()
    n_m = len(methods)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(obs_counts))
    width = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr = [sub[sub["num_obstacles"] == n]["collision_rate"].values[0] * 100
              if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr_lo = [sub[sub["num_obstacles"] == n]["coll_ci_lo"].values[0] * 100
                 if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr_hi = [sub[sub["num_obstacles"] == n]["coll_ci_hi"].values[0] * 100
                 if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        clr = [sub[sub["num_obstacles"] == n]["mean_clearance"].values[0]
               if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        clr_p5 = [sub[sub["num_obstacles"] == n]["p5_clearance"].values[0]
                  if len(sub[sub["num_obstacles"] == n]) else 0 for n in obs_counts]
        clr_n = [float(sub[sub["num_obstacles"] == n]["n_rollouts"].values[0])
                 if len(sub[sub["num_obstacles"] == n]) else 1 for n in obs_counts]
        clr = np.array(clr)
        clr_p5 = np.array(clr_p5)
        clr_n = np.array(clr_n)
        std_est = np.maximum((clr - clr_p5) / 1.645, 1e-6)
        ci_margin = 1.96 * std_est / np.sqrt(clr_n)
        clr_err = np.array([ci_margin, ci_margin])

        offset = (i - n_m / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.9, yerr=err, label=method,
                color=_c(method), capsize=3, error_kw={"linewidth": 0.8})
        ax2.bar(x + offset, clr, width * 0.9, yerr=clr_err, color=_c(method),
                label=method, capsize=3, error_kw={"linewidth": 0.8})

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in obs_counts])
    ax1.set_xlabel("Number of Obstacles")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate: Multi-Obstacle Scaling")
    ax1.legend(fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in obs_counts])
    ax2.set_xlabel("Number of Obstacles")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance: Multi-Obstacle Scaling")
    ax2.legend(fontsize=7.5)

    fig.suptitle("Multi-Obstacle Scaling at Sweet Spot (Tight-S, v=1.0, N=1500)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f9_multi_obs_sweet_spot.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F9 saved")

    # DRO benefit line plot
    fig2, ax = plt.subplots(figsize=(8, 5))
    for method in [m for m in methods if m != "Base"]:
        sub = df[df["method"] == method].sort_values("num_obstacles")
        base_sub = df[df["method"] == "Base"].sort_values("num_obstacles")

        nobs = sub["num_obstacles"].values
        benefit = (base_sub["collision_rate"].values - sub["collision_rate"].values) * 100

        ax.plot(nobs, benefit, label=method, color=_c(method), marker="o",
                markersize=7, linewidth=2)

    ax.set_xlabel("Number of Obstacles")
    ax.set_ylabel("DRO Benefit (pp reduction)")
    ax.set_title("DRO Collision Rate Reduction vs # Obstacles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(obs_counts)

    fig2.suptitle("DRO Benefit Scaling (Tight-S, v=1.0)", fontsize=12, y=1.01)
    fig2.tight_layout()
    for ext in ["pdf", "png"]:
        fig2.savefig(os.path.join(DIR, f"fig_f9_benefit_scaling.{ext}"), bbox_inches="tight")
    plt.close(fig2)
    print("  Fig F9 benefit scaling saved")


# ============================================================================
# Fig F10: Tight-S Speed Sweep with DRO Crossover
# ============================================================================
def fig_f10():
    fpath = os.path.join(DIR, "f10_tight_s_speed_sweep.csv")
    if not os.path.exists(fpath):
        print("  Fig F10: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    methods = df["method"].unique()
    markers = {"Base": "s", "WDRO-inject-K1": "o", "WDRO-sampling": "D", "TopRisk-K1": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for method in methods:
        sub = df[df["method"] == method].sort_values("obstacle_speed")
        sp = sub["obstacle_speed"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        clr = sub["mean_clearance"].values

        style = dict(color=_c(method), marker=markers.get(method, "o"),
                     markersize=7, linewidth=2)
        ax1.plot(sp, cr, label=method, **style)
        ax1.fill_between(sp, clo, chi, alpha=0.15, color=_c(method))
        ax2.plot(sp, clr, label=method, **style)

    # Mark crossover zone if visible
    ax1.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax1.set_xlabel("Obstacle Speed (m/s)")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs Speed (Tight-S)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.set_xlabel("Obstacle Speed (m/s)")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance vs Speed (Tight-S)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("F10: Tight-S Speed Sweep (N=1500)", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f10_tight_s_speed_sweep.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F10 saved")

    # Also make a DRO benefit plot (difference from Base)
    fig2, ax = plt.subplots(figsize=(10, 5.5))
    base = df[df["method"] == "Base"].sort_values("obstacle_speed")
    base_cr = base["collision_rate"].values * 100
    base_sp = base["obstacle_speed"].values

    for method in [m for m in methods if m != "Base"]:
        sub = df[df["method"] == method].sort_values("obstacle_speed")
        sp = sub["obstacle_speed"].values
        cr = sub["collision_rate"].values * 100
        benefit = base_cr - cr

        style = dict(color=_c(method), marker=markers.get(method, "o"),
                     markersize=7, linewidth=2)
        ax.plot(sp, benefit, label=method, **style)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="DRO neutral")
    ax.fill_between(base_sp, 0, -20, alpha=0.05, color="red")
    ax.set_xlabel("Obstacle Speed (m/s)")
    ax.set_ylabel("DRO Benefit (pp reduction)")
    ax.set_title("DRO Collision Rate Reduction on Tight-S")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig2.suptitle("F10: DRO Benefit vs Speed (Tight-S, N=1500)", fontsize=12, y=1.01)
    fig2.tight_layout()
    for ext in ["pdf", "png"]:
        fig2.savefig(os.path.join(DIR, f"fig_f10_benefit_vs_speed.{ext}"), bbox_inches="tight")
    plt.close(fig2)
    print("  Fig F10 benefit curve saved")


# ============================================================================
# Fig F11: Path Geometry Effect at High Speeds
# ============================================================================
def fig_f11():
    fpath = os.path.join(DIR, "f11_path_geometry_high_speed.csv")
    if not os.path.exists(fpath):
        print("  Fig F11: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    paths = df["path"].unique()
    speeds = sorted(df["obstacle_speed"].unique())

    # DRO benefit (inject-K1 reduction vs Base) as heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title, fmt, cmap in [
        (ax1, "collision_rate", "DRO Collision Rate Reduction (pp)\nWDRO-inject vs Base", ".1f", "RdYlGn"),
        (ax2, "mean_clearance", "DRO Clearance Change (m)\nWDRO-inject vs Base", ".2f", "RdYlGn"),
    ]:
        matrix = np.zeros((len(paths), len(speeds)))
        for i, p in enumerate(paths):
            for j, s in enumerate(speeds):
                base = df[(df["path"] == p) & (df["obstacle_speed"] == s) & (df["method"] == "Base")]
                wdro = df[(df["path"] == p) & (df["obstacle_speed"] == s) & (df["method"] == "WDRO-inject-K1")]
                if not base.empty and not wdro.empty:
                    b = base.iloc[0][metric]
                    w = wdro.iloc[0][metric]
                    if "collision" in metric:
                        matrix[i, j] = (b - w) * 100  # positive = DRO better
                    else:
                        matrix[i, j] = w - b  # positive = DRO more clearance

        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(speeds)))
        ax.set_xticklabels([f"{s}" for s in speeds])
        ax.set_xlabel("Obstacle Speed (m/s)")
        ax.set_yticks(range(len(paths)))
        ax.set_yticklabels(paths)
        ax.set_title(title, fontsize=9)
        for i_r in range(len(paths)):
            for j_c in range(len(speeds)):
                val = matrix[i_r, j_c]
                txt_color = "white" if abs(val) > max(abs(matrix.max()), abs(matrix.min())) * 0.5 else "black"
                ax.text(j_c, i_r, f"{val:{fmt}}", ha="center", va="center", fontsize=10,
                        color=txt_color)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("F11: Path Geometry Effect on DRO at High Speeds (N=1500)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_f11_path_geometry_heatmap.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig F11 heatmap saved")

    # Line plots: collision rate by speed, one panel per path
    fig2, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5), sharey=True)
    if len(paths) == 1:
        axes = [axes]

    markers = {"Base": "s", "WDRO-inject-K1": "o", "TopRisk-K1": "^"}
    for ax_i, path in enumerate(paths):
        ax = axes[ax_i]
        pdf = df[df["path"] == path]
        for method in pdf["method"].unique():
            sub = pdf[pdf["method"] == method].sort_values("obstacle_speed")
            sp = sub["obstacle_speed"].values
            cr = sub["collision_rate"].values * 100
            clo = sub["coll_ci_lo"].values * 100
            chi = sub["coll_ci_hi"].values * 100
            style = dict(color=_c(method), marker=markers.get(method, "o"),
                         markersize=7, linewidth=2)
            ax.plot(sp, cr, label=method, **style)
            ax.fill_between(sp, clo, chi, alpha=0.15, color=_c(method))

        ax.set_xlabel("Obstacle Speed (m/s)")
        ax.set_ylabel("Collision Rate (%)" if ax_i == 0 else "")
        ax.set_title(f"{path}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig2.suptitle("F11: Collision Rate by Path at High Speeds (N=1500)", fontsize=12, y=1.02)
    fig2.tight_layout()
    for ext in ["pdf", "png"]:
        fig2.savefig(os.path.join(DIR, f"fig_f11_path_comparison.{ext}"), bbox_inches="tight")
    plt.close(fig2)
    print("  Fig F11 path comparison saved")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating focused experiment figures...")
    fig_f1()
    fig_f2()
    fig_f3()
    fig_f4()
    fig_f5()
    fig_f6()
    fig_f7()
    fig_f8()
    fig_f9()
    fig_f10()
    fig_f11()
    print("Done.")
