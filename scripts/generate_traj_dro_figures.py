#!/usr/bin/env python3
"""Generate figures from T1–T5 trajectory DRO generalization experiments.

Fig T1: Path geometry — mode DRO vs trajectory DRO (grouped bars)
Fig T2: Multi-obstacle scaling (grouped bars)
Fig T3: Switching dynamics sweep (line plots)
Fig T4: Wasserstein radius sweep (line plots)
Fig T5: Path × obstacle heatmaps
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

DIR = "traj_dro_figures"
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
    "Mode-DRO(q*)":    "#FF7F00",
    "Mode-DRO(inj)":   "#2166AC",
    "Traj-DRO(q*)":    "#B2182B",
    "Traj-DRO(inj)":   "#4DAF4A",
    "Traj-DRO(comb)":  "#984EA3",
}
MARKERS = {
    "Base":             "s",
    "Mode-DRO(q*)":    "D",
    "Mode-DRO(inj)":   "o",
    "Traj-DRO(q*)":    "^",
    "Traj-DRO(inj)":   "v",
    "Traj-DRO(comb)":  "P",
}

def _c(m): return COLORS.get(m, "#666")
def _m(m): return MARKERS.get(m, "o")


def fig_t1():
    fpath = os.path.join(DIR, "t1_path_geometry.csv")
    if not os.path.exists(fpath):
        print("  Fig T1: no data, skipping"); return

    df = pd.read_csv(fpath)
    paths = df["path"].unique()
    methods = df["method"].unique()
    n_m = len(methods)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    x = np.arange(len(paths))
    w = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr = np.array([sub[sub["path"]==p]["collision_rate"].values[0]*100 if len(sub[sub["path"]==p]) else 0 for p in paths])
        cr_lo = np.array([sub[sub["path"]==p]["coll_ci_lo"].values[0]*100 if len(sub[sub["path"]==p]) else 0 for p in paths])
        cr_hi = np.array([sub[sub["path"]==p]["coll_ci_hi"].values[0]*100 if len(sub[sub["path"]==p]) else 0 for p in paths])
        err = np.array([cr - cr_lo, cr_hi - cr])

        mm = np.array([sub[sub["path"]==p]["missed_mode_rate"].values[0]*100 if len(sub[sub["path"]==p]) else 0 for p in paths])

        off = (i - n_m/2 + 0.5) * w
        ax1.bar(x+off, cr, w*0.9, yerr=err, label=method, color=_c(method),
                capsize=2, error_kw={"linewidth": 0.7})
        ax2.bar(x+off, mm, w*0.9, color=_c(method))

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate by Path Geometry")
    ax2.set_ylabel("Missed Mode Rate (%)")
    ax2.set_title("Missed Mode Rate by Path Geometry")
    ax2.set_xticks(x); ax2.set_xticklabels(paths)
    ax1.legend(loc="upper right", fontsize=7, ncol=2)

    fig.suptitle("T1: Mode DRO vs Trajectory DRO — Path Geometry", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_t1_path_geometry.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig T1 saved")


def fig_t2():
    fpath = os.path.join(DIR, "t2_multi_obstacle.csv")
    if not os.path.exists(fpath):
        print("  Fig T2: no data, skipping"); return

    df = pd.read_csv(fpath)
    # Remove trajectory-level DRO strategies
    df = df[~df["method"].str.startswith("Traj-")]
    obs_counts = sorted(df["num_obstacles"].unique())
    methods = df["method"].unique()
    n_m = len(methods)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    x = np.arange(len(obs_counts))
    w = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr = np.array([sub[sub["num_obstacles"]==n]["collision_rate"].values[0]*100 if len(sub[sub["num_obstacles"]==n]) else 0 for n in obs_counts])
        cr_lo = np.array([sub[sub["num_obstacles"]==n]["coll_ci_lo"].values[0]*100 if len(sub[sub["num_obstacles"]==n]) else 0 for n in obs_counts])
        cr_hi = np.array([sub[sub["num_obstacles"]==n]["coll_ci_hi"].values[0]*100 if len(sub[sub["num_obstacles"]==n]) else 0 for n in obs_counts])
        err = np.array([cr - cr_lo, cr_hi - cr])
        mm = np.array([sub[sub["num_obstacles"]==n]["missed_mode_rate"].values[0]*100 if len(sub[sub["num_obstacles"]==n]) else 0 for n in obs_counts])
        mm_lo = np.array([sub[sub["num_obstacles"]==n]["mm_ci_lo"].values[0]*100 if len(sub[sub["num_obstacles"]==n]) else 0 for n in obs_counts])
        mm_hi = np.array([sub[sub["num_obstacles"]==n]["mm_ci_hi"].values[0]*100 if len(sub[sub["num_obstacles"]==n]) else 0 for n in obs_counts])
        mm_err = np.array([mm - mm_lo, mm_hi - mm])

        off = (i - n_m/2 + 0.5) * w
        ax1.bar(x+off, cr, w*0.9, yerr=err, label=method, color=_c(method),
                capsize=2, error_kw={"linewidth": 0.7})
        ax2.bar(x+off, mm, w*0.9, yerr=mm_err, color=_c(method),
                capsize=2, error_kw={"linewidth": 0.7})

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs Obstacle Count")
    ax2.set_ylabel("Missed Mode Rate (%)")
    ax2.set_title("Missed Mode Rate vs Obstacle Count")
    ax2.set_xticks(x); ax2.set_xticklabels([str(n) for n in obs_counts])
    ax2.set_xlabel("Number of Obstacles")
    ax1.legend(loc="upper left", fontsize=7, ncol=2)

    fig.suptitle("T2: Multi-Obstacle Scaling — Mode DRO", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_t2_multi_obstacle.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig T2 saved")


def fig_t3():
    fpath = os.path.join(DIR, "t3_switch_dynamics.csv")
    if not os.path.exists(fpath):
        print("  Fig T3: no data, skipping"); return

    df = pd.read_csv(fpath)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("switch_prob")
        sp = sub["switch_prob"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        mm = sub["missed_mode_rate"].values * 100

        style = dict(color=_c(method), marker=_m(method), markersize=5, linewidth=1.3)
        ax1.plot(sp, cr, label=method, **style)
        ax1.fill_between(sp, clo, chi, alpha=0.1, color=_c(method))
        ax2.plot(sp, mm, label=method, **style)

    ax1.set_xlabel("Switch Probability"); ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Switch Probability"); ax2.set_ylabel("Missed Mode Rate (%)")
    ax2.set_title("Missed Mode Rate")

    fig.suptitle("T3: Switching Dynamics — Mode DRO vs Traj DRO", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_t3_switch_dynamics.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig T3 saved")


def fig_t4():
    fpath = os.path.join(DIR, "t4_rho_sweep.csv")
    if not os.path.exists(fpath):
        print("  Fig T4: no data, skipping"); return

    df = pd.read_csv(fpath)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("rho")
        rho = sub["rho"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        mm = sub["missed_mode_rate"].values * 100

        style = dict(color=_c(method), marker=_m(method), markersize=5, linewidth=1.3)
        ax1.plot(rho, cr, label=method, **style)
        ax1.fill_between(rho, clo, chi, alpha=0.1, color=_c(method))
        ax2.plot(rho, mm, label=method, **style)

    ax1.set_xlabel("Wasserstein Radius (rho)"); ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs rho"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Wasserstein Radius (rho)"); ax2.set_ylabel("Missed Mode Rate (%)")
    ax2.set_title("Missed Mode Rate vs rho")

    fig.suptitle("T4: Wasserstein Radius Sweep", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_t4_rho_sweep.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig T4 saved")


def fig_t5():
    fpath = os.path.join(DIR, "t5_path_obstacle_grid.csv")
    if not os.path.exists(fpath):
        print("  Fig T5: no data, skipping"); return

    df = pd.read_csv(fpath)
    paths = df["path"].unique()
    obs_counts = sorted(df["num_obstacles"].unique())

    # Compare Mode-DRO(inj) vs Traj-DRO(inj) improvement over Base
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, compare_method, title in [
        (axes[0], "Mode-DRO(inj)", "Mode-DRO(inj) Coll. Reduction (%)"),
        (axes[1], "Traj-DRO(comb)", "Traj-DRO(comb) Coll. Reduction (%)"),
    ]:
        matrix = np.zeros((len(paths), len(obs_counts)))
        for i, p in enumerate(paths):
            for j, n in enumerate(obs_counts):
                base = df[(df["path"]==p) & (df["num_obstacles"]==n) & (df["method"]=="Base")]
                comp = df[(df["path"]==p) & (df["num_obstacles"]==n) & (df["method"]==compare_method)]
                if not base.empty and not comp.empty:
                    b = base.iloc[0]["collision_rate"] * 100
                    c = comp.iloc[0]["collision_rate"] * 100
                    matrix[i, j] = b - c

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(obs_counts)))
        ax.set_xticklabels([str(n) for n in obs_counts])
        ax.set_yticks(range(len(paths)))
        ax.set_yticklabels(paths, fontsize=8)
        ax.set_xlabel("Num Obstacles")
        ax.set_title(title)
        for i in range(len(paths)):
            for j in range(len(obs_counts)):
                v = matrix[i,j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 3 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("T5: Collision Rate Reduction over Base", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_t5_path_obstacle.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig T5 saved")


if __name__ == "__main__":
    print("Generating trajectory DRO figures...")
    fig_t1()
    fig_t2()
    fig_t3()
    fig_t4()
    fig_t5()
    print("Done.")
