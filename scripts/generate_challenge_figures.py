#!/usr/bin/env python3
"""Generate figures from G6–G8 / T6–T7 challenging environment experiments.

Fig G6: Challenging environment variants (grouped bars)
Fig G7: Obstacle speed sweep (line plots)
Fig G8: Path x challenge heatmaps
Fig T6: Challenging envs — Mode DRO vs Traj DRO
Fig T7: Speed sweep — Mode DRO vs Traj DRO
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

G_DIR = "generalization_figures"
T_DIR = "traj_dro_figures"
os.makedirs(G_DIR, exist_ok=True)
os.makedirs(T_DIR, exist_ok=True)

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

G_COLORS = {
    "Base":             "#999999",
    "WDRO-sampling":    "#FF7F00",
    "WDRO-inject-K1":   "#2166AC",
    "WDRO-inject-K2":   "#4393C3",
    "TopRisk-K1":       "#B2182B",
    "TopRisk-K2":       "#E06666",
    "DiverseRisk-K1":   "#4DAF4A",
    "Softmax-tau5":     "#984EA3",
}
T_COLORS = {
    "Base":             "#999999",
    "Mode-DRO(q*)":     "#FF7F00",
    "Mode-DRO(inj)":    "#2166AC",
    "Traj-DRO(q*)":     "#E41A1C",
    "Traj-DRO(inj)":    "#4DAF4A",
    "Traj-DRO(comb)":   "#984EA3",
}
G_MARKERS = {
    "Base": "s", "WDRO-sampling": "D", "WDRO-inject-K1": "o",
    "WDRO-inject-K2": "v", "TopRisk-K1": "^", "DiverseRisk-K1": "P", "Softmax-tau5": "X",
}
T_MARKERS = {
    "Base": "s", "Mode-DRO(q*)": "D", "Mode-DRO(inj)": "o",
    "Traj-DRO(q*)": "^", "Traj-DRO(inj)": "P", "Traj-DRO(comb)": "X",
}


def _gc(m): return G_COLORS.get(m, "#666666")
def _tc(m): return T_COLORS.get(m, "#666666")
def _gm(m): return G_MARKERS.get(m, "o")
def _tm(m): return T_MARKERS.get(m, "o")


# ============================================================================
# Fig G6: Challenging Environments
# ============================================================================

def _draw_challenge_diagram(ax, challenge):
    """Draw a schematic of the challenging environment."""
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.65, 0.65)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- road (always a straight path for ego) ---
    t = np.linspace(0, 1, 200)
    px, py = t, np.zeros_like(t)
    ax.plot(px, py, color="#C0C0C0", linewidth=10, solid_capstyle="round", zorder=1)
    ax.plot(px, py, color="#E8E8E8", linewidth=6, solid_capstyle="round", zorder=2)

    def _car(ax, cx, cy, tx, ty, color, light_color, size=1.0):
        """Draw a car rectangle at (cx,cy) facing (tx,ty)."""
        nx, ny = -ty, tx
        ew, eh = 0.07 * size, 0.035 * size
        corners = [
            (cx + ew*tx + eh*nx, cy + ew*ty + eh*ny),
            (cx + ew*tx - eh*nx, cy + ew*ty - eh*ny),
            (cx - ew*tx - eh*nx, cy - ew*ty - eh*ny),
            (cx - ew*tx + eh*nx, cy - ew*ty + eh*ny),
        ]
        ax.add_patch(plt.Polygon(corners, closed=True, facecolor=color,
                                 edgecolor="white", linewidth=0.8, zorder=4))
        ws = 0.025 * size
        tri = plt.Polygon([
            (cx + (ew-0.008)*tx, cy + (ew-0.008)*ty),
            (cx + (ew-0.028)*tx + ws*nx, cy + (ew-0.028)*ty + ws*ny),
            (cx + (ew-0.028)*tx - ws*nx, cy + (ew-0.028)*ty - ws*ny),
        ], closed=True, facecolor=light_color, edgecolor="none", zorder=5)
        ax.add_patch(tri)

    def _arrow(ax, cx, cy, tx, ty, color, length=0.12):
        ax.annotate("", xy=(cx + length*tx, cy + length*ty),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                    zorder=5)

    # --- ego (always at 15%, facing right) ---
    ei = int(0.15 * len(t))
    _car(ax, px[ei], py[ei], 1.0, 0.0, "#2166AC", "#4A90D9")
    _arrow(ax, px[ei], py[ei], 1.0, 0.0, "#2166AC", length=0.12)
    ax.text(px[ei], py[ei] - 0.10, "ego", ha="center", va="top", fontsize=7,
            color="#2166AC", fontweight="bold", zorder=6)

    if challenge == "Crossing":
        # One obstacle crossing perpendicularly from above
        ox, oy = 0.60, 0.45
        _car(ax, ox, oy, 0.0, -1.0, "#CC2222", "#E06060")
        _arrow(ax, ox, oy, 0.0, -1.0, "#CC2222")
    elif challenge == "Diagonal":
        # One obstacle approaching diagonally
        ox, oy = 0.70, 0.40
        d = np.sqrt(2) / 2
        _car(ax, ox, oy, -d, -d, "#CC2222", "#E06060")
        _arrow(ax, ox, oy, -d, -d, "#CC2222")
    elif challenge == "Mixed-2obs":
        # Oncoming + crossing
        _car(ax, 0.55, 0.0, -1.0, 0.0, "#CC2222", "#E06060")
        _arrow(ax, 0.55, 0.0, -1.0, 0.0, "#CC2222", length=0.10)
        _car(ax, 0.50, 0.42, 0.0, -1.0, "#CC2222", "#E06060", size=0.85)
        _arrow(ax, 0.50, 0.42, 0.0, -1.0, "#CC2222", length=0.10)
    elif challenge == "Mixed-3obs":
        # Oncoming + crossing + diagonal
        _car(ax, 0.55, 0.0, -1.0, 0.0, "#CC2222", "#E06060", size=0.85)
        _arrow(ax, 0.55, 0.0, -1.0, 0.0, "#CC2222", length=0.09)
        _car(ax, 0.45, 0.42, 0.0, -1.0, "#CC2222", "#E06060", size=0.85)
        _arrow(ax, 0.45, 0.42, 0.0, -1.0, "#CC2222", length=0.09)
        d = np.sqrt(2) / 2
        _car(ax, 0.70, 0.35, -d, -d, "#CC2222", "#E06060", size=0.85)
        _arrow(ax, 0.70, 0.35, -d, -d, "#CC2222", length=0.09)
    elif challenge == "Dense-4obs":
        # 4 obstacles surrounding
        _car(ax, 0.55, 0.0, -1.0, 0.0, "#CC2222", "#E06060", size=0.75)
        _arrow(ax, 0.55, 0.0, -1.0, 0.0, "#CC2222", length=0.08)
        _car(ax, 0.45, 0.40, 0.0, -1.0, "#CC2222", "#E06060", size=0.75)
        _arrow(ax, 0.45, 0.40, 0.0, -1.0, "#CC2222", length=0.08)
        d = np.sqrt(2) / 2
        _car(ax, 0.68, 0.32, -d, -d, "#CC2222", "#E06060", size=0.75)
        _arrow(ax, 0.68, 0.32, -d, -d, "#CC2222", length=0.08)
        _car(ax, 0.50, -0.40, 0.0, 1.0, "#CC2222", "#E06060", size=0.75)
        _arrow(ax, 0.50, -0.40, 0.0, 1.0, "#CC2222", length=0.08)

    # Nice label
    nice = {"Crossing": "Crossing", "Diagonal": "Diagonal",
            "Mixed-2obs": "Mixed (2 obs)", "Mixed-3obs": "Mixed (3 obs)",
            "Dense-4obs": "Dense (4 obs)"}
    ax.text(0.5, -0.58, nice.get(challenge, challenge), ha="center", va="top",
            fontsize=10, fontweight="bold")


def fig_g6():
    fpath = os.path.join(G_DIR, "g6_challenging_envs.csv")
    if not os.path.exists(fpath):
        print("  Fig G6: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    # Drop high-speed challenges
    drop_challenges = {"HighSpeed-1.5", "HighSpeed-2.0", "HighSpeed-Dense"}
    df = df[~df["challenge"].isin(drop_challenges)]

    # Keep specific methods in order
    G6_METHODS = ["Base", "WDRO-sampling", "WDRO-inject-K1", "WDRO-inject-K2"]
    df = df[df["method"].isin(G6_METHODS)]

    challenges = [c for c in df["challenge"].unique()]
    methods = [m for m in G6_METHODS if m in df["method"].values]
    n_ch = len(challenges)
    n_m = len(methods)

    NICE = {
        "Base":           "Base (no DRO)",
        "WDRO-sampling":  "WDRO sampling",
        "WDRO-inject-K1": "WDRO inject K=1",
        "WDRO-inject-K2": "WDRO inject K=2",
        "TopRisk-K1":     "TopRisk K=1",
        "TopRisk-K2":     "TopRisk K=2",
    }

    fig = plt.figure(figsize=(3.0 * n_ch, 9.5))
    gs = fig.add_gridspec(3, n_ch, height_ratios=[1.2, 1.4, 1.4],
                          hspace=0.30, wspace=0.35)

    x = np.arange(n_m)
    width = 0.72

    # Compute shared y-limits across columns
    cr_max = 0
    clr_max = 0
    for ch in challenges:
        for method in methods:
            sub = df[(df["challenge"] == ch) & (df["method"] == method)]
            if not sub.empty:
                cr_hi = sub["coll_ci_hi"].values[0] * 100
                cr_max = max(cr_max, cr_hi)
                clr_max = max(clr_max, sub["mean_clearance"].values[0])
    cr_max = cr_max * 1.15
    clr_max = clr_max * 1.15

    ax_cr_first = None
    ax_cl_first = None

    for col, ch in enumerate(challenges):
        # --- Row 0: diagram ---
        ax_d = fig.add_subplot(gs[0, col])
        _draw_challenge_diagram(ax_d, ch)

        # --- Row 1: Collision Rate ---
        ax_cr = fig.add_subplot(gs[1, col], sharey=ax_cr_first)
        if ax_cr_first is None:
            ax_cr_first = ax_cr

        # --- Row 2: Clearance ---
        ax_cl = fig.add_subplot(gs[2, col], sharey=ax_cl_first)
        if ax_cl_first is None:
            ax_cl_first = ax_cl

        cr_vals = []
        cr_errs_lo = []
        cr_errs_hi = []
        clr_vals = []
        clr_p5 = []
        clr_n = []
        for method in methods:
            sub = df[(df["challenge"] == ch) & (df["method"] == method)]
            if not sub.empty:
                cr_vals.append(sub["collision_rate"].values[0] * 100)
                cr_errs_lo.append(sub["coll_ci_lo"].values[0] * 100)
                cr_errs_hi.append(sub["coll_ci_hi"].values[0] * 100)
                clr_vals.append(sub["mean_clearance"].values[0])
                clr_p5.append(sub["p5_clearance"].values[0])
                clr_n.append(float(sub["n_rollouts"].values[0]))
            else:
                cr_vals.append(0)
                cr_errs_lo.append(0)
                cr_errs_hi.append(0)
                clr_vals.append(0)
                clr_p5.append(0)
                clr_n.append(1)

        cr_vals = np.array(cr_vals)
        err = np.array([cr_vals - np.array(cr_errs_lo),
                        np.array(cr_errs_hi) - cr_vals])

        # Wilson-style CI for clearance: estimate std from p5, then CI = mean ± z*std/sqrt(n)
        clr_vals = np.array(clr_vals)
        clr_p5 = np.array(clr_p5)
        clr_n = np.array(clr_n)
        std_est = np.maximum((clr_vals - clr_p5) / 1.645, 1e-6)
        ci_margin = 1.96 * std_est / np.sqrt(clr_n)
        clr_err = np.array([ci_margin, ci_margin])

        colors = [_gc(m) for m in methods]
        ax_cr.bar(x, cr_vals, width, yerr=err, color=colors,
                  capsize=3, error_kw={"linewidth": 0.7},
                  edgecolor="white", linewidth=0.3)
        ax_cl.bar(x, clr_vals, width, yerr=clr_err, color=colors,
                  capsize=3, error_kw={"linewidth": 0.7},
                  edgecolor="white", linewidth=0.3)

        ax_cr.set_ylim(0, cr_max)
        ax_cl.set_ylim(0, clr_max)

        # X-axis labels only on bottom row
        ax_cr.set_xticks([])
        ax_cl.set_xticks(x)
        ax_cl.set_xticklabels([NICE.get(m, m) for m in methods],
                              rotation=40, ha="right", fontsize=7)

        # Y-axis labels only on first column
        if col == 0:
            ax_cr.set_ylabel("Collision Rate (%)")
            ax_cl.set_ylabel("Mean Clearance (m)")
        else:
            ax_cr.tick_params(labelleft=False)
            ax_cl.tick_params(labelleft=False)

    # Shared legend at top
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=_gc(m), edgecolor="white",
               linewidth=0.3) for m in methods]
    labels = [NICE.get(m, m) for m in methods]
    fig.legend(handles, labels, loc="upper center", ncol=n_m, fontsize=11,
               framealpha=0.9, edgecolor="0.85", columnspacing=2.0,
               handlelength=2.0, handleheight=1.2, borderpad=0.8,
               bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Challenging Environment Variants  (N = 500)",
                 fontsize=13, y=1.04)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(G_DIR, f"fig_g6_challenging_envs.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig G6 saved")


# ============================================================================
# Fig G7: Speed Sweep
# ============================================================================
def fig_g7():
    fpath = os.path.join(G_DIR, "g7_speed_sweep.csv")
    if not os.path.exists(fpath):
        print("  Fig G7: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("obstacle_speed")
        sp = sub["obstacle_speed"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        clr = sub["mean_clearance"].values

        style = dict(color=_gc(method), marker=_gm(method), markersize=5, linewidth=1.3)
        ax1.plot(sp, cr, label=method, **style)
        ax1.fill_between(sp, clo, chi, alpha=0.1, color=_gc(method))
        ax2.plot(sp, clr, label=method, **style)

    ax1.set_xlabel("Obstacle Speed (m/s)")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs Obstacle Speed")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Obstacle Speed (m/s)")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance vs Obstacle Speed")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("G7: Obstacle Speed Sweep", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(G_DIR, f"fig_g7_speed_sweep.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig G7 saved")


# ============================================================================
# Fig G8: Path x Challenge Heatmaps
# ============================================================================
def fig_g8():
    fpath = os.path.join(G_DIR, "g8_path_challenge.csv")
    if not os.path.exists(fpath):
        print("  Fig G8: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    paths = df["path"].unique()
    obs_types = df["obstacle_type"].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title, fmt in [
        (ax1, "collision_rate", "Collision Rate Reduction (%)\nWDRO-inject vs Base", ".1f"),
        (ax2, "mean_clearance", "Clearance Improvement (m)\nWDRO-inject vs Base", ".2f"),
    ]:
        matrix = np.zeros((len(paths), len(obs_types)))
        for i, p in enumerate(paths):
            for j, ot in enumerate(obs_types):
                base = df[(df["path"] == p) & (df["obstacle_type"] == ot) & (df["method"] == "Base")]
                wdro = df[(df["path"] == p) & (df["obstacle_type"] == ot) & (df["method"] == "WDRO-inject-K1")]
                if not base.empty and not wdro.empty:
                    b = base.iloc[0][metric]
                    w = wdro.iloc[0][metric]
                    if "collision" in metric:
                        matrix[i, j] = (b - w) * 100  # Reduction
                    else:
                        matrix[i, j] = w - b  # Improvement

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(obs_types)))
        ax.set_xticklabels(obs_types, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(paths)))
        ax.set_yticklabels(paths, fontsize=9)
        ax.set_title(title, fontsize=9)
        for i_r in range(len(paths)):
            for j_c in range(len(obs_types)):
                val = matrix[i_r, j_c]
                ax.text(j_c, i_r, f"{val:{fmt}}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > 5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("G8: Path Geometry x Obstacle Type Interaction", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(G_DIR, f"fig_g8_path_challenge.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig G8 saved")


# ============================================================================
# Fig T6: Challenging Envs — Mode DRO vs Traj DRO
# ============================================================================
def fig_t6():
    fpath = os.path.join(T_DIR, "t6_challenging_envs.csv")
    if not os.path.exists(fpath):
        print("  Fig T6: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    challenges = df["challenge"].unique()
    methods = df["method"].unique()
    n_m = len(methods)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    x = np.arange(len(challenges))
    width = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr = [sub[sub["challenge"] == c]["collision_rate"].values[0] * 100
              if len(sub[sub["challenge"] == c]) else 0 for c in challenges]
        cr_lo = [sub[sub["challenge"] == c]["coll_ci_lo"].values[0] * 100
                 if len(sub[sub["challenge"] == c]) else 0 for c in challenges]
        cr_hi = [sub[sub["challenge"] == c]["coll_ci_hi"].values[0] * 100
                 if len(sub[sub["challenge"] == c]) else 0 for c in challenges]
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        clr = [sub[sub["challenge"] == c]["mean_clearance"].values[0]
               if len(sub[sub["challenge"] == c]) else 0 for c in challenges]

        offset = (i - n_m / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.9, yerr=err, label=method,
                color=_tc(method), capsize=2, error_kw={"linewidth": 0.7})
        ax2.bar(x + offset, clr, width * 0.9, color=_tc(method))

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate: Mode DRO vs Trajectory DRO")
    ax1.legend(fontsize=6.5, ncol=3, loc="upper center")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance: Mode DRO vs Trajectory DRO")
    ax2.set_xticks(x)
    ax2.set_xticklabels(challenges, rotation=25, ha="right")

    fig.suptitle("T6: Challenging Environments — Mode DRO vs Traj DRO", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(T_DIR, f"fig_t6_challenging_envs.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig T6 saved")


# ============================================================================
# Fig T7: Speed Sweep — Mode DRO vs Traj DRO
# ============================================================================
def fig_t7():
    fpath = os.path.join(T_DIR, "t7_speed_sweep.csv")
    if not os.path.exists(fpath):
        print("  Fig T7: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("obstacle_speed")
        sp = sub["obstacle_speed"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        clr = sub["mean_clearance"].values

        style = dict(color=_tc(method), marker=_tm(method), markersize=5, linewidth=1.3)
        ax1.plot(sp, cr, label=method, **style)
        ax1.fill_between(sp, clo, chi, alpha=0.1, color=_tc(method))
        ax2.plot(sp, clr, label=method, **style)

    ax1.set_xlabel("Obstacle Speed (m/s)")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs Obstacle Speed")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Obstacle Speed (m/s)")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Clearance vs Obstacle Speed")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("T7: Speed Sweep — Mode DRO vs Traj DRO", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(T_DIR, f"fig_t7_speed_sweep.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig T7 saved")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating challenging environment figures...")
    fig_g6()
    fig_g7()
    fig_g8()
    fig_t6()
    fig_t7()
    print("Done.")
