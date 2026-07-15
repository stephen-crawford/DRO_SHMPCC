#!/usr/bin/env python3
"""Generate publication-quality figures for CDC paper: DRO vs Base.

Outputs to cdc_paper_figures/ in both PDF and PNG format.

Fig 1: Flagship collision rate results across best scenarios (N=2000)
Fig 2: Speed sweep on Tight-S path showing DRO benefit curve (N=1500)
Fig 3: Multi-obstacle robustness at DRO sweet spot (N=1500)
Fig 4: Clearance CDFs showing distributional shift (N=2000)
Fig 5: Path geometry × speed interaction heatmap (N=1500)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FOCUSED = "focused_figures"
OUT_DIR = "cdc_paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linewidth":   0.5,
    "lines.linewidth":  1.8,
    "lines.markersize": 6,
})

# Consistent colour palette — Base is grey, DRO methods get distinct colours
PAL = {
    "Base":             "#888888",
    "WDRO-inject-K1":   "#2166AC",
    "WDRO-sampling":    "#FF7F00",
    "TopRisk-K1":       "#B2182B",
}
# Readable names for legend
NICE = {
    "Base":             "Base (no DRO)",
    "WDRO-inject-K1":   "WDRO injection",
    "WDRO-sampling":    "WDRO sampling",
    "TopRisk-K1":       "TopRisk injection",
}
MARKERS = {
    "Base":             "s",
    "WDRO-inject-K1":   "o",
    "WDRO-sampling":    "D",
    "TopRisk-K1":       "^",
}

def _c(m):  return PAL.get(m, "#666666")
def _n(m):  return NICE.get(m, m)
def _m(m):  return MARKERS.get(m, "o")

def _save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ===================================================================
# Fig 1 — Flagship collision rate results (F8, N=2000)
# ===================================================================
def fig1_flagship():
    fpath = os.path.join(FOCUSED, "f8_best_dro_showcase.csv")
    if not os.path.exists(fpath):
        print("  Fig 1: no data, skipping"); return

    df = pd.read_csv(fpath)

    # Pretty scenario labels
    SCEN_NICE = {
        "Oncoming-default": "Oncoming",
        "Tight-S@v1.0":    "Tight-S (v=1.0)",
        "Tight-S@v1.3":    "Tight-S (v=1.3)",
        "S-curve-crossing": "Crossing",
    }
    scenarios = list(SCEN_NICE.keys())
    methods   = ["Base", "WDRO-inject-K1", "TopRisk-K1", "WDRO-sampling"]
    n_m = len(methods)

    fig, (ax_cr, ax_cl) = plt.subplots(1, 2, figsize=(10, 4.2))

    x = np.arange(len(scenarios))
    width = 0.72 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr, cr_lo, cr_hi, clr = [], [], [], []
        for sc in scenarios:
            row = sub[sub["scenario"] == sc]
            if row.empty:
                cr.append(0); cr_lo.append(0); cr_hi.append(0); clr.append(0)
            else:
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
                clr.append(row.iloc[0]["mean_clearance"])
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        offset = (i - n_m / 2 + 0.5) * width
        ax_cr.bar(x + offset, cr, width * 0.88, yerr=err, label=_n(method),
                  color=_c(method), capsize=3, error_kw={"linewidth": 0.8},
                  edgecolor="white", linewidth=0.4)
        ax_cl.bar(x + offset, clr, width * 0.88, color=_c(method),
                  edgecolor="white", linewidth=0.4)

    ax_cr.set_ylabel("Collision Rate (%)")
    ax_cr.set_xticks(x)
    ax_cr.set_xticklabels([SCEN_NICE[s] for s in scenarios], fontsize=9)
    ax_cr.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax_cr.legend(loc="upper right", framealpha=0.9, edgecolor="0.8")
    ax_cr.set_title("(a)  Collision Rate")

    ax_cl.set_ylabel("Mean Clearance (m)")
    ax_cl.set_xticks(x)
    ax_cl.set_xticklabels([SCEN_NICE[s] for s in scenarios], fontsize=9)
    ax_cl.set_ylim(bottom=0)
    ax_cl.set_title("(b)  Mean Clearance")

    fig.suptitle("Collision Rate and Clearance Across Scenarios  (N = 2 000)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_dro_flagship")
    print("  Fig 1 saved")


# ===================================================================
# Fig 2 — Speed sweep on Tight-S (F10, N=1500)
# ===================================================================
def fig2_speed_sweep():
    fpath = os.path.join(FOCUSED, "f10_tight_s_speed_sweep.csv")
    if not os.path.exists(fpath):
        print("  Fig 2: no data, skipping"); return

    df = pd.read_csv(fpath)
    methods = ["Base", "WDRO-inject-K1", "TopRisk-K1", "WDRO-sampling"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    for method in methods:
        sub = df[df["method"] == method].sort_values("obstacle_speed")
        sp  = sub["obstacle_speed"].values
        cr  = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        clr = sub["mean_clearance"].values

        kw = dict(color=_c(method), marker=_m(method), label=_n(method))
        ax1.plot(sp, cr, **kw)
        ax1.fill_between(sp, clo, chi, alpha=0.12, color=_c(method))
        ax2.plot(sp, clr, **kw)

    ax1.set_xlabel("Obstacle Speed (m/s)")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right", framealpha=0.9, edgecolor="0.8")
    ax1.set_title("(a)  Collision Rate")

    ax2.set_xlabel("Obstacle Speed (m/s)")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", framealpha=0.9, edgecolor="0.8")
    ax2.set_title("(b)  Mean Clearance")

    fig.suptitle("Tight-S Speed Sweep  (N = 1 500)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_dro_speed_sweep")
    print("  Fig 2 saved")


# ===================================================================
# Fig 3 — Multi-obstacle robustness (F9, N=1500)
# ===================================================================
def fig3_multi_obstacle():
    fpath = os.path.join(FOCUSED, "f9_multi_obs_sweet_spot.csv")
    if not os.path.exists(fpath):
        print("  Fig 3: no data, skipping"); return

    df = pd.read_csv(fpath)
    obs_counts = sorted(df["num_obstacles"].unique())
    methods = ["Base", "WDRO-inject-K1", "TopRisk-K1", "WDRO-sampling"]
    n_m = len(methods)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    x = np.arange(len(obs_counts))
    width = 0.72 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr, cr_lo, cr_hi, clr = [], [], [], []
        for n in obs_counts:
            row = sub[sub["num_obstacles"] == n]
            if row.empty:
                cr.append(0); cr_lo.append(0); cr_hi.append(0); clr.append(0)
            else:
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
                clr.append(row.iloc[0]["mean_clearance"])
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        offset = (i - n_m / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.88, yerr=err, label=_n(method),
                color=_c(method), capsize=3, error_kw={"linewidth": 0.8},
                edgecolor="white", linewidth=0.4)
        ax2.bar(x + offset, clr, width * 0.88, color=_c(method),
                edgecolor="white", linewidth=0.4)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in obs_counts])
    ax1.set_xlabel("Number of Obstacles")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.legend(loc="upper left", framealpha=0.9, edgecolor="0.8")
    ax1.set_title("(a)  Collision Rate")

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in obs_counts])
    ax2.set_xlabel("Number of Obstacles")
    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_ylim(bottom=0)
    ax2.set_title("(b)  Mean Clearance")

    fig.suptitle("Multi-Obstacle Scaling at Sweet Spot  (Tight-S, v = 1.0 m/s, N = 1 500)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "fig_dro_multi_obstacle")
    print("  Fig 3 saved")


# ===================================================================
# Fig 4 — Clearance CDFs (F8 raw, N=2000)
# ===================================================================
def fig4_clearance_cdf():
    fpath = os.path.join(FOCUSED, "f8_showcase_clearance_raw.csv")
    if not os.path.exists(fpath):
        print("  Fig 4: no data, skipping"); return

    df = pd.read_csv(fpath)

    SCEN_NICE = {
        "Oncoming-default": "Oncoming",
        "Tight-S@v1.0":    "Tight-S (v = 1.0)",
        "Tight-S@v1.3":    "Tight-S (v = 1.3)",
        "S-curve-crossing": "Crossing",
    }
    scenarios = list(SCEN_NICE.keys())
    methods   = ["Base", "WDRO-inject-K1", "TopRisk-K1", "WDRO-sampling"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)

    for ax_i, sc in enumerate(scenarios):
        ax = axes[ax_i]
        sdf = df[df["scenario"] == sc]

        for method in methods:
            sub = sdf[sdf["method"] == method]
            if sub.empty:
                continue
            vals = np.sort(sub["min_clearance"].values)
            cdf  = np.arange(1, len(vals) + 1) / len(vals)
            lw = 1.3 if method != "Base" else 2.0
            ls = "-" if method != "Base" else "--"
            ax.plot(vals, cdf, color=_c(method), linewidth=lw,
                    linestyle=ls, label=_n(method))

        ax.axvline(x=0.5, color="#CC0000", linestyle="--", linewidth=0.9,
                   alpha=0.55, label="Safety radius" if ax_i == 0 else None)
        ax.set_xlabel("Min Clearance (m)")
        if ax_i == 0:
            ax.set_ylabel("CDF")
        ax.set_title(SCEN_NICE[sc], fontsize=10)
        ax.set_xlim(left=0, right=3.0)
        ax.set_ylim(0, 1)

    # Single shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               framealpha=0.9, edgecolor="0.8", fontsize=9,
               bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Clearance Distribution  (N = 2 000 per method)",
                 fontsize=12, y=1.12)
    fig.tight_layout()
    _save(fig, "fig_dro_clearance_cdf")
    print("  Fig 4 saved")


# ===================================================================
# Fig 5 — Path geometry × speed heatmap (F11, N=1500)
# ===================================================================
def fig5_path_heatmap():
    fpath = os.path.join(FOCUSED, "f11_path_geometry_high_speed.csv")
    if not os.path.exists(fpath):
        print("  Fig 5: no data, skipping"); return

    df = pd.read_csv(fpath)

    path_order  = ["Straight", "S-curve", "Tight-S"]
    speed_order = sorted(df["obstacle_speed"].unique())

    # Build benefit matrix (inject-K1 vs Base)
    matrix_cr  = np.full((len(path_order), len(speed_order)), np.nan)
    matrix_clr = np.full((len(path_order), len(speed_order)), np.nan)

    for i, p in enumerate(path_order):
        for j, s in enumerate(speed_order):
            base = df[(df["path"] == p) & (df["obstacle_speed"] == s) &
                      (df["method"] == "Base")]
            dro  = df[(df["path"] == p) & (df["obstacle_speed"] == s) &
                      (df["method"] == "WDRO-inject-K1")]
            if not base.empty and not dro.empty:
                matrix_cr[i, j]  = (base.iloc[0]["collision_rate"]
                                    - dro.iloc[0]["collision_rate"]) * 100
                matrix_clr[i, j] = (dro.iloc[0]["mean_clearance"]
                                    - base.iloc[0]["mean_clearance"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))

    # --- Collision rate reduction ---
    vmax_cr = max(abs(np.nanmin(matrix_cr)), np.nanmax(matrix_cr))
    im1 = ax1.imshow(matrix_cr, cmap="RdYlGn", aspect="auto",
                     vmin=-vmax_cr, vmax=vmax_cr)
    ax1.set_xticks(range(len(speed_order)))
    ax1.set_xticklabels([f"{s:.1f}" for s in speed_order])
    ax1.set_xlabel("Obstacle Speed (m/s)")
    ax1.set_yticks(range(len(path_order)))
    ax1.set_yticklabels(path_order)
    ax1.set_title("(a)  Collision Rate Reduction (pp)")
    for ir in range(len(path_order)):
        for jc in range(len(speed_order)):
            v = matrix_cr[ir, jc]
            if np.isnan(v):
                continue
            txt = f"{v:+.1f}" if v != 0 else "0.0"
            brightness = abs(v) / vmax_cr if vmax_cr else 0
            ax1.text(jc, ir, txt, ha="center", va="center", fontsize=9,
                     fontweight="bold" if abs(v) > 20 else "normal",
                     color="white" if brightness > 0.55 else "black")
    fig.colorbar(im1, ax=ax1, shrink=0.85, label="pp")

    # --- Clearance change ---
    vmax_cl = max(abs(np.nanmin(matrix_clr)), np.nanmax(matrix_clr))
    im2 = ax2.imshow(matrix_clr, cmap="RdYlGn", aspect="auto",
                     vmin=-vmax_cl, vmax=vmax_cl)
    ax2.set_xticks(range(len(speed_order)))
    ax2.set_xticklabels([f"{s:.1f}" for s in speed_order])
    ax2.set_xlabel("Obstacle Speed (m/s)")
    ax2.set_yticks(range(len(path_order)))
    ax2.set_yticklabels(path_order)
    ax2.set_title("(b)  Clearance Change (m)")
    for ir in range(len(path_order)):
        for jc in range(len(speed_order)):
            v = matrix_clr[ir, jc]
            if np.isnan(v):
                continue
            txt = f"{v:+.2f}"
            brightness = abs(v) / vmax_cl if vmax_cl else 0
            ax2.text(jc, ir, txt, ha="center", va="center", fontsize=9,
                     fontweight="bold" if abs(v) > 0.15 else "normal",
                     color="white" if brightness > 0.55 else "black")
    fig.colorbar(im2, ax=ax2, shrink=0.85, label="m")

    fig.suptitle("WDRO-injection Benefit vs Base by Path and Speed  (N = 1 500)",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    _save(fig, "fig_dro_path_heatmap")
    print("  Fig 5 saved")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating CDC paper DRO figures ...")
    fig1_flagship()
    fig2_speed_sweep()
    fig3_multi_obstacle()
    fig4_clearance_cdf()
    fig5_path_heatmap()
    print("Done.  Output in", OUT_DIR)
