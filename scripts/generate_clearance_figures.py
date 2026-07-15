#!/usr/bin/env python3
"""Generate clearance analysis figures from C1 and C2 CDC experiments.

Fig C1a: Clearance percentile comparison across environments (4 envs x 3 methods)
Fig C1b: Clearance CDFs from raw per-rollout data (Oncoming + Intersection)
Fig C1c: Conditional clearance bar chart (collision vs non-collision runs)
Fig C2:  Baseline clearance comparison on Oncoming (7 methods)
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

DIR = "cdc_paper_figures"
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
    "WDRO-injection":   "#2166AC",
    "Uniform-coverage": "#4DAF4A",
    "Softmax-risk":     "#984EA3",
    "Eps-greedy":       "#E41A1C",
    "Top-risk-inject":  "#377EB8",
}

METHOD_ORDER_C1 = ["Base", "WDRO-sampling", "WDRO-injection"]
METHOD_ORDER_C2 = ["Base", "WDRO-sampling", "WDRO-injection",
                   "Uniform-coverage", "Softmax-risk", "Eps-greedy", "Top-risk-inject"]
ENV_ORDER = ["Straight", "Narrow", "Intersection", "Oncoming"]
PERCENTILES = ["p5", "p10", "p25", "p50", "p75", "p90", "p95"]


def _color(method):
    return COLORS.get(method, "#666666")


# ============================================================================
# Fig C1a: Clearance percentile ladder across environments
# ============================================================================
def fig_c1a():
    fpath = os.path.join(DIR, "cdc_c1_clearance_analysis.csv")
    if not os.path.exists(fpath):
        print("  Fig C1a: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)

    for ax_i, env in enumerate(ENV_ORDER):
        ax = axes[ax_i]
        edf = df[df["environment"] == env]
        methods = [m for m in METHOD_ORDER_C1 if m in edf["method"].values]

        x = np.arange(len(PERCENTILES))
        n_m = len(methods)
        width = 0.8 / n_m

        for i, method in enumerate(methods):
            row = edf[edf["method"] == method].iloc[0]
            vals = [row[p] for p in PERCENTILES]
            offset = (i - n_m / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, label=method, color=_color(method))

        ax.set_xticks(x)
        ax.set_xticklabels(PERCENTILES, rotation=45, ha="right")
        ax.set_title(env)
        ax.set_ylabel("Clearance (m)" if ax_i == 0 else "")
        ax.set_ylim(bottom=0)
        if ax_i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("C1: Clearance Percentiles Across Environments", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_c1a_clearance_percentiles.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  Fig C1a saved")


# ============================================================================
# Fig C1b: Clearance CDFs from raw data (Oncoming + Intersection)
# ============================================================================
def fig_c1b():
    fpath = os.path.join(DIR, "cdc_c1_clearance_raw.csv")
    if not os.path.exists(fpath):
        print("  Fig C1b: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    envs_to_plot = ["Oncoming", "Intersection"]
    fig, axes = plt.subplots(1, len(envs_to_plot), figsize=(10, 4))

    for ax_i, env in enumerate(envs_to_plot):
        ax = axes[ax_i]
        edf = df[df["environment"] == env]

        for method in METHOD_ORDER_C1:
            sub = edf[edf["method"] == method]
            if sub.empty:
                continue
            clr = np.sort(sub["min_clearance"].values)
            cdf = np.arange(1, len(clr) + 1) / len(clr)
            ax.plot(clr, cdf, label=method, color=_color(method), linewidth=1.5)

        # Mark safety radius
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Safety r")
        ax.set_xlabel("Min Clearance (m)")
        ax.set_ylabel("CDF" if ax_i == 0 else "")
        ax.set_title(f"{env}: Clearance CDF")
        ax.legend(fontsize=7)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("C1: Clearance Distribution (CDF)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_c1b_clearance_cdf.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig C1b saved")


# ============================================================================
# Fig C1c: Conditional clearance (collision vs non-collision runs)
# ============================================================================
def fig_c1c():
    fpath = os.path.join(DIR, "cdc_c1_clearance_analysis.csv")
    if not os.path.exists(fpath):
        print("  Fig C1c: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    # Only environments with non-trivial collision rates
    envs = ["Intersection", "Oncoming"]

    fig, axes = plt.subplots(1, len(envs), figsize=(10, 4))

    for ax_i, env in enumerate(envs):
        ax = axes[ax_i]
        edf = df[df["environment"] == env]
        methods = [m for m in METHOD_ORDER_C1 if m in edf["method"].values]

        x = np.arange(len(methods))
        width = 0.35

        coll_clr = []
        nocoll_clr = []
        for method in methods:
            row = edf[edf["method"] == method].iloc[0]
            coll_clr.append(row["mean_clearance_coll"])
            nocoll_clr.append(row["mean_clearance_nocoll"])

        ax.bar(x - width / 2, nocoll_clr, width, label="No collision", color="#4DAF4A", alpha=0.85)
        ax.bar(x + width / 2, coll_clr, width, label="Collision", color="#E41A1C", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("Mean Clearance (m)" if ax_i == 0 else "")
        ax.set_title(f"{env}: Clearance by Outcome")
        ax.legend(fontsize=7)
        ax.set_ylim(bottom=0)

    fig.suptitle("C1: Conditional Clearance (Collision vs Non-Collision)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_c1c_conditional_clearance.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("  Fig C1c saved")


# ============================================================================
# Fig C1d: Combined collision rate + mean clearance overview
# ============================================================================
def fig_c1d():
    fpath = os.path.join(DIR, "cdc_c1_clearance_analysis.csv")
    if not os.path.exists(fpath):
        print("  Fig C1d: no data found, skipping")
        return

    df = pd.read_csv(fpath)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    methods = METHOD_ORDER_C1
    n_m = len(methods)
    x = np.arange(len(ENV_ORDER))
    width = 0.8 / n_m

    for i, method in enumerate(methods):
        cr = []
        cr_lo = []
        cr_hi = []
        clr = []
        clr_lo = []
        clr_hi = []
        for env in ENV_ORDER:
            row = df[(df["environment"] == env) & (df["method"] == method)]
            if row.empty:
                cr.append(0); cr_lo.append(0); cr_hi.append(0)
                clr.append(0); clr_lo.append(0); clr_hi.append(0)
            else:
                r = row.iloc[0]
                cr.append(r["collision_rate"] * 100)
                cr_lo.append(r["coll_ci_lo"] * 100)
                cr_hi.append(r["coll_ci_hi"] * 100)
                clr.append(r["mean_clearance"])
                clr_lo.append(r["clearance_ci_lo"])
                clr_hi.append(r["clearance_ci_hi"])

        cr = np.array(cr)
        cr_err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])
        clr = np.array(clr)
        clr_err = np.array([clr - np.array(clr_lo), np.array(clr_hi) - clr])

        offset = (i - n_m / 2 + 0.5) * width
        ax1.bar(x + offset, cr, width * 0.9, yerr=cr_err, label=method,
                color=_color(method), capsize=2, error_kw={"linewidth": 0.7})
        ax2.bar(x + offset, clr, width * 0.9, yerr=clr_err,
                color=_color(method), capsize=2, error_kw={"linewidth": 0.7})

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate by Environment")
    ax1.legend(fontsize=7, loc="upper left")

    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Mean Clearance by Environment")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ENV_ORDER)

    fig.suptitle("C1: Collision Rate vs Clearance Across Environments", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_c1d_coll_vs_clearance.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig C1d saved")


# ============================================================================
# Fig C2: All baselines clearance comparison on Oncoming
# ============================================================================
def fig_c2():
    fpath = os.path.join(DIR, "cdc_c2_clearance_baselines.csv")
    if not os.path.exists(fpath):
        print("  Fig C2: no data found, skipping")
        return

    df = pd.read_csv(fpath)
    methods = [m for m in METHOD_ORDER_C2 if m in df["method"].values]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Collision rate
    ax = axes[0]
    cr = [df[df["method"] == m].iloc[0]["collision_rate"] * 100 for m in methods]
    cr_lo = [df[df["method"] == m].iloc[0]["coll_ci_lo"] * 100 for m in methods]
    cr_hi = [df[df["method"] == m].iloc[0]["coll_ci_hi"] * 100 for m in methods]
    cr = np.array(cr)
    err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])
    colors = [_color(m) for m in methods]
    ax.barh(np.arange(len(methods)), cr, xerr=err, color=colors,
            capsize=3, error_kw={"linewidth": 0.8})
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("Collision Rate (%)")
    ax.set_title("Collision Rate")
    ax.invert_yaxis()

    # Panel 2: Clearance percentile profile
    ax = axes[1]
    pcts = ["p5", "p25", "p50", "p75", "p95"]
    x_pct = np.arange(len(pcts))
    for method in methods:
        row = df[df["method"] == method].iloc[0]
        vals = [row[p] for p in pcts]
        ax.plot(x_pct, vals, marker="o", markersize=4, label=method,
                color=_color(method), linewidth=1.3)
    ax.set_xticks(x_pct)
    ax.set_xticklabels(pcts)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Clearance (m)")
    ax.set_title("Clearance Percentile Profile")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 3: Conditional clearance
    ax = axes[2]
    x = np.arange(len(methods))
    w = 0.35
    coll_clr = [df[df["method"] == m].iloc[0]["mean_clearance_coll"] for m in methods]
    nocoll_clr = [df[df["method"] == m].iloc[0]["mean_clearance_nocoll"] for m in methods]

    ax.bar(x - w / 2, nocoll_clr, w, label="No collision", color="#4DAF4A", alpha=0.85)
    ax.bar(x + w / 2, coll_clr, w, label="Collision", color="#E41A1C", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("Mean Clearance (m)")
    ax.set_title("Conditional Clearance")
    ax.legend(fontsize=7)

    fig.suptitle("C2: All Baselines on Oncoming — Clearance Analysis", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_c2_baseline_clearance.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig C2 saved")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating clearance analysis figures...")
    fig_c1a()
    fig_c1b()
    fig_c1c()
    fig_c1d()
    fig_c2()
    print("Done.")
