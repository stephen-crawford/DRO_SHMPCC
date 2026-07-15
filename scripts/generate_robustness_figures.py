#!/usr/bin/env python3
"""Generate figures for robustness experiments R1–R4."""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = pathlib.Path("figures/robustness")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Base": "#888888",
    "WDRO-sampling": "#1f77b4",
    "WDRO-inject-K1": "#2ca02c",
    "WDRO-inject-K2": "#17becf",
    "TopRisk-K1": "#ff7f0e",
    "TopRisk-K2": "#d62728",
}

def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}")


# ─── R1: Novel Unseen Modes ──────────────────────────────────────────────
def wilson_clearance_ci(mean_clr, p5_clr, n):
    """Estimate CI for clearance using std recovered from p5 percentile."""
    std_est = max((mean_clr - p5_clr) / 1.645, 1e-6)
    margin = 1.96 * std_est / np.sqrt(n)
    return mean_clr - margin, mean_clr + margin

def fig_r1():
    df = pd.read_csv(OUT / "r1_novel_mode.csv")
    modes = df["novel_mode"].unique()
    methods = df["method"].unique()
    n_methods = len(methods)
    x = np.arange(len(modes))
    w = 0.8 / n_methods

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collision rate with Wilson CI whiskers
    ax = axes[0]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["novel_mode"] == nm]["collision_rate"].values[0] for nm in modes]
        lo = [sub[sub["novel_mode"] == nm]["coll_ci_lo"].values[0] for nm in modes]
        hi = [sub[sub["novel_mode"] == nm]["coll_ci_hi"].values[0] for nm in modes]
        err_lo = [v - l for v, l in zip(vals, lo)]
        err_hi = [h - v for v, h in zip(vals, hi)]
        ax.bar(x + i * w, vals, w, yerr=[err_lo, err_hi],
               label=m, color=COLORS.get(m, "#1f77b4"), capsize=3, alpha=0.85)
    ax.set_xticks(x + w * (n_methods - 1) / 2)
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.set_ylabel("Collision Rate")
    ax.set_title("Collision Rate — Novel Unseen Modes")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Mean clearance with Wilson-style CI whiskers
    ax = axes[1]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals, err_lo, err_hi = [], [], []
        for nm in modes:
            row = sub[sub["novel_mode"] == nm].iloc[0]
            mc = row["mean_clearance"]
            p5 = row["p5_clearance"]
            n = row["n_rollouts"]
            clo, chi = wilson_clearance_ci(mc, p5, n)
            vals.append(mc)
            err_lo.append(mc - clo)
            err_hi.append(chi - mc)
        ax.bar(x + i * w, vals, w, yerr=[err_lo, err_hi],
               label=m, color=COLORS.get(m, "#1f77b4"), capsize=3, alpha=0.85)
    ax.set_xticks(x + w * (n_methods - 1) / 2)
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.set_ylabel("Mean Clearance")
    ax.set_title("Mean Clearance — Novel Unseen Modes")
    ax.legend(fontsize=8)

    fig.suptitle("Novel (Unseen) Mode Robustness", fontsize=14, y=1.02)
    save(fig, "fig_r1_novel_mode")


# ─── R2: Misclassification Sweep ─────────────────────────────────────────
def fig_r2():
    df = pd.read_csv(OUT / "r2_misclassification.csv")
    rates = sorted(df["misclass_rate"].unique())
    methods = df["method"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for m in methods:
        sub = df[df["method"] == m].sort_values("misclass_rate")
        ax.plot(sub["misclass_rate"] * 100, sub["collision_rate"],
                "o-", label=m, color=COLORS.get(m, "#1f77b4"), linewidth=2)
        ax.fill_between(sub["misclass_rate"] * 100, sub["coll_ci_lo"], sub["coll_ci_hi"],
                        alpha=0.15, color=COLORS.get(m, "#1f77b4"))
    ax.set_xlabel("Misclassification Rate (%)")
    ax.set_ylabel("Collision Rate")
    ax.set_title("R2: Collision vs Misclassification Rate")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    ax = axes[1]
    for m in methods:
        sub = df[df["method"] == m].sort_values("misclass_rate")
        ax.plot(sub["misclass_rate"] * 100, sub["mean_clearance"],
                "s-", label=m, color=COLORS.get(m, "#1f77b4"), linewidth=2)
    ax.set_xlabel("Misclassification Rate (%)")
    ax.set_ylabel("Mean Clearance")
    ax.set_title("R2: Clearance vs Misclassification Rate")
    ax.legend(fontsize=8)

    fig.suptitle("R2: Mode Misclassification (turn_left → constant_velocity)",
                 fontsize=14, y=1.02)
    save(fig, "fig_r2_misclassification")


# ─── R3: Novel Mode Severity ─────────────────────────────────────────────
def fig_r3():
    df = pd.read_csv(OUT / "r3_novel_severity.csv")
    methods = df["method"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for m in methods:
        sub = df[df["method"] == m].sort_values("swerve_amplitude")
        ax.plot(sub["swerve_amplitude"], sub["collision_rate"],
                "o-", label=m, color=COLORS.get(m, "#1f77b4"), linewidth=2)
        ax.fill_between(sub["swerve_amplitude"], sub["coll_ci_lo"], sub["coll_ci_hi"],
                        alpha=0.15, color=COLORS.get(m, "#1f77b4"))
    ax.set_xlabel("Swerve Amplitude")
    ax.set_ylabel("Collision Rate")
    ax.set_title("R3: Collision vs Novel Mode Severity")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    ax = axes[1]
    for m in methods:
        sub = df[df["method"] == m].sort_values("swerve_amplitude")
        ax.plot(sub["swerve_amplitude"], sub["mean_clearance"],
                "s-", label=m, color=COLORS.get(m, "#1f77b4"), linewidth=2)
    ax.set_xlabel("Swerve Amplitude")
    ax.set_ylabel("Mean Clearance")
    ax.set_title("R3: Clearance vs Novel Mode Severity")
    ax.legend(fontsize=8)

    fig.suptitle("R3: Novel Mode Severity Sweep", fontsize=14, y=1.02)
    save(fig, "fig_r3_novel_severity")


# ─── R4: Confusion Pairs ─────────────────────────────────────────────────
def fig_r4():
    df = pd.read_csv(OUT / "r4_confusion_pairs.csv")
    pairs = df["confusion_pair"].unique()
    methods = df["method"].unique()
    x = np.arange(len(pairs))
    w = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["confusion_pair"] == p]["collision_rate"].values[0] for p in pairs]
        lo = [sub[sub["confusion_pair"] == p]["coll_ci_lo"].values[0] for p in pairs]
        hi = [sub[sub["confusion_pair"] == p]["coll_ci_hi"].values[0] for p in pairs]
        err_lo = [v - l for v, l in zip(vals, lo)]
        err_hi = [h - v for v, h in zip(vals, hi)]
        ax.bar(x + i * w, vals, w, yerr=[err_lo, err_hi],
               label=m, color=COLORS.get(m, "#1f77b4"), capsize=3, alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(pairs, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Collision Rate")
    ax.set_title("R4: Collision Rate by Confusion Pair (50% misclass)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    ax = axes[1]
    for i, m in enumerate(methods):
        sub = df[df["method"] == m]
        vals = [sub[sub["confusion_pair"] == p]["mean_clearance"].values[0] for p in pairs]
        ax.bar(x + i * w, vals, w, label=m,
               color=COLORS.get(m, "#1f77b4"), alpha=0.85)
    ax.set_xticks(x + w)
    ax.set_xticklabels(pairs, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Clearance")
    ax.set_title("R4: Mean Clearance by Confusion Pair")
    ax.legend(fontsize=8)

    fig.suptitle("R4: Different Mode Confusion Pairs at 50% Misclassification",
                 fontsize=14, y=1.02)
    save(fig, "fig_r4_confusion_pairs")


if __name__ == "__main__":
    print("Generating robustness figures...")
    fns = {"r1": fig_r1, "r2": fig_r2, "r3": fig_r3, "r4": fig_r4}
    which = sys.argv[1:] if len(sys.argv) > 1 else fns.keys()
    for k in which:
        try:
            fns[k]()
        except Exception as e:
            print(f"  SKIP {k}: {e}")
    print("Done.")
