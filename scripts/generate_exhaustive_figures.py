#!/usr/bin/env python3
"""Generate figures for exhaustive suites F1, F9, G1, G3, G6.

Each suite has its own figures/<suite>/ directory with:
  - Primary comparison figure (the main finding)
  - Sensitivity panel (showing the finding holds across swept parameters)
"""

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Consistent colors across all suites
COLORS = {
    "Base":             "#999999",
    "WDRO-sampling":    "#FF7F00",
    "WDRO-inject-K1":   "#2166AC",
    "WDRO-inject-K2":   "#4393C3",
    "TopRisk-K1":       "#B2182B",
    "TopRisk-K2":       "#E08080",
    "DiverseRisk-K1":   "#4DAF4A",
    "Softmax-tau5":     "#984EA3",
}

def save(fig, stem, d):
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {d / stem}")


# ─── F1 ────────────────────────────────────────────────────────────────
def fig_f1():
    d = pathlib.Path("figures/f1")
    df = pd.read_csv(d / "f1_high_n_oncoming.csv")
    speeds = sorted(df["obstacle_speed"].unique())
    methods = df["method"].unique()

    fig, axes = plt.subplots(1, len(speeds), figsize=(7 * len(speeds), 6), sharey=True)
    if len(speeds) == 1:
        axes = [axes]

    for ax, spd in zip(axes, speeds):
        sub = df[df["obstacle_speed"] == spd].sort_values("collision_rate")
        y = np.arange(len(sub))
        colors = [COLORS.get(m, "#1f77b4") for m in sub["method"]]
        xerr_lo = sub["collision_rate"] - sub["coll_ci_lo"]
        xerr_hi = sub["coll_ci_hi"] - sub["collision_rate"]
        ax.barh(y, sub["collision_rate"], xerr=[xerr_lo, xerr_hi],
                color=colors, capsize=4, alpha=0.85, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["method"], fontsize=9)
        ax.set_xlabel("Collision Rate")
        ax.set_title(f"Obstacle Speed = {spd:.1f} m/s  (N=2000)")
        ax.set_xlim(0, max(0.15, sub["coll_ci_hi"].max() * 1.2))
        # Add value labels
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(row["coll_ci_hi"] + 0.003, i,
                    f'{row["collision_rate"]*100:.1f}%', va="center", fontsize=8)

    fig.suptitle("F1: High-N Variant Differentiation — Method Ranking by Speed",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "fig_f1_method_ranking", d)

    # Clearance comparison
    fig, axes = plt.subplots(1, len(speeds), figsize=(7 * len(speeds), 5), sharey=True)
    if len(speeds) == 1:
        axes = [axes]
    for ax, spd in zip(axes, speeds):
        sub = df[df["obstacle_speed"] == spd]
        y = np.arange(len(sub))
        colors = [COLORS.get(m, "#1f77b4") for m in sub["method"]]
        ax.barh(y, sub["mean_clearance"], color=colors, alpha=0.85, height=0.7)
        ax.barh(y, sub["p5_clearance"], color=colors, alpha=0.4, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["method"], fontsize=9)
        ax.set_xlabel("Clearance (solid=mean, faint=p5)")
        ax.set_title(f"Speed = {spd:.1f} m/s")
    fig.suptitle("F1: Clearance by Method and Speed", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "fig_f1_clearance", d)


# ─── F9 ────────────────────────────────────────────────────────────────
def fig_f9():
    d = pathlib.Path("figures/f9")
    df = pd.read_csv(d / "f9_multi_obs_sweet_spot.csv")
    paths = df["path"].unique()
    methods = df["method"].unique()
    obs_counts = sorted(df["num_obstacles"].unique())

    fig, axes = plt.subplots(1, len(paths), figsize=(7 * len(paths), 5), sharey=True)
    if len(paths) == 1:
        axes = [axes]

    for ax, pth in zip(axes, paths):
        sub = df[df["path"] == pth]
        x = np.arange(len(obs_counts))
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            msub = sub[sub["method"] == m].sort_values("num_obstacles")
            vals = msub["collision_rate"].values
            lo = msub["collision_rate"].values - msub["coll_ci_lo"].values
            hi = msub["coll_ci_hi"].values - msub["collision_rate"].values
            ax.bar(x + i * w, vals, w, yerr=[lo, hi],
                   label=m, color=COLORS.get(m, "#1f77b4"), capsize=3, alpha=0.85)
        ax.set_xticks(x + w * (len(methods) - 1) / 2)
        ax.set_xticklabels([str(n) for n in obs_counts])
        ax.set_xlabel("Number of Obstacles")
        ax.set_ylabel("Collision Rate")
        ax.set_title(f"Path: {pth}")
        if ax == axes[0]:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("F9: Multi-Obstacle Scaling at Sweet Spot (v=1.0, N=1000)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "fig_f9_scaling", d)

    # DRO benefit (Base - WDRO-inject-K1) across conditions
    fig, ax = plt.subplots(figsize=(8, 5))
    for pth in paths:
        sub = df[df["path"] == pth]
        benefits = []
        for nobs in obs_counts:
            base = sub[(sub["method"] == "Base") & (sub["num_obstacles"] == nobs)]["collision_rate"].values
            dro = sub[(sub["method"] == "WDRO-inject-K1") & (sub["num_obstacles"] == nobs)]["collision_rate"].values
            if len(base) > 0 and len(dro) > 0:
                benefits.append((base[0] - dro[0]) * 100)
            else:
                benefits.append(0)
        ax.plot(obs_counts, benefits, "o-", label=pth, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Obstacles")
    ax.set_ylabel("DRO Benefit (pp collision reduction)")
    ax.set_title("F9: WDRO-inject-K1 Benefit Over Base by Path")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig_f9_dro_benefit", d)


# ─── G1 ────────────────────────────────────────────────────────────────
def fig_g1():
    d = pathlib.Path("figures/g1")
    df = pd.read_csv(d / "g1_path_geometry.csv")
    speeds = sorted(df["obstacle_speed"].unique())
    paths = df["path"].unique()
    methods = df["method"].unique()

    for spd in speeds:
        sub = df[df["obstacle_speed"] == spd]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        x = np.arange(len(paths))
        w = 0.8 / len(methods)

        ax = axes[0]
        for i, m in enumerate(methods):
            msub = sub[sub["method"] == m]
            vals = [msub[msub["path"] == p]["collision_rate"].values[0] for p in paths]
            lo = [msub[msub["path"] == p]["coll_ci_lo"].values[0] for p in paths]
            hi = [msub[msub["path"] == p]["coll_ci_hi"].values[0] for p in paths]
            elo = [v - l for v, l in zip(vals, lo)]
            ehi = [h - v for v, h in zip(vals, hi)]
            ax.bar(x + i * w, vals, w, yerr=[elo, ehi],
                   label=m, color=COLORS.get(m, "#1f77b4"), capsize=2, alpha=0.85)
        ax.set_xticks(x + w * (len(methods) - 1) / 2)
        ax.set_xticklabels(paths, rotation=20, ha="right")
        ax.set_ylabel("Collision Rate")
        ax.set_title(f"Collision Rate (v={spd:.1f})")
        ax.legend(fontsize=7)

        ax = axes[1]
        for i, m in enumerate(methods):
            msub = sub[sub["method"] == m]
            vals = [msub[msub["path"] == p]["mean_clearance"].values[0] for p in paths]
            ax.bar(x + i * w, vals, w, label=m,
                   color=COLORS.get(m, "#1f77b4"), alpha=0.85)
        ax.set_xticks(x + w * (len(methods) - 1) / 2)
        ax.set_xticklabels(paths, rotation=20, ha="right")
        ax.set_ylabel("Mean Clearance")
        ax.set_title(f"Clearance (v={spd:.1f})")
        ax.legend(fontsize=7)

        fig.suptitle(f"G1: Path Geometry — Obstacle Speed {spd:.1f} m/s (N=500)",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save(fig, f"fig_g1_v{spd:.1f}", d)

    # Sensitivity: collision rate Base vs best-DRO across speeds
    fig, ax = plt.subplots(figsize=(10, 5))
    for spd in speeds:
        sub = df[df["obstacle_speed"] == spd]
        for m in ["Base", "WDRO-inject-K1", "TopRisk-K1"]:
            msub = sub[sub["method"] == m]
            vals = [msub[msub["path"] == p]["collision_rate"].values[0] for p in paths]
            style = "-" if spd == speeds[0] else "--"
            ax.plot(paths, vals, f"o{style}", label=f"{m} v={spd:.1f}",
                    color=COLORS.get(m, "#1f77b4"), linewidth=2)
    ax.set_ylabel("Collision Rate")
    ax.set_title("G1: Key Methods Across Paths and Speeds")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    save(fig, "fig_g1_sensitivity", d)


# ─── G3 ────────────────────────────────────────────────────────────────
def fig_g3():
    d = pathlib.Path("figures/g3")
    df = pd.read_csv(d / "g3_switch_dynamics.csv")
    paths = df["path"].unique()
    methods = df["method"].unique()

    for pth in paths:
        sub = df[df["path"] == pth]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for m in methods:
            msub = sub[sub["method"] == m].sort_values("switch_prob")
            ax.plot(msub["switch_prob"] * 100, msub["collision_rate"],
                    "o-", label=m, color=COLORS.get(m, "#1f77b4"), linewidth=2)
            ax.fill_between(msub["switch_prob"] * 100,
                            msub["coll_ci_lo"], msub["coll_ci_hi"],
                            alpha=0.12, color=COLORS.get(m, "#1f77b4"))
        ax.set_xlabel("Switch Probability (%)")
        ax.set_ylabel("Collision Rate")
        ax.set_title(f"Collision Rate ({pth})")
        ax.legend(fontsize=8)

        ax = axes[1]
        for m in methods:
            msub = sub[sub["method"] == m].sort_values("switch_prob")
            ax.plot(msub["switch_prob"] * 100, msub["mean_clearance"],
                    "s-", label=m, color=COLORS.get(m, "#1f77b4"), linewidth=2)
        ax.set_xlabel("Switch Probability (%)")
        ax.set_ylabel("Mean Clearance")
        ax.set_title(f"Clearance ({pth})")
        ax.legend(fontsize=8)

        fig.suptitle(f"G3: Switching Dynamics — {pth} (N=500)",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save(fig, f"fig_g3_{pth.lower().replace('-','_')}", d)

    # Sensitivity: overlay paths for key methods
    fig, ax = plt.subplots(figsize=(10, 5))
    for pth in paths:
        sub = df[df["path"] == pth]
        for m in ["Base", "WDRO-inject-K1", "TopRisk-K1"]:
            msub = sub[sub["method"] == m].sort_values("switch_prob")
            style = "-" if pth == paths[0] else "--"
            ax.plot(msub["switch_prob"] * 100, msub["collision_rate"],
                    f"o{style}", label=f"{m} ({pth})",
                    color=COLORS.get(m, "#1f77b4"), linewidth=2)
    ax.set_xlabel("Switch Probability (%)")
    ax.set_ylabel("Collision Rate")
    ax.set_title("G3: Key Methods Across Paths")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    save(fig, "fig_g3_sensitivity", d)


# ─── G6 ────────────────────────────────────────────────────────────────
def fig_g6():
    d = pathlib.Path("figures/g6")
    df = pd.read_csv(d / "g6_challenging_envs.csv")
    switch_probs = sorted(df["switch_prob"].unique())
    challenges = df["challenge"].unique()
    methods = df["method"].unique()

    for sp in switch_probs:
        sub = df[df["switch_prob"] == sp]
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        x = np.arange(len(challenges))
        w = 0.8 / len(methods)

        ax = axes[0]
        for i, m in enumerate(methods):
            msub = sub[sub["method"] == m]
            vals, elo, ehi = [], [], []
            for ch in challenges:
                row = msub[msub["challenge"] == ch]
                if len(row) == 0:
                    vals.append(0); elo.append(0); ehi.append(0)
                    continue
                row = row.iloc[0]
                vals.append(row["collision_rate"])
                elo.append(row["collision_rate"] - row["coll_ci_lo"])
                ehi.append(row["coll_ci_hi"] - row["collision_rate"])
            ax.bar(x + i * w, vals, w, yerr=[elo, ehi],
                   label=m, color=COLORS.get(m, "#1f77b4"), capsize=2, alpha=0.85)
        ax.set_xticks(x + w * (len(methods) - 1) / 2)
        ax.set_xticklabels(challenges, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Collision Rate")
        ax.set_title(f"Collision (switch_prob={sp:.1f})")
        ax.legend(fontsize=6, ncol=2)

        ax = axes[1]
        for i, m in enumerate(methods):
            msub = sub[sub["method"] == m]
            vals = []
            for ch in challenges:
                row = msub[msub["challenge"] == ch]
                vals.append(row["mean_clearance"].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * w, vals, w, label=m,
                   color=COLORS.get(m, "#1f77b4"), alpha=0.85)
        ax.set_xticks(x + w * (len(methods) - 1) / 2)
        ax.set_xticklabels(challenges, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean Clearance")
        ax.set_title(f"Clearance (switch_prob={sp:.1f})")
        ax.legend(fontsize=6, ncol=2)

        fig.suptitle(f"G6: Challenging Environments — switch_prob={sp:.1f} (N=500)",
                     fontsize=14, y=1.02)
        fig.tight_layout()
        save(fig, f"fig_g6_sp{sp:.1f}", d)

    # Sensitivity: DRO benefit across switch probs
    fig, ax = plt.subplots(figsize=(12, 5))
    for sp in switch_probs:
        sub = df[df["switch_prob"] == sp]
        benefits = []
        for ch in challenges:
            base = sub[(sub["method"] == "Base") & (sub["challenge"] == ch)]["collision_rate"].values
            dro = sub[(sub["method"] == "WDRO-inject-K1") & (sub["challenge"] == ch)]["collision_rate"].values
            if len(base) > 0 and len(dro) > 0:
                benefits.append((base[0] - dro[0]) * 100)
            else:
                benefits.append(0)
        ax.bar(np.arange(len(challenges)) + (switch_probs.index(sp)) * 0.35,
               benefits, 0.35, label=f"sp={sp:.1f}", alpha=0.8)
    ax.set_xticks(np.arange(len(challenges)) + 0.175)
    ax.set_xticklabels(challenges, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("DRO Benefit (pp)")
    ax.set_title("G6: WDRO-inject-K1 Benefit Over Base by Challenge & Switch Prob")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "fig_g6_sensitivity", d)


# ─── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating exhaustive suite figures...")
    fns = {"f1": fig_f1, "f9": fig_f9, "g1": fig_g1, "g3": fig_g3, "g6": fig_g6}
    which = sys.argv[1:] if len(sys.argv) > 1 else list(fns.keys())
    for k in which:
        try:
            print(f"\n--- {k.upper()} ---")
            fns[k]()
        except Exception as e:
            print(f"  SKIP {k}: {e}")
    print("\nDone.")
