#!/usr/bin/env python3
"""
Generate publication figures for Q1-Q3 experiments (SH-only variants).

Reads CSVs from paper_figures/ and writes PNGs + a LaTeX summary to paper_figures/.

Figures:
  Q1-a: Collision rate vs scenario budget S          (Exp T)
  Q1-b: Missed-mode rate vs scenario budget S        (Exp T)
  Q2-a: Rare-mode missed fraction vs rare_prob       (Exp V)
  Q2-b: Collision rate vs rare_prob                   (Exp V)
  Q2-c: Coverage baselines comparison                 (Exp X)
  Q3-a: Geometry ablation (ground cost types)         (Exp Y)
  Q3-b: Robustness across environments                (Exp AA)
  Q3-c: Hyperparameter Pareto frontier                (Exp AB)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

DATA_DIR = "paper_figures/"
OUT_DIR = "paper_figures/"

# SH-only variant styling
VARIANT_COLORS = {
    "Base+SH":           "#1f77b4",
    "OT+SH":            "#2ca02c",
    "DRO+SH":           "#ff7f0e",
    "OT+DRO+SH":        "#d62728",
    "OT+DRO+SH+Blend":  "#9467bd",
}
VARIANT_MARKERS = {
    "Base+SH":           "o",
    "OT+SH":            "^",
    "DRO+SH":           "s",
    "OT+DRO+SH":        "D",
    "OT+DRO+SH+Blend":  "P",
}
VARIANT_ORDER = ["Base+SH", "OT+SH", "DRO+SH", "OT+DRO+SH", "OT+DRO+SH+Blend"]
VARIANT_LABELS = {
    "Base+SH":           "SH (baseline)",
    "OT+SH":            "OT+SH",
    "DRO+SH":           "DRO+SH",
    "OT+DRO+SH":        "OT+DRO+SH",
    "OT+DRO+SH+Blend":  "OT+DRO+SH (blended)",
}


def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return None
    return pd.read_csv(path).dropna(how="all")


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ============================================================================
# Q1-a: Collision Rate vs Scenario Budget S  (Exp T)
# ============================================================================

def fig_q1a_collision_vs_s():
    df = load_csv("exp_t_missed_mode_vs_s.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        ax.errorbar(
            sub["num_scenarios"], sub["collision_rate"],
            yerr=[sub["collision_rate"] - sub["ci_lo"],
                  sub["ci_hi"] - sub["collision_rate"]],
            label=VARIANT_LABELS.get(vname, vname),
            color=VARIANT_COLORS.get(vname, "gray"),
            marker=VARIANT_MARKERS.get(vname, "o"),
            capsize=3, linewidth=1.5, markersize=6,
        )
    ax.set_xlabel("Number of Scenarios $S$")
    ax.set_ylabel("Collision Rate")
    ax.set_title("Q1: Collision Rate vs Scenario Budget")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "q1a_collision_vs_s.png")


# ============================================================================
# Q1-b: Missed-Mode Rate vs Scenario Budget S  (Exp T)
# ============================================================================

def fig_q1b_missed_mode_vs_s():
    df = load_csv("exp_t_missed_mode_vs_s.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        ax.plot(
            sub["num_scenarios"], sub["missed_mode_rate"],
            label=VARIANT_LABELS.get(vname, vname),
            color=VARIANT_COLORS.get(vname, "gray"),
            marker=VARIANT_MARKERS.get(vname, "o"),
            linewidth=1.5, markersize=6,
        )
    ax.set_xlabel("Number of Scenarios $S$")
    ax.set_ylabel("Missed-Mode Rate")
    ax.set_title("Q1: Mode Coverage vs Scenario Budget")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "q1b_missed_mode_vs_s.png")


# ============================================================================
# Q2-a: Rare-Mode Missed Fraction vs Rare Probability  (Exp V)
# ============================================================================

def fig_q2a_rare_mode_coverage():
    df = load_csv("exp_v_rare_mode_sweep.csv")
    if df is None:
        return
    # Use 4obs_4class config
    df4 = df[df["obs_config"] == "4obs_4class"] if "obs_config" in df.columns else df

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    for vname in VARIANT_ORDER:
        sub = df4[df4["variant"] == vname].sort_values("rare_prob")
        if sub.empty:
            continue
        color = VARIANT_COLORS.get(vname, "gray")
        marker = VARIANT_MARKERS.get(vname, "o")
        label = VARIANT_LABELS.get(vname, vname)

        ax1.plot(sub["rare_prob"], sub["rare_mode_missed_frac"],
                 label=label, color=color, marker=marker,
                 linewidth=1.5, markersize=6)

        ax2.errorbar(sub["rare_prob"], sub["collision_rate"],
                     yerr=[sub["collision_rate"] - sub["ci_lo"],
                           sub["ci_hi"] - sub["collision_rate"]],
                     label=label, color=color, marker=marker,
                     capsize=3, linewidth=1.5, markersize=6)

    ax1.set_xlabel("Rare-Mode Probability")
    ax1.set_ylabel("Rare-Mode Miss Fraction")
    ax1.set_title("Q2: Rare-Mode Coverage")
    ax1.legend(loc="upper right")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Rare-Mode Probability")
    ax2.set_ylabel("Collision Rate")
    ax2.set_title("Q2: Collision Rate under Rare Modes")
    ax2.legend(loc="upper right")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "q2a_rare_mode_sweep.png")


# ============================================================================
# Q2-b: Coverage Baselines Comparison  (Exp X)
# ============================================================================

def fig_q2b_baselines():
    df = load_csv("exp_x_baselines_rare_mode.csv")
    if df is None:
        return
    # Group by baseline at rare_prob=0.10 (medium difficulty)
    df10 = df[df["rare_prob"].between(0.09, 0.11)]
    if df10.empty:
        df10 = df.groupby("baseline").mean(numeric_only=True).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    baselines = df10["baseline"].unique()
    x = np.arange(len(baselines))
    width = 0.35

    colors = [VARIANT_COLORS.get(b, "#888888") for b in baselines]

    ax1.bar(x, df10["collision_rate"], width, color=colors)
    if "ci_lo" in df10.columns:
        yerr = [df10["collision_rate"] - df10["ci_lo"],
                df10["ci_hi"] - df10["collision_rate"]]
        ax1.errorbar(x, df10["collision_rate"], yerr=yerr,
                     fmt="none", ecolor="black", capsize=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(baselines, rotation=30, ha="right")
    ax1.set_ylabel("Collision Rate")
    ax1.set_title("Q2: Collision Rate at rare_p=0.10")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, df10["missed_mode_rate"], width, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(baselines, rotation=30, ha="right")
    ax2.set_ylabel("Missed-Mode Rate")
    ax2.set_title("Q2: Mode Coverage at rare_p=0.10")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, "q2b_baselines_comparison.png")


# ============================================================================
# Q3-a: Geometry Ablation  (Exp Y)
# ============================================================================

def fig_q3a_geometry_ablation():
    df = load_csv("exp_y_geometry_ablation.csv")
    if df is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # For OT variants, show different ground costs
    ot_variants = [v for v in df["variant"].unique() if "OT" in v]
    non_ot_variants = [v for v in df["variant"].unique() if "OT" not in v]

    # Left panel: collision rate by ground cost for OT variants
    cost_types = df["ground_cost"].unique()
    x = np.arange(len(cost_types))
    width = 0.8 / max(len(ot_variants), 1)

    for i, vname in enumerate(sorted(ot_variants)):
        sub = df[df["variant"] == vname]
        vals = []
        ci_lo_vals = []
        ci_hi_vals = []
        for ct in cost_types:
            row = sub[sub["ground_cost"] == ct]
            if len(row) > 0:
                vals.append(row["collision_rate"].values[0])
                ci_lo_vals.append(row["ci_lo"].values[0])
                ci_hi_vals.append(row["ci_hi"].values[0])
            else:
                vals.append(0)
                ci_lo_vals.append(0)
                ci_hi_vals.append(0)
        vals = np.array(vals)
        ci_lo_vals = np.array(ci_lo_vals)
        ci_hi_vals = np.array(ci_hi_vals)
        offset = (i - len(ot_variants)/2 + 0.5) * width
        color = VARIANT_COLORS.get(vname, f"C{i}")
        ax1.bar(x + offset, vals, width, label=VARIANT_LABELS.get(vname, vname),
                color=color, alpha=0.8)
        ax1.errorbar(x + offset, vals,
                     yerr=[vals - ci_lo_vals, ci_hi_vals - vals],
                     fmt="none", ecolor="black", capsize=2)

    # Add non-OT baselines as horizontal lines
    for vname in non_ot_variants:
        sub = df[df["variant"] == vname]
        if not sub.empty:
            cr = sub["collision_rate"].values[0]
            color = VARIANT_COLORS.get(vname, "gray")
            ax1.axhline(cr, color=color, linestyle="--", alpha=0.7,
                       label=f"{VARIANT_LABELS.get(vname, vname)} (no OT)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(cost_types, rotation=30, ha="right")
    ax1.set_ylabel("Collision Rate")
    ax1.set_title("Q3: Collision Rate by Ground Cost")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right panel: missed-mode rate by ground cost for OT variants
    for i, vname in enumerate(sorted(ot_variants)):
        sub = df[df["variant"] == vname]
        vals = []
        for ct in cost_types:
            row = sub[sub["ground_cost"] == ct]
            vals.append(row["missed_mode_rate"].values[0] if len(row) > 0 else 0)
        offset = (i - len(ot_variants)/2 + 0.5) * width
        color = VARIANT_COLORS.get(vname, f"C{i}")
        ax2.bar(x + offset, vals, width, label=VARIANT_LABELS.get(vname, vname),
                color=color, alpha=0.8)

    for vname in non_ot_variants:
        sub = df[df["variant"] == vname]
        if not sub.empty:
            mmr = sub["missed_mode_rate"].values[0]
            color = VARIANT_COLORS.get(vname, "gray")
            ax2.axhline(mmr, color=color, linestyle="--", alpha=0.7,
                       label=f"{VARIANT_LABELS.get(vname, vname)} (no OT)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(cost_types, rotation=30, ha="right")
    ax2.set_ylabel("Missed-Mode Rate")
    ax2.set_title("Q3: Mode Coverage by Ground Cost")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, "q3a_geometry_ablation.png")


# ============================================================================
# Q3-b: Robustness Across Environments  (Exp AA)
# ============================================================================

def fig_q3b_environment_robustness():
    df = load_csv("exp_aa_robustness_per_seed.csv")
    if df is None:
        return

    # Aggregate per (environment, variant)
    agg = df.groupby(["environment", "variant"]).agg(
        collision_rate=("collision", "mean"),
        num_rollouts=("collision", "count"),
        avg_progress=("progress", "mean"),
        avg_clearance=("min_clearance", "mean"),
        mean_solve_ms=("mean_solve_ms", "mean"),
    ).reset_index()

    envs = agg["environment"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: collision rate by environment
    ax = axes[0]
    x = np.arange(len(envs))
    width = 0.8 / max(len(VARIANT_ORDER), 1)

    for i, vname in enumerate(VARIANT_ORDER):
        vals = []
        for env in envs:
            row = agg[(agg["environment"] == env) & (agg["variant"] == vname)]
            vals.append(row["collision_rate"].values[0] if len(row) > 0 else 0)
        offset = (i - len(VARIANT_ORDER)/2 + 0.5) * width
        color = VARIANT_COLORS.get(vname, f"C{i}")
        ax.bar(x + offset, vals, width,
               label=VARIANT_LABELS.get(vname, vname), color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15)
    ax.set_ylabel("Collision Rate")
    ax.set_title("Q3: Collision Rate by Environment")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: progress by environment
    ax = axes[1]
    for i, vname in enumerate(VARIANT_ORDER):
        vals = []
        for env in envs:
            row = agg[(agg["environment"] == env) & (agg["variant"] == vname)]
            vals.append(row["avg_progress"].values[0] if len(row) > 0 else 0)
        offset = (i - len(VARIANT_ORDER)/2 + 0.5) * width
        color = VARIANT_COLORS.get(vname, f"C{i}")
        ax.bar(x + offset, vals, width,
               label=VARIANT_LABELS.get(vname, vname), color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=15)
    ax.set_ylabel("Average Path Progress")
    ax.set_title("Q3: Progress by Environment")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, "q3b_environment_robustness.png")


# ============================================================================
# Q3-c: Hyperparameter Pareto Frontier  (Exp AB)
# ============================================================================

def fig_q3c_pareto_frontier():
    df = load_csv("exp_ab_pareto_frontier.csv")
    if df is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    variants = df["variant"].unique()
    for vname in variants:
        sub = df[df["variant"] == vname]
        color = VARIANT_COLORS.get(vname, "gray")
        label = VARIANT_LABELS.get(vname, vname)

        # Left: collision rate vs missed_mode_rate (Pareto)
        ax1.scatter(sub["missed_mode_rate"], sub["collision_rate"],
                   c=color, marker=VARIANT_MARKERS.get(vname, "o"),
                   s=50, label=label, alpha=0.8, edgecolors="black", linewidth=0.5)

        # Annotate with epsilon values
        for _, row in sub.iterrows():
            ax1.annotate(f"e={row['epsilon']:.2f}",
                        (row["missed_mode_rate"], row["collision_rate"]),
                        fontsize=6, alpha=0.6,
                        textcoords="offset points", xytext=(3, 3))

    ax1.set_xlabel("Missed-Mode Rate")
    ax1.set_ylabel("Collision Rate")
    ax1.set_title("Q3: Safety vs Coverage Pareto")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: collision rate vs epsilon (sensitivity)
    for vname in variants:
        sub = df[df["variant"] == vname]
        # Average over uncertainty_scale for each epsilon
        by_eps = sub.groupby("epsilon").agg(
            collision_rate=("collision_rate", "mean"),
            ci_lo=("ci_lo", "mean"),
            ci_hi=("ci_hi", "mean"),
        ).reset_index()
        color = VARIANT_COLORS.get(vname, "gray")
        label = VARIANT_LABELS.get(vname, vname)
        ax2.plot(by_eps["epsilon"], by_eps["collision_rate"],
                color=color, marker=VARIANT_MARKERS.get(vname, "o"),
                label=label, linewidth=1.5, markersize=6)

    ax2.set_xlabel("Sinkhorn $\\epsilon$")
    ax2.set_ylabel("Collision Rate")
    ax2.set_title("Q3: Sensitivity to OT Regularization")
    ax2.set_xscale("log")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "q3c_pareto_frontier.png")


# ============================================================================
# Summary Table (LaTeX)
# ============================================================================

def generate_summary_table():
    """Write a LaTeX table summarizing key metrics across Q1-Q3."""
    out_path = os.path.join(OUT_DIR, "q1q2q3_summary_table.tex")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary of Q1--Q3 results (SH variants only).}")
    lines.append(r"\label{tab:q1q2q3}")
    lines.append(r"\begin{tabular}{l|cccc}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Base+SH & OT+SH & DRO+SH & OT+DRO+SH \\")
    lines.append(r"\midrule")

    # Q1: From Exp T at S=40
    df_t = load_csv("exp_t_missed_mode_vs_s.csv")
    if df_t is not None:
        t40 = df_t[df_t["num_scenarios"] == 40]
        vals = []
        for vname in VARIANT_ORDER:
            row = t40[t40["variant"] == vname]
            if len(row) > 0:
                cr = row["collision_rate"].values[0]
                mmr = row["missed_mode_rate"].values[0]
                vals.append(f"{cr:.3f} / {mmr:.3f}")
            else:
                vals.append("--")
        lines.append(f"Coll / Miss (S=40) & {' & '.join(vals)} \\\\")

    # Q2: From Exp V at rare_p=0.10
    df_v = load_csv("exp_v_rare_mode_sweep.csv")
    if df_v is not None:
        v10 = df_v[(df_v["rare_prob"].between(0.09, 0.11))]
        if "obs_config" in v10.columns:
            v10 = v10[v10["obs_config"] == "4obs_4class"]
        vals = []
        for vname in VARIANT_ORDER:
            row = v10[v10["variant"] == vname]
            if len(row) > 0:
                rmf = row["rare_mode_missed_frac"].values[0]
                vals.append(f"{rmf:.3f}")
            else:
                vals.append("--")
        lines.append(f"Rare miss (p=0.10) & {' & '.join(vals)} \\\\")

    # Q3: From Exp AA (average across environments)
    df_aa = load_csv("exp_aa_robustness_per_seed.csv")
    if df_aa is not None:
        agg = df_aa.groupby("variant").agg(
            collision_rate=("collision", "mean"),
        ).reset_index()
        vals = []
        for vname in VARIANT_ORDER:
            row = agg[agg["variant"] == vname]
            if len(row) > 0:
                cr = row["collision_rate"].values[0]
                vals.append(f"{cr:.3f}")
            else:
                vals.append("--")
        lines.append(f"Coll (avg envs) & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {out_path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating Q1 figures (Exp T)...")
    fig_q1a_collision_vs_s()
    fig_q1b_missed_mode_vs_s()

    print("Generating Q2 figures (Exp V, X)...")
    fig_q2a_rare_mode_coverage()
    fig_q2b_baselines()

    print("Generating Q3 figures (Exp Y, AA, AB)...")
    fig_q3a_geometry_ablation()
    fig_q3b_environment_robustness()
    fig_q3c_pareto_frontier()

    print("Generating summary table...")
    generate_summary_table()

    print("\nDone. All figures in paper_figures/")
