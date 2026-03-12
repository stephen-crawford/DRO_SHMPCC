#!/usr/bin/env python3
"""
Generate figures for Trajectory-Level W2 DRO vs SHMPCC Nominal Baseline.

Reads:  paper_figures/trajectory_dro_results.csv
        paper_figures/trajectory_dro_summary.csv
Writes: traj_paper_figures/fig_*.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_DIR = "paper_figures"
OUTPUT_DIR = "traj_paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(INPUT_DIR, "trajectory_dro_results.csv")
SUMMARY_CSV = os.path.join(INPUT_DIR, "trajectory_dro_summary.csv")

# Consistent strategy colors + labels
STRATEGY_ORDER = [
    "SHMPCC-Nom+SH",
    "Traj-W2-DRO+SH",
    "Mode-DRO(inj)+SH",
    "OT+Mode-DRO+SH",
]
STRATEGY_LABELS = {
    "SHMPCC-Nom+SH":     "SHMPCC Nominal\n(Baseline)",
    "Traj-W2-DRO+SH":    "Traj-W2-DRO\n(Proposed)",
    "Mode-DRO(inj)+SH":  "Mode-DRO\n(Injection)",
    "OT+Mode-DRO+SH":    "OT + Mode-DRO",
}
STRATEGY_COLORS = {
    "SHMPCC-Nom+SH":     "#7f8c8d",   # grey
    "Traj-W2-DRO+SH":    "#2ecc71",   # green
    "Mode-DRO(inj)+SH":  "#3498db",   # blue
    "OT+Mode-DRO+SH":    "#e74c3c",   # red
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(RESULTS_CSV)
summary = pd.read_csv(SUMMARY_CSV)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return p_hat, max(0, center - half), min(1, center + half)


# ============================================================================
# Figure 1: Collision Rate Bar Chart with Wilson CIs
# ============================================================================
def fig_collision_rate():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(STRATEGY_ORDER))
    width = 0.55

    rates, ci_los, ci_his = [], [], []
    for s in STRATEGY_ORDER:
        row = summary[summary["strategy"] == s].iloc[0]
        rates.append(row["collision_rate"] * 100)
        ci_los.append(row["collision_ci_lo"] * 100)
        ci_his.append(row["collision_ci_hi"] * 100)

    colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
    bars = ax.bar(x, rates, width, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)

    # Error bars (Wilson CI)
    yerr_lo = [r - lo for r, lo in zip(rates, ci_los)]
    yerr_hi = [hi - r for r, hi in zip(rates, ci_his)]
    ax.errorbar(x, rates, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black",
                capsize=5, capthick=1.5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=9)
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Collision Rate Comparison (2000 rollouts, 95% Wilson CI)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    # Annotate rates
    for i, (r, hi) in enumerate(zip(rates, ci_his)):
        ax.text(i, hi + 0.05, f"{r:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_collision_rate.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_collision_rate.pdf")


# ============================================================================
# Figure 2: Mode Coverage Bar Chart with Wilson CIs
# ============================================================================
def fig_mode_coverage():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(STRATEGY_ORDER))
    width = 0.55

    rates, ci_los, ci_his = [], [], []
    for s in STRATEGY_ORDER:
        row = summary[summary["strategy"] == s].iloc[0]
        rates.append(row["mode_coverage"] * 100)
        ci_los.append(row["mode_cov_ci_lo"] * 100)
        ci_his.append(row["mode_cov_ci_hi"] * 100)

    colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
    bars = ax.bar(x, rates, width, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)

    yerr_lo = [r - lo for r, lo in zip(rates, ci_los)]
    yerr_hi = [hi - r for r, hi in zip(rates, ci_his)]
    ax.errorbar(x, rates, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black",
                capsize=5, capthick=1.5, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER], fontsize=9)
    ax.set_ylabel("Mode Coverage (%)")
    ax.set_title("Mode Coverage Comparison (2000 rollouts, 95% Wilson CI)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for i, (r, hi) in enumerate(zip(rates, ci_his)):
        ax.text(i, min(hi + 0.5, 104), f"{r:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_mode_coverage.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_mode_coverage.pdf")


# ============================================================================
# Figure 3: Clearance Distribution (Violin + Box)
# ============================================================================
def fig_clearance_distribution():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    data = []
    labels = []
    colors = []
    for s in STRATEGY_ORDER:
        vals = df[df["strategy"] == s]["clearance_5pct"].values
        data.append(vals)
        labels.append(STRATEGY_LABELS[s])
        colors.append(STRATEGY_COLORS[s])

    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True,
                          showextrema=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.5)
    parts["cmeans"].set_color("black")

    bp = ax.boxplot(data, positions=range(len(data)), widths=0.25,
                    patch_artist=True, showfliers=False, zorder=3)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("black")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("5th Percentile Clearance (m)")
    ax.set_title("Clearance Distribution (5th %ile per rollout)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_clearance_distribution.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_clearance_distribution.pdf")


# ============================================================================
# Figure 4: Solve Time Distribution
# ============================================================================
def fig_solve_time():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    y = np.arange(len(STRATEGY_ORDER))
    solve_means = []
    labels = []
    colors_list = []
    for s in STRATEGY_ORDER:
        row = summary[summary["strategy"] == s].iloc[0]
        solve_means.append(row["mean_solve_ms"])
        labels.append(STRATEGY_LABELS[s].replace("\n", " "))
        colors_list.append(STRATEGY_COLORS[s])

    bars = ax.barh(y, solve_means, height=0.5, color=colors_list,
                   edgecolor="black", linewidth=0.7, alpha=0.85)
    for i, v in enumerate(solve_means):
        ax.text(v * 1.05, i, f"{v:.2f} ms", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean Solve Time (ms)")
    ax.set_xscale("log")
    ax.set_title("Mean Solve Time Comparison (log scale)")
    ax.grid(axis="x", alpha=0.3, which="both")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_solve_time.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_solve_time.pdf")


# ============================================================================
# Figure 5: Safety vs Coverage Pareto Plot
# ============================================================================
def fig_safety_vs_coverage():
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for s in STRATEGY_ORDER:
        row = summary[summary["strategy"] == s].iloc[0]
        coll = row["collision_rate"] * 100
        cov = row["mode_coverage"] * 100

        coll_lo = row["collision_ci_lo"] * 100
        coll_hi = row["collision_ci_hi"] * 100
        cov_lo = row["mode_cov_ci_lo"] * 100
        cov_hi = row["mode_cov_ci_hi"] * 100

        color = STRATEGY_COLORS[s]
        label = STRATEGY_LABELS[s].replace("\n", " ")

        ax.errorbar(cov, coll,
                    xerr=[[cov - cov_lo], [cov_hi - cov]],
                    yerr=[[coll - coll_lo], [coll_hi - coll]],
                    fmt="o", color=color, markersize=10, capsize=4,
                    capthick=1.2, linewidth=1.2, label=label, zorder=5)

    ax.set_xlabel("Mode Coverage (%)")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Safety vs Coverage Trade-off (95% CIs)")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.3)

    # Annotate ideal corner
    ax.annotate("Ideal\n(low coll, high cov)", xy=(95, 0.1),
                fontsize=8, ha="center", color="gray", style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_safety_vs_coverage.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_safety_vs_coverage.pdf")


# ============================================================================
# Figure 6: Combined Overview (2x2 subplots)
# ============================================================================
def fig_combined_overview():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    x = np.arange(len(STRATEGY_ORDER))
    width = 0.5
    colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
    xlabels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]

    # (0,0) Collision Rate
    ax = axes[0, 0]
    rates, ci_los, ci_his = [], [], []
    for s in STRATEGY_ORDER:
        row = summary[summary["strategy"] == s].iloc[0]
        rates.append(row["collision_rate"] * 100)
        ci_los.append(row["collision_ci_lo"] * 100)
        ci_his.append(row["collision_ci_hi"] * 100)
    ax.bar(x, rates, width, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)
    yerr_lo = [r - lo for r, lo in zip(rates, ci_los)]
    yerr_hi = [hi - r for r, hi in zip(rates, ci_his)]
    ax.errorbar(x, rates, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black",
                capsize=4, capthick=1.2, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("(a) Collision Rate")
    ax.grid(axis="y", alpha=0.3)
    for i, (r, hi) in enumerate(zip(rates, ci_his)):
        ax.text(i, hi + 0.03, f"{r:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # (0,1) Mode Coverage
    ax = axes[0, 1]
    rates, ci_los, ci_his = [], [], []
    for s in STRATEGY_ORDER:
        row = summary[summary["strategy"] == s].iloc[0]
        rates.append(row["mode_coverage"] * 100)
        ci_los.append(row["mode_cov_ci_lo"] * 100)
        ci_his.append(row["mode_cov_ci_hi"] * 100)
    ax.bar(x, rates, width, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)
    yerr_lo = [r - lo for r, lo in zip(rates, ci_los)]
    yerr_hi = [hi - r for r, hi in zip(rates, ci_his)]
    ax.errorbar(x, rates, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="black",
                capsize=4, capthick=1.2, linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Mode Coverage (%)")
    ax.set_title("(b) Mode Coverage")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    for i, (r, hi) in enumerate(zip(rates, ci_his)):
        ax.text(i, min(hi + 0.3, 104), f"{r:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # (1,0) Clearance 5th %ile
    ax = axes[1, 0]
    data_clear = []
    for s in STRATEGY_ORDER:
        data_clear.append(df[df["strategy"] == s]["clearance_5pct"].values)
    bp = ax.boxplot(data_clear, positions=x, widths=0.45,
                    patch_artist=True, showfliers=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("black")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("5th %ile Clearance (m)")
    ax.set_title("(c) Minimum Clearance")
    ax.grid(axis="y", alpha=0.3)

    # (1,1) Solve Time — use bar chart of means with log scale
    ax = axes[1, 1]
    solve_means = []
    for s in STRATEGY_ORDER:
        row = summary[summary["strategy"] == s].iloc[0]
        solve_means.append(row["mean_solve_ms"])
    ax.bar(x, solve_means, width, color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)
    for i, v in enumerate(solve_means):
        ax.text(i, v * 1.08, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Mean Solve Time (ms, log)")
    ax.set_title("(d) Solve Time")
    ax.grid(axis="y", alpha=0.3, which="both")

    fig.suptitle("Trajectory-Level W2 DRO vs SHMPCC Baselines (N=2000)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_combined_overview.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_combined_overview.pdf")


# ============================================================================
# Figure 7: Convergence Plot (collision rate CI over rollouts)
# ============================================================================
def fig_convergence():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    checkpoints = list(range(50, 2001, 50))

    # Panel (a): Collision rate convergence
    ax = axes[0]
    for s in STRATEGY_ORDER:
        sdf = df[df["strategy"] == s]
        coll_vals = sdf["collision"].values
        means, lo_list, hi_list = [], [], []
        for cp in checkpoints:
            n = min(cp, len(coll_vals))
            k = int(coll_vals[:n].sum())
            p, lo, hi = wilson_ci(k, n)
            means.append(p * 100)
            lo_list.append(lo * 100)
            hi_list.append(hi * 100)
        color = STRATEGY_COLORS[s]
        label = STRATEGY_LABELS[s].replace("\n", " ")
        ax.plot(checkpoints, means, color=color, linewidth=1.5, label=label)
        ax.fill_between(checkpoints, lo_list, hi_list, color=color, alpha=0.15)

    ax.set_xlabel("Number of Rollouts")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("(a) Collision Rate Convergence")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel (b): Mode coverage convergence
    ax = axes[1]
    for s in STRATEGY_ORDER:
        sdf = df[df["strategy"] == s]
        missed = sdf["missed_mode_steps"].values
        total = sdf["total_mode_checks"].values
        means, lo_list, hi_list = [], [], []
        for cp in checkpoints:
            n = min(cp, len(missed))
            tot = int(total[:n].sum())
            cov = tot - int(missed[:n].sum())
            p, lo, hi = wilson_ci(cov, tot)
            means.append(p * 100)
            lo_list.append(lo * 100)
            hi_list.append(hi * 100)
        color = STRATEGY_COLORS[s]
        label = STRATEGY_LABELS[s].replace("\n", " ")
        ax.plot(checkpoints, means, color=color, linewidth=1.5, label=label)
        ax.fill_between(checkpoints, lo_list, hi_list, color=color, alpha=0.15)

    ax.set_xlabel("Number of Rollouts")
    ax.set_ylabel("Mode Coverage (%)")
    ax.set_title("(b) Mode Coverage Convergence")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig_convergence.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  -> fig_convergence.pdf")


# ============================================================================
# Figure 8: Summary Table as LaTeX
# ============================================================================
def write_summary_table():
    filepath = os.path.join(OUTPUT_DIR, "summary_table.tex")
    with open(filepath, "w") as f:
        f.write("\\begin{tabular}{l c c c c c}\n")
        f.write("\\toprule\n")
        f.write("Strategy & Coll.\\% & Coll. 95\\% CI & Mode Cov.\\% & Cov. 95\\% CI & Solve (ms) \\\\\n")
        f.write("\\midrule\n")
        for s in STRATEGY_ORDER:
            row = summary[summary["strategy"] == s].iloc[0]
            cr = row["collision_rate"] * 100
            clo = row["collision_ci_lo"] * 100
            chi = row["collision_ci_hi"] * 100
            mc = row["mode_coverage"] * 100
            mlo = row["mode_cov_ci_lo"] * 100
            mhi = row["mode_cov_ci_hi"] * 100
            st = row["mean_solve_ms"]
            label = s.replace("+", "{+}").replace("-", "{-}")
            f.write(f"{label} & {cr:.2f} & [{clo:.2f}, {chi:.2f}] "
                    f"& {mc:.1f} & [{mlo:.1f}, {mhi:.1f}] & {st:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"  -> summary_table.tex")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print(f"Generating trajectory DRO figures from {RESULTS_CSV}...")
    print(f"Output directory: {OUTPUT_DIR}/\n")

    fig_collision_rate()
    fig_mode_coverage()
    fig_clearance_distribution()
    fig_solve_time()
    fig_safety_vs_coverage()
    fig_combined_overview()
    fig_convergence()
    write_summary_table()

    print(f"\nDone. {len(os.listdir(OUTPUT_DIR))} files in {OUTPUT_DIR}/")
