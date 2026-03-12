#!/usr/bin/env python3
"""
Generate publication figures for the 6-strategy OT/DRO/SH comparison.

Reads:
  paper_figures/strategy_comparison_summary.csv
  paper_figures/strategy_comparison_results.csv

Writes:
  paper_figures/fig_collision_rate.pdf
  paper_figures/fig_mode_coverage.pdf
  paper_figures/fig_rare_mode_coverage.pdf
  paper_figures/fig_solve_time.pdf
  paper_figures/fig_safety_vs_coverage.pdf
  paper_figures/fig_clearance_distribution.pdf
  paper_figures/fig_combined_overview.pdf
  paper_figures/strategy_summary_table.tex
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
OUT_DIR  = "paper_figures/"

# Strategy ordering and styling
STRATEGY_ORDER = [
    "Base", "OT", "DRO(inj)",
    "SH", "OT+SH", "DRO(inj)+SH",
    "OT+DRO(inj)+SH", "DRO(q*)+SH", "OT+DRO(q*)+SH",
]
STRATEGY_LABELS = {
    "Base":             "Base",
    "OT":               "OT",
    "DRO(inj)":         "DRO(inj)",
    "SH":               "SH",
    "OT+SH":            "OT+SH",
    "DRO(inj)+SH":      "DRO(inj)+SH",
    "OT+DRO(inj)+SH":   "OT+DRO(inj)+SH",
    "DRO(q*)+SH":       r"DRO($q^*$)+SH",
    "OT+DRO(q*)+SH":    r"OT+DRO($q^*$)+SH",
}
STRATEGY_COLORS = {
    "Base":             "#7f7f7f",  # gray
    "OT":               "#17becf",  # cyan
    "DRO(inj)":         "#bcbd22",  # olive
    "SH":               "#1f77b4",  # blue
    "OT+SH":            "#2ca02c",  # green
    "DRO(inj)+SH":      "#ff7f0e",  # orange
    "OT+DRO(inj)+SH":   "#9467bd",  # purple
    "DRO(q*)+SH":       "#d62728",  # red
    "OT+DRO(q*)+SH":    "#8c564b",  # brown
}
STRATEGY_HATCHES = {
    "Base":             "",
    "OT":               "",
    "DRO(inj)":         "",
    "SH":               "",
    "OT+SH":            "",
    "DRO(inj)+SH":      "//",
    "OT+DRO(inj)+SH":   "xx",
    "DRO(q*)+SH":       "//",
    "OT+DRO(q*)+SH":    "xx",
}


def load_summary():
    path = os.path.join(DATA_DIR, "strategy_comparison_summary.csv")
    return pd.read_csv(path)


def load_detail():
    path = os.path.join(DATA_DIR, "strategy_comparison_results.csv")
    return pd.read_csv(path)


def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


def get_ordered(df, col):
    """Return values in STRATEGY_ORDER."""
    vals = []
    for s in STRATEGY_ORDER:
        row = df[df["strategy"] == s]
        vals.append(row[col].values[0] if len(row) > 0 else 0.0)
    return np.array(vals)


# ============================================================================
# Fig 1: Collision Rate with Wilson CI whiskers
# ============================================================================

def fig_collision_rate(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(STRATEGY_ORDER))

    rates = get_ordered(df, "collision_rate") * 100
    ci_lo = get_ordered(df, "collision_ci_lo") * 100
    ci_hi = get_ordered(df, "collision_ci_hi") * 100
    yerr = np.array([rates - ci_lo, ci_hi - rates])

    colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
    hatches = [STRATEGY_HATCHES[s] for s in STRATEGY_ORDER]
    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]

    bars = ax.bar(x, rates, 0.6, yerr=yerr, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Collision Rate (95% Wilson CI)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate rates
    for i, r in enumerate(rates):
        ax.text(i, ci_hi[i] + 0.3, f"{r:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    save_fig(fig, "fig_collision_rate")


# ============================================================================
# Fig 2: Mode Coverage with Wilson CI whiskers
# ============================================================================

def fig_mode_coverage(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(len(STRATEGY_ORDER))
    colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
    hatches = [STRATEGY_HATCHES[s] for s in STRATEGY_ORDER]
    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]

    # Left: overall mode coverage
    mc = get_ordered(df, "mode_coverage") * 100
    mc_lo = get_ordered(df, "mode_cov_ci_lo") * 100
    mc_hi = get_ordered(df, "mode_cov_ci_hi") * 100
    yerr_mc = np.array([mc - mc_lo, mc_hi - mc])

    bars = ax1.bar(x, mc, 0.6, yerr=yerr_mc, capsize=4,
                   color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("Mode Coverage (%)")
    ax1.set_title("Overall Mode Coverage (95% CI)")
    ax1.set_ylim(0, min(105, max(85, mc_hi.max() * 1.08)))
    ax1.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(mc):
        ax1.text(i, mc_hi[i] + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    # Right: rare mode coverage
    rc = get_ordered(df, "rare_mode_coverage") * 100
    rc_lo = get_ordered(df, "rare_cov_ci_lo") * 100
    rc_hi = get_ordered(df, "rare_cov_ci_hi") * 100
    yerr_rc = np.array([rc - rc_lo, rc_hi - rc])

    bars = ax2.bar(x, rc, 0.6, yerr=yerr_rc, capsize=4,
                   color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.set_ylabel("Rare Mode Coverage (%)")
    ax2.set_title("Rare Mode Coverage (95% CI)")
    ax2.set_ylim(0, min(105, max(70, rc_hi.max() * 1.08)))
    ax2.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(rc):
        ax2.text(i, rc_hi[i] + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig_mode_coverage")


# ============================================================================
# Fig 3: Solve Time comparison (mean, p50, p95)
# ============================================================================

def fig_solve_time(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(STRATEGY_ORDER))
    width = 0.22

    mean_t = get_ordered(df, "mean_solve_ms")
    med_t  = get_ordered(df, "median_solve_ms")
    p95_t  = get_ordered(df, "p95_solve_ms")

    labels_x = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
    colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]

    ax.bar(x - width, mean_t, width, label="Mean", color=colors, alpha=0.9,
           edgecolor="black", linewidth=0.5)
    ax.bar(x, med_t, width, label="Median (p50)", color=colors, alpha=0.65,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width, p95_t, width, label="p95", color=colors, alpha=0.4,
           edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=25, ha="right")
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("MPC Solve Time per Step")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig_solve_time")


# ============================================================================
# Fig 4: Safety vs Coverage tradeoff (scatter)
# ============================================================================

def fig_safety_vs_coverage(df):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for s in STRATEGY_ORDER:
        row = df[df["strategy"] == s]
        if row.empty:
            continue
        cr = row["collision_rate"].values[0] * 100
        mc = row["mode_coverage"].values[0] * 100
        cr_lo = row["collision_ci_lo"].values[0] * 100
        cr_hi = row["collision_ci_hi"].values[0] * 100
        mc_lo = row["mode_cov_ci_lo"].values[0] * 100
        mc_hi = row["mode_cov_ci_hi"].values[0] * 100

        color = STRATEGY_COLORS[s]
        label = STRATEGY_LABELS[s]

        ax.errorbar(mc, cr,
                    xerr=[[mc - mc_lo], [mc_hi - mc]],
                    yerr=[[cr - cr_lo], [cr_hi - cr]],
                    fmt="o", color=color, markersize=9,
                    capsize=4, linewidth=1.5, label=label,
                    markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Mode Coverage (%)")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Safety vs Mode Coverage Tradeoff")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Ideal direction arrow
    ax.annotate("", xy=(75, 5), xytext=(40, 18),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
    ax.text(58, 10, "ideal", color="gray", fontsize=9, fontstyle="italic",
            ha="center", rotation=-15)

    fig.tight_layout()
    save_fig(fig, "fig_safety_vs_coverage")


# ============================================================================
# Fig 5: Per-rollout clearance distribution (box plot)
# ============================================================================

def fig_clearance_distribution(detail_df):
    fig, ax = plt.subplots(figsize=(9, 4))

    data = []
    labels = []
    colors_list = []
    for s in STRATEGY_ORDER:
        sub = detail_df[detail_df["strategy"] == s]
        data.append(sub["clearance_5pct"].values)
        labels.append(STRATEGY_LABELS[s])
        colors_list.append(STRATEGY_COLORS[s])

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showfliers=False, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("5th Percentile Clearance (m)")
    ax.set_title("Minimum Clearance Distribution")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=25, ha="right")

    fig.tight_layout()
    save_fig(fig, "fig_clearance_distribution")


# ============================================================================
# Fig 6: Combined overview (2x2 panel)
# ============================================================================

def fig_combined_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(STRATEGY_ORDER))
    colors = [STRATEGY_COLORS[s] for s in STRATEGY_ORDER]
    hatches = [STRATEGY_HATCHES[s] for s in STRATEGY_ORDER]
    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]

    def make_bar(ax, vals, lo, hi, ylabel, title, ylim_top=None, pct=True):
        if pct:
            vals, lo, hi = vals * 100, lo * 100, hi * 100
        yerr = np.array([vals - lo, hi - vals])
        bars = ax.bar(x, vals, 0.6, yerr=yerr, capsize=3,
                      color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0, top=ylim_top)
        ax.grid(True, alpha=0.3, axis="y")
        suffix = "%" if pct else ""
        for i, v in enumerate(vals):
            ax.text(i, hi[i] + 0.3, f"{v:.1f}{suffix}",
                    ha="center", va="bottom", fontsize=7)

    # (a) Collision rate
    coll_hi = get_ordered(df, "collision_ci_hi").max() * 100
    make_bar(axes[0, 0],
             get_ordered(df, "collision_rate"),
             get_ordered(df, "collision_ci_lo"),
             get_ordered(df, "collision_ci_hi"),
             "Collision Rate (%)", "(a) Collision Rate",
             ylim_top=max(22, coll_hi * 1.2))

    # (b) Mode coverage
    mc_hi = get_ordered(df, "mode_cov_ci_hi").max() * 100
    make_bar(axes[0, 1],
             get_ordered(df, "mode_coverage"),
             get_ordered(df, "mode_cov_ci_lo"),
             get_ordered(df, "mode_cov_ci_hi"),
             "Mode Coverage (%)", "(b) Mode Coverage",
             ylim_top=min(105, max(85, mc_hi * 1.08)))

    # (c) Rare mode coverage
    rc_hi = get_ordered(df, "rare_cov_ci_hi").max() * 100
    make_bar(axes[1, 0],
             get_ordered(df, "rare_mode_coverage"),
             get_ordered(df, "rare_cov_ci_lo"),
             get_ordered(df, "rare_cov_ci_hi"),
             "Rare Mode Coverage (%)", "(c) Rare Mode Coverage",
             ylim_top=min(105, max(70, rc_hi * 1.08)))

    # (d) Solve time
    ax = axes[1, 1]
    mean_t = get_ordered(df, "mean_solve_ms")
    p95_t  = get_ordered(df, "p95_solve_ms")
    bars = ax.bar(x, mean_t, 0.6, color=colors, edgecolor="black",
                  linewidth=0.5, alpha=0.85, label="Mean")
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    # p95 as error cap
    ax.errorbar(x, mean_t, yerr=[np.zeros_like(mean_t), p95_t - mean_t],
                fmt="none", ecolor="black", capsize=4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Solve Time (ms)")
    ax.set_title("(d) MPC Solve Time (bar=mean, cap=p95)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(mean_t):
        ax.text(i, p95_t[i] + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig_combined_overview")


# ============================================================================
# LaTeX summary table
# ============================================================================

def generate_latex_table(df):
    path = os.path.join(OUT_DIR, "strategy_summary_table.tex")
    n_rollouts = int(df["n"].iloc[0])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Strategy comparison: collision rate, mode coverage, and solve time "
        rf"({n_rollouts} rollouts per strategy, 95\% Wilson CIs).}}",  # double }} for f-string escape
        r"\label{tab:strategy_comparison}",
        r"\small",
        r"\begin{tabular}{l c c c c c}",
        r"\toprule",
        r"Strategy & Coll.\ Rate & Mode Cov.\ & Rare Cov.\ & Solve (ms) & Compl.\ \\",
        r"\midrule",
    ]

    for s in STRATEGY_ORDER:
        row = df[df["strategy"] == s]
        if row.empty:
            continue
        r = row.iloc[0]
        cr = r["collision_rate"] * 100
        clo = r["collision_ci_lo"] * 100
        chi = r["collision_ci_hi"] * 100
        mc = r["mode_coverage"] * 100
        mlo = r["mode_cov_ci_lo"] * 100
        mhi = r["mode_cov_ci_hi"] * 100
        rc = r["rare_mode_coverage"] * 100
        rlo = r["rare_cov_ci_lo"] * 100
        rhi = r["rare_cov_ci_hi"] * 100
        st = r["mean_solve_ms"]
        comp = r["completion_rate"] * 100

        label = STRATEGY_LABELS[s].replace("$", r"$")
        cr_str = f"{cr:.1f}\\% [{clo:.1f}, {chi:.1f}]"
        mc_str = f"{mc:.1f}\\% [{mlo:.1f}, {mhi:.1f}]"
        rc_str = f"{rc:.1f}\\% [{rlo:.1f}, {rhi:.1f}]"
        st_str = f"{st:.2f}"
        comp_str = f"{comp:.1f}\\%"

        lines.append(f"{label} & {cr_str} & {mc_str} & {rc_str} & {st_str} & {comp_str} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Generate figures for each available dataset
    for suffix in ["", "_4obs"]:
        tag = suffix if suffix else ""
        label = suffix.strip("_") if suffix else "1obs"
        sum_path = os.path.join(DATA_DIR, f"strategy_comparison_summary{tag}.csv")
        det_path = os.path.join(DATA_DIR, f"strategy_comparison_results{tag}.csv")

        if not os.path.exists(sum_path):
            print(f"SKIP ({label}): {sum_path} not found")
            continue

        print(f"Loading data ({label})...")
        summary = pd.read_csv(sum_path)
        n = summary["n"].iloc[0]
        print(f"  {len(summary)} strategies, {n} rollouts each\n")

        print(f"Generating figures ({label})...")

        # Use a custom save that appends the suffix
        def save_fig_tagged(fig, name, _tag=tag):
            for ext in ["pdf", "png"]:
                fig.savefig(os.path.join(OUT_DIR, f"{name}{_tag}.{ext}"))
            plt.close(fig)
            print(f"  -> {name}{_tag}.pdf / .png")

        # Temporarily replace save_fig
        import types
        old_save = globals()["save_fig"]
        globals()["save_fig"] = save_fig_tagged

        fig_collision_rate(summary)
        fig_mode_coverage(summary)
        fig_solve_time(summary)
        fig_safety_vs_coverage(summary)
        fig_combined_overview(summary)

        if os.path.exists(det_path):
            detail = pd.read_csv(det_path)
            fig_clearance_distribution(detail)

        globals()["save_fig"] = old_save

        print(f"\nGenerating LaTeX table ({label})...")
        generate_latex_table(summary)
        if tag:
            os.rename(os.path.join(OUT_DIR, "strategy_summary_table.tex"),
                      os.path.join(OUT_DIR, f"strategy_summary_table{tag}.tex"))

    print("\nDone. All outputs in paper_figures/")
