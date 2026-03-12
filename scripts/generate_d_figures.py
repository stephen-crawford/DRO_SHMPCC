#!/usr/bin/env python3
"""Generate Fig A/B/C from D1–D4 experiments.

Fig A: D1 — WDRO-K vs TopRisk-K vs DiverseRisk-K selection (grouped bar + line)
Fig B: D2 — WDRO-sampling vs Softmax-risk (bar chart, mode coverage focus)
Fig C: D3 — Low-observation regime (line plots over history length)
Fig D: D4 — Ground-cost ablation table (printed + heatmap)

Each figure has Oncoming (left) + Intersection (right) panels where available.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import sys

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

WDRO_COLOR = "#2166AC"
TOPRISK_COLOR = "#B2182B"
DIVERSE_COLOR = "#4DAF4A"
BASE_COLOR = "#999999"
SAMPLING_COLOR = "#FF7F00"

# ============================================================================
# Fig A: D1 — Selection Comparison (collision rate by K)
# ============================================================================
def fig_a():
    files = {
        "Oncoming":      os.path.join(DIR, "cdc_d1_selection_oncoming.csv"),
        "Intersection":  os.path.join(DIR, "cdc_d1_selection_intersection.csv"),
    }
    avail = {k: v for k, v in files.items() if os.path.exists(v)}
    if not avail:
        # Fall back to original single file
        orig = os.path.join(DIR, "cdc_d1_selection_comparison.csv")
        if os.path.exists(orig):
            avail = {"Oncoming": orig}
    if not avail:
        print("  Fig A: no D1 data found, skipping")
        return

    n_panels = len(avail)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.5), squeeze=False)

    for col, (env_name, fpath) in enumerate(avail.items()):
        ax = axes[0, col]
        df = pd.read_csv(fpath)

        selectors = df["selector"].unique()
        K_values = sorted(df["K"].unique())
        x = np.arange(len(K_values))
        width = 0.25
        colors = {"WDRO": WDRO_COLOR, "TopRisk": TOPRISK_COLOR, "DiverseRisk": DIVERSE_COLOR}

        for i, sel in enumerate(["WDRO", "TopRisk", "DiverseRisk"]):
            sub = df[df["selector"] == sel].sort_values("K")
            if sub.empty:
                continue
            rates = sub["collision_rate"].values * 100
            lo = sub["coll_ci_lo"].values * 100
            hi = sub["coll_ci_hi"].values * 100
            err = np.array([rates - lo, hi - rates])
            ax.bar(x + (i - 1) * width, rates, width * 0.9, yerr=err,
                   label=sel, color=colors.get(sel, "#666"),
                   capsize=2, error_kw={"linewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in K_values])
        ax.set_xlabel("Injection Count K")
        ax.set_ylabel("Collision Rate (%)")
        ax.set_title(f"{env_name}")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

        # Tight y-window per environment so CI whiskers are clearly separated
        if env_name == "Oncoming":
            ax.set_ylim(12.5, 18.0)
        elif env_name == "Intersection":
            ax.set_ylim(3.0, 7.0)

    # Place legend between panels, above the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.98), framealpha=0.9)
    fig.suptitle("Fig A: Mode Selection Rule Comparison", fontsize=11, y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_a_selection.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig A saved")


# ============================================================================
# Fig B: D2 — Sampling Comparison (collision + missed mode rate)
# ============================================================================
def fig_b():
    files = {
        "Oncoming":      os.path.join(DIR, "cdc_d2_sampling_oncoming.csv"),
        "Intersection":  os.path.join(DIR, "cdc_d2_sampling_intersection.csv"),
    }
    avail = {k: v for k, v in files.items() if os.path.exists(v)}
    if not avail:
        orig = os.path.join(DIR, "cdc_d2_sampling_comparison.csv")
        if os.path.exists(orig):
            avail = {"Oncoming": orig}
    if not avail:
        print("  Fig B: no D2 data found, skipping")
        return

    n_panels = len(avail)
    fig, axes = plt.subplots(2, n_panels, figsize=(4.5 * n_panels, 5.5), squeeze=False)

    for col, (env_name, fpath) in enumerate(avail.items()):
        df = pd.read_csv(fpath)

        # Prepare data: WDRO-sampling, Softmax variants, Base
        methods = []
        coll_rates = []
        coll_lo = []
        coll_hi = []
        mm_rates = []
        mm_lo = []
        mm_hi = []
        colors_list = []

        # WDRO-sampling
        wdro = df[df["variant"] == "WDRO"]
        if not wdro.empty:
            methods.append("WDRO\nsampling")
            coll_rates.append(wdro.iloc[0]["collision_rate"] * 100)
            coll_lo.append(wdro.iloc[0]["coll_ci_lo"] * 100)
            coll_hi.append(wdro.iloc[0]["coll_ci_hi"] * 100)
            mm_rates.append(wdro.iloc[0]["missed_mode_rate"] * 100)
            mm_lo.append(wdro.iloc[0]["mm_ci_lo"] * 100)
            mm_hi.append(wdro.iloc[0]["mm_ci_hi"] * 100)
            colors_list.append(WDRO_COLOR)

        # Softmax variants
        softmax = df[df["variant"] == "Softmax"].sort_values("tau")
        for _, row in softmax.iterrows():
            tau = int(row["tau"])
            methods.append(f"Softmax\n" + r"$\tau$" + f"={tau}")
            coll_rates.append(row["collision_rate"] * 100)
            coll_lo.append(row["coll_ci_lo"] * 100)
            coll_hi.append(row["coll_ci_hi"] * 100)
            mm_rates.append(row["missed_mode_rate"] * 100)
            mm_lo.append(row["mm_ci_lo"] * 100)
            mm_hi.append(row["mm_ci_hi"] * 100)
            colors_list.append(SAMPLING_COLOR)

        # Base
        base = df[df["variant"] == "Base"]
        if not base.empty:
            methods.append("Base")
            coll_rates.append(base.iloc[0]["collision_rate"] * 100)
            coll_lo.append(base.iloc[0]["coll_ci_lo"] * 100)
            coll_hi.append(base.iloc[0]["coll_ci_hi"] * 100)
            mm_rates.append(base.iloc[0]["missed_mode_rate"] * 100)
            mm_lo.append(base.iloc[0]["mm_ci_lo"] * 100)
            mm_hi.append(base.iloc[0]["mm_ci_hi"] * 100)
            colors_list.append(BASE_COLOR)

        x = np.arange(len(methods))
        cr = np.array(coll_rates)
        cr_err = np.array([cr - np.array(coll_lo), np.array(coll_hi) - cr])
        mr = np.array(mm_rates)
        mr_err = np.array([mr - np.array(mm_lo), np.array(mm_hi) - mr])

        # Top: collision rate
        ax_c = axes[0, col]
        ax_c.bar(x, cr, color=colors_list, yerr=cr_err, capsize=2, error_kw={"linewidth": 0.8})
        ax_c.set_ylabel("Collision Rate (%)")
        ax_c.set_title(f"{env_name} — Collision")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(methods, fontsize=7)

        # Bottom: missed mode rate
        ax_m = axes[1, col]
        ax_m.bar(x, mr, color=colors_list, yerr=mr_err, capsize=2, error_kw={"linewidth": 0.8})
        ax_m.set_ylabel("Missed Mode Rate (%)")
        ax_m.set_title(f"{env_name} — Mode Coverage")
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(methods, fontsize=7)

    fig.suptitle("Fig B: WDRO-sampling vs Softmax-risk Sampling", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_b_sampling.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig B saved")


# ============================================================================
# Fig C: D3 — Low-Observation Regime (line plots)
# ============================================================================
def fig_c():
    files = {
        "Oncoming":      os.path.join(DIR, "cdc_d3_observation_oncoming.csv"),
        "Intersection":  os.path.join(DIR, "cdc_d3_observation_intersection.csv"),
    }
    avail = {k: v for k, v in files.items() if os.path.exists(v)}
    if not avail:
        orig = os.path.join(DIR, "cdc_d3_low_observation.csv")
        if os.path.exists(orig):
            avail = {"Oncoming": orig}
    if not avail:
        print("  Fig C: no D3 data found, skipping")
        return

    n_panels = len(avail)
    fig, axes = plt.subplots(2, n_panels, figsize=(4.5 * n_panels, 5.5), squeeze=False)

    method_styles = {
        "Base":           {"color": BASE_COLOR,    "marker": "s", "ls": "--"},
        "WDRO-injection": {"color": WDRO_COLOR,    "marker": "o", "ls": "-"},
        "TopRisk-inject": {"color": TOPRISK_COLOR, "marker": "^", "ls": "-"},
        "WDRO-sampling":  {"color": SAMPLING_COLOR,"marker": "D", "ls": "-."},
    }

    for col, (env_name, fpath) in enumerate(avail.items()):
        df = pd.read_csv(fpath)

        for method_name, style in method_styles.items():
            sub = df[df["method"] == method_name].sort_values("max_history")
            if sub.empty:
                continue
            H = sub["max_history"].values

            # Collision rate
            cr = sub["collision_rate"].values * 100
            clo = sub["coll_ci_lo"].values * 100
            chi = sub["coll_ci_hi"].values * 100
            ax_c = axes[0, col]
            ax_c.plot(H, cr, label=method_name, **style, markersize=4, linewidth=1.3)
            ax_c.fill_between(H, clo, chi, alpha=0.12, color=style["color"])

            # Missed mode rate
            mr = sub["missed_mode_rate"].values * 100
            mlo = sub["mm_ci_lo"].values * 100
            mhi = sub["mm_ci_hi"].values * 100
            ax_m = axes[1, col]
            ax_m.plot(H, mr, label=method_name, **style, markersize=4, linewidth=1.3)
            ax_m.fill_between(H, mlo, mhi, alpha=0.12, color=style["color"])

        axes[0, col].set_title(f"{env_name} — Collision Rate")
        axes[0, col].set_xlabel("Max History Length")
        axes[0, col].set_ylabel("Collision Rate (%)")

        axes[1, col].set_title(f"{env_name} — Missed Mode Rate")
        axes[1, col].set_xlabel("Max History Length")
        axes[1, col].set_ylabel("Missed Mode Rate (%)")

    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Fig C: Low-Observation Regime", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_c_observation.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Fig C saved")


# ============================================================================
# Fig D: D4 — Ground-cost ablation (heatmap + table)
# ============================================================================
def fig_d():
    fpath = os.path.join(DIR, "cdc_d4_ground_cost_ablation.csv")
    if not os.path.exists(fpath):
        print("  Fig D: no D4 data found, skipping")
        return

    df = pd.read_csv(fpath)

    methods = df["method"].unique()
    costs = df["ground_cost"].unique()

    # Build collision rate and missed mode rate matrices
    coll_matrix = np.zeros((len(methods), len(costs)))
    mm_matrix = np.zeros((len(methods), len(costs)))
    for i, m in enumerate(methods):
        for j, c in enumerate(costs):
            row = df[(df["method"] == m) & (df["ground_cost"] == c)]
            if not row.empty:
                coll_matrix[i, j] = row.iloc[0]["collision_rate"] * 100
                mm_matrix[i, j] = row.iloc[0]["missed_mode_rate"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    # Collision rate heatmap
    im1 = ax1.imshow(coll_matrix, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(len(costs)))
    ax1.set_xticklabels(costs, fontsize=8)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels(methods, fontsize=8)
    ax1.set_title("Collision Rate (%)")
    for i in range(len(methods)):
        for j in range(len(costs)):
            ax1.text(j, i, f"{coll_matrix[i,j]:.1f}", ha="center", va="center", fontsize=8,
                     color="white" if coll_matrix[i,j] > 15 else "black")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # Missed mode rate heatmap
    im2 = ax2.imshow(mm_matrix, cmap="RdYlGn_r", aspect="auto")
    ax2.set_xticks(range(len(costs)))
    ax2.set_xticklabels(costs, fontsize=8)
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods, fontsize=8)
    ax2.set_title("Missed Mode Rate (%)")
    for i in range(len(methods)):
        for j in range(len(costs)):
            ax2.text(j, i, f"{mm_matrix[i,j]:.1f}", ha="center", va="center", fontsize=8,
                     color="white" if mm_matrix[i,j] > 24 else "black")
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Fig D: Ground-Cost Ablation (Oncoming)", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_d_ground_cost.{ext}"), bbox_inches="tight")
    plt.close(fig)

    # Also print LaTeX table
    tex_path = os.path.join(DIR, "d4_ground_cost_table.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{l" + "cc" * len(costs) + "}\n")
        f.write("\\toprule\n")
        header = " & ".join([f"\\multicolumn{{2}}{{c}}{{{c}}}" for c in costs])
        f.write(f"Method & {header} \\\\\n")
        subheader = " & ".join(["Coll.\\% & MM\\%"] * len(costs))
        f.write(f" & {subheader} \\\\\n")
        f.write("\\midrule\n")
        for i, m in enumerate(methods):
            vals = " & ".join([f"{coll_matrix[i,j]:.1f} & {mm_matrix[i,j]:.1f}" for j in range(len(costs))])
            f.write(f"{m} & {vals} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    print(f"  Fig D saved + {tex_path}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating D-experiment figures...")
    fig_a()
    fig_b()
    fig_c()
    fig_d()
    print("Done.")
