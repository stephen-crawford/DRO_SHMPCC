#!/usr/bin/env python3
"""Generate figures from the parameter tuning sweep results."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

DATA_DIR = "paper_figures/"
OUT_DIR  = "paper_figures/"


def load():
    path = os.path.join(DATA_DIR, "tuning_sweep_4obs.csv")
    return pd.read_csv(path)


def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


def fig_collision_vs_coverage(df):
    """Scatter plot: collision rate vs mode coverage, color-coded by category."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Categorize configs
    categories = {
        "Reference":       {"color": "#7f7f7f", "marker": "s", "size": 100},
        "DRO rho":         {"color": "#1f77b4", "marker": "^", "size": 70},
        "SH min":          {"color": "#2ca02c", "marker": "D", "size": 70},
        "SH forced":       {"color": "#ff7f0e", "marker": "v", "size": 70},
        "OT+DRO alpha":    {"color": "#d62728", "marker": "x", "size": 70},
        "More scenarios":  {"color": "#9467bd", "marker": "p", "size": 90},
        "Combos":          {"color": "#8c564b", "marker": "*", "size": 100},
    }

    def categorize(name):
        if name in ("Base", "DRO(inj)", "DRO(inj)+SH", "OT+SH"):
            return "Reference"
        if "rho=" in name and "min=" not in name and "S=" not in name:
            return "DRO rho"
        if "min=" in name and "rho=" not in name:
            return "SH min"
        if "forced=" in name:
            return "SH forced"
        if "a=" in name or "OT(freq)" in name:
            return "OT+DRO alpha"
        if "S=" in name and "rho=" not in name:
            return "More scenarios"
        return "Combos"

    # Get base CI for reference lines
    base = df[df["config"] == "Base"].iloc[0]

    # Plot reference lines
    ax.axhline(y=base["collision_rate"] * 100, color="#cccccc", ls="--", lw=1, zorder=0)
    ax.axvline(x=base["mode_coverage"] * 100, color="#cccccc", ls="--", lw=1, zorder=0)
    ax.text(base["mode_coverage"] * 100 + 0.5, 26, "Base collision", fontsize=7, color="#999")
    ax.text(30, base["collision_rate"] * 100 + 0.3, "Base mode cov", fontsize=7, color="#999")

    # Plot each config
    plotted_categories = set()
    for _, row in df.iterrows():
        cat = categorize(row["config"])
        style = categories[cat]
        label = cat if cat not in plotted_categories else None
        plotted_categories.add(cat)

        cr = row["collision_rate"] * 100
        mc = row["mode_coverage"] * 100
        cr_lo = row["coll_ci_lo"] * 100
        cr_hi = row["coll_ci_hi"] * 100
        mc_lo = row["mode_ci_lo"] * 100
        mc_hi = row["mode_ci_hi"] * 100

        ax.errorbar(mc, cr,
                     xerr=[[mc - mc_lo], [mc_hi - mc]],
                     yerr=[[cr - cr_lo], [cr_hi - cr]],
                     fmt="none", ecolor=style["color"], capsize=2, alpha=0.4, lw=1)
        ax.scatter(mc, cr, c=style["color"], marker=style["marker"],
                   s=style["size"], label=label, zorder=5,
                   edgecolors="black", linewidth=0.3, alpha=0.9)

        # Annotate key configs
        if row["config"] in ("Base", "DRO(inj)", "DRO(inj)+SH", "DRO(inj) S=60",
                              "DRO(inj)+SH min=12", "OT+SH"):
            offset = (5, 5) if cr > 10 else (5, -10)
            ax.annotate(row["config"], (mc, cr), textcoords="offset points",
                        xytext=offset, fontsize=7, color=style["color"])

    ax.set_xlabel("Mode Coverage (%)")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Parameter Tuning: Safety vs Mode Coverage (4 obstacles)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(25, 101)
    ax.set_ylim(0, 32)
    ax.grid(True, alpha=0.2)

    # Arrow to ideal
    ax.annotate("", xy=(100, 2), xytext=(60, 15),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))
    ax.text(80, 7, "ideal", color="gray", fontsize=9, fontstyle="italic",
            ha="center", rotation=-15)

    fig.tight_layout()
    save_fig(fig, "fig_tuning_scatter_4obs")


def fig_sh_min_trend(df):
    """Line plot showing collision rate vs safe_horizon_min for DRO(inj)+SH."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Extract SH min sweep data
    sh_data = []
    for _, row in df.iterrows():
        name = row["config"]
        if name == "DRO(inj)+SH":
            sh_data.append((3, row))
        elif "DRO(inj)+SH min=" in name and "rho=" not in name:
            m = int(name.split("min=")[1])
            sh_data.append((m, row))
    sh_data.sort()

    mins = [d[0] for d in sh_data]
    crs = [d[1]["collision_rate"] * 100 for d in sh_data]
    cr_los = [d[1]["coll_ci_lo"] * 100 for d in sh_data]
    cr_his = [d[1]["coll_ci_hi"] * 100 for d in sh_data]
    mcs = [d[1]["mode_coverage"] * 100 for d in sh_data]
    mc_los = [d[1]["mode_ci_lo"] * 100 for d in sh_data]
    mc_his = [d[1]["mode_ci_hi"] * 100 for d in sh_data]
    rcs = [d[1]["rare_coverage"] * 100 for d in sh_data]

    # DRO(inj) reference
    dro = df[df["config"] == "DRO(inj)"].iloc[0]
    dro_cr = dro["collision_rate"] * 100
    dro_mc = dro["mode_coverage"] * 100

    # Left: collision rate vs min
    ax1.errorbar(mins, crs, yerr=[np.array(crs) - cr_los, np.array(cr_his) - crs],
                 fmt="o-", color="#2ca02c", capsize=4, markersize=8, label="DRO(inj)+SH")
    ax1.axhline(y=dro_cr, color="#1f77b4", ls="--", lw=1.5, label=f"DRO(inj): {dro_cr:.1f}%")
    ax1.set_xlabel("safe_horizon_min")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs SH Floor")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(mins)

    # Right: mode coverage vs min
    ax2.errorbar(mins, mcs, yerr=[np.array(mcs) - mc_los, np.array(mc_his) - mcs],
                 fmt="o-", color="#2ca02c", capsize=4, markersize=8, label="DRO(inj)+SH mode cov")
    ax2.plot(mins, rcs, "s--", color="#ff7f0e", markersize=6, label="Rare mode cov")
    ax2.axhline(y=dro_mc, color="#1f77b4", ls="--", lw=1.5, label=f"DRO(inj): {dro_mc:.1f}%")
    ax2.set_xlabel("safe_horizon_min")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Mode Coverage vs SH Floor")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(mins)
    ax2.set_ylim(98.5, 100.2)

    fig.tight_layout()
    save_fig(fig, "fig_tuning_sh_min_4obs")


def fig_full_ranking(df):
    """Horizontal bar chart ranking all configs by collision rate."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by collision rate
    df_sorted = df.sort_values("collision_rate", ascending=True).reset_index(drop=True)

    y = np.arange(len(df_sorted))
    crs = df_sorted["collision_rate"].values * 100
    cr_los = df_sorted["coll_ci_lo"].values * 100
    cr_his = df_sorted["coll_ci_hi"].values * 100
    xerr = np.array([crs - cr_los, cr_his - crs])

    # Color by whether they beat Base
    base_cr_hi = df[df["config"] == "Base"]["coll_ci_hi"].values[0] * 100
    colors = ["#2ca02c" if chi < base_cr_hi else "#d62728" for chi in cr_his]

    bars = ax.barh(y, crs, xerr=xerr, capsize=3, color=colors,
                   edgecolor="black", linewidth=0.5, alpha=0.8, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted["config"].values, fontsize=8)
    ax.set_xlabel("Collision Rate (%)")
    ax.set_title("All Configs Ranked by Collision Rate (4 obstacles, 2000 rollouts)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate with mode coverage
    for i, row in df_sorted.iterrows():
        mc = row["mode_coverage"] * 100
        ax.text(cr_his[i] + 0.3, i, f"mcov={mc:.1f}%", va="center", fontsize=7)

    fig.tight_layout()
    save_fig(fig, "fig_tuning_ranking_4obs")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading tuning sweep data...")
    df = load()
    print(f"  {len(df)} configurations\n")

    print("Generating figures...")
    fig_collision_vs_coverage(df)
    fig_sh_min_trend(df)
    fig_full_ranking(df)
    print("\nDone.")
