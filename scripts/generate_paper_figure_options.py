#!/usr/bin/env python3
"""Generate multiple figure-set options for the WDRO paper.

Each option set produces up to 4 figures that collectively support the
paper's findings. The user selects one option set to include in the paper.

Usage:
    python3 scripts/generate_paper_figure_options.py [option_a|option_b|option_c|all]

Data sources (CSVs from test_generalization):
    figures/generalization_figures/g1_path_geometry.csv
    figures/generalization_figures/g3_switch_dynamics.csv
    figures/generalization_figures/g6_challenging_envs.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import sys
import pathlib

# ============================================================================
# Setup
# ============================================================================

ROOT = pathlib.Path(__file__).resolve().parent.parent
G_DIR = ROOT / "figures" / "generalization_figures"
OUT = ROOT / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Try multiple CSV locations
def _find_csv(name):
    for d in [G_DIR,
              ROOT / "figures" / "generalization",
              ROOT / "figures" / "g1",
              ROOT / "figures" / "g6"]:
        p = d / name
        if p.exists():
            return p
    return None

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette — consistent across all options
COLORS = {
    "Base":             "#888888",
    "WDRO-sampling":    "#E6850E",
    "WDRO-inject-K1":   "#2166AC",
    "WDRO-inject-K2":   "#4393C3",
    "WDRO-inject-K3":   "#92C5DE",
    "TopRisk-K1":       "#B2182B",
    "TopRisk-K2":       "#E06666",
    "TopRisk-K3":       "#F4A582",
    "DiverseRisk-K1":   "#4DAF4A",
    "Softmax-tau5":     "#984EA3",
}

NICE = {
    "Base":             "Base SH-MPCC",
    "WDRO-sampling":    "WDRO-sampling",
    "WDRO-inject-K1":   "WDRO-$1$",
    "WDRO-inject-K2":   "WDRO-$2$",
    "WDRO-inject-K3":   "WDRO-$3$",
    "TopRisk-K1":       "TopRisk-$1$",
    "TopRisk-K2":       "TopRisk-$2$",
    "TopRisk-K3":       "TopRisk-$3$",
    "DiverseRisk-K1":   "DiverseRisk-$1$",
    "Softmax-tau5":     r"Softmax $\tau{=}5$",
}

MARKERS = {
    "Base": "s", "WDRO-sampling": "D", "WDRO-inject-K1": "o",
    "WDRO-inject-K2": "v", "WDRO-inject-K3": "p", "TopRisk-K1": "^",
    "TopRisk-K2": ">", "TopRisk-K3": "<", "DiverseRisk-K1": "P",
    "Softmax-tau5": "X",
}

def _c(m): return COLORS.get(m, "#666")
def _n(m): return NICE.get(m, m)
def _m(m): return MARKERS.get(m, "o")


# ============================================================================
# Shared path-diagram drawing
# ============================================================================

def _draw_path_diagram(ax, path_name):
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.65, 0.65)
    ax.set_aspect("equal")
    ax.axis("off")

    t = np.linspace(0, 1, 200)
    amp = {"Straight": 0, "Gentle-S": 0.08, "S-curve": 0.22, "Tight-S": 0.42}
    a = amp.get(path_name, 0.22)
    px, py = t, a * np.sin(2 * np.pi * t)

    ax.plot(px, py, color="#C0C0C0", lw=10, solid_capstyle="round", zorder=1)
    ax.plot(px, py, color="#E8E8E8", lw=6, solid_capstyle="round", zorder=2)

    def _tn(idx):
        d = 5
        dx = px[min(idx+d, len(t)-1)] - px[max(idx-d, 0)]
        dy = py[min(idx+d, len(t)-1)] - py[max(idx-d, 0)]
        n = np.sqrt(dx**2 + dy**2) + 1e-9
        return dx/n, dy/n, -dy/n, dx/n

    # Ego
    ei = int(0.15 * len(t))
    ex, ey = px[ei], py[ei]
    etx, ety, enx, eny = _tn(ei)
    ew, eh = 0.09, 0.045
    corners = [(ex+ew*etx+eh*enx, ey+ew*ety+eh*eny),
               (ex+ew*etx-eh*enx, ey+ew*ety-eh*eny),
               (ex-ew*etx-eh*enx, ey-ew*ety-eh*eny),
               (ex-ew*etx+eh*enx, ey-ew*ety+eh*eny)]
    ax.add_patch(plt.Polygon(corners, closed=True, fc="#2166AC", ec="white", lw=0.8, zorder=4))
    ax.text(ex, ey - eh - 0.06, "ego", ha="center", va="top", fontsize=7,
            color="#2166AC", fontweight="bold")

    # Obstacle
    oi = int(0.55 * len(t))
    ox, oy = px[oi], py[oi]
    otx, oty, onx, ony = _tn(oi)
    otx, oty = -otx, -oty
    ow, oh = 0.07, 0.038
    corners_o = [(ox+ow*otx+oh*onx, oy+ow*oty+oh*ony),
                 (ox+ow*otx-oh*onx, oy+ow*oty-oh*ony),
                 (ox-ow*otx-oh*onx, oy-ow*oty-oh*ony),
                 (ox-ow*otx+oh*onx, oy-ow*oty+oh*ony)]
    ax.add_patch(plt.Polygon(corners_o, closed=True, fc="#CC2222", ec="white", lw=0.8, zorder=4))
    ax.annotate("", xy=(ox+0.14*otx, oy+0.14*oty), xytext=(ox, oy),
                arrowprops=dict(arrowstyle="-|>", color="#CC2222", lw=1.8), zorder=5)

    ax.text(0.5, -0.58, path_name, ha="center", va="top", fontsize=10, fontweight="bold")


def _draw_challenge_diagram(ax, challenge):
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.65, 0.65)
    ax.set_aspect("equal")
    ax.axis("off")

    t = np.linspace(0, 1, 200)
    px = t
    py = 0.22 * np.sin(2 * np.pi * t)
    ax.plot(px, py, color="#C0C0C0", lw=8, solid_capstyle="round", zorder=1)
    ax.plot(px, py, color="#E8E8E8", lw=5, solid_capstyle="round", zorder=2)

    def _tn(idx):
        d = 5
        dx = px[min(idx+d, len(t)-1)] - px[max(idx-d, 0)]
        dy = py[min(idx+d, len(t)-1)] - py[max(idx-d, 0)]
        n = np.sqrt(dx**2 + dy**2) + 1e-9
        return dx/n, dy/n, -dy/n, dx/n

    # Ego at start
    ei = int(0.15 * len(t))
    ex, ey = px[ei], py[ei]
    etx, ety, _, _ = _tn(ei)
    ax.plot(ex, ey, "s", color="#2166AC", ms=8, zorder=4)
    ax.annotate("", xy=(ex+0.08*etx, ey+0.08*ety), xytext=(ex, ey),
                arrowprops=dict(arrowstyle="-|>", color="#2166AC", lw=1.5), zorder=5)

    nice = {"Crossing": "Crossing", "Diagonal": "Diagonal",
            "Mixed-2obs": "Mixed (2)", "Mixed-3obs": "Mixed (3)",
            "Dense-4obs": "Dense (4)"}

    # Obstacle positions depend on challenge type
    if "Crossing" in challenge:
        mid_i = int(0.5 * len(t))
        ox, oy = px[mid_i], py[mid_i] + 0.35
        ax.plot(ox, oy, "o", color="#CC2222", ms=7, zorder=4)
        ax.annotate("", xy=(ox, oy-0.12), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color="#CC2222", lw=1.5), zorder=5)
    elif "Diagonal" in challenge:
        mid_i = int(0.5 * len(t))
        ox, oy = px[mid_i]+0.15, py[mid_i]+0.25
        ax.plot(ox, oy, "o", color="#CC2222", ms=7, zorder=4)
        ax.annotate("", xy=(ox-0.08, oy-0.08), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color="#CC2222", lw=1.5), zorder=5)
    elif "Mixed" in challenge or "Dense" in challenge:
        n_obs = 2 if "2" in challenge else (3 if "3" in challenge else 4)
        positions = [(0.35, 0.3), (0.6, -0.25), (0.45, -0.35), (0.75, 0.2)]
        angles = [(-0.06, -0.08), (0.04, 0.08), (-0.08, 0.04), (-0.06, -0.06)]
        for i in range(min(n_obs, 4)):
            fx, fy = positions[i]
            mi = int(fx * len(t))
            ox = px[mi] + fy * 0.3
            oy = py[mi] + fy
            ax.plot(ox, oy, "o", color="#CC2222", ms=6, zorder=4)
            dx, dy = angles[i]
            ax.annotate("", xy=(ox+dx, oy+dy), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle="-|>", color="#CC2222", lw=1.2), zorder=5)
    else:
        mid_i = int(0.55 * len(t))
        ox, oy = px[mid_i], py[mid_i]
        ax.plot(ox, oy, "o", color="#CC2222", ms=7, zorder=4)

    ax.text(0.5, -0.58, nice.get(challenge, challenge), ha="center", va="top",
            fontsize=9, fontweight="bold")


# ============================================================================
# OPTION A: Paper-Aligned (one figure per results subsection)
#   Fig 1: Path geometry — collision + missed mode bars with diagrams
#   Fig 2: Challenging environments — collision + clearance bars with diagrams
#   Fig 3: Switching dynamics sweep — line plot showing robustness
#   Fig 4: Summary table as figure — all results at a glance
# ============================================================================

def option_a_fig1(df_g1):
    """Path geometry comparison: diagrams + collision rate + missed mode rate."""
    paths = [p for p in ["Straight", "Gentle-S", "S-curve", "Tight-S"] if p in df_g1["path"].unique()]
    methods = ["Base", "WDRO-sampling", "WDRO-inject-K1", "WDRO-inject-K2"]
    methods = [m for m in methods if m in df_g1["method"].unique()]
    df = df_g1[df_g1["method"].isin(methods) & df_g1["path"].isin(paths)]

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.6, 1.6], hspace=0.28)

    # Row 0: diagrams
    gs_d = gs[0].subgridspec(1, len(paths), wspace=0.12)
    for col, pname in enumerate(paths):
        ax = fig.add_subplot(gs_d[0, col])
        _draw_path_diagram(ax, pname)

    # Row 1: collision rate
    ax1 = fig.add_subplot(gs[1])
    # Row 2: missed mode rate
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    x = np.arange(len(paths))
    n_m = len(methods)
    w = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr, cr_lo, cr_hi, mm = [], [], [], []
        for p in paths:
            row = sub[sub["path"] == p]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
                mm.append(row.iloc[0]["missed_mode_rate"] * 100)
            else:
                cr.append(0); cr_lo.append(0); cr_hi.append(0); mm.append(0)

        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])
        offset = (i - n_m/2 + 0.5) * w

        ax1.bar(x + offset, cr, w*0.88, yerr=err, label=_n(method),
                color=_c(method), capsize=2, error_kw={"linewidth": 0.7},
                edgecolor="white", linewidth=0.3)
        ax2.bar(x + offset, np.array(mm), w*0.88,
                color=_c(method), edgecolor="white", linewidth=0.3)

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("(a) Collision Rate", fontsize=9.5)
    ax1.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9,
               edgecolor="0.85")
    ax1.tick_params(labelbottom=False)

    ax2.set_ylabel("Missed-Mode Rate (%)")
    ax2.set_title("(b) Missed-Mode Rate", fontsize=9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(paths, fontsize=10)

    fig.suptitle("WDRO Reduces Collision and Missed-Mode Rates across Path Geometries",
                 fontsize=11, y=0.995)
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optA_fig1_path_geometry.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option A, Fig 1 saved")


def option_a_fig2(df_g6):
    """Challenging environments: diagrams + collision rate + mean clearance."""
    challenges = ["Crossing", "Diagonal", "Mixed-2obs", "Mixed-3obs", "Dense-4obs"]
    challenges = [c for c in challenges if c in df_g6["challenge"].unique()]
    methods = ["Base", "WDRO-sampling", "WDRO-inject-K1", "WDRO-inject-K2"]
    methods = [m for m in methods if m in df_g6["method"].unique()]
    df = df_g6[df_g6["method"].isin(methods) & df_g6["challenge"].isin(challenges)]

    nice_ch = {"Crossing": "Crossing", "Diagonal": "Diagonal",
               "Mixed-2obs": "Mixed (2)", "Mixed-3obs": "Mixed (3)",
               "Dense-4obs": "Dense (4)"}

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.6, 1.6], hspace=0.28)

    # Row 0: environment diagrams
    gs_d = gs[0].subgridspec(1, len(challenges), wspace=0.08)
    for col, ch in enumerate(challenges):
        ax = fig.add_subplot(gs_d[0, col])
        _draw_challenge_diagram(ax, ch)

    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)

    x = np.arange(len(challenges))
    n_m = len(methods)
    w = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        cr, cr_lo, cr_hi, clr = [], [], [], []
        for ch in challenges:
            row = sub[sub["challenge"] == ch]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
                clr.append(row.iloc[0]["mean_clearance"])
            else:
                cr.append(0); cr_lo.append(0); cr_hi.append(0); clr.append(0)

        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])
        offset = (i - n_m/2 + 0.5) * w

        ax1.bar(x + offset, cr, w*0.88, yerr=err, label=_n(method),
                color=_c(method), capsize=2, error_kw={"linewidth": 0.7},
                edgecolor="white", linewidth=0.3)
        ax2.bar(x + offset, np.array(clr), w*0.88,
                color=_c(method), edgecolor="white", linewidth=0.3)

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("(a) Collision Rate", fontsize=9.5)
    ax1.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.9,
               edgecolor="0.85")
    ax1.tick_params(labelbottom=False)

    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("(b) Mean Clearance", fontsize=9.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([nice_ch.get(c, c) for c in challenges], fontsize=9)

    fig.suptitle("WDRO Improves Safety and Clearance in Challenging Multi-Agent Settings",
                 fontsize=11, y=0.995)
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optA_fig2_challenging_envs.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option A, Fig 2 saved")


def option_a_fig3(df_g3):
    """Switching dynamics sweep: line plots with CI bands."""
    methods = ["Base", "WDRO-sampling", "WDRO-inject-K1", "TopRisk-K1"]
    methods = [m for m in methods if m in df_g3["method"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method in methods:
        sub = df_g3[df_g3["method"] == method].sort_values("switch_prob")
        if sub.empty:
            continue
        sp = sub["switch_prob"].values
        cr = sub["collision_rate"].values * 100
        clo = sub["coll_ci_lo"].values * 100
        chi = sub["coll_ci_hi"].values * 100
        mm = sub["missed_mode_rate"].values * 100

        style = dict(color=_c(method), marker=_m(method), ms=5, lw=1.5)
        ax1.plot(sp, cr, label=_n(method), **style)
        ax1.fill_between(sp, clo, chi, alpha=0.12, color=_c(method))
        ax2.plot(sp, mm, label=_n(method), **style)

    ax1.set_xlabel("Switch Probability")
    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate vs. Switching Frequency", fontweight="bold")
    ax1.legend(fontsize=7.5, framealpha=0.9)

    ax2.set_xlabel("Switch Probability")
    ax2.set_ylabel("Missed-Mode Rate (%)")
    ax2.set_title("Missed-Mode Rate vs. Switching Frequency", fontweight="bold")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optA_fig3_switch_dynamics.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option A, Fig 3 saved")


def option_a_fig4(df_g1, df_g6):
    """Summary table figure: collision rate reduction across all settings."""
    # Combine path geometry and challenging env results
    rows = []

    # Path geometry
    for p in ["Straight", "Gentle-S", "S-curve", "Tight-S"]:
        for method in ["Base", "WDRO-sampling", "WDRO-inject-K1", "TopRisk-K1"]:
            row = df_g1[(df_g1["path"] == p) & (df_g1["method"] == method)]
            if len(row):
                rows.append({
                    "Setting": p, "Method": method,
                    "Collision %": f"{row.iloc[0]['collision_rate']*100:.1f}",
                    "Missed-Mode %": f"{row.iloc[0]['missed_mode_rate']*100:.1f}",
                    "Clearance (m)": f"{row.iloc[0]['mean_clearance']:.2f}",
                })

    nice_ch = {"Crossing": "Crossing", "Diagonal": "Diagonal",
               "Mixed-2obs": "Mixed (2)", "Mixed-3obs": "Mixed (3)",
               "Dense-4obs": "Dense (4)"}
    for ch in ["Crossing", "Diagonal", "Mixed-2obs", "Mixed-3obs", "Dense-4obs"]:
        for method in ["Base", "WDRO-sampling", "WDRO-inject-K1", "TopRisk-K1"]:
            row = df_g6[(df_g6["challenge"] == ch) & (df_g6["method"] == method)]
            if len(row):
                rows.append({
                    "Setting": nice_ch.get(ch, ch), "Method": method,
                    "Collision %": f"{row.iloc[0]['collision_rate']*100:.1f}",
                    "Missed-Mode %": f"{row.iloc[0]['missed_mode_rate']*100:.1f}",
                    "Clearance (m)": f"{row.iloc[0]['mean_clearance']:.2f}",
                })

    if not rows:
        print("  Option A, Fig 4: no data, skipping")
        return

    tab_df = pd.DataFrame(rows)

    # Pivot for a clean table
    settings = tab_df["Setting"].unique()
    methods_order = ["Base", "WDRO-sampling", "WDRO-inject-K1", "TopRisk-K1"]

    fig, ax = plt.subplots(figsize=(12, 0.38 * len(settings) + 2.0))
    ax.axis("off")

    cell_text = []
    row_colors = []
    for s in settings:
        sub = tab_df[tab_df["Setting"] == s]
        base_cr = sub[sub["Method"] == "Base"]
        base_val = float(base_cr["Collision %"].values[0]) if len(base_cr) else 0

        row = [s]
        for m in methods_order:
            r = sub[sub["Method"] == m]
            if len(r):
                row.append(f"{r.iloc[0]['Collision %']}  /  {r.iloc[0]['Missed-Mode %']}  /  {r.iloc[0]['Clearance (m)']}")
            else:
                row.append("--")
        cell_text.append(row)
        row_colors.append("#f5f5f5" if len(cell_text) % 2 == 0 else "white")

    col_labels = ["Setting"] + [_n(m) for m in methods_order]
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2166AC")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i, rc in enumerate(row_colors):
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(rc)

    ax.set_title("Summary: Collision % / Missed-Mode % / Clearance (m) across all settings",
                 fontsize=10, fontweight="bold", pad=15)

    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optA_fig4_summary_table.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option A, Fig 4 saved")


# ============================================================================
# OPTION B: Narrative-Focused (compact, story-driven)
#   Fig 1: Collision rate reduction — horizontal bar chart (forest-plot style)
#   Fig 2: Path geometry — side-by-side collision + missed mode (compact)
#   Fig 3: Challenging environments — collision + clearance (compact)
#   Fig 4: Injection comparison — WDRO-K vs TopRisk-K across all settings
# ============================================================================

def option_b_fig1(df_g1, df_g6):
    """Forest-plot style: collision rate reduction across all settings."""
    methods_show = ["WDRO-sampling", "WDRO-inject-K1"]
    settings = []
    reductions = {m: [] for m in methods_show}
    base_rates = []
    ci_lo = {m: [] for m in methods_show}
    ci_hi = {m: [] for m in methods_show}

    # Path geometries
    for p in ["Straight", "Gentle-S", "S-curve", "Tight-S"]:
        base = df_g1[(df_g1["path"] == p) & (df_g1["method"] == "Base")]
        if base.empty:
            continue
        br = base.iloc[0]["collision_rate"] * 100
        base_rates.append(br)
        settings.append(p)
        for m in methods_show:
            row = df_g1[(df_g1["path"] == p) & (df_g1["method"] == m)]
            if len(row):
                mr = row.iloc[0]["collision_rate"] * 100
                reductions[m].append(br - mr)
                ci_lo[m].append(br - row.iloc[0]["coll_ci_hi"] * 100)
                ci_hi[m].append(br - row.iloc[0]["coll_ci_lo"] * 100)
            else:
                reductions[m].append(0)
                ci_lo[m].append(0)
                ci_hi[m].append(0)

    # Challenging envs
    nice_ch = {"Crossing": "Crossing", "Diagonal": "Diagonal",
               "Mixed-2obs": "Mixed (2)", "Mixed-3obs": "Mixed (3)",
               "Dense-4obs": "Dense (4)"}
    for ch in ["Crossing", "Diagonal", "Mixed-2obs", "Mixed-3obs", "Dense-4obs"]:
        base = df_g6[(df_g6["challenge"] == ch) & (df_g6["method"] == "Base")]
        if base.empty:
            continue
        br = base.iloc[0]["collision_rate"] * 100
        base_rates.append(br)
        settings.append(nice_ch.get(ch, ch))
        for m in methods_show:
            row = df_g6[(df_g6["challenge"] == ch) & (df_g6["method"] == m)]
            if len(row):
                mr = row.iloc[0]["collision_rate"] * 100
                reductions[m].append(br - mr)
                ci_lo[m].append(br - row.iloc[0]["coll_ci_hi"] * 100)
                ci_hi[m].append(br - row.iloc[0]["coll_ci_lo"] * 100)
            else:
                reductions[m].append(0)
                ci_lo[m].append(0)
                ci_hi[m].append(0)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    y = np.arange(len(settings))
    h = 0.35

    for i, m in enumerate(methods_show):
        offset = (i - 0.5) * h
        vals = np.array(reductions[m])
        lo = np.array(ci_lo[m])
        hi = np.array(ci_hi[m])
        err = np.array([vals - lo, hi - vals])
        err = np.clip(err, 0, None)
        ax.barh(y + offset, vals, h * 0.9, xerr=err, label=_n(m),
                color=_c(m), capsize=2, error_kw={"linewidth": 0.7},
                edgecolor="white", linewidth=0.3)

    # Add base collision rate annotations on right
    for j, (s, br) in enumerate(zip(settings, base_rates)):
        ax.text(max(max(reductions["WDRO-sampling"]), max(reductions["WDRO-inject-K1"])) + 5,
                j, f"Base: {br:.0f}%", va="center", ha="left", fontsize=7.5, color="#888")

    ax.set_yticks(y)
    ax.set_yticklabels(settings)
    ax.set_xlabel("Collision Rate Reduction (pp)")
    ax.set_title("Collision Rate Improvement over Base SH-MPCC", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.5, ls="--")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optB_fig1_forest_plot.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option B, Fig 1 saved")


def option_b_fig2(df_g1):
    """Compact path geometry: 2x2 grid, one panel per path, bars for methods."""
    paths = [p for p in ["Straight", "Gentle-S", "S-curve", "Tight-S"]
             if p in df_g1["path"].unique()]
    methods = ["Base", "WDRO-sampling", "WDRO-inject-K1", "TopRisk-K1"]
    methods = [m for m in methods if m in df_g1["method"].unique()]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey="row")
    axes_flat = axes.flatten()

    for idx, p in enumerate(paths):
        ax = axes_flat[idx]
        sub = df_g1[(df_g1["path"] == p) & df_g1["method"].isin(methods)]

        x = np.arange(len(methods))
        cr = [sub[sub["method"] == m].iloc[0]["collision_rate"] * 100
              if len(sub[sub["method"] == m]) else 0 for m in methods]
        cr_lo = [sub[sub["method"] == m].iloc[0]["coll_ci_lo"] * 100
                 if len(sub[sub["method"] == m]) else 0 for m in methods]
        cr_hi = [sub[sub["method"] == m].iloc[0]["coll_ci_hi"] * 100
                 if len(sub[sub["method"] == m]) else 0 for m in methods]
        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])

        bars = ax.bar(x, cr, 0.65, yerr=err, color=[_c(m) for m in methods],
                      capsize=3, error_kw={"linewidth": 0.8}, edgecolor="white", linewidth=0.3)
        ax.set_title(p, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([_n(m) for m in methods], fontsize=7, rotation=25, ha="right")
        if idx % 2 == 0:
            ax.set_ylabel("Collision Rate (%)")

    fig.suptitle("Collision Rate across Path Geometries (N=500)", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optB_fig2_path_panels.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option B, Fig 2 saved")


def option_b_fig3(df_g6):
    """Compact challenging envs: collision + clearance in 2-row layout."""
    challenges = [c for c in ["Crossing", "Diagonal", "Mixed-2obs", "Mixed-3obs", "Dense-4obs"]
                  if c in df_g6["challenge"].unique()]
    methods = ["Base", "WDRO-sampling", "WDRO-inject-K1", "TopRisk-K1"]
    methods = [m for m in methods if m in df_g6["method"].unique()]
    nice_ch = {"Crossing": "Crossing", "Diagonal": "Diagonal",
               "Mixed-2obs": "Mixed (2)", "Mixed-3obs": "Mixed (3)",
               "Dense-4obs": "Dense (4)"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    x = np.arange(len(challenges))
    n_m = len(methods)
    w = 0.8 / n_m

    for i, method in enumerate(methods):
        sub = df_g6[df_g6["method"] == method]
        cr, cr_lo, cr_hi, clr = [], [], [], []
        for ch in challenges:
            row = sub[sub["challenge"] == ch]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
                clr.append(row.iloc[0]["mean_clearance"])
            else:
                cr.append(0); cr_lo.append(0); cr_hi.append(0); clr.append(0)

        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])
        offset = (i - n_m/2 + 0.5) * w

        ax1.bar(x + offset, cr, w*0.88, yerr=err, label=_n(method),
                color=_c(method), capsize=2, error_kw={"linewidth": 0.7},
                edgecolor="white", linewidth=0.3)
        ax2.bar(x + offset, np.array(clr), w*0.88,
                color=_c(method), edgecolor="white", linewidth=0.3)

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate in Challenging Environments", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7.5, ncol=2)

    ax2.set_ylabel("Mean Clearance (m)")
    ax2.set_title("Mean Clearance in Challenging Environments", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([nice_ch.get(c, c) for c in challenges], fontsize=9)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optB_fig3_challenging_compact.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option B, Fig 3 saved")


def option_b_fig4(df_g1, df_g6):
    """Injection comparison: WDRO vs TopRisk at K=1,2,3 across all 9 settings."""
    methods = [
        ("WDRO-inject-K1", "#2166AC"),
        ("WDRO-inject-K2", "#4393C3"),
        ("WDRO-inject-K3", "#92C5DE"),
        ("TopRisk-K1",     "#B2182B"),
        ("TopRisk-K2",     "#D6604D"),
        ("TopRisk-K3",     "#F4A582"),
    ]

    paths = ["Straight", "Gentle-S", "S-curve", "Tight-S"]
    challenges = ["Crossing", "Diagonal", "Mixed-2obs", "Mixed-3obs", "Dense-4obs"]
    nice_ch = {"Crossing": "Cross.", "Diagonal": "Diag.",
               "Mixed-2obs": "Mix(2)", "Mixed-3obs": "Mix(3)",
               "Dense-4obs": "Dense"}
    settings = paths + [nice_ch.get(c, c) for c in challenges]

    fig, ax = plt.subplots(figsize=(13, 5))

    x = np.arange(len(settings))
    n_m = len(methods)
    w = 0.8 / n_m

    for i, (mname, mcolor) in enumerate(methods):
        cr, cr_lo, cr_hi = [], [], []

        for p in paths:
            row = df_g1[(df_g1["path"] == p) & (df_g1["method"] == mname)]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
            else:
                cr.append(np.nan); cr_lo.append(np.nan); cr_hi.append(np.nan)

        for ch in challenges:
            row = df_g6[(df_g6["challenge"] == ch) & (df_g6["method"] == mname)]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
            else:
                cr.append(np.nan); cr_lo.append(np.nan); cr_hi.append(np.nan)

        cr = np.array(cr, dtype=float)
        lo = np.array(cr_lo, dtype=float)
        hi = np.array(cr_hi, dtype=float)
        err = np.array([cr - lo, hi - cr])
        err = np.nan_to_num(err, nan=0.0)
        offset = (i - n_m/2 + 0.5) * w

        # Short label for legend
        short = mname.replace("WDRO-inject-", "WDRO-").replace("TopRisk-", "TopRisk-")
        mask = ~np.isnan(cr)
        ax.bar(x[mask] + offset, cr[mask], w*0.88, yerr=err[:, mask],
               label=short, color=mcolor,
               capsize=2, error_kw={"linewidth": 0.7},
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Collision Rate (%)")
    n_rollouts = int(df_g1["n_rollouts"].iloc[0]) if "n_rollouts" in df_g1.columns else 500
    ax.set_title(f"Injection Selection Method Does Not Significantly Affect Collision Rate  (N={n_rollouts})",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9)
    ax.axvline(3.5, color="gray", lw=0.5, ls=":")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optB_fig4_injection_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option B, Fig 4 saved")


# ============================================================================
# OPTION C: Maximum Information Density (3 figures, packed with data)
#   Fig 1: Combined path geometry + challenging envs in single multi-panel
#   Fig 2: Mechanism figure — missed mode rate vs collision rate scatter
#   Fig 3: Switching dynamics + injection comparison combined
# ============================================================================

def option_c_fig1(df_g1, df_g6):
    """Combined: collision rate across ALL settings in one figure."""
    methods = ["Base", "WDRO-sampling", "WDRO-inject-K1", "TopRisk-K1"]
    methods = [m for m in methods if m in df_g1["method"].unique()]

    paths = [p for p in ["Straight", "Gentle-S", "S-curve", "Tight-S"]
             if p in df_g1["path"].unique()]
    challenges = [c for c in ["Crossing", "Diagonal", "Mixed-2obs", "Mixed-3obs", "Dense-4obs"]
                  if c in df_g6["challenge"].unique()]
    nice_ch = {"Crossing": "Crossing", "Diagonal": "Diagonal",
               "Mixed-2obs": "Mixed (2)", "Mixed-3obs": "Mixed (3)",
               "Dense-4obs": "Dense (4)"}

    all_settings = paths + [nice_ch.get(c, c) for c in challenges]
    n_settings = len(all_settings)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    x = np.arange(n_settings)
    n_m = len(methods)
    w = 0.8 / n_m

    for i, method in enumerate(methods):
        cr, cr_lo, cr_hi, mm = [], [], [], []

        for p in paths:
            row = df_g1[(df_g1["path"] == p) & (df_g1["method"] == method)]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
                mm.append(row.iloc[0]["missed_mode_rate"] * 100)
            else:
                cr.append(0); cr_lo.append(0); cr_hi.append(0); mm.append(0)

        for ch in challenges:
            row = df_g6[(df_g6["challenge"] == ch) & (df_g6["method"] == method)]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
                mm.append(row.iloc[0]["missed_mode_rate"] * 100)
            else:
                cr.append(0); cr_lo.append(0); cr_hi.append(0); mm.append(0)

        cr = np.array(cr)
        err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])
        offset = (i - n_m/2 + 0.5) * w

        ax1.bar(x + offset, cr, w*0.88, yerr=err, label=_n(method),
                color=_c(method), capsize=2, error_kw={"linewidth": 0.7},
                edgecolor="white", linewidth=0.3)
        ax2.bar(x + offset, np.array(mm), w*0.88,
                color=_c(method), edgecolor="white", linewidth=0.3)

    # Divider
    div_x = len(paths) - 0.5
    ax1.axvline(div_x, color="gray", lw=0.8, ls=":")
    ax2.axvline(div_x, color="gray", lw=0.8, ls=":")
    ax1.text(len(paths)/2 - 0.5, ax1.get_ylim()[1]*0.93, "Path Geometry",
             ha="center", fontsize=8, fontstyle="italic", color="gray")
    ax1.text(len(paths) + len(challenges)/2 - 0.5, ax1.get_ylim()[1]*0.93,
             "Challenging Environments", ha="center", fontsize=8, fontstyle="italic", color="gray")

    ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("Collision Rate across All Settings", fontweight="bold", fontsize=11)
    ax1.legend(loc="upper left", fontsize=7.5, ncol=2, framealpha=0.9)
    ax1.tick_params(labelbottom=False)

    ax2.set_ylabel("Missed-Mode Rate (%)")
    ax2.set_title("Missed-Mode Rate across All Settings", fontweight="bold", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_settings, fontsize=8.5, rotation=20, ha="right")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optC_fig1_combined_all.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option C, Fig 1 saved")


def option_c_fig2(df_g1, df_g6):
    """Mechanism scatter: missed-mode rate vs collision rate, each point = one (setting, method)."""
    methods = ["Base", "WDRO-sampling", "WDRO-inject-K1"]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for method in methods:
        mm_vals, cr_vals = [], []

        for p in df_g1["path"].unique():
            if p == "Circle":
                continue
            row = df_g1[(df_g1["path"] == p) & (df_g1["method"] == method)]
            if len(row):
                mm_vals.append(row.iloc[0]["missed_mode_rate"] * 100)
                cr_vals.append(row.iloc[0]["collision_rate"] * 100)

        for ch in df_g6["challenge"].unique():
            if "HighSpeed" in ch:
                continue
            row = df_g6[(df_g6["challenge"] == ch) & (df_g6["method"] == method)]
            if len(row):
                mm_vals.append(row.iloc[0]["missed_mode_rate"] * 100)
                cr_vals.append(row.iloc[0]["collision_rate"] * 100)

        ax.scatter(mm_vals, cr_vals, label=_n(method), color=_c(method),
                   marker=_m(method), s=60, alpha=0.8, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Missed-Mode Rate (%)")
    ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Missed-Mode Rate Predicts Collision Rate across Methods and Settings",
                 fontsize=10.5)
    ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optC_fig2_mechanism_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option C, Fig 2 saved")


def option_c_fig3(df_g3, df_g1, df_g6):
    """Injection comparison: WDRO vs TopRisk at K=1,2,3 across all 9 settings (N=1000)."""
    # Filter for 1000-rollout data if available
    if "n_rollouts" in df_g1.columns:
        df_g1_f = df_g1[df_g1["n_rollouts"] == 1000]
        if df_g1_f.empty:
            df_g1_f = df_g1  # fall back to whatever is available
    else:
        df_g1_f = df_g1
    if "n_rollouts" in df_g6.columns:
        df_g6_f = df_g6[df_g6["n_rollouts"] == 1000]
        if df_g6_f.empty:
            df_g6_f = df_g6
    else:
        df_g6_f = df_g6

    inj_methods = [
        "WDRO-inject-K1", "WDRO-inject-K2",
        "TopRisk-K1",     "TopRisk-K2",
    ]
    # Keep only methods present in the data
    avail = set(df_g1_f["method"].unique()) | set(df_g6_f["method"].unique())
    inj_methods = [m for m in inj_methods if m in avail]

    paths = ["Straight", "Gentle-S", "S-curve", "Tight-S"]
    challenges = ["Crossing", "Diagonal", "Mixed-2obs", "Mixed-3obs", "Dense-4obs"]
    nice_ch = {"Crossing": "Cross.", "Diagonal": "Diag.",
               "Mixed-2obs": "Mix(2)", "Mixed-3obs": "Mix(3)",
               "Dense-4obs": "Dense"}
    settings = paths + [nice_ch.get(c, c) for c in challenges]

    fig, ax = plt.subplots(figsize=(13, 5))

    x = np.arange(len(settings))
    n_m = len(inj_methods)
    w = 0.8 / n_m

    for i, mname in enumerate(inj_methods):
        cr, cr_lo, cr_hi = [], [], []

        for p in paths:
            row = df_g1_f[(df_g1_f["path"] == p) & (df_g1_f["method"] == mname)]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
            else:
                cr.append(np.nan); cr_lo.append(np.nan); cr_hi.append(np.nan)

        for ch in challenges:
            row = df_g6_f[(df_g6_f["challenge"] == ch) & (df_g6_f["method"] == mname)]
            if len(row):
                cr.append(row.iloc[0]["collision_rate"] * 100)
                cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
                cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
            else:
                cr.append(np.nan); cr_lo.append(np.nan); cr_hi.append(np.nan)

        cr = np.array(cr, dtype=float)
        lo = np.array(cr_lo, dtype=float)
        hi = np.array(cr_hi, dtype=float)
        err = np.array([cr - lo, hi - cr])
        err = np.nan_to_num(err, nan=0.0)
        offset = (i - n_m/2 + 0.5) * w

        mask = ~np.isnan(cr)
        ax.bar(x[mask] + offset, cr[mask], w*0.88, yerr=err[:, mask],
               label=_n(mname), color=_c(mname),
               capsize=2, error_kw={"linewidth": 0.7},
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Collision Rate (%)")
    n_rollouts = int(df_g1_f["n_rollouts"].iloc[0]) if "n_rollouts" in df_g1_f.columns else "?"
    ax.set_title(f"Injection Selection Method Does Not Significantly Affect Collision Rate  (N={n_rollouts})",
                 fontsize=10.5)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", framealpha=0.9)
    ax.axvline(3.5, color="gray", lw=0.5, ls=":")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"optC_fig3_sweep_injection.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  Option C, Fig 3 saved")


# ============================================================================
# Main
# ============================================================================

def load_data():
    g1_path = _find_csv("g1_path_geometry.csv")
    g3_path = _find_csv("g3_switch_dynamics.csv")
    g6_path = _find_csv("g6_challenging_envs.csv")

    df_g1 = pd.read_csv(g1_path) if g1_path else None
    df_g3 = pd.read_csv(g3_path) if g3_path else None
    df_g6 = pd.read_csv(g6_path) if g6_path else None

    print(f"  G1 data: {'loaded' if df_g1 is not None else 'NOT FOUND'}")
    print(f"  G3 data: {'loaded' if df_g3 is not None else 'NOT FOUND'}")
    print(f"  G6 data: {'loaded' if df_g6 is not None else 'NOT FOUND'}")

    return df_g1, df_g3, df_g6


def run_option_a(df_g1, df_g3, df_g6):
    print("\n=== OPTION A: Paper-Aligned (diagrams + bars + sweep + table) ===")
    if df_g1 is not None:
        option_a_fig1(df_g1)
    if df_g6 is not None:
        option_a_fig2(df_g6)
    if df_g3 is not None:
        option_a_fig3(df_g3)
    if df_g1 is not None and df_g6 is not None:
        option_a_fig4(df_g1, df_g6)


def run_option_b(df_g1, df_g3, df_g6):
    print("\n=== OPTION B: Narrative-Focused (forest plot + panels + compact + injection) ===")
    if df_g1 is not None and df_g6 is not None:
        option_b_fig1(df_g1, df_g6)
    if df_g1 is not None:
        option_b_fig2(df_g1)
    if df_g6 is not None:
        option_b_fig3(df_g6)
    if df_g1 is not None and df_g6 is not None:
        option_b_fig4(df_g1, df_g6)


def run_option_c(df_g1, df_g3, df_g6):
    print("\n=== OPTION C: Maximum Density (combined + scatter + sweep+injection) ===")
    if df_g1 is not None and df_g6 is not None:
        option_c_fig1(df_g1, df_g6)
    if df_g1 is not None and df_g6 is not None:
        option_c_fig2(df_g1, df_g6)
    if df_g1 is not None and df_g6 is not None:
        option_c_fig3(None, df_g1, df_g6)


if __name__ == "__main__":
    df_g1, df_g3, df_g6 = load_data()

    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("option_a", "a", "all"):
        run_option_a(df_g1, df_g3, df_g6)
    if which in ("option_b", "b", "all"):
        run_option_b(df_g1, df_g3, df_g6)
    if which in ("option_c", "c", "all"):
        run_option_c(df_g1, df_g3, df_g6)

    print(f"\nAll figures saved to: {OUT}/")
    print("Files:")
    for f in sorted(OUT.glob("opt*.*")):
        print(f"  {f.name}")
