#!/usr/bin/env python3
"""
Generate figures from the path / obstacle / behavior sweep experiment.

Reads:  paper_figures/path_obstacle_sweep.csv   (summary)
        paper_figures/path_obstacle_raw.csv     (per-rollout)
Writes: paper_figures/fig_path_*.pdf
        paper_figures/fig_obs_count_*.pdf
        paper_figures/fig_behavior_*.pdf
        paper_figures/fig_full_sweep_*.pdf
        paper_figures/path_obstacle_findings.tex
"""

import os
import sys
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
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

DATA_DIR = "paper_figures/"
OUT_DIR = "paper_figures/"

# CDC strategies (no OT)
VARIANT_ORDER = ["Base", "SH", "DRO(inj)", "DRO(inj)+SH", "DRO(q*)+SH"]
VARIANT_COLORS = {
    "Base":         "#7f7f7f",
    "SH":           "#1f77b4",
    "DRO(inj)":     "#2ca02c",
    "DRO(inj)+SH":  "#d62728",
    "DRO(q*)+SH":   "#ff7f0e",
}
VARIANT_MARKERS = {
    "Base": "o", "SH": "s", "DRO(inj)": "^",
    "DRO(inj)+SH": "D", "DRO(q*)+SH": "P",
}
VARIANT_LABELS = {
    "Base":         "Base",
    "SH":           "SH",
    "DRO(inj)":     "DRO(inj)",
    "DRO(inj)+SH":  "DRO(inj)+SH",
    "DRO(q*)+SH":   r"DRO($q^*$)+SH",
}

PATH_ORDER = ["straight", "s_curve", "chicane", "wide_arc", "hairpin"]
PATH_LABELS = {
    "straight": "Straight", "s_curve": "S-Curve", "chicane": "Chicane",
    "wide_arc": "Wide Arc", "hairpin": "Hairpin",
}

BEHAVIOR_ORDER = ["conservative", "balanced", "aggressive", "erratic"]
BEHAVIOR_LABELS = {
    "conservative": "Conservative", "balanced": "Balanced",
    "aggressive": "Aggressive", "erratic": "Erratic",
}


def load_summary():
    path = os.path.join(DATA_DIR, "path_obstacle_sweep.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found"); sys.exit(1)
    return pd.read_csv(path)


def load_raw():
    path = os.path.join(DATA_DIR, "path_obstacle_raw.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def save_fig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> {name}")


def get_variants_in_data(df):
    """Return variants present in data, in canonical order."""
    present = set(df["variant"].unique())
    return [v for v in VARIANT_ORDER if v in present]


# ============================================================================
# Fig 1: Collision rate & mode coverage by path shape (grouped bars)
# ============================================================================

def fig_path_collision(df):
    sub = df[df["path_shape"].isin(PATH_ORDER)]
    if sub.empty: return
    variants = get_variants_in_data(sub)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(PATH_ORDER))
    n_var = len(variants)
    width = 0.8 / n_var

    for i, v in enumerate(variants):
        cr, mc, cr_lo, cr_hi, mc_lo, mc_hi = [], [], [], [], [], []
        for ps in PATH_ORDER:
            row = sub[(sub["variant"] == v) & (sub["path_shape"] == ps)]
            if row.empty:
                cr.append(0); mc.append(0)
                cr_lo.append(0); cr_hi.append(0); mc_lo.append(0); mc_hi.append(0)
            else:
                r = row.iloc[0]
                cr.append(r["collision_rate"] * 100)
                mc.append(r["mode_coverage"] * 100)
                cr_lo.append((r["collision_rate"] - r["collision_ci_lo"]) * 100)
                cr_hi.append((r["collision_ci_hi"] - r["collision_rate"]) * 100)
                mc_lo.append((r["mode_coverage"] - r["mode_cov_ci_lo"]) * 100)
                mc_hi.append((r["mode_cov_ci_hi"] - r["mode_coverage"]) * 100)

        offset = (i - n_var / 2 + 0.5) * width
        c = VARIANT_COLORS.get(v, "#333")
        lab = VARIANT_LABELS.get(v, v)
        ax1.bar(x + offset, cr, width, yerr=[cr_lo, cr_hi], label=lab,
                color=c, capsize=2, edgecolor="black", linewidth=0.4, alpha=0.85)
        ax2.bar(x + offset, mc, width, yerr=[mc_lo, mc_hi], label=lab,
                color=c, capsize=2, edgecolor="black", linewidth=0.4, alpha=0.85)

    labels = [PATH_LABELS.get(p, p) for p in PATH_ORDER]
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("Collision Rate (%)"); ax1.set_title("(a) Collision Rate by Path Shape")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3, axis="y"); ax1.set_ylim(bottom=0)

    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Mode Coverage (%)"); ax2.set_title("(b) Mode Coverage by Path Shape")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig_path_collision_coverage")


# ============================================================================
# Fig 2: Solve time and completion by path shape
# ============================================================================

def fig_path_perf(df):
    sub = df[df["path_shape"].isin(PATH_ORDER)]
    if sub.empty: return
    variants = get_variants_in_data(sub)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(PATH_ORDER))
    n_var = len(variants)
    width = 0.8 / n_var

    for i, v in enumerate(variants):
        st, comp = [], []
        for ps in PATH_ORDER:
            row = sub[(sub["variant"] == v) & (sub["path_shape"] == ps)]
            if row.empty: st.append(0); comp.append(0)
            else:
                r = row.iloc[0]
                st.append(r["mean_solve_ms"]); comp.append(r["completion_rate"] * 100)
        offset = (i - n_var / 2 + 0.5) * width
        c = VARIANT_COLORS.get(v, "#333")
        lab = VARIANT_LABELS.get(v, v)
        ax1.bar(x + offset, st, width, label=lab, color=c,
                edgecolor="black", linewidth=0.4, alpha=0.85)
        ax2.bar(x + offset, comp, width, label=lab, color=c,
                edgecolor="black", linewidth=0.4, alpha=0.85)

    labels = [PATH_LABELS.get(p, p) for p in PATH_ORDER]
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel("Avg Solve Time (ms)"); ax1.set_title("(a) Solve Time by Path Shape")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3, axis="y"); ax1.set_ylim(bottom=0)

    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Completion Rate (%)"); ax2.set_title("(b) Path Completion by Path Shape")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3, axis="y"); ax2.set_ylim(0, 105)

    fig.tight_layout()
    save_fig(fig, "fig_path_performance")


# ============================================================================
# Fig 3: Collision rate by obstacle count (line plot)
# ============================================================================

def fig_obs_count(df):
    obs_vals = sorted(df["num_obstacles"].unique())
    if len(obs_vals) < 2: return
    variants = get_variants_in_data(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for v in variants:
        x_vals, cr_vals, mc_vals, cr_lo, cr_hi = [], [], [], [], []
        for nobs in obs_vals:
            row = df[(df["variant"] == v) & (df["num_obstacles"] == nobs)]
            if row.empty: continue
            r = row.iloc[0]
            x_vals.append(nobs)
            cr_vals.append(r["collision_rate"] * 100)
            mc_vals.append(r["mode_coverage"] * 100)
            cr_lo.append((r["collision_rate"] - r["collision_ci_lo"]) * 100)
            cr_hi.append((r["collision_ci_hi"] - r["collision_rate"]) * 100)
        if not x_vals: continue
        c = VARIANT_COLORS.get(v, "#333")
        m = VARIANT_MARKERS.get(v, "o")
        lab = VARIANT_LABELS.get(v, v)
        ax1.errorbar(x_vals, cr_vals, yerr=[cr_lo, cr_hi], label=lab,
                     color=c, marker=m, capsize=4, linewidth=2, markersize=8)
        ax2.plot(x_vals, mc_vals, label=lab, color=c, marker=m, linewidth=2, markersize=8)

    ax1.set_xlabel("Number of Obstacles"); ax1.set_ylabel("Collision Rate (%)")
    ax1.set_title("(a) Collision Rate vs Obstacle Count")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(bottom=0); ax1.set_xticks(obs_vals)

    ax2.set_xlabel("Number of Obstacles"); ax2.set_ylabel("Mode Coverage (%)")
    ax2.set_title("(b) Mode Coverage vs Obstacle Count")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_xticks(obs_vals)

    fig.tight_layout()
    save_fig(fig, "fig_obs_count_sweep")


# ============================================================================
# Fig 4: Behavior profile comparison (grouped bars)
# ============================================================================

def fig_behavior(df):
    sub = df[df["behavior"].isin(BEHAVIOR_ORDER)]
    if sub.empty: return
    variants = get_variants_in_data(sub)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    x = np.arange(len(BEHAVIOR_ORDER))
    n_var = len(variants)
    width = 0.8 / n_var

    for i, v in enumerate(variants):
        cr, mc, cl, cr_lo, cr_hi = [], [], [], [], []
        for b in BEHAVIOR_ORDER:
            row = sub[(sub["variant"] == v) & (sub["behavior"] == b)]
            if row.empty:
                cr.append(0); mc.append(0); cl.append(0); cr_lo.append(0); cr_hi.append(0)
            else:
                r = row.iloc[0]
                cr.append(r["collision_rate"] * 100); mc.append(r["mode_coverage"] * 100)
                cl.append(r["mean_clearance_5pct"])
                cr_lo.append((r["collision_rate"] - r["collision_ci_lo"]) * 100)
                cr_hi.append((r["collision_ci_hi"] - r["collision_rate"]) * 100)
        offset = (i - n_var / 2 + 0.5) * width
        c = VARIANT_COLORS.get(v, "#333"); lab = VARIANT_LABELS.get(v, v)

        axes[0].bar(x + offset, cr, width, yerr=[cr_lo, cr_hi], label=lab, color=c,
                    capsize=2, edgecolor="black", linewidth=0.4, alpha=0.85)
        axes[1].bar(x + offset, mc, width, label=lab, color=c,
                    edgecolor="black", linewidth=0.4, alpha=0.85)
        axes[2].bar(x + offset, cl, width, label=lab, color=c,
                    edgecolor="black", linewidth=0.4, alpha=0.85)

    labels = [BEHAVIOR_LABELS.get(b, b) for b in BEHAVIOR_ORDER]
    for ai, (ax, ylabel, title) in enumerate(zip(axes,
            ["Collision Rate (%)", "Mode Coverage (%)", "5th-Pct Clearance (m)"],
            ["(a) Collision Rate", "(b) Mode Coverage", "(c) Safety Margin (5th Pct)"])):
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3, axis="y")
        if ai != 1: ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_fig(fig, "fig_behavior_sweep")


# ============================================================================
# Fig 5: Safety vs Coverage scatter (all conditions)
# ============================================================================

def fig_safety_coverage_scatter(df):
    if df.empty: return
    variants = get_variants_in_data(df)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for v in variants:
        sub = df[df["variant"] == v]
        if sub.empty: continue
        ax.scatter(sub["mode_coverage"] * 100, sub["collision_rate"] * 100,
                   c=VARIANT_COLORS.get(v, "#333"), marker=VARIANT_MARKERS.get(v, "o"),
                   s=60, label=VARIANT_LABELS.get(v, v), alpha=0.7,
                   edgecolors="black", linewidth=0.3)

    ax.set_xlabel("Mode Coverage (%)"); ax.set_ylabel("Collision Rate (%)")
    ax.set_title("Safety vs Coverage (CDC Strategies)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_fig(fig, "fig_path_obs_safety_vs_coverage")


# ============================================================================
# Fig 6: Heatmap — path x behavior -> collision rate (one panel per variant)
# ============================================================================

def fig_heatmap(df):
    paths_present = [p for p in PATH_ORDER if p in df["path_shape"].values]
    behavs_present = [b for b in BEHAVIOR_ORDER if b in df["behavior"].values]
    if len(paths_present) < 2 or len(behavs_present) < 2: return
    variants = get_variants_in_data(df)

    n_var = len(variants)
    fig, axes = plt.subplots(1, n_var, figsize=(4.5 * n_var, 4.5))
    if n_var == 1: axes = [axes]

    for vi, v in enumerate(variants):
        ax = axes[vi]
        mat = np.full((len(behavs_present), len(paths_present)), np.nan)
        for ri, b in enumerate(behavs_present):
            for ci, p in enumerate(paths_present):
                row = df[(df["variant"] == v) & (df["path_shape"] == p) & (df["behavior"] == b)]
                if not row.empty:
                    mat[ri, ci] = row.iloc[0]["collision_rate"] * 100

        vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 20
        im = ax.imshow(mat, aspect="auto", origin="lower", vmin=0,
                       vmax=max(vmax, 1), cmap="YlOrRd")
        ax.set_xticks(range(len(paths_present)))
        ax.set_xticklabels([PATH_LABELS.get(p, p) for p in paths_present], rotation=30, fontsize=8)
        ax.set_yticks(range(len(behavs_present)))
        ax.set_yticklabels([BEHAVIOR_LABELS.get(b, b) for b in behavs_present], fontsize=8)
        ax.set_title(VARIANT_LABELS.get(v, v))
        for ri in range(len(behavs_present)):
            for ci in range(len(paths_present)):
                if not np.isnan(mat[ri, ci]):
                    ax.text(ci, ri, f"{mat[ri, ci]:.1f}", ha="center", va="center",
                            fontsize=8, color="white" if mat[ri, ci] > vmax * 0.6 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Collision %")

    fig.suptitle("Collision Rate: Path Shape x Behavior (CDC Strategies)", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_full_sweep_heatmap")


# ============================================================================
# Fig 7: Clearance distribution from raw data
# ============================================================================

def fig_clearance_dist(raw_df):
    if raw_df is None or raw_df.empty: return
    paths_present = [p for p in PATH_ORDER if p in raw_df["path_shape"].values]
    if not paths_present: return
    variants = get_variants_in_data(raw_df)

    fig, axes = plt.subplots(1, len(paths_present),
                              figsize=(4 * len(paths_present), 4), squeeze=False)
    axes = axes.flatten()
    for pi, ps in enumerate(paths_present):
        ax = axes[pi]
        for v in variants:
            sub = raw_df[(raw_df["variant"] == v) & (raw_df["path_shape"] == ps)]
            vals = sub["min_clearance"].dropna().values if not sub.empty else np.array([])
            if len(vals) == 0: continue
            ax.hist(vals, bins=30, alpha=0.4, label=VARIANT_LABELS.get(v, v),
                    color=VARIANT_COLORS.get(v, "#333"), density=True)
        ax.set_xlabel("Min Clearance (m)"); ax.set_ylabel("Density")
        ax.set_title(PATH_LABELS.get(ps, ps)); ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

    fig.suptitle("Clearance Distributions by Path Shape", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "fig_path_clearance_dist")


# ============================================================================
# Fig 8: Rare mode coverage (erratic behavior)
# ============================================================================

def fig_rare_coverage(df):
    sub = df[(df["behavior"] == "erratic") & df["path_shape"].isin(PATH_ORDER)]
    if sub.empty: return
    paths_in = [p for p in PATH_ORDER if p in sub["path_shape"].values]
    if not paths_in: return
    variants = get_variants_in_data(sub)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(paths_in)); n_var = len(variants); width = 0.8 / n_var

    for i, v in enumerate(variants):
        rc_vals, rc_lo, rc_hi = [], [], []
        for ps in paths_in:
            row = sub[(sub["variant"] == v) & (sub["path_shape"] == ps)]
            if row.empty: rc_vals.append(0); rc_lo.append(0); rc_hi.append(0)
            else:
                r = row.iloc[0]
                rc_vals.append(r["rare_coverage"] * 100)
                rc_lo.append((r["rare_coverage"] - r["rare_cov_ci_lo"]) * 100)
                rc_hi.append((r["rare_cov_ci_hi"] - r["rare_coverage"]) * 100)
        offset = (i - n_var / 2 + 0.5) * width
        ax.bar(x + offset, rc_vals, width, yerr=[rc_lo, rc_hi],
               label=VARIANT_LABELS.get(v, v), color=VARIANT_COLORS.get(v, "#333"),
               capsize=2, edgecolor="black", linewidth=0.4, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels([PATH_LABELS.get(p, p) for p in paths_in], rotation=15)
    ax.set_ylabel("Rare Mode Coverage (%)"); ax.set_title("Rare Mode Coverage (Erratic Behavior)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig_rare_mode_coverage")


# ============================================================================
# LaTeX summary table
# ============================================================================

def generate_latex_table(df):
    path = os.path.join(OUT_DIR, "path_obstacle_findings.tex")
    variants = get_variants_in_data(df)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Path/obstacle/behavior sweep: CDC strategy comparison (no OT).}",
        r"\label{tab:path_obstacle_sweep}", r"\small",
        r"\begin{tabular}{l l l r c c c c}", r"\toprule",
        r"Path & Behavior & Strategy & Obs & Coll.\ (\%) & [95\% CI] & Mode Cov.\ (\%) & Solve (ms) \\",
        r"\midrule",
    ]
    prev_path = ""
    for _, row in df.iterrows():
        ps = PATH_LABELS.get(row["path_shape"], row["path_shape"])
        beh = BEHAVIOR_LABELS.get(row["behavior"], row["behavior"])
        var = VARIANT_LABELS.get(row["variant"], row["variant"])
        nobs = int(row["num_obstacles"])
        cr = row["collision_rate"] * 100
        ci_lo = row["collision_ci_lo"] * 100; ci_hi = row["collision_ci_hi"] * 100
        mc = row["mode_coverage"] * 100; st = row["mean_solve_ms"]
        if ps != prev_path:
            if prev_path: lines.append(r"\midrule")
            prev_path = ps
        lines.append(f"{ps} & {beh} & {var} & {nobs} & {cr:.1f} & [{ci_lo:.1f}, {ci_hi:.1f}] & {mc:.1f} & {st:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading sweep data...")
    df = load_summary()
    raw_df = load_raw()
    print(f"  {len(df)} conditions, variants: {sorted(df['variant'].unique())}\n")

    has_paths = len(df["path_shape"].unique()) > 1
    has_obs = len(df["num_obstacles"].unique()) > 1
    has_behavs = len(df["behavior"].unique()) > 1

    print("Generating figures...")
    if has_paths:
        fig_path_collision(df); fig_path_perf(df)
        if raw_df is not None: fig_clearance_dist(raw_df)

    if has_obs:
        obs_df = df[(df["path_shape"] == "s_curve") & (df["behavior"] == "balanced")]
        if not obs_df.empty: fig_obs_count(obs_df)

    if has_behavs:
        beh_df = df[(df["path_shape"] == "s_curve") & (df["num_obstacles"] == 2)]
        if not beh_df.empty: fig_behavior(beh_df)

    fig_safety_coverage_scatter(df)
    if has_paths and has_behavs: fig_heatmap(df)
    fig_rare_coverage(df)

    print("\nGenerating LaTeX table...")
    generate_latex_table(df)
    print("\nDone.")
