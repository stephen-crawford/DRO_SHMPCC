#!/usr/bin/env python3
"""
Generate publication figures from results-section CSV data.

Reads CSVs from paper_figures/ and writes PNGs to paper_figures/.

Figures:
  1. Collision probability vs switching severity  (Exp A)
  2. Mode-mismatch (missed-mode fraction) vs time (Exp A, sp=0.2)
  3. Safety-performance Pareto                    (Exp A, sp=0.2)
  4. Collision rate conditioned on rare mode       (Exp B)
  5. Solve-time distributions (box plots)          (Exp C)
  5b. Safety vs Runtime Pareto                     (Exp C)
  6. Ablation table as figure                      (Exp A)
  7. Calibration plot: predicted vs observed risk   (Exp D)
  8. Buffer size sensitivity                        (Exp E)
  9. Conservatism & control smoothness comparison   (Exp G)
 10. Forest plot: collision rate differences + bootstrap CIs (Exp H1)
 11. Missed-mode significance: bar chart w/ p-values        (Exp H2)
 12. Updated ablation table (1000 rollouts)                  (Exp H4)
 13. Full ablation matrix bar chart                          (Exp H)
 14. Safe horizon scaling                                    (Exp I)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style
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

VARIANT_COLORS = {
    "Base": "#1f77b4",
    "DRO": "#ff7f0e",
    "OT": "#2ca02c",
    "OT+DRO": "#d62728",
    "Base+SH": "#aec7e8",
    "OT+SH": "#98df8a",
    "OT+ADV": "#9467bd",
    "OT+ADV+SH": "#e377c2",
}
VARIANT_MARKERS = {
    "Base": "o",
    "DRO": "s",
    "OT": "^",
    "OT+DRO": "D",
    "Base+SH": "v",
    "OT+SH": "<",
    "OT+ADV": ">",
    "OT+ADV+SH": "P",
}
VARIANT_ORDER = ["Base", "DRO", "OT", "OT+DRO",
                 "Base+SH", "OT+SH", "OT+ADV", "OT+ADV+SH"]

EPS_TARGET = 0.05  # epsilon target for horizontal dashed line


def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return None
    return pd.read_csv(path).dropna(how="all")


# ============================================================================
# Figure 1: Collision Probability vs Switching Severity
# ============================================================================

def fig1_collision_vs_switching():
    df = load_csv("exp_a_collision_vs_switching.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("switch_prob")
        if sub.empty:
            continue
        ax.errorbar(
            sub["switch_prob"], sub["collision_rate"],
            yerr=[sub["collision_rate"] - sub["ci_lo"],
                  sub["ci_hi"] - sub["collision_rate"]],
            label=vname, color=VARIANT_COLORS[vname],
            marker=VARIANT_MARKERS[vname], markersize=5,
            capsize=3, linewidth=1.5,
        )

    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, label=r"$\varepsilon$ target")
    ax.set_xlabel("Mode-switch probability")
    ax.set_ylabel("Empirical collision rate")
    ax.set_title("Collision Rate vs. Switching Severity")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=-0.01)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "fig1_collision_vs_switching.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 2: Mismatch vs Time
# ============================================================================

def fig2_mismatch_vs_time():
    df = load_csv("exp_a_w2_vs_time.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("step")
        if sub.empty:
            continue
        # Smooth with rolling window
        vals = sub["missed_fraction"].values
        window = 5
        if len(vals) >= window:
            smoothed = np.convolve(vals, np.ones(window)/window, mode="valid")
            steps = sub["step"].values[:len(smoothed)]
        else:
            smoothed = vals
            steps = sub["step"].values
        ax.plot(steps * 0.1, smoothed, label=vname,
                color=VARIANT_COLORS[vname], linewidth=1.3)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Missed-mode fraction")
    ax.set_title("Distributional Mismatch vs. Time (switch prob = 0.2)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=-0.02, top=1.02)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "fig2_mismatch_vs_time.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 3: Safety-Performance Pareto
# ============================================================================

def fig3_safety_performance_pareto():
    df = load_csv("exp_a_missed_mode_rate.csv")
    if df is None:
        return

    # Use switch_prob = 0.2 as representative
    df2 = df[np.isclose(df["switch_prob"], 0.2)]
    df_coll = load_csv("exp_a_collision_vs_switching.csv")
    if df_coll is None:
        return
    df_coll2 = df_coll[np.isclose(df_coll["switch_prob"], 0.2)]

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    for vname in VARIANT_ORDER:
        sub_miss = df2[df2["variant"] == vname]
        sub_coll = df_coll2[df_coll2["variant"] == vname]
        if sub_miss.empty or sub_coll.empty:
            continue
        progress = sub_miss["avg_progress"].values[0]
        coll_rate = sub_coll["collision_rate"].values[0]
        ci_lo = sub_coll["ci_lo"].values[0]
        ci_hi = sub_coll["ci_hi"].values[0]

        ax.errorbar(
            progress, coll_rate,
            yerr=[[coll_rate - ci_lo], [ci_hi - coll_rate]],
            marker=VARIANT_MARKERS[vname], color=VARIANT_COLORS[vname],
            markersize=10, capsize=4, linewidth=1.5,
            label=vname,
        )

    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Average progress [m]")
    ax.set_ylabel("Collision rate")
    ax.set_title("Safety vs. Performance (switch prob = 0.2)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "fig3_safety_performance_pareto.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 4: Collision Rate Conditioned on Rare Mode
# ============================================================================

def fig4_rare_mode_collision():
    df = load_csv("exp_b_collision_given_rare.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Left: overall collision rate vs rare_prob
    ax = axes[0]
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("rare_prob")
        if sub.empty:
            continue
        ax.errorbar(
            sub["rare_prob"], sub["collision_rate"],
            yerr=[sub["collision_rate"] - sub["ci_lo"],
                  sub["ci_hi"] - sub["collision_rate"]],
            label=vname, color=VARIANT_COLORS[vname],
            marker=VARIANT_MARKERS[vname], markersize=5,
            capsize=3, linewidth=1.5,
        )
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Rare-mode probability")
    ax.set_ylabel("Collision rate")
    ax.set_title("Overall Collision Rate")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right: collision rate conditioned on rare mode occurring
    ax = axes[1]
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("rare_prob")
        if sub.empty:
            continue
        ax.plot(
            sub["rare_prob"], sub["collision_given_rare"],
            label=vname, color=VARIANT_COLORS[vname],
            marker=VARIANT_MARKERS[vname], markersize=5,
            linewidth=1.5,
        )
    ax.set_xlabel("Rare-mode probability")
    ax.set_ylabel("P(collision | rare mode)")
    ax.set_title("Collision Rate Given Rare Mode")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_rare_mode_collision.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 5: Solve-Time Distributions
# ============================================================================

def fig5_solve_time_distributions():
    df = load_csv("exp_c_solve_times.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # Left: median solve time vs scenario count
    ax = axes[0]
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        ax.plot(sub["num_scenarios"], sub["median_ms"],
                label=vname, color=VARIANT_COLORS[vname],
                marker=VARIANT_MARKERS[vname], markersize=5, linewidth=1.5)
    ax.set_xlabel("Number of scenarios S")
    ax.set_ylabel("Median solve time [ms]")
    ax.set_title("Median Solve Time")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right: p99 solve time vs scenario count
    ax = axes[1]
    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        ax.plot(sub["num_scenarios"], sub["p99_ms"],
                label=vname, color=VARIANT_COLORS[vname],
                marker=VARIANT_MARKERS[vname], markersize=5, linewidth=1.5)
    ax.set_xlabel("Number of scenarios S")
    ax.set_ylabel("p99 solve time [ms]")
    ax.set_title("Tail Solve Time (p99)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_solve_time_distributions.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")

    # Also: safety vs runtime Pareto
    df_safety = load_csv("exp_c_safety_vs_runtime.csv")
    if df_safety is None:
        return

    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    for vname in VARIANT_ORDER:
        sub = df_safety[df_safety["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        ax2.plot(sub["avg_solve_ms"], sub["collision_rate"],
                 label=vname, color=VARIANT_COLORS[vname],
                 marker=VARIANT_MARKERS[vname], markersize=7, linewidth=1.5)
        # Annotate with S values
        for _, row in sub.iterrows():
            ax2.annotate(f"S={int(row['num_scenarios'])}",
                        (row["avg_solve_ms"], row["collision_rate"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax2.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Average solve time [ms]")
    ax2.set_ylabel("Collision rate")
    ax2.set_title("Safety vs. Runtime Tradeoff")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    out2 = os.path.join(OUT_DIR, "fig5b_safety_vs_runtime.png")
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"  Saved {out2}")


# ============================================================================
# Figure 6: Ablation Table
# ============================================================================

def fig6_ablation_table():
    df = load_csv("exp_a_ablation_table.csv")
    if df is None:
        return

    # Detect column layout
    has_adv = "uses_adversarial" in df.columns
    has_sh = "uses_sh" in df.columns

    fig_height = max(2.2, 0.5 + 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    ax.axis("off")

    columns = ["Variant", "OT?", "DRO?"]
    if has_adv:
        columns.append("ADV?")
    if has_sh:
        columns.append("SH?")
    columns += ["Collision Rate", "95% CI",
                "Missed Mode", "Avg Progress", "Avg Clearance", "Solve [ms]"]

    dro_col = "uses_dro" if "uses_dro" in df.columns else "uses_extremes"

    cell_text = []
    cell_colors = []
    for _, row in df.iterrows():
        ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        cells = [
            row["variant"],
            row.get("uses_ot", ""),
            row.get(dro_col, ""),
        ]
        if has_adv:
            cells.append(row.get("uses_adversarial", ""))
        if has_sh:
            cells.append(row.get("uses_sh", ""))
        cells += [
            f"{row['collision_rate']:.3f}",
            ci,
            f"{row['missed_mode_rate']:.3f}",
            f"{row['avg_progress']:.1f}",
            f"{row['avg_clearance']:.2f}",
            f"{row['avg_solve_ms']:.1f}",
        ]
        cell_text.append(cells)
        if row["collision_rate"] < EPS_TARGET:
            cell_colors.append(["#e8f5e9"] * len(columns))
        else:
            cell_colors.append(["#ffebee"] * len(columns))

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellColours=cell_colors if cell_colors else None,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for j in range(len(columns)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#e0e0e0")

    ax.set_title("Ablation Study (switch prob = 0.2)", fontsize=11, pad=15)

    out = os.path.join(OUT_DIR, "fig6_ablation_table.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 7: Calibration Plot (Predicted Risk vs Observed Collision Rate)
# ============================================================================

def fig7_calibration_plot():
    df = load_csv("exp_d_calibration.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    # Perfect calibration line
    ax.plot([0, 0.55], [0, 0.55], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")

    for vname in VARIANT_ORDER:
        sub = df[df["variant"] == vname].sort_values("predicted_risk")
        if sub.empty:
            continue
        ax.errorbar(
            sub["predicted_risk"], sub["observed_collision_rate"],
            yerr=[sub["observed_collision_rate"] - sub["ci_lo"],
                  sub["ci_hi"] - sub["observed_collision_rate"]],
            label=vname, color=VARIANT_COLORS[vname],
            marker=VARIANT_MARKERS[vname], markersize=5,
            capsize=3, linewidth=1.5,
        )

    ax.set_xlabel(r"Predicted risk ($\varepsilon$ target)")
    ax.set_ylabel("Observed collision rate")
    ax.set_title("Calibration: Predicted vs. Observed Risk")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_xlim(-0.01, 0.55)
    ax.set_ylim(-0.01, 0.55)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out = os.path.join(OUT_DIR, "fig7_calibration_plot.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 8: Buffer Size Sensitivity
# ============================================================================

def fig8_buffer_sensitivity():
    df = load_csv("exp_e_buffer_sensitivity.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Left: collision rate vs buffer size
    ax = axes[0]
    ax.errorbar(df["buffer_size"], df["collision_rate"],
                yerr=[df["collision_rate"] - df["ci_lo"],
                      df["ci_hi"] - df["collision_rate"]],
                color="#d62728", marker="D", markersize=6, capsize=3, linewidth=1.5)
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("History buffer size")
    ax.set_ylabel("Collision rate")
    ax.set_title("Safety vs. Buffer Size")
    ax.grid(True, alpha=0.3)

    # Middle: missed mode rate vs buffer size
    ax = axes[1]
    ax.plot(df["buffer_size"], df["missed_mode_rate"],
            color="#2ca02c", marker="^", markersize=6, linewidth=1.5)
    ax.set_xlabel("History buffer size")
    ax.set_ylabel("Missed-mode rate")
    ax.set_title("Mode Coverage vs. Buffer Size")
    ax.grid(True, alpha=0.3)

    # Right: clearance vs buffer size
    ax = axes[2]
    ax.plot(df["buffer_size"], df["avg_clearance"],
            color="#1f77b4", marker="o", markersize=6, linewidth=1.5)
    ax.set_xlabel("History buffer size")
    ax.set_ylabel("Average min clearance [m]")
    ax.set_title("Conservatism vs. Buffer Size")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig8_buffer_sensitivity.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 9: Conservatism & Control Smoothness
# ============================================================================

def fig9_conservatism_smoothness():
    df = load_csv("exp_g_conservatism_metrics.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    variants = df["variant"].tolist()
    x = np.arange(len(variants))
    width = 0.6
    colors = [VARIANT_COLORS.get(v, "#999999") for v in variants]

    # Average speed
    ax = axes[0]
    ax.bar(x, df["avg_speed"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg speed [m/s]")
    ax.set_title("Average Speed")
    ax.grid(True, alpha=0.3, axis="y")

    # Min clearance
    ax = axes[1]
    ax.bar(x, df["min_clearance_mean"], width, color=colors,
           yerr=df["min_clearance_std"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Min clearance [m]")
    ax.set_title("Minimum Clearance")
    ax.grid(True, alpha=0.3, axis="y")

    # Control effort
    ax = axes[2]
    ax.bar(x, df["control_effort_mean"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Control effort")
    ax.set_title("Control Effort")
    ax.grid(True, alpha=0.3, axis="y")

    # Steering variation (smoothness)
    ax = axes[3]
    ax.bar(x, df["steering_variation"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Steering variation")
    ax.set_title("Control Smoothness")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig9_conservatism_smoothness.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 10: Forest Plot — Collision Rate Differences with Bootstrap CIs
# ============================================================================

def fig10_forest_plot():
    df = load_csv("exp_h1_bootstrap_ci.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    comparisons = df["comparison"].tolist()
    means = df["mean_diff"].values
    ci_lo = df["ci_lo"].values
    ci_hi = df["ci_hi"].values

    # Clean up labels
    labels = [c.replace("_vs_", " vs ").replace("_", " ") for c in comparisons]

    y_pos = np.arange(len(labels))
    colors = ["#d62728", "#2ca02c", "#ff7f0e"]  # match variant colors

    for i in range(len(labels)):
        ax.errorbar(
            means[i], y_pos[i],
            xerr=[[means[i] - ci_lo[i]], [ci_hi[i] - means[i]]],
            fmt="o", color=colors[i % len(colors)],
            markersize=8, capsize=5, linewidth=2, capthick=1.5,
        )

    ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Collision rate difference (Base $-$ variant)")
    ax.set_title("Collision Rate Differences with 95% Bootstrap CIs")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Annotate with values
    for i in range(len(labels)):
        ax.annotate(
            f"{means[i]:.3f} [{ci_lo[i]:.3f}, {ci_hi[i]:.3f}]",
            (ci_hi[i], y_pos[i]),
            textcoords="offset points", xytext=(8, 0),
            fontsize=8, va="center",
        )

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig10_forest_plot.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 11: Missed-Mode Significance — Bar Chart with p-values
# ============================================================================

def fig11_missed_mode_significance():
    df = load_csv("exp_h2_missed_mode_significance.csv")
    if df is None:
        return

    # Check if expected columns exist
    if "switch_prob" not in df.columns:
        print("  WARNING: exp_h2_missed_mode_significance.csv has incompatible schema, skipping fig11.")
        return

    switch_probs = sorted(df["switch_prob"].unique())

    fig, axes = plt.subplots(1, len(switch_probs), figsize=(5 * len(switch_probs), 4.5))
    if len(switch_probs) == 1:
        axes = [axes]

    for ax, sp in zip(axes, switch_probs):
        sub = df[np.isclose(df["switch_prob"], sp)]

        variants = []
        rates = []
        colors = []
        for vname in VARIANT_ORDER:
            row = sub[sub["variant"] == vname]
            if row.empty:
                continue
            variants.append(vname)
            rates.append(row["missed_mode_rate"].values[0])
            colors.append(VARIANT_COLORS.get(vname, "#999999"))

        x = np.arange(len(variants))
        bars = ax.bar(x, rates, 0.6, color=colors)

        # Annotate p-values for non-Base variants
        for i, vname in enumerate(variants):
            if vname == "Base":
                continue
            row = sub[sub["variant"] == vname]
            if row.empty:
                continue
            p_z = row["p_vs_base"].values[0]
            h = row["cohens_h_vs_base"].values[0]
            perm_p = row["permutation_p_vs_base"].values[0]

            # Star annotation
            if p_z < 0.001:
                stars = "***"
            elif p_z < 0.01:
                stars = "**"
            elif p_z < 0.05:
                stars = "*"
            else:
                stars = "n.s."

            ax.annotate(
                f"p={p_z:.3f} {stars}\nh={h:.2f}",
                (x[i], rates[i]),
                textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=7,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Missed-mode rate")
        ax.set_title(f"switch prob = {sp}")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Missed-Mode Rates with Statistical Significance vs Base", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig11_missed_mode_significance.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 12: Updated Ablation Table (1000 rollouts, tighter CIs)
# ============================================================================

def fig12_ablation_table_1000():
    df = load_csv("exp_h4_ablation_table_1000.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.axis("off")

    columns = ["Variant", "OT?", "DRO?", "Collision Rate", "95% CI",
               "Missed Mode", "Avg Progress", "Avg Clearance", "Solve [ms]"]

    # Handle both old ("uses_extremes") and new ("uses_dro") column names
    dro_col = "uses_dro" if "uses_dro" in df.columns else "uses_extremes"

    cell_text = []
    cell_colors = []
    for _, row in df.iterrows():
        ci = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
        cells = [
            row["variant"],
            row["uses_ot"],
            row[dro_col],
            f"{row['collision_rate']:.4f}",
            ci,
            f"{row['missed_mode_rate']:.4f}",
            f"{row['avg_progress']:.1f}",
            f"{row['avg_clearance']:.2f}",
            f"{row['avg_solve_ms']:.1f}",
        ]
        cell_text.append(cells)
        if row["collision_rate"] < EPS_TARGET:
            cell_colors.append(["#e8f5e9"] * len(columns))
        else:
            cell_colors.append(["#ffebee"] * len(columns))

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellColours=cell_colors if cell_colors else None,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for j in range(len(columns)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#e0e0e0")

    ax.set_title("Ablation Study — 1000 Rollouts (switch prob = 0.2)", fontsize=11, pad=15)

    out = os.path.join(OUT_DIR, "fig12_ablation_table_1000.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 13: Full Ablation Matrix (Exp H)
# ============================================================================

def fig13_full_ablation():
    df = load_csv("exp_h_ablation_full.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    variants = df["variant"].tolist()
    x = np.arange(len(variants))
    width = 0.55

    colors = [VARIANT_COLORS.get(v, "#999999") for v in variants]

    # Left: collision rate with CIs
    ax = axes[0]
    ax.bar(x, df["collision_rate"], width, color=colors,
           yerr=[df["collision_rate"] - df["ci_lo"],
                 df["ci_hi"] - df["collision_rate"]],
           capsize=3)
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=r"$\varepsilon$ target")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Collision rate")
    ax.set_title("Collision Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Middle: missed mode rate
    ax = axes[1]
    ax.bar(x, df["missed_mode_rate"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Missed-mode rate")
    ax.set_title("Mode Coverage")
    ax.grid(True, alpha=0.3, axis="y")

    # Right: avg progress
    ax = axes[2]
    ax.bar(x, df["avg_progress"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average progress [m]")
    ax.set_title("Progress (lower conservatism)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Full Ablation Matrix (switch prob = 0.2, S=40)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig13_full_ablation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 14: Safe Horizon Scaling (Exp I)
# ============================================================================

def fig14_sh_scaling():
    df = load_csv("exp_i_sh_scaling.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Group by variant name if column exists
    if "variant" in df.columns:
        variant_groups = df.groupby("variant")
    else:
        # Legacy format: partition by sh_enabled
        sh_on = df[df["sh_enabled"].astype(str).str.lower() == "true"].copy()
        sh_off = df[df["sh_enabled"].astype(str).str.lower() == "false"].copy()
        sh_on["variant"] = "Base+SH"
        sh_off["variant"] = "Base"
        df = pd.concat([sh_off, sh_on])
        variant_groups = df.groupby("variant")

    # Left: collision rate vs S
    ax = axes[0]
    for vname, sub in variant_groups:
        sub = sub.sort_values("num_scenarios")
        color = VARIANT_COLORS.get(vname, "#999999")
        marker = VARIANT_MARKERS.get(vname, "o")
        ax.errorbar(sub["num_scenarios"], sub["collision_rate"],
                     yerr=[sub["collision_rate"] - sub["ci_lo"],
                           sub["ci_hi"] - sub["collision_rate"]],
                     label=vname, color=color, marker=marker, capsize=3, linewidth=1.5)
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Number of scenarios S")
    ax.set_ylabel("Collision rate")
    ax.set_title("Collision Rate vs. S")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: predicted N_safe vs S (SH variants only)
    ax = axes[1]
    sh_variants = df[df["sh_enabled"].astype(str).str.lower() == "true"]
    for vname, sub in sh_variants.groupby("variant"):
        sub = sub.sort_values("num_scenarios")
        color = VARIANT_COLORS.get(vname, "#d62728")
        marker = VARIANT_MARKERS.get(vname, "s")
        ax.plot(sub["num_scenarios"], sub["predicted_n_safe"],
                label=vname, color=color, marker=marker, linewidth=1.5, markersize=7)
    ax.axhline(15, color="gray", linestyle="--", linewidth=1, alpha=0.5,
                label="Full horizon (N=15)")
    ax.axhline(3, color="orange", linestyle=":", linewidth=1, alpha=0.7,
                label="safe_horizon_min=3")
    ax.set_xlabel("Number of scenarios S")
    ax.set_ylabel("Effective N_safe")
    ax.set_title("Safe Horizon Truncation (Practical Mode)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 17)
    ax.grid(True, alpha=0.3)

    # Right: solve time vs S
    ax = axes[2]
    for vname, sub in variant_groups:
        sub = sub.sort_values("num_scenarios")
        color = VARIANT_COLORS.get(vname, "#999999")
        marker = VARIANT_MARKERS.get(vname, "o")
        ax.plot(sub["num_scenarios"], sub["avg_solve_ms"],
                label=vname, color=color, marker=marker, linewidth=1.5)
    ax.set_xlabel("Number of scenarios S")
    ax.set_ylabel("Average solve time [ms]")
    ax.set_title("Solve Time vs. S")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Safe Horizon Scaling Analysis (Practical Mode)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig14_sh_scaling.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 15: OT vs Simple Coverage Baselines (Exp J)
# ============================================================================

def fig15_ot_vs_baselines():
    df = load_csv("exp_j_ot_vs_baselines.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    baselines = df["baseline"].tolist()
    x = np.arange(len(baselines))
    width = 0.55

    BASELINE_COLORS = {
        "Standard": "#1f77b4",
        "OT": "#2ca02c",
        "Stratified": "#ff7f0e",
        "Temperature": "#d62728",
        "EpsilonGreedy": "#9467bd",
        "RiskBiased": "#8c564b",
    }
    colors = [BASELINE_COLORS.get(b, "#999999") for b in baselines]

    # Left: collision rate with CIs
    ax = axes[0]
    ax.bar(x, df["collision_rate"], width, color=colors,
           yerr=[df["collision_rate"] - df["ci_lo"],
                 df["ci_hi"] - df["collision_rate"]],
           capsize=3)
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=r"$\varepsilon$ target")
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Collision rate")
    ax.set_title("Collision Rate by Sampling Baseline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: missed mode rate
    ax = axes[1]
    ax.bar(x, df["missed_mode_rate"], width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Missed-mode rate")
    ax.set_title("Mode Coverage by Sampling Baseline")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("OT vs. Simple Coverage Baselines (S=40, switch prob=0.2)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig15_ot_vs_baselines.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 16: Environment Generalization (Exp K)
# ============================================================================

def fig16_environment_generalization():
    df = load_csv("exp_k_environment_generalization.csv")
    if df is None:
        return

    envs = df["environment"].unique()
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(4 * n_envs, 4.5))
    if n_envs == 1:
        axes = [axes]

    k_variants = ["Base", "OT", "Base+SH", "OT+SH"]
    k_colors = [VARIANT_COLORS.get(v, "#999999") for v in k_variants]

    for ax, env_name in zip(axes, envs):
        sub = df[df["environment"] == env_name]
        variants_present = [v for v in k_variants if v in sub["variant"].values]
        x = np.arange(len(variants_present))
        width = 0.55
        colors_present = [VARIANT_COLORS.get(v, "#999999") for v in variants_present]

        rates = []
        ci_los = []
        ci_his = []
        for v in variants_present:
            row = sub[sub["variant"] == v]
            rates.append(row["collision_rate"].values[0])
            ci_los.append(row["ci_lo"].values[0])
            ci_his.append(row["ci_hi"].values[0])

        rates = np.array(rates)
        ci_los = np.array(ci_los)
        ci_his = np.array(ci_his)

        ax.bar(x, rates, width, color=colors_present,
               yerr=[rates - ci_los, ci_his - rates], capsize=3)
        ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(variants_present, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Collision rate")
        ax.set_title(f"{env_name}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(bottom=-0.01)

    fig.suptitle("Environment Generalization (S=40, switch prob=0.2)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig16_environment_generalization.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 17: Empirical Violation Rate (Exp L)
# ============================================================================

def fig17_empirical_violation():
    df = load_csv("exp_l_empirical_violation.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Perfect calibration line
    ax.plot([0, 0.15], [0, 0.15], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")

    for vname in ["Base", "Base+SH", "OT+SH"]:
        sub = df[df["variant"] == vname]
        if sub.empty:
            continue
        color = VARIANT_COLORS.get(vname, "#999999")
        marker = VARIANT_MARKERS.get(vname, "o")

        ax.errorbar(
            sub["epsilon_target"], sub["epsilon_hat"],
            yerr=[sub["epsilon_hat"] - sub["ci_lo"],
                  sub["ci_hi"] - sub["epsilon_hat"]],
            label=vname, color=color,
            marker=marker, markersize=7,
            capsize=4, linewidth=1.5, linestyle="none",
        )
        # Annotate with S values
        for _, row in sub.iterrows():
            ax.annotate(f"S={int(row['num_scenarios'])}",
                       (row["epsilon_target"], row["epsilon_hat"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.axhline(0.05, color="red", linestyle=":", linewidth=1, alpha=0.5,
               label=r"$\varepsilon$=0.05")
    ax.set_xlabel(r"$\varepsilon$ target")
    ax.set_ylabel(r"$\hat{\varepsilon}$ (empirical violation rate)")
    ax.set_title("Empirical Joint Violation Rate vs. Target")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.005, 0.08)
    ax.set_ylim(-0.005, max(0.08, df["ci_hi"].max() * 1.1 if not df.empty else 0.08))

    out = os.path.join(OUT_DIR, "fig17_empirical_violation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 18: Safe Horizon Length Sweep (Exp M)
# ============================================================================

def fig18_sh_sweep():
    df = load_csv("exp_m_sh_sweep.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    for vname in ["Base+SH", "OT+SH"]:
        sub = df[df["variant"] == vname].sort_values("forced_n_safe")
        if sub.empty:
            continue
        color = VARIANT_COLORS.get(vname, "#999999")
        marker = VARIANT_MARKERS.get(vname, "o")

        # Left: collision rate
        ax = axes[0]
        ax.errorbar(sub["forced_n_safe"], sub["collision_rate"],
                    yerr=[sub["collision_rate"] - sub["ci_lo"],
                          sub["ci_hi"] - sub["collision_rate"]],
                    label=vname, color=color, marker=marker,
                    capsize=3, linewidth=1.5, markersize=6)

        # Middle: progress
        ax = axes[1]
        ax.plot(sub["forced_n_safe"], sub["avg_progress"],
                label=vname, color=color, marker=marker,
                linewidth=1.5, markersize=6)

        # Right: solve time
        ax = axes[2]
        ax.plot(sub["forced_n_safe"], sub["avg_solve_ms"],
                label=vname, color=color, marker=marker,
                linewidth=1.5, markersize=6)

    axes[0].axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                     label=r"$\varepsilon$ target")
    axes[0].set_xlabel(r"Forced $N_{\mathrm{safe}}$")
    axes[0].set_ylabel("Collision rate")
    axes[0].set_title("Collision Rate vs. Safe Horizon")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r"Forced $N_{\mathrm{safe}}$")
    axes[1].set_ylabel("Average progress [m]")
    axes[1].set_title("Progress vs. Safe Horizon")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel(r"Forced $N_{\mathrm{safe}}$")
    axes[2].set_ylabel("Average solve time [ms]")
    axes[2].set_title("Solve Time vs. Safe Horizon")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Safe Horizon Length Sweep (S=40, N=15)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig18_sh_sweep.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 19: Runtime Scaling Breakdown (Exp N)
# ============================================================================

def fig19_runtime_scaling():
    df = load_csv("exp_n_runtime_scaling.csv")
    if df is None:
        return

    dims = df["sweep_dim"].unique()
    n_dims = len(dims)
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4.0))
    if n_dims == 1:
        axes = [axes]

    dim_labels = {"S": "Scenario Count S", "D": "Disc Count D", "N_safe": r"$N_{\mathrm{safe}}$"}

    for ax, dim in zip(axes, dims):
        sub = df[df["sweep_dim"] == dim].sort_values("sweep_value")
        x = sub["sweep_value"].values

        # Stacked bar: constraint time + QP time
        bar_width = (x[-1] - x[0]) / (len(x) * 2) if len(x) > 1 else 1
        ax.bar(x, sub["avg_constraint_ms"], bar_width, label="Constraint build",
               color="#1f77b4", alpha=0.8)
        ax.bar(x, sub["avg_qp_ms"], bar_width, bottom=sub["avg_constraint_ms"],
               label="QP solve", color="#ff7f0e", alpha=0.8)

        # Also plot total as line
        ax.plot(x, sub["avg_total_ms"], "k-o", label="Total", markersize=5, linewidth=1.5)

        ax.set_xlabel(dim_labels.get(dim, dim))
        ax.set_ylabel("Time [ms]")
        ax.set_title(f"Runtime vs. {dim_labels.get(dim, dim)}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Runtime Scaling Breakdown (OT+SH)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig19_runtime_scaling.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 20: Distribution Shift / Mismatch (Exp O)
# ============================================================================

def fig20_distribution_shift():
    df = load_csv("exp_o_distribution_shift.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    conditions = df["condition"].unique()
    o_variants = ["Base", "OT", "OT+SH"]

    # Group by condition, bars for each variant
    x = np.arange(len(conditions))
    width = 0.22
    offsets = np.arange(len(o_variants)) * width - width

    # Left: collision rate
    ax = axes[0]
    for i, vname in enumerate(o_variants):
        rates = []
        ci_los = []
        ci_his = []
        for cond in conditions:
            row = df[(df["condition"] == cond) & (df["variant"] == vname)]
            if row.empty:
                rates.append(0)
                ci_los.append(0)
                ci_his.append(0)
            else:
                rates.append(row["collision_rate"].values[0])
                ci_los.append(row["ci_lo"].values[0])
                ci_his.append(row["ci_hi"].values[0])
        rates = np.array(rates)
        ci_los = np.array(ci_los)
        ci_his = np.array(ci_his)

        color = VARIANT_COLORS.get(vname, "#999999")
        ax.bar(x + offsets[i], rates, width, label=vname, color=color,
               yerr=[rates - ci_los, ci_his - rates], capsize=2)

    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("Collision rate")
    ax.set_title("Collision Rate Under Distribution Shift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: missed mode rate
    ax = axes[1]
    for i, vname in enumerate(o_variants):
        rates = []
        for cond in conditions:
            row = df[(df["condition"] == cond) & (df["variant"] == vname)]
            rates.append(row["missed_mode_rate"].values[0] if not row.empty else 0)
        color = VARIANT_COLORS.get(vname, "#999999")
        ax.bar(x + offsets[i], rates, width, label=vname, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("Missed-mode rate")
    ax.set_title("Mode Coverage Under Distribution Shift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Robustness to Distribution Mismatch", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig20_distribution_shift.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 21: Safety-Coverage Tradeoff Plane (from existing exp_h data)
# ============================================================================

def fig21_tradeoff_plane():
    df = load_csv("exp_h_ablation_full.csv")
    if df is None:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for _, row in df.iterrows():
        vname = row["variant"]
        color = VARIANT_COLORS.get(vname, "#999999")
        marker = VARIANT_MARKERS.get(vname, "o")
        # Marker size proportional to progress
        size = (row["avg_progress"] / df["avg_progress"].max()) * 300 + 50
        ax.scatter(row["missed_mode_rate"], row["collision_rate"],
                   c=color, marker=marker, s=size, label=vname,
                   edgecolors="black", linewidth=0.5, zorder=5)
        # Vertical CI bars for collision rate
        ax.plot([row["missed_mode_rate"]] * 2, [row["ci_lo"], row["ci_hi"]],
                color=color, linewidth=1, alpha=0.5)

    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=r"$\varepsilon$ target")
    ax.set_xlabel("Missed-mode rate")
    ax.set_ylabel("Collision rate")
    ax.set_title("Safety\u2013Coverage Tradeoff (sp=0.2, S=40)")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # Annotate ideal corner
    ax.annotate("Ideal", xy=(0, 0), fontsize=8, color="gray",
                ha="left", va="bottom")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig21_tradeoff_plane.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 22: Coverage Strategy Baselines (Exp P)
# ============================================================================

def fig22_coverage_baselines():
    df = load_csv("exp_p_coverage_baselines.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    baselines = df["baseline"].values
    x = np.arange(len(baselines))
    bar_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2"][:len(baselines)]

    # Left: collision rate with CIs
    ax = axes[0]
    cr = df["collision_rate"].values
    ci_lo = df["ci_lo"].values
    ci_hi = df["ci_hi"].values
    ax.bar(x, cr, 0.6, color=bar_colors,
           yerr=[cr - ci_lo, ci_hi - cr], capsize=3, edgecolor="black", linewidth=0.5)
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=r"$\varepsilon$ target")
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Collision rate")
    ax.set_title("Collision Rate by Strategy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: missed-mode rate
    ax = axes[1]
    ax.bar(x, df["missed_mode_rate"].values, 0.6, color=bar_colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Missed-mode rate")
    ax.set_title("Mode Coverage by Strategy")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Coverage Strategy Comparison (sp=0.2, S=40)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig22_coverage_baselines.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 23: OT Internal Ablation (Exp Q)
# ============================================================================

def fig23_ot_ablation():
    df = load_csv("exp_q_ot_ablation.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    configs = df["config"].values
    x = np.arange(len(configs))
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))

    # Left: collision rate
    ax = axes[0]
    cr = df["collision_rate"].values
    ci_lo = df["ci_lo"].values
    ci_hi = df["ci_hi"].values
    ax.bar(x, cr, 0.6, color=colors,
           yerr=[cr - ci_lo, ci_hi - cr], capsize=2, edgecolor="black", linewidth=0.5)
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Collision rate")
    ax.set_title("Collision Rate")
    ax.grid(True, alpha=0.3, axis="y")

    # Middle: missed-mode rate
    ax = axes[1]
    ax.bar(x, df["missed_mode_rate"].values, 0.6, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Missed-mode rate")
    ax.set_title("Mode Coverage")
    ax.grid(True, alpha=0.3, axis="y")

    # Right: solve time
    ax = axes[2]
    ax.bar(x, df["avg_solve_ms"].values, 0.6, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Solve time [ms]")
    ax.set_title("Runtime")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("OT Predictor Parameter Sensitivity", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig23_ot_ablation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 24: Mode Coverage Diagnostic (Exp R)
# ============================================================================

def fig24_mode_coverage():
    df = load_csv("exp_r_mode_coverage.csv")
    if df is None:
        return

    variants = df["variant"].unique()
    n_variants = len(variants)
    fig, axes = plt.subplots(1, n_variants, figsize=(5.5 * n_variants, 4.5))
    if n_variants == 1:
        axes = [axes]

    mode_colors = {"constant_velocity": "#1f77b4", "turn_left": "#ff7f0e",
                   "turn_right": "#2ca02c", "decelerating": "#d62728"}
    # Human-readable behavior labels for x-axis (full behavior description)
    mode_labels = {
        "constant_velocity": "Constant velocity",
        "turn_left": "Turn left",
        "turn_right": "Turn right",
        "decelerating": "Decelerating",
    }

    # Use a common y-axis scale across subplots so coverage fractions are comparable
    global_max = 0.0

    for ax, vname in zip(axes, variants):
        sub = df[df["variant"] == vname]
        modes = sub["mode_name"].values
        true_frac = sub["true_fraction"].values
        samp_frac = sub["sampled_fraction"].values
        # Track maximum fraction across all variants for a shared y-limit
        if len(true_frac) > 0 and len(samp_frac) > 0:
            local_max = max(true_frac.max(), samp_frac.max())
            if local_max > global_max:
                global_max = local_max

        x = np.arange(len(modes))
        width = 0.35

        bars1 = ax.bar(x - width/2, true_frac, width, label="True frequency",
                       color="#aec7e8", edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width/2, samp_frac, width, label="Sampled frequency",
                       color="#ffbb78", edgecolor="black", linewidth=0.5)

        # Perfect coverage line
        ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, alpha=0.5,
                   label="Uniform (1/M)")

        # Annotate absolute discrepancy |sampled - true| above each mode
        diff = np.abs(samp_frac - true_frac)
        for xi, t, s, d in zip(x, true_frac, samp_frac, diff):
            y_top = max(t, s)
            ax.text(xi, y_top + 0.02, f"{d:.2f}", ha="center", va="bottom", fontsize=7)

        # Use full behavior names on x-axis (or fall back to cleaned mode name)
        labels = [mode_labels.get(m, m.replace("_", " ")) for m in modes]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
        ax.set_ylabel("Fraction")
        ax.set_title(f"{vname}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # Apply a common y-scale to all subplots
    if global_max <= 0:
        global_max = 0.25
    y_max = min(1.0, global_max * 1.2)
    for ax in axes:
        ax.set_ylim(0, y_max)

    fig.suptitle("Mode Coverage Diagnostic: True vs Sampled Frequencies (numbers = |sampled - true|)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig24_mode_coverage.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 25: Missed-Mode Rate vs Scenario Count (Exp T)
# ============================================================================

def fig25_missed_mode_vs_s():
    df = load_csv("exp_t_missed_mode_vs_s.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for vname in ["Base", "OT", "OT+SH"]:
        sub = df[df["variant"] == vname].sort_values("num_scenarios")
        if sub.empty:
            continue
        color = VARIANT_COLORS.get(vname, "#999999")
        marker = VARIANT_MARKERS.get(vname, "o")

        # Left: missed-mode rate vs S
        axes[0].plot(sub["num_scenarios"], sub["missed_mode_rate"],
                     label=vname, color=color, marker=marker,
                     linewidth=1.5, markersize=6)

        # Right: collision rate vs S with CI
        axes[1].errorbar(sub["num_scenarios"], sub["collision_rate"],
                         yerr=[sub["collision_rate"] - sub["ci_lo"],
                               sub["ci_hi"] - sub["collision_rate"]],
                         label=vname, color=color, marker=marker,
                         linewidth=1.5, markersize=6, capsize=3)

    axes[0].set_xlabel("Scenario count S")
    axes[0].set_ylabel("Missed-mode rate")
    axes[0].set_title("Mode Coverage vs. Scenario Budget")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("log", base=2)

    axes[1].axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7,
                     label=r"$\varepsilon$ target")
    axes[1].set_xlabel("Scenario count S")
    axes[1].set_ylabel("Collision rate")
    axes[1].set_title("Collision Rate vs. Scenario Budget")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("log", base=2)

    fig.suptitle("Impact of Scenario Count on Coverage and Safety", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig25_missed_mode_vs_s.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 26: Ground-Cost Ablation (Exp U)
# ============================================================================

def fig26_ground_cost():
    df = load_csv("exp_u_ground_cost.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    costs = df["ground_cost"].values
    cr = df["collision_rate"].values
    ci_lo = df["ci_lo"].values
    ci_hi = df["ci_hi"].values
    x = np.arange(len(costs))
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"][:len(costs)]

    # Try to load paired data for McNemar tests
    paired_df = load_csv("exp_u_paired.csv")
    mcnemar_labels = {}
    if paired_df is not None and "W2-Euclidean" in paired_df.columns:
        from scipy.stats import chi2 as chi2_dist
        w2_col = paired_df["W2-Euclidean"].values
        for c in costs:
            if c == "W2-Euclidean":
                continue
            if c in paired_df.columns:
                other_col = paired_df[c].values
                b = int(np.sum((w2_col == 0) & (other_col == 1)))
                mc_c = int(np.sum((w2_col == 1) & (other_col == 0)))
                if b + mc_c > 0:
                    chi2_val = (abs(b - mc_c) - 1) ** 2 / (b + mc_c)
                    p_val = 1 - chi2_dist.cdf(chi2_val, df=1)
                else:
                    p_val = 1.0
                if p_val < 0.001:
                    mcnemar_labels[c] = "***"
                elif p_val < 0.01:
                    mcnemar_labels[c] = "**"
                elif p_val < 0.05:
                    mcnemar_labels[c] = "*"
                else:
                    mcnemar_labels[c] = f"p={p_val:.2f}"

    # Left: collision rate with McNemar annotations
    ax = axes[0]
    ax.bar(x, cr, 0.6, color=colors,
           yerr=[cr - ci_lo, ci_hi - cr], capsize=3, edgecolor="black", linewidth=0.5)
    ax.axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=r"$\varepsilon$ target")
    ax.set_xticks(x)
    ax.set_xticklabels(costs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Collision rate")
    ax.set_title("Collision Rate by Ground Cost")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Add McNemar p-value annotations
    for i, c in enumerate(costs):
        if c in mcnemar_labels:
            ax.text(i, ci_hi[i] + 0.003, mcnemar_labels[c],
                    ha="center", va="bottom", fontsize=7,
                    color="darkred" if mcnemar_labels[c].startswith("*") else "gray")

    # Right: missed-mode rate
    ax = axes[1]
    ax.bar(x, df["missed_mode_rate"].values, 0.6, color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(costs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Missed-mode rate")
    ax.set_title("Mode Coverage by Ground Cost")
    ax.grid(True, alpha=0.3, axis="y")

    n_rollouts = max(int(round(1.0 / (ci_hi[0] - ci_lo[0]) * 4)), 200)
    fig.suptitle(f"OT Ground-Cost Ablation (sp=0.2, S=40, N={n_rollouts})", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig26_ground_cost.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 27: Rare-Mode Probability Sweep (Exp V)
# ============================================================================

def fig27_rare_mode_sweep():
    df = load_csv("exp_v_rare_mode_sweep.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for vname in ["Base", "OT", "OT+SH"]:
        sub = df[df["variant"] == vname].sort_values("rare_prob")
        if sub.empty:
            continue
        color = VARIANT_COLORS.get(vname, "#999999")
        marker = VARIANT_MARKERS.get(vname, "o")

        # Left: collision rate vs rare_prob
        axes[0].errorbar(sub["rare_prob"], sub["collision_rate"],
                         yerr=[sub["collision_rate"] - sub["ci_lo"],
                               sub["ci_hi"] - sub["collision_rate"]],
                         label=vname, color=color, marker=marker,
                         linewidth=1.5, markersize=6, capsize=3)

        # Middle: overall missed-mode rate
        axes[1].plot(sub["rare_prob"], sub["missed_mode_rate"],
                     label=vname, color=color, marker=marker,
                     linewidth=1.5, markersize=6)

        # Right: rare-mode specific miss fraction
        axes[2].plot(sub["rare_prob"], sub["rare_mode_missed_frac"],
                     label=vname, color=color, marker=marker,
                     linewidth=1.5, markersize=6)

    axes[0].axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Rare-mode probability")
    axes[0].set_ylabel("Collision rate")
    axes[0].set_title("Collision Rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Rare-mode probability")
    axes[1].set_ylabel("Missed-mode rate (all modes)")
    axes[1].set_title("Overall Mode Coverage")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Rare-mode probability")
    axes[2].set_ylabel("Rare-mode miss fraction")
    axes[2].set_title("Rare-Mode Miss Rate")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Rare-Mode Probability Sweep (S=40, sp=0.2)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig27_rare_mode_sweep.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Figure 28: Mode Count Scaling (Exp W)
# ============================================================================

def fig28_mode_scaling():
    df = load_csv("exp_w_mode_scaling.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for vname in ["Base", "OT", "OT+SH"]:
        sub = df[df["variant"] == vname].sort_values("num_modes")
        if sub.empty:
            continue
        color = VARIANT_COLORS.get(vname, "#999999")
        marker = VARIANT_MARKERS.get(vname, "o")

        # Left: collision rate vs M
        axes[0].errorbar(sub["num_modes"], sub["collision_rate"],
                         yerr=[sub["collision_rate"] - sub["ci_lo"],
                               sub["ci_hi"] - sub["collision_rate"]],
                         label=vname, color=color, marker=marker,
                         linewidth=1.5, markersize=6, capsize=3)

        # Middle: missed-mode rate vs M
        axes[1].plot(sub["num_modes"], sub["missed_mode_rate"],
                     label=vname, color=color, marker=marker,
                     linewidth=1.5, markersize=6)

        # Right: solve time vs M
        axes[2].plot(sub["num_modes"], sub["avg_solve_ms"],
                     label=vname, color=color, marker=marker,
                     linewidth=1.5, markersize=6)

    axes[0].axhline(EPS_TARGET, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Number of modes M")
    axes[0].set_ylabel("Collision rate")
    axes[0].set_title("Collision Rate vs. M")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(df["num_modes"].unique())

    axes[1].set_xlabel("Number of modes M")
    axes[1].set_ylabel("Missed-mode rate")
    axes[1].set_title("Mode Coverage vs. M")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(df["num_modes"].unique())

    axes[2].set_xlabel("Number of modes M")
    axes[2].set_ylabel("Average solve time [ms]")
    axes[2].set_title("Runtime vs. M")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(df["num_modes"].unique())

    fig.suptitle("Scaling with Number of Modes (S=40, sp=0.2)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig28_mode_scaling.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Fig 29: Coverage Baselines on Rare-Mode Stress Test (Exp X)
# ============================================================================

def fig29_baselines_rare_mode():
    df = load_csv("exp_x_baselines_rare_mode.csv")
    if df is None:
        return

    baselines = df["baseline"].unique()
    rare_probs = sorted(df["rare_prob"].unique())

    baseline_colors = {
        "Standard": "#1f77b4", "OT": "#2ca02c", "OT+SH": "#98df8a",
        "Uniform": "#ff7f0e", "Temperature": "#d62728",
        "Stratified": "#9467bd", "EpsGreedy": "#8c564b",
    }
    baseline_markers = {
        "Standard": "s", "OT": "o", "OT+SH": "D",
        "Uniform": "^", "Temperature": "v",
        "Stratified": "P", "EpsGreedy": "X",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Coverage Baselines vs OT on Rare-Mode Stress Test (S=40, sp=0.2)", fontsize=13)

    for bl in baselines:
        sub = df[df["baseline"] == bl]
        c = baseline_colors.get(bl, "#333333")
        m = baseline_markers.get(bl, "o")

        # Panel 1: Collision rate
        axes[0].errorbar(sub["rare_prob"], sub["collision_rate"],
                         yerr=[sub["collision_rate"] - sub["ci_lo"],
                               sub["ci_hi"] - sub["collision_rate"]],
                         label=bl, color=c, marker=m, markersize=5, capsize=3, linewidth=1.5)

        # Panel 2: Missed-mode rate
        axes[1].plot(sub["rare_prob"], sub["missed_mode_rate"],
                     label=bl, color=c, marker=m, markersize=5, linewidth=1.5)

        # Panel 3: Rare-mode miss fraction
        axes[2].plot(sub["rare_prob"], sub["rare_mode_missed_frac"],
                     label=bl, color=c, marker=m, markersize=5, linewidth=1.5)

    axes[0].set_xlabel("Rare-mode probability")
    axes[0].set_ylabel("Collision rate")
    axes[0].set_title("Collision Rate")
    axes[0].axhline(0.05, ls="--", color="gray", alpha=0.5, label=r"$\varepsilon$ target")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel("Rare-mode probability")
    axes[1].set_ylabel("Missed-mode rate (all modes)")
    axes[1].set_title("Overall Mode Coverage")
    axes[1].legend(fontsize=7, ncol=2)

    axes[2].set_xlabel("Rare-mode probability")
    axes[2].set_ylabel("Rare-mode miss fraction")
    axes[2].set_title("Rare-Mode Miss Rate")
    axes[2].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig29_baselines_rare_mode.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Fig 30: Geometry Ablation with Shuffled/Random Cost (Exp Y)
# ============================================================================

def fig30_geometry_ablation():
    df = load_csv("exp_y_geometry_ablation.csv")
    if df is None:
        return

    costs = df["ground_cost"].values
    cr = df["collision_rate"].values
    ci_lo = df["ci_lo"].values
    ci_hi = df["ci_hi"].values
    mmr = df["missed_mode_rate"].values
    p99 = df["p99_solve_ms"].values

    # Try to load paired data for McNemar tests
    paired_df = load_csv("exp_y_paired.csv")
    mcnemar_labels = {}
    if paired_df is not None and "W2-Euclidean" in paired_df.columns:
        from scipy.stats import chi2 as chi2_dist
        w2_col = paired_df["W2-Euclidean"].values
        for c in costs:
            if c == "W2-Euclidean":
                continue
            if c in paired_df.columns:
                other_col = paired_df[c].values
                # McNemar: b = W2 safe & other collision, c = W2 coll & other safe
                b = int(np.sum((w2_col == 0) & (other_col == 1)))
                mc_c = int(np.sum((w2_col == 1) & (other_col == 0)))
                if b + mc_c > 0:
                    chi2_val = (abs(b - mc_c) - 1) ** 2 / (b + mc_c)
                    p_val = 1 - chi2_dist.cdf(chi2_val, df=1)
                else:
                    p_val = 1.0
                if p_val < 0.001:
                    mcnemar_labels[c] = "***"
                elif p_val < 0.01:
                    mcnemar_labels[c] = "**"
                elif p_val < 0.05:
                    mcnemar_labels[c] = "*"
                else:
                    mcnemar_labels[c] = f"p={p_val:.2f}"

    # Color W2 green (our method), controls red/orange
    bar_colors = []
    for c in costs:
        if c == "W2-Euclidean":
            bar_colors.append("#2ca02c")
        elif c in ("Random-Permuted", "Constant"):
            bar_colors.append("#d62728")
        else:
            bar_colors.append("#ff7f0e")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    n_rollouts = max(int(round(1.0 / (ci_hi[0] - ci_lo[0]) * 4)), 200)  # estimate
    n_rollouts_str = str(n_rollouts) if paired_df is None else str(len(paired_df))
    fig.suptitle(f"OT Geometry Ablation: Semantic vs Broken Cost Matrices (sp=0.28, S=40, N={n_rollouts_str})",
                 fontsize=13)

    # Panel 1: Collision rate with McNemar annotations
    x = np.arange(len(costs))
    yerr = [cr - ci_lo, ci_hi - cr]
    bars = axes[0].bar(x, cr, color=bar_colors, yerr=yerr, capsize=4,
                       edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(costs, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Collision rate")
    axes[0].set_title("Collision Rate by Ground Cost")
    axes[0].axhline(0.05, ls="--", color="gray", alpha=0.5, label=r"$\varepsilon$ target")

    # Add McNemar p-value annotations
    for i, c in enumerate(costs):
        if c in mcnemar_labels:
            axes[0].text(i, ci_hi[i] + 0.005, mcnemar_labels[c],
                         ha="center", va="bottom", fontsize=7,
                         color="darkred" if mcnemar_labels[c].startswith("*") else "gray")

    axes[0].legend(fontsize=8)

    # Panel 2: Missed-mode rate
    axes[1].bar(x, mmr, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(costs, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("Missed-mode rate")
    axes[1].set_title("Mode Coverage by Ground Cost")

    # Panel 3: P99 solve time
    axes[2].bar(x, p99, color=bar_colors, edgecolor="black", linewidth=0.5)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(costs, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("P99 solve time (ms)")
    axes[2].set_title("Tail Latency (P99)")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig30_geometry_ablation.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Fig 31: Qualitative Rollout Montage (Exp Z)
# ============================================================================

def fig31_qualitative_rollouts():
    df = load_csv("exp_z_qualitative_trajectories.csv")
    if df is None:
        return

    seeds = df["seed"].unique()
    variants = ["Base", "OT", "OT_SH"]
    variant_labels = {"Base": "Base (Freq.)", "OT": "OT", "OT_SH": "OT+SH"}
    variant_colors = {"Base": "#1f77b4", "OT": "#2ca02c", "OT_SH": "#98df8a"}

    n_seeds = min(len(seeds), 4)
    fig, axes = plt.subplots(n_seeds, 3, figsize=(14, 3.5 * n_seeds))
    if n_seeds == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Qualitative Rollout Comparison: Base vs OT vs OT+SH", fontsize=14, y=1.02)

    for row, seed in enumerate(seeds[:n_seeds]):
        for col, var in enumerate(variants):
            ax = axes[row, col]
            sub = df[(df["seed"] == seed) & (df["variant"] == var)]
            if sub.empty:
                ax.set_visible(False)
                continue

            # Plot ego trajectory
            ax.plot(sub["ego_x"], sub["ego_y"], "-", color=variant_colors[var],
                    linewidth=1.5, label="Ego", zorder=3)

            # Plot obstacle trajectory
            ax.plot(sub["obs_x"], sub["obs_y"], "--", color="#666666",
                    linewidth=1.0, label="Obstacle", zorder=2)

            # Mark collisions
            coll = sub[sub["collision"] == 1]
            if not coll.empty:
                ax.scatter(coll["ego_x"], coll["ego_y"], c="red", s=40,
                           marker="x", zorder=5, label="Collision")

            # Mark missed modes
            missed = sub[sub["missed_mode"] == 1]
            if len(missed) > 0:
                # Sample a few to avoid clutter
                sample_idx = missed.index[::max(1, len(missed) // 8)]
                ax.scatter(missed.loc[sample_idx, "obs_x"],
                           missed.loc[sample_idx, "obs_y"],
                           c="orange", s=15, marker="^", alpha=0.6, zorder=4,
                           label="Missed mode")

            # Color background by mode
            mode_changes = []
            prev_mode = None
            for _, r in sub.iterrows():
                if r["obs_mode"] != prev_mode:
                    mode_changes.append((r["step"], r["obs_mode"]))
                    prev_mode = r["obs_mode"]

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if row == 0:
                ax.set_title(variant_labels[var], fontsize=11, fontweight="bold")

            has_coll = not coll.empty
            min_clear = sub["clearance"].min() if "clearance" in sub.columns else 0
            ax.text(0.02, 0.98,
                    f"{'COLLISION' if has_coll else 'Safe'}\nmin d={min_clear:.2f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    color="red" if has_coll else "green",
                    fontweight="bold")

            if col == 0:
                ax.set_ylabel(f"Seed {seed}\ny", fontsize=9)

            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig31_qualitative_rollouts.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Fig 32: Robustness Across Environments (Boxplots, Exp AA)
# ============================================================================

def fig32_robustness_boxplots():
    df = load_csv("exp_aa_robustness_per_seed.csv")
    if df is None:
        return

    envs = df["environment"].unique()
    variants = ["Base", "OT", "OT+SH"]
    var_colors = {"Base": "#1f77b4", "OT": "#2ca02c", "OT+SH": "#98df8a"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Robustness Across Environments: Per-Seed Distributions", fontsize=14)

    metrics = [
        ("collision", "Collision (0/1)", axes[0, 0]),
        ("missed_mode_rate", "Missed-Mode Rate", axes[0, 1]),
        ("progress", "Goal Progress (x)", axes[1, 0]),
        ("min_clearance", "Min Clearance (m)", axes[1, 1]),
    ]

    for metric_col, metric_label, ax in metrics:
        positions = []
        data = []
        labels = []
        colors = []
        width = 0.6
        n_variants = len(variants)
        n_envs = len(envs)

        for i, env in enumerate(envs):
            for j, var in enumerate(variants):
                sub = df[(df["environment"] == env) & (df["variant"] == var)]
                if sub.empty:
                    continue
                pos = i * (n_variants + 1) + j
                positions.append(pos)
                data.append(sub[metric_col].values)
                labels.append(f"{env}\n{var}" if j == 1 else "")
                colors.append(var_colors.get(var, "#999"))

        bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                        showfliers=True, flierprops=dict(markersize=2))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Set x-axis labels at environment centers
        env_centers = [i * (n_variants + 1) + 1 for i in range(n_envs)]
        ax.set_xticks(env_centers)
        ax.set_xticklabels(envs, fontsize=9)
        ax.set_ylabel(metric_label)

        # For collision, show mean rate as text
        if metric_col == "collision":
            for i, env in enumerate(envs):
                for j, var in enumerate(variants):
                    sub = df[(df["environment"] == env) & (df["variant"] == var)]
                    if not sub.empty:
                        cr = sub["collision"].mean()
                        pos = i * (n_variants + 1) + j
                        ax.text(pos, ax.get_ylim()[1] * 0.95, f"{cr:.0%}",
                                ha="center", va="top", fontsize=7, fontweight="bold",
                                color=var_colors.get(var, "#999"))

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=var_colors[v], alpha=0.7, label=v)
                       for v in variants]
    axes[0, 0].legend(handles=legend_elements, fontsize=8, loc="upper right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig32_robustness_boxplots.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Fig 33: Pareto Frontier - Collision vs Missed-Mode (Exp AB)
# ============================================================================

def fig33_pareto_frontier():
    df = load_csv("exp_ab_pareto_frontier.csv")
    if df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(r"OT Regularization Pareto Frontier: Collision $\leftrightarrow$ Mode Coverage",
                 fontsize=13)

    # Split into OT and OT+SH
    df_ot = df[df["use_sh"] == 0]
    df_sh = df[df["use_sh"] == 1]

    # Panel 1: Pareto plane (collision vs missed-mode), colored by epsilon
    ax = axes[0]

    # Color by log(epsilon)
    import matplotlib.colors as mcolors
    norm = mcolors.LogNorm(vmin=df["epsilon"].min(), vmax=df["epsilon"].max())
    cmap = plt.cm.viridis

    sc1 = ax.scatter(df_ot["collision_rate"], df_ot["missed_mode_rate"],
                     c=df_ot["epsilon"], cmap=cmap, norm=norm,
                     s=40, marker="o", edgecolor="black", linewidth=0.3,
                     label="OT", alpha=0.8, zorder=3)
    ax.scatter(df_sh["collision_rate"], df_sh["missed_mode_rate"],
               c=df_sh["epsilon"], cmap=cmap, norm=norm,
               s=60, marker="D", edgecolor="black", linewidth=0.5,
               label="OT+SH", alpha=0.9, zorder=4)

    # Mark the default operating point (eps=0.1, scale=1.0)
    default_ot = df_ot[(df_ot["epsilon"].between(0.09, 0.11)) &
                        (df_ot["uncertainty_scale"].between(0.9, 1.1))]
    if not default_ot.empty:
        ax.scatter(default_ot["collision_rate"].values[0],
                   default_ot["missed_mode_rate"].values[0],
                   c="red", s=150, marker="*", zorder=5,
                   label=r"Default ($\varepsilon$=0.1, $\sigma$=1.0)")

    ax.set_xlabel("Collision Rate")
    ax.set_ylabel("Missed-Mode Rate")
    ax.set_title("Safety vs Coverage Tradeoff")
    ax.axhline(0.05, ls=":", color="gray", alpha=0.3)
    ax.axvline(0.05, ls=":", color="gray", alpha=0.3)
    ax.legend(fontsize=8)
    cb = plt.colorbar(sc1, ax=ax, label=r"Sinkhorn $\varepsilon$")

    # Panel 2: Collision rate vs epsilon, grouped by uncertainty_scale
    ax2 = axes[1]
    scales = sorted(df_ot["uncertainty_scale"].unique())
    scale_colors = plt.cm.coolwarm(np.linspace(0, 1, len(scales)))

    for sc_val, color in zip(scales, scale_colors):
        sub = df_ot[df_ot["uncertainty_scale"] == sc_val].sort_values("epsilon")
        ax2.plot(sub["epsilon"], sub["collision_rate"],
                 "o-", color=color, markersize=5, linewidth=1.2,
                 label=fr"$\sigma$={sc_val}")

    ax2.set_xscale("log")
    ax2.set_xlabel(r"Sinkhorn $\varepsilon$ (log scale)")
    ax2.set_ylabel("Collision Rate")
    ax2.set_title(r"Collision Rate vs $\varepsilon$ by Uncertainty Scale")
    ax2.axhline(0.05, ls="--", color="gray", alpha=0.5, label=r"$\varepsilon$ target")
    ax2.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fig33_pareto_frontier.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Generating figures from CSV data...\n")

    fig1_collision_vs_switching()
    fig2_mismatch_vs_time()
    fig3_safety_performance_pareto()
    fig4_rare_mode_collision()
    fig5_solve_time_distributions()
    fig6_ablation_table()
    fig7_calibration_plot()
    fig8_buffer_sensitivity()
    fig9_conservatism_smoothness()
    fig10_forest_plot()
    fig11_missed_mode_significance()
    fig12_ablation_table_1000()
    fig13_full_ablation()
    fig14_sh_scaling()
    fig15_ot_vs_baselines()
    fig16_environment_generalization()
    fig17_empirical_violation()
    fig18_sh_sweep()
    fig19_runtime_scaling()
    fig20_distribution_shift()
    fig21_tradeoff_plane()
    fig22_coverage_baselines()
    fig23_ot_ablation()
    fig24_mode_coverage()
    fig25_missed_mode_vs_s()
    fig26_ground_cost()
    fig27_rare_mode_sweep()
    fig28_mode_scaling()
    fig29_baselines_rare_mode()
    fig30_geometry_ablation()
    fig31_qualitative_rollouts()
    fig32_robustness_boxplots()
    fig33_pareto_frontier()

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
