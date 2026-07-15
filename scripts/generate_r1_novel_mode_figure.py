#!/usr/bin/env python3
"""Generate a compact figure summarizing R1: Novel (Unseen) Mode results.

Matches the style of optA_fig2_challenging_envs exactly:
  - Same rcParams, color palette, nice names
  - Panel (a): collision rate with Wilson CI whiskers
  - Panel (b): mean clearance, no whiskers (plain bars)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = "figures/robustness/r1_novel_mode.csv"
OUT = "figures/robustness/fig_r1_novel_mode"

# --- Standardized style (matches generate_paper_figure_options.py) -----------
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

# --- Shared palette (matches generate_paper_figure_options.py) ---------------
COLORS = {
    "Base":             "#888888",
    "WDRO-sampling":    "#E6850E",
    "WDRO-inject-K1":   "#2166AC",
    "WDRO-inject-K2":   "#4393C3",
}
NICE = {
    "Base":             "Base SH-MPCC",
    "WDRO-sampling":    "WDRO-sampling",
    "WDRO-inject-K1":   "WDRO-$1$",
    "WDRO-inject-K2":   "WDRO-$2$",
}
NICE_NOVEL = {
    "Spiral": "Spiral",
    "HardBrake": "Hard Brake",
    "U-turn": "U-turn",
    "Skid": "Skid",
}

def _c(m): return COLORS.get(m, "#666")
def _n(m): return NICE.get(m, m)

df = pd.read_csv(CSV)

methods = ["Base", "WDRO-sampling", "WDRO-inject-K1", "WDRO-inject-K2"]
methods = [m for m in methods if m in df["method"].unique()]
novels = ["Spiral", "HardBrake", "U-turn", "Skid"]
novels = [n for n in novels if n in df["novel_mode"].unique()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.32})

x = np.arange(len(novels))
n_m = len(methods)
w = 0.8 / n_m

for i, method in enumerate(methods):
    sub = df[df["method"] == method]
    cr, cr_lo, cr_hi, clr = [], [], [], []
    for n in novels:
        row = sub[sub["novel_mode"] == n]
        if len(row):
            cr.append(row.iloc[0]["collision_rate"] * 100)
            cr_lo.append(row.iloc[0]["coll_ci_lo"] * 100)
            cr_hi.append(row.iloc[0]["coll_ci_hi"] * 100)
            clr.append(row.iloc[0]["mean_clearance"])
        else:
            cr.append(0); cr_lo.append(0); cr_hi.append(0); clr.append(0)

    cr = np.array(cr)
    err = np.array([cr - np.array(cr_lo), np.array(cr_hi) - cr])
    offset = (i - n_m / 2 + 0.5) * w

    # Panel (a): collision rate with Wilson CI whiskers
    ax1.bar(x + offset, cr, w * 0.88, yerr=err, label=_n(method),
            color=_c(method), capsize=2,
            error_kw={"linewidth": 0.7}, edgecolor="white", linewidth=0.3)

    # Panel (b): mean clearance, no whiskers
    ax2.bar(x + offset, np.array(clr), w * 0.88,
            color=_c(method), edgecolor="white", linewidth=0.3)

ax1.set_xticks(x)
ax1.set_xticklabels([NICE_NOVEL.get(n, n) for n in novels])
ax1.set_ylabel("Collision Rate (%)")
ax1.set_title("(a) Collision Rate", fontsize=9.5)
ax1.set_ylim(0, 90)

ax2.set_xticks(x)
ax2.set_xticklabels([NICE_NOVEL.get(n, n) for n in novels])
ax2.set_ylabel("Mean Clearance (m)")
ax2.set_title("(b) Mean Clearance", fontsize=9.5)
ax2.set_ylim(0, 1.35)

fig.suptitle("Mode-Based WDRO Maintains Safety Against Unseen Obstacle Behaviors",
             fontsize=11, y=1.08)

# Shared legend above both panels
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=len(methods),
           bbox_to_anchor=(0.5, 1.02), framealpha=0.9, edgecolor="0.85")

fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}.pdf and {OUT}.png")
