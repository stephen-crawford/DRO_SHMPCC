#!/usr/bin/env python3
"""Generate figure for G9: Mode Scaling — WDRO vs TopRisk injection as mode count grows."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = "figures/generalization/g9_mode_scaling.csv"
OUT = "paper_figures/fig_g9_mode_scaling"

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

COLORS = {
    "Base":             "#888888",
    "WDRO-sampling":    "#E6850E",
    "WDRO-inject-K1":   "#2166AC",
    "WDRO-inject-K2":   "#4393C3",
    "WDRO-inject-K3":   "#92C5DE",
    "TopRisk-K1":       "#B2182B",
    "TopRisk-K2":       "#E06666",
    "TopRisk-K3":       "#F4A582",
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
}
MARKERS = {
    "Base": "s", "WDRO-sampling": "D",
    "WDRO-inject-K1": "o", "WDRO-inject-K2": "v", "WDRO-inject-K3": "^",
    "TopRisk-K1": "o", "TopRisk-K2": "v", "TopRisk-K3": "^",
}
LINESTYLES = {
    "Base": "-", "WDRO-sampling": "-",
    "WDRO-inject-K1": "-", "WDRO-inject-K2": "--", "WDRO-inject-K3": ":",
    "TopRisk-K1": "-", "TopRisk-K2": "--", "TopRisk-K3": ":",
}

df = pd.read_csv(CSV)

methods = ["WDRO-inject-K1", "WDRO-inject-K2", "WDRO-inject-K3",
           "TopRisk-K1", "TopRisk-K2", "TopRisk-K3"]
methods = [m for m in methods if m in df["method"].unique()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.30})

mode_counts = sorted(df["num_modes"].unique())

for method in methods:
    sub = df[df["method"] == method].sort_values("num_modes")
    x = sub["num_modes"].values
    cr = sub["collision_rate"].values * 100
    cr_lo = sub["coll_ci_lo"].values * 100
    cr_hi = sub["coll_ci_hi"].values * 100
    clr = sub["mean_clearance"].values

    ax1.plot(x, cr, color=COLORS.get(method, "#666"), marker=MARKERS.get(method, "o"),
             ls=LINESTYLES.get(method, "-"), ms=4, lw=1.3, label=NICE.get(method, method))
    ax1.fill_between(x, cr_lo, cr_hi, color=COLORS.get(method, "#666"), alpha=0.12)

    ax2.plot(x, clr, color=COLORS.get(method, "#666"), marker=MARKERS.get(method, "o"),
             ls=LINESTYLES.get(method, "-"), ms=4, lw=1.3)

ax1.set_xlabel("Number of Modes")
ax1.set_ylabel("Collision Rate (%)")
ax1.set_title("(a) Collision Rate vs. Mode Count", fontsize=9.5)
ax1.set_xticks(mode_counts)

ax2.set_xlabel("Number of Modes")
ax2.set_ylabel("Mean Clearance (m)")
ax2.set_title("(b) Mean Clearance vs. Mode Count", fontsize=9.5)
ax2.set_xticks(mode_counts)

fig.suptitle("WDRO and TopRisk Injection across Mode Count",
             fontsize=11, y=1.06)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6,
           bbox_to_anchor=(0.5, 1.01), framealpha=0.9, edgecolor="0.85")

fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}.pdf and {OUT}.png")
