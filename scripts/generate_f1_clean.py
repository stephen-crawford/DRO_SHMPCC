#!/usr/bin/env python3
"""Generate clean F1 figure: 6 methods, consistent row order across panels."""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIR = pathlib.Path("figures/f1")

COLORS = {
    "Base":            "#999999",
    "WDRO-sampling":   "#FF7F00",
    "WDRO-inject-K1":  "#2166AC",
    "WDRO-inject-K2":  "#4393C3",
    "TopRisk-K1":      "#B2182B",
    "TopRisk-K2":      "#E08080",
}

# Fixed display order (top to bottom)
METHOD_ORDER = [
    "Base",
    "WDRO-sampling",
    "WDRO-inject-K1",
    "WDRO-inject-K2",
    "TopRisk-K1",
    "TopRisk-K2",
]

df = pd.read_csv(DIR / "f1_high_n_oncoming.csv")
df = df[df["method"].isin(METHOD_ORDER)]
speeds = sorted(df["obstacle_speed"].unique())

fig, axes = plt.subplots(1, len(speeds), figsize=(7 * len(speeds), 5.5))
if len(speeds) == 1:
    axes = [axes]

for ax, spd in zip(axes, speeds):
    sub = df[df["obstacle_speed"] == spd]

    # Enforce consistent order
    sub = sub.set_index("method").loc[METHOD_ORDER].reset_index()
    y = np.arange(len(sub))

    colors = [COLORS[m] for m in sub["method"]]
    cr = sub["collision_rate"].values
    lo = cr - sub["coll_ci_lo"].values
    hi = sub["coll_ci_hi"].values - cr

    bars = ax.barh(y, cr, xerr=[lo, hi], color=colors, capsize=5,
                   alpha=0.88, height=0.65, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["method"], fontsize=10)
    ax.set_xlabel("Collision Rate", fontsize=11)
    ax.set_title(f"Obstacle Speed = {spd:.1f} m/s   (N = 2000)", fontsize=12)
    ax.invert_yaxis()

    # Value labels
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(row["coll_ci_hi"] + 0.005, i,
                f'{row["collision_rate"]*100:.1f}%',
                va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(0.25, cr.max() * 1.12))
    ax.grid(axis="x", alpha=0.2)

fig.suptitle("Injection Method Comparison — Collision Rate with 95% Wilson CIs",
             fontsize=14, y=1.01)
fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(DIR / f"fig_f1_method_ranking.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved fig_f1_method_ranking")
