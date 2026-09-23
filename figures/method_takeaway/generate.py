"""Standalone paper artwork; does not import or modify controller mathematics."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dro-paper-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

OUT = Path(__file__).resolve().parent
INK, MUTED = "#172C40", "#536577"
BLUE, TEAL, ORANGE = "#3175AC", "#087F78", "#C76530"
PALE, LINE = "#EDF5F8", "#CCD9E1"
NOMINAL = np.array([.70, .20, .10])
REWEIGHTED = np.array([.35, .25, .40])
RATES = [28.24, 28.94, 3.31]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "text.color": INK, "axes.labelcolor": MUTED,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def label(ax, x, y, s, size=10, color=INK, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, va="center", **kw)


def box(ax, x, y, w, h, s, color=PALE, edge=LINE, size=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.008,rounding_size=0.012",
                 facecolor=color, edgecolor=edge, linewidth=1))
    label(ax, x+w/2, y+h/2, s, size, ha="center", linespacing=1.5)


def arrow(ax, start, end, color=MUTED, **kw):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.25,
                                shrinkA=2, shrinkB=2, **kw))


def base(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")


def method(ax):
    base(ax)
    label(ax, 0, .965, "(a)  From observations to scenario weights", 15, weight="bold")
    label(ax, 0, .924, "Calibrated ambiguity  →  geometry-aware reweighting", 10, MUTED)
    box(ax, .015, .812, .45, .065, "Observed mode history")
    box(ax, .015, .698, .45, .074, "Nominal class belief  $\\widehat{p}_t$")
    arrow(ax, (.24, .812), (.24, .772))
    box(ax, .015, .575, .45, .08, "Clopper–Pearson confidence region\nwith simultaneous coverage", size=9)
    arrow(ax, (.24, .698), (.24, .655))
    box(ax, .015, .454, .45, .077, "Calibrated Wasserstein ball\n$\\mathcal{B}_W(\\widehat{p}_t,\\rho_t^v)$", size=10)
    arrow(ax, (.24, .575), (.24, .531))
    box(ax, .55, .698, .43, .074, "Predicted mode trajectories")
    box(ax, .55, .454, .43, .077, "Geometry-aware risk scores\n$r_m$ for each mode", color="#FBF1E9")
    arrow(ax, (.765, .698), (.765, .531))
    box(ax, .22, .317, .57, .087,
        "Worst-case mode reweighting\n$\\max_{q\\in\\mathcal{B}_W}\\;\\sum_m q_m r_m$", color="#E3F1EF", edge=TEAL)
    arrow(ax, (.24, .454), (.36, .404))
    arrow(ax, (.765, .454), (.65, .404))
    box(ax, .22, .192, .57, .075, "Reweighted distribution  $q^\\star$")
    arrow(ax, (.505, .317), (.505, .267))
    box(ax, .045, .064, .40, .074, "Scenario sampling")
    box(ax, .60, .064, .36, .074, "SH-MPCC")
    arrow(ax, (.505, .192), (.245, .138))
    arrow(ax, (.445, .101), (.60, .101))


def reweighting(ax):
    base(ax)
    label(ax, 0, .97, "Illustrative probability transport", 13, weight="bold")
    label(ax, 0, .905, "Example weights; not a rollout or an optimization result.", 8.5, MUTED)
    for i, (name, p, q) in enumerate(zip(["Straight", "Turn left", "Turn right"], NOMINAL, REWEIGHTED)):
        y = .74-i*.205
        label(ax, 0, y, name, 9, weight="bold")
        ax.barh(y+.037, p*.48, height=.053, left=.23, color=BLUE)
        ax.barh(y-.037, q*.48, height=.053, left=.23, color=ORANGE if i == 2 else TEAL)
        label(ax, .24+p*.48, y+.037, f"{p:.2f}", 8)
        label(ax, .24+q*.48, y-.037, f"{q:.2f}", 8)
    label(ax, .23, .115, "Nominal", 9, BLUE)
    label(ax, .43, .115, "Reweighted", 9, TEAL)
    label(ax, 0, .015, "Higher-risk turn: 0.10 → 0.40", 10, ORANGE, weight="bold")


def simplex(ax):
    base(ax)
    label(ax, .5, .97, "Transport-budget intuition", 12, ha="center", weight="bold")
    vertices = np.array([[.13, .20], [.90, .20], [.515, .80]])
    ax.add_patch(Polygon(vertices, fill=False, edgecolor=LINE, lw=1.4))
    p, q = NOMINAL @ vertices, REWEIGHTED @ vertices
    # A deliberately schematic region, not a computed Wasserstein ball.
    region = np.array([[.25, .205], [.53, .205], [.61, .31], q,
                       [.30, .43], [.22, .30]])
    ax.add_patch(Polygon(region, facecolor="#E3F1EF", edgecolor=TEAL,
                         linestyle="--", linewidth=1.2))
    arrow(ax, p, q, TEAL)
    ax.scatter(*p, s=35, color=BLUE, zorder=4)
    ax.scatter(*q, s=35, color=ORANGE, zorder=4)
    label(ax, p[0]-.07, p[1]-.025, "$\\widehat{p}_t$", 12)
    label(ax, q[0]+.035, q[1]+.015, "$q^\\star$", 12)
    label(ax, .10, .145, "Mode 1", 8)
    label(ax, .79, .145, "Mode 2", 8)
    label(ax, .515, .85, "Mode 3 · higher risk", 9, ORANGE, ha="center")
    label(ax, .5, .065, "Transport limited by $\\rho_t^v$", 10, TEAL, ha="center")
    label(ax, .5, -.015, "Schematic region; geometry is not calibrated here.", 7.5, MUTED, ha="center")


def hybrid(ax):
    base(ax)
    label(ax, 0, .965, "(b)  Hybrid control and empirical takeaway", 15, weight="bold")
    label(ax, 0, .914, "Retry with nominal weights when the WDRO attempt fails", 10, MUTED)
    box(ax, .025, .735, .36, .095, "WDRO-SH-MPCC\nadmissible control?", color="#E3F1EF", edge=TEAL)
    box(ax, .60, .735, .37, .095, "Baseline SH-MPCC\nadmissible control?")
    arrow(ax, (.385, .782), (.60, .782))
    label(ax, .49, .817, "No · retry", 8, MUTED, ha="center")
    box(ax, .025, .566, .36, .075, "Apply WDRO control", color="#E3F1EF")
    box(ax, .60, .566, .37, .075, "Apply baseline control")
    for x in [.205, .785]:
        arrow(ax, (x, .735), (x, .641))
        label(ax, x+.025, .688, "Yes", 8, TEAL)
    arrow(ax, (.97, .782), (.985, .473), connectionstyle="angle,angleA=0,angleB=90,rad=8")
    label(ax, .98, .665, "No", 8, ORANGE, ha="right")
    label(ax, .97, .46, "Neither admissible → stop; log failure", 9, ORANGE, ha="right")
    label(ax, 0, .37, "No-admissible-control rate", 13, weight="bold")
    label(ax, 0, .317, "Overall matched rollouts · lower is better", 9, MUTED)
    for y, name, rate, color in zip([.241, .156, .071], ["SH-MPCC", "WDRO", "Hybrid"], RATES, [BLUE, MUTED, TEAL]):
        label(ax, 0, y, name, 10, weight="bold" if name == "Hybrid" else "normal")
        ax.barh(y, rate/35*.63, left=.22, height=.05, color=color)
        label(ax, .235+rate/35*.63, y, f"{rate:.2f}%", 10, color, weight="bold")


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Exported {name}: PDF, SVG, PNG")


def main():
    fig = plt.figure(figsize=(16, 10))
    label(fig.add_axes([.045, .92, .91, .07], frameon=False), 0, .5,
          "Risk-aware sampling, backed by a nominal retry", 22, weight="bold")
    fig.axes[-1].axis("off")
    method(fig.add_axes([.045, .32, .43, .59]))
    reweighting(fig.add_axes([.045, .075, .25, .235]))
    simplex(fig.add_axes([.31, .075, .165, .235]))
    hybrid(fig.add_axes([.54, .32, .415, .59]))
    fig.text(.54, .225, "3.31%", fontsize=43, color=TEAL, weight="bold")
    fig.text(.54, .18, "Hybrid no-admissible-control rate", fontsize=13)
    fig.text(.54, .125, "24.93 percentage points below baseline", fontsize=12, color=TEAL)
    fig.text(.045, .025, "Rates: supplied paper recommendation; not independently reproduced.  |  Probability weights and simplex region: illustrative.", fontsize=9, color=MUTED)
    save(fig, "method_takeaway")

    fig = plt.figure(figsize=(9, 10))
    method(fig.add_axes([.07, .35, .86, .62]))
    reweighting(fig.add_axes([.07, .055, .50, .27]))
    simplex(fig.add_axes([.64, .055, .29, .27]))
    save(fig, "mode_reweighting")

    fig = plt.figure(figsize=(9, 7))
    hybrid(fig.add_axes([.08, .12, .84, .84]))
    fig.text(.08, .035, "Source: supplied paper recommendation. Rates not independently reproduced.", fontsize=9, color=MUTED)
    save(fig, "hybrid_results")


if __name__ == "__main__":
    main()
