#!/usr/bin/env python3
"""Visualize all generalization path geometries used in G1–G5 / T1–T5 experiments.

Reproduces the reference paths mathematically and shows:
  - All five path geometries in a single overview figure
  - Obstacle placement zones for each path
  - Path curvature profiles
  - Multi-obstacle placement for G2/G5 experiments
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

DIR = "generalization_figures"
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

PATH_COLORS = {
    "Straight": "#1f77b4",
    "Gentle-S": "#2ca02c",
    "S-curve":  "#ff7f0e",
    "Tight-S":  "#d62728",
    "Circle":   "#9467bd",
}


# ============================================================================
# Path geometry generators (matching C++ ReferencePath implementations)
# ============================================================================

def make_straight(start=(0, 0), end=(25, 0), n=500):
    """Straight line from start to end."""
    t = np.linspace(0, 1, n)
    x = start[0] + t * (end[0] - start[0])
    y = start[1] + t * (end[1] - start[1])
    length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    s = t * length
    heading = np.full(n, np.arctan2(end[1] - start[1], end[0] - start[0]))
    curvature = np.zeros(n)
    return x, y, s, heading, curvature, length


def make_s_curve(length=25.0, amplitude=3.0, n=500):
    """S-curve: y = amplitude * sin(2*pi*x/length)."""
    x_vals = np.linspace(0, length, n)
    y_vals = amplitude * np.sin(2 * np.pi * x_vals / length)

    # Compute arc length numerically
    dx = np.diff(x_vals)
    dy = np.diff(y_vals)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.zeros(n)
    s[1:] = np.cumsum(ds)
    total_length = s[-1]

    # Heading
    dy_dx = amplitude * 2 * np.pi / length * np.cos(2 * np.pi * x_vals / length)
    heading = np.arctan2(dy_dx, np.ones_like(dy_dx))

    # Curvature: kappa = y'' / (1 + y'^2)^(3/2)
    d2y_dx2 = -amplitude * (2 * np.pi / length)**2 * np.sin(2 * np.pi * x_vals / length)
    curvature = d2y_dx2 / (1 + dy_dx**2)**(1.5)

    return x_vals, y_vals, s, heading, curvature, total_length


def make_circle(center=(0, -10), radius=10.0, start_angle=np.pi / 2, end_angle=0.0, n=500):
    """Circular arc."""
    angles = np.linspace(start_angle, end_angle, n)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    angle_span = end_angle - start_angle
    total_length = abs(radius * angle_span)
    t = np.linspace(0, 1, n)
    s = t * total_length
    heading = angles - np.pi / 2  # tangent is 90 deg from radius
    if angle_span < 0:
        heading = angles + np.pi / 2
    curvature = np.full(n, 1.0 / radius)
    return x, y, s, heading, curvature, total_length


def obstacle_position_on_path(x, y, s, heading, total_length, arc_fraction, oncoming=True):
    """Get obstacle position at given arc-length fraction."""
    target_s = arc_fraction * total_length
    idx = np.argmin(np.abs(s - target_s))
    ox, oy = x[idx], y[idx]
    h = heading[idx]
    # Obstacle velocity direction
    if oncoming:
        vx = -np.cos(h) * 0.8
        vy = -np.sin(h) * 0.8
    else:
        vx = np.cos(h) * 0.8
        vy = np.sin(h) * 0.8
    return ox, oy, vx, vy, idx


# ============================================================================
# Figure 1: All path geometries overview
# ============================================================================
def fig_path_overview():
    paths = {
        "Straight": make_straight(),
        "Gentle-S": make_s_curve(25.0, 1.0),
        "S-curve":  make_s_curve(25.0, 3.0),
        "Tight-S":  make_s_curve(20.0, 5.0),
        "Circle":   make_circle(),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    # Panel 0: All paths overlaid
    ax = axes[0]
    for name, (x, y, s, h, k, L) in paths.items():
        ax.plot(x, y, label=f"{name} (L={L:.1f}m)", color=PATH_COLORS[name], linewidth=1.8)
        # Ego start
        ax.plot(x[0], y[0], 'o', color=PATH_COLORS[name], markersize=6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("All Path Geometries")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Panels 1-5: Individual paths with obstacle placement
    for panel_i, (name, (x, y, s, h, k, L)) in enumerate(paths.items(), start=1):
        ax = axes[panel_i]
        ax.plot(x, y, color=PATH_COLORS[name], linewidth=2)

        # Mark ego start
        ax.plot(x[0], y[0], 's', color='green', markersize=8, zorder=5, label='Ego start')

        # Obstacle at 55% arc length (single-obstacle experiments)
        ox, oy, vx, vy, oidx = obstacle_position_on_path(x, y, s, h, L, 0.55)
        ax.plot(ox, oy, 'o', color='red', markersize=8, zorder=5, label='Obstacle (55%)')
        ax.annotate('', xy=(ox + vx * 2, oy + vy * 2), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # Multi-obstacle positions (G2/T2: 30%, 45%, 65%)
        for frac, marker in [(0.30, '^'), (0.45, 'D'), (0.65, 'v')]:
            mx, my, mvx, mvy, _ = obstacle_position_on_path(x, y, s, h, L, frac)
            ax.plot(mx, my, marker, color='orange', markersize=6, zorder=4, alpha=0.6)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{name} (L={L:.1f}m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if panel_i == 1:
            ax.legend(fontsize=6.5, loc="best")

    fig.suptitle("Generalization Experiment: Path Geometries & Obstacle Placement", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_path_overview.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Path overview saved")


# ============================================================================
# Figure 2: Curvature profiles
# ============================================================================
def fig_curvature_profiles():
    paths = {
        "Straight": make_straight(),
        "Gentle-S": make_s_curve(25.0, 1.0),
        "S-curve":  make_s_curve(25.0, 3.0),
        "Tight-S":  make_s_curve(20.0, 5.0),
        "Circle":   make_circle(),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Curvature vs arc length
    for name, (x, y, s, h, k, L) in paths.items():
        # Normalize arc length to [0, 1]
        ax1.plot(s / L, np.abs(k), label=name, color=PATH_COLORS[name], linewidth=1.5)

    ax1.set_xlabel("Normalized Arc Length")
    ax1.set_ylabel("|Curvature| (1/m)")
    ax1.set_title("Curvature Profile")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Heading angle vs arc length
    for name, (x, y, s, h, k, L) in paths.items():
        ax2.plot(s / L, np.degrees(h), label=name, color=PATH_COLORS[name], linewidth=1.5)

    ax2.set_xlabel("Normalized Arc Length")
    ax2.set_ylabel("Heading (deg)")
    ax2.set_title("Heading Profile")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Path Curvature and Heading Profiles", fontsize=12, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_curvature_profiles.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Curvature profiles saved")


# ============================================================================
# Figure 3: Multi-obstacle configurations
# ============================================================================
def fig_multi_obstacle():
    """Show 1, 2, 3-obstacle configurations on S-curve (G2/T2 setup)."""

    obs_fracs = {
        1: [0.50],
        2: [0.40, 0.60],
        3: [0.30, 0.45, 0.65],
    }

    x, y, s, h, k, L = make_s_curve(25.0, 3.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    obs_colors = ['#d62728', '#ff7f0e', '#9467bd']

    for ax_i, (n_obs, fracs) in enumerate(obs_fracs.items()):
        ax = axes[ax_i]
        ax.plot(x, y, color='#1f77b4', linewidth=2, zorder=1)

        # Ego start
        ax.plot(x[0], y[0], 's', color='green', markersize=10, zorder=5, label='Ego start')

        for oi, frac in enumerate(fracs):
            ox, oy, vx, vy, _ = obstacle_position_on_path(x, y, s, h, L, frac, oncoming=True)
            ax.plot(ox, oy, 'o', color=obs_colors[oi], markersize=10, zorder=5,
                    label=f'Obs {oi+1} ({frac*100:.0f}%)')
            ax.annotate('', xy=(ox + vx * 2, oy + vy * 2), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle='->', color=obs_colors[oi], lw=2))

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{n_obs} Obstacle{'s' if n_obs > 1 else ''}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Multi-Obstacle Configurations on S-curve (G2/T2)", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_multi_obstacle.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Multi-obstacle configs saved")


# ============================================================================
# Figure 4: Path x Obstacle grid (G5/T5 setup)
# ============================================================================
def fig_path_obstacle_grid():
    """Show all path x obstacle combinations from G5/T5."""

    path_defs = {
        "Straight": make_straight(),
        "S-curve":  make_s_curve(25.0, 3.0),
        "Tight-S":  make_s_curve(20.0, 5.0),
        "Circle":   make_circle(),
    }

    obs_fracs = {
        1: [0.50],
        2: [0.40, 0.60],
        3: [0.30, 0.45, 0.65],
    }
    obs_colors = ['#d62728', '#ff7f0e', '#9467bd']

    n_paths = len(path_defs)
    n_configs = len(obs_fracs)

    fig, axes = plt.subplots(n_configs, n_paths, figsize=(16, 10))

    for col_i, (pname, (x, y, s, h, k, L)) in enumerate(path_defs.items()):
        for row_i, (n_obs, fracs) in enumerate(obs_fracs.items()):
            ax = axes[row_i, col_i]
            ax.plot(x, y, color=PATH_COLORS[pname], linewidth=1.8, zorder=1)
            ax.plot(x[0], y[0], 's', color='green', markersize=7, zorder=5)

            for oi, frac in enumerate(fracs):
                ox, oy, vx, vy, _ = obstacle_position_on_path(x, y, s, h, L, frac)
                ax.plot(ox, oy, 'o', color=obs_colors[oi], markersize=7, zorder=5)
                ax.annotate('', xy=(ox + vx * 1.5, oy + vy * 1.5), xytext=(ox, oy),
                            arrowprops=dict(arrowstyle='->', color=obs_colors[oi], lw=1.2))

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            if row_i == 0:
                ax.set_title(pname, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"{n_obs} obs", fontsize=9)
            ax.tick_params(labelsize=6)

    fig.suptitle("G5/T5: Path x Obstacle Interaction Grid", fontsize=13, y=1.01)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_path_obstacle_grid.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Path x obstacle grid saved")


# ============================================================================
# Figure 5: Experiment parameter space summary
# ============================================================================
def fig_parameter_summary():
    """Summary table-like figure of all generalization experiment dimensions."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    experiments = [
        ["G1 / T1", "Path Geometry", "Straight, Gentle-S, S-curve, Tight-S, Circle",
         "1 obs @ 55%", "sp=0.2"],
        ["G2 / T2", "Multi-Obstacle", "S-curve",
         "1, 2, 3 obs", "sp=0.2"],
        ["G3 / T3", "Switching Dynamics", "S-curve",
         "1 obs @ 55%", "sp=0.02..0.50"],
        ["G4", "Mode Diversity", "S-curve",
         "1 obs @ 55%", "3, 4, 5, 6 modes"],
        ["G5 / T5", "Path x Obstacle", "Straight, S-curve, Tight-S, Circle",
         "1, 2, 3 obs", "sp=0.2"],
        ["T4", "Wasserstein Radius", "S-curve",
         "1 obs @ 55%", "rho=0.01..0.5"],
    ]

    headers = ["Experiment", "Dimension", "Paths", "Obstacles", "Key Param"]
    cell_colors = [["#f0f0f0"] * len(headers)] * len(experiments)

    table = ax.table(cellText=experiments, colLabels=headers,
                     loc='center', cellLoc='center',
                     colColours=["#4472C4"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Header colors
    for j in range(len(headers)):
        table[0, j].set_text_props(color='white', fontweight='bold')

    fig.suptitle("Generalization Experiment Design Summary", fontsize=12, y=0.92)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_experiment_summary.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("  Experiment summary saved")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("Generating path visualization figures...")
    fig_path_overview()
    fig_curvature_profiles()
    fig_multi_obstacle()
    fig_path_obstacle_grid()
    fig_parameter_summary()
    print("Done.")
