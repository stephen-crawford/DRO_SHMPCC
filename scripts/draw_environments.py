#!/usr/bin/env python3
"""Draw diagrams of the Oncoming and Intersection experiment setups."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# S-curve parameters (matching C++: L=25, A=3)
L = 25.0
A = 3.0
N = 200

x = np.linspace(0, L, N)
y = A * np.sin(2 * np.pi * x / L)

# Compute arc length
dx = np.diff(x)
dy = np.diff(y)
ds = np.sqrt(dx**2 + dy**2)
s = np.concatenate([[0], np.cumsum(ds)])
total_s = s[-1]

# Headings
headings = np.arctan2(np.gradient(y, x), np.ones_like(x))

def point_at_frac(frac):
    idx = np.argmin(np.abs(s - frac * total_s))
    return x[idx], y[idx], headings[idx]

def normal_at(heading):
    return np.array([-np.sin(heading), np.cos(heading)])

def tangent_at(heading):
    return np.array([np.cos(heading), np.sin(heading)])


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (env_name, env_frac, setup_fn) in enumerate([
    ("Oncoming", 0.60, "oncoming"),
    ("Intersection", 0.40, "intersection"),
]):
    ax = axes[ax_idx]

    # Draw path
    ax.plot(x, y, 'k-', linewidth=2, label='Ego reference path', zorder=2)

    # Draw road boundaries (width ~7m, so +/-3.5m from center)
    road_half = 3.5
    nx = -np.sin(headings)
    ny = np.cos(headings)
    ax.plot(x + road_half * nx, y + road_half * ny, 'k--', linewidth=0.5, alpha=0.3)
    ax.plot(x - road_half * nx, y - road_half * ny, 'k--', linewidth=0.5, alpha=0.3)

    # Ego start
    ax.plot(0, 0, 'bs', markersize=10, zorder=5, label='Ego start')
    ax.annotate('Ego start\n(0, 0)', (0, 0), textcoords="offset points",
                xytext=(10, -20), fontsize=8, color='blue')

    # Draw ego direction arrow
    ax.annotate('', xy=(1.5, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    # Get obstacle position
    ox, oy, oh = point_at_frac(env_frac)
    n = normal_at(oh)
    t = tangent_at(oh)

    if setup_fn == "oncoming":
        # Obstacle at 60% arc, on the path, velocity ~opposite to path tangent
        obs_pos = np.array([ox, oy])
        obs_vel = -1.0 * t  # heading back toward ego

        ax.plot(obs_pos[0], obs_pos[1], 'r^', markersize=12, zorder=5, label='Obstacle')
        ax.annotate(f'Obstacle\nat 60% arc',
                    obs_pos, textcoords="offset points",
                    xytext=(12, 10), fontsize=8, color='red')

        # Velocity arrow
        ax.annotate('', xy=(obs_pos[0] + 1.5 * obs_vel[0], obs_pos[1] + 1.5 * obs_vel[1]),
                    xytext=(obs_pos[0], obs_pos[1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.annotate('v = -1.0 t\n(head-on)',
                    (obs_pos[0] + 1.5 * obs_vel[0], obs_pos[1] + 1.5 * obs_vel[1]),
                    textcoords="offset points", xytext=(-15, -18), fontsize=7, color='red')

        # Mark the 60% point on path
        ax.plot(ox, oy, 'rx', markersize=8, zorder=4)

        # Show mode switching possibilities
        mode_offsets = [
            ("const vel", obs_vel * 2.5),
            ("turn L", obs_vel * 2.0 + n * 1.5),
            ("turn R", obs_vel * 2.0 - n * 1.5),
            ("decel", obs_vel * 1.0),
        ]
        for mode_name, offset in mode_offsets:
            end = obs_pos + offset
            ax.annotate('', xy=(end[0], end[1]), xytext=(obs_pos[0], obs_pos[1]),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=0.8,
                                       linestyle='--', alpha=0.6))
            ax.text(end[0], end[1], mode_name, fontsize=6, color='orange', alpha=0.8,
                    ha='center', va='bottom')

        ax.set_title("Oncoming Environment", fontsize=12, fontweight='bold')
        description = (
            "Obstacle placed at 60% of S-curve arc length,\n"
            "moving head-on toward ego at ~1 m/s.\n"
            "Highest collision risk (67% base rate).\n"
            "4 switching modes + rare lane-change."
        )

    elif setup_fn == "intersection":
        # Obstacle at 40% arc, offset 2m in normal direction, velocity toward path
        obs_pos = np.array([ox, oy]) + 2.0 * n
        obs_vel = -1.0 * n  # crossing perpendicular to path

        ax.plot(obs_pos[0], obs_pos[1], 'r^', markersize=12, zorder=5, label='Obstacle')
        ax.annotate(f'Obstacle\n2m off path\nat 40% arc',
                    obs_pos, textcoords="offset points",
                    xytext=(12, 10), fontsize=8, color='red')

        # Velocity arrow
        ax.annotate('', xy=(obs_pos[0] + 1.5 * obs_vel[0], obs_pos[1] + 1.5 * obs_vel[1]),
                    xytext=(obs_pos[0], obs_pos[1]),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.annotate('v = -1.0 n\n(crossing)',
                    (obs_pos[0] + 1.5 * obs_vel[0], obs_pos[1] + 1.5 * obs_vel[1]),
                    textcoords="offset points", xytext=(-15, -18), fontsize=7, color='red')

        # Show where it would cross the path
        ax.plot(ox, oy, 'rx', markersize=8, zorder=4)
        ax.annotate('crossing\npoint', (ox, oy), textcoords="offset points",
                    xytext=(-30, -15), fontsize=7, color='darkred')

        # Show mode switching possibilities
        mode_offsets = [
            ("const vel", obs_vel * 2.5),
            ("turn L", obs_vel * 2.0 + t * 1.2),
            ("turn R", obs_vel * 2.0 - t * 1.2),
            ("decel", obs_vel * 1.0),
        ]
        for mode_name, offset in mode_offsets:
            end = obs_pos + offset
            ax.annotate('', xy=(end[0], end[1]), xytext=(obs_pos[0], obs_pos[1]),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=0.8,
                                       linestyle='--', alpha=0.6))
            ax.text(end[0], end[1], mode_name, fontsize=6, color='orange', alpha=0.8,
                    ha='center', va='bottom')

        ax.set_title("Intersection Environment", fontsize=12, fontweight='bold')
        description = (
            "Obstacle placed 2m off-path at 40% arc,\n"
            "moving perpendicular (crossing) at ~1 m/s.\n"
            "Moderate collision risk (4-7% base rate).\n"
            "4 switching modes + rare lane-change."
        )

    # Add description text box
    ax.text(0.02, 0.02, description, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.2)

fig.suptitle("Experiment Environment Setups (S-curve path, L=25m, A=3m)",
             fontsize=13, fontweight='bold', y=1.02)
fig.tight_layout()

for ext in ["pdf", "png"]:
    fig.savefig(f"cdc_paper_figures/environment_diagrams.{ext}",
                bbox_inches='tight', dpi=200)
print("Saved cdc_paper_figures/environment_diagrams.{pdf,png}")
