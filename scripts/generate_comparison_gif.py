#!/usr/bin/env python3
"""Generate frame-by-frame PNGs comparing Baseline SHMPCC (collision) vs Mode-DRO (safe).

Outputs numbered frames to generalization_figures/comparison_frames/ for use with
the LaTeX animate package:

    \\animategraphics[autoplay,loop,width=\\linewidth]{10}
        {generalization_figures/comparison_frames/frame_}{000}{052}

Uses the S-curve path geometry from G1 experiments where:
  - Baseline:  78.2% collision rate, mean clearance 0.63m
  - Mode-DRO:  12.8% collision rate, mean clearance 1.09m
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

# ============================================================================
# Path geometry (matching C++ and generate_path_visualizations.py)
# ============================================================================

def make_s_curve(length=25.0, amplitude=3.0, n=500):
    x = np.linspace(0, length, n)
    y = amplitude * np.sin(2 * np.pi * x / length)
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.zeros(n)
    s[1:] = np.cumsum(ds)
    total_length = s[-1]
    dy_dx = amplitude * 2 * np.pi / length * np.cos(2 * np.pi * x / length)
    heading = np.arctan2(dy_dx, np.ones_like(dy_dx))
    return x, y, s, heading, total_length


def interp_path(s_vals, target_s):
    """Interpolate x, y, heading at a given arc-length."""
    px, py, ps, ph, _ = PATH_DATA
    idx = np.searchsorted(ps, target_s, side='right') - 1
    idx = np.clip(idx, 0, len(ps) - 2)
    t = (target_s - ps[idx]) / max(ps[idx + 1] - ps[idx], 1e-9)
    t = np.clip(t, 0, 1)
    x = px[idx] + t * (px[idx + 1] - px[idx])
    y = py[idx] + t * (py[idx + 1] - py[idx])
    h = ph[idx] + t * (ph[idx + 1] - ph[idx])
    return x, y, h


# Global path data
PATH_DATA = make_s_curve(25.0, 3.0, 500)
TOTAL_LENGTH = PATH_DATA[4]

# ============================================================================
# Simulation parameters
# ============================================================================

DT = 0.1
N_STEPS = 180
EGO_SPEED = 1.5        # m/s along path
OBS_SPEED = 0.8        # m/s oncoming
EGO_RADIUS = 0.75      # vehicle half-length (single disc)
OBS_RADIUS = 0.50
COLLISION_DIST = EGO_RADIUS + OBS_RADIUS  # 1.25m

# Obstacle starts at 55% arc length, moving toward ego (oncoming)
OBS_START_FRAC = 0.55
# Mode switch: obstacle swerves laterally at step 55 (near encounter)
SWERVE_STEP = 52
SWERVE_DURATION = 25
SWERVE_AMPLITUDE = 0.06  # lateral velocity component (rad/s equivalent)

# ============================================================================
# Simulate obstacle trajectory (shared between both controllers)
# ============================================================================

def simulate_obstacle():
    """Obstacle moves oncoming along path, then swerves at encounter."""
    obs_s = OBS_START_FRAC * TOTAL_LENGTH
    positions = []

    # Get initial position and heading on path
    ox, oy, oh = interp_path(PATH_DATA[2], obs_s)
    # Oncoming velocity
    vx = -np.cos(oh) * OBS_SPEED
    vy = -np.sin(oh) * OBS_SPEED

    for step in range(N_STEPS):
        positions.append((ox, oy))

        # After swerve step, add lateral deviation (lane change mode)
        if SWERVE_STEP <= step < SWERVE_STEP + SWERVE_DURATION:
            # Swerve perpendicular to heading (to the left of obstacle's travel)
            perp_angle = oh + np.pi / 2
            swerve_vx = np.cos(perp_angle) * SWERVE_AMPLITUDE * OBS_SPEED * 3.0
            swerve_vy = np.sin(perp_angle) * SWERVE_AMPLITUDE * OBS_SPEED * 3.0
            ox += (vx + swerve_vx) * DT
            oy += (vy + swerve_vy) * DT
        else:
            ox += vx * DT
            oy += vy * DT

        # Update heading based on path (approximate)
        obs_s -= OBS_SPEED * DT
        if obs_s > 0:
            _, _, oh = interp_path(PATH_DATA[2], max(obs_s, 0))
            vx = -np.cos(oh) * OBS_SPEED
            vy = -np.sin(oh) * OBS_SPEED

    return positions


# ============================================================================
# Simulate ego trajectories
# ============================================================================

def simulate_ego_baseline(obs_positions):
    """Baseline SHMPCC: follows reference path closely, reacts late to swerve."""
    ego_s = 0.0
    positions = []
    collided = False
    collision_step = -1

    for step in range(N_STEPS):
        ex, ey, eh = interp_path(PATH_DATA[2], min(ego_s, TOTAL_LENGTH * 0.99))
        positions.append((ex, ey, eh, collided))

        # Check collision
        if not collided:
            ox, oy = obs_positions[step]
            dist = np.sqrt((ex - ox)**2 + (ey - oy)**2)
            if dist < COLLISION_DIST:
                collided = True
                collision_step = step

        # Baseline: stays very close to reference path
        # Only slight lateral deviation when obstacle is very close
        if not collided:
            if step > SWERVE_STEP + 5:
                # Late, weak reaction
                ox, oy = obs_positions[min(step, len(obs_positions) - 1)]
                dist = np.sqrt((ex - ox)**2 + (ey - oy)**2)
                if dist < 3.0:
                    # Small lateral nudge (too little, too late)
                    perp = eh + np.pi / 2
                    ex += np.cos(perp) * 0.02
                    ey += np.sin(perp) * 0.02
            ego_s += EGO_SPEED * DT
        else:
            # Stop after collision
            ego_s += EGO_SPEED * DT * 0.0

    return positions, collision_step


def simulate_ego_dro(obs_positions):
    """Mode-DRO: anticipates worst-case modes, steers wider early."""
    ego_s = 0.0
    positions = []
    lateral_offset = 0.0  # perpendicular offset from reference path
    collided = False

    # DRO anticipation window: starts reacting earlier
    ANTICIPATE_START = 35  # starts planning avoidance well before encounter
    ANTICIPATE_PEAK = 55
    RECOVER_END = 100

    for step in range(N_STEPS):
        ex, ey, eh = interp_path(PATH_DATA[2], min(ego_s, TOTAL_LENGTH * 0.99))

        # Apply lateral offset (DRO steers wider)
        perp = eh - np.pi / 2  # offset to the right of travel direction
        ex += np.cos(perp) * lateral_offset
        ey += np.sin(perp) * lateral_offset

        positions.append((ex, ey, eh, collided))

        # DRO avoidance: smooth lateral offset that peaks during encounter
        if ANTICIPATE_START <= step <= RECOVER_END:
            ox, oy = obs_positions[step]
            dist_along = ego_s - (OBS_START_FRAC * TOTAL_LENGTH - OBS_SPEED * step * DT)

            if step < ANTICIPATE_PEAK:
                # Ramp up: DRO injects worst-case modes → plans conservatively
                progress = (step - ANTICIPATE_START) / (ANTICIPATE_PEAK - ANTICIPATE_START)
                target_offset = 1.8 * smooth_step(progress)
            else:
                # Ramp down: return to reference path after safe passage
                progress = (step - ANTICIPATE_PEAK) / (RECOVER_END - ANTICIPATE_PEAK)
                target_offset = 1.8 * (1.0 - smooth_step(progress))

            # Smooth tracking toward target
            lateral_offset += (target_offset - lateral_offset) * 0.15
        else:
            lateral_offset *= 0.95  # decay

        # Check collision
        ox, oy = obs_positions[step]
        dist = np.sqrt((ex - ox)**2 + (ey - oy)**2)
        if dist < COLLISION_DIST:
            collided = True

        ego_s += EGO_SPEED * DT

    return positions, -1


def smooth_step(t):
    """Smooth step function (Hermite interpolation)."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# ============================================================================
# Rendering
# ============================================================================

def draw_vehicle(ax, x, y, heading, color, alpha=1.0, label=None):
    """Draw ego vehicle as a rounded rectangle."""
    w, h = 1.5, 0.8  # length, width
    corners = np.array([
        [-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2], [-w/2, -h/2]
    ])
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    rotated = (rot @ corners.T).T + np.array([x, y])
    ax.fill(rotated[:, 0], rotated[:, 1], color=color, alpha=alpha, zorder=10,
            label=label)
    ax.plot(rotated[:, 0], rotated[:, 1], color='black', linewidth=0.8, zorder=11)
    # Heading indicator
    front_x = x + np.cos(heading) * w / 2
    front_y = y + np.sin(heading) * w / 2
    ax.plot([x, front_x], [y, front_y], color='white', linewidth=1.5, zorder=12)


def draw_obstacle(ax, x, y, radius, color='#d62728', alpha=1.0):
    """Draw obstacle as a circle."""
    circle = plt.Circle((x, y), radius, color=color, alpha=alpha, zorder=9)
    ax.add_patch(circle)
    circle_border = plt.Circle((x, y), radius, fill=False, edgecolor='black',
                                linewidth=0.8, zorder=10)
    ax.add_patch(circle_border)


def draw_clearance_line(ax, ex, ey, ox, oy, dist, safe):
    """Draw dashed line showing clearance between ego and obstacle."""
    color = '#2ca02c' if safe else '#d62728'
    ax.plot([ex, ox], [ey, oy], '--', color=color, linewidth=1.0, alpha=0.7, zorder=8)
    mid_x, mid_y = (ex + ox) / 2, (ey + oy) / 2
    ax.annotate(f'{dist:.2f}m', xy=(mid_x, mid_y), fontsize=7,
                color=color, fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8),
                zorder=15)


def render_frame(step, obs_pos, ego_base, ego_dro, collision_step_base):
    """Render a single frame with side-by-side comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    px, py = PATH_DATA[0], PATH_DATA[1]

    for ax, ego_positions, title, is_dro in [
        (ax1, ego_base, 'Baseline SHMPCC', False),
        (ax2, ego_dro, 'Mode-DRO SHMPCC', True)
    ]:
        # Background
        ax.set_facecolor('#f8f9fa')

        # Reference path
        ax.plot(px, py, color='#adb5bd', linewidth=3.0, alpha=0.5, zorder=1,
                label='Reference path')
        ax.plot(px, py, '--', color='#6c757d', linewidth=0.8, alpha=0.6, zorder=1)

        # Trail (past positions)
        trail_start = max(0, step - 40)
        trail_x = [ego_positions[i][0] for i in range(trail_start, step + 1)]
        trail_y = [ego_positions[i][1] for i in range(trail_start, step + 1)]
        ego_color = '#2ca02c' if is_dro else '#1f77b4'
        ax.plot(trail_x, trail_y, color=ego_color, linewidth=2.0, alpha=0.4, zorder=5)

        # Obstacle trail
        obs_trail_x = [obs_pos[i][0] for i in range(trail_start, step + 1)]
        obs_trail_y = [obs_pos[i][1] for i in range(trail_start, step + 1)]
        ax.plot(obs_trail_x, obs_trail_y, color='#d62728', linewidth=1.5, alpha=0.3, zorder=4)

        # Current positions
        ex, ey, eh, collided = ego_positions[step]
        ox, oy = obs_pos[step]

        # Distance
        dist = np.sqrt((ex - ox)**2 + (ey - oy)**2)
        safe = dist >= COLLISION_DIST

        # Draw clearance line when close
        if dist < 5.0:
            draw_clearance_line(ax, ex, ey, ox, oy, dist, safe)

        # Draw collision radius (faint)
        if dist < 4.0:
            ego_zone = plt.Circle((ex, ey), EGO_RADIUS, fill=False,
                                   edgecolor=ego_color, linewidth=0.5,
                                   linestyle=':', alpha=0.4, zorder=7)
            obs_zone = plt.Circle((ox, oy), OBS_RADIUS, fill=False,
                                   edgecolor='#d62728', linewidth=0.5,
                                   linestyle=':', alpha=0.4, zorder=7)
            ax.add_patch(ego_zone)
            ax.add_patch(obs_zone)

        # Draw vehicles
        draw_obstacle(ax, ox, oy, OBS_RADIUS)
        if not is_dro and collision_step_base >= 0 and step >= collision_step_base:
            draw_vehicle(ax, ex, ey, eh, '#ff6b6b', alpha=0.9)
        else:
            draw_vehicle(ax, ex, ey, eh, ego_color, alpha=0.9)

        # Collision flash effect
        if not is_dro and collision_step_base >= 0 and step == collision_step_base:
            flash = plt.Circle((ex, ey), 2.0, color='red', alpha=0.3, zorder=6)
            ax.add_patch(flash)

        # Status panel
        panel_y = 0.97
        if not is_dro:
            if collision_step_base >= 0 and step >= collision_step_base:
                status_text = "COLLISION"
                status_color = '#d62728'
            else:
                status_text = "Tracking..."
                status_color = '#1f77b4'
        else:
            if step >= SWERVE_STEP - 17 and step < SWERVE_STEP + SWERVE_DURATION + 30:
                status_text = "DRO: Anticipating worst-case"
                status_color = '#ff8c00'
            else:
                status_text = "Safe"
                status_color = '#2ca02c'

        ax.text(0.5, panel_y, status_text, transform=ax.transAxes,
                fontsize=12, fontweight='bold', color='white', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=status_color, alpha=0.85),
                zorder=20)

        # Title
        method_label = title
        if not is_dro:
            method_label += f'  (78.2% collision rate)'
        else:
            method_label += f'  (12.8% collision rate)'
        ax.set_title(method_label, fontsize=11, fontweight='bold', pad=8)

        # Axis setup
        # Follow the action — center view on midpoint of ego and obstacle
        cx = (ex + ox) / 2
        cy = (ey + oy) / 2
        view_w = 10
        view_h = 7
        ax.set_xlim(cx - view_w, cx + view_w)
        ax.set_ylim(cy - view_h, cy + view_h)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('x (m)', fontsize=9)
        ax.set_ylabel('y (m)', fontsize=9)

        # Time label
        ax.text(0.02, 0.02, f't = {step * DT:.1f}s', transform=ax.transAxes,
                fontsize=9, color='#495057', family='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Swerve indicator
        if SWERVE_STEP <= step < SWERVE_STEP + SWERVE_DURATION:
            ax.text(0.98, 0.02, 'Obstacle: SWERVE', transform=ax.transAxes,
                    fontsize=8, color='#d62728', fontweight='bold', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff3f3', alpha=0.9))

    # Legend (shared)
    legend_elements = [
        mpatches.Patch(facecolor='#1f77b4', edgecolor='black', label='Baseline ego'),
        mpatches.Patch(facecolor='#2ca02c', edgecolor='black', label='Mode-DRO ego'),
        mpatches.Patch(facecolor='#d62728', edgecolor='black', label='Obstacle'),
        plt.Line2D([0], [0], color='#adb5bd', linewidth=3, alpha=0.5, label='Reference path'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Baseline SHMPCC vs Mode-DRO: Collision Avoidance Comparison',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    print("Simulating trajectories...")
    obs_positions = simulate_obstacle()
    ego_base_positions, collision_step = simulate_ego_baseline(obs_positions)
    ego_dro_positions, _ = simulate_ego_dro(obs_positions)

    print(f"  Baseline collision at step {collision_step} (t={collision_step * DT:.1f}s)")

    # Output directory for individual frames
    frames_dir = os.path.join(os.path.dirname(__file__), '..',
                              'generalization_figures', 'comparison_frames')
    frames_dir = os.path.abspath(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Focus on the interesting part: steps 15-120 (approach + encounter + aftermath)
    frame_start = 15
    frame_end = min(120, N_STEPS)
    frame_skip = 2  # every other step for smooth animation

    frame_indices = list(range(frame_start, frame_end, frame_skip))
    n_frames = len(frame_indices)

    print(f"Rendering {n_frames} frames to {frames_dir}/")
    for i, step in enumerate(frame_indices):
        if i % 10 == 0:
            print(f"  Frame {i}/{n_frames} (step {step}, t={step * DT:.1f}s)")
        fig = render_frame(step, obs_positions, ego_base_positions,
                           ego_dro_positions, collision_step)
        fig.savefig(os.path.join(frames_dir, f'frame_{i:03d}.png'),
                    dpi=150, facecolor='white', edgecolor='none')
        plt.close(fig)

    last_idx = n_frames - 1
    print(f"\nDone! {n_frames} frames saved to {frames_dir}/")
    print(f"  frame_000.png .. frame_{last_idx:03d}.png")
    print(f"  Frame rate: 10 fps (100ms per frame)")
    print(f"\nLaTeX usage (animate package):")
    print(f"  \\animategraphics[autoplay,loop,width=\\linewidth]{{10}}")
    print(f"      {{generalization_figures/comparison_frames/frame_}}{{000}}{{{last_idx:03d}}}")


if __name__ == '__main__':
    main()
