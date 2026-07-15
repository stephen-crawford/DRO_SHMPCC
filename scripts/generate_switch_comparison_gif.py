#!/usr/bin/env python3
"""GIF: Baseline SHMPCC avoids a non-switching obstacle but collides when it switches.

Left panel:  obstacle holds constant_velocity → baseline avoids safely
Right panel: obstacle switches to turn_left mid-encounter → baseline collides

S-curve path, single oncoming obstacle at 55% arc length.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import io, os

# ── Path geometry (S-curve, matching C++) ──────────────────────────────────

def make_s_curve(length=25.0, amplitude=3.0, n=500):
    x = np.linspace(0, length, n)
    y = amplitude * np.sin(2 * np.pi * x / length)
    dx, dy = np.diff(x), np.diff(y)
    s = np.zeros(n); s[1:] = np.cumsum(np.sqrt(dx**2 + dy**2))
    dy_dx = amplitude * 2 * np.pi / length * np.cos(2 * np.pi * x / length)
    heading = np.arctan2(dy_dx, np.ones_like(dy_dx))
    return x, y, s, heading, s[-1]

PX, PY, PS, PH, TOTAL_L = make_s_curve()

def interp(target_s):
    idx = np.clip(np.searchsorted(PS, target_s, side='right') - 1, 0, len(PS) - 2)
    t = np.clip((target_s - PS[idx]) / max(PS[idx+1] - PS[idx], 1e-9), 0, 1)
    return (PX[idx] + t*(PX[idx+1]-PX[idx]),
            PY[idx] + t*(PY[idx+1]-PY[idx]),
            PH[idx] + t*(PH[idx+1]-PH[idx]))

# ── Sim constants ──────────────────────────────────────────────────────────

DT          = 0.1
N_STEPS     = 160
EGO_SPEED   = 1.5
OBS_SPEED   = 0.8
EGO_R       = 0.75
OBS_R       = 0.50
COLL_DIST   = EGO_R + OBS_R
OBS_FRAC    = 0.55
SWITCH_STEP = 48          # when the mode switch happens (right panel only)

# ── Obstacle simulation ───────────────────────────────────────────────────

def sim_obstacle(do_switch):
    """Oncoming obstacle. If do_switch, it swerves into the ego's escape path.

    The ego always dodges to the RIGHT of its travel direction (south-ish on
    the S-curve at the encounter zone). When the obstacle switches, it veers
    to its LEFT — which is the ego's RIGHT — cutting into the escape corridor.
    """
    obs_s = OBS_FRAC * TOTAL_L
    ox, oy, oh = interp(obs_s)
    vx, vy = -np.cos(oh) * OBS_SPEED, -np.sin(oh) * OBS_SPEED
    positions, modes = [], []

    for step in range(N_STEPS):
        positions.append((ox, oy))

        if do_switch and step >= SWITCH_STEP:
            modes.append("lane_change")
            # Swerve to obstacle's LEFT = ego's RIGHT (south-ish on S-curve)
            travel_angle = np.arctan2(vy, vx)
            perp = travel_angle + np.pi / 2   # LEFT of obstacle travel = ego's RIGHT
            swerve_strength = 0.55
            ox += (vx + np.cos(perp) * swerve_strength) * DT
            oy += (vy + np.sin(perp) * swerve_strength) * DT
        else:
            modes.append("constant_velocity")
            ox += vx * DT
            oy += vy * DT

        obs_s -= OBS_SPEED * DT
        if obs_s > 0 and not (do_switch and step >= SWITCH_STEP):
            _, _, oh = interp(max(obs_s, 0))
            vx, vy = -np.cos(oh) * OBS_SPEED, -np.sin(oh) * OBS_SPEED

    return positions, modes

# ── Ego simulation (baseline SHMPCC) ──────────────────────────────────────

def sim_ego(obs_positions):
    """Baseline SHMPCC: dodges RIGHT of travel when obstacle is detected.

    Once the obstacle enters detection range, the ego commits to a full
    lateral dodge to the right of the path. The offset builds smoothly.
    This is sufficient when the obstacle is predictable, but when the
    obstacle swerves into the escape corridor the committed plan fails.
    """
    ego_s = 0.0
    positions = []
    collided = False
    collision_step = -1
    lateral = 0.0
    DODGE_OFFSET = 2.0      # target lateral offset (> COLL_DIST = 1.25m)
    DETECT_RANGE = 10.0
    TRACKING_RATE = 0.08    # smooth build-up per step

    for step in range(N_STEPS):
        ex, ey, eh = interp(min(ego_s, TOTAL_L * 0.99))
        perp_x = np.cos(eh - np.pi / 2)
        perp_y = np.sin(eh - np.pi / 2)
        ex += perp_x * lateral
        ey += perp_y * lateral

        positions.append((ex, ey, eh, collided))

        ox, oy = obs_positions[step]
        dist = np.sqrt((ex - ox)**2 + (ey - oy)**2)

        if not collided and dist < COLL_DIST:
            collided = True
            collision_step = step

        if not collided:
            if dist < DETECT_RANGE:
                # Commit: full dodge to the right
                lateral += (DODGE_OFFSET - lateral) * TRACKING_RATE
            else:
                lateral *= 0.93

            ego_s += EGO_SPEED * DT

    return positions, collision_step


def smooth_step(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

# ── Drawing helpers ────────────────────────────────────────────────────────

def draw_vehicle(ax, x, y, h, color, alpha=1.0):
    w, ht = 1.5, 0.8
    c = np.array([[-w/2,-ht/2],[w/2,-ht/2],[w/2,ht/2],[-w/2,ht/2],[-w/2,-ht/2]])
    R = np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
    pts = (R @ c.T).T + [x, y]
    ax.fill(pts[:,0], pts[:,1], color=color, alpha=alpha, zorder=10)
    ax.plot(pts[:,0], pts[:,1], 'k-', lw=0.8, zorder=11)
    fx, fy = x + np.cos(h)*w/2, y + np.sin(h)*w/2
    ax.plot([x,fx],[y,fy], 'w-', lw=1.5, zorder=12)

def draw_obs(ax, x, y, r, color='#d62728'):
    ax.add_patch(plt.Circle((x,y), r, color=color, zorder=9))
    ax.add_patch(plt.Circle((x,y), r, fill=False, ec='k', lw=0.8, zorder=10))

def clearance_line(ax, ex, ey, ox, oy, d, safe):
    c = '#2ca02c' if safe else '#d62728'
    ax.plot([ex,ox],[ey,oy], '--', color=c, lw=1.0, alpha=0.7, zorder=8)
    mx, my = (ex+ox)/2, (ey+oy)/2
    ax.annotate(f'{d:.2f}m', (mx,my), fontsize=7, color=c, fontweight='bold',
                ha='center', va='bottom', zorder=15,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8))

# ── Render one frame ──────────────────────────────────────────────────────

def render_frame(step, data_no_sw, data_sw):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    panels = [
        (ax1, *data_no_sw, 'No Mode Switch', False),
        (ax2, *data_sw,    'Mode Switch',    True),
    ]

    for ax, obs_pos, obs_modes, ego_pos, coll_step, title, is_switch in panels:
        ax.set_facecolor('#f8f9fa')
        ax.plot(PX, PY, color='#adb5bd', lw=3, alpha=0.5, zorder=1)
        ax.plot(PX, PY, '--', color='#6c757d', lw=0.8, alpha=0.6, zorder=1)

        # Trails
        t0 = max(0, step - 40)
        ax.plot([ego_pos[i][0] for i in range(t0, step+1)],
                [ego_pos[i][1] for i in range(t0, step+1)],
                color='#1f77b4', lw=2, alpha=0.4, zorder=5)
        ax.plot([obs_pos[i][0] for i in range(t0, step+1)],
                [obs_pos[i][1] for i in range(t0, step+1)],
                color='#d62728', lw=1.5, alpha=0.3, zorder=4)

        ex, ey, eh, collided = ego_pos[step]
        ox, oy = obs_pos[step]
        dist = np.sqrt((ex-ox)**2 + (ey-oy)**2)

        if dist < 5.0:
            clearance_line(ax, ex, ey, ox, oy, dist, dist >= COLL_DIST)
        if dist < 4.0:
            ax.add_patch(plt.Circle((ex,ey), EGO_R, fill=False, ec='#1f77b4',
                                     lw=0.5, ls=':', alpha=0.4, zorder=7))
            ax.add_patch(plt.Circle((ox,oy), OBS_R, fill=False, ec='#d62728',
                                     lw=0.5, ls=':', alpha=0.4, zorder=7))

        draw_obs(ax, ox, oy, OBS_R)
        veh_color = '#ff6b6b' if (coll_step >= 0 and step >= coll_step) else '#1f77b4'
        draw_vehicle(ax, ex, ey, eh, veh_color, 0.9)

        # Collision flash
        if coll_step >= 0 and step == coll_step:
            ax.add_patch(plt.Circle((ex,ey), 2.0, color='red', alpha=0.3, zorder=6))

        # Status badge
        if coll_step >= 0 and step >= coll_step:
            stxt, scol = "COLLISION", '#d62728'
        elif dist < 4.5 and not collided:
            stxt, scol = "Avoiding...", '#ff8c00'
        else:
            stxt, scol = "Safe", '#2ca02c'
        ax.text(0.5, 0.97, stxt, transform=ax.transAxes, fontsize=12,
                fontweight='bold', color='white', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.4', fc=scol, alpha=0.85), zorder=20)

        # Mode badge
        mode = obs_modes[step]
        mode_color = '#6c757d' if mode == 'constant_velocity' else '#9467bd'
        mode_label = 'CV' if mode == 'constant_velocity' else 'TURN LEFT'
        ax.text(0.98, 0.02, f'Obs mode: {mode_label}', transform=ax.transAxes,
                fontsize=8, color=mode_color, fontweight='bold', ha='right',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.9), zorder=20)

        ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
        cx, cy = (ex+ox)/2, (ey+oy)/2
        ax.set_xlim(cx-10, cx+10); ax.set_ylim(cy-7, cy+7)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.set_xlabel('x (m)', fontsize=9); ax.set_ylabel('y (m)', fontsize=9)
        ax.text(0.02, 0.02, f't = {step*DT:.1f}s', transform=ax.transAxes,
                fontsize=9, color='#495057', family='monospace',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    legend_el = [
        mpatches.Patch(fc='#1f77b4', ec='k', label='Baseline SHMPCC ego'),
        mpatches.Patch(fc='#d62728', ec='k', label='Obstacle'),
        plt.Line2D([0],[0], color='#adb5bd', lw=3, alpha=0.5, label='Reference path'),
    ]
    fig.legend(handles=legend_el, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Baseline SHMPCC: Safe Without Switching, Collision With Switching',
                 fontsize=13, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy().convert('RGB')

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Simulating trajectories...")
    obs_no, modes_no = sim_obstacle(do_switch=False)
    obs_sw, modes_sw = sim_obstacle(do_switch=True)
    ego_no, coll_no  = sim_ego(obs_no)
    ego_sw, coll_sw  = sim_ego(obs_sw)

    print(f"  No-switch collision step: {coll_no}")
    print(f"  Switch collision step:    {coll_sw} (t={coll_sw*DT:.1f}s)")

    data_no = (obs_no, modes_no, ego_no, coll_no)
    data_sw = (obs_sw, modes_sw, ego_sw, coll_sw)

    start, end, skip = 10, min(110, N_STEPS), 2
    indices = list(range(start, end, skip))
    frames = []

    print(f"Rendering {len(indices)} frames...")
    for i, step in enumerate(indices):
        if i % 10 == 0:
            print(f"  {i}/{len(indices)} (step {step})")
        frames.append(render_frame(step, data_no, data_sw))

    # Hold final frame
    for _ in range(10):
        frames.append(frames[-1])

    out = os.path.join(os.path.dirname(__file__), '..',
                       'generalization_figures', 'switch_collision.gif')
    out = os.path.abspath(out)
    print(f"Saving {out}...")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)
    sz = os.path.getsize(out) / (1024*1024)
    print(f"Done! {sz:.1f} MB, {len(frames)} frames")

if __name__ == '__main__':
    main()
