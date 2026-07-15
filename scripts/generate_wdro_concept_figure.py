#!/usr/bin/env python3
"""
Generate a 3D simplex figure showing WDRO reweighting.
Tight framing, centered simplex, clean non-overlapping annotations.
"""

import pathlib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
import matplotlib.colors as mcolors

OUT = pathlib.Path("figures/robustness")
OUT.mkdir(parents=True, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────
MODES = ["Constant\nVelocity", "Turn\nLeft", "Decelerate"]
# q* must lie inside B_rho(p_hat), so place them close enough
# that the ball visually contains q*.
p_hat = np.array([0.50, 0.30, 0.20])
q_star = np.array([0.35, 0.45, 0.20])  # shifted toward Turn Left (high risk)
risk = np.array([0.08, 0.85, 0.05])

# Simplex vertices — equilateral triangle
V = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.866, 0.0],
])

def bary_to_cart(w):
    return w @ V

def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  saved {stem}")


def make_figure():
    fig = plt.figure(figsize=(13, 8), facecolor="white")
    # Simplex on the left ~72%, legend/key on the right ~28%
    ax = fig.add_axes([-0.02, -0.04, 0.76, 1.0], projection="3d", computed_zorder=False)
    ax.set_facecolor("white")
    ax.view_init(elev=30, azim=-55)

    # ── Risk-shaded simplex face ─────────────────────────────────────
    N = 45
    triangles, colors = [], []
    for i in range(N):
        for j in range(N - i):
            k = N - i - j
            w0 = np.array([i, j, k]) / N
            w1 = np.array([i+1, j, k-1]) / N
            w2 = np.array([i, j+1, k-1]) / N
            if k >= 1:
                triangles.append([bary_to_cart(w0), bary_to_cart(w1), bary_to_cart(w2)])
                colors.append((w0 + w1 + w2) / 3 @ risk)
            if k >= 2:
                w3 = np.array([i+1, j+1, k-2]) / N
                triangles.append([bary_to_cart(w1), bary_to_cart(w3), bary_to_cart(w2)])
                colors.append((w1 + w3 + w2) / 3 @ risk)

    colors = np.array(colors)
    cmap = matplotlib.colormaps["YlOrRd"]
    norm = mcolors.Normalize(vmin=colors.min() * 0.7, vmax=colors.max() * 1.15)
    fc = [(*cmap(norm(c))[:3], 0.65) for c in colors]
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=fc, edgecolors="none", zorder=1))

    # ── Simplex wireframe edges ──────────────────────────────────────
    for i in range(3):
        j = (i + 1) % 3
        ax.plot(*zip(V[i], V[j]), color="#cc3333", linewidth=2.5, zorder=3, alpha=0.9)

    # ── Vertex markers ───────────────────────────────────────────────
    rcmap = matplotlib.colormaps["YlOrRd"]
    for i in range(3):
        rn = (risk[i] - risk.min()) / (risk.max() - risk.min() + 1e-9)
        ax.scatter(*V[i], s=140, color=rcmap(0.25 + 0.65 * rn),
                   edgecolors="black", linewidth=1.5, zorder=10, depthshade=False)

    # ── Points ───────────────────────────────────────────────────────
    p_pt = bary_to_cart(p_hat)
    q_pt = bary_to_cart(q_star)
    ax.scatter(*p_pt, s=200, color="#4e79a7", edgecolors="white", linewidth=2.5,
               marker="o", zorder=12, depthshade=False)
    ax.scatter(*q_pt, s=200, color="#e15759", edgecolors="white", linewidth=2.5,
               marker="D", zorder=12, depthshade=False)

    # ── Wasserstein ball (clipped to simplex) ─────────────────────────
    # The ball must lie entirely within the probability simplex.
    # We draw a dome whose equator ring is clipped to barycentric >= 0.
    ball_r_xy = 0.19
    ball_r_z = 0.065

    # Helper: Cartesian -> barycentric (inverse of bary_to_cart for 2D simplex)
    def cart_to_bary(pt):
        # Solve w @ V = pt for w with w summing to 1 (z=0 plane)
        # V[0]=(0,0), V[1]=(1,0), V[2]=(0.5, 0.866)
        x, y = pt[0], pt[1]
        w2 = y / 0.866
        w1 = x - 0.5 * w2
        w0 = 1.0 - w1 - w2
        return np.array([w0, w1, w2])

    def inside_simplex(pt):
        b = cart_to_bary(pt)
        return np.all(b >= -0.005)  # tiny tolerance

    # In-plane basis vectors on the simplex face
    t1 = V[1] - V[0]; t1 /= np.linalg.norm(t1)
    t2r = V[2] - V[0]; t2r -= np.dot(t2r, t1) * t1; t2 = t2r / np.linalg.norm(t2r)

    # Build dome surface, masking points outside simplex
    n_u, n_v = 60, 15
    u_angles = np.linspace(0, 2 * np.pi, n_u)
    v_angles = np.linspace(0, np.pi / 2, n_v)
    sx = np.full((n_u, n_v), np.nan)
    sy = np.full((n_u, n_v), np.nan)
    sz = np.full((n_u, n_v), np.nan)
    for ui in range(n_u):
        for vi in range(n_v):
            pt_base = p_pt[:2] + ball_r_xy * np.sin(v_angles[vi]) * (
                np.cos(u_angles[ui]) * t1[:2] + np.sin(u_angles[ui]) * t2[:2])
            pt3 = np.array([pt_base[0], pt_base[1], 0.0])
            if inside_simplex(pt3):
                sx[ui, vi] = pt_base[0]
                sy[ui, vi] = pt_base[1]
                sz[ui, vi] = ball_r_z * np.cos(v_angles[vi])
    ax.plot_surface(sx, sy, sz, color="#4e79a7", alpha=0.14,
                    edgecolor="#4e79a7", linewidth=0.1, zorder=5)

    # Equator ring clipped to simplex
    ring_pts = []
    for t in np.linspace(0, 2 * np.pi, 200):
        pt = p_pt + ball_r_xy * (np.cos(t) * t1 + np.sin(t) * t2)
        if inside_simplex(pt):
            ring_pts.append(pt)
    if ring_pts:
        ring = np.array(ring_pts)
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2],
                color="#4e79a7", linewidth=1.8, linestyle="--", alpha=0.55, zorder=6)

    # ── Transport arrow ──────────────────────────────────────────────
    normal = np.array([0, 0, 1.0])
    ts = np.linspace(0, 1, 40)
    arc = np.array([(1-t)*p_pt + t*q_pt + 0.025*np.sin(np.pi*t)*normal for t in ts])
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="#333333", linewidth=2.5, alpha=0.85, zorder=11)
    # arrowhead
    tip, pre = arc[-1], arc[-3]
    d = tip - pre; d /= np.linalg.norm(d)
    perp = np.cross(d, normal); perp /= np.linalg.norm(perp)
    head = np.array([tip, tip - 0.025*d + 0.012*perp, tip - 0.025*d - 0.012*perp])
    ax.add_collection3d(Poly3DCollection([head], facecolors="#333333", edgecolors="#333333", zorder=11))

    # ── Tight axis limits (center the simplex, fill the space) ───────
    cx, cy = 0.5, 0.29  # simplex centroid approx
    span = 0.50
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_zlim(-0.04, 0.15)
    ax.set_axis_off()

    # ── Force draw for projection ────────────────────────────────────
    fig.canvas.draw()

    # ── Annotation helper ────────────────────────────────────────────
    def ann(text, pt3d, dx, dy, color="#333333", fs=10, fw="bold", ha="center"):
        x2, y2, _ = proj3d.proj_transform(*pt3d, ax.get_proj())
        ax.annotate(
            text, xy=(x2, y2), xycoords="data",
            xytext=(dx, dy), textcoords="offset points",
            fontsize=fs, color=color, fontweight=fw, ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.6,
                            shrinkA=0, shrinkB=3),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, alpha=0.9, linewidth=0.7),
            zorder=20)

    # Vertices — pushed well outside
    ann(f"{MODES[0]}\nrisk = {risk[0]:.0%}", V[0],
        -55, 40, color="#666666", fs=9)
    ann(f"{MODES[1]}\nrisk = {risk[1]:.0%}", V[1],
        75, -30, color="#aa2222", fs=9)
    ann(f"{MODES[2]}\nrisk = {risk[2]:.0%}", V[2],
        75, 25, color="#666666", fs=9)

    # p̂ — upper right
    ann(r"$\hat{p}$" + f"  ({p_hat[0]:.0%}, {p_hat[1]:.0%}, {p_hat[2]:.0%})",
        p_pt, 90, 55, color="#4e79a7", fs=10.5)

    # q* — left
    ann(r"$q^*$" + f"  ({q_star[0]:.0%}, {q_star[1]:.0%}, {q_star[2]:.0%})",
        q_pt, -110, -40, color="#e15759", fs=10.5)

    # Ball label — above
    ball_top = p_pt + np.array([0, 0, ball_r_z])
    ann(r"$\mathcal{B}_\rho(\hat{p})$", ball_top,
        55, 40, color="#4e79a7", fs=11)

    # Arrow label
    arc_mid = arc[16] + np.array([0, 0, 0.01])
    ann(r"$\hat{p} \to q^*$", arc_mid,
        0, -40, color="#333333", fs=9, fw="normal")

    # ── Right-side legend / key panel ────────────────────────────────
    ax_key = fig.add_axes([0.64, 0.05, 0.35, 0.88])
    ax_key.set_xlim(0, 1)
    ax_key.set_ylim(0, 1)
    ax_key.axis("off")

    y = 0.95
    gap = 0.048

    def key_heading(text, yy):
        ax_key.text(0.05, yy, text, fontsize=11, fontweight="bold",
                    color="#222222", va="top")
        return yy - gap * 1.2

    def key_entry(symbol, desc, yy, sym_color="#222222"):
        ax_key.text(0.05, yy, symbol, fontsize=11, color=sym_color,
                    fontweight="bold", va="top", fontfamily="serif")
        ax_key.text(0.30, yy, desc, fontsize=9.5, color="#444444",
                    va="top", linespacing=1.4)
        return yy - gap * 1.5

    def key_line(yy):
        ax_key.plot([0.03, 0.95], [yy + 0.01, yy + 0.01],
                    color="#cccccc", linewidth=0.8)
        return yy - gap * 0.3

    # Objective
    y = key_heading("Objective", y)
    ax_key.text(0.05, y, r"$\sup_{q \in \mathcal{B}_\rho(\hat{p})} \sum_m q_m \, r_m$",
                fontsize=13, color="#222222", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5ff",
                          edgecolor="#ccccdd", linewidth=0.8))
    y -= gap * 2.5

    y = key_line(y)

    # Symbol definitions
    y = key_heading("Symbols", y)

    y = key_entry(r"$q_m$",
                  "Weight on mode $m$ in\ncandidate distribution $q$",
                  y)
    y = key_entry(r"$r_m$",
                  "Collision risk of mode $m$\n(color intensity on simplex)",
                  y, sym_color="#cc6600")
    y = key_entry(r"$\hat{p}$",
                  "Nominal distribution\n(observed mode frequencies)",
                  y, sym_color="#4e79a7")
    y = key_entry(r"$q^*$",
                  "Worst-case distribution\n(maximizes expected risk)",
                  y, sym_color="#e15759")
    y = key_entry(r"$\mathcal{B}_\rho(\hat{p})$",
                  "Wasserstein ball of radius $\\rho$\naround $\\hat{p}$ (ambiguity set)",
                  y, sym_color="#4e79a7")
    y = key_entry(r"$\rho$",
                  "Wasserstein radius\n(controls robustness level)",
                  y)

    y = key_line(y)

    # Why a simplex?
    y = key_heading("Why a simplex?", y)
    ax_key.text(0.05, y,
                "Each point on the simplex is a\n"
                "probability distribution over modes\n"
                "($q_m \\geq 0$, $\\sum_m q_m = 1$).\n\n"
                "DRO finds the distribution $q^*$\n"
                "within the Wasserstein ball that\n"
                "maximizes expected collision risk\n"
                "$\\sum_m q_m r_m$, shifting mass\n"
                "toward dangerous modes.",
                fontsize=9, color="#555555", va="top", linespacing=1.45)

    # Title
    fig.suptitle("Wasserstein DRO Reweighting on the Probability Simplex",
                 fontsize=14, fontweight="bold", y=0.97)

    save(fig, "fig_wdro_simplex")


if __name__ == "__main__":
    print("Generating WDRO 3D simplex figure...")
    make_figure()
    print(f"Done. Figure in {OUT}/")
