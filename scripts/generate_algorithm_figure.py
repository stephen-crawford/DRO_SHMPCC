#!/usr/bin/env python3
"""Generate a compact horizontal algorithm overview figure."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

DIR = "cdc_paper_figures"
os.makedirs(DIR, exist_ok=True)

C_BG   = "#FAFAFA"
C_WDRO = "#2166AC"
C_INJ  = "#B2182B"
C_SAMP = "#2CA02C"
C_EGO  = "#1B7837"
C_LOOP = "#7570B3"

C_EGO_BOX   = "#2E7D32"
C_OBS_BOX   = "#E65100"
C_BEL_BOX   = "#1565C0"
C_RISK_BOX  = "#C62828"
C_BUILD_BOX = "#37474F"
C_SOLVE_BOX = "#7B1FA2"

plt.rcParams.update({
    "font.size": 8, "font.family": "sans-serif",
    "figure.facecolor": C_BG, "mathtext.fontset": "cm",
})


def box(ax, cx, cy, w, h, lines, *, bg="#FFF", border="#333", lw=1.3,
        fontsize=8, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h, boxstyle="round,pad=0.018",
        facecolor=bg, edgecolor=border, linewidth=lw, zorder=zorder))
    n = len(lines)
    sp = h / (n + 1)
    for i, (txt, sty) in enumerate(lines):
        ax.text(cx, cy + h/2 - (i+1)*sp, txt, ha="center", va="center",
                fontsize=sty.get("fs", fontsize),
                fontweight=sty.get("w", "normal"),
                color=sty.get("c", "black"),
                fontstyle=sty.get("s", "normal"), zorder=zorder+1)


def ar(ax, xy0, xy1, color="#444", lw=1.2):
    ax.add_patch(FancyArrowPatch(
        xy0, xy1, arrowstyle="-|>,head_width=0.10,head_length=0.06",
        mutation_scale=7, color=color, linewidth=lw,
        shrinkA=1, shrinkB=1, zorder=5))


def make_figure():
    W, H = 8.5, 3.8
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(C_BG)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # title
    ax.text(W/2, H - 0.17,
            "Algorithm 1:  Scenario SH-MPCC with WDRO Reweighting",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # ── layout ───────────────────────────────────────────────────
    y_mid = 2.15          # vertical centre of main flow
    bh_in = 0.38          # input box height
    bw_in = 1.05          # input box width
    gap_in = 0.06         # gap between stacked input boxes

    x_in   = 0.70         # inputs
    x_wdro = 2.20         # WDRO
    x_sc   = 3.85         # sample/inject
    x_bld  = 5.30         # build
    x_slv  = 6.65         # solve
    x_app  = 7.80         # apply

    # ── COL 1: inputs (stacked) ──────────────────────────────────
    total_h = 4*bh_in + 3*gap_in
    y_top = y_mid + total_h/2 - bh_in/2

    labels = [
        ("Ego State &\nRef. Path",          "#C8E6C9", C_EGO_BOX),
        ("Obstacle\nPredictions",           "#FFE0B2", C_OBS_BOX),
        (r"Mode Belief $\hat{p}_t^v$",      "#BBDEFB", C_BEL_BOX),
        ("Risk Scores r",                   "#FFCDD2", C_RISK_BOX),
    ]
    in_ys = []
    in_borders = []
    for i, (lbl, bg, brd) in enumerate(labels):
        iy = y_top - i*(bh_in + gap_in)
        in_ys.append(iy)
        in_borders.append(brd)
        box(ax, x_in, iy, bw_in, bh_in,
            [(lbl, {"w": "bold", "fs": 7})],
            bg=bg, border=brd, lw=1.3)

    # ── COL 2: WDRO ─────────────────────────────────────────────
    wdro_w = 1.30
    wdro_h = total_h
    box(ax, x_wdro, y_mid, wdro_w, wdro_h,
        [("WDRO", {"w": "bold", "fs": 9.5}),
         ("Reweighting", {"w": "bold", "fs": 9.5}),
         (r"$\downarrow$  $q^\star$", {"w": "bold", "fs": 9}),
         (r"$\max \langle q, r \rangle$", {"fs": 7.5, "c": "#333"}),
         (r"s.t. $W_D(q, \hat{p}) \leq \rho$", {"fs": 7.5, "c": "#333"})],
        bg="#D1E5F0", border=C_WDRO, lw=2.0)

    # inputs → WDRO
    in_r = x_in + bw_in/2
    wl = x_wdro - wdro_w/2
    spread = wdro_h * 0.30
    for i in range(4):
        ey = y_mid + spread*(1.5 - i)
        ar(ax, (in_r, in_ys[i]), (wl, ey), color=C_WDRO, lw=1.1)

    # ── COL 3: sample / inject ───────────────────────────────────
    bw_sc = 1.25
    bh_sc = 0.50
    sc_gap = 0.35
    y_samp = y_mid + (bh_sc + sc_gap)/2
    y_inj  = y_mid - (bh_sc + sc_gap)/2

    box(ax, x_sc, y_samp, bw_sc, bh_sc,
        [("Sample S Scenarios", {"w": "bold", "fs": 7.5}),
         (r"modes from $q^\star$", {"fs": 6.5, "c": "#555", "s": "italic"})],
        bg="#E8F5E9", border=C_SAMP, lw=1.4)

    box(ax, x_sc, y_inj, bw_sc, bh_sc,
        [(r"Inject $K$ Modes", {"w": "bold", "fs": 7.5}),
         ("deterministic", {"fs": 6.5, "c": "#555", "s": "italic"})],
        bg="#FDDBC7", border=C_INJ, lw=1.4)

    ax.text(x_sc, y_mid, "and/or", ha="center", va="center",
            fontsize=6.5, color="#777",
            bbox=dict(boxstyle="round,pad=0.08", fc=C_BG, ec="#CCC", lw=0.4),
            zorder=6)

    wr = x_wdro + wdro_w/2
    sl = x_sc - bw_sc/2
    ar(ax, (wr, y_mid + 0.20), (sl, y_samp), color=C_SAMP, lw=1.2)
    ar(ax, (wr, y_mid - 0.20), (sl, y_inj),  color=C_INJ, lw=1.2)

    # ── COL 4: build constraints ─────────────────────────────────
    bw_b = 1.10
    bh_b = 0.85
    box(ax, x_bld, y_mid, bw_b, bh_b,
        [("Build Half-Space", {"w": "bold", "fs": 7.5}),
         ("Constraints", {"w": "bold", "fs": 7.5}),
         (r"$k = 1 \ldots N_s$", {"fs": 6.5, "c": "#555", "s": "italic"})],
        bg="#C6DBEF", border=C_BUILD_BOX, lw=1.4)

    sr = x_sc + bw_sc/2
    bl = x_bld - bw_b/2
    ar(ax, (sr, y_samp), (bl, y_mid + 0.12), color=C_BUILD_BOX, lw=1.1)
    ar(ax, (sr, y_inj),  (bl, y_mid - 0.12), color=C_BUILD_BOX, lw=1.1)

    # ── COL 5: solve ─────────────────────────────────────────────
    bw_s = 1.10
    bh_s = 0.85
    box(ax, x_slv, y_mid, bw_s, bh_s,
        [("Solve Scenario", {"w": "bold", "fs": 8}),
         ("MPCC", {"w": "bold", "fs": 8}),
         (r"$\min \sum \ell_{\mathrm{MPCC}}$", {"fs": 6.5, "c": "#444"})],
        bg="#EDE7F6", border=C_SOLVE_BOX, lw=1.6)

    br = x_bld + bw_b/2
    svl = x_slv - bw_s/2
    ar(ax, (br, y_mid), (svl, y_mid), color=C_SOLVE_BOX, lw=1.3)

    # ── COL 6: apply ─────────────────────────────────────────────
    bw_a = 0.85
    bh_a = 0.65
    box(ax, x_app, y_mid, bw_a, bh_a,
        [("Apply", {"w": "bold", "fs": 8.5}),
         (r"$u^\star_{t|t}$", {"w": "bold", "fs": 9.5})],
        bg="#C8E6C9", border=C_EGO, lw=1.8)

    svr = x_slv + bw_s/2
    al = x_app - bw_a/2
    ar(ax, (svr, y_mid), (al, y_mid), color=C_EGO, lw=1.4)

    # ── loop ─────────────────────────────────────────────────────
    loop_y = 0.48
    ax.plot([x_app, x_app], [y_mid - bh_a/2, loop_y],
            color=C_LOOP, lw=1.3, zorder=3)
    ax.plot([x_in, x_app], [loop_y, loop_y],
            color=C_LOOP, lw=1.3, zorder=3)
    ax.plot([x_in, x_in], [loop_y, in_ys[-1] - bh_in/2],
            color=C_LOOP, lw=1.3, zorder=3)

    for i in range(4):
        bb = in_ys[i] - bh_in/2
        ar(ax, (x_in, bb - 0.06), (x_in, bb), color=in_borders[i], lw=1.1)

    ax.text((x_in + x_app)/2, loop_y - 0.18,
            "receding horizon:  advance to t+1, update all inputs, repeat",
            ha="center", va="center", fontsize=7, color=C_LOOP, fontweight="bold")

    # ── notes ────────────────────────────────────────────────────
    notes = [
        r"$\bullet$ $q^\star$ upweights dangerous yet reachable modes",
        r"$\bullet$ $\mathcal{F}_{S,\mathrm{inj}} \subseteq \mathcal{F}_S$ (guarantees preserved)",
        r"$\bullet$ $\rho = \rho_0(1 + \alpha/\sqrt{n_{\mathrm{obs}}})$",
    ]
    cw = W / len(notes)
    for i, n in enumerate(notes):
        ax.text(cw/2 + i*cw, 0.12, n, ha="center", va="center",
                fontsize=6, color="#555")

    # ── save ─────────────────────────────────────────────────────
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(DIR, f"fig_algorithm_overview.{ext}"),
                    bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    plt.close(fig)
    print(f"Saved {DIR}/fig_algorithm_overview.{{pdf,png}}")


if __name__ == "__main__":
    make_figure()
