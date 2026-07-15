#!/usr/bin/env python3
"""Generate paper-quality figure for DRO injection count sweep."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

DATA = Path('figures/paper')
OUT = Path('figures/paper')

STYLES = {
    'DRO':         dict(color='#2ca02c', marker='^', label='DRO (mean)', ls='-', zorder=3),
    'ADVERSARIAL': dict(color='#9467bd', marker='v', label='Adversarial ($\\mu + \\sigma$)', ls='--', zorder=2),
}


def load():
    csv = DATA / 'dro_injection_count_sweep.csv'
    df = pd.read_csv(csv)
    k_map = {'0': 0, '1': 1, '2': 2, '3': 3, '5': 5, 'all': 6}
    df['x'] = df['K_label'].astype(str).map(k_map)
    return df


def fig_main():
    """Paper figure: 2x2 grid — DRO injection only."""
    df = load()

    # Filter to DRO injection only
    sub = df[df['injection_mode'] == 'DRO'].sort_values('x')
    x = sub['x'].values
    sty = STYLES['DRO']

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

    # Single legend line handle for the figure-level legend
    line_handle = None

    # (a) Collision rate
    ax = axes[0, 0]
    ln, = ax.plot(x, sub['collision_rate'] * 100, color=sty['color'], marker=sty['marker'],
                  ms=7, linewidth=1.8, ls=sty['ls'], zorder=sty['zorder'])
    line_handle = ln

    # (b) Missed-mode rate
    ax = axes[0, 1]
    ax.fill_between(x, sub['mm_ci_lo'] * 100, sub['mm_ci_hi'] * 100,
                    color=sty['color'], alpha=0.15)
    ax.plot(x, sub['missed_mode_rate'] * 100, color=sty['color'], marker=sty['marker'],
            ms=7, linewidth=1.8, ls=sty['ls'], zorder=sty['zorder'])

    # (c) Mean clearance
    ax = axes[1, 0]
    ax.plot(x, sub['mean_clearance'], color=sty['color'], marker=sty['marker'],
            ms=7, linewidth=1.8, ls=sty['ls'], zorder=sty['zorder'])

    # (d) 5th-percentile clearance
    ax = axes[1, 1]
    ax.plot(x, sub['p5_clearance'], color=sty['color'], marker=sty['marker'],
            ms=7, linewidth=1.8, ls=sty['ls'], zorder=sty['zorder'])

    # --- Annotation: mark K=1 as optimal ---
    dro1 = sub[sub['K'] == 1].iloc[0]
    axes[0, 0].annotate(f'$K\\!=\\!1$: {dro1["collision_rate"]*100:.1f}%',
                         xy=(1, dro1['collision_rate'] * 100),
                         xytext=(2.5, 35), fontsize=9,
                         arrowprops=dict(arrowstyle='->', color=sty['color'], lw=1.3),
                         color=sty['color'], fontweight='bold')

    # --- Formatting ---
    xticks = [0, 1, 2, 3, 5, 6]
    xlabels = ['0\n(base)', '1', '2', '3', '5', 'all\n($M$)']
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    titles = ['Collision Rate', 'Missed-Mode Rate', 'Mean Clearance', 'Clearance (5th Percentile)']
    ylabels = ['Collision rate (%)', 'Missed-mode rate (%)',
               'Mean clearance (m)', 'Clearance, 5th pct. (m)']

    for i, (ax, pl, t, yl) in enumerate(zip(axes.flat, panel_labels, titles, ylabels)):
        ax.set_xlabel('Injection count $K$')
        ax.set_ylabel(yl)
        ax.set_title(f'{pl} {t}', loc='left', fontweight='bold', fontsize=11)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        if i < 2:
            ax.set_ylim(bottom=0)

    # Baseline reference line on collision plot
    base_coll = sub[sub['K'] == 0].iloc[0]['collision_rate'] * 100
    axes[0, 0].axhline(base_coll, color='gray', ls=':', lw=0.8, alpha=0.5)
    axes[0, 0].text(5.5, base_coll + 1.5, f'Base: {base_coll:.0f}%', fontsize=7.5,
                     color='gray', ha='center')

    # Figure-level suptitle with legend to the right
    fig.suptitle('Effect of Injection Count $K$ on DRO Performance',
                 fontsize=13, x=0.42, ha='center', y=0.98)
    fig.legend([line_handle], ['DRO (mean trajectory injection)'],
               loc='upper right', bbox_to_anchor=(0.98, 0.955),
               fontsize=10, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / 'fig_injection_count_sweep.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close(fig)
    print(f"  Saved {out} + .png")


def fig_focused():
    """Compact 1x3 version for inline use."""
    df = load()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    xticks = [0, 1, 2, 3, 5, 6]
    xlabels = ['0', '1', '2', '3', '5', 'all']

    for mode_name, sty in STYLES.items():
        sub = df[df['injection_mode'] == mode_name].sort_values('x')
        x = sub['x'].values

        # Collision
        ax = axes[0]
        ax.fill_between(x, sub['coll_ci_lo'] * 100, sub['coll_ci_hi'] * 100,
                        color=sty['color'], alpha=0.12)
        ax.plot(x, sub['collision_rate'] * 100, color=sty['color'], marker=sty['marker'],
                ms=7, lw=1.8, ls=sty['ls'], label=sty['label'], zorder=sty['zorder'])

        # Missed-mode
        ax = axes[1]
        ax.fill_between(x, sub['mm_ci_lo'] * 100, sub['mm_ci_hi'] * 100,
                        color=sty['color'], alpha=0.12)
        ax.plot(x, sub['missed_mode_rate'] * 100, color=sty['color'], marker=sty['marker'],
                ms=7, lw=1.8, ls=sty['ls'], label=sty['label'], zorder=sty['zorder'])

        # Clearance
        ax = axes[2]
        ax.plot(x, sub['mean_clearance'], color=sty['color'], marker=sty['marker'],
                ms=7, lw=1.8, ls=sty['ls'], label=sty['label'], zorder=sty['zorder'])

    for ax, t, yl in zip(axes,
            ['(a) Collision Rate', '(b) Missed-Mode Rate', '(c) Mean Clearance'],
            ['Collision rate (%)', 'Missed-mode rate (%)', 'Mean clearance (m)']):
        ax.set_xlabel('Injection count $K$')
        ax.set_ylabel(yl)
        ax.set_title(t, loc='left', fontweight='bold', fontsize=11)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.legend(framealpha=0.9)

    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    out = OUT / 'fig_injection_count_focused.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close(fig)
    print(f"  Saved {out} + .png")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Generating DRO injection count figures...")
    fig_main()
    fig_focused()
    print("Done.")


if __name__ == '__main__':
    main()
