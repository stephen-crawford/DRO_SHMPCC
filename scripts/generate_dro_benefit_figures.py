#!/usr/bin/env python3
"""Generate figures from DRO benefit experiments."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

COLORS = {
    'Base': '#1f77b4', 'WDRO-sampling': '#ff7f0e',
    'WDRO-injection': '#2ca02c', 'WDRO-combined': '#d62728',
}
MARKERS = {'Base': 'o', 'WDRO-sampling': 's', 'WDRO-injection': '^', 'WDRO-combined': 'D'}

DATA = Path('paper_figures')
OUT = Path('paper_figures')


def fig_env_sweep():
    """Exp A: collision rate and missed-mode rate across envs and switch_probs."""
    csv = DATA / 'dro_benefit_env_sweep.csv'
    if not csv.exists():
        print(f"  Skip: {csv}")
        return
    df = pd.read_csv(csv)

    envs = df['environment'].unique()
    for env_name in envs:
        edf = df[df['environment'] == env_name]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for method in edf['method'].unique():
            sub = edf[edf['method'] == method].sort_values('switch_prob')
            c = COLORS.get(method, 'gray')
            mk = MARKERS.get(method, 'o')
            sp = sub['switch_prob'].values

            # Collision rate
            ax = axes[0]
            ax.errorbar(sp, sub['collision_rate'] * 100,
                        yerr=[(sub['collision_rate'] - sub['coll_ci_lo']) * 100,
                              (sub['coll_ci_hi'] - sub['collision_rate']) * 100],
                        label=method, color=c, marker=mk, capsize=3, ms=6)

            # Missed-mode rate
            ax = axes[1]
            ax.errorbar(sp, sub['missed_mode_rate'] * 100,
                        yerr=[(sub['missed_mode_rate'] - sub['mm_ci_lo']) * 100,
                              (sub['mm_ci_hi'] - sub['missed_mode_rate']) * 100],
                        label=method, color=c, marker=mk, capsize=3, ms=6)

            # Rare-mode miss rate
            ax = axes[2]
            ax.errorbar(sp, sub['rare_miss_rate'] * 100,
                        yerr=[(sub['rare_miss_rate'] - sub['rare_ci_lo']) * 100,
                              (sub['rare_ci_hi'] - sub['rare_miss_rate']) * 100],
                        label=method, color=c, marker=mk, capsize=3, ms=6)

        titles = [f'{env_name}: Collision Rate', f'{env_name}: Missed-Mode Rate',
                  f'{env_name}: Rare-Mode Miss Rate']
        ylabels = ['Collision rate (%)', 'Missed-mode rate (%)', 'Rare-mode miss rate (%)']
        for ax, t, yl in zip(axes, titles, ylabels):
            ax.set_xlabel('Switch probability')
            ax.set_ylabel(yl)
            ax.set_title(t)
            ax.legend()
            ax.set_ylim(bottom=0)
            ax.set_xticks([0.1, 0.2, 0.3])

        plt.tight_layout()
        out = OUT / f'dro_benefit_{env_name.lower()}_sweep.png'
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved {out}")


def fig_multi_obs():
    """Exp B: collision and coverage vs number of obstacles."""
    csv = DATA / 'dro_benefit_multi_obs.csv'
    if not csv.exists():
        print(f"  Skip: {csv}")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for method in df['method'].unique():
        sub = df[df['method'] == method].sort_values('num_obstacles')
        c = COLORS.get(method, 'gray')
        mk = MARKERS.get(method, 'o')
        x = sub['num_obstacles'].values

        axes[0].errorbar(x, sub['collision_rate'] * 100,
                         yerr=[(sub['collision_rate'] - sub['coll_ci_lo']) * 100,
                               (sub['coll_ci_hi'] - sub['collision_rate']) * 100],
                         label=method, color=c, marker=mk, capsize=3, ms=6)

        axes[1].errorbar(x, sub['missed_mode_rate'] * 100,
                         yerr=[(sub['missed_mode_rate'] - sub['mm_ci_lo']) * 100,
                               (sub['mm_ci_hi'] - sub['missed_mode_rate']) * 100],
                         label=method, color=c, marker=mk, capsize=3, ms=6)

        axes[2].errorbar(x, sub['rare_miss_rate'] * 100,
                         yerr=[(sub['rare_miss_rate'] - sub['rare_ci_lo']) * 100,
                               (sub['rare_ci_hi'] - sub['rare_miss_rate']) * 100],
                         label=method, color=c, marker=mk, capsize=3, ms=6)

    for ax, t, yl in zip(axes,
            ['Collision Rate vs Obstacles', 'Missed-Mode Rate vs Obstacles',
             'Rare-Mode Miss Rate vs Obstacles'],
            ['Collision rate (%)', 'Missed-mode rate (%)', 'Rare-mode miss rate (%)']):
        ax.set_xlabel('Number of obstacles')
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.legend()
        ax.set_ylim(bottom=0)
        ax.set_xticks([1, 2, 4])

    plt.tight_layout()
    out = OUT / 'dro_benefit_multi_obs.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig_mechanism():
    """Exp C: mechanism comparison bar chart."""
    csv = DATA / 'dro_benefit_mechanism.csv'
    if not csv.exists():
        print(f"  Skip: {csv}")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(df))
    colors = [COLORS.get(m, 'gray') for m in df['method']]

    # Collision
    ax = axes[0]
    bars = ax.bar(x, df['collision_rate'] * 100, 0.6, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    ax.errorbar(x, df['collision_rate'] * 100,
                yerr=[(df['collision_rate'] - df['coll_ci_lo']) * 100,
                      (df['coll_ci_hi'] - df['collision_rate']) * 100],
                fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['method'], rotation=15, ha='right')
    ax.set_ylabel('Collision rate (%)')
    ax.set_title('Collision Rate (Oncoming, sp=0.2)')
    ax.set_ylim(bottom=0)

    # Missed-mode
    ax = axes[1]
    ax.bar(x, df['missed_mode_rate'] * 100, 0.6, color=colors, alpha=0.85,
           edgecolor='black', linewidth=0.5)
    ax.errorbar(x, df['missed_mode_rate'] * 100,
                yerr=[(df['missed_mode_rate'] - df['mm_ci_lo']) * 100,
                      (df['mm_ci_hi'] - df['missed_mode_rate']) * 100],
                fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['method'], rotation=15, ha='right')
    ax.set_ylabel('Missed-mode rate (%)')
    ax.set_title('Missed-Mode Rate (Oncoming, sp=0.2)')
    ax.set_ylim(bottom=0)

    # Rare mode miss
    ax = axes[2]
    ax.bar(x, df['rare_miss_rate'] * 100, 0.6, color=colors, alpha=0.85,
           edgecolor='black', linewidth=0.5)
    ax.errorbar(x, df['rare_miss_rate'] * 100,
                yerr=[(df['rare_miss_rate'] - df['rare_ci_lo']) * 100,
                      (df['rare_ci_hi'] - df['rare_miss_rate']) * 100],
                fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['method'], rotation=15, ha='right')
    ax.set_ylabel('Rare-mode miss rate (%)')
    ax.set_title('Rare-Mode Miss Rate (Oncoming, sp=0.2)')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = OUT / 'dro_benefit_mechanism.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Generating DRO benefit figures...")
    fig_env_sweep()
    fig_multi_obs()
    fig_mechanism()
    print("Done.")


if __name__ == '__main__':
    main()
