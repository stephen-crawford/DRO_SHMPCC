#!/usr/bin/env python3
"""Generate paper figures 40-47 from CSV data produced by test_paper_figures."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

METHOD_COLORS = {
    'Base': '#1f77b4',
    'WDRO-sampling': '#ff7f0e',
    'WDRO-injection': '#2ca02c',
    'WDRO-combined': '#d62728',
}
METHOD_MARKERS = {
    'Base': 'o',
    'WDRO-sampling': 's',
    'WDRO-injection': '^',
    'WDRO-combined': 'D',
}

DATA_DIR = Path('paper_figures')
OUT_DIR = Path('paper_figures')


def fig40_risk_lift():
    """Risk targeting validation: distribution of risk lift and injected-mode risk."""
    csv = DATA_DIR / 'fig40_risk_lift.csv'
    if not csv.exists():
        print(f"  Skipping fig40: {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: histogram of risk_lift
    ax = axes[0]
    ax.hist(df['risk_lift'], bins=40, color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', ls='--', lw=1, label='$\\Delta r = 0$')
    ax.axvline(df['risk_lift'].mean(), color='red', ls='-', lw=1.5,
               label=f"mean = {df['risk_lift'].mean():.3f}")
    ax.set_xlabel('Risk lift $\\Delta r = \\mathbb{E}_{q^*}[r] - \\mathbb{E}_{\\hat{p}}[r]$')
    ax.set_ylabel('Count')
    ax.set_title('(a) WDRO risk lift per MPC solve')
    ax.legend(loc='upper right')

    # Right: histogram of injected_mode_risk vs nominal_expected_risk
    ax = axes[1]
    ax.hist(df['injected_mode_risk'], bins=40, color='#2ca02c', alpha=0.7,
            label='Injected mode $r_{m^*}$', edgecolor='black', linewidth=0.5)
    ax.hist(df['nominal_expected_risk'], bins=40, color='#1f77b4', alpha=0.5,
            label='Nominal $\\mathbb{E}_{\\hat{p}}[r]$', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Risk score')
    ax.set_ylabel('Count')
    ax.set_title('(b) Injected-mode risk vs. nominal')
    ax.legend(loc='upper left')

    plt.tight_layout()
    out = OUT_DIR / 'fig40_risk_lift.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig41_budget_scaling():
    """Scenario budget scaling: missed-mode rate and collision rate vs S."""
    csv = DATA_DIR / 'fig41_budget_scaling.csv'
    if not csv.exists():
        print(f"  Skipping fig41: {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for method in df['method'].unique():
        sub = df[df['method'] == method].sort_values('S')
        c = METHOD_COLORS.get(method, 'gray')
        mk = METHOD_MARKERS.get(method, 'o')

        # Left: missed-mode rate
        ax = axes[0]
        ax.errorbar(sub['S'], sub['missed_mode_rate'] * 100,
                     yerr=[
                         (sub['missed_mode_rate'] - sub['mm_ci_lo']) * 100,
                         (sub['mm_ci_hi'] - sub['missed_mode_rate']) * 100,
                     ],
                     label=method, color=c, marker=mk, capsize=3, ms=5)

        # Right: collision rate
        ax = axes[1]
        ax.errorbar(sub['S'], sub['collision_rate'] * 100,
                     yerr=[
                         (sub['collision_rate'] - sub['coll_ci_lo']) * 100,
                         (sub['coll_ci_hi'] - sub['collision_rate']) * 100,
                     ],
                     label=method, color=c, marker=mk, capsize=3, ms=5)

    axes[0].set_xlabel('Scenario budget $S$')
    axes[0].set_ylabel('Missed-mode rate (%)')
    axes[0].set_title('(a) Mode coverage vs. scenario budget')
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel('Scenario budget $S$')
    axes[1].set_ylabel('Collision rate (%)')
    axes[1].set_title('(b) Collision rate vs. scenario budget')
    axes[1].legend()
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    out = OUT_DIR / 'fig41_budget_scaling.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig42_rare_mode_sweep():
    """Rare-mode stress test: rare-mode miss rate and collision rate vs rare_prob."""
    csv = DATA_DIR / 'fig42_rare_mode_sweep.csv'
    if not csv.exists():
        print(f"  Skipping fig42: {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for method in df['method'].unique():
        sub = df[df['method'] == method].sort_values('rare_prob')
        c = METHOD_COLORS.get(method, 'gray')
        mk = METHOD_MARKERS.get(method, 'o')

        # Left: rare-mode miss rate
        ax = axes[0]
        ax.errorbar(sub['rare_prob'] * 100, sub['rare_miss_rate'] * 100,
                     yerr=[
                         (sub['rare_miss_rate'] - sub['rare_ci_lo']) * 100,
                         (sub['rare_ci_hi'] - sub['rare_miss_rate']) * 100,
                     ],
                     label=method, color=c, marker=mk, capsize=3, ms=5)

        # Right: collision rate
        ax = axes[1]
        ax.errorbar(sub['rare_prob'] * 100, sub['collision_rate'] * 100,
                     yerr=[
                         (sub['collision_rate'] - sub['coll_ci_lo']) * 100,
                         (sub['coll_ci_hi'] - sub['collision_rate']) * 100,
                     ],
                     label=method, color=c, marker=mk, capsize=3, ms=5)

    axes[0].set_xlabel('Rare-mode probability (%)')
    axes[0].set_ylabel('Rare-mode miss rate (%)')
    axes[0].set_title('(a) Rare-mode coverage')
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel('Rare-mode probability (%)')
    axes[1].set_ylabel('Collision rate (%)')
    axes[1].set_title('(b) Collision rate')
    axes[1].legend()
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    out = OUT_DIR / 'fig42_rare_mode_sweep.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig43_geometry_ablation():
    """Geometry ablation: semantic vs broken ground costs."""
    csv = DATA_DIR / 'fig43_geometry_ablation.csv'
    if not csv.exists():
        print(f"  Skipping fig43: {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    x = np.arange(len(df))
    w = 0.35

    # Left: collision rate
    ax = axes[0]
    bars = ax.bar(x, df['collision_rate'] * 100, w, color='#2ca02c', alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.errorbar(x, df['collision_rate'] * 100,
                yerr=[
                    (df['collision_rate'] - df['coll_ci_lo']) * 100,
                    (df['coll_ci_hi'] - df['collision_rate']) * 100,
                ],
                fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['ground_cost'], rotation=20, ha='right')
    ax.set_ylabel('Collision rate (%)')
    ax.set_title('(a) Collision rate by ground cost')
    ax.set_ylim(bottom=0)

    # Right: missed-mode rate
    ax = axes[1]
    bars = ax.bar(x, df['missed_mode_rate'] * 100, w, color='#ff7f0e', alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.errorbar(x, df['missed_mode_rate'] * 100,
                yerr=[
                    (df['missed_mode_rate'] - df['mm_ci_lo']) * 100,
                    (df['mm_ci_hi'] - df['missed_mode_rate']) * 100,
                ],
                fmt='none', color='black', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['ground_cost'], rotation=20, ha='right')
    ax.set_ylabel('Missed-mode rate (%)')
    ax.set_title('(b) Missed-mode rate by ground cost')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = OUT_DIR / 'fig43_geometry_ablation.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig44_rho_sweep():
    """Rho sweep / Pareto: collision rate vs missed-mode rate for different rho."""
    csv = DATA_DIR / 'fig44_rho_sweep_pareto.csv'
    if not csv.exists():
        print(f"  Skipping fig44: {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    # Base reference (single point)
    base = df[df['method'] == 'Base']
    if not base.empty:
        ax.scatter(base['missed_mode_rate'] * 100, base['collision_rate'] * 100,
                   s=120, color='#1f77b4', marker='*', zorder=5, label='Base (no DRO)')

    # WDRO-sampling at different rho
    dro = df[df['method'] != 'Base'].sort_values('rho')
    if not dro.empty:
        sc = ax.scatter(dro['missed_mode_rate'] * 100, dro['collision_rate'] * 100,
                        c=dro['rho'], cmap='YlOrRd', s=80, marker='s',
                        edgecolors='black', linewidths=0.5, zorder=4)
        for _, row in dro.iterrows():
            ax.annotate(f"$\\rho$={row['rho']:.2f}",
                        (row['missed_mode_rate'] * 100, row['collision_rate'] * 100),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
        plt.colorbar(sc, ax=ax, label='$\\rho$')

    ax.set_xlabel('Missed-mode rate (%)')
    ax.set_ylabel('Collision rate (%)')
    ax.set_title('WDRO radius sweep: safety--coverage tradeoff')
    ax.legend(loc='upper right')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = OUT_DIR / 'fig44_rho_sweep_pareto.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig45_feasibility():
    """Recovery feasibility: transport cost ratio and feasibility rate."""
    csv = DATA_DIR / 'fig45_recovery_feasibility.csv'
    if not csv.exists():
        print(f"  Skipping fig45: {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: histogram of transport_ratio
    ax = axes[0]
    feas = df[df['feasible'] == 1]['transport_ratio']
    infeas = df[df['feasible'] == 0]['transport_ratio']
    ax.hist(feas, bins=40, color='#2ca02c', alpha=0.7, label='Feasible', edgecolor='black', linewidth=0.5)
    ax.hist(infeas, bins=40, color='#d62728', alpha=0.7, label='Infeasible', edgecolor='black', linewidth=0.5)
    ax.axvline(1.0, color='black', ls='--', lw=1.5, label='$\\rho$ boundary')
    ax.set_xlabel('Implied transport cost / $\\rho$')
    ax.set_ylabel('Count')
    ax.set_title('(a) Transport cost ratio distribution')
    ax.legend()

    # Right: feasibility rate (pie or bar)
    ax = axes[1]
    feas_rate = df['feasible'].mean() * 100
    infeas_rate = 100 - feas_rate
    bars = ax.bar(['Feasible', 'Infeasible'], [feas_rate, infeas_rate],
                  color=['#2ca02c', '#d62728'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Fraction (%)')
    ax.set_title(f'(b) Recovery feasibility rate ({feas_rate:.1f}%)')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, [feas_rate, infeas_rate]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    out = OUT_DIR / 'fig45_recovery_feasibility.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig46_env_robustness():
    """Environment robustness: per-environment collision rate and mode coverage."""
    csv = DATA_DIR / 'fig46_env_robustness.csv'
    if not csv.exists():
        print(f"  Skipping fig46: {csv} not found")
        return
    df = pd.read_csv(csv)

    envs = df['environment'].unique()
    methods = df['method'].unique()
    n_envs = len(envs)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(n_envs)
    w = 0.8 / n_methods

    for i, method in enumerate(methods):
        sub = df[df['method'] == method]
        c = METHOD_COLORS.get(method, 'gray')

        # Collision rate
        ax = axes[0]
        vals = [sub[sub['environment'] == e]['collision_rate'].values[0] * 100 if len(sub[sub['environment'] == e]) > 0 else 0 for e in envs]
        lo = [sub[sub['environment'] == e]['coll_ci_lo'].values[0] * 100 if len(sub[sub['environment'] == e]) > 0 else 0 for e in envs]
        hi = [sub[sub['environment'] == e]['coll_ci_hi'].values[0] * 100 if len(sub[sub['environment'] == e]) > 0 else 0 for e in envs]
        err_lo = [v - l for v, l in zip(vals, lo)]
        err_hi = [h - v for v, h in zip(vals, hi)]
        ax.bar(x + i * w, vals, w, color=c, alpha=0.8, label=method, edgecolor='black', linewidth=0.5)
        ax.errorbar(x + i * w, vals, yerr=[err_lo, err_hi], fmt='none', color='black', capsize=2)

        # Missed-mode rate
        ax = axes[1]
        vals = [sub[sub['environment'] == e]['missed_mode_rate'].values[0] * 100 if len(sub[sub['environment'] == e]) > 0 else 0 for e in envs]
        lo = [sub[sub['environment'] == e]['mm_ci_lo'].values[0] * 100 if len(sub[sub['environment'] == e]) > 0 else 0 for e in envs]
        hi = [sub[sub['environment'] == e]['mm_ci_hi'].values[0] * 100 if len(sub[sub['environment'] == e]) > 0 else 0 for e in envs]
        err_lo = [v - l for v, l in zip(vals, lo)]
        err_hi = [h - v for v, h in zip(vals, hi)]
        ax.bar(x + i * w, vals, w, color=c, alpha=0.8, label=method, edgecolor='black', linewidth=0.5)
        ax.errorbar(x + i * w, vals, yerr=[err_lo, err_hi], fmt='none', color='black', capsize=2)

    for ax_idx, (ax, title, ylabel) in enumerate(zip(axes,
        ['(a) Collision rate', '(b) Missed-mode rate'],
        ['Collision rate (%)', 'Missed-mode rate (%)'])):
        ax.set_xticks(x + w * (n_methods - 1) / 2)
        ax.set_xticklabels(envs, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / 'fig46_env_robustness.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def fig47_runtime():
    """Runtime overhead per MPC solve."""
    csv = DATA_DIR / 'fig47_runtime.csv'
    if not csv.exists():
        print(f"  Skipping fig47: {csv} not found")
        return
    df = pd.read_csv(csv)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    x = np.arange(len(df))
    colors = [METHOD_COLORS.get(m, 'gray') for m in df['method']]

    bars = ax.bar(x, df['mean_solve_ms'], 0.5, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    # Add p95 as error bars
    ax.errorbar(x, df['mean_solve_ms'],
                yerr=[np.zeros(len(df)), df['p95_solve_ms'] - df['mean_solve_ms']],
                fmt='none', color='black', capsize=5, label='P95')

    ax.set_xticks(x)
    ax.set_xticklabels(df['method'])
    ax.set_ylabel('Solve time (ms)')
    ax.set_title('Runtime per MPC solve (mean + P95)')

    # Add text labels
    for bar, mean_val, p95_val, method in zip(bars, df['mean_solve_ms'], df['p95_solve_ms'], df['method']):
        if method == 'WDRO-combined':
            ax.text(bar.get_x() + bar.get_width() / 2 + 0.20, p95_val - 0.15,
                    f'{mean_val:.2f}ms', ha='center', va='top', fontsize=9)
        elif method == 'WDRO-sampling':
            ax.text(bar.get_x() + bar.get_width() / 2 + 0.25, p95_val - 0.10,
                    f'{mean_val:.2f}ms', ha='center', va='top', fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, p95_val + 0.05,
                    f'{mean_val:.2f}ms', ha='center', va='bottom', fontsize=9)

    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = OUT_DIR / 'fig47_runtime.png'
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    figs = {
        'fig40': fig40_risk_lift,
        'fig41': fig41_budget_scaling,
        'fig42': fig42_rare_mode_sweep,
        'fig43': fig43_geometry_ablation,
        'fig44': fig44_rho_sweep,
        'fig45': fig45_feasibility,
        'fig46': fig46_env_robustness,
        'fig47': fig47_runtime,
    }

    filters = sys.argv[1:] if len(sys.argv) > 1 else list(figs.keys())

    print("Generating paper figures...")
    for name in filters:
        if name in figs:
            print(f"\n--- {name} ---")
            figs[name]()
        else:
            print(f"  Unknown figure: {name}")

    print("\nDone.")


if __name__ == '__main__':
    main()
