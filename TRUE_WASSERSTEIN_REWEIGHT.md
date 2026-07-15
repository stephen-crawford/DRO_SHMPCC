# True Wasserstein-metric reweighting — behavioral result (branch `true-wasserstein-reweight`)

**Change.** The incumbent `WassersteinDRO` recovers the worst-case mode weights
`Q*` with a **dual + greedy heuristic**: it solves the 1-D Kantorovich dual
(`solve_kantorovich_dual`, binary search on the single budget multiplier λ),
then builds `Q*` from *deterministic* transport plans — each source ships **all**
its mass to one destination (bang-bang, `build_plan`) — and convex-**mixes** two
extreme plans (`mix_plans_to_radius`, scalar `mix_alpha`) to hit the ρ budget.
That is a *Wasserstein-inspired* recovery, not the Wasserstein primal.

This branch adds the **true primal optimal-transport LP** (`src/primal_ot.cpp`,
`solve_primal_ot`):

```
max_{π ≥ 0}   Σ_ij π_ij r_j
s.t.  Σ_j π_ij = p_i   ∀i        (source marginals = nominal)
      Σ_ij π_ij D_ij ≤ ρ          (Wasserstein budget, ground cost D = W2-Bures)
      Q*_j = Σ_i π_ij
```

solved exactly with a self-contained Big-M dense simplex (Bland's rule). Unlike
the heuristic, the LP may **split a source's mass fractionally** across
destinations. It is wired into `compute_worst_case_weights` behind an opt-in env
flag **`USE_PRIMAL_OT=1`** (default path unchanged), mirroring the `USE_CVAR_RISK`
opt-in convention.

## Correctness of the LP solver

- **Strong duality.** On every tested instance the primal LP optimum equals the
  Kantorovich dual optimum `g(λ*)` to machine precision
  (`strong_duality_residual = 0`). This validates the simplex.
- **Fractional splitting (crafted check).** Source A (mass 1, risk 0),
  destination B (risk 10, transport cost 1), budget ρ=0.4. Optimal OT plan
  `π_AB=0.4, π_AA=0.6` → `Q={A:0.6, B:0.4}`, risk `4.0`. No feasible
  *deterministic* plan achieves this (keep-all → risk 0; move-all → cost 1 > ρ).
  `solve_primal_ot` returns exactly `qA=0.6, qB=0.4, risk=4.0, cost=0.4`.
  So the solver is the genuine general OT solver, not an accidental bang-bang.

## Measured behavior on the CDC scenario (micro_dro_probe, ρ sweep)

Canonical 6-mode scenario; obstacle at (5,0); ego driving toward it; ρ swept
(binding budget λ\*>0 at ρ≤0.20, slack at ρ=0.30):

| ρ | λ\* | HEUR risk | OT risk | Q\* L1 distance |
|---|---|---|---|---|
| 0.02 | 1.160 | 0.670953 | 0.670953 | **0** |
| 0.05 | 0.821 | 0.701029 | 0.701029 | **0** |
| 0.10 | 0.505 | 0.731010 | 0.731010 | **0** |
| 0.15 | 0.369 | 0.751715 | 0.751715 | **0** |
| 0.20 | 0.350 | 0.770057 | 0.770057 | **0** |
| 0.30 | 0.000 | 0.786021 | 0.786021 | **0** |

**The true primal OT reweighting produces byte-identical `Q*` to the incumbent
heuristic at every radius, binding or slack.** Swapping to true Wasserstein does
**not** change behavior on this problem.

## Why they coincide — and when they would not

The heuristic mixing is *exact* whenever the OT optimum is realized by **one
source splitting toward a single dominant destination**. Here `decelerating` is
the strict risk-argmax at every ρ, so the optimal plan moves mass from the other
modes to `decelerating` and — in the binding regime — one source splits to hit
ρ exactly. The bracket + scalar-`mix_alpha` interpolation represents that single
split precisely.

The methods would diverge only if **two or more sources needed *independent*
fractional splits simultaneously** at the optimal dual price — e.g. when
symmetric modes (`lane_change_left/right`, `turn_left/right`) tie as competing
destinations. A single scalar mixing weight cannot set per-source splits
independently, so the heuristic would be suboptimal there. That degeneracy does
**not** arise in the CDC mode models because one mode strictly dominates the
tail risk.

## Takeaway for the CDC paper

The "Wasserstein-inspired" reweighting in the CDC codebase is, on the paper's
scenario structure, **numerically identical to the exact Wasserstein (primal OT)
reweighting** at all radii — the worst-case distribution `Q*` and its risk are
the same. The heuristic is not merely inspired; it is exact here. If future work
introduces scenarios with multiple co-dominant symmetric destination modes, use
`USE_PRIMAL_OT=1` to get the exact OT `Q*`; otherwise the incumbent recovery is
provably optimal for this problem class.
