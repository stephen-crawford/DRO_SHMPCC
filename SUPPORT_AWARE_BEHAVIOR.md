# Support-aware Wasserstein floor — closed-loop behavior (base vs enforced)

Branch `support-aware-wasserstein` (based on `main`). Floor:
`q ← α·p̂ + (1−α)·q*`, exposed as `DROConfig::support_aware_alpha`
(`ScenarioMPCConfig::dro_support_aware_alpha`, `ExperimentConfig::support_aware_alpha`).
`α = 0` reproduces the base raw-LP WDRO reweighting exactly.

## Unit level (`support_aware_probe`)

At ρ ≥ 0.2 the base LP collapses to support-1 (q_floor → 0, coverage functional
Ψ_M = 4.0 on the 5-mode instance). With α = 0.2 / 0.5 the recovered q keeps full
support-5, q_floor ≥ α·min(p̂), and Ψ_M stays under the guarantee bound
Σ_i e^{−Mα·p̂_i}. The floor does exactly what it is designed to do.

## Closed loop (`support_aware_ab`, N = 200 seeds/arm, DRO on, rare mode @5%)

| S | α | coll% [95% CI] | miss% | rareMiss% | prog | minClr | contour | velErr | ms |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.00 | 17.0% [11.8, 22.2] | 25.25 | 13.68 | 0.765 | 1.579 | 1.955 | 2.367 | 2.37 |
| 5 | 0.20 | 16.5% [11.4, 21.6] | **0.00** | **0.00** | 0.768 | 1.587 | 1.958 | 2.367 | 2.36 |
| 5 | 0.50 | 17.5% [12.2, 22.8] | **0.00** | **0.00** | 0.763 | 1.581 | 1.956 | 2.367 | 2.35 |
| 20 | 0.00 | 15.5% [10.5, 20.5] | 24.89 | 14.00 | 0.792 | 1.624 | 1.936 | 2.367 | 2.48 |
| 20 | 0.20 | 16.0% [10.9, 21.1] | **0.00** | **0.00** | 0.794 | 1.622 | 1.918 | 2.367 | 2.48 |
| 20 | 0.50 | 16.0% [10.9, 21.1] | **0.00** | **0.00** | 0.791 | 1.630 | 1.933 | 2.367 | 2.48 |

`missed`/`rareMiss` = fraction of (rare-mode) steps where the obstacle's TRUE mode
appears in NONE of the S sampled scenarios.

## Reading

1. **Coverage: the floor eliminates missed modes.** Marginal miss 25% → 0% and
   rare-mode miss ~14% → 0%, at BOTH scenario budgets. Mechanism: full-support q
   feeds the mode-coverage sampler, which then represents every mode (S ≥ #modes),
   so the true mode is always in the sampled set. This is a *finite-sample*
   coverage guarantee, not just a support-of-q statement.
2. **Collision rate: no significant change.** Every α row's 95% CI overlaps the
   base heavily (≈15–17% throughout). Fixing coverage did **not** reduce collisions.
3. **No cost.** Progress, min clearance, planning conservatism (contour / velocity
   error), and solve time are all statistically flat across α. The floor is free.

## Verdict

The support-enforced version delivers a **provable, measured coverage guarantee**
(0% missed modes vs ~25%) at **zero conservatism and zero runtime cost** — it makes
Theorem 1's full-support premise (finite sampling certificate L ≤ 1/q_min) actually
hold in closed loop. But in these scenarios that coverage gain does **not** translate
into fewer collisions: mode coverage is not the collision mechanism here (consistent
with the earlier "missed-mode flat across S" and "coverage is NOT the collision
mechanism at any V" findings). The honest contribution is a **certificate/guarantee**
story, not a collision-rate win — do not claim the latter.

Caveat: at p ≈ 0.16, N = 200 the collision CI half-width is ≈5%, so a sub-2% effect
could be masked; but since coverage was driven fully to 0 with no visible collision
movement, a coverage-driven collision reduction of any size is not in evidence.

## Road-constrained replication (`enable_contouring_constraints=true`)

Re-run of the same A/B on the paper's road-constrained benchmark (ego confined to
the road), so the numbers sit on the same footing as `tab:road_on`:

| S | α | coll% [95% CI] | missed% | rareMiss% | contour | velErr |
|---|---|---|---|---|---|---|
| 5 | 0 (base) | 23.0 [17.2,28.8] | 25.1 | 14.2 | 1.67 | 2.37 |
| 5 | 0.2 | 23.5 [17.6,29.4] | **0.0** | **0.0** | 1.65 | 2.37 |
| 5 | 0.5 | 22.5 [16.7,28.3] | **0.0** | **0.0** | 1.67 | 2.37 |
| 20 | 0 (base) | 18.5 [13.1,23.9] | 24.9 | 14.3 | 1.69 | 2.37 |
| 20 | 0.2 | 20.5 [14.9,26.1] | **0.0** | **0.0** | 1.70 | 2.37 |
| 20 | 0.5 | 20.0 [14.5,25.5] | **0.0** | **0.0** | 1.68 | 2.37 |

Identical qualitative result with road constraints on: coverage → exactly 0,
collision statistically unchanged (overlapping CIs), conservatism flat. This is the
interventional evidence cited in the ACC draft's §VI-B (`sec:res_mechanism`). Note the
α=0 collision here (~19–23%) is higher than the paper's 0.140 headline because this is
a single-obstacle S-curve config, not the ONCOMING benchmark of `tab:road_on`; the
claim rests on the *intervention* (Δcoverage large, Δcollision zero), not the level.
