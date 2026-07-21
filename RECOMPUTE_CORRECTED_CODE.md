# Corrected-code recompute of the ACC paper's figures/tables (2026-07)

Branch `main` (all corrections: J_d/fixed-normal collision constraint, hard velocity
bounds, Dirichlet belief, calibrated radius, Bonferroni VaR, primal-OT recovery).
Reproducible runners: `tests/paper_recompute.cpp`, `tests/paper_recompute_coverage.cpp`
(the original generators were ad-hoc and absent from the repo — these replace them).

## fig:offset — lateral-offset sweep (road ON, S=40, N=250/arm)

| offset | base | WDRO | benefit (NEW) | benefit (OLD paper) |
|---|---|---|---|---|
| 0.0 | 0.548 | 0.540 | **+0.8 pp** | 47.6 pp |
| 0.5 | 0.340 | 0.360 | −2.0 pp | 33.6 pp |
| 1.0 | 0.256 | 0.260 | −0.4 pp | 38.4 pp |
| 1.5 | 0.148 | 0.152 | −0.4 pp | 16.4 pp |
| 2.0 | 0.088 | 0.092 | −0.4 pp | 4.4 pp |
| 3.0 | 0.008 | 0.008 | 0.0 pp | 1.2 pp |

**WDRO shows no collision benefit at any offset (±2 pp, within noise).** The paper's
47.6 pp headline does not survive the corrected code.

## tab:road_on — 5-arm at offset 0 (road ON, S=40, N=250/arm)

| Method | Coll. [95% CI] | Clear. | Contour | Vel.err | Effort | ms |
|---|---|---|---|---|---|---|
| base (no DRO) | 0.556 [.494,.618] | 0.774 | 1.471 | 2.375 | 156.3 | 1.92 |
| ε-greedy belief | 0.564 [.503,.625] | 0.757 | 1.473 | 2.375 | 156.4 | 1.92 |
| uniform belief | 0.612 [.552,.672] | 0.731 | 1.464 | 2.379 | 155.3 | 1.91 |
| WDRO raw-LP | 0.524 [.462,.586] | 0.833 | 1.468 | 2.378 | 154.5 | 1.91 |
| WDRO entropic | 0.568 [.507,.629] | 0.795 | 1.470 | 2.375 | 157.2 | 1.95 |

All five arms are statistically indistinguishable (CIs overlap). Old paper: base
0.656 → WDRO **0.140**. That 52 pp gap is gone. Control effort is flat (~155–157),
so R7.5 is answered — but moot, since there is no safety gain to trade against.

## fig:coverage — joint-missed-mode gap (base planner)

| V | jm\|collided | jm\|safe | gap (NEW) | gap (OLD) |
|---|---|---|---|---|
| 3 | 0.684 | 0.762 | −0.078 | −0.037 |
| 4 | 0.855 | 0.891 | −0.035 | −0.030 |

Negative gap at both V — the coverage-null mechanism finding **reproduces**. This is
the important cross-check: the reconstruction faithfully reproduces the paper's OTHER
result, so the vanished safety benefit is a real property of the corrected code, not a
reconstruction artifact.

## Bottom line

On the corrected code the WDRO layer is **safety-neutral** in this benchmark. The
previously reported 0.656→0.140 benefit was an artifact of the pre-correction code.
The only surviving empirical claim is a negative one (coverage is not a collision
mechanism) — and with no safety benefit there is no positive effect left to attribute.
Diagnosis of WHICH correction removed the benefit (leading suspect: the fixed-normal /
J_d collision-constraint fix, which changes the base and WDRO evasion geometry) is the
open question that decides whether the empirical paper is salvageable or dead.

## Diagnosis — why the benefit vanished (2026-07)

Reproducible: `tests/diagnose_wdro_benefit.cpp`, `tests/shift_test.cpp`.

**It is NOT the DRO tuning.** Offset-0 bisection (base 0.540), each toggled back to old:
| WDRO config | coll | benefit |
|---|---|---|
| NEW default | 0.570 | −3.0 pp |
| calib radius OFF (old heuristic ρ) | 0.570 | −3.0 pp |
| SURROGATE_VAR (old risk) | 0.530 | +1.0 pp |
| primal-OT OFF (old recovery) | 0.545 | −0.5 pp |
| ALL OLD DRO | 0.570 | −3.0 pp |

Even the exact pre-correction DRO config gives no benefit. `num_discs=1` rules out the
J_d collision fix. So the 47.6 pp headline was not produced by anything in the DRO layer.

**It does not reappear under distribution shift** (offset 0, N=200):
| shift | base | WDRO | benefit |
|---|---|---|---|
| none | 0.555 | 0.565 | −1.0 pp |
| dangerous-boost 0.2 | 0.555 | 0.535 | +2.0 pp |
| dangerous-boost 0.4 | 0.360 | 0.350 | +1.0 pp |
| random 0.3 | 0.665 | 0.655 | +1.0 pp |
| random 0.3 + boost 0.3 | 0.695 | 0.685 | +1.0 pp |

All within noise. CAVEAT: `boosted_mode=-1` forces `lane_change_left`, which for the
oncoming geometry steers the obstacle *away* (base drops at boost 0.4), so this did NOT
cleanly test "a high-risk mode under-represented in the belief" — a targeted boost of the
actual collision-causing mode is the one regime not yet ruled out.

**Conclusion.** Under corrected code the WDRO layer is safety-neutral in the paper's
headline scenario and every shift regime tested. The 0.656→0.140 benefit was a
pre-correction artifact (leading suspect: the zero-mask belief bug, which WDRO's
risk-reweighting compensated for). The certificate/coverage theory (support-aware floor,
entropic allocator, calibrated radius) is independent of this and unaffected.

## Mechanism analysis — WDRO hedges correctly but the hedge is counterproductive (2026-07)

Reproducible: `tests/wdro_mechanism_probe.cpp`, `tests/fixed_rho_test.cpp` (adds
`ExperimentConfig::fixed_rho` to force a constant radius).

**The mechanism IS active and hedges toward the most dangerous mode.** On a head-on
scenario q* up-weights argmax(r): at rho>=0.3 it goes bang-bang, q*=1.0 on the highest-
risk mode (decelerating, r=0.965). Hedge check +0.83 vs nominal. Not broken.

**The calibrated radius defangs it in-loop.** rho shrinks with observation count
(0.39 -> 0.037 as n_obs 1 -> 200; q*[danger] 1.0 -> 0.33), and the controller sets
observation_count = mode-history size, so a 200-step rollout runs mostly at rho~0.04.

**Forcing a constant LARGE radius makes WDRO WORSE, not better** (offset 0, road ON, N=200):
| WDRO rho | offset 0 coll | benefit | offset 1 coll | benefit |
|---|---|---|---|---|
| base | 0.565 | -- | 0.260 | -- |
| calibrated (~0.04 in-loop) | 0.550 | +1.5pp | 0.245 | +1.5pp |
| fixed 0.30 | 0.610 | -4.5pp | 0.280 | -2.0pp |
| fixed 0.50 | 0.650 | -8.5pp | 0.260 | 0.0pp |

No fixed radius recovers a benefit; a stronger hedge is monotonically harmful at offset 0.

**Mechanistic conclusion.** WDRO's worst-case reweighting concentrates the S scenarios on
the single highest-risk mode. For a SWITCHING obstacle this destroys the scenario DIVERSITY
needed to cover the other modes the obstacle actually switches into, so a stronger hedge
raises collisions. Worst-case mode concentration is strategically mismatched to scenario-
based collision avoidance under switching dynamics: the useful objective is coverage/
diversity, not worst-case concentration -- and coverage itself is not the collision
bottleneck here (fig:coverage). This is the honest, theory-consistent explanation of the
neutral-to-negative WDRO effect, and it is the opposite of the original headline claim.

(NOTE: `RolloutRecord::active_constraints` reads 0 because the controller never populates
`MPCResult::active_scenarios` -- a reporting stub, not evidence about binding. The plan-
change is instead proven by the collision shift itself: fixed rho=0.5 collides +8.5pp.)

## Full sweep under the ρ-cap fix (rho_max=0.10, graded q*)

After capping rho_max at 0.10 so q* is graded (not bang-bang), the full offset sweep and
5-arm table were re-run (road ON, S=40, N=250/arm):

| offset | base | WDRO | benefit |
|---|---|---|---|
| 0.0 | 0.560 | 0.556 | +0.4 |
| 0.5 | 0.356 | 0.360 | −0.4 |
| 1.0 | 0.276 | 0.260 | +1.6 |
| 1.5 | 0.152 | 0.144 | +0.8 |
| 2.0 | 0.092 | 0.092 | 0.0 |
| 3.0 | 0.012 | 0.012 | 0.0 |

5-arm at offset 0: base 0.544 [.482,.606], ε-greedy 0.580, uniform 0.616,
WDRO raw-LP 0.576 [.515,.637], WDRO entropic 0.548 [.486,.610] — all CIs overlap;
conservatism (contour 1.47 / vel 2.37 / effort ~158) flat across arms.

**Conclusion:** the ρ-cap fixes the reweighting mechanism (q* graded), but WDRO is
safety-neutral across the ENTIRE offset regime and every belief/allocator arm. Consistent
with coverage-null: scenario mode-composition is not the collision driver in this benchmark.

## CORRECTION (2026-07): the base arm was silently running WDRO

A harness bug invalidated every prior base-vs-WDRO comparison above. `ExperimentConfig::
ablation` defaulted to `DRO_FULL`, and the legacy guard (`!enable_dro && ablation !=
NO_INJECTION`) fired on every "base" arm, silently enabling DRO (injection_mode=NONE falls
through to the QSTAR_SAMPLE branch). So "base" == WDRO in all the tables above, which is why
WDRO looked neutral. Fixed: default ablation -> NO_INJECTION.

With a TRUE base, WDRO works and matches the paper's offset-sweep story:

| offset | base (true) | WDRO | benefit |
|---|---|---|---|
| 0.0 | 0.800 | 0.596 | **+20.4 pp** |
| 0.5 | 0.704 | 0.372 | **+33.2 pp** |
| 1.0 | 0.464 | 0.256 | **+20.8 pp** |
| 1.5 | 0.288 | 0.156 | **+13.2 pp** |
| 2.0 | 0.096 | 0.100 | −0.4 pp |
| 3.0 | 0.004 | 0.004 | 0.0 pp |

Large benefit in the lane-intrusion regime (offset <=1.5), vanishing once the obstacle
cannot enter the ego lane -- exactly the original characterization. WDRO also holds more
clearance (e.g. 0.77 vs 0.51 at offset 0). 5-arm base(true) = 0.824 [.777,.871].

ALL earlier "WDRO neutral / not working / mechanism defective" conclusions in this file and
the diagnosis commits are ARTIFACTS of the confounded base and are retracted. The pipeline,
the risk model, and the reweighting were working; the experiment's baseline was wrong.
