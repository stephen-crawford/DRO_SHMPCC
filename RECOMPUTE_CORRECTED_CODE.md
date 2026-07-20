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
