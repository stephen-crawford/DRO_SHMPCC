
---

## [2026-07-16] CORRECTED RE-RUN — supersedes all earlier numbers

**Config finally correct**: Dirichlet belief (zero-mask bug fixed), road-boundary
constraints wired + ON, honest baselines. N=250/arm, identical seeds, oncoming
S-curve, S=40. THESE REPLACE the CDC numbers — every earlier margin was measured
against a zero-mask-crippled baseline with the ego unconstrained to the road.

### Road constraints OFF (the old CDC regime)
| arm | collision | 95% CI | clear | contour | vel_err |
|---|---|---|---|---|---|
| base (no DRO) | 0.720 | [.661,.772] | 0.657 | 1.839 | 5.333 |
| eps-greedy belief | 0.700 | [.641,.753] | 0.667 | 1.832 | 5.339 |
| uniform belief | 0.704 | [.645,.757] | 0.660 | 1.836 | 5.333 |
| WDRO-sampling (raw LP) | 0.048 | [.028,.082] | 1.124 | 1.884 | 5.294 |
| WDRO entropic (τ=0.05) | 0.084 | [.056,.125] | 1.104 | 1.893 | 5.267 |

### Road constraints ON (width 4.0 m — realistic lane)
| arm | collision | 95% CI | clear | contour | vel_err |
|---|---|---|---|---|---|
| base (no DRO) | 0.656 | [.595,.712] | 0.798 | 1.349 | 5.340 |
| eps-greedy belief | 0.656 | [.595,.712] | 0.791 | 1.349 | 5.342 |
| uniform belief | 0.620 | [.558,.678] | 0.809 | 1.353 | 5.336 |
| WDRO-sampling (raw LP) | 0.140 | [.102,.188] | 1.134 | 1.404 | 5.373 |
| WDRO entropic (τ=0.05) | 0.152 | [.113,.202] | 1.139 | 1.401 | 5.376 |

### What the corrected numbers say
1. **The mechanism is NOT belief/coverage.** base ≈ eps-greedy ≈ uniform (all ~0.70
   / ~0.65), CIs fully overlapping. Fixing the belief did not move the baseline.
   Confirms the CDC "missed-mode → collision" mechanism is dead; WDRO's benefit is
   risk-aware allocation → earlier evasion, independent of the belief estimator.
2. **WDRO's safety benefit SURVIVES the corrections** — big and real. Road ON:
   0.656 → 0.140 (base → WDRO). That is the empirical paper's headline and it holds
   on the honest config.
3. **Road constraints shrink the margin and RAISE residual collision**: WDRO 0.048 →
   0.140 when the ego is constrained to the road. Reviewer 2's "20% is still
   substantial" is now honestly characterised: ~14% residual on a realistic lane.
4. **Conservatism cost is negligible** (Reviewer 3 comment 5, answered): WDRO
   contouring/velocity error ≈ base (1.40 vs 1.35; 5.38 vs 5.34). WDRO buys safety
   WITHOUT more conservative planning. POSITIVE finding.
5. **Raw LP ≈ entropic** on safety (0.140 vs 0.152, CIs overlap) — the entropic
   certificate costs nothing closed-loop, as predicted.

### Caveat that reruns cannot fix
The obstacle is ONCOMING ON THE REFERENCE CENTRELINE. Even with road constraints
that is a near-degenerate head-on setup, which is why residual collision stays ~14%.
A defensible paper needs the obstacle OFFSET to an adjacent lane — a scenario-design
change, not a config fix. Flagged for the empirical re-submission.
