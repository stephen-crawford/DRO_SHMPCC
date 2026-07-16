
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

---

## [2026-07-16] OBSTACLE-OFFSET SWEEP — the scenario fix that answers Reviewers 2 & 8

The oncoming obstacle was on the ego's reference centreline (degenerate head-on).
Offsetting it laterally to an adjacent lane and sweeping the offset. Road ON (4.0m,
ego confined to ±2.0m). N=250/arm, identical seeds. Combined ego+obstacle radius ~0.85m.

| lateral offset | base coll | WDRO coll | benefit | base clr | WDRO clr |
|---|---|---|---|---|---|
| 0.0 (centreline) | 0.640 | 0.164 | 47.6 pp | 0.778 | 1.115 |
| 0.5 | 0.572 | 0.236 | 33.6 pp | 0.886 | 1.023 |
| 1.0 | 0.536 | 0.152 | 38.4 pp | 0.876 | 1.030 |
| 1.5 | 0.236 | 0.072 | 16.4 pp | 1.221 | 1.072 |
| 2.0 (road edge) | 0.076 | 0.032 | 4.4 pp | 1.419 | 1.383 |
| 3.0 (own lane) | 0.012 | 0.000 | 1.2 pp | 2.031 | 2.101 |

### This is the empirical paper's headline, and it answers the reviewers
1. **Reviewer 2 ("unreasonable scenario / 20% still substantial") — RESOLVED.** The
   64% base collision was the centreline artifact. Pull the obstacle to a real
   adjacent lane and the baseline becomes sane: 7.6% at offset 2.0m, 1.2% at 3.0m.
   The high collision rates were a degenerate placement, now characterised, not hidden.
2. **Reviewer 8 ("significant advantage over existing methods?") — ANSWERED as a
   BOUNDARY.** WDRO's benefit is large precisely where the obstacle can intrude into
   the ego's lane (offset ≤ ~1.5m: 16–48 pp) and vanishes once it cannot (offset ≥
   2.0m: ≤ 4.4 pp). That is the parameter-selection guidance R8 asked for: *use WDRO
   when obstacles may enter your lane; it is redundant otherwise.*
3. **Residual collision is now defensible.** At offset 1.0m (obstacle genuinely in the
   ego's lane, mode-switching) WDRO is 15.2% — a hard, legitimate conflict, not a
   rigged head-on.
4. Minor non-monotonicity at offset 0.5 (WDRO 0.236 vs 0.164/0.152 at 0.0/1.0) — a
   geometry effect at that specific overlap; CIs nearly touch. Note but not load-bearing.

### Standing residual (honest)
This is single-obstacle. The V>1 multi-obstacle joint-mode-space argument (M^V) is
untouched by this sweep and remains the scaling story for a theory framing.
