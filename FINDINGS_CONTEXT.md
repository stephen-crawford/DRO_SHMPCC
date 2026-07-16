
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

---

## [2026-07-16] Road-geometry + obstacle-count experiments (corrected config)

Road ON (4.0m), obstacle(s) oncoming at offset 1.0m, S=40, N=200/arm, identical seeds.
**Directional, NOT publication-clean — both variables are confounded (see below).**

### (A) Road geometry (obstacle at arc 0.55, offset 1.0m)
| geometry | base coll | WDRO coll | benefit | base clr | WDRO clr |
|---|---|---|---|---|---|
| Straight | 0.130 | 0.085 | 4.5 pp | 1.189 | 1.157 |
| Gentle-S (A=1.5) | 0.250 | 0.025 | 22.5 pp | 1.211 | 1.125 |
| S-curve (A=3.0) | 0.465 | 0.160 | 30.5 pp | 0.932 | 1.016 |
| Tight-S (A=5.0) | 0.000 | 0.000 | 0.0 pp | 1.796 | 1.810 |

**CONFOUND**: a fixed 1.0m normal-offset is NOT a fixed conflict level across
curvatures. On Tight-S (A=5.0, path swings ±5m) the offset places the obstacle away
from the ego's actual swept path → 0 collisions for both (artifact, not "tight-S is
safe"). Clean signal: WDRO helps on the moderate-curvature road shapes (Gentle-S,
S-curve: 22–30 pp) where the road constraint limits evasion. Straight = mild conflict
at this offset. A clean version must calibrate offset per-curvature or fix the
obstacle at a world position and vary only the path.

### (B) Obstacle count — clean placement rule (V obstacles even over arc [0.40,0.80], all offset +1.0m)
| V | M^V | base coll | WDRO coll | benefit |
|---|---|---|---|---|
| 1 | 5 | 0.410 | 0.230 | 18.0 pp |
| 2 | 25 | 0.300 | 0.250 | 5.0 pp |
| 3 | 125 | 0.760 | 0.400 | 36.0 pp |
| 4 | 625 | 0.885 | 0.565 | 32.0 pp |

**CONFOUND**: obstacle COUNT is inseparable from DENSITY and raw conflict exposure —
adding obstacles adds joint-mode-space M^V *and* more chances to hit something. V=2
(0.300 base) is a sparse-spacing easy case; the trend is not monotone. What IS clean
and interesting: **WDRO's OWN collision rises 0.23 → 0.57 as V:1→4**, i.e. even the
reweighted planner degrades as M^V (5→625) outruns the fixed budget S=40 — the
concrete "fixed budget cannot cover the joint mode space" story. WDRO's benefit is
largest in the dense regime (V=3,4: 32–36 pp).

**The clean isolation of the M^V effect is an S-SWEEP AT FIXED V** (vary the budget,
show base needs S ≳ M^V while WDRO does not), NOT a count sweep. That is the real
crossover experiment and remains to be run on the corrected config.

### Bottom line
The OFFSET SWEEP (2026-07-16) remains the single cleanest result and the empirical
paper's headline. Geometry and count are supporting/directional and need
confound-controlled redesigns before they can be figures.

---

## [2026-07-16] DECIDING MEASUREMENT: coverage is NOT the mechanism, at ANY V

Does joint-mode coverage predict collision at high V? (The one fact that would let a
coverage/sample-complexity theory framing explain the RIGHT mechanism.) Base planner,
S-curve, road ON, N=300/arm. Split rollouts into collided vs safe; compare
joint-missed-mode rate (NO scenario covers all obstacles' true modes at once).

| V | collided | safe | joint-miss (collided) | joint-miss (safe) | gap |
|---|---|---|---|---|---|
| 3 | 231 | 69 | 0.7248 | 0.7615 | **−0.037** |
| 4 | 253 | 47 | 0.8891 | 0.9187 | **−0.030** |

**Verdict: coverage does NOT predict collision.** The gap is NEGATIVE at both V —
collided rollouts have slightly *lower* joint-miss than safe ones. No positive
correlation. (Marginal per-obstacle missed-mode is 0.0000 throughout because
ensure_mode_coverage is hardcoded on — uninformative.)

**Two things are simultaneously true and that is the whole point:**
1. The joint mode config is genuinely rarely covered — joint-miss 72% at V=3, 89% at
   V=4 (S=40 vs M^V=125, 625). The "fixed budget cannot cover M^V" observation is REAL.
2. It is CAUSALLY INERT. The planner does not collide because it missed the exact joint
   mode; it collides from dense multi-agent geometry. Coverage and collision are
   uncorrelated (slightly anti-correlated).

**Consequence for a theory framing**: the coverage/sample-complexity-of-joint-modes
crossover is DEAD. Refuted at V=1 (missed 0.0000 / coll 0.71) and now at V=3,4
(joint-miss uncorrelated with collision). WDRO's benefit is NOT coverage; it is
risk-aware allocation (earlier evasion) — which is risk-averse MPC (Sopasakis, Singh),
already pre-empted. No novel sample-complexity theorem survives this. The M^V-vs-S
result stays an EMPIRICAL scaling observation, not a theorem.
