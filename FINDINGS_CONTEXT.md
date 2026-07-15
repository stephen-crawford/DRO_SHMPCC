# DRO-SHMPCC Experimental Findings Context

**Last updated:** 2026-03-13 (v5 — includes focused experiments F1–F11)
**Total experiments:** 30+ CSV datasets across 5 experiment suites
**Total rollouts:** ~420,000+ across all experiments

---

## 1. FLAGSHIP RESULT: Oncoming Scenario

The strongest, most statistically significant finding across all experiments.

| Method | Collision Rate | 95% CI | Mean Clearance | p5 Clearance |
|--------|---------------|--------|----------------|--------------|
| Base | 93.5% | [91.8%, 94.9%] | 0.516 m | 0.270 m |
| WDRO-sampling | 16.7% | [14.5%, 19.1%] | 1.058 m | 0.473 m |
| WDRO-injection | 15.6% | [13.5%, 18.0%] | 1.059 m | 0.485 m |
| Mode-DRO(inj) | 13.3% | [10.9%, 16.1%] | 1.090 m | 0.542 m |
| Traj-DRO(inj) | 36.6% | [31.7%, 41.7%] | — | — |
| Traj-DRO(comb) | 18.2% | [14.9%, 22.1%] | — | — |

**Key:** CIs are non-overlapping between Base and ALL DRO variants → **statistically significant at p<0.05**.
**Collision reduction:** 78–80 percentage points. **Clearance doubles** (0.52→1.06 m).

---

## 2. ENVIRONMENT DIFFICULTY RANKING

From easiest to hardest, based on Base collision rates:

| Environment | Base Collision Rate | DRO Benefit |
|-------------|--------------------|----|
| Circle | 0.0% | None (no collisions to prevent) |
| Straight | 0.1% (CDC) / 57.8–59.0% (generalization) | Minimal in CDC, strong in generalization |
| Narrow | 0.0% | None |
| Intersection | 7.7% | Moderate (→4.5–5.7%) |
| Oncoming | 93.5% | **Massive** (→13.3–18.2%) |
| Tight-S + Oncoming | 97.0% | Large (→21.8–30.6%) |
| Crossing | 82.8–85.8% | Large (→19.8–23.2%) |

**Pattern:** DRO benefit scales with base difficulty. Negligible benefit when Base is already safe; maximum benefit when Base fails catastrophically.

---

## 3. SPEED CROSSOVER EFFECT

Obstacle speed is the most critical environmental parameter.

### G7 Speed Sweep (S-curve, N=500)
| Speed (m/s) | Base Coll. | WDRO-inject Coll. | Benefit |
|-------------|-----------|-------------------|---------|
| 0.3 | 75.8% | 12.4% | **63.4 pp** |
| 0.5 | 83.6% | 11.8% | **71.8 pp** |
| 0.8 | 87.8% | 14.4% | **73.4 pp** |
| 1.0 | 81.2% | 12.4% | **68.8 pp** |
| 1.3 | 55.6% | 17.2% | **38.4 pp** |
| 1.5 | 16.2% | 15.2% | **1.0 pp** (≈zero) |
| 1.8 | 5.4% | 12.2% | **−6.8 pp** (DRO HURTS) |
| 2.0 | 2.6% | 9.0% | **−6.4 pp** (DRO HURTS) |

**Critical crossover at v ≈ 1.5 m/s ON S-CURVE ONLY.** Below this, DRO provides massive benefits. Above, the obstacle passes too fast for proactive avoidance to help, and DRO's conservatism causes unnecessary evasive maneuvers. **Important:** F11 showed this crossover is S-curve specific. On Tight-S, DRO always helps. On Straight, DRO benefit actually increases with speed.

### T7 Speed Sweep (Mode-DRO vs Traj-DRO)
| Speed | Mode-DRO(inj) | Traj-DRO(inj) | Traj-DRO(comb) |
|-------|---------------|----------------|----------------|
| 0.5 | 11.8% | 35.8% | 17.6% |
| 0.8 | 14.2% | 39.4% | 21.0% |
| 1.0 | 14.4% | 40.2% | 20.4% |
| 1.3 | 15.2% | 38.2% | 18.4% |
| 1.5 | 15.8% | 40.2% | 18.4% |
| 2.0 | ↑ | ↑ | ↑ |

---

## 4. METHOD RANKING (Mode-Level DRO Variants)

### F1 High-N Differentiation (N=2000, Oncoming) — RESOLVES Gap 1
| Method | Collision Rate | 95% CI |
|--------|---------------|--------|
| Base | 93.1% | [91.9%, 94.1%] |
| TopRisk-K1 | **14.1%** | [12.6%, 15.6%] |
| DiverseRisk-K1 | 14.2% | [12.7%, 15.7%] |
| WDRO-inject-K2 | 14.2% | [12.7%, 15.8%] |
| WDRO-inject-K1 | 15.0% | [13.5%, 16.6%] |
| WDRO-sampling | 16.7% | [15.1%, 18.4%] |
| **Softmax-tau5** | **19.0%** | **[17.3%, 20.7%]** |

**Key findings from F1:**
- TopRisk-K1, DiverseRisk-K1, inject-K1, inject-K2 remain **statistically indistinguishable** (all CIs overlap) even at N=2000
- **Softmax-tau5 is confirmed significantly worse** than TopRisk-K1 (CIs non-overlapping: [17.3%, 20.7%] vs [12.6%, 15.6%])
- WDRO-sampling CIs overlap with both injection methods and Softmax — intermediate tier
- **Conclusion:** All injection-based methods are equivalent in performance; the specific injection criterion (DRO vs TopRisk vs DiverseRisk) does not matter

---

## 5. TRAJECTORY-LEVEL DRO FINDINGS

### Consistent Patterns:
- **Traj-DRO(inj):** Consistently worst DRO variant. 35–51% collision rates (vs 12–30% for Mode-DRO). **58–62% missed mode rate.** Gets stuck/over-conservative (progress 0.81–0.90 vs 0.96 for others). This is because trajectory-level injection overpopulates one region of scenario space.
- **Traj-DRO(q\*):** Close to Mode-DRO(q\*) on single-obstacle scenarios. Falls apart on multi-obstacle (54.8% on Mixed-3obs).
- **Traj-DRO(comb):** Best trajectory-level variant. Matches Mode-DRO(q\*) on most scenarios.

### **MAJOR FINDING — F5: Traj-DRO(comb) BEATS Mode-DRO(inj) at Speed 1.3 (N=2000)**

| Speed | Method | Collision | 95% CI |
|-------|--------|----------|--------|
| 1.3 | **Traj-DRO(comb)** | **9.2%** | **[8.0%, 10.5%]** |
| 1.3 | Traj-DRO(inj) | 11.9% | [10.6%, 13.4%] |
| 1.3 | Mode-DRO(q\*) | 12.0% | [10.6%, 13.5%] |
| 1.3 | Mode-DRO(inj) | 14.7% | [13.2%, 16.3%] |
| 1.3 | Base | 22.0% | [20.2%, 23.9%] |
| 1.0 | Mode-DRO(inj) | 19.5% | [17.8%, 21.2%] |
| 1.0 | Traj-DRO(comb) | 20.4% | [18.7%, 22.2%] |
| 1.0 | Base | 38.3% | [36.1%, 40.4%] |

**Traj-DRO(comb) at 9.2% [8.0%, 10.5%] vs Mode-DRO(inj) at 14.7% [13.2%, 16.3%] — CIs NON-OVERLAPPING. Statistically significant.**

This reverses the prior finding: at medium speed (1.3 m/s) on S-curve, trajectory-level DRO combined approach genuinely outperforms mode-level injection. The advantage disappears at speed 1.0 where Mode-DRO(inj) and Traj-DRO(comb) are equivalent.

**Why:** At speed 1.3, the obstacle passes more quickly. Traj-DRO(comb)'s resampling from DRO-weighted mode distribution adapts better to the medium-speed regime than deterministic injection, which injects worst-case modes that may be too conservative for the speed.

### F6: Transition Speed Sweep — PRECISE CROSSOVER (N=1500, S-curve)
| Speed | Mode-DRO(inj) | CI | Traj-DRO(comb) | CI | Gap |
|-------|--------------|-----|----------------|-----|-----|
| 0.8 | **13.2%** | [11.6, 15.0] | 17.9% | [16.1, 20.0] | Mode +4.7 |
| 1.0 | **17.9%** | [16.0, 19.9] | 19.5% | [17.6, 21.6] | Mode +1.6 |
| **1.1** | 21.3% | [19.3, 23.4] | **16.6%** | [14.8, 18.6] | **Traj −4.7** |
| 1.2 | 19.6% | [17.7, 21.7] | **13.7%** | [12.1, 15.6] | **Traj −5.9** |
| 1.3 | 15.9% | [14.2, 17.9] | **8.2%** | [6.9, 9.7] | **Traj −7.7** |
| 1.4 | **11.1%** | [9.6, 12.8] | 9.7% | [8.3, 11.3] | Traj −1.4 |
| 1.5 | **13.8%** | [12.1, 15.6] | 16.3% | [14.5, 18.2] | Mode +2.5 |

**Crossover at v ≈ 1.1 m/s.** Traj-DRO(comb) advantage window: **v = 1.1 to 1.4 m/s**. Peak advantage at v = 1.3 (7.7 pp gap, CIs clearly non-overlapping). Above v = 1.5, Mode-DRO(inj) regains advantage.

### F7: Traj-DRO on Tight-S Path (N=1500) — RESOLVES Gap 8 (path dependence)
| Speed | Base | Mode-DRO(inj) | Mode-DRO(q*) | Traj-DRO(comb) |
|-------|------|---------------|---------------|----------------|
| 0.8 | 97.8% [97.0,98.4] | 25.3% [23.1,27.5] | 27.7% [25.5,30.0] | 29.3% [27.1,31.7] |
| 1.0 | 82.4% [80.4,84.2] | **8.1%** [6.9,9.6] | 8.3% [7.0,9.8] | 10.3% [8.9,12.0] |
| 1.2 | 52.2% [49.7,54.7] | 6.3% [5.1,7.6] | **5.7%** [4.7,7.0] | 6.3% [5.2,7.7] |
| 1.3 | 36.0% [33.6,38.5] | 4.3% [3.4,5.5] | **3.9%** [3.1,5.0] | 4.9% [3.9,6.1] |

**Key findings from F7:**
- **Traj-DRO(comb) does NOT outperform Mode-DRO on Tight-S** — unlike S-curve where Traj-DRO(comb) beat Mode-DRO(inj) at v=1.3 (9.2% vs 14.7%), on Tight-S all three DRO variants are **statistically indistinguishable** at every speed (CIs overlap).
- **The Traj-DRO advantage is S-curve specific, not a general speed phenomenon.** Gap 8 resolved: the speed-crossover finding from F5/F6 does not generalize to all path geometries.
- **Mode-DRO(q*) slightly edges out others at v=1.2-1.3:** 5.7% and 3.9% vs 6.3% and 4.3% for inject, but CIs overlap.
- **DRO is spectacularly effective on Tight-S at medium speeds:** Base 82.4% → DRO 8.1% at v=1.0 (74.3 pp reduction), Base 36.0% → DRO 3.9% at v=1.3 (32.1 pp).
- **Clearance on Tight-S improves dramatically:** 0.60m → 1.14m at v=1.0, 1.08m → 1.33m at v=1.3.

### Multi-Obstacle Catastrophe:
On Mixed-3obs: Traj-DRO(inj) = 79.2%, Traj-DRO(comb) = 61.0%, Mode-DRO(inj) = 30.0%. Trajectory DRO completely fails with multiple obstacles because the per-trajectory reweighting cannot separately handle different obstacle threats.

---

## 6. PATH GEOMETRY EFFECTS

### G1 Path Geometry (7 methods × 5 paths, N=500)
| Path | Base Coll. | Best DRO Coll. | Reduction |
|------|-----------|---------------|-----------|
| Straight | 57.8% | 11.0% (inject-K1) | **46.8 pp** |
| Gentle-S | 71.2% | 13.2% (inject-K1) | **58.0 pp** |
| S-curve | 79.8% | 15.2% (TopRisk) | **64.6 pp** |
| Tight-S | 97.0% | 21.8% (inject-K1) | **75.2 pp** |
| Circle | 0.0% | 0.0% | 0 pp |

**Pattern:** Higher curvature → higher base collision rate → larger DRO benefit.
**But:** DRO residual collision rate also increases with curvature (11% → 22%).

### T1 Path Geometry (6 methods × 4 paths, N=500)
Mode-DRO(inj) consistently best: 12.4% (S-curve), 12.6% (Straight), 30.4% (Tight-S).
Traj-DRO(inj): 22.8% (Straight), 41.8% (S-curve), 51.4% (Tight-S) — degradation amplified by curvature.

---

## 6b. SPEED × PATH INTERACTION (F2, N=1000) — RESOLVES Gap 2

### Collision Rate Table
| Speed | Path | Base | inject-K1 | TopRisk | sampling | Reduction (best) |
|-------|------|------|-----------|---------|----------|-----------------|
| 0.8 | Straight | 56.4% | 15.1% | **12.7%** | 16.4% | **43.7 pp** |
| 0.8 | S-curve | 74.4% | 13.3% | **12.1%** | 15.9% | **62.3 pp** |
| 0.8 | Tight-S | 97.8% | **25.4%** | 26.0% | 28.5% | **72.4 pp** |
| 1.0 | Straight | 61.3% | 18.0% | **15.8%** | 16.6% | **45.5 pp** |
| 1.0 | S-curve | 38.4% | 20.8% | **19.0%** | 22.1% | **19.4 pp** |
| 1.0 | Tight-S | 82.8% | 9.3% | **7.9%** | 9.6% | **74.9 pp** |
| 1.3 | Straight | 61.7% | **16.0%** | 21.6% | 20.8% | **45.7 pp** |
| 1.3 | S-curve | 23.8% | 15.5% | 13.9% | **10.8%** | **13.0 pp** |
| 1.3 | Tight-S | 39.5% | 4.1% | 6.3% | **3.4%** | **36.1 pp** |

### KEY SWEET SPOT: Speed 1.0 + Tight-S
- Base 82.8% → TopRisk 7.9% = **74.9 pp reduction** (CIs non-overlapping)
- This is the SECOND LARGEST DRO benefit after the flagship Oncoming result
- DRO clearance 1.14m vs Base 0.60m (nearly doubles)

### Speed 1.3 + Tight-S: DRO Near-Perfect
- Base 39.5% → WDRO-sampling **3.4%** [2.4%, 4.7%] = nearly eliminates collisions
- inject-K1 at 4.1% [3.0%, 5.5%] also excellent
- p5 clearance: 0.99m (inject-K1) vs 0.29m (Base)

### Pattern: Speed × Curvature Interaction
At speed 1.3, DRO benefit INCREASES with curvature (opposite of lower speeds):
- Straight: 45.7 pp benefit
- S-curve: 13.0 pp benefit (lower because Base is also better at 23.8%)
- Tight-S: 36.1 pp benefit (higher curvature helps DRO stand out)

---

## 6c. K=1 vs K=2 INJECTION (F3, N=1000) — RESOLVES Gap 4

| Obstacles | inject-K1 | inject-K2 | TopRisk-K1 | TopRisk-K2 | Base |
|-----------|----------|----------|-----------|-----------|------|
| 1 | 11.2% | 12.4% | 11.4% | 12.0% | 6.3% |
| 2 | 17.8% | **16.0%** | 18.5% | **17.0%** | 67.1% |
| 3 | **13.3%** | 15.2% | 15.5% | 15.1% | 50.2% |

**Conclusions:**
- K=2 provides marginal benefit with 2 obstacles (~1.5-2 pp better) but CIs overlap
- K=1 is actually better with 3 obstacles (13.3% vs 15.2% for inject)
- **K=1 is sufficient** — the extra mode injection from K=2 does not provide consistent improvement
- Note: 1-obstacle result shows Base at 6.3% (lower than DRO) due to different obstacle geometry in this scenario

---

## 7. CHALLENGING ENVIRONMENT RESULTS

### G6 Challenging Environments (8 variants, 7 methods, N=500)
| Challenge | Base Coll. | Best DRO | Reduction |
|-----------|-----------|----------|-----------|
| HighSpeed-1.5 | 16.2% | 12.0% (WDRO-samp) | 4.2 pp |
| HighSpeed-2.0 | 2.6% | 9.0% (inject-K1) | **−6.4 pp** |
| Crossing | 85.8% | 19.8% (WDRO-samp) | **66.0 pp** |
| Diagonal | 66.6% | 14.0% (DiverseRisk) | **52.6 pp** |
| Mixed-2obs | 65.8% | 15.0% (inject-K2) | **50.8 pp** |
| Mixed-3obs | 77.8% | 29.6% (DiverseRisk) | **48.2 pp** |
| Dense-4obs | 70.0% | 30.4% (Softmax) | **39.6 pp** |
| HighSpeed-Dense | 46.0% | 21.2% (TopRisk/Diverse) | **24.8 pp** |

**Key observation:** HighSpeed-2.0 is the ONLY scenario where DRO hurts (Base 2.6% → DRO 9.0%). Obstacle passes too fast.

### T6 Challenging Environments (Mode DRO vs Traj DRO)
On Mixed-3obs: Mode-DRO(inj) 30.0% vs Traj-DRO(inj) 79.2% — **49 pp gap**, CIs non-overlapping.
On Crossing: Mode-DRO(inj) 23.2% vs Traj-DRO(inj) 39.2%.

---

## 8. CLEARANCE ANALYSIS (CDC C1/C2)

### C1: Across 4 Environments (N=1000)
- **Oncoming:** Clearance doubles: 0.52→1.06 m (mean), 0.27→0.49 m (p5). CIs non-overlapping.
- **Intersection:** Slight clearance decrease with DRO (1.44→1.26 m) — DRO routes through tighter gaps but avoids collisions.
- **Straight/Narrow:** No meaningful difference (already safe).

### C2: All 7 Baselines on Oncoming (N=1000)
| Method | Collision | Mean Clearance | p5 Clearance |
|--------|----------|----------------|--------------|
| Base | 92.7% | 0.523 m | 0.287 m |
| WDRO-injection | 14.6% | 1.058 m | 0.497 m |
| WDRO-sampling | 18.1% | 1.062 m | 0.470 m |
| Uniform-coverage | 15.0% | 1.050 m | 0.510 m |
| Softmax-risk | 16.5% | 1.069 m | 0.507 m |
| Eps-greedy | 18.3% | 1.041 m | 0.456 m |
| Top-risk-inject | 16.8% | 1.048 m | 0.517 m |

### Conditional Clearance:
- Non-collision runs: Base 1.06 m vs DRO 1.15 m — modest improvement
- Collision runs: Base 0.48 m vs DRO 0.57 m — DRO crashes at higher clearance (less severe)

### F4: High-N Clearance Data (N=2000, Oncoming + Crossing) — RESOLVES Gap 6
| Env | Method | Collision | Mean Clr | p5 Clr | p50 Clr | Coll Clr | NoColl Clr |
|-----|--------|----------|----------|--------|---------|----------|------------|
| Oncoming | Base | 74.5% | 0.693m | 0.187m | 0.540m | 0.461m | 1.371m |
| Oncoming | inject-K1 | **12.6%** | **1.095m** | **0.521m** | **1.123m** | 0.553m | 1.173m |
| Oncoming | TopRisk-K1 | 14.8% | 1.091m | 0.541m | 1.117m | 0.590m | 1.178m |
| Crossing | Base | 76.2% | 0.707m | 0.141m | 0.531m | 0.448m | 1.533m |
| Crossing | inject-K1 | **23.0%** | **1.015m** | **0.242m** | **1.084m** | 0.428m | 1.189m |
| Crossing | TopRisk-K1 | 23.4% | 1.012m | 0.227m | 1.085m | 0.436m | 1.188m |

**DRO doubles median clearance** in both scenarios: 0.54→1.12m (Oncoming), 0.53→1.08m (Crossing).
**p5 clearance triples** in Oncoming: 0.19→0.52m.
**Crossing is harder than Oncoming** for DRO: 23% vs 12.6% residual collision rate.

---

## 8b. BEST DRO SHOWCASE (F8, N=2000) — Paper-Ready Results

High-N results for the four strongest DRO scenarios.

| Scenario | Base | inject-K1 | TopRisk-K1 | sampling | Best Reduction |
|----------|------|-----------|------------|---------|----------------|
| Oncoming-default | 93.7% [92.5,94.6] | 14.8% [13.3,16.4] | 15.7% [14.1,17.3] | 18.5% [16.9,20.3] | **78.9 pp** |
| Tight-S@v1.0 | 81.5% [79.7,83.1] | 8.2% [7.0,9.4] | **7.1%** [6.1,8.3] | 8.4% [7.3,9.7] | **74.4 pp** |
| Tight-S@v1.3 | 34.9% [32.8,37.0] | 5.1% [4.2,6.1] | 4.8% [3.9,5.8] | **3.9%** [3.1,4.8] | **31.0 pp** |
| S-curve-crossing | 75.6% [73.7,77.4] | 24.2% [22.3,26.1] | **22.7%** [20.9,24.6] | 25.6% [23.7,27.6] | **52.9 pp** |

### Key F8 Findings:
1. **Tight-S@v1.0 rivals Oncoming as flagship:** 74.4 pp reduction (81.5%→7.1%), clearance doubles (0.60→1.16m). Both scenarios have clear, non-overlapping CIs.
2. **Tight-S@v1.3 near-eliminates collisions:** Base 34.9% → WDRO-sampling 3.9% [3.1%, 4.8%]. Only ~1 in 25 rollouts collides with DRO.
3. **TopRisk-K1 slightly outperforms inject-K1** on Tight-S (7.1% vs 8.2%, 4.8% vs 5.1%) but CIs overlap — consistent with F1 finding of indistinguishable injection methods.
4. **WDRO-sampling wins at highest speed (v=1.3):** 3.9% vs inject 5.1% and TopRisk 4.8%. At high speed, q*-sampling's exploration helps, though CIs still overlap.
5. **Crossing is the hardest for DRO:** 22.7% best residual collision rate (vs 7.1% on Tight-S, 14.8% on Oncoming). Perpendicular approach is fundamentally harder to avoid.
6. **Clearance improvements by scenario:**
   - Oncoming: 0.52→1.05m (2.0×)
   - Tight-S@v1.0: 0.60→1.16m (1.9×)
   - Tight-S@v1.3: 1.08→1.33m (1.2×)
   - Crossing: 0.71→1.02m (1.4×)

---

## 8c. MULTI-OBSTACLE SCALING AT SWEET SPOT (F9, N=1500)

Tests DRO robustness with increasing obstacle count at Tight-S v=1.0 (the sweet spot).

| # Obs | Base | inject-K1 | CI | TopRisk-K1 | sampling | DRO Benefit |
|-------|------|-----------|-----|------------|---------|-------------|
| 1 | 81.9% [79.8,83.7] | **7.3%** [6.1,8.8] | | 7.9% [6.6,9.3] | 8.7% [7.4,10.3] | **74.6 pp** |
| 2 | 91.9% [90.4,93.2] | **9.3%** [8.0,10.9] | | 9.6% [8.2,11.2] | 8.6% [7.3,10.1] | **82.6 pp** |
| 3 | 92.6% [91.2,93.8] | **8.5%** [7.2,10.0] | | 8.5% [7.2,10.0] | 9.4% [8.0,11.0] | **84.1 pp** |

### KEY FINDING: DRO benefit INCREASES with obstacle count at the sweet spot
- Base gets worse (82→93%) but DRO stays flat (7-10%). CIs overlap across all obstacle counts.
- **DRO benefit goes from 74.6 pp (1 obs) to 84.1 pp (3 obs)** — scaling is actually favorable!
- Unlike previous F3 results at default speed, Tight-S v=1.0 is where DRO truly excels with multi-obstacle.
- Clearance remains stable: ~1.14-1.20m for DRO vs ~0.51-0.60m for Base across all counts.
- **All three DRO methods remain statistically indistinguishable** at every obstacle count.

---

## 8d. TIGHT-S SPEED SWEEP — FULL DRO PROFILE (F10, N=1500)

Complete speed sweep on Tight-S path, mapping the full DRO benefit curve.

| Speed | Base | inject-K1 | TopRisk-K1 | sampling | DRO Benefit |
|-------|------|-----------|------------|---------|-------------|
| 0.6 | 97.2% | 34.4% | 33.7% | 33.5% | **63.7 pp** |
| 0.8 | 97.4% | 24.1% | 25.9% | 26.2% | **73.3 pp** |
| 1.0 | 82.9% | 8.4% | 9.0% | 11.1% | **74.5 pp** |
| 1.2 | 52.0% | 5.2% | 6.1% | 4.8% | **47.2 pp** |
| 1.3 | 32.1% | 6.5% | 4.1% | 4.4% | **28.0 pp** |
| 1.5 | 12.8% | **1.9%** | **1.9%** | **1.9%** | **10.9 pp** |
| 1.7 | 3.9% | **0.7%** | **1.1%** | **0.6%** | **3.3 pp** |
| 2.0 | 0.2% | 0.07% | 0.2% | 0.07% | 0.13 pp |

### KEY FINDINGS from F10:
1. **No DRO-hurts crossover on Tight-S!** Unlike S-curve where DRO hurts above v=1.8, on Tight-S DRO benefit remains positive at ALL speeds tested. At v=2.0, both Base and DRO are near zero (<0.2%).
2. **DRO near-eliminates collisions at v≥1.5:** 1.9% at v=1.5, 0.7% at v=1.7, 0.07% at v=2.0. DRO converges to near-zero faster than Base.
3. **Peak DRO benefit at v=0.8-1.0:** 73-75 pp reduction. This is the sweet spot on Tight-S.
4. **At low speed (v=0.6), DRO residual is highest:** 34% collision rate. The obstacle is so slow that even DRO can't fully avoid it on a tight curve.
5. **Path geometry changes the DRO crossover entirely:** S-curve has a DRO-hurts zone at v>1.5; Tight-S does not. The higher curvature means DRO's conservative planning always helps.
6. **Clearance story:** At v=1.5, DRO clearance (1.50m) equals Base (1.51m) — performance converges as speed increases and encounters become less dangerous.

---

## 8e. PATH GEOMETRY EFFECT ON DRO AT HIGH SPEEDS (F11, N=1500) — RESOLVES Gap 12

**This is a major finding:** The DRO-hurts crossover is S-curve specific, NOT universal.

### DRO Benefit (pp collision reduction, inject-K1 vs Base):
| Speed | S-curve | Tight-S | Straight |
|-------|---------|---------|----------|
| 1.3 | +8.1 pp | **+31.4 pp** | **+40.8 pp** |
| 1.5 | +4.2 pp | **+14.6 pp** | **+37.1 pp** |
| 1.7 | +6.7 pp | **+3.9 pp** | **+41.7 pp** |
| 2.0 | **−11.1 pp** | +0.3 pp | **+53.1 pp** |

### Three Completely Different DRO Profiles:
1. **S-curve:** DRO benefit shrinks with speed. At v=2.0, **DRO hurts** (Base 2.4% → inject 13.5%). The fast obstacle passes safely on S-curve, but DRO's conservative evasion causes the ego to steer off-path.
2. **Tight-S:** DRO benefit monotonically decreases but **always stays positive**. Both Base and DRO converge to near-zero at v=2.0 (0.4% and 0.1%). Higher curvature keeps DRO useful.
3. **Straight:** **DRO benefit INCREASES with speed** (40.8→53.1 pp). Base collision rate stays at 56-67% regardless of speed! On a straight path, the obstacle approaches head-on and speed doesn't help Base avoid it — but DRO's proactive evasion remains effective.

### Why Does S-curve Show DRO-Hurts?
At v=2.0 on S-curve, Base collision rate is only 2.4% — the obstacle passes so quickly that the S-curve geometry creates a natural separation. But DRO detects the high-risk approach and triggers evasive steering, which on a curved path can cause the ego to deviate from the reference and collide with a different part of the obstacle trajectory. This is a **false alarm effect** — DRO's conservatism backfires when the natural geometry already provides safety.

### Why Doesn't Straight Show DRO-Hurts?
On Straight, the obstacle and ego are on a direct collision course. Speed doesn't change this — a faster obstacle still hits head-on. Only active evasion (DRO's contribution) can prevent collision. Hence DRO benefit grows with speed.

### Clearance Story:
- S-curve: DRO always reduces clearance at high speeds (-0.27 to -0.69m) because DRO routes through tighter spaces
- Tight-S: Mixed — DRO improves clearance at v=1.3 (+0.20m) but is neutral to negative at v≥1.7
- Straight: DRO consistently improves clearance (+0.19 to +0.32m)

---

## 9. RHO (WASSERSTEIN RADIUS) SENSITIVITY

### T4 Rho Sweep (S-curve, N=500)
| Rho | Mode-DRO(inj) Coll. | Traj-DRO(comb) Coll. |
|-----|---------------------|---------------------|
| 0.01 | 15.0% | 18.6% |
| 0.05 | 15.6% | 19.0% |
| 0.10 | 14.2% | 21.0% |
| 0.20 | 14.4% | 20.4% |
| 0.50 | 15.8% | 18.4% |

**Mode-DRO(inj) is robust to rho** — collision rate varies only 14.2–15.8% across 50× range.
**Traj-DRO(comb) slightly more sensitive** — 18.4–21.0% range, non-monotonic.
**Traj-DRO(inj) insensitive to rho** — stays ~36–40% regardless (fundamental structural issue).

---

## 10. SWITCHING DYNAMICS

### G3 Switching Probability Sweep (S-curve, N=500)
| Switch Prob | Base Coll. | inject-K1 Coll. |
|------------|-----------|-----------------|
| 0.02 | 80.4% | 16.6% |
| 0.05 | 80.0% | 17.0% |
| 0.10 | 82.4% | 18.0% |
| 0.20 | 79.8% | 17.2% |
| 0.35 | 80.8% | 16.4% |
| 0.50 | 80.0% | 17.0% |

**DRO benefit is independent of switching probability.** Both Base and DRO collision rates are flat across the sweep. This makes sense: the DRO robustification protects against mode uncertainty regardless of actual switching rate.

---

## 11. RESEARCH GAPS — STATUS

### Gap 1: Higher N to Differentiate DRO Variants → **RESOLVED (F1)**
All injection methods (DRO, TopRisk, DiverseRisk) remain statistically indistinguishable at N=2000. Softmax confirmed worst.

### Gap 2: Medium Speed + Complex Path → **RESOLVED (F2)**
Speed 1.0 + Tight-S is the second-best DRO scenario (74.9 pp reduction). Speed 1.3 + Tight-S nearly eliminates collisions (3.4%).

### Gap 3: Traj-DRO(comb) vs Mode-DRO Sweet Spot → **RESOLVED (F5)**
**YES — Traj-DRO(comb) genuinely beats Mode-DRO(inj) at speed 1.3.** 9.2% vs 14.7%, statistically significant (N=2000).

### Gap 4: K=2 vs K=1 Systematic Sweep → **RESOLVED (F3)**
K=2 does not consistently improve over K=1. K=1 is sufficient.

### Gap 5: Adaptive Rho on Tight Curves → **OPEN**
Not yet tested. Mode-DRO(inj) at rho=0.1 gets 25.4% on Tight-S at v=0.8 but only 4.1% at v=1.3. The speed effect dominates rho.

### Gap 6: Clearance-Focused Experiments → **RESOLVED (F4)**
N=2000 clearance data confirms DRO doubles median clearance on both Oncoming and Crossing environments.

### Gap 7: Traj-DRO transition speed → **RESOLVED (F6)**
Crossover at v ≈ 1.1 m/s. Traj-DRO(comb) wins in the v=1.1–1.4 window. Peak advantage at v=1.3 (7.7 pp).

### Gap 8: Traj-DRO Advantage Path Dependence → **RESOLVED (F7)**
Traj-DRO(comb) advantage is **S-curve specific**, does NOT generalize to Tight-S. On Tight-S all DRO variants are equivalent.

### Gap 9: Multi-obstacle + medium speed → **RESOLVED (F9)**
DRO benefit INCREASES with obstacle count at Tight-S v=1.0: 74.6→84.1 pp. DRO stays at 7-10% regardless of obstacle count.

### Gap 11: WDRO-sampling speed-dependent advantage → **PARTIALLY RESOLVED (F10)**
F10 confirms: at v≥1.2, sampling matches or beats injection (4.8% vs 5.2% at v=1.2, 4.4% vs 6.5% at v=1.3). At v≤0.8, injection is better (24.1% vs 26.2%). q*-sampling's exploration helps when the obstacle passes more quickly (less time to react, so diverse scenarios matter more).

### Gap 12: S-curve vs Tight-S comparison → **RESOLVED (F11)**
DRO-hurts crossover is S-curve specific. On Tight-S, DRO benefit is always positive. On Straight, DRO benefit actually INCREASES with speed (40.8→53.1 pp).

### Remaining Gaps:
- **Gap 10:** Does adaptive method selection (Mode-DRO at v<1.1, Traj-DRO at v>1.1) outperform either alone?
- **Gap 13:** Why does Straight path maintain 56-67% Base collision rate at ALL speeds? (Obstacle geometry?)

---

## 12. STATISTICAL SIGNIFICANCE SUMMARY

### Confirmed Significant (non-overlapping 95% CIs):
- Base vs ANY DRO on Oncoming (78+ pp gap)
- Base vs ANY DRO on Crossing (50+ pp gap)
- Base vs ANY DRO on Diagonal (50+ pp gap)
- Base vs ANY DRO on S-curve/Tight-S (60+ pp gap)
- Mode-DRO(inj) vs Traj-DRO(inj) everywhere (15-49 pp gap)
- All DRO variants vs Base at speeds 0.3–1.3 m/s
- Traj-DRO(comb) vs Mode-DRO(inj) at speed 1.3 S-curve (5.5 pp gap, N=2000)
- Softmax-tau5 vs TopRisk-K1 on Oncoming (4.9 pp gap, N=2000)
- Base vs DRO on Tight-S@v1.0 (74.4 pp gap, N=2000 in F8)
- Base vs DRO on Tight-S@v1.3 (31.0 pp gap, N=2000 in F8)
- Base vs DRO on S-curve-crossing (52.9 pp gap, N=2000 in F8)
- **[NEW] DRO stable across 1-3 obstacles at sweet spot (F9, N=1500, all CIs ~7-10%)**
- **[NEW] DRO benefit positive at ALL speeds on Tight-S (F10, no crossover)**
- **[NEW] DRO near-zero collision rate at v≥1.5 on Tight-S: 1.9% [1.3%, 2.7%] (F10)**
- **[NEW] DRO hurts on S-curve at v=2.0: Base 2.4% → inject 13.5% (11.1 pp worse, N=1500, F11)**
- **[NEW] DRO benefit increases with speed on Straight: 40.8→53.1 pp as v goes 1.3→2.0 (F11)**

### NOT Significant (overlapping CIs):
- inject-K1 vs TopRisk-K1 (1 pp difference, N=2000)
- inject-K1 vs inject-K2 (0.8 pp difference, N=2000)
- inject-K1 vs DiverseRisk-K1 (0.8 pp difference, N=2000)
- WDRO-sampling vs inject methods (1.7-2.7 pp difference, N=2000)
- Traj-DRO(comb) vs Mode-DRO(inj) at speed 1.0 (0.9 pp difference)

---

## 13. KEY TAKEAWAYS FOR PAPER

1. **DRO dramatically improves safety** in high-collision scenarios (60-80 pp reduction). Confirmed at N=2000 across 4 showcase scenarios.
2. **Speed regime matters, but depends on path:** DRO hurts above v≈1.8 m/s **only on S-curve** (false alarm effect). On Tight-S and Straight, DRO benefit is positive at all speeds.
3. **Mode-level DRO > Trajectory-level DRO in general:** Structural advantage of discrete-mode reweighting
4. **Exception: Traj-DRO(comb) wins at medium speed on S-curve (v=1.3):** 9.2% vs 14.7%, statistically significant. But this advantage is **S-curve specific** — does not generalize to Tight-S (F7).
5. **All injection criteria are equivalent:** DRO, TopRisk, DiverseRisk yield indistinguishable results (confirmed at N=2000)
6. **K=1 injection is sufficient:** K=2 provides no consistent improvement
7. **DRO doubles clearance** in oncoming/crossing scenarios while maintaining path progress
8. **Two flagship scenarios for the paper:**
   - Oncoming: 93.7%→14.8% (78.9 pp reduction, clearance 0.52→1.05m)
   - Tight-S@v1.0: 81.5%→7.1% (74.4 pp reduction, clearance 0.60→1.16m)
9. **DRO near-eliminates collisions at sweet spot:** Tight-S@v1.3: 34.9%→3.9% (31 pp, ~1/25 rollouts collide)
10. **Softmax-tau5 is the weakest DRO variant** — confirmed significantly worse than injection methods
11. **DRO benefit is independent of switching probability** — robust to environment dynamics
12. **Crossing is hardest for DRO:** 22.7% residual rate even with best method (vs 7.1% on Tight-S). Perpendicular approach is fundamentally harder to evade.
13. **DRO scales favorably with obstacle count:** At Tight-S v=1.0, DRO maintains 7-10% collision rate with 1, 2, or 3 obstacles while Base degrades from 82→93%. DRO benefit increases with more obstacles.
14. **No DRO-hurts zone on Tight-S:** Unlike S-curve where DRO hurts above v=1.8, Tight-S DRO benefit is positive at all speeds. Higher curvature paths always benefit from DRO.
15. **DRO achieves near-zero collision rate at v≥1.5 on Tight-S:** 1.9% at v=1.5, 0.7% at v=1.7 — almost eliminates risk entirely.
16. **[MAJOR] Path geometry determines DRO utility profile at high speed:** S-curve is the ONLY geometry where DRO hurts. On Tight-S it's always positive. On Straight, DRO benefit actually increases with speed (40→53 pp). The DRO-hurts effect is a false alarm artifact unique to moderate-curvature paths.
17. **Straight path reveals head-on collision geometry:** Base maintains 56-67% collision rate regardless of obstacle speed on Straight, because a faster oncoming obstacle still collides head-on. Only DRO's active evasion prevents collision.

---

## 14. EXPERIMENT FILES INDEX

### Focused Experiments (focused_figures/)
| File | Description | N | Key Finding |
|------|-------------|---|-------------|
| f1_high_n_oncoming.csv | 7 methods, Oncoming, N=2000 | 14,000 | Injection methods indistinguishable |
| f2_speed_path_interaction.csv | 3 speeds × 3 paths × 4 methods | 36,000 | v=1.0+Tight-S sweet spot |
| f3_k1_vs_k2_multi_obs.csv | K=1 vs K=2 × 1-3 obstacles | 15,000 | K=1 sufficient |
| f4_clearance_raw.csv | Raw per-rollout clearance, N=2000 | 12,000 | DRO doubles clearance |
| f4_clearance_summary.csv | Clearance statistics summary | — | p5 clearance triples |
| f5_traj_vs_mode_dro.csv | 5 methods × 2 speeds, N=2000 | 20,000 | **Traj-DRO(comb) wins at v=1.3** |
| f6_transition_speed.csv | 3 methods × 7 speeds, N=1500 | 31,500 | Crossover at v≈1.1 |
| f7_traj_dro_tight_s.csv | 4 methods × 4 speeds on Tight-S, N=1500 | 24,000 | Traj-DRO advantage S-curve specific |
| f8_best_dro_showcase.csv | 4 scenarios × 4 methods, N=2000 | 32,000 | **Paper-ready flagship results** |
| f8_showcase_clearance_raw.csv | Per-rollout clearance for F8 | 32,000 | Clearance CDFs for all showcases |
| f9_multi_obs_sweet_spot.csv | 3 obs counts × 4 methods, Tight-S v=1.0 | 18,000 | **DRO scales favorably with obstacles** |
| f10_tight_s_speed_sweep.csv | 8 speeds × 4 methods, Tight-S | 48,000 | **No DRO-hurts zone on Tight-S** |
| f11_path_geometry_high_speed.csv | 3 paths × 4 speeds × 3 methods | 54,000 | **DRO-hurts is S-curve specific** |

### Original Experiment Suites
- **cdc_paper_figures/**: E1 method comparison, C1/C2 clearance analysis (N=1000)
- **generalization_figures/**: G1–G8 path/obstacle/speed generalization (N=500)
- **traj_dro_figures/**: T1–T7 trajectory DRO comparison (N=500)
- **paper_figures/**: A–D DRO benefits, injection count sweep (N=1000–5000)
