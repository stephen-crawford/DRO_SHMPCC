# Confidence-calibrated ambiguity radius (branch `radius-calibration`)

**Change.** `WassersteinDRO::get_adaptive_rho()` gains an opt-in
(`DROConfig::use_calibrated_radius`, default **off**) confidence-calibrated
radius that replaces the ad-hoc heuristic

```
rho = rho_base * (1 + alpha/sqrt(n)) * (1 + gamma * H/H_max)     [legacy]
```

with the **simplex-concentration** radius

```
rho_n(beta) = rho_min + rho_base * sqrt( 2 (N ln2 + ln(1/beta)) / n )   [calibrated]
```

`N` = number of modes (`exp(max_entropy_)`), `n` = observed interactions
(`observation_count_`), `beta` = target miscoverage (`confidence_beta`, default
0.05 → 95% coverage).

**Derivation / guarantee.** The nominal belief `p_hat` is an empirical
categorical distribution over `N` modes from `n` interactions. It concentrates in
L1 as `P(||p_hat - p*||_1 >= eps) <= 2^N exp(-n eps^2/2)` (Devroye), so at
miscoverage `beta` the half-width is `eps_n = sqrt(2(N ln2 + ln(1/beta))/n)`. With
a metric ground cost, `W1(p_hat,p*) <= diam * eps`, the diameter folded into
`rho_base`. Then the **true belief `p*` lies in the ball with probability
>= 1 - beta**, so the reweighted worst-case risk upper-bounds the true risk at
confidence `1 - beta`. This is the confidence-calibrated form of the paper's
`1/sqrt(n)` shrink; because the CDC object is the finite **mode simplex** (not a
continuous density), the correct rate is the parametric `sqrt(N/n)` — no
Fournier–Guillin curse.

## Measured behavior (`tests/radius_probe.cpp`, canonical 6-mode scenario, beta=0.05)

| n_obs | HEUR rho | HEUR wc_risk | HEUR Q\* L1 | CALIB rho | CALIB wc_risk | CALIB Q\* L1 |
|---|---|---|---|---|---|---|
| 1     | 0.3000 | 0.7860 | 1.667 | 0.3883 | 0.7860 | 1.667 |
| 2     | 0.2561 | 0.7860 | 1.667 | 0.2775 | 0.7860 | 1.667 |
| 5     | 0.2171 | 0.7760 | 1.483 | 0.1792 | 0.7625 | 1.333 |
| 10    | 0.1974 | 0.7692 | 1.357 | 0.1296 | 0.7435 | 1.219 |
| 25    | 0.1800 | 0.7628 | 1.333 | 0.0857 | 0.7238 | 0.860 |
| 100   | 0.1650 | 0.7573 | 1.333 | 0.0478 | 0.6992 | 0.469 |
| 1000  | 0.1547 | 0.7535 | 1.333 | 0.0220 | 0.6732 | 0.203 |
| 10000 | 0.1515 | 0.7523 | 1.333 | 0.0138 | 0.6637 | 0.127 |

## Does it significantly change the results?

**Yes — the degree of conservatism, especially in the data-rich regime. No — the
qualitative reweighting structure.**

- **The heuristic never becomes confident.** Its radius plateaus at
  `~rho_base * h_term ≈ 0.15` no matter how many interactions are observed
  (0.15 even at n=10000), and its worst-case reweighting saturates at Q\* L1 =
  1.333. This is a genuine flaw: infinite data buys no reduction in conservatism.
- **The calibrated radius is statistically consistent.** It shrinks toward
  `rho_min` as `n → ∞` (0.39 → 0.014 across the sweep), and the worst-case
  distribution relaxes toward nominal (Q\* L1 1.67 → 0.13). More data → tighter
  ball → less conservative, as it should be.
- **Crossover at n ≈ 3–5.** For very few observations (n ≤ 2) the calibrated
  radius is *larger* (correctly more cautious when the belief is barely
  estimated); from n ≳ 5 it is smaller, and the gap widens fast: at n=100 it is
  3.4× smaller (0.048 vs 0.165), at n=1000 **7× smaller** (0.022 vs 0.155),
  producing **6.6× less reweighting** (Q\* L1 0.20 vs 1.33) and an 11% lower
  worst-case risk (0.673 vs 0.753).
- **Structure unchanged.** The worst-case mode and the risk ranking are identical
  under both radii — only the *magnitude* of the reweighting changes. This is
  consistent with the LP-structural invariance established earlier (the radius
  sets how much mass moves, not which mode receives it).

**Net:** the swap is a significant, well-motivated improvement in the data-rich
regime — it fixes the heuristic's "never gets confident" pathology and gives a
finite-sample coverage interpretation — while leaving small-data behavior similar
(both large/clamped) and the qualitative worst case untouched. Default behavior
is preserved (`use_calibrated_radius=false`), so existing experiments are
unaffected until the flag is set.

**Caveat on the constant.** The *rate* (`sqrt(N/n)`) and the `N`/`beta`
dependence are now principled; the absolute scale is still carried by `rho_base`
as a proxy for the ground-metric diameter. For a fully rigorous constant, set
`rho_base` to the actual Bures ground-cost diameter `max_ij C_ij` per scenario,
or cross-validate it against a target out-of-sample violation rate. The sharper
Robust Wasserstein Profile Inference radius (`~ chi2_{1-beta}/n`, O(1/n)) is the
next step if an even less conservative, optimizer-covering calibration is wanted.
