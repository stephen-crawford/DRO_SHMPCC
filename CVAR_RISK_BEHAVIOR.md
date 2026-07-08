# CVaR risk swap — behavioral result (branch `cvar-risk`)

**Change.** The one-sided directional risk in `WassersteinDRO::compute_risk_vector`
previously used the **VaR** (quantile) coefficient `z_alpha = Phi^{-1}(alpha)`.
On this branch it uses the **CVaR** (expected-shortfall) coefficient

```
k_alpha = phi(Phi^{-1}(alpha)) / (1 - alpha)
```

implemented as `cvar_coefficient(alpha)` in `src/wasserstein_dro.cpp`. The risk
formula is otherwise unchanged:

```
r_kd = max(0, (safety_radius + coeff * sigma_dir) - dist)
```

At `alpha = 0.95`: VaR coeff `1.645` -> CVaR coeff `2.063` (a **1.25x** margin inflation).

## Measured behavior (micro_dro_probe, canonical 6-mode scenario)

Fast deterministic probe (`tests/micro_dro_probe.cpp`): obstacle at (5,0), ego
driving toward it so along-horizon distances land near the safety margin (the
regime where VaR and CVaR actually differ). Full experiment suites are
impractical for before/after (heavy runners time out at >280s); this probe runs
in milliseconds.

| mode | VaR risk | CVaR risk | delta |
|---|---|---|---|
| decelerating      | 0.7860 | 0.8561 | +0.070 |
| turn_left/right   | 0.6773 | 0.7470 | +0.070 |
| constant_velocity | 0.6757 | 0.7457 | +0.070 |
| lane_change_l/r   | 0.5351 | 0.6052 | +0.070 |
| **worst_case_risk** | **0.7860** | **0.8561** | **+0.070** |

**Q\* (worst-case weights): identical — 100% on `decelerating` in both cases.**
`optimal_lambda`, `rho_used`, and `implied_transport_cost` are unchanged.

## Interpretation

Swapping VaR -> CVaR is a (near-)**uniform additive inflation** of the risk
vector by `(k_alpha - z_alpha) * sigma_dir`. Because the shift is monotone and
roughly mode-independent (the `sigma_dir` values are similar across modes here),
it **preserves the risk ranking**, so the LP reweighting selects the **same
worst-case mode** and `Q*` is invariant. The only downstream effect is a more
conservative absolute `worst_case_risk` (0.786 -> 0.856, +9%), which inflates
constraint tightening but does **not** change which mode the planner robustifies
against.

This empirically confirms, in the actual CDC C++ codebase, the analysis recorded
as Q38 in `wiki/concepts/paper-a-concepts-qa.md`: the reweighting is
LP-structural and invariant to a monotone risk-measure change. CVaR would only
change `Q*` if per-mode `sigma_dir` were heterogeneous enough that the larger
CVaR multiplier reordered modes — not the case for these mode models.

**Caveat.** Unlike the collision-cost `1/(dist+floor)` model used in the Python
template (where CVaR-of-cost != margin-shift and Jensen/saturation effects
appear), the CDC risk here is a *directional margin* on distance, so the clean
`z_alpha -> k_alpha` coefficient swap is the correct CVaR reformulation with no
inverse-cost pathology.

See `CVAR_RISK_UPDATE.md` for the future-CDC-paper text action item.
