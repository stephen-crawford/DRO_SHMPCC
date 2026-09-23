# Offline WDRO → true-law risk transfer

For bounds recomputed from a returned trajectory, see
[Posterior Gaussian collision diagnostic](POSTERIOR.md). That extension keeps
posterior assessment separate from pre-solve sample-count claims.

This diagnostic implements the requested first milestone for **one obstacle and
2–6 held modes**. It does not change controller sampling, issued certificates,
confidence bounds, solver acceptance, or Lean proofs. The result is an offline
floating-point calculation under explicitly stated assumptions, not a new safety
certificate.

## Probability model

Condition on information available **before** drawing the scenarios for a solve.
Let P* be the unknown true categorical law and Q=q the frozen sampling law. For
a returned plan theta, v_m is its probability of violating the existing joint-
horizon safety event conditional on mode m. The within-mode trajectory/noise
law must be identical under P* and Q. Then V_P*(theta)=sum p*_m v_m and
V_Q(theta)=sum q_m v_m. The scenario theorem must independently justify its
Q-risk statement for the actual support, removal and optimization procedure.
Q depending on these same scenario draws would need additional justification.

Categorical counts define a simultaneous Bonferroni/Clopper–Pearson box,
intersected with the probability simplex. CP coverage assumes iid stationary
categorical observations. Switching, adaptive, temporally dependent, or pooled
class histories do not automatically meet that assumption. An observed mode
frequency is not proof of the true-law model.

For multiple obstacles, joint P* and Q cannot be inferred by merely multiplying
marginals without conditional independence of obstacle modes AND appropriate
within-mode coupling assumptions. Shared histories do not establish that
independence. Marginal confidence events also need a simultaneous allocation
across obstacles/classes. This first tool rejects multiple obstacles rather
than silently imposing those assumptions or treating marginal CP vertices as
joint-law vertices. Extension is deferred until a justified single-state
sample-reduction example exists.

## Calculation

For every vertex p of the box/simplex intersection, the inner program maximizes
p·v subject to q·v <= eta and 0 <= v <= b. This one-budget LP is solved as a
fractional knapsack: fill q=0 modes without consuming budget, then fill remaining
modes in decreasing p_m/q_m order. Maximization over CP vertices gives Psi(eta).
A deterministic 64-iteration bisection finds the feasible side of its inversion.
If Psi(0)>epsilon, no allowable eta exists, and S_WDRO is blank. Zero q entries
are not clipped or replaced by arbitrary floors.

The default is **b_m=1**. Then the constant vector v_m=eta is feasible, so
Psi(eta)>=eta. Thus **eta_star cannot exceed epsilon** with only these bounds.
A sample-count improvement requires additional justified, mode-conditioned
upper bounds. The surrogate risk r used to choose q is not automatically such a
bound. Supplying b<1 requires `b_justification`; the text records an assumption,
not a machine-checked proof. A bound proved only for one observed plan also
requires justification before using it to dimension a different reduced-S solve.

The C++ adapter calls the existing production CP and de Groot functions unchanged.
It discards the auxiliary unit-ground-cost radius used when requesting CP bounds:
L/U depend on counts and beta, not that auxiliary geometry. The snapshot's actual
rho is retained separately. No Python approximation substitutes for production
sample sizing. The helper checks the returned S meets the risk threshold and
exports the risk at S-1 to inspect minimality. Reaching the production search cap
without satisfying the target is an error.

The same beta_cert is used for baseline and WDRO counts as requested. A valid
transfer adds beta_cp via the union bound, so these are **not equal-total-
confidence comparisons** with an ordinary baseline that spends only beta_cert.
The output exposes beta_cp, beta_cert and their sum. Simultaneous guarantees
across time or data-selected snapshots would need additional accounting.

## Support/removal audit

`RuntimeConfig::compute_required_scenarios` uses total cap n_bar+R (defaults 6+2=8).
In `MPCController::solve_optimization_sqp`, the support set starts with
`pre_support_scenarios`; callers pass the removed-scenario IDs. Active or violated
scenario IDs are then unioned over SQP iterates. The exported support size on
that path already includes removed scenarios; **do not add R again to that
measured union**. Dimensioning here uses configured 6+2, not an inferred measured
support count. Branches that clear removal/rebuild recovery need their own
certificate eligibility check; existing controller restrictions are unchanged.

The snapshot may include `observed_support` when its provenance is known. If it
is absent it remains null; rho, count, or final active constraints are not used
to invent n_hat. No removal-accounting proof is claimed by this source audit.

## Run

```bash
cmake -S . -B build-certificate -DDRO_MPC_ENABLE_ROS2=OFF \
  -DACADOS_ROOT=/home/stephen/Documents/ACC_Development/Development/acados
cmake --build build-certificate --target certificate_numeric_backend -j 2
python3 tools/analyze_wdro_certificate.py \
  --frozen-weights build-rare-mode/rare-pilot/weights.csv \
  --output results/rare-mode-certificate-diagnostic
```

For a supplied snapshot, use `--snapshot snapshot.json` instead. Example schema:

```json
{
  "obstacles": 1,
  "epsilon": 0.05,
  "beta_cp": 0.05,
  "beta_cert": 0.01,
  "nonremoved_support_cap": 6,
  "removal_budget": 2,
  "observed_support": null,
  "rho": 0.04,
  "modes": [
    {"mode": "mode_a", "count": 900, "p_hat": 0.9, "q": 0.8},
    {"mode": "mode_b", "count": 100, "p_hat": 0.1, "q": 0.2}
  ]
}
```

Counts must be actual observations, not rounded pseudo-counts reconstructed from
p_hat. p_hat and q are distinct inputs; neither is renormalized by this tool.
This schema is illustrative, not an empirical result. The frozen-weights adapter
reads the saved production rare-mode fixture (counts 90/10/900); it represents
a frozen geometric reference, not an issued closed-loop certificate.

Outputs: `snapshot.json`, `mode_bounds.csv`, `certificate_summary.csv`, detailed
`certificate_diagnostic.json`, and source/executable/script hashes in
`provenance.json`. Existing output directories are refused.

## Executed validation and current result

Build succeeded in the separate build directory. Executed:

```bash
python3 tests/test_wdro_certificate.py
python3 tools/analyze_wdro_certificate.py \
  --frozen-weights build-rare-mode/rare-pilot/weights.csv \
  --output build-certificate/rare-mode-transfer-final
```

Three tests passed: 30 random inner LP comparisons against independent SciPy
HiGHS solutions (including zero-q modes), conservative-b/infeasible-transfer
cases, and production CP/sample-size checks including S-1. A synthetic example
with an explicitly assumed zero-risk mode yielded eta=0.202797 and S ratio
0.169190. This is a software test, **not evidence about the real controller**.

The saved three-mode production fixture, using b=1, yielded:

| Quantity | Observed diagnostic |
|---|---:|
| epsilon_Q_max | 0.039662359007474085 |
| S_SH at epsilon=.05, beta_cert=.01, total support=8 | 1123 |
| S_WDRO | 1484 |
| S_WDRO / S_SH | 1.3214603739982191 |
| 30% reduction threshold met | No |

The current result is **no-go for claiming reduced certified sample complexity**
from this fixture. Better mode inclusion alone is insufficient. Need justified
conditional bounds and valid confidence/history/support assumptions before
claiming an improvement or changing the controller. Floating-point vertex
feasibility uses 1e-12 tolerance and is not verified interval arithmetic.
Candidate diagnostic implemented and tested; behavior requires user verification.
