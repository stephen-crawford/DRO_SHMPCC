# Returned-plan Gaussian collision diagnostic

The next implementation follows the a-posteriori branch of the proposal. It
computes mode-conditioned bounds from a **fixed returned plan**, not from the
surrogate reference geometry used to choose WDRO weights. It never changes the
controller, ambiguity set, constraints, scenario count, or issued certificates.
Trust-region constraints, pre-solve uniform bounds and adaptive stopping remain
unimplemented: those are separate mathematical/algorithmic changes.

## Calculation and event

For each supplied future step, ego disc center c, and mode-conditioned Gaussian
position X~N(mu,Sigma), choose n=(mu-c)/||mu-c||. At exactly coincident means use
the deterministic unit vector (1,0). The strict event ||X-c||<R implies
R-n·(X-c)>0 for either direction choice. The tool computes the projected mean
**R-n·(mu-c)** and variance n' Sigma n, then the Gaussian upper tail at zero.
With variance zero the probability is exactly 1 if the projected mean is
positive and 0 otherwise. Thus deterministic equality at the radius is excluded
under this explicitly strict event. Counting contact as collision would require
a different degenerate-boundary convention.

The per-mode upper bound is min(1,sum of disc/step bounds). No independence
across discs or times is needed for the union bound. Gaussian marginals, correct
conditioning, and the actual final disc geometry are required. Input covariance
matrices must be finite, symmetric and positive semidefinite; invalid matrices
are rejected rather than silently repaired. The certificate concerns supplied
**discrete future times**, not intersample collision or closed-loop replanning.

The tool reports two different quantities:

1. `direct_cp_bound = max_{p in CP vertices} p·b`, which needs no Q-risk theorem.
2. `transferred_bound = Psi_theta(epsilon_SH(S,total_support,beta_cert))`, only
   if the snapshot explicitly supplies actual q, sample count, a valid total
   support bound, `scenario_theorem_eligible: true`, and an eligibility
   justification. Production `RuntimeConfig::degroot_violation_risk` is called
   unchanged. Removed IDs must already be included in the total support bound;
   they are not added again.

The eligibility flag records the caller's assumption, not a proof checker.
The direct calculation uses beta_cp; a valid scenario transfer additionally
spends beta_cert. Both remain conditional on valid CP history assumptions and
mode-conditioned predictive laws. Floating-point evaluation/LP enumeration is
not verified interval arithmetic. Surrogate risk scores are never substituted
for b. No S_SH/S_WDRO comparison is emitted, and
`presolve_sample_reduction_claim` is always false: final-plan bounds do not
justify choosing fewer scenarios before obtaining that plan.

## Snapshot interface

```json
{
  "obstacles": 1,
  "collision_radius": 0.95,
  "epsilon": 0.05,
  "beta_cp": 0.05,
  "beta_cert": 0.01,
  "scenario_theorem_eligible": false,
  "modes": [
    {"mode": "away", "count": 900, "p_hat": 0.9, "q": 0.8},
    {"mode": "crossing", "count": 100, "p_hat": 0.1, "q": 0.2}
  ],
  "steps": [
    {
      "k": 1,
      "disc_centers": [[0.0, 0.0]],
      "gaussians": {
        "away": {"mean": [3.0, 0.0], "covariance": [[0.1, 0.0], [0.0, 0.1]]},
        "crossing": {"mean": [1.0, 0.5], "covariance": [[0.2, 0.0], [0.0, 0.2]]}
      }
    }
  ]
}
```

This is a schema example, not experimental evidence. Supply all discs and all
future stages of the plan for a full-horizon claim. Only one obstacle and 2–6
held modes are currently supported. Actual q may be omitted for the direct
bound; missing q prevents scenario transfer. For eligible transfer also provide
`sample_count`, `total_support_bound`, and `scenario_theorem_justification`.

```bash
cmake --build build-certificate --target certificate_numeric_backend -j 2
python3 tools/analyze_posterior_certificate.py \
  --snapshot snapshot.json --output results/posterior-snapshot
```

Outputs include the original snapshot, per-disc/step projected means and
variances, zero-distance/variance flags, mode b/L/U bounds, conditional posterior
results and code/backend hashes. Existing output directories are refused.

## Evaluation on saved production plans

The fixture adapter is specifically for `tools/rare_mode_experiment.cpp`:
three ego discs at longitudinal offsets -1,0,1 (length 2), R=.5+.35+.1=.95,
20 future stages, and deterministic obstacle modes. Zero covariance is part of
that fixture's model, not an estimate inferred from lack of logged noise.

```bash
python3 tools/evaluate_frozen_posterior.py \
  --fixture build-rare-mode/rare-pilot --output results/posterior-rare-pilot
```

It reads actual accepted trajectories from `plans.csv`, Gaussian means from
`scene.csv`, and counts/p_hat from `weights.csv`. It includes rejected/error
trials in the summary as `no_accepted_plan` rather than selecting them away.
Only the cut-in component of actual per-attempt q was previously recorded, so
it does not invent the remaining q components from the frozen-reference law.
It leaves scenario transfer unset. Synthetic fixed history counts also do not
establish iid statistical coverage; all CP interpretations are conditional.

Executed results under `build-certificate/posterior-rare-pilot`:

- 96 trials, all with accepted saved plans.
- 13 plans: cut-in b=0; 83 plans: cut-in b=1. The other modes had b=0.
- Corresponding direct CP bounds were 0 or 0.02037248921162961.
- All 24 plans in each of the four arms met the 0.05 direct-bound threshold.
- These results do not distinguish WDRO as safer. They concern this deterministic
  fixture and its assumed history model, not general Gaussian closed-loop safety.

The adapter's `posterior_summary.csv` retains every trial and plan outcome;
individual plan directories contain full diagnostic evidence and snapshots.
Input CSV hashes are saved in `source_hashes.json`.

## Validation and status

Build: `cmake --build build-certificate --target certificate_numeric_backend -j 2`.
Tests: `python3 tests/test_posterior_certificate.py` and
`python3 tests/test_wdro_certificate.py` — **3+3 tests passed**.
Tests cover deterministic strict-boundary behavior, zero-distance projection,
invalid covariance rejection, transfer eligibility gating, and absence of a
pre-solve reduction claim. A seeded 100,000-sample Gaussian test checked
pointwise containment of the collision event in the projected event and its
analytic probability. It is a regression test, not proof by simulation.
All 96 saved deterministic-plan nominal violation masses from `plan_risk.csv`
were bounded by both the computed per-mode geometry and direct CP diagnostics.
No controller mathematical guarantee or formulation was modified. The only
existing-code addition is an offline backend command exposing the unchanged
production risk calculation. Candidate diagnostic implemented and tested;
behavior requires user verification.
