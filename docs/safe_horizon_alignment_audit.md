# Safe Horizon MPC alignment and validation

Status: candidate implementation tested; behavior requires user verification.
No proof of equivalence to the authors' complete solver implementation is claimed.

## Specification

The authority is the supplied
`/home/stephen/Documents/publications/SafeHorizonMPC.pdf`,
*Scenario-Based Motion Planning with Bounded Probability of Collision*,
arXiv:2307.01070v1, 3 July 2023. Its SHA-256 is
`0238d9646fbdba23404eb5c9b9c4286e29eb44623c4aebc1ed053e6a41e74ba0`.

The user explicitly instructed that unspecified details should use the existing
framework. Accordingly, RK4 dynamics, state/input ordering, path projection,
fixed collision normals, disc geometry, QP construction and regularization,
acados/HPIPM and deterministic ID ordering remain in use. Following the
certificate diagnostic, the user approved replacing ordinary SH sampled
lateral/DR anchor preparation with a frozen pre-sampling reference/backup
anchor. The hybrid extension retains its sampled preparation. These choices
do not establish identity to FORCES Pro or to a
particular revision of the authors' public module. The optional hybrid homotopy
recovery and DRO transfer/calibration remain extensions.

## Changes and equation mapping

Subsequent source inspection found additional differences from the local
`mpc_planner` / `scenario_module` checkouts. See
[the reference implementation comparison](reference_implementation_alignment.md).
Those checkouts postdate the supplied preprint. In particular, the existing
anchor routine is not arithmetically identical to their DR operator, and their
shared slack/progress dynamics and cap handling differ. The numerical comparison
does not establish publication/controller equivalence.

| Paper requirement | Candidate implementation |
| --- | --- |
| Eq. (8), bisection of the support risk bound | Existing `RuntimeConfig::degroot_violation_risk()` and integer bisection are preserved. |
| Eqs. (13)–(14), support union over feasible convex iterations | `solve_optimization_sqp()` reads residuals `C*delta_u-d` at the actual QP primal solution and accumulates active joint scenario IDs. Removed IDs remain in support. Nonlinear activity is diagnostic only. |
| Assumptions 2–3, convex and inequality-feasible iterations | Squared-term objective weights must be finite and nonnegative. Counted QPs must converge and satisfy all inequality and box rows to the existing QP tolerance. Failed QPs cannot produce a certificate merely by returning a finite vector. |
| Algorithm 1 lines 9–11 / Theorem 3 | Save eligible nonlinear trajectories, controls, support, removal sets, collision rows, and iteration counts. On cap exceedance, restore the saved iterate; if none exists, try the existing slowing fallback without certification. |
| Eq. (24), cumulative active/infeasibility removal up to R | Ordinary SH solves remove joint scenarios during SQP. Candidate IDs are processed in sorted order, which the PDF does not specify. Pre-solve shifted-backup removal is retained only for the optional hybrid extension. |
| Theorem 5 | Every removed joint ID is included in support, including infeasibility-causing removals. Returning an earlier iterate restores its own removal set. |
| Eqs. (26)–(31), halfspaces and polygon reduction | All original source halfspaces remain available. Reducing again after removal exposes previously redundant facets. Without removal, reuse the facets valid on the same reachable domain. Validate returned nonlinear trajectories against every retained source halfspace. |
| Convex iteration support interpretation | Ordinary SH uses full SQP steps instead of sample-dependent collision backtracking. The latter would introduce scenario dependencies outside the counted QP active set. Non-SH controllers retain their existing line search. |
| Sec. VIII-C / IX-B and Table II | `configs/safe_horizon_paper.yaml` selects N=20, prediction dt=0.2, execution dt=0.05, at most 15 iterations, R=1, total cap 9, epsilon=0.05, beta=0.01, S=1237 and the Gaussian weights: velocity 0.05, angular input 0.05, contour 0.001, lag 0.1. It disables the extra progress reward. Unlisted objective terms remain framework choices. |

The public configuration uses a **nonremoved** support cap, so the paper profile
sets 8+R=9. Setting 9+R would instead size for support 10. Existing experiment
configurations retain their chosen parameter values; the new profile provides
the paper's reported controller settings without rewriting unrelated experiments.

The four Table II weights can now be loaded from YAML and are written to the
resolved configuration, so artifact replay preserves them.

## Mathematical impact and limits

This change affects mathematical behavior and was made only after explicit
approval to match the supplied paper and retain framework details it omits.
The changes affect support estimation, online removal, cap termination,
SQP stepping, and admission of returned plans. The sample-complexity equation,
DRO confidence formulas, ambiguity construction and transfer derivation were
not changed.

The PDF uses exact equality for active constraints and exact inequalities for
feasibility. Numerical checks retain the existing activity tolerance 0.001 and
QP tolerance (default 0.0001). The previous 0.01 collision-admission threshold
is replaced by the QP tolerance on SH solves. SH trajectories are checked at
that same numerical tolerance for road, input and velocity constraints.
Controls are not saturated afterward to create a different trajectory than the
counted SH iterate. Non-SH velocity saturation remains in place. These are
floating-point implementations of the formulation, not exact-arithmetic proofs.

The global sample-feasibility assumption in Assumptions 1 and 4 remains
conditional on the chosen uncertainty model and available removal budget. A
finite removal budget is not a proof that every possible sample extraction is
feasible. Infeasible plans and admitted braking fallbacks are never certified.
Execution and prediction intervals are now separate: the paper profiles execute
at 0.05 s and predict at 0.2 s. Existing profiles inherit prediction dt for
execution. Ego and obstacle plant models use execution dt; controller forecasts
retain prediction dt. Fractional-step warm starts sample the previous held
controls on the new grid and re-roll from the actual measured state. This is a
numerical initialization choice, not a recursive-feasibility proof.

The opt-in `gaussian_position` model implements Eqs. (34)/(36) using the existing
four-state mode interface: position increments are `(v + noise)*dt`, velocities
remain fixed, and increment covariance is `dt^2*Sigma_w`. The static mode has
zero velocity. `configs/safe_horizon_paper_static.yaml` selects Eq. (34), the
static Table II weights, stationary plant obstacle (6,0), and radii 0.325/0.3.
Its `obstacle_position_noise_variance: 0.3` follows the literal diagonal entries
of Sigma_w in IX-C. Eq. (35) uses sigma_w squared and omits dt squared; its
marginal-baseline convention is not silently substituted into the SH joint
trajectory model. This ambiguity prevents a claim of identical experimental
covariance across all baselines. The default predictor and all existing mode
catalog entries retain their original acceleration-noise behavior.

When ordinary SH cannot return an eligible iterate, its uncertified slowing
fallback is checked against every original sampled disc obstacle, including
removed scenarios, with zero collision tolerance and the existing hard limits.
The initial reference's tangent planes can exclude a physically admissible
braking trajectory; those planes no longer determine this fallback's admission.
Fallbacks remain uncertified. Synthetic row-only tests and the optional hybrid
extension retain their prior fallback halfspace checks.

## Build prerequisite

Commit `8439eb43` deleted `src/collision_constraints.cpp` while CMake still
required it. A fresh configure failed with `Cannot find source file:
src/collision_constraints.cpp`. The source was restored byte-for-byte from
`HEAD^` (`61aaa7de`), without modifying its geometry. Its SHA-256 is
`4c36a804a6b8cce43721fa529903d5d1bf373d840b7afecbc056387faf0058ee`.
The unchanged source reports warnings for unused `start` and unused
`project_state_to_collision_halfspace`; no unrelated cleanup was included.

## Validation

Build directory: `/tmp/dro-shmpcc-paper-audit`.

```sh
cmake -S . -B /tmp/dro-shmpcc-paper-audit \
  -DACADOS_ROOT=/home/stephen/Documents/ACC_Development/Development/acados \
  -DDRO_MPC_ENABLE_ROS2=OFF
cmake --build /tmp/dro-shmpcc-paper-audit -j2 --target \
  test_safe_horizon_algorithm test_velocity_bounds experiment_runner \
  test_experiment_artifacts test_distribution_transfer test_paper_formulation \
  test_collision_halfspaces test_sample_complexity test_safe_horizon_configuration \
  test_support_accounting test_pruning_and_linearization test_mpcc_progress \
  test_sh_anchor_preparation test_bounded_braking test_plan_acceptance \
  test_dro_fallback test_acados_reproducibility
ctest --test-dir /tmp/dro-shmpcc-paper-audit --timeout 60 \
  -R '^(test_safe_horizon_algorithm|test_sample_complexity|test_safe_horizon_configuration|test_collision_halfspaces|test_pruning_and_linearization|test_sh_anchor_preparation|test_bounded_braking|test_support_accounting|test_distribution_transfer|test_paper_formulation|test_velocity_bounds|test_mpcc_progress|test_plan_acceptance|test_experiment_artifacts|test_acados_qp_solver|test_dro_fallback|test_acados_reproducibility)$' \
  --output-on-failure
```

Build succeeded. The 17-test run reported 13 passing, three failing, and one
60-second timeout. Full output: `/tmp/shmpc-alignment-final-tests.log`.
No existing test assertions or expectations were changed.

- `test_pruning_and_linearization`: existing failure, expects excess facets to
  throw, but the unchanged restored geometry retains all facets.
- `test_mpcc_progress`: existing failure, expects reference-speed independence,
  while the existing objective tracks reference velocity.
- `test_support_accounting`: three expectations fail because its deliberately
  infeasible obstacle scene no longer counts violated nonlinear rows as feasible
  convex support. Its sample-count, unique-ID and zero-support checks pass.
- `test_plan_acceptance`: timed out after 60 seconds; its full behavior was not
  validated. An earlier unbounded run was interrupted while this test ran.
- Passing tests include distribution transfer, the existing DRO paper
  formulation, optional hybrid DRO fallback, acados reproducibility, velocity
  bounds, artifacts, sample sizing, halfspaces, anchors and braking.

The new test exercises the actual private SQP loop with deterministic rows:

- R=1 removes scenario 11 (x1<=0.6) and exposes scenario 12 (x1<=1.1).
  Returned x1=1.1; support is {11,12}; nonlinear rollout reproduces controls.
- Total cap=1 returns the saved first iterate: x1=0.6, support={11}, no removals,
  returned iteration=0 and one counted QP. Controls and trajectory are compared
  directly with a separate one-iteration solve.
- No saved eligible iterate triggers the slowing fallback on cap exceedance.
- An unreachable row x1>=100 is rejected with R=0; R=1 removes its joint
  scenario and keeps it in support.
- A finite unconverged QP (status=2, one iteration, primal residual=0.692677)
  does not obtain support certification.
- An offset disc's feasible QP requires theta>=0.3, while its nonlinear
  y=sin(theta) violates the row by 0.00446851. Its QP support is counted but the
  nonlinear plan and infeasible slowing fallback are rejected.
- The paper profile obtains S=1237 and the Table II weights. Negative squared
  weights are rejected to preserve convex QP construction.

Sample-complexity checks also reported S=288 and S=1248 for the illustrative
cases rounded to about 290 and 1250 in the PDF; all existing assertions passed.

## Rollout evidence

An eight-decision requested smoke fixture used the paper profile, seed 77,
a stationary obstacle at (6,0), one ego disc of radius 0.325, obstacle radius
0.3, and zero plant noise. It retained the existing stochastic predictor.
Commands and outputs are recorded separately from unit-test evidence:

```sh
/tmp/dro-shmpcc-paper-audit/experiment_runner \
  --config /tmp/shmpc-paper-smoke.yaml --seed 77 \
  --output /tmp/shmpc-paper-rollout --label paper-velocity-tracking \
  --no-gif --no-rviz
```

With the framework's extra progress reward enabled, the first decision was
rejected: final nonlinear halfspace violation 1.01233, braking halfspace
violation 5.99542, no propagated steps and no recorded collision. This is why
the paper settings profile selects reference-velocity tracking with progress
reward disabled.

With progress reward disabled, one decision executed and the next was rejected:
final nonlinear halfspace violation 0.000194014 exceeded the 0.0001 tolerance;
braking halfspace violation was 4.66951. The harness reported
`no_admissible_control`, with no recorded collision. This result demonstrates
rejection, not completion of the obstacle task or a probability guarantee.
Full output is in `/tmp/shmpc-paper-velocity-rollout.log`; artifacts are in
`/tmp/shmpc-paper-rollout/paper-velocity-tracking/`.

User verification of the candidate behavior is still required.

## Follow-up changes and validation

Authorized by the request to make remaining paper-alignment changes. The edits
are localized to timing configuration/serialization, harness plant/forecast
construction, controller warm starts and fallback admission, and the optional
Gaussian mode factory. A tracking-only objective now seeds toward reference
velocity; progress-reward objectives retain their speed-bound seed. Unspecified
objective terms and solver settings remain unchanged.

Build command (ROS disabled standalone build, same acados root as above):

```sh
cmake --build /tmp/dro-shmpcc-paper-audit --target \
  test_safe_horizon_algorithm experiment_runner test_experiment_artifacts \
  test_bounded_braking test_dynamics test_safe_horizon_configuration -- -j2
ctest --test-dir /tmp/dro-shmpcc-paper-audit \
  -R '^test_(safe_horizon_algorithm|safe_horizon_configuration|bounded_braking|dynamics|experiment_artifacts)$' \
  --timeout 60 --output-on-failure
```

All five targeted tests passed. New algorithm checks exercise default timing
inheritance, 0.05-second plant versus 0.2-second prediction updates, fractional
warm-start re-roll, Eqs. (34)/(36) with deterministic noise, covariance 0.012 for
dt=0.2 and Sigma_w=0.3, static profile parsing, physically admissible slowing,
and rejection of slowing that intersects any original sampled obstacle. The
existing tests were not weakened or changed. Compilation initially caught an
incorrect serialization helper and fixed-size test noise type; the subsequent
build and tests succeeded.

Eight-step integration command:

```sh
/tmp/dro-shmpcc-paper-audit/experiment_runner \
  --config /tmp/shmpc-paper-static-smoke.yaml --seed 77 \
  --output /tmp/shmpc-paper-rollout --label paper-static-multirate \
  --no-gif --no-rviz
```

The temporary configuration is the static paper profile plus rollout_steps=8
and disabled GIF/RVIZ output. Observed: S=1237, eight executed decisions, eight
implementation-certified decisions, zero recorded collisions, minimum clearance
4.8188 m, and `step_limit` termination. Trace timestamps run from 0 to 0.4 s.
This short rollout does not demonstrate passing the obstacle, path completion,
or the probability guarantee. Reported mean solve time was 110933.3881 ms,
maximum 880162.2524 ms; subsequent last solve was 1041.7605 ms. The configured
20 Hz execution grid is not evidence of meeting a 50 ms real-time deadline.
Log: `/tmp/shmpc-paper-static-multirate.log`; artifacts:
`/tmp/shmpc-paper-rollout/paper-static-multirate/`.

Current status: candidate changes implemented and tests executed; behavior
requires user verification. Exact identity to an unpublished solver detail,
FORCES Pro, or exact-arithmetic guarantees is not established.

Follow-up regression command (all six executables rebuilt against this candidate):

```sh
ctest --test-dir /tmp/dro-shmpcc-paper-audit \
  -R '^test_(support_accounting|velocity_bounds|distribution_transfer|mpcc_progress|pruning_and_linearization|mode_catalog)$' \
  --timeout 60 --output-on-failure
```

Three passed: mode_catalog, distribution_transfer, velocity_bounds. Three still
failed: pruning_and_linearization's facet-limit expectation, mpcc_progress's
reference-speed-independent cost expectation, and support_accounting's three
expectations counting infeasible/violated nonlinear scenarios as support.
These are the previously reported discrepancies; no existing test expectations
were changed. Full outputs: `/tmp/shmpc-remaining-tests.log`,
`/tmp/shmpc-remaining-regressions.log`, `/tmp/shmpc-remaining-algorithm.log`.
The older plan_acceptance timeout has not been re-established as a pass.

## Week 2 support-theorem audit

See [the follow-up code/theory audit](week2_support_certificate_audit.md).
Earlier-iterate logging follows the saved decision and the PDF's Theorem 3,
but the complete numerical inequality n_logged >= n_theorem remains unproved.
In the Week 2 implementation, sample-dependent lateral/DR anchor preparation
occurred before the counted QPs. The subsequent approved restriction removes
that preparation from ordinary SH; exact feasibility/activity assumptions still
require a numerical justification. See [the diagnostic and proposal](certificate_anchor_audit.md)
and [post-change results](../results/frozen-anchor-validation/README.md).
The Week 2 experiments measure implementation certificate flags;
they do not close these theorem obligations. The three baseline regression
expectation discrepancies were rerun and remain failures; tests were not weakened.
