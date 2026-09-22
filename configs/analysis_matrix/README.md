# Repeatable analysis matrix

`tests/run_analysis_matrix.py` generates **720 configurations**:

- 1–4 obstacles;
- 1–4 classes, restricted to classes ≤ obstacles;
- straight highway, S-curve, four-way intersection, two-lane roundabout;
- 1–6 modes per class;
- SH-MPCC, SH-MPCC with DRO, and `sh_mpcc_dro_fallback`.

All settings are in `settings.json`. The default schedule uses seeds 77–86,
with one execution per seed: **7,200 seed trials / 7,200 executions**.
These are experiments, not assertions that every controller avoids collisions
or completes its route. Solver failures and early terminations are retained.

From the repository root:

```sh
cmake --build build-base --target experiment_runner -j 4

# Review every generated YAML and the frozen manifest without starting rollouts.
python3 tests/run_analysis_matrix.py --output build-base/analysis-matrix --generate-only

# Run all experiments; rerun this command to resume after interruption.
python3 tests/run_analysis_matrix.py --output build-base/analysis-matrix --resume

# Run one configuration across the same seed schedule.
python3 tests/run_analysis_matrix.py --output build-base/analysis-matrix --resume \
  --case sh_mpcc_dro_roundabout_o4_c3_m6
```

Use `--settings my_settings.json` for standardized batch changes. Edit
`overrides` for common horizon, speed, noise, risk, tolerance, rollout length,
and visualization options. Edit the axis lists, mode catalog, seeds and repeats
at the top level. The generator selects the first M entries of `mode_catalog`,
so mode sets are nested. Each class uses that same catalog; classes represent
separate shared-history groups, not different vehicle geometries. Obstacle i
belongs to class `i % C`, giving exactly C occupied classes, with balanced sizes.
Unknown mode names, duplicate seeds and conflicting axis overrides are rejected.
Canonical automatic scenario sizing remains enabled in the default experiment.

`--jobs 1` is the default for comparable compute-time measurements. Parallel
execution is available with `--jobs N`; competing processes affect timings.
`--timeout` controls the per-execution limit in seconds. The manifest freezes
settings, all configuration hashes, executable and analysis-script hashes,
concurrency and timeout. Changed settings, code or executable require a new
output directory; resume will not mix their results. Use the same arguments for
generation and execution. Successful seed trials are reused by `--resume`,
including trials that observed a collision or early termination; errors retry.

## Pairing and repeatability

Every combination uses the same master-seed list and canonical derived plant /
controller seeds. Plant and controller RNGs are separate. Obstacles start at
fixed route fractions `[0.20, 0.35, 0.50, 0.65]`, with seeded lateral and velocity
perturbations. The single-obstacle case uses the same placement rule. Thus
obstacle i has the same initial position and velocity for a given environment
and seed across solver styles, class counts, mode counts, and all obstacle
counts that include i. The runner checks those common placements and records
actual initial states. Different environment shapes necessarily have different
world coordinates. Different mode supports/counts can change subsequent plant
motion; matching seeds does not assert identical motion across those axes.

The plant follows the configured mode models, with no ego-following policy.
For Markov experiments, the diagnostic metadata retains the actual sampled
mode sequence without drawing additional random numbers or altering dynamics.
Every repeat checks realized numerical traces, mode coverage, decision outcomes,
control effort and geometric metrics. Timings are excluded from equality checks.
Only repeat 0 of a seed trial whose executions completed and matched contributes
to aggregate rates; repeats are not independent statistical samples.

## Outputs and exact metric definitions

`matrix.json` records the complete design. `configs/*.yaml` contains frozen
configurations. `results.json` and `rollouts.csv` hold per-seed/per-repeat metrics;
`summary.csv` and `summary.json` hold one row per configuration, including pending,
measured and error counts. Missing rates are null/blank, never zero.

`summary_per_seed.csv` holds one row per configuration and seed, with the same
summary metrics plus `seed`, `status`, and `error`. It uses the existing repeat-0
aggregation rule; repeated executions do not create extra rows. Error trials
remain visible with blank rates.

The script exits automatically after all selected configuration/seed trials
have been processed and reports written, printing `Matrix complete` with the
coverage and error counts. It exits with status 0 when all trials are OK, or 1
when any trial errored. There is no automatic retry loop; a later `--resume`
invocation retries errors.

Each `<case>/seed_<seed>/repeat_<index>/` contains resolved configuration, seed /
backend manifest, `rollout.csv`, `trace.csv`, `decisions.csv`, `mode_coverage.csv`,
`geometry.csv` and `conservatism.csv`, plus selected visualization artifacts.
Sibling `.log` files retain process errors. `result.json` survives interrupted
batches; `--resume` reconstructs the aggregate summaries. A process error or
repeat mismatch produces an ERROR record and nonzero batch exit status. An
observed collision or `no_admissible_control` is a measured outcome, not a broken
experiment. Error trials are excluded from rates, with explicit denominator and
error counts; an incomplete batch is not a complete comparison.

| Metric | Definition |
|---|---|
| Collision rate | Fraction of measured seed trials with the harness collision flag. Detection is at executed simulation states, not continuous-time swept geometry. |
| SH certification rate | Total `CERTIFIED` decisions / total decisions where SH certification was requested, pooled across measured seed trials. Failed decisions count if certification was requested. Zero requests gives null. |
| Missed-mode rate | Fraction of measured seed trials with **any decision and obstacle** whose current true dynamic mode is absent from the union of **all sampled scenarios' modes for that obstacle at that decision**. A mode anywhere in a sampled Markov horizon counts as represented. This is mode identity coverage, not a same-horizon-time sequence-match probability. |
| Compute time | Mean and maximum controller solve time in milliseconds, including failed returned solves. Raw per-decision times and per-repeat end-to-end wall time are retained. Aggregate mean pools decisions; repeats do not inflate its denominator. |
| Control effort | Sum of `acceleration² + angular_velocity²` over applied controls, matching the harness convention; no dt factor or unit normalization. Aggregate reports mean rollout effort. |
| Max conservatism | User-requested closest approach: **minimum signed margin** across time and obstacles, despite the word “max.” For each obstacle/frame, margin = minimum ego-disc-center Euclidean distance − (ego radius + obstacle radius + safety margin). Negative means combined-radius penetration. |
| Average conservatism | Arithmetic mean of those nearest-disc signed margins over all recorded frame/obstacle pairs, including the initial state. Aggregate pools these pairs. Individual disc centers are not averaged. |

The certification counts are the existing solver reports, not a new guarantee.
Conservatism uses realized states, not forecast half-spaces, and is sampled at
trace times. Completion counts, executed steps and termination reasons accompany
the metrics so a short failed rollout is not confused with a successful low-effort
run. Initial margins are included in conservatism; the existing harness collision
flag is left unchanged.

## Regression tests

```sh
cmake --build build-base --target experiment_runner test_analysis_diagnostics -j 4
ctest --test-dir build-base -R 'test_analysis_(diagnostics|matrix|matrix_smoke)$' --output-on-failure
```

The matrix smoke test runs **every one of the 480 combinations twice**, with one
seed, one decision, horizon 2 and four manually selected scenarios. This checks
configuration parsing, all mode-support sizes, class assignment, metrics,
repeatability, paired starts and resume behavior. It does not establish route
completion, collision rates or certification performance for the full settings.
Separate unit tests check signed multi-disc geometry, missed-mode rollout events,
weighted rate denominators, missing data and all-scenario Markov mode coverage.

### Traffic maneuvers and support-only forecasts

The current catalog order puts left and right turns immediately after constant
velocity, so low-mode-count cases exercise turns instead of only braking/speed
changes. `artifact_show_support_scenarios: true` is enabled in the matrix settings:
GIF/SVG display solver-reported support forecasts rather than white constraint
boundaries. See the [easy vehicle-turn test and visualization options](../traffic_tests/README.md).
Generate into a new output directory to apply these changes to an existing batch.

### Exact case filtering

`--case sh_mpcc_dro_roundabout_o2_c2_m4` schedules only that exact case and its
configured seed/repeat list. Generated YAML updates, printed counts, saved-result
collection, placement checks, and root CSV/JSON reports are scoped to that same
selection. Other case directories and YAML files remain untouched. Root reports
represent the latest invocation; omit `--case` and use `--resume` to rebuild the
full matrix reports. The manifest still describes the complete experiment design,
so different filtered invocations can share a batch directory.

Unknown names are rejected before any output is written, including with
`--generate-only`. The obstacle-count prefix is the letter `o`, not the digit `0`.
As with other runner updates, its changed script fingerprint requires a new
output directory when starting from a batch created by an older script.

### Failure classification

New runs include these diagnostic columns in `decisions.csv`. Values are `1`
(true), `0` (false), or `-1` (not evaluated). Every `no_admissible_control`
also emits a `[FAILURE CLASSIFICATION]` line in the repeat log (one-based failed
step; decision CSV steps remain zero-based).

| Column | Evidence |
|---|---|
| `backup_available` | Shifted backup passed the existing deterministic availability check. |
| `backup_removal_budget_exceeded` | Backup's incompatible sampled scenarios exceeded the removal budget. |
| `backup_dro_failed` | Currently `-1`: no separate DRO feasibility test is performed on the shifted backup. The removal check is reported separately. |
| `braking_collision_feasible` | Recovery's direct braking trajectory had no incompatible sampled scenarios; unknown when that recovery branch was not entered. Does not include hard-limit feasibility. |
| `any_homotopy_geometrically_feasible` | At least one attempted PATH/AUTO/LEFT/RIGHT anchor passed the existing geometric preparation/check; does not establish dynamic feasibility. |
| `last_qp_converged` | Final SQP subproblem's QP convergence flag; unknown if none was solved. |
| `sqp_sampled_collision_feasible` | Rejected SQP candidate checked against actual sampled disc geometry. |
| `fallback_sampled_collision_feasible` | Gentle-braking fallback checked against actual sampled disc geometry; distinct from recovery's direct braking. |

The last two checks use all original sampled scenarios (including removed ones),
future stages, combined radius and the existing compatibility tolerance of
`1e-9`. They run only on the SQP rejection/fallback path, do not affect acceptance,
and remain unknown for nonfinite trajectories. Linearized collision-row
rejection does not itself imply a collision in this geometry check.

`rollouts.csv`/`results.json` retain the failed decision's fields and
`failure_class`: `sampled_collision`, `solver_nonconvergence`,
`solver_nonconvergence_and_sampled_collision`, `other_rejection`, or `unknown`.
Other terminations use `not_applicable`. A sampled collision means at least one
of the tested SQP/fallback candidates collided; it is **not** proof that the
sampled optimization problem has no feasible solution. QP nonconvergence is
observed evidence, not a claim of causation or an SQP convergence certificate.

Summaries count `failure_<column>_{true,false,unknown}_rollouts` and
`failure_<class>_rollouts`, using only `no_admissible_control` outcomes from
measured seed trials, once per seed. Older evidence without these columns is
unknown. Use a new output directory after rebuilding; the existing manifest
checks prevent mixing old and new executable/script versions.

## Investigating nominal completion with DRO refusal

Use the [matrix investigation tool](../../tests/MATRIX_INVESTIGATION.md) to scan
saved seed pairs and probe a nominal plan at the DRO run's frozen refusal state
under `p`, `qstar`, and boosted modes. It includes exact replay checks and an
optional frozen-state radius sweep, with commands for roundabout seeds 79 and 85.

## DRO with nominal fallback

Select `mpc_type: sh_mpcc_dro_fallback` in YAML or `sh_mpcc_dro_fallback`
in the matrix `solver_styles`. This type enables DRO. Every decision first
runs the existing DRO solve and its homotopy/checked-braking recovery. Only
when that returns no admissible plan does it draw fresh nominal scenarios and
run standard SH-MPCC, including its existing checked-braking fallback. If that
also fails, the harness retains its `no_admissible_control` termination.
The next decision always tries DRO again, using the last executed plan.

`decisions.csv` reports `nominal_fallback_attempted` and `used_nominal_fallback`.
Certificates and scenario diagnostics describe the returned attempt; nominal
fallback is not a DRO certificate, and this switching policy does not establish
a new closed-loop probabilistic guarantee. Solve time includes both attempts.
