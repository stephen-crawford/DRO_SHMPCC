# Repeatable analysis matrix

`tests/run_analysis_matrix.py` generates **480 configurations**:

- 1–4 obstacles;
- 1–4 classes, restricted to classes ≤ obstacles;
- straight highway, S-curve, four-way intersection, two-lane roundabout;
- 1–6 modes per class;
- SH-MPCC and SH-MPCC with DRO.

All settings are in `settings.json`. The default schedule uses seeds 77–86,
with two executions per seed: **4,800 seed trials / 9,600 executions**.
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
