# Mode mismatch and sample efficiency experiment

This preset tests whether WDRO improves representation and closed-loop outcomes
under graded plant shifts, including an extra-sampling nominal competitor. It
retains every predetermined seed. `matches.json` remains a qualitative debugging
aid; use `primary_summary.csv` for paired outcome comparisons.

## Build and run

From the repository root, use a separate build to preserve existing experiment
executables and their manifests:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S . -B build-concentration -DDRO_MPC_ENABLE_ROS2=OFF \
  -DACADOS_ROOT=/home/stephen/Documents/ACC_Development/Development/acados
cmake --build build-concentration --target experiment_runner rare_mode_experiment \
  test_attempt_diagnostics -j 2

# Small full-horizon pilot: 40 seed trials × 2 repeats = 80 executions.
python3 tests/run_comparison_matrix.py \
  --settings configs/comparison_matrix/sample_efficiency_pilot.json \
  --runner build-concentration/experiment_runner --output results/mismatch-pilot-v3

# Freeze the design: 3,600 seed trials × 2 repeats = 7,200 executions.
python3 tests/run_comparison_matrix.py \
  --settings configs/comparison_matrix/sample_efficiency.json \
  --runner build-concentration/experiment_runner --output results/mismatch-concentration-v3 \
  --generate-only

# Execute; repeat this command to resume an interrupted run.
python3 tests/run_comparison_matrix.py \
  --settings configs/comparison_matrix/sample_efficiency.json \
  --runner build-concentration/experiment_runner --output results/mismatch-concentration-v3 --resume

# After completion, export curves and their underlying denominators.
python3 tools/plot_comparison_evidence.py \
  --input results/mismatch-concentration-v3 --output figures/mismatch-large
```

Adjust `ACADOS_ROOT` for another installation. Use new output directories after
changing binaries, scripts or settings. Default `--jobs 1` avoids concurrent
solver contention in timing comparisons. A `--case` selects one setup and all its
arms; do not use selected cases as the complete paper dataset. Filtered runs
replace top-level reports with that selection. Re-run the unfiltered command
with `--resume` to regenerate complete reports.

## Predetermined design

The large preset uses straight and intersection environments; (obstacles,
classes) = (1,1), (2,1), (2,2); three modes; S = 20, 40, 80; seeds 77–86;
and plant override probabilities 0, .05, .15, .30. Each seed trial runs twice;
only repeat zero contributes to statistical sample size. All profiles boost
`turn_right` (index 2). There are 72 setup/profile/budget combinations.

| Arm | First attempt | Retry on rejection | Maximum draws per decision |
|---|---|---|---:|
| `sh_mpcc` | Nominal S | None | S |
| `sh_mpcc_dro` | WDRO S | None | S |
| `sh_mpcc_extra` | Nominal 2S | None | 2S |
| `sh_mpcc_resample` | Nominal S | Fresh nominal S | 2S |
| `sh_mpcc_dro_fallback` | WDRO S | Fresh nominal S | 2S |

This permits same-S comparisons and comparisons with extra nominal sampling.
Maximum draws are not actual draws or equal CPU budgets: retries are conditional,
and QP iterations vary. Actual draws, outer attempts, QP calls, and cycle time are
recorded. Manual budgets disable automatic sample-size selection; these are
exploratory budget experiments and do not acquire the automatic sample-count
guarantee. Existing acceptance logic and certificate diagnostics are preserved.

The shift parameters act on plant mode selection in `experiment_harness.cpp`.
They are not injected into the nominal sampling law by
`ExperimentConfig::to_scenario_mpc_config()`. The controller does observe the
resulting modes and updates its history, so mismatch need not persist. The x axis
is **configured override probability**, not a measured divergence or an exact
plant marginal. Likewise, `turn_right` is a focus mode, not automatically a
dangerous mode: inspect its geometry and logged risk scores.

Generated YAML must contain every requested mismatch key or generation fails.
Each execution checks requested values against `resolved_config.yaml`. All arm
pairs must share seeds, initial placement, backend, and realized obstacle traces
over their shared prefix. With diagnostics enabled they must also share nominal
beliefs over that prefix. Early termination can shorten the compared prefix.

## Evidence and interpretation

`artifact_capture_attempt_diagnostics` is opt-in and defaults to false. When
enabled, each artifact bundle includes:

- `attempts.csv`: first/retry success, actual initial batch size, QP calls, time.
- `mode_mechanism.csv`: each obstacle/mode's nominal p, sampling q, risk r, radius
  rho, sampled count, and realized mode. Nominal attempts leave r/rho blank.
  Counts precede scenario removal; switching trajectories count their initial
  mode. The preset uses held prediction modes.
- Existing `decisions.csv`, `trace.csv`, metrics and resolved configurations.

Top-level reports include:

- `all_comparisons.json/csv`: all ten arm pairs per setup and seed; legacy
  `nondro_`/`dro_` prefixes mean controller A/B as identified by their columns.
- `primary_summary.csv`: complete paired collision tables (neither, A only,
  B only, both), collision rates, separately reported collision-free completion,
  errors/pending denominators, B-minus-A difference, exact two-sided McNemar p,
  and a descriptive paired seed bootstrap interval. Tests are per fixed setup,
  not pooled across dependent configurations; multiple comparisons are unadjusted.
  The interval uses 2,000 resamples, RNG seed 1729, sorted indices 49 and 1949.
  Few seeds or zero events can yield misleadingly narrow intervals.
- `mechanism_per_seed.csv`: first-attempt refusal, final no-admissible-control,
  fallback, draws, QP calls, cycle time, safety margin, mode inclusion and actual
  exposure. A refusal is not a physical collision.
- `mechanism_summary.csv`: descriptive marginals per fixed setup, shift and
  budget, including first/final admissibility, fallback, actual draws, QP calls,
  total cycle cost and minimum safety margins. Expected/measured/missing counts
  remain explicit; denominators can differ by arm. Use complete paired records
  for comparative claims.
- `evidence_notes.json`: limitations accompanying the machine-readable evidence.

The plot tool uses only groups with every arm present and all pairing checks
passing, separately for each environment/obstacle/class/budget setup. It plots
collision, collision-free completion, focus-mode omission and time against boost,
and exports the plotted denominators. Omission rates are over observed
obstacle/decision opportunities; differing stopping times affect exposure.
Timing includes diagnostic collection overhead and failed attempts. None of these descriptive curves
is a safety certificate. Predetermine more seeds before a confirmatory run; two
pilot seeds cannot establish a collision-rate advantage.

## Frozen rare-mode mechanism experiment

This complementary fixture isolates mode representation from adaptive histories
and conditional fallback. It uses the production sampler, DRO and controller,
with one obstacle and three deterministic paths (stationary, away, cut-in).
Synthetic history counts are 900/90/10; the production KT posterior, rather
than a hard-coded .01, determines nominal probabilities. `weights.csv` records
actual p/q/r/rho and geometric clearance to the fixed ego reference.

```bash
python3 tests/run_rare_mode_experiment.py \
  --runner build-concentration/rare_mode_experiment \
  --output results/rare-mode --coverage-replicates 2000 --solve-seeds 40
python3 tests/test_sample_efficiency.py --rare-artifacts results/rare-mode/artifacts
```

Coverage trials compare nominal single, nominal split, WDRO single, and
WDRO-plus-nominal unconditional split at identical total budgets 16/40/80/160.
The analytic inclusion formula is used only for these fixed-law, unconditional
draws. Nominal split continues the same RNG stream and should match nominal
single exactly. This is not the law of a conditional retry.

Separate real controller trials use four policies (nominal, nominal retry, WDRO,
WDRO retry), base S = 8/20/40, and two designs: equal per-attempt S, or equal
maximum draws (singles 2S versus retries S+S). These are one-decision cold-start
comparisons; the matrix above supplies closed-loop trajectories and warm starts.
The reporter retains failures and summarizes complete four-policy seed groups.

`plan_risk.csv` evaluates accepted open-loop plans exactly over the three
deterministic modes, with specified cut-in mass and other masses renormalized.
It records collision mass, safety-margin violation mass, mean penetration and
worst-5%-tail mean penetration. Rejected plans have no executable-plan risk row;
accepted-only risks cannot establish an unconditional closed-loop advantage.
`scene.csv` and `plans.csv` expose the geometry for inspection.

See [VALIDATION.md](VALIDATION.md) for executed checks and pilot observations.

## Repeatability and paper claims

The five arms are labeled A (nominal), B (extra nominal), C (WDRO), D (nominal
retry), E (WDRO then nominal). Keep three claims separate: the frozen experiment
measures sampling coverage; A/C and B/C assess closed-loop shift robustness;
D/E assesses the contribution beyond a second stochastic attempt.

Both supplied presets require two fresh executions per seed. Numerical traces,
controller decisions, metrics, per-attempt outcomes/QP counts, and p/q/r/rho/mode
counts must match exactly or the trial is ERROR. Wall-clock durations are excluded
from that comparison, so repeatability does not mean identical execution times.
Resume re-reads artifact evidence and checks it against recorded signatures,
rather than trusting a cached OK status. Altered evidence produces ERROR; a
subsequent resume reruns that trial. Hashes freeze settings, generated YAML,
runner executable and reporting scripts. Use the same build/environment and
`--jobs 1`; do not edit code or rebuild the executable during the run. Changed
inputs require a new output directory.

The large preset therefore has 3,600 unique controller/seed trials and 7,200
executions, with 7,200 arm-pair comparisons over 720 setup/seed groups. Replicates
are checks, not additional independent evidence. Ten seeds per fixed setup can
still be insufficient to establish a rare collision-rate difference.

## Vertex reachability and concentration logging

With `artifact_capture_attempt_diagnostics: true` (already enabled in both
presets), each bundle now includes `transport_costs.csv`: the actual labelled
D matrix and `radius_observation_count` from each WDRO obstacle/attempt. It is
header-only for nominal attempts. The matrix and counts are copied from the
existing DRO result, without recomputing geometry or changing the optimization.
Their numerical contents are included in repeat and resume integrity checks.

At matrix completion, three additional reports are generated from repeat zero:

- `vertex_reachability.csv`: each target mode's `vertex_distance = sum_i p_i D_ij`,
  `rho_over_vertex_distance`, reachability (`rho >= distance`), p, q, and risk.
- `concentration_per_solve.csv`: each WDRO obstacle/attempt's observation count g,
  rho, minimum vertex distance, rho/minimum distance, maximum q, and concentration
  flags. Join by case/seed/step/attempt/obstacle to the other evidence. These raw
  rows allow plotting concentration against g and rho without arbitrary bins.
- `concentration_summary.csv`: rates with max(q) > 0.9, max(q) >= 1-1e-9, and
  max(q) exactly 1; reachable-vertex fraction; range/median of max(q); observation
  count and radius ranges, separately for each setup/profile/budget/controller.

The summary denominator is WDRO **obstacle/attempt evaluations**, not independent
rollouts; nominal retries are excluded. The 1e-9 near-one threshold is an explicit
reporting convention, not an allocator tolerance or guarantee. Reachability is a
direct floating-point comparison with no added tolerance; raw distances are
retained for boundary inspection. Zero distances have blank ratios and explicit
zero-distance flags, while reachability remains separately reported.

For an entropic allocator, reachable vertices need not be attained. Neither CP
calibration nor these logs assert that rho contracts monotonically or that
concentration is impossible. The observation count is the radius calibration's
reported count; do not treat it as an independent-observation guarantee.

These new artifacts require the new executable. Do not resume an older manifest
with changed scripts or binary; use the new output directory in the commands
above. The previous build remains available for inspecting earlier experiments.
