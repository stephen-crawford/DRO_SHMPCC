# Risk-directed sampling experiments

These additions address both feedback documents. The primary experiment remains
nominal/WDRO at the same S, extra nominal at 2S, nominal resampling, and WDRO
fallback. The optional sixth arm is a separate stratification ablation.

## Run commands

From the repository root, build the affected executables and run the targeted tests:

```bash
source /opt/ros/jazzy/setup.bash
cmake --build build-base --target experiment_runner rare_mode_experiment test_scenario_sampler test_attempt_diagnostics test_reviewer_controls -j 2
ctest --test-dir build-base -R '^(test_scenario_sampler|test_attempt_diagnostics|test_reviewer_controls)$' --output-on-failure
python3 -m unittest discover -s tests -p test_risk_sampling.py
python3 -m unittest discover -s tests -p test_causal_matrix.py
python3 -m unittest discover -s tests -p test_comparison_matrix.py
python3 -m unittest discover -s tests -p test_scrub_analysis_logs.py
```

`build-base` is the existing configured build (including its acados path). The
Python rare-history test skips when its compiled executable is absent.

Run the hard two-obstacle intersection matrix, both class layouts, S=20/40/80,
boost=0/5/15/30, seeds 77–86, two repeatability runs. This is 2,400 executions.
Use a NEW directory: historical manifest hashes must remain immutable.
Serial execution retains comparable timing measurements.

```bash
python3 tests/run_comparison_matrix.py --settings configs/causal_matrix/hard_mismatch.json --runner build-base/experiment_runner --output results/hard-mismatch-v2 --generate-only
python3 tests/run_comparison_matrix.py --settings configs/causal_matrix/hard_mismatch.json --runner build-base/experiment_runner --output results/hard-mismatch-v2 --resume
python3 tools/scrub_artifacts.py results/hard-mismatch-v2 --out results/hard-mismatch-v2-csv
python3 tools/analyze_comparison_results.py results/hard-mismatch-v2-csv --out results/hard-mismatch-v2-outcomes
python3 tools/analyze_risk_sampling.py results/hard-mismatch-v2-csv --out results/hard-mismatch-v2-risk
```

For the separate sixth-arm ablation (2,880 executions), replace the settings with
`configs/causal_matrix/stratified_ablation.json` and use a new output root such as
`results/stratified-v2`. Run the same scrubber and risk analyzer on that root.
The established five-arm analyzer intentionally retains its original five-arm
population. The risk analyzer includes raw-WDRO versus stratified comparisons.
`--case four_way_intersection_o2_c1_m3_boost_15_s20` can select a pilot condition;
the manifest still lists all scheduled conditions. A later full `--resume` uses
the same manifest, and generates full comparison reports.

`switch_regime=hold` means each sampled prediction holds its mode over the
horizon. The existing plant redraws its realized mode at each real-world step;
these experiments therefore contain realized switching events. This is not the
Markov prediction model, which the stratified option currently rejects.

Frozen rare-mode experiment, with 100/50/20/10 dangerous observations out of 1000:

```bash
for count in 100 50 20 10; do
  python3 tests/run_rare_mode_experiment.py --runner build-base/rare_mode_experiment --output "results/rare-history-${count}-v2" --dangerous-count "$count" --coverage-replicates 2000 --solve-seeds 40
done
```

Production Dirichlet smoothing is unchanged: inspect the actual probabilities
in each `artifacts/weights.csv` (approximately .10035/.05042/.02047/.01048).
`coverage_summary.csv` compares empirical inclusion with the existing exact IID
formula using those probabilities. `wdro_stratified` is an additional frozen
sampling arm with one guaranteed draw per mode; the four controller/recovery
arms in `policy_summary.csv` remain unchanged. Frozen geometry establishes a
sampling mechanism, not closed-loop safety or certification.

## Reports and interpretation

The matrix runner already writes `primary_summary.csv`, `all_comparisons.csv`,
`vertex_reachability.csv`, and `concentration_summary.csv`. The scrubber exports
these with `report_` prefixes. Do not infer vertex reachability from q alone.
The new risk analyzer provides:

- `risk_per_decision.csv`, `risk_per_seed.csv`: top-risk count/fraction, risk
  coverage, empirical sampled risk, q-weighted target risk, absolute error,
  support size, maximum q, exact zero mass, realized-mode loss diagnostics.
- `support_distribution.csv`: support histogram per seed and experimental cell.
- `risk_mass_scatter.csv`: pre-reweighting risk versus q−p, with source provenance.
- `matched_mode_allocation.csv`, `allocation_effects_per_seed.csv`: q−p versus
  WDRO-minus-comparison sample count, including positive unique top-risk modes,
  increased/decreased mass groups, wins/ties/losses, and actual budgets.
- `matched_risk_coverage.csv`: risk coverage and sampled risk for each comparison
  arm evaluated using the SAME logged WDRO risk vector. Extra nominal keeps 2S.
- `matched_events.csv`, `events_per_seed.csv`: realized non-MAP, mode-switch,
  dangerous-realized, dangerous-non-MAP and dangerous-switch representation.
- `event_decisions_per_seed.csv`: decision inadmissibility, deduplicated across
  obstacles. `event_rollout_outcomes.csv`: collision, completion, clearance and
  solve time for rollouts containing each event, once per seed/arm/event.
- `ablation_pairs.csv`: completion AND collision discordance, including WDRO
  versus WDRO+fallback and nominal versus nominal-resample. Exact two-sided
  McNemar tests remain per fixed cell; four wins/zero losses yields p=.125.
- `excluded_evidence.csv`: missing/invalid risk vectors, counts, or probabilities;
  duplicate records raise errors. No clipping or imputation is performed.

Repeat 0 is the default; `--repeat 1` writes a separate repeatability report.
Never count repeats or decision rows as independent seeds. Tied or zero-risk
maxima have no selected top mode; zero total risk gives blank risk coverage.
Support uses EXACT q>0, not a tolerance. Raw distributions remain exported.
Matched comparisons require runner-validated paired plant/belief trajectories,
complete OK logs, identical mode support, beliefs and realized modes at that
step. No matching on seed alone. Missing runner reports produce empty matched
outputs while per-arm diagnostics remain available. Finish the matrix and
rescrub to obtain its comparison reports.

Default danger means the realized mode is the unique positive risk maximum.
Alternatively preregister a numerical cutoff using `--risk-threshold VALUE`.
The score is computed before reweighting, but on the WDRO controller's ego plan:
this is a COMMON, exploratory label, not controller-independent ground truth.
The current artifacts do not contain independent reference-plan danger scores.
The tool does not infer next-mode labels, stepwise physical collision/clearance,
or certified safe actions from the available fields. Conditional collision,
completion and minimum clearance are explicitly whole-rollout outcomes among
rollouts with observed events; shared-prefix survival can bias event selection.

## Opt-in stratified arm and changed guarantees

`wdro_stratified_sampling: true` is explicitly experimental. For each obstacle
with M available modes, its first M scenarios force one of each mode in stable
library order, including modes with q=0. The remaining S−M draws use q*. Multiple
obstacles share joint scenario indices; this ensures marginal mode coverage,
NOT every joint mode combination. It spends exactly S joint scenarios and
rejects S<M. All existing arms set the option false and retain their IID paths.

The forced draws make the batch non-IID. Existing scenario certificates and
sample-sufficiency guarantees DO NOT apply. Configuration validation rejects
Safe Horizon, automatic sample sizing, Markov prediction and fallback when this
option is enabled. The runtime emits a warning, and `resolved_config.yaml`
records `scenario_guarantee_status: not_applicable_non_iid_stratified`; scrubbed
rows retain it. No replacement certificate is claimed or implemented.
For this arm, logged `sampling_probability` remains the residual q* law and
`q_times_S` remains the raw-q diagnostic. Expected full-batch count is
`1 + (S-M)*q_m`, not `S*q_m`. Observed counts are authoritative.

The certificate/transfer-sample question remains separate. These presets all
use `safe_horizon_enabled=false`; none establish certified sample reduction.

## Validation performed in this change

- Targeted build succeeded. The initial rebuild reported two existing unused-code
  warnings in `src/collision_constraints.cpp`.
- 25 Python tests passed across risk sampling, causal matrix, comparison matrix,
  scrubber and analysis matrix suites. Three targeted CTest tests passed.
- Vertex fixture: raw q=(0,0,1), S=20 gives stratified counts (1,1,18).
  Default and explicit-false IID paths produced identical trajectories and RNG state.
- Six-arm integration: o2/c1/intersection, boost=.15, S=20, seed=77,
  three steps, two repeats: 12 OK executions, zero pairing errors. All 36
  stratified mode records had at least one draw; all six stratified decisions
  requested/issued no certificate. Existing arms recorded stratification=false.
- Four frozen smoke runs used 32 coverage replicates and one controller seed
  each. All completed. At p=.0104843 and S=16, nominal/WDRO/stratified inclusion
  was 5/32, 16/32, 32/32. These are plumbing checks, not performance evidence.
- The first new live-test geometry produced an empty free-space polygon. The
  fixture was changed to feasible separated trajectories; production acceptance
  logic and existing assertions were not changed.
- Existing 2.7 GB mechanism CSV: 126,029 raw-WDRO risk decisions, 90 excluded
  invalid probability/count records. No paired events were admitted because
  that scrubbed directory lacked runner comparison reports. The complete smoke
  artifacts yielded six risk decisions, 90 matched mode rows and 24 event rows.

Full hard/stratified matrices and large rare-mode runs have not been executed.
Candidate change implemented and tested; behavior requires user verification.
