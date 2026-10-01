# Targeted causal-link experiments

The broad `results/risk-stress-v3` matrix is complete (396/396 scheduled cells) and frozen. No broad-grid expansion is performed by these scripts.

Completed results:

- `results/causal-link-followup/{selected_p50,selected_p20,negative_control}/report/`: 100 new seeds per case and arm, full mechanism logs and matched dangerous-event tables.
- `results/causal-link-followup/extension-v2/`: outcome-blind geometry/calibration prescreen, commitment-relative scheduling and its failed attrition gate, initial braking pilots.
- `results/causal-link-followup/refinement/`: complete frozen count strata, additional braking pilots and 50-seed confirmation, live sampling/constraint/QP timings for 2–4 obstacles.
- `results/causal-link-followup/extension-v2/REPORT.md`: results and limitations; `figures/` contains SVG and PNG figures.

The first incomplete `extension/` directory is retained as an audit of a checkpoint filename bug, caught before outcome runs. `extension-v2/frozen` is an initial limited-count pilot; use `refinement/frozen` for the final analysis. Neither is silently pooled with final confirmation data.

## Build and tests

From the repository root, in bash:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S . -B build-base
cmake --build build-base --target causal_link_probe causal_link_rollout test_attempt_diagnostics -j 2
ctest --test-dir build-base -R '^(test_causal_link|test_causal_link_runtime|test_attempt_diagnostics)$' --output-on-failure
python3 -m unittest discover -s tests -p 'test_risk_sampling.py'
python3 -m unittest discover -s tests -p 'test_dangerous_events.py'
python3 -m unittest discover -s tests -p 'test_long_stress_pilots.py'
```

## Run a fresh follow-up and write analyzed CSVs

Use NEW output directories. These runs do not rerun the already completed 100-seed selected cases or modify the frozen grid.

```bash
python3 tests/run_causal_link.py --output results/causal-link-new --stage all
python3 tests/run_causal_link_refinement.py \
  --prior results/causal-link-new --output results/causal-link-new-refinement
python3 tools/report_causal_link.py \
  --root results/causal-link-followup \
  --extension results/causal-link-new \
  --frozen-root results/causal-link-new-refinement
```

The main runner writes CSV analyses automatically. The refinement completes the additional braking screen and confirmation and records live timings. Successful runs have per-run command/config/binary hashes and are reused only when those match. Failed or incomplete folders are retained and cause an explicit error; choose a new output root after source changes. This protects against mixing builds or silently treating a missing run as a successful one. The `--stage` option supports `prescreen`, `frozen`, `commitment`, `braking`, and `scaling`; frozen requires an existing prescreen selection.

To regenerate figures and reports from the completed evidence only:

```bash
python3 tools/report_causal_link.py \
  --extension results/causal-link-followup/extension-v2 \
  --frozen-root results/causal-link-followup/refinement
```

## Interpretation

- Geometries are ranked by ΔU2 before their closed-loop outcomes. All calibration radii come from the existing estimator. Raw history proportions remain 85/10/5%; posterior-predictive p changes slightly with evidence size because smoothing is unchanged. Do not describe this as an exactly fixed posterior center.
- Each frozen geometry has 5,000 sets per sampling law. Coverage checks use the exact binomial formula for k=1,2,3 and a simultaneous 99% Hoeffding tolerance over each geometry's six checks.
- Conditional runs use outcome-blind seed rejection to control INITIAL Nd. The controller must reproduce the selected count. Later draws retain the frozen categorical law but are not count-controlled. Every accepted seed's collision, refusal, and 40-step survival is retained; survival is not path completion.
- No collisions were observed in the conditional study. It cannot establish a nonzero gamma collision gap. Gamma CSVs retain missing count strata and uncertainty, and do not misweight the deliberately balanced count quotas. Outcome intervals are Wilson; the combined simultaneous intervals are approximate.
- Commitment timing comes from an independent no-switch nominal reference for each seed, not the treated trajectory. References that do not reach commitment remain in the denominator. The current fixture fails the 95% reach gate.
- Braking selection uses 20 fresh pilot seeds per design. Confirmation uses 50 separate seeds. A failure means collision or no-admissible-control; the two outcomes are also reported separately.
- `controller_trajectory_generation_ms` measures actual sampler calls with diagnostics enabled, excluding DRO computation. `trajectory_generation_ms` and `fixed_reference_constraint_ms` are separate production-function microbenchmarks. Fixed-reference raw counts include k=0. `scaling_constraint_groups.csv` records actual controller raw/facet counts per logged horizon/disc group. QP row counts come from actual solver logs.
- All experiments are uncertified and use discrete-time collision checks. No mathematical guarantee, constraint, sampling law, or acceptance rule was modified. The production change is an opt-in wall-clock diagnostic, covered by the existing exact control/outcome/RNG-invariance regression.
