# Reviewer-testing validation record

Executed on 2026-09-22. These are candidate changes and measured diagnostics;
behavior and paper conclusions require user verification.

## Change made

- `include/config.hpp`, `include/experiment_harness.hpp`, YAML parsing/defaults,
  and resolved-config serialization: one opt-in `nominal_resampling_baseline`
  flag makes the existing hybrid wrapper run nominal sampling on both attempts.
  It reuses the existing retry path rather than adding a second controller loop.
- `src/dro.cpp`: direct and mixture Gaussian surrogates now retain
  `R - n.dot(diff)` when the fallback normal is selected. The ordinary nonzero
  formula is algebraically equivalent, subject to floating-point roundoff.
- `tests/run_reviewer_matrix.py` and settings: six matched variants, standard and
  shift/boost profiles, shared-plant checks, frozen manifests, and resume.
- `tools/report_reviewer_experiments.py`: read-only matched outcomes, descriptive
  seed-block intervals, paired overlap, total-cycle and fallback latency,
  clearance/penetration/exposure metrics, scenario counts, and mode persistence.
- New controller/report regression tests and calibration-assumption audit.
  Existing assertions were not weakened. No Lean source was edited by this task.

## Mathematical impact

This change affects mathematical behavior and was made only after explicit
approval. The nominal-resampling arm changes the first-attempt sampling law only
when opted in. The projected-mean substitution affects the near-zero fallback
region and aligns the mean with its chosen normal. Neither modification changes
collision inequalities, acceptance thresholds, sample-size formulas, CP bounds,
solver tolerances, or the existing retry/recovery implementation.

## Validation performed

Builds:

```bash
cmake -S . -B build-base
cmake --build build-base --target test_reviewer_controls test_dro_fallback test_safe_horizon_configuration test_risk_scoring_models experiment_runner -j 2
cmake --build build-base --target calibration_assumption_audit -j 2
cmake --build build-base --target test_experiment_artifacts test_experiment_runner_cli test_ambiguity_calculation -j 2
```

Targeted checks:

```bash
ctest --test-dir build-base -R '^test_(reviewer_controls|reviewer_experiments|reviewer_experiments_smoke|dro_fallback|safe_horizon_configuration|risk_scoring_models|analysis_matrix|comparison_matrix)$' --output-on-failure
ctest --test-dir build-base -R '^test_(experiment_artifacts|experiment_runner_cli|ambiguity_calculation)$' --output-on-failure
python3 tests/test_reviewer_experiments.py --runner build-base/experiment_runner
```

Results: **8/8 + 3/3 CTest targets passed**. The standalone Python suite also
passed after adding resolved-YAML replay validation. Its smoke execution covered
12 variant/profile trials with two repeats each, exact numerical repeatability,
equal eight-scenario budgets, preserved nominal-resampling flag after replay,
pairing, resume, reporting, and stale-manifest rejection.

The C++ resampling fixture recovered on seed 2 after its first nominal attempt
failed. A threat-only fixture rejected both attempts. For displacement
`(-5e-13, 0)` the direct projected mean was `0.9500000000005`, matching the expected
projection; the held-mode mixture matched as well. Exact zero, perpendicular
near-zero, and ordinary nonzero displacements passed. The final-stage threat
produced full-horizon score `0.95000115034938037`, versus zero for first-stage-only
evaluation. No assertion was relaxed to obtain these results.

The build reported existing unused-variable/function warnings in
`src/collision_constraints.cpp`. No selected test failed. The first build command
for the newly added target required CMake reconfiguration before that target
existed; configuration and subsequent builds succeeded.

### Full-horizon pilot

```bash
python3 tests/run_reviewer_matrix.py --settings configs/reviewer_tests/pilot.json --output build-base/reviewer-pilot --timeout 180
```

**48/48 trials returned OK; 8/8 matched groups passed shared-plant checks.**
Horizon 20, dt 0.1 s, 1,123 scenarios per attempt, up to 450 steps, two layouts,
two obstacles/classes, three modes, two seeds, six variants, and two profiles.
OK is an execution/report status, not a completion or safety claim.

| Profile | Method | No-admissible-control | Route completions |
|---|---|---:|---:|
| Standard | Nominal | 0/4 | 4/4 |
| Standard | Direct WDRO | 2/4 | 2/4 |
| Standard | Nominal resampling | 0/4 | 4/4 |
| Standard | WDRO + nominal | 0/4 | 4/4 |
| Standard | Uniform-cost hybrid | 0/4 | 4/4 |
| Standard | Fixed-radius hybrid | 0/4 | 4/4 |
| Shift/boost | Nominal | 1/4 | 3/4 |
| Shift/boost | Direct WDRO | 0/4 | 3/4 |
| Shift/boost | Nominal resampling | 0/4 | 4/4 |
| Shift/boost | WDRO + nominal | 0/4 | 3/4 |
| Shift/boost | Uniform-cost hybrid | 0/4 | 3/4 |
| Shift/boost | Fixed-radius hybrid | 0/4 | 4/4 |

All pilot collision flags were zero. **The pilot does not demonstrate a WDRO
advantage over ordinary resampling.** Only two master seeds were tested. Timing
from this validation pilot is not a controlled benchmark: development checks
ran on the same host while it executed. Use a dedicated serial run for paper
latency claims. Reports live in `build-base/reviewer-pilot-report` and
`build-base/reviewer-resampling-comparison`.

The full design was generated in `build-base/reviewer-full-design`: 2,880
configurations / 28,800 scheduled trials. **The full sweep was not executed.**

### Historical report audit

The read-only report covered 2,263 complete matched groups (6,789 rollouts),
excluding 137 incomplete/error groups. Results in
`build-base/reviewer-historical-full` preserve the original saved experiments,
which predate this task's risk-mean change.

- No-admissible-control counts: nominal 639, WDRO 655, hybrid 75.
- Nominal/WDRO overlap: neither refuses 1,387; only nominal 221; only WDRO 237;
  both 418. Standalone overlap is not a same-state causal test of the hybrid.
- Hybrid cycle median 48.316286 ms, p95 195.8856406 ms.
- Hybrid fallback-conditioned cycle median 330.647049 ms, p95 595.30055775 ms.
- Hybrid fallback fraction: 1,228 / 515,973 decisions = 0.237997%.
- At a 100 ms deadline, historical hybrid miss rate: 16.40299%.
- Actual scenarios per attempt: 1,123 for all three historical controllers.

The historical timing result does not support treating rare fallback cycles as
cheap simply because the overall median is similar. Margins remained positive
at recorded states; this does not establish continuous-time collision safety.

### Calibration assumption audit

```bash
./build-base/calibration_assumption_audit build-base/reviewer-calibration-audit.csv
```

For 100 observations and 2,000 replications per law:

| Observation law | CP-region coverage | Raw ball coverage |
|---|---:|---:|
| Stationary i.i.d. | 1,960/2,000 = 98.0% | 1,980/2,000 = 99.0% |
| Stationary blocks of ten | 775/2,000 = 38.75% | 1,149/2,000 = 57.45% |
| Mid-history distribution shift | 385/2,000 = 19.25% | 385/2,000 = 19.25% |

These results demonstrate sensitivity in the specified synthetic experiment;
they are not empirical coverage estimates for the complete simulator, and not
new guarantees. Both 20- and 100-observation results are retained in the CSV.

## Current status

Candidate fix implemented and test executed; behavior requires user verification.
The controlled experiments and diagnostics are available; the reviewer’s causal
question remains open until a sufficiently broad resampling comparison is run.
