# Candidate experiment implementation: validation record

## Fixed-budget, uncertified comparison

Change made: added `fixed_budget_uncertified.json` (larger horizon-8 suite),
`fixed_budget_uncertified_pilot.json` (original horizon-20 probe), and
`fixed_budget_uncertified_short_horizon.json` (horizon-8 diagnostic suite).
Both arms use 40 scenarios, manual sizing, no outer retry, and
`safe_horizon_enabled: false`. The reporting helper checks resolved flags and
actual batch sizes and exports explicit `certification_status=not_requested`
with requested/issued certificate counts. These localized configuration/report
changes use existing APIs; no C++ build or solver change was needed.

Mathematical impact: no mathematical guarantee or formulation was modified in
controller code. The user-requested configuration disables Safe Horizon and
its certification. That existing switch also selects non-SH recovery/constraint
behavior and its clearance filter; it is not a label-only change. The separate
horizon-8 experiment changes the configured prediction horizon, not the code.

Validation commands:

```bash
python3 tests/run_comparison_matrix.py \
  --settings configs/comparison_matrix/fixed_budget_uncertified_pilot.json \
  --runner build-concentration/experiment_runner \
  --output build-concentration/fixed-budget-uncertified-pilot --jobs 1
python3 tests/run_comparison_matrix.py \
  --settings configs/comparison_matrix/fixed_budget_uncertified_short_horizon.json \
  --runner build-concentration/experiment_runner \
  --output build-concentration/fixed-budget-uncertified-short-horizon --jobs 1
python3 tests/test_sample_efficiency.py --runner build-concentration/experiment_runner
```

The original horizon-20 probe failed all 32 trials at initial constraint
construction: 16 logs reported `Free-space polygon became empty at k=17,
disc=2`, and 16 at `k=16, disc=2`. The runner returned nonzero with 16 pair
errors. Those artifacts are preserved; no failure was counted collision-free.
The horizon-8 run is separately named and retains two environments, four shift
levels and two seeds with two repeats and a 60-decision rollout limit.

The horizon-8 run completed: **32/32 trials OK with exact repeatability and
16/16 paired comparisons OK**. The artifact audit command
`python3 tests/test_sample_efficiency.py --uncertified-artifacts build-concentration/fixed-budget-uncertified-short-horizon`
passed: **3,840 decisions/attempts**, each at exactly **40 scenarios**, with
**zero certificate requests and zero certificates** across both repeats.
Each arm had 16 unique trials, zero collisions, zero route completions, and
16 `step_limit` outcomes. This short-run result does not demonstrate a safety
advantage. Reports are in that output directory; descriptive figures are under
`build-concentration/fixed-budget-uncertified-figures`. Matplotlib emitted its
existing unavailable-Axes3D warning; these plots are two-dimensional.

Regression checks passed: 40 trials × 2 repeats, 80 pair checks, repeat/resume
and tampered-evidence rejection. Five unit tests passed, including the new
fixed-budget configuration check. Large-design generation printed
`24 comparison setups; 240 seed groups; 960 executions`; the larger suite has
not been run. See [FIXED_BUDGET.md](FIXED_BUDGET.md) for runnable commands and
interpretation. Candidate changes tested; behavior requires user verification.

## Concentration logging follow-up

Change made: `SolveAttemptDiagnostics` and the controller/artifact code now copy
and export the existing DRO transport matrix and radius observation count to
`transport_costs.csv`. `comparison_evidence.py` derives vertex distances,
radius/distance ratios, reachability, max(q), and concentration summaries.
`run_analysis_matrix.py` includes the new raw matrix in repeat signatures.
This is observational logging only; **no mathematical guarantee or formulation
was modified**. No allocator, confidence radius, probability floor, cap, or
solver tolerance was changed.

Build and tests executed:

```bash
cmake -S . -B build-concentration -DDRO_MPC_ENABLE_ROS2=OFF \
  -DACADOS_ROOT=/home/stephen/Documents/ACC_Development/Development/acados
cmake --build build-concentration --target experiment_runner test_attempt_diagnostics -j 2
python3 tests/test_sample_efficiency.py --runner build-concentration/experiment_runner
ctest --test-dir build-concentration -R '^test_attempt_diagnostics$' --output-on-failure
```

Observed: build succeeded; 4/4 Python unit tests passed; 40 trials × 2 repeats
with 80 successful pair checks; concentration rows present only for WDRO arms;
modified transport geometry was rejected on resume. The C++ instrumentation
test passed (0.17 seconds), preserving controls/outcomes on successive solves.
The hand-calculated p=(.75,.25), off-diagonal costs 2 fixture produced vertex
distances (.5,1.5) at rho=.5: only the first vertex reachable. Zero-cost geometry
produced blank ratios plus zero-distance flags, while q=(1,0) triggered the
near-one flag. These are diagnostic checks, not performance evidence.

The previous `build-rare-mode` executable was left intact. For the new logs, use
`build-concentration/experiment_runner` and a new output directory such as
`results/mismatch-concentration-v3`; the earlier v2 run command does not include
the new executable. See [SAMPLE_EFFICIENCY.md](SAMPLE_EFFICIENCY.md) for definitions
and the updated command. Candidate fix implemented and test executed; behavior
requires user verification.

## Repeatability/reporting follow-up

The feedback follow-up changes only experiment settings, Python evidence checks,
reports, tests and documentation. No mathematical guarantee or formulation was
modified; no C++ rebuild was needed for this follow-up.

- Both sample-efficiency presets now require two executions per seed. The large
  design remains 3,600 unique controller/seed trials, now **7,200 executions**.
  Historical execution counts below describe the earlier single-repeat presets.
- Mechanism summaries now retain each setup/profile/scenario budget separately,
  with explicit first-attempt and final admissibility, minimum safety margins,
  QP calls and full-cycle timing. Arms A–E identify nominal, extra nominal, WDRO,
  nominal retry, and WDRO→nominal respectively. All ten pair comparisons remain.
- Resume re-reads numerical evidence and recomputes repeat signatures, including
  attempt records and p/q/r/rho/mode counts. It rejects changed numeric evidence;
  timings remain excluded from exact equality. This preserves the existing
  signatures while adding an integrity check to reused results.

Executed after these changes:

```bash
python3 tests/test_sample_efficiency.py --runner build-rare-mode/experiment_runner
python3 tests/test_comparison_matrix.py --runner build-rare-mode/experiment_runner
python3 tests/run_comparison_matrix.py \
  --settings configs/comparison_matrix/sample_efficiency.json \
  --runner build-rare-mode/experiment_runner \
  --output build-rare-mode/repeatable-v2-generated --jobs 1 --generate-only
```

Observed: sample-efficiency unit tests **3/3 passed**; integration **40 trials ×
2 repeats, 80 pair checks passed**. Additional checks passed for budget-separated
reports and final admissibility; changing only recorded attempt time allowed
resume, changing a recorded QP count caused nonzero exit and a repeat-signature
error, and a subsequent resume reran that trial and restored the original paired
summary. Legacy integration passed 10 profiles / 40 executions, repeatability,
pairing, filtering, resume, and stale-manifest rejection. Design generation
printed `72 comparison setups; 720 seed groups; 7200 executions`.
`git diff --check` passed. The 7,200-execution suite has not been run here.

Use a **new output directory** for these changed scripts/settings. The command
in [SAMPLE_EFFICIENCY.md](SAMPLE_EFFICIENCY.md) uses `results/mismatch-repeatable-v2`.
Candidate fix implemented and test executed; behavior requires user verification.

## Change made

- `tests/run_comparison_matrix.py`: retains the existing nominal/WDRO search,
  adds three optional comparator arms and explicit scenario budgets, and fails
  generation if requested mismatch/variant keys are absent.
- `tests/comparison_evidence.py`: checks resolved experimental settings,
  attempt/sample accounting and paired nominal histories; reports all arm pairs,
  paired collision inference, completion and mechanism metrics.
- `include/types.hpp`, `include/mpc_controller.hpp`, `src/mpc_controller.cpp`,
  and harness/artifact configuration files: opt-in observational attempt records.
  Both failed first attempts and nominal retries survive result replacement.
  Diagnostics default off and add no RNG calls.
- `sample_efficiency*.json`: separate graded-shift, five-arm presets, leaving
  the existing exploratory settings intact. The large preset has 360
  configurations and 3,600 rollouts, not executed as part of implementation.
- `tools/rare_mode_experiment.cpp`, `tests/run_rare_mode_experiment.py`, and
  `tools/report_rare_mode_experiment.py`: isolated fixed-scene coverage and
  real conditional-retry experiment using production APIs.
- `tools/plot_comparison_evidence.py`: complete-group descriptive curves and
  denominator CSVs; separate configurations/budgets are not pooled in figures.
- New targeted tests and CMake registrations exercise these paths.

The additions reuse the existing sampler, controller, acceptance criteria and
artifact runner. Experimental presets and reporting tools can be removed
independently of the default-off telemetry.

## Mathematical impact

No mathematical guarantee or formulation was modified by these additions.
The manual-budget presets deliberately select experimental budgets through
existing configuration APIs; they do not claim automatic sample-count coverage.
The previously approved nominal-resampling flag is reused. The earlier approved
projected-mean change belongs to the reviewer-suite work, not this telemetry
change. Exact inclusion formulas are comparisons for fixed-law unconditional
sampling, not assertions about conditional fallback or closed-loop safety.

## Validation performed

Build (separate from `build-base`):

```bash
cmake -S . -B build-rare-mode -DDRO_MPC_ENABLE_ROS2=OFF \
  -DACADOS_ROOT=/home/stephen/Documents/ACC_Development/Development/acados
cmake --build build-rare-mode --target experiment_runner rare_mode_experiment \
  test_attempt_diagnostics test_reviewer_controls test_dro_fallback \
  test_experiment_artifacts -j 2
ctest --test-dir build-rare-mode \
  -R '^test_(attempt_diagnostics|sample_efficiency|sample_efficiency_smoke|reviewer_controls|dro_fallback|experiment_artifacts)$' \
  --output-on-failure
```

Observed: **6/6 passed**, 8.02 seconds. The diagnostics regression compares
instrumented/uninstrumented controls and success exactly across two decisions,
including a failed nominal first attempt and successful nominal retry. This
checks that observation did not consume random draws or change these outcomes.

Additional executed tests:

```bash
python3 tests/test_comparison_matrix.py --runner build-rare-mode/experiment_runner
python3 tests/test_sample_efficiency.py --runner build-rare-mode/experiment_runner
python3 tests/test_reviewer_experiments.py
python3 tests/test_analysis_matrix.py
```

- Legacy comparison: 10 profiles, 40 executions; pairing, repeatability, resume,
  filtering and stale-manifest rejection passed.
- Sample efficiency: 40 trials × 2 repeats, five policies, four severities,
  two budgets, **80 successful pair checks**; nominal beliefs/plant traces,
  mode probabilities/counts, blank nominal risk fields and resume checked.
- Reviewer report unit tests: 5 passed; analysis matrix unit tests: 8 passed.
- Large-design generation printed `72 comparison setups; 720 seed groups;
  3600 executions`. Saved under `build-rare-mode/sample-efficiency-generated`.

Frozen pilot:

```bash
./build-rare-mode/rare_mode_experiment build-rare-mode/rare-pilot 500 4
python3 tools/report_rare_mode_experiment.py \
  --input build-rare-mode/rare-pilot --output build-rare-mode/rare-pilot-report
python3 tests/test_sample_efficiency.py --rare-artifacts build-rare-mode/rare-pilot
```

Observed: 8,000 sampling trials and 96 controller trials, **0 errors**,
24 complete four-policy groups. Logged cut-in probabilities:
nominal `0.0104843`, frozen-reference WDRO `0.0298836`.

| Scheme at 40 total draws | Measured inclusion (500 seeds) | Analytic inclusion |
|---|---:|---:|
| Nominal single | 0.328 | 0.343994 |
| Nominal unconditional split | 0.328 | 0.343994 |
| WDRO single | 0.700 | 0.702866 |
| WDRO/nominal unconditional split | 0.560 | 0.558500 |

All recorded fixed-law coverage rates passed the test's six-Monte-Carlo-standard-
error plus 1/n comparison; every nominal split had exactly the same mode count
as its paired nominal single. Actual conditional controller attempt counts and
draw totals also passed. These observations demonstrate sampling behavior in
this fixture, not a general collision or admissibility advantage.

The manifest-producing wrapper was separately exercised with 10 coverage
replicates and one controller seed under `build-rare-mode/rare-wrapper-smoke`:
16 coverage summary rows, 24 policy rows, six matched groups, zero excluded;
artifact checks passed for all 24 controller records.

Plot runtime check: a three-step/horizon-four version of the pilot with seed 77
ran 20 trials and all 40 arm-pair checks. The plotting command exported
20 rate rows and PDF/SVG/PNG under `build-rare-mode/plot-smoke-figures`; the PNG
was visually inspected. Those short-run curves are only a plumbing check.
Matplotlib warned that Axes3D was unavailable due to mixed installations; the
two-dimensional exports completed. Build warnings included existing unused
variables/functions. `git diff --check` produced no errors.

The earlier single-repeat, full-horizon 40-rollout pilot completed under
`build-rare-mode/comparison-pilot` (log `/tmp/comparison-pilot.log`), with all
80 arm-pair checks OK. This is not validation of two-repeat full-horizon behavior;
no closed-loop performance claim is made from the frozen or short-run checks.

## Current status

Candidate fix implemented and test executed; behavior requires user verification.
