# Nominal versus WDRO with Safe Horizon disabled

These configurations compare two sampling laws at exactly **40 scenarios per
attempt**, with `safe_horizon_enabled: false` and automatic sample sizing off.
There is no outer retry arm. All other settings are shared within each pair.
The legacy controller identifiers remain `sh_mpcc` and `sh_mpcc_dro`, but the
explicit flag disables Safe Horizon for both; those names are not certificates.

`mechanism_per_seed.csv` and `mechanism_summary.csv` explicitly record:

- `safe_horizon_enabled=False`;
- `automatic_sample_sizing=False`;
- `certification_status=not_requested`;
- zero requested and issued certificates.

The runner checks the resolved flags, each attempt's exact scenario count, and
every decision's certificate counters. A mismatch is an ERROR. Existing paired
outcomes, clearance, p/q/r/rho, concentration, sample counts and timings are
retained. Each seed runs twice with exact numerical repeat checks, excluding
wall-clock timing. Repeats count only once in statistical summaries.

## Important behavior of the existing controller

Disabling Safe Horizon also selects the existing non-SH constraint/recovery
path, disables support-cap enforcement, and enables its ordinary clearance
filter. Therefore this compares nominal versus WDRO **within that non-SH path**;
it is not an isolated ablation of the certification label in the SH algorithm.
No controller formulation or recovery code was changed for this experiment.

The first pilot used prediction horizon 20. All 32 trials failed at initial
constraint construction with `Free-space polygon became empty at k=16/17,
disc=2`; these are errors, not refusals, collisions, or successful runs. The
original configuration is retained as `fixed_budget_uncertified_pilot.json`,
and its output as `build-concentration/fixed-budget-uncertified-pilot`.

The subsequent diagnostic suite, `fixed_budget_uncertified_short_horizon.json`,
uses horizon 8, two environments, four shift levels, seeds 77/78 and up to 60
decisions per rollout: 32 unique trials × 2 repeats = 64 executions. It is a
separately identified experiment, not a replacement of the failed horizon-20
results. Short-horizon findings do not establish behavior at horizon 20.

## Commands

Reproduce the short-horizon diagnostic suite from the repository root:

```bash
python3 tests/run_comparison_matrix.py \
  --settings configs/comparison_matrix/fixed_budget_uncertified_short_horizon.json \
  --runner build-concentration/experiment_runner \
  --output results/fixed-budget-uncertified-h8 --jobs 1 --resume
python3 tests/test_sample_efficiency.py \
  --uncertified-artifacts results/fixed-budget-uncertified-h8
```

The larger `fixed_budget_uncertified.json` preset also uses horizon 8, retains the
450-decision rollout limit, varies obstacle/class counts, and uses ten seeds:
48 configurations × 10 seeds × 2 repeats = **960 executions**. It has not been
fully executed here. Other configurations can still encounter non-SH feasibility
errors; the runner preserves those errors rather than counting them safe.

```bash
python3 tests/run_comparison_matrix.py \
  --settings configs/comparison_matrix/fixed_budget_uncertified.json \
  --runner build-concentration/experiment_runner \
  --output results/fixed-budget-uncertified-large --jobs 1 --resume
```

Use a new output directory after changing settings, scripts or the executable.
The current executable already supports these flags; no C++ rebuild is needed.
Neither this suite nor observed collision-free runs carry Safe Horizon
certification. See [VALIDATION.md](VALIDATION.md) for measured results.
