# Scrubbing experiment logs and artifacts

```bash
python3 tools/scrub_analysis_logs.py --help
python3 tools/scrub_analysis_logs.py build-base/reviewer-pilot --out scrubbed/reviewer-pilot
python3 tools/scrub_analysis_logs.py build-concentration/fixed-budget-uncertified-short-horizon --out scrubbed/fixed-budget-h8
```

Pass an experiment output directory, not its settings file. A parent containing
multiple suites also works. The input is read-only. Output may be beneath the
input root (it is excluded from discovery), but cannot equal or contain the input
root. Re-running overwrites exported CSVs; unchanged inputs produce identical
CSV bytes. Empty categories have header-only files to avoid leaving stale rows.

Compatible inputs:

| Runner/layout | Exported test-suite label |
|---|---|
| `run_analysis_matrix.py` | `analysis_matrix` |
| `run_reviewer_matrix.py`, pilot/large/ablation runs | `reviewer_matrix` |
| `run_comparison_matrix.py`, original paired search | `comparison_matrix` |
| `sample_efficiency*.json`, including concentration logging | `sample_efficiency` |
| `fixed_budget_uncertified*.json`, including failed probes | `fixed_budget_uncertified` |
| `run_rare_mode_experiment.py` wrapper or raw artifact directory | `rare_mode` |
| Older `case/seed_N/*.log` without a manifest | `legacy_logs` |

Every row has `test_suite`, `test_name` (the source experiment directory name),
`test_root` (absolute source root), and `matrix_identity` where available. Log
rows retain `log_file` and `run_id`; artifact/report rows have `source_artifact`.
The nearest manifest determines provenance when several suites share a parent.
Modern case/controller/profile/budget metadata come from `result.json` instead
of assuming the controller name appears before `seed_N` in the directory path.
Nominal resampling and extra nominal sampling remain distinct from WDRO/retry.

## CSV outputs

- Existing log-derived files remain `run_summary.csv`, `control_steps.csv`,
  `dro_steps.csv`, `dro_modes.csv`, and `dro_samples.csv`.
- `artifact_decisions.csv`, `artifact_attempts.csv`, `artifact_mode_coverage.csv`,
  `artifact_mode_mechanism.csv`, `artifact_transport_costs.csv` preserve detailed
  structured evidence across controllers and repeats. No missing rows are invented.
- `report_*.csv` copies available authoritative matrix reports with provenance,
  including paired comparisons/statistics, mechanism summaries, vertex
  reachability and concentration. These reports are not recalculated from log
  fragments, so their original pairing/error denominators are preserved.
- `rare_*.csv` preserves frozen-scene coverage trials, controller trials, weights,
  geometry, attempts, plans and plan-risk tables; these need no `seed_N` logs.

`run_summary.csv` includes recorded trial status/error, repeat index and
repeatability status. Incomplete logs have `log_complete=0` and missing final
outcomes stay blank; a missing final collision record is not collision-free.
Resolved configuration flags and `certification_status=not_requested` are
retained when available. Requested/issued certificate counts come from actual
decision CSVs. Legacy `support_certified_*` fields count support-limit events
and **must not be interpreted as issued certificates**.

Each repeat remains a separate execution in raw exports. Do not treat repeats
as independent seeds or pool different tests simply because their case names
match. Use the copied paired reports for the experiment's statistical results.
When scrubbing a live run, results are only a snapshot of the files available;
scrubbing does not re-run or establish the runner's repeatability/pairing checks.

## Validation record

Changed `scrub_analysis_logs.py` identity/export/help sections and added the
read-only `scrub_artifacts.py` adapter. Existing log calculations are retained;
new structured tables supply the diagnostics absent from logs. No mathematical
guarantee or formulation was modified. No build is required for these Python
changes.

Executed `python3 tests/test_scrub_analysis_logs.py`: two tests passed, covering
legacy fallback behavior, mixed suites, modern layouts, all five arm labels,
two repeats, failures, inactive certification, concentration and rare-mode
exports, help text and byte-identical re-scrubbing.

Actual exports were also created under `build-concentration/scrubbed/`:

| Source | Parsed logs | Structured artifact rows | Report/frozen rows |
|---|---:|---:|---:|
| Fixed-budget horizon-8 suite | 64 | 40,320 | 3,944 |
| Reviewer pilot | 48 | 34,050 | 72 |
| Concentration smoke suite | 40 | 1,152 | 232 |
| Frozen rare-mode wrapper smoke | 0 | 0 | 895 |
| Failed horizon-20 probe | 32 | 0 | 48 |

The fixed-budget decision export contained 3,840 rows, all with zero requested
and issued certificates; both repeat indices were retained. All 32 failed probe
logs retained ERROR status. No runtime warnings or errors occurred in these
scrubber runs. Candidate changes tested; behavior requires user verification.
