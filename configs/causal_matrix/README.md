# Current empirical causal-chain matrix

One entry point: `tests/run_comparison_matrix.py` (defaults to this directory's
`settings.json`). It reuses `run_analysis_matrix.py` for execution, evidence
validation, manifest hashes and resume. No controller or certificate formula is changed.

| Axis | Values |
| --- | --- |
| Scene | straight, four-way intersection |
| Obstacles / classes | 1/1, 2/1, 2/2 |
| Modes | constant velocity, left turn, right turn |
| Mismatch boost | 0, 0.05, 0.15, 0.30, applied to mode index 2 |
| Base scenario budget S | 20, 40, 80 |
| Arms | nominal, WDRO, extra nominal (2S), nominal resample, WDRO fallback |
| Seeds / repeats | 77–86 / two repeats per seed |
| Rollout / horizon | 450 steps / 8 |
| Certification | Safe Horizon disabled; empirical fixed-budget experiment |

72 conditions × five arms × ten seeds × two repeats = **7,200 executions**.
These settings extend the existing uncertified fixed-budget experiment; they do
not establish formal safety guarantees. The certification pilot remains separate.
Serial execution is the default for comparable runtime measurements.
Every arm writes `rollout.svg` and `rollout.gif` in each `repeat_N/` artifact
bundle, matching the analysis matrix. Older frozen configs with visualization
disabled require a new output directory to adopt this setting.

```bash
python3 tests/run_comparison_matrix.py --runner build-base/experiment_runner \
  --output results/causal-matrix --generate-only
python3 tests/run_comparison_matrix.py --runner build-base/experiment_runner \
  --output results/causal-matrix --resume
python3 tools/scrub_artifacts.py results/causal-matrix --out results/causal-csv
python3 tools/analyze_comparison_results.py results/causal-csv --out results/causal-report
```

Use `--case straight_o1_c1_m3_boost_0_s20` to select one condition with every arm.
The manifest always retains the full schedule, so unrun cases remain visible.
Use a new output directory after changing settings, scripts or runner binary.
Both `scrub_artifacts.py` and `scrub_analysis_logs.py` run the same normalizer.

| Causal link / check | Evidence |
| --- | --- |
| Distribution mismatch | frozen config, profile, nominal probabilities, true mode |
| Representation | `artifact_mode_mechanism.csv`: p, sampling q, mode counts; `mode_representation.csv`: WDRO shift group, n/S, inclusion |
| Controller behavior | `artifact_attempts.csv`, `artifact_decisions.csv`: failed attempts, retry/fallback and RF diagnostics |
| Physical safety / completion | `run_summary.csv`, `artifact_rollout.csv`: realized clearance, collision, completion, final progress |
| Cost | log timing decomposition, attempts, scenario counts, QP calls |
| Missing/error jobs | `expected_runs.csv`, `completeness.csv`, trial error text |
| Certificates | requested/issued fields in `artifact_decisions.csv`; never infer from support counts |

The postprocessor requires all five unique completed arms, trial status OK and
successful runner checks of paired plant trajectories/beliefs before admitting a
cell. Missing, failed, duplicate and unrun cells remain in completeness. A normal
`no_admissible_control` termination remains an outcome; an execution/evidence
ERROR is excluded. Reports include nominal-relative differences, collision
contingency counts (n10 = nominal-only collision), outcomes and mechanism by seed.
Conditions and repeat indices remain separate; repeats do not increase seed n.

Mechanism classification uses the WDRO attempt-zero q−p at the same seed,
repeat, decision, obstacle and mode for every arm. Retry attempts remain in raw
tables. Shared-prefix rows only are used; mode-check denominators may differ
when an arm terminates early. These are descriptive comparisons, not independent
decision-level statistical samples. A boosted mode is not automatically dangerous.
Compare actual budgets (extra nominal uses 2S) when assessing representation
efficiency; no unobserved matching budget or efficiency gain is inferred.
Predicted MPC objective and SH anchor clearance are not rollout cost or physical safety.

## Consolidation

The unrelated `run_single_obstacle_matrix.py` visualization launcher was removed;
its YAML fixtures remain available for direct runner use. The analysis/reviewer
modules remain because existing tests and historical resume depend on them.
Existing regression tests and certification tools are retained. Historical
comparison presets require explicit `--settings`; this is the current default.

## Validation observed

`python3 -m unittest discover -s tests -p test_causal_matrix.py` exercises matrix
counts/arm budgets, missing and duplicate cells, required plant pairing, paired
clearance differences, collision discordance, and manifests with no logs.

A real three-step, one-seed, two-repeat integration fixture produced five OK
arms (ten executions), zero pair errors and two complete five-arm cells after
scrubbing. Earlier repeatability rejections were traced to wall-clock `qp_ms`
and `constraint_ms` being included in repeat signatures. The candidate change
excludes those timing fields while retaining numerical evidence checks; nine
analysis-matrix regressions passed, including sensitivity to sample counts,
QP call counts, success flags and mode evidence. User verification remains
pending. The 7,200-execution experiment has not been run to completion.
