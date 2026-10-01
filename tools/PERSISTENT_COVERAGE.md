# Persistent coverage experiments

Results: `results/persistent-coverage/REPORT.md`.

## Completed protocol

The new transition cells `(p,S) ≈ (.02,20), (.05,10), (.08,10)` use 100 paired seeds each (1001–1100), fixed late-switch geometry, switch step 10, lag 5, and a 120-step limit. Matched `(.02,10)` and `(.05,20)` anchors use the same seeds and timing. The earlier `(.05,20,switch5)` negative-control evidence is retained separately; different switch times are not silently pooled.

`p` denotes raw history proportions; actual posterior-predictive probabilities, WDRO probabilities, risk scores, and counts are exported at every solve. In particular, `across` is exported BEFORE it becomes the realized true mode. ΔU is provided for k=1,2,3.

The count intervention starts from a paired pre-switch state and enforces Nd=0,1,2,3,5,10 for five replanning steps. Shared non-dangerous scenario slots, stationary-obstacle trajectories, per-step bank seeds, and subsequent nominal draws are paired. The original modes have zero diffusion; this does not establish behavior with nonzero process noise. Refusal stops the run; missing future decisions are never counted as successfully controlled steps.

A one-shot `set_experimental_scenarios` API accepts exactly S supplied scenarios only for raw, uncertified SH_MPCC without DRO, Markov switching, or stratification. It rejects certified and retry configurations. The original sampler still advances its RNG normally. Unset intervention batches leave ordinary behavior unchanged. `reset()` clears pending batches. This is the explicitly requested non-IID sampling intervention, not a new certified controller arm.

## CSV locations

| File | Contents |
|---|---|
| `across_every_solve.csv` | Every recorded across-mode solve, including pre-switch p, q, r, Nd, and ΔU1/2/3 |
| `across_paired_decisions.csv` | Matched decision counts/mass shifts and paired seed outcomes |
| `paired_seed_outcomes.csv` | One record per paired seed, including a fixed pre-switch exposure summary |
| `transition_summary.csv` | Five matched cells, collision/refusal/completion counts and intervals |
| `all_run_outcomes.csv` | All 1,000 transition runs |
| `intervention-updated-belief/confirmation_outcomes.csv` | 600 runs: 100 seeds × six forced-count arms |
| `intervention-updated-belief/confirmation_summary.csv` | Per-count collision/refusal/completion and window-completion rates |
| `intervention-updated-belief/paired_outcomes.csv` | Each count arm paired against Nd=0 |
| `recoverability-updated-belief/pilot_selection_scores.csv` | Larger-separation and longer-window screening, including failures |
| `*/<case>/seed_<seed>/nd_<n>/paired_slots.csv` | Actual trajectory slots through the controlled window and first post-window decision |

`intervention/` and `recoverability/` retain earlier fixed-belief pilots. Use the **updated-belief** directories for final conclusions: their post-window sampling uses the same lagged history updates as the production controller. These datasets are not pooled.

In the forced-count fixture, `reference_iid_probability` is a nominal reference quantity, NOT the intervention's sampling law. Binomial undercoverage calculations apply to the ordinary transition runs only. All intervention runs log `guarantees=not_requested`. Nominal risk scores that were not computed remain blank; they are not imputed from the WDRO controller's different reference trajectory.

## Build and tests

From the repository root, in bash:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S . -B build-base
cmake --build build-base --target persistent_coverage_experiment test_persistent_intervention test_attempt_diagnostics -j 2
ctest --test-dir build-base -R '^(test_persistent_intervention|test_persistent_coverage|test_attempt_diagnostics)$' --output-on-failure
python3 -m unittest discover -s tests -p 'test_persistent_analysis.py'
```

Ordinary nominal and WDRO runs were also compared with the prior executable: controls, plans, plant trajectories, mode probabilities/counts, and outcomes matched exactly, excluding timing.

## Generate reports without rerunning simulations

```bash
python3 tools/analyze_persistent_coverage.py --root results/persistent-coverage
```

## Simulation commands

The committed settings contain the exact 100-seed transition designs:

```bash
for cell in p20_s20 p50_s10 p80_s10 p20_s10 p50_s20; do
  python3 tests/run_risk_stress.py \
    --settings "configs/persistent_coverage/$cell.json" \
    --output "results/persistent-coverage/$cell" --resume || break
done
python3 tests/run_persistent_intervention.py
python3 tests/run_persistent_recoverability.py
python3 tools/analyze_persistent_coverage.py
```

Completed simulations are reused only when their recorded configuration, source, and executable signatures match. A changed build deliberately prevents resuming into old results. The primary intervention runner supports `--output NEW_DIRECTORY`. Preserve the recorded manifests and selection files when reproducing results.

All rates concern these prescribed trajectories and finite simulation horizons. They are not population collision guarantees. Outcome inference uses paired seeds, not repeated decisions as independent observations. A failed recoverability screen or flat count-response curve must remain visible and must not be described as evidence for a coverage threshold.
