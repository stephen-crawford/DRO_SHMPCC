# Investigating DRO refusal and nominal completion

`investigate_matrix.py` works on saved **analysis-matrix or comparison-matrix**
results. It identifies paired seeds where the nominal controller completes
collision-free and DRO terminates with `no_admissible_control`, then optionally
investigates a nominal plan from the **DRO run's actual decision state**.

## The exact roundabout cases: seeds 79 and 85

Use the letter `o` in `o2` (two obstacles), not the digit zero `02`.

```sh
cmake -S . -B build-base
cmake --build build-base --target counterfactual_probe experiment_runner -j 2

# Read existing results only, without running controllers.
python3 tests/investigate_matrix.py \
  --matrix build-base/analysis-matrix \
  --pair roundabout_o2_c2_m4 --seeds 79 85 \
  --output build-base/roundabout-refusal-report

# Replay each saved DRO run to its refusal and perform frozen-state experiments.
python3 tests/investigate_matrix.py \
  --matrix build-base/analysis-matrix \
  --pair roundabout_o2_c2_m4 --seeds 79 85 \
  --output build-base/roundabout-counterfactuals \
  --probe --samples 1000 --mc-seed 12345 \
  --radius-scales 0,0.25,0.5,0.75,1
```

The saved failures are at **zero-based decision 185 (seed 79)** and
**104 (seed 85)**, corresponding to one-based decisions 186 and 105. By default
the script takes this index from each DRO run's executed-step count. Use `--step`
to select a preceding decision, for example `--seeds 79 --step 184`.
The unrelated baseline run at the same step has a different ego state; its
trajectory is not used as the frozen nominal candidate.

Use a new output directory for every probe invocation. Existing probe directories
are not overwritten. The source matrix is read-only and outputs must be outside
it. Per-probe reports are saved as each investigation finishes.

## Apply to either matrix

Omit `--pair` and `--seeds` to inventory all saved pairs. Add `--probe` to probe
all discovered refusal/nominal-completion cases (this can take substantial time).
For a comparison matrix, use its profile-prefixed pair name:

```sh
python3 tests/investigate_matrix.py \
  --matrix build-base/comparison-matrix \
  --pair shift_and_boost_roundabout_o2_c2_m4 \
  --output build-base/comparison-refusal-report
```

The script preserves pair errors, incomplete pairs, reverse outcomes and ordinary
win flags. Pairing checks use the existing comparison suite's initial-placement,
seed, backend and shared-obstacle-trace checks. A missing baseline is not evidence
of nominal success. Successful repeats contribute once per seed, after the
matrix's repeatability checks. The inventory uses saved repeat-zero artifacts.

## What the frozen probe does

1. Loads the saved DRO `resolved_config.yaml`, reruns that seed, and stops after
   the selected decision. No current matrix settings are substituted.
2. Immediately before solving that decision, copies the controller's full
   history, reference trajectory, backup, sampling RNG and configuration into
   isolated nominal and DRO controllers. All mode observations for that decision
   are already present. Only `dro.enabled` differs between these two forks.
   QP workspaces are fresh because the backend starts each subproblem
   deterministically; they contain no solver warm start or RNG state.
3. Solves both forks and records their trajectories and controls. If the nominal
   fork rejects its candidate, replay reports `nominal_plan_inadmissible` rather
   than counting that candidate as an executable nominal action.
4. Holds the accepted nominal **control sequence** fixed for the entire horizon,
   using its dynamically rolled-out ego trajectory, and independently samples
   counterfactual obstacle trajectories under:
   - nominal per-obstacle mode weights `p`;
   - the DRO fork's `qstar` weights;
   - each mode with `q > p`, conditioning that obstacle on the mode while other
     obstacles keep their `qstar` distributions.
5. Counts any collision at future sampled stages `k=1..N`, separately counts
   first-step collisions, and retains every trial's first collision stage and
   minimum disc-to-obstacle margin. A collision is strict margin `< 0`, using the
   existing combined radius (including the configured safety margin).

Sampling calls the existing scenario sampler with a separate `--mc-seed`. It uses
held modes and the prediction models' noise matrices, exactly as that sampler
specifies. This is a **prediction-law, open-loop experiment**, not feedback MPC,
continuous-time collision testing, or the harness's plant process-noise/speed-cap
law. It does not replay the shift/boost routine within the counterfactual horizon.
The two laws share the MC seed for comparability but use separate RNG instances;
neither affects the original plant or controller. Markov-jump prediction and
externally supplied sampling weights are explicitly rejected rather than silently
approximated by held-mode sampling.

`qstar` is the adversary computed against the DRO controller's existing risk
reference trajectory. It is **not** a newly optimized worst-case distribution for
the nominal candidate. Exported `risk_score` values are the configured risk scores,
not necessarily collision probabilities. Empirical collision counts have finite
Monte Carlo uncertainty; zero observations do not establish zero risk. No automatic
"justified refusal" or "excessive conservatism" label is assigned.

## Optional radius sensitivity

`--radius-scales 0,0.25,0.5,0.75,1` performs an additional **frozen-state** sweep.
For every obstacle, it takes the unscaled fork's actual `rho_used` and requests
`alpha * rho_used` through the existing fixed-radius DRO API, retaining the same
nominal weights, risk reference, mode models and history. Each resulting
categorical distribution is passed to another isolated controller through its
existing custom-weight interface, with the same pre-decision RNG and warm start.
The requested and actual radii and weights are written for inspection.

This reports **plan admissibility at that state**, not route completion, rollout
collision rate, or a first failure step along a new trajectory. Alpha zero is a
zero-radius DRO experiment, not an assertion that every solver detail equals the
DRO-disabled run. These experimental radii do not inherit the original calibrated
confidence guarantee. No production radius formula, clamp, feasibility rule or
controller decision is modified. A full closed-loop time-varying radius-multiplier
experiment is not implemented by this tool.

## Reproduction checks and outputs

`paired_outcomes.{csv,json}`, `refusals.{csv,json}`, and `summary.json` describe
saved outcomes. Probe outputs include:

- `snapshot.csv`, `obstacles.csv`, `mode_history.csv`, `controller_rng.txt`,
  `source_warmstart_{plan,controls}.csv`: inputs and controller-state evidence;
- `nominal_*`, `dro_*`: candidate plans and controls;
- `weights.csv`: per-obstacle `p`, `q`, probability change, risk score and radius;
- `counterfactual_trials.csv`, `counterfactual_summary.csv`: raw and grouped counts;
- `radius_sweep.csv`, `radius_sweep_weights.csv`, `alpha_*`: optional sensitivity;
- `replay/source/`: ordinary harness artifacts from the unmodified source rollout;
- `step_<index>.log` and `step_<index>_report.json`: execution/provenance checks;
- top-level `probes.json`: combined investigation reports;
- top-level `counterfactuals.csv` and `radius_sweep.csv`: combined rows with pair,
  seed, decision index and replay status (filter to `probe_status=OK`).

The report hashes the source artifacts, executable and investigation script.
After the probe it requires the saved state/obstacle prefix and original decision
columns to match exactly, excluding solve timings. A difference yields
`REPLAY_MISMATCH` and a nonzero exit; the resulting Monte Carlo data must not be
presented as an investigation of the exact historical state. Source artifacts
must also remain unchanged. Equality of these numerical outputs does not prove
identity with the historical executable or its dirty source tree.

## Validation

```sh
python3 tests/test_matrix_investigation.py
python3 tests/test_matrix_investigation.py --integration \
  build-base/counterfactual_probe build-base/experiment_runner
```

The integration fixture checks that the probe leaves the live replay unchanged,
that repeated Monte Carlo runs reproduce exactly, and that alpha 1 reproduces the
unscaled DRO fork's controls. These are instrumentation checks, not safety proofs.
