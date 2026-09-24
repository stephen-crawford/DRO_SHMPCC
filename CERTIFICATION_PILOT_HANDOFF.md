# Certification pilot handoff

**Latest status (2026-09-24): candidate logging/diagnostic implementation tested;
15/15 pilot rollouts completed, 450 snapshots captured. Full bound/transfer
analysis is blocked pending explicit user approval for offline covariance
roundoff treatment. No controller formulation was changed. No transfer or
sample-saving result is claimed.**

The latest consolidated findings and required change summary are at the end of
this file. Earlier checkpoints are retained as a chronological audit trail.

## Task and authorization

User requested implementation of the attached certification diagnostic plan and
a detailed file usable in ChatGPT online. The two attachments read on 2026-09-24
contain the same plan. Primary question: are returned-plan mode bounds below one
and useful for risk transfer? Do not run the full matrix. Do not change controller
constraints, sampling, guarantees, or acceptance. Offline calculations explicitly
requested by the plan are authorized; existing protected logic is preserved.

## Initial state and work log

- Read AGENTS.md and both attachments.
- Shell sandbox cannot start: `error building bubblewrap command: mountinfo path
  is not absolute`. Read commands require approved escalation. One earlier
  inspection was denied; the user subsequently authorized needed reads.
- Repository was already dirty before this task. Existing edits include bundle
  sampling, certification tubes, artifact diagnostics, and posterior analysis.
  Do not attribute those changes to this task or revert them.
- Existing `tools/analyze_posterior_certificate.py` computes fixed returned-plan
  projected Gaussian bounds and retains pair probabilities.
- Existing `tools/analyze_wdro_certificate.py` implements CP vertex enumeration,
  fractional-knapsack inner LP, and transfer inversion. Backend calls production
  CP and Safe-Horizon formulas. Prior documented results are NOT new test results.
- Existing rollout artifacts omit complete returned-plan Gaussian snapshots.
- `src/mpc_controller.cpp`, `solve_optimization_sqp`, initializes support union
  with `pre_support_scenarios`; callers pass removed IDs. Configured sizing uses
  nonremoved cap 6 + removal budget 2 = total cap 8. Measured union must not have
  the removal count added a second time. Recovery paths require explicit audit.
- Proposed logging: reuse opt-in attempt diagnostics; capture each attempt's
  final trajectory, actual disc centers, held affine-Gaussian marginal forecasts,
  observation counts, p_hat, actual sampling q, rho, active/support/removed IDs,
  sample count and production certificate status. Compute CP intervals offline.

## Planned sequence / progress

1. Add observational snapshot logging (in progress).
2. Add batch bounds and probability-shift summaries; retain pair probabilities.
3. Build and test logging invariance and support accounting.
4. Pilot: straight o1 m2, straight o1 m3, s-curve o1 m2; five seeds each.
5. Inspect bound distributions before enabling any transfer analysis.
6. If useful, run existing exact-structure transfer LP offline, inversion, and
   hypothetical sizing with explicit total-support conventions.
7. Expand only if evidence supports it; otherwise document no-go reasons.

## Interpretation constraints

Posterior bounds concern supplied discrete future times and strict disc overlap.
Zero covariance uses the existing strict-event boundary convention. Held affine
Gaussian laws are required; Markov switching and nonlinear body-frame motion do
not satisfy this model. CP coverage requires appropriate iid history assumptions.
Numerical outputs are conditional diagnostics, not newly issued certificates or
proof that a different reduced-sample controller is safe. Existing production
certificate status is logged separately from offline statistical assumptions.

## Current status

Implementation and validation pending. No experiment results from this task yet.

## Implementation checkpoint

- Added `include/certification_snapshot.hpp`, a deterministic serializer called
  only when existing attempt diagnostics are enabled. Includes all attempts,
  final trajectory/discs, held Gaussian forecasts, raw counts, actual sampling
  probabilities, radius, and support/removal provenance. Unsupported predictive
  models are explicitly flagged. No RNG is consumed.
- Added one string field to SolveAttemptDiagnostics, one capture call after each
  attempt returns, and artifact output under `rollout/certification/step_*_attempt_*.json`.
- Added `tools/compute_mode_collision_bounds.py` and `tools/summarize_mode_bounds.py`.
  CP endpoints are reconstructed offline by the unchanged production backend and
  retained in mode_bounds.csv. Raw snapshots retain exact counts and beta_cp.
  Failed/nonreturned/nominal attempts remain visible in outcomes.csv.
- Added `tools/run_certification_pilot.py`: only the three requested cells,
  five seeds beginning at 77, 30 decisions maximum per run, 180-second per-run
  timeout, existing canonical horizon and automatic sample count unchanged.
- Added C++ actual-SQP removed-support regression and Python batch-analysis tests.
- First build: experiment_runner succeeded; combined command exited 2 because
  the old build-base Makefile lacked certificate_numeric_backend. Switched to
  reconfiguring the existing standalone build-certificate with ROS disabled.
  New logger indentation warnings were removed before the second build.
- No new transfer script implemented yet: waiting for observed mode-bound gate.

## Validation checkpoint and numerical blocker

- Standalone build completed successfully for experiment_runner,
  certificate_numeric_backend, test_certification_snapshot,
  test_attempt_diagnostics, test_support_accounting.
- C++ tests passed: actual SQP pre-support IDs [3,17] yielded support union [3,17]
  and count 2; logging on/off preserved controls/outcomes/successive RNG draws;
  existing support-accounting test reported zero failed checks.
- Python tests: 3 new batch tests + 3 existing posterior tests + 3 existing
  transfer tests passed. A syntax typo in the new fixture failed initially and
  was corrected before these runs. No existing expectations were changed.
- Raw logs: results/certification-pilot-validation/*.log.
- Pilot started: python3 tools/run_certification_pilot.py --output
  results/certification-pilot --seeds 5 --steps 30 --timeout 180.
- First completed seed has 30 snapshots but zero evaluated by the unchanged
  helper: exact covariance symmetry check rejects floating-point skew from the
  production Lyapunov recursion. Example turn_left k=2 off-diagonals:
  +1.0587911840678754e-22 and -1.0587911840678754e-22; diagonal
  6.248875084372472e-05. This is a numerical input-validation blocker, not evidence
  that b_m=1. Do not draw a union-bound no-go conclusion from these exclusions.
- Asked user for approval under AGENTS.md 1.2 for offline-only roundoff-scale
  symmetric-part treatment, retaining raw matrices and rejecting larger skew
  or negative eigenvalues. No such treatment has been implemented yet.
- Partial raw support/skew audit saved as partial_raw_audit.json; full pilot
  still running at this checkpoint. No transfer stage enabled.

## Resume instructions for ChatGPT online / another coding session

Read this file first, then AGENTS.md. Preserve pre-existing uncommitted changes.
Current outputs are under results/certification-pilot and
results/certification-pilot-validation. The pilot manifest records each exact
command, executable hash, seed, configuration, exit code, and bounded run length.
The pilot is limited to 15 rollouts, not the full comparison matrix.

1. Check pilot_manifest.json status; do not start duplicate rollouts over these
   paths. Inspect unfinished run.log files if the process has stopped.
2. Check whether user approved the explicitly requested offline covariance
   roundoff treatment. If not, retain exact-symmetry rejection. Do not silently
   symmetrize, add jitter, clip eigenvalues, or modify controller covariance logic.
3. After approved handling and its regression test (if authorized), run:
   `PYTHONDONTWRITEBYTECODE=1 python3 tools/compute_mode_collision_bounds.py
   --root results/certification-pilot --output results/certification-pilot/mode_bounds.csv`
   followed by `python3 tools/summarize_mode_bounds.py
   results/certification-pilot/mode_bounds.csv --output results/certification-pilot/mode_bounds_summary.json`.
   Existing outputs are deliberately refused: use a fresh filename after a failed
   or exploratory analysis. first_seed_bounds.csv is a pre-treatment exclusion
   record and must not be represented as an empirical distribution of b.
4. Inspect failed/unsupported snapshots and mass-decreased modes, not only modes
   that benefit from WDRO. Report fraction b<1, median, q90, q95, pair maxima and
   untruncated sums for each configuration and probability-shift group.
5. Only if nontrivial bounds exist, reuse vertices/transfer_at/invert from
   tools/analyze_wdro_certificate.py in a batch adapter. For fixed p use the
   one-budget fractional-knapsack LP; maximize over CP/simplex vertices.
   Evaluate at production epsilon_SH(S,measured support,beta_cert) only when
   support was actually evaluated. Record production eligibility separately.
6. Invert Psi over eta in [0,1], explicitly reporting if Psi(0)>0.05 (empty
   feasible set) or Psi(1)<=0.05 (all probability budgets feasible). No arbitrary
   clipping of q=0 modes. These are posterior diagnostics, not justification for
   selecting fewer scenarios before a new solve.
7. For hypothetical sample counts use the unchanged production backend with
   identical confidence for baseline/transfer. Report n=6 as the requested
   formula illustration AND production cap n=6+2=8, identifying each convention.
   Do not add removal budget to an already measured support union.
8. Do not expand to a moderate matrix merely because some b values are below
   one. Check eta>0.05 and useful savings, including the important mass-decreased
   modes; otherwise provide the no-go explanation supported by the data.

Current permission boundary: no changes to controller mathematical formulations
were requested or made. Offline formula implementation in the attachment was
explicitly requested. Covariance numerical repair was NOT explicitly in that
plan, hence the additional question. No answer had arrived at this checkpoint.

## Additional observed evidence

At the 150-snapshot checkpoint: production statuses were 99 certified,
44 support_not_evaluated, and 7 support_exceeded. There were 53 snapshots with
nonempty removal sets and zero count/uniqueness/removed-in-union/active-in-union
mismatches where support was evaluated. Example seed 77, step 15:
removed=[180,353], union=[75,180,261,334,353,417,508], measured count=7, not 9.
Maximum raw covariance skew was 1.0842021724855044e-18 across 6300 matrices;
maximum absolute covariance entry was 0.06662500000000002.

A separate PARTIAL exact-validator analysis examined 210 accepted
constant_velocity records: fraction b<1=1.0, median=0.008422240216156222,
q90=0.026934700342023124, q95=0.03449431468446847; 55 had decreased probability.
Turning modes remained excluded. These figures are incomplete checkpoint data,
not the final three-configuration distribution. File: partial_exact_modes.json.
They show the blocker is numerical input validation, not universal saturation.

Build logs have been copied from /tmp into results/certification-pilot-validation.
implementation_provenance.json records source and executable hashes. The only
remaining build warnings in the second build were pre-existing unused code in
src/collision_constraints.cpp (lines 315 and 561); no new logger warnings remain.

## Final checkpoint: completed pilot and partial results

All 15 runs exited 0; all reached the configured 30-step limit. All 15 rollout
collision flags were 0. This is not a collision-rate study or a safety proof.
Five seeds 77–81 were used for each of straight/o1/m2, straight/o1/m3, and
s-curve/o1/m2. Horizon remained 20, automatic scenario count 1123, and support
cap/removal budget remained 6/2. The existing matrix configuration generator
was reused; its other overrides, including 25 m road length and predicted
noise enabled, are frozen in pilot_manifest.json and each generated YAML.
No tube or bundle sampling was enabled for these runs.

- 450 returned-attempt snapshots and 1050 obstacle/mode CP metadata rows.
- Production certificate status: 294 certified, 133 support_not_evaluated,
  23 support_exceeded. Success/executability does not imply certification.
- 148 snapshots with removals; zero observed support count, uniqueness, or
  evaluated-set inclusion mismatches. Final active IDs need not equal the SQP
  support union, which retains earlier active/violated IDs and removals.
- 450 constant_velocity mode bounds accepted by the unchanged validator;
  27,000 associated disc/stage probabilities retained (20 stages × 3 discs).
- 600 turning-mode records rejected by the exact symmetry check. Maximum raw
  covariance skew: 1.0842021724855044e-18. No covariance values were altered.
- Zero complete mode-law snapshots evaluated. No transfer LP, eta inversion,
  S_transfer, or moderate matrix executed. This is neither a transfer success
  nor a bound-saturation no-go result: a numerical validation gate remains open.

### Constant-velocity-only distributions (incomplete mode sets)

These rows must not be represented as all-mode results. Every row below has
fraction b<1 equal to 1.0. Quantiles use NumPy's default linear interpolation.

| Configuration | Records | Median b | q90 | q95 |
| --- | ---: | ---: | ---: | ---: |
| ALL | 450 | 0.00684701759414 | 0.0251921970159 | 0.0319541011071 |
| sh_mpcc_dro_s_curve_o1_c1_m2 | 150 | 0.00772599284397 | 0.0252966960051 | 0.0316528162196 |
| sh_mpcc_dro_straight_o1_c1_m2 | 150 | 0.0104139335072 | 0.0318640599124 | 0.0369306119528 |
| sh_mpcc_dro_straight_o1_c1_m3 | 150 | 0.00433665105803 | 0.0128549245837 | 0.0160573456978 |

Across all configurations, constant_velocity probability decreased in 158
records: median b=0.002496736853859552, q90=0.010865011035619085,
q95=0.014425886087225446. Probability increased in 120 records and was unchanged
in 172. The omitted turning modes may change the transfer conclusion entirely.
The largest constant_velocity pair probability was 0.0094085890622846.

### Support convention and baseline counts

Executed unchanged backend commands:

```text
build-certificate/certificate_numeric_backend size 0.05 0.01 6 0
895 0.04998627040121828 0.05003273693590915

build-certificate/certificate_numeric_backend size 0.05 0.01 6 2
1123 0.049997772590821166 0.050034655506853598
```

Output columns are S, epsilon_SH(S), epsilon_SH(S-1). The first command is the
plan's n=6 illustrative formula. The second uses the production total support
cap 8, including removal budget 2. These are baseline counts only. There is
no measured or hypothetical S_transfer yet. Removed scenarios already in a
measured support union must not be added again.

### Artifact map

- `results/certification-pilot/pilot_manifest.json`: settings, exact commands,
  executable hash, seeds, exit codes, final completed status.
- `results/certification-pilot/pilot_audit.json`: consolidated counts and blocker.
- `results/certification-pilot/*/seed_*/rollout/certification/*.json`: immutable
  raw returned-plan snapshots, all stages 0..N, raw covariance, p_hat/q/counts,
  rho and support/removed IDs.
- `mode_metadata_cp.csv`: all 1050 mode records with production CP L/U, raw counts,
  beta_cp, p_hat, q_star and rho. These are marginal single-obstacle CP intervals;
  iid coverage assumptions are not established by logging them.
- `mode_bounds_exact.csv`: empty all-mode result due to numerical exclusions.
  Its `_outcomes.csv` and `_support_audit.csv` retain every snapshot's status.
- `mode_bounds_partial_exact.csv`: only the 450 accepted constant_velocity rows.
- `mode_pairs_partial_exact.csv`: all 27,000 pair probabilities for those rows.
- `mode_exclusions_partial_exact.csv`: the 600 rejected turning-mode rows and skew.
- `mode_bounds_partial_exact_summary.json`: per-case and probability-shift stats.
- `results/certification-pilot-validation/`: raw test/build logs, early checkpoint
  outputs (clearly partial), implementation source and executable SHA256 hashes.

Partial-mode consolidation was a one-off read-only Python analysis using the
same `projected_probability` helper, stages 1..N and every actual disc center,
with `math.fsum` and min(1,sum). Each unsupported mode was retained separately;
no mode was renormalized or assigned an invented bound. CP intervals came from
`certificate_numeric_backend intervals beta_cp count...` for the full mode law.
The reusable batch tool deliberately requires complete mode laws; after an
approved numerical treatment it should replace this partial exploratory output
with a fresh complete analysis, without overwriting these raw/exclusion records.

## Change made

- Files: include/certification_snapshot.hpp; include/types.hpp;
  src/mpc_controller.cpp; src/experiment_artifacts.cpp;
  tools/compute_mode_collision_bounds.py; tools/summarize_mode_bounds.py;
  tools/run_certification_pilot.py; tests/test_certification_snapshot.cpp;
  tests/test_mode_collision_bounds.py; CMakeLists.txt; this handoff file.
- Sections: existing opt-in attempt capture and artifact emission; new standalone
  offline tools and new regression target. The CMake change only adds that target.
- Exact behavior: enabled diagnostics now serialize returned-plan evidence;
  scripts analyze saved evidence without altering controller decisions.
- Smallest appropriate approach: reused existing diagnostic switch, production
  forecast propagation, disc geometry, CP backend and projected-probability helper.
  No new solver, probability model, or production configuration was introduced.
- Existing files had unrelated pre-existing edits. Revert only the named new
  field/call/emission block/target and new files to reverse this task, not entire
  files or the repository's current diff. Generated results may be retained.

## Mathematical impact

No mathematical guarantee or formulation was modified.

The new offline diagnostic applies the requested Gaussian projection and union
bound using existing formulas. Controller predictions, support counting,
constraints, RNG and acceptance are unchanged. Covariance symmetrization was
proposed but has NOT been applied. Existing prior-session tube/bundle/controller
changes must not be attributed to this task. Offline floating-point values do
not establish theorem assumptions, CP iid coverage, intersample safety, or
validity of pre-solve sample reduction.

## Validation performed

Build commands:

```bash
cmake --build build-base --target experiment_runner certificate_numeric_backend test_support_accounting -j 2
# Runner built; command stopped because the cached Makefile lacked the backend target.
cmake -S . -B build-certificate -DDRO_MPC_ENABLE_ROS2=OFF
cmake --build build-certificate --target experiment_runner certificate_numeric_backend test_certification_snapshot test_attempt_diagnostics test_support_accounting -j 2
```

The second configure/build succeeded. Existing unused-code warnings remain in
collision_constraints.cpp. The logger warnings from the initial build were
removed. `git diff --check` produced no output and exited 0.

Tests actually executed:

```bash
build-certificate/test_certification_snapshot
build-certificate/test_attempt_diagnostics
build-certificate/test_support_accounting
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_mode_collision_bounds.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_posterior_certificate.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_wdro_certificate.py
```

All three C++ executables exited 0. Python: 3 + 3 + 3 tests passed. See earlier
checkpoint for the initial new-fixture syntax failure and subsequent correction.
Relevant runtime outputs: seeded removals [3,17] -> union [3,17], count 2;
existing logging-on/off test reports identical controls/outcomes and successive
RNG draws. New synthetic final-center test has expected mode bounds [0,1].

Pilot and all-mode analysis commands:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/run_certification_pilot.py --output results/certification-pilot --seeds 5 --steps 30 --timeout 180
PYTHONDONTWRITEBYTECODE=1 python3 tools/compute_mode_collision_bounds.py --root results/certification-pilot --output results/certification-pilot/mode_bounds_exact.csv
```

The pilot completed 15/15 runs. The all-mode analysis recorded 450 exclusions,
not usable complete bounds. The partial analysis and raw support audit above
were subsequently performed without altering covariances.

## Current status

Candidate fix implemented and test executed; behavior requires user verification.

This status applies to logging and the diagnostic implementation, NOT completion
of the entire requested research plan. Stages 1–3 and the support audit produced
observable evidence but complete bounds remain blocked. Stages 4–6 (transfer,
inversion, hypothetical savings) and stage 8 (conditional expansion) remain undone.

Required next decision: approve or decline offline-only roundoff-scale use of
(Σ+Σᵀ)/2, retaining raw matrices and reporting skew, while rejecting larger
asymmetry or negative eigenvalues. AGENTS.md §1.2 says: "DO NOT modify protected
mathematical code without explicit approval from the user." That is why this
numerical treatment has not been implemented. No user response to that question
had arrived as this final checkpoint was written. After approval, specify and
test the roundoff tolerance explicitly; do not silently repair arbitrary inputs.
