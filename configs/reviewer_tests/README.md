# Reviewer-response experiments

This suite isolates the extra-solve effect from WDRO, audits complete controller
cycle cost, and measures safety-sensitive outcomes. It uses the existing rollout
harness and acceptance checks. The two protected changes (opt-in nominal first
attempt and projected-Gaussian mean) were explicitly approved by the user.

## Controller comparisons

| Variant | First attempt | Retry after inadmissibility | Isolated change |
|---|---|---|---|
| `sh_mpcc` | Nominal | None at outer wrapper | Existing baseline |
| `sh_mpcc_dro` | Calibrated W2-Bures WDRO | None at outer wrapper | Existing direct WDRO |
| `sh_mpcc_resample` | Nominal | Fresh nominal draw | Resampling control |
| `sh_mpcc_dro_fallback` | Calibrated W2-Bures WDRO | Fresh nominal draw | Existing hybrid |
| `hybrid_zero_one` | Calibrated WDRO, unit off-diagonal cost | Fresh nominal draw | Removes trajectory-dependent transport geometry |
| `hybrid_fixed_radius` | W2-Bures WDRO, fixed radius 0.05 | Fresh nominal draw | Removes data-dependent radius calibration |

The main comparison is `sh_mpcc_resample` versus `sh_mpcc_dro_fallback`.
They use the same hybrid type, recovery paths, pre-attempt warm start restoration,
scenario count, horizon, SQP settings, plant stream, and acceptance logic. The
opt-in `nominal_resampling_baseline: true` changes only whether DRO is enabled for
the first attempt. It is rejected for non-hybrid types. Both have at most two
**outer attempts**; each attempt may contain the existing recovery machinery.
This equalizes attempt/scenario budgets, not measured CPU time. Deadline misses
are measured; no new deadline cutoff is imposed on the controller.

Uniform ground cost is **not** uniform mode sampling. Risk-proportional heuristic
sampling and uniform mode oversampling are not implemented. The fixed-radius
arm is a sensitivity experiment without the calibrated-radius coverage claim;
the number 0.05 has not been optimized. A single radius cannot establish general
calibration superiority. Use a fresh settings/output pair for additional radii.

## Experiment design and execution

`settings.json` retains the original four environments, ten valid obstacle/class
combinations, six nested mode supports, and seeds 77–86. Six controller variants
and two plant profiles produce **2,880 configurations / 28,800 scheduled trials**.
Automatic scenario sizing, horizon 20, dt 0.1 s, 450-step limit, and 95% route
completion are preserved. The pilot uses the same numerical settings but only
straight/roundabout, two obstacles/classes, three modes, and seeds 77–78:
**48 trials**. It is a plumbing and sensitivity pilot, not a powered paper study.

`standard` uses the original plant. `shift_and_boost` uses existing plant options:
uniform mode replacement with probability 0.25 followed by a last-supported-mode
override with probability 0.15. The support itself stays unchanged. This stresses
model mismatch; it does not guarantee a rare or dangerous realization. No noise,
collision threshold, solver tolerance, or certificate formula is changed.

From the repository root:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S . -B build-base
cmake --build build-base --target experiment_runner test_reviewer_controls calibration_assumption_audit -j 2

# Inspect the entire frozen design without running 28,800 trials.
python3 tests/run_reviewer_matrix.py --output build-base/reviewer-full --generate-only

# Full-horizon pilot (new output; serial execution for timing).
python3 tests/run_reviewer_matrix.py --settings configs/reviewer_tests/pilot.json \
  --output build-base/reviewer-pilot

# Execute/resume the full suite, or select all variants for one setup.
python3 tests/run_reviewer_matrix.py --output build-base/reviewer-full --resume \
  --pair standard_straight_o2_c2_m3
```

Omit `--pair` to run the full suite. Existing per-trial resume checks include
executable, generator, analysis, pairing code, frozen configurations, and settings
hashes. Changed inputs require a new output directory. Filtered runs rewrite
top-level summaries for that selection. The runner checks all variants' derived
seeds, initial placement, backend, and obstacle trajectories over shared trace
prefixes; a mismatch is an ERROR, not a favorable outcome. Repeats test numerical
reproducibility and never increase the statistical denominator.

## Read-only reports

```bash
python3 tools/report_reviewer_experiments.py --matrix build-base/reviewer-pilot \
  --output build-base/reviewer-pilot-report \
  --methods sh_mpcc sh_mpcc_dro sh_mpcc_resample sh_mpcc_dro_fallback hybrid_zero_one hybrid_fixed_radius

# Essential two-controller comparison with its own complete-pair denominator.
python3 tools/report_reviewer_experiments.py --matrix build-base/reviewer-pilot \
  --output build-base/reviewer-resampling-comparison \
  --methods sh_mpcc_resample sh_mpcc_dro_fallback

# Existing data: no reruns and no changes to historical artifacts.
python3 tools/report_reviewer_experiments.py --matrix build-base/analysis-matrix \
  --output build-base/reviewer-historical-full
```

Use a new empty output directory outside the source matrix. `--outcomes-only`
skips per-decision file reads. `--case-contains` selects a configuration substring.
The default report requires all requested controllers to have OK results on each
configuration/seed. Errors and incomplete groups are excluded with counts.
Legacy reports trust recorded OK status; new experiments additionally run the
shared-plant-prefix checks described above.

Outputs:

- `rates.csv`: no-admissible-control counts/rates, completion, collisions, and
  descriptive 95% bootstrap intervals.
- `paired_overlap.csv`: neither refuses, only A refuses, only B refuses, both
  refuse; paired rate difference and interval. Non-refusal is not called success.
- `cycle_latency.csv`: pooled control-cycle median/p95, conditional fallback
  median/p95, fallback fraction, deadline-miss rate, and rollout clearance tail.
- `rollout_diagnostics.csv`: per-rollout latency, actual scenario-count range,
  executed exposure, signed-clearance tails, penetration, near-collision events,
  and observed mode persistence.
- `summary.json`: selection rules, exclusions, source hash, thresholds, bootstrap
  seed, and interpretation limits.

Bootstrap resamples whole master-seed blocks, preserving within-seed controller
pairing and dependence across configurations. The default is 2,000 draws, seed
1729. These are descriptive percentile intervals, not exact confidence guarantees;
ten seed blocks limit precision, and the two-seed pilot is especially weak for
inference. With fewer than two blocks, intervals are null. Degenerate intervals
at zero events do not prove zero probability. Complete-case selection can bias
results. No unsupported independent-trial significance test is supplied.

The default deadline is 100 ms (`--deadline-ms`). `solve_ms` is measured around
the entire `MPCController::solve` wrapper, starting before the first attempt and
ending after fallback. It includes failed initial attempts and terminal rejected
cycles, but not harness observation/plant propagation or end-to-end I/O. Pooling
uses individual cycles, not medians of rollout averages. Conditional fields are
null if no relevant cycles exist; missing legacy flags are not treated as zero.

Clearance uses saved disc-to-obstacle distances minus ego radius, obstacle radius,
and safety margin. Near-collision means a rollout has a recorded margin **< 0.2 m**
by default (`--near-margin`); safety-envelope violation means **< 0**. Physical
penetration removes the added safety margin. Metrics include initial and executed
recorded states, not continuous-time collision checks or prospective constraint
certificates. Early-stopped runs have less exposure; compare completion and
executed steps alongside clearance and collision rates. Actual scenario counts
are reported rather than assumed (the canonical pilot uses 1,123 per attempt).

## Exact hybrid algorithm

See `MPCController::solve` in `src/mpc_controller.cpp`.

1. Start the controller-cycle timer and increment the decision counter once.
2. Save the previous reference trajectory, feasible controls, and backup flag.
3. Call `solve_attempt` with the configured first-attempt distribution. Its
   ordinary SQP, braking, homotopy, support/removal accounting, and acceptance
   paths run as implemented; the outer wrapper tests the returned `success`.
4. If that result is unsuccessful, restore the saved trajectory/controls/backup,
   clear external mode weights, and call `solve_attempt` with DRO disabled.
   The sampler consumes a fresh draw from the continuing controller RNG stream;
   the RNG is **not reset** to replay the first sample. This is a new pseudorandom
   realization, not an independently seeded controller.
5. Return the second result if attempted, including its failure. The harness stops
   without advancing the plant if no admissible control exists.
6. Record elapsed time across both attempts. The next decision again begins with
   the configured first-attempt policy; there is no permanent switch to nominal.

The original non-hybrid types have different error/recovery handling in some
branches. Reusing the hybrid type for both two-attempt policies controls that
confound. Existing internal braking remains subject to its existing acceptance
and certificate labeling; an executable fallback is not automatically certified.

## Risk, horizon, and statistical-assumption tests

`test_reviewer_controls` exercises a failed first nominal solve followed by a
successful fresh nominal draw (deterministic seed 2), both-attempt rejection,
configuration validation, one decision count, and enclosing cycle timing.
Near-zero tests use negative and perpendicular displacements below 1e-12, exact
zero, and an ordinary displacement. They require `R - n.dot(diff)` in both the
direct surrogate and the held-mode Gaussian mixture. A threat that appears only
at stage N tests that full-horizon evaluation includes the last stage.

The controller requests the full prediction horizon N for DRO. The DRO API also
honors an explicit positive `risk_horizon` override, capped by N; the suite leaves
that override at -1. Bonferroni counts the **effective evaluated horizon × disc
count**. Current controller comments describe full-horizon joint SH certification;
there is no newly introduced shorter N_s in these experiments. This does not
repair notation in a manuscript absent from the repository. State N_s=N for this
setup or explain any distinct manuscript notion with its actual implementation.

```bash
./build-base/calibration_assumption_audit build-base/reviewer-calibration-audit.csv
```

This audit calls the production `finite_sample_wasserstein_radius` API for two
categories with unit off-diagonal transport cost, beta=0.05, 2,000 replications,
and 20/100 observations. It compares independent stationary observations,
stationary labels held in blocks of ten, and a halfway change from p=0.1 to p=0.4.
It reports simultaneous CP-region and raw-radius-ball coverage of the target p.
For the shift case the target is the current p=0.4, not the time-average law.
The two-category transport cost is exactly |p_1-q_1|. This audits the raw
calibration API, not every live controller normalization or a feedback safety law.
It deliberately does not assert that dependent/nonstationary coverage meets 95%.

The harness records one realized mode label per obstacle per world step and pools
class histories. Successive labels may be dependent under held/switching behavior.
Different obstacles sharing a class do not make temporal samples independent.
The diagnostic persistence fraction can expose repeated labels; it is not an
independence test. No history thinning, effective-sample-size substitution, or new
dependence-aware calibration has been silently introduced.

## Inspectable Lean artifact and limits

- [IndependentModeHitsToBinomial.lean](../../certification/DROSafety/DRO/IndependentModeHitsToBinomial.lean):
  `categoricalModeCount_hasLaw_binomial_of_iIndep` connects independence assumptions
  to a binomial count law.
- [ClopperPearsonBonferroniCertified.lean](../../certification/DROSafety/DRO/ClopperPearsonBonferroniCertified.lean):
  exact marginal interval coverage and simultaneous allocation statements.
- [FiniteEnumeratorExactClopperPearsonCoverage.lean](../../certification/DROSafety/DRO/FiniteEnumeratorExactClopperPearsonCoverage.lean):
  `exactClopperPearson_finiteEnumerator_trueDistributionCoverage` packages the
  finite-enumerator ambiguity-set coverage under explicit binomial-law premises.
  The combined true-risk theorem in the same file has additional worst-case/risk
  hypotheses; its presence does not establish those hypotheses for the simulator.

Inspect the complete theorem assumptions, not just their names. Rebuild with
`cd certification && lake build` when reviewing the proof artifact. Proof files
are under active user edits and were not changed or revalidated by this task.
No empirical outcome or scenario certificate is promoted here into a guarantee
under the unknown plant distribution. Ambiguity-set coverage, surrogate risk,
sampling-law certification, and closed-loop collision performance are distinct.

## Validation commands

```bash
python3 tests/test_reviewer_experiments.py --runner build-base/experiment_runner
ctest --test-dir build-base -R '^test_(reviewer_controls|reviewer_experiments|reviewer_experiments_smoke|dro_fallback|safe_horizon_configuration|risk_scoring_models|analysis_matrix|comparison_matrix)$' --output-on-failure
```

The smoke test runs six variants × two profiles × two repeats with horizon 4,
eight manually selected scenarios, and three rollout steps. It checks executed
configuration, equal budgets, pairing, repeatability, resume, report generation,
and stale-manifest rejection. This small-budget smoke test is not safety evidence.
The full-horizon pilot is separate. All behavior remains subject to user review.
