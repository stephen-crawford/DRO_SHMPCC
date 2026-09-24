# Adversarial bundles on `a_priori_pruning`

This opt-in implementation follows the conservative explicit-draw route in the
provided derivation. It changes the scenario unit to an independent bundle.
It does not replace conditional trajectories with a highest-scoring trajectory
or assume an unproved Gaussian order-statistic kernel.

## Controller behavior

For one obstacle with 2–6 held modes and a nonempty categorical history:

1. Obtain simultaneous CP intervals using the existing production backend.
   Intersect with the simplex to obtain each coordinate maximum `U_m`.
2. Before drawing scenarios, freeze `K_m = ceil(c0 U_m)` plus a configured number
   of extras. Extras use deterministic largest-remainder allocation proportional
   to `q_m * max(score_m, 0)`, or uniform allocation when all scores vanish.
   The certification floor remains even when a WDRO weight is zero.
3. Compute `c_K = min(K_m/U_m)` over positive `U_m`, and
   `eta = -expm1(c_K log1p(-epsilon))`.
4. Automatic sizing uses the unchanged de Groot sample calculation with `eta`
   as the **bundle** violation threshold. A manual sample count means bundles;
   insufficient counts remain uncertified.
5. Draw every conditional trajectory independently using the existing sampler.
   Raw trajectory IDs stay unique; a separate bundle ID controls support and
   scenario removal. Removing a bundle removes every internal trajectory.
6. At every constraint rebuild, prune only same-bundle facets with exactly equal
   normals, stage, obstacle and disc map: retain the largest right-hand side in
   `a·c >= b`. No angular tolerance, risk-score dominance or facet cap is used.
   The conjunction of all future-stage halfspaces is preserved.
7. Before accepting the result, check collision clearance against **every raw
   candidate in every unremoved bundle**, including fallback results. Nonfinite
   or incomplete trajectories fail this check. Existing support, feasibility and
   recovery restrictions on certification remain in force.

Example YAML options, added to an otherwise complete existing experiment config:

```yaml
mpc_type: sh_mpcc
num_obstacles: 1
automatically_compute_sample_size: true
bundle_amplification: 2.0
bundle_extra_draws: 2
bundle_beta_cp: 0.05
```

Zero amplification disables bundles. Unsupported multi-obstacle, Markov-jump,
empty-history, external-weight-override and outer nominal-retry uses are rejected,
rather than assigned the single-obstacle bundle certificate. The tube is optional;
the simple amplification argument does not require nontrivial mode-risk caps.

## Build, test, run and verify

```bash
cmake -S . -B build-bundles -DCMAKE_BUILD_TYPE=Release -DACADOS_ROOT=/home/stephen/Documents/ACC_Development/Development/acados
cmake --build build-bundles --target adversarial_bundle_experiment test_adversarial_bundles certificate_numeric_backend -j 3
ctest --test-dir build-bundles -R '^test_adversarial_bundles$' --output-on-failure
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_bundle_certificate.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_adversarial_bundles.py --seeds 3 --output results/adversarial-bundles
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_adversarial_bundles.py --verify-existing results/adversarial-bundles
```

Use a new output directory for each run. The fixture compares baseline sampling
with c0=2 and 4, with zero or two extras, deterministic and noisy predictions,
three seeds and two repetitions. Each rollout has two decisions. Increase
`--seeds` to repeat this same bounded study over additional seeds.

Artifacts include `cycles.csv`, `modes.csv`, `draws.csv`, `plans.csv`, `summary.csv`,
`manifest.json`, `verification.json` and the complete solver log. The verifier
checks repeated data except timing, raw/bundle cardinalities, mode multiplicities,
coverage amplification, saved-plan clearance against every raw draw, complete
horizons and cohort completion. Seeds repeat all saved trajectory coordinates.

## Optional refined threshold and mode pruning

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/analyze_bundle_certificate.py --snapshot configs/certificate_diagnostic/BUNDLE_SNAPSHOT.json --output results/bundle-threshold
```

The standalone analyzer evaluates the simple threshold, explicit sample-count
corollary, CP-vertex refinement using KKT bisection, and the omitted-mode risk
budget. It handles zero multiplicities, zero-probability coordinates, infeasible
unsafe sets, and logarithmic boundaries explicitly. It reports primal/dual gaps.
It requires a stated uniform justification for any supplied `b_m < 1`.
The runtime uses the simple threshold; refined thresholds and mode omission are
diagnostic only. No scalar order-statistic sampling or general dominance kernel
is implemented without the corresponding nesting/amplification proof.

## Observed validation and limits

The initial saved comparison in `build-bundles/comparison` accepted 120/120
decisions. Repeats matched except timing; the artifact verifier found no group,
multiplicity, horizon or raw-collision violations. For this fixture:

| Configuration | Bundles | Raw draws |
|---|---:|---:|
| Baseline | 895 | 895 |
| c0=2, no extras | 388 | 1164 |
| c0=2, two extras | 239 | 1195 |
| c0=4, no extras | 169 | 845 |
| c0=4, two extras | 129 | 903 |

The final expanded run in `build-bundles/validated-comparison` accepted **200/200**
decisions across five seeds with the same count table. Independent saved-artifact
verification also recomputes the CP upper bounds and sample-size formula.
`modes.csv` includes the WDRO weights, scores and ambiguity radius used for allocation.

**This was not a runtime improvement.** The conservative within-bundle pruner
retained far more facets than the existing baseline free-space reducer. In noisy
cases, slightly different normals generally prevent exact pruning. Saved timing
and retained-facet counts expose this cost; no solver-speed or general raw-draw
reduction is claimed.

The tests include actual binding support (three bundles containing multiple raw
candidates count as three), zero-weight coverage floors, noisy conditional draws,
exact conjunction preservation, 1,000 randomized amplification checks, a 100,000
trial detection experiment, and 40 comparisons of the refined threshold with an
independent constrained optimizer. The optimizer test uses analytic derivatives
after finite-difference SLSQP reported a line-search failure; assertions and
tolerances were retained.

The history in this fixture is synthetic. These tests do not prove stationary IID
history, support-theorem eligibility, numerical enclosure or a new formal Lean
theorem. New conditional confidence is `1 - beta_CP - beta_cert` (0.94 here), while
baseline uses beta_cert alone (0.99). Counts are not an equal-confidence empirical
comparison. The existing controller's `certified` status is conditional on its
scenario/support assumptions and, with bundles enabled, additionally on the CP
event and bundle construction. It is not a proof supplied by this simulation.

Status: candidate implementation tested; awaiting user verification of behavior.
