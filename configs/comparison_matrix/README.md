# Comparison matrix

Search for seed-paired configurations where **SH-MPCC collides and SH-MPCC with
DRO completes the route without collision**. The suite also retains reverse
outcomes, collision-free incomplete runs, and experiment errors.

## Run

From the repository root:

```sh
cmake --build build-base --target experiment_runner -j 2

# Generate frozen configurations and manifest; no simulations.
python3 tests/run_comparison_matrix.py --output build-base/comparison-matrix --generate-only

# Execute or resume the search.
python3 tests/run_comparison_matrix.py --output build-base/comparison-matrix --resume

# Run both controllers for one comparison case across the configured seeds.
python3 tests/run_comparison_matrix.py --output build-base/comparison-matrix --resume \
  --case shift_and_boost_roundabout_o2_c2_m4
```

Use `--settings path/to/settings.json` for a smaller search or different shift
strengths. `--jobs` defaults to 1; parallel execution changes timing measurements.
`--timeout` is per execution, in seconds (default 1800). A changed executable,
script, generated configuration, settings, jobs, or timeout requires a new output
directory. `--case` selects a pair, always running both controllers. Filtered runs
rewrite top-level reports for that selection and leave other trial directories
intact. Per-trial `result.json` files allow interrupted runs to resume.

## Default search

The independent settings file uses:

- 1–4 obstacles and classes, with classes ≤ obstacles;
- straight highway, S-curve, intersection, and roundabout environments;
- 2, 4, or 6 modes from the ordered catalog;
- seeds 77–86, one execution per controller and seed;
- 450 rollout steps with the existing 95% path-completion criterion;
- four mismatch profiles below.

This produces **480 comparison cases, 4,800 seed pairs, and 9,600 executions**.
Visualization is disabled by default to limit search storage; traces, decision
CSV files, manifests and logs are retained. Common `overrides` apply identically
to both controllers. Increase `repeats` to require same-seed repeatability;
repeats never increase the number of measured seed pairs.

## Mismatch profiles

| Profile | `shift_psi` | `shift_boost` | `boosted_mode` |
|---|---:|---:|---:|
| baseline | 0 | 0 | -1 |
| distribution_shift | 0.25 | 0 | -1 |
| rare_boost | 0 | 0.15 | -1 |
| shift_and_boost | 0.25 | 0.15 | -1 |

These are exploratory settings, not known winning configurations.

The existing harness first applies ordinary mode switching, then:

1. With probability `shift_psi`, it replaces the current mode with a uniformly
   selected available mode (possibly the same mode).
2. With probability `shift_boost`, it overrides that choice with `boosted_mode`.
   Index `-1` means the last available mode. With the default catalog this is
   `turn_left` for two modes, `accelerating` for four, and `stop` for six.

The shift routine runs each decision, even under the ordinary `hold` switching
regime. These options alter plant mode selection; they do not directly perturb
positions or the controller's belief. The controller still observes the actual
mode and updates its history. Thus they introduce a stress on the nominal mode
model but do **not** guarantee persistent belief mismatch, rarity, collisions,
or a DRO advantage. The profile name `rare_boost` describes the intended stress;
it does not establish that the selected mode was empirically rare.

A profile can additionally set `rare_mode` and `rare_mode_probability` to use the
existing rare-switch branch. This branch obeys the normal switching schedule,
before the shift overrides. The named mode must already occur in every configured
mode support, preventing silent enlargement of the mode-count axis. All profile
probabilities must be in [0, 1]; shift keys belong in `mismatch_profiles`, not
common overrides. No solver, risk-bound, or controller formulation is changed.

## Pairing and interpretation

For each pair, the generated YAML differs only in `dro_enabled`, method name and
scenario tag. Both controllers receive the same master seed, derived plant and
controller seeds, mode support, shift settings, placement and solver backend.
Plant behavior remains the existing ego-independent `mode_switching` policy.
The suite checks initial placement and obstacle positions, velocities and modes
at every shared trace step, for every repeat. A pairing mismatch is an ERROR,
excluded from match counts. When one controller stops sooner, only their shared
prefix can be compared; the other controller may continue to completion.

A **target match** requires a measured collision flag for non-DRO, no collision
flag for DRO, and DRO path completion. `no_admissible_control` or `step_limit`
without completion is not successful navigation. Collisions use the harness's
existing collision definition. Collision-free completion is an observed outcome,
not a formal safety certificate; certification and failure diagnostics remain in
the per-controller metrics. Searching and selecting favorable seeds is exploratory
and does not establish a general performance advantage.

## Outputs

- `matches.csv` / `matches.json`: target matches with seed, both artifact paths,
  and side-by-side controller metrics. An empty match CSV means no matches.
- `pairs.csv` / `pairs.json`: every selected seed pair, including target/reverse
  flags, completion, collision, termination, clearance, timing, mode coverage,
  certificate counters, and failure diagnostics prefixed `nondro_` and `dro_`.
- `summary.csv` / `summary.json`: one row per comparison case, ranked by target
  count; target seed lists, reverse counts, incomplete outcomes, rates, and
  measured/error/pending denominators. Rates are null if no pairs were measured.
- `results.json`: complete per-controller repeat results.
- `matrix.json`, `configs/*.yaml`: frozen design and executable/script hashes.
- `<controller>_<pair>/seed_<seed>/`: per-trial result, repeat logs and artifact
  bundles, using the existing analysis-matrix runner and metrics.

Errors and repeat mismatches remain errors, never wins. Rates count repeat zero
once per successful seed pair after all requested repeats and pairing checks pass.
The script exits nonzero on pair errors; finding zero matches is a valid result.

## Targeted validation

```sh
python3 tests/test_comparison_matrix.py
python3 tests/test_comparison_matrix.py --runner build-base/experiment_runner
```

The integration fixture uses a short horizon and four samples solely to exercise
all four profiles, paired evidence, repeatability, resume and manifest checks.
It is not a safety-performance experiment. Unit fixtures separately exercise
positive matches, reverse matches, incomplete runs, and invalid pairing evidence.

## Investigating nominal completion with DRO refusal

Use the [matrix investigation tool](../../tests/MATRIX_INVESTIGATION.md) to scan
saved seed pairs and probe a nominal plan at the DRO run's frozen refusal state
under `p`, `qstar`, and boosted modes. It includes exact replay checks and an
optional frozen-state radius sweep, with commands for roundabout seeds 79 and 85.
