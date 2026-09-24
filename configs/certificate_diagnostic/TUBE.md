# Fixed certification tube experiment

This implements recommendation **#2**, as an opt-in experiment. It does not yet
implement an adaptive reduced-scenario certificate or replace the VaR objective.

Set `certification_tube_radius: 0.3` in an experiment YAML to enable a 0.3 m tube.
The default is zero (disabled). Existing default YAML and existing binaries were
left unchanged. Build a separate executable for the new configuration.

At each decision with a previously accepted complete control sequence, the
controller shifts those controls, repeats the final input, and rolls them from
the measured state. It freezes the resulting reference **before scenario draws**.
That reference remains fixed through SQP, homotopy, braking and nominal retry.
Without a previous accepted plan, no tube is active and the configured baseline
scenario budget is used. The budget is unchanged on subsequent decisions too.

For every future stage and vehicle disc, the QP receives a linearization of
`||c - c_ref||² <= r²`. Line search and final acceptance check the exact Euclidean
disc distance, including heading-dependent disc offsets. There is no added
acceptance tolerance. A failed check rejects execution and suppresses the horizon
certificate before updating the accepted-plan cache. Recovery may still propose
an outside candidate; the final guard rejects it. This can increase inadmissibility.

For each held affine Gaussian mode, before drawing scenarios, the diagnostic uses
the fixed reference normal and projected mean `R - n·(mu - c_ref) + r`, then sums
the Gaussian upper tails over future stages/discs and caps the result at one.
Coincident means use the unit x direction; zero variance retains strict overlap.
Switching or nonlinear body-frame modes conservatively receive `b = 1`.
The bounds are conditional on the supplied model and exact tube containment;
floating-point evaluation is not a verified numerical enclosure.

## Build and run

```bash
cmake -S . -B build-tube -DCMAKE_BUILD_TYPE=Release -DACADOS_ROOT=/home/stephen/Documents/ACC_Development/Development/acados
cmake --build build-tube --target certification_tube_experiment test_certification_tube certificate_numeric_backend -j 3
ctest --test-dir build-tube -R '^test_certification_tube$' --output-on-failure
python3 tests/run_certification_tube.py --backend build-tube/certificate_numeric_backend --scenarios 1123 --output results/certification-tube-baseline-budget
```

Use a new output directory each time. `--scenarios 40` gives the smaller smoke
test, deliberately below the baseline scenario requirement. The fixed fixture
has one obstacle, two held affine Gaussian modes (stationary and lateral cut),
position process noise 0.02 per coordinate, horizon 8, three ego discs, and four
closed-loop decisions. Every arm uses the same measured obstacle sequence and
synthetic 95:5 history. Nominal resampling and WDRO both use the hybrid retry
path; radii are disabled, 0.1, 0.3 and 0.75 m. Seeds 77–79 each run twice.
An inadmissible decision ends its rollout; the controller never executes it.

Artifacts:

- `manifest.json`: binary/source hashes, fixture settings, completion and repeat checks.
- `cycles.csv`: acceptance, active/rejected tube, maximum displacement, recovery,
  actual sample count, existing certificate status, mode bounds and actual final-attempt p/q.
- `disc_geometry.csv`: final and fixed-reference disc centers at every future stage.
- `summary.csv`: acceptance and tube rejection counts per arm/radius.
- `conditional_sample_counts.csv`: hypothetical transfer inversion using those
  pre-sampling bounds, explicitly labeled `certificate_issued=False`.
- `experiment.log`: pre-sampling bounds and solver/recovery output.

Repeat comparisons include every logged disc position, bound, sampling weight,
status and other CSV field except elapsed time and repeat identity.

## Interpretation limits

The mode history is a synthetic fixture, not independently observed IID evidence.
The hypothetical transfer calculation assumes the existing scenario theorem and
support hypotheses; the tube alone does not establish them. It uses beta_CP=0.05
and beta_cert=0.01 (combined union budget 0.06), whereas the baseline sample count
uses beta_cert=0.01 alone. The first pilot report assumed two removals (1123);
logging the actual fixture settings showed zero removals and a baseline of 895.
The extended diagnostics use the logged support/removal settings. Ratios are **not equal-confidence performance
comparisons**. A target already met by the direct mode bound can yield a very
small hypothetical scenario count without demonstrating any WDRO advantage.

The next decision should depend jointly on admissibility, progress, useful mode
bounds and confidence-matched sample requirements. Do not enable live reduced
sampling from these diagnostic ratios. Existing Safe-Horizon status is logged
separately and can remain uncertified even at the baseline budget (for example,
because of support limits or fallback). No new certificate is issued here.

## Validation

The targeted C++ test checks the actual condensed QP tube rows against control
finite differences (including the inequality sign), exact boundaries, heading effects, malformed
horizons, Gaussian tails, cold/warm controller calls, a real nominal retry with an
active tube, and conservative handling of nonlinear modes. Existing fallback and
reviewer-control regression tests also passed in `build-tube`.

The 40-scenario pilot in `build-tube/pilot-final` produced 188 decisions, 186
accepted. All repeats matched except timing, and no accepted plan left its tube.
WDRO at 0.1 m rejected two decisions (one seed repeated twice); all other arms
accepted all 24 attempted decisions. This is evidence of a feasibility tradeoff,
not demonstrated safety or sample-efficiency improvement.

The 1123-scenario run in `build-tube/validation-budget` produced 192/192 accepted
decisions, identical repeats except timing, and zero accepted tube violations.
Existing controller status reported 156 `certified` and 36 `support_not_evaluated`;
these are existing status labels, not independent validation of the theorem.
The hypothetical ratios still ranged from about 0.008 to 2.623, and the smallest
ratios occurred for nominal as well as WDRO sampling. This pilot therefore does
not establish a WDRO-specific sample-efficiency benefit.

Status: candidate implementation tested; user verification of behavior pending.

## Extended decomposition study

The runner now includes radii 0.2, 0.4 and 0.5 m, all attempt weights and radii,
saved Gaussian predictions, support settings and an independent geometry report.
The same bound vector is evaluated under nominal and actual attempt weights.
`decomposition.csv` records both weighted bounds, confidence-region transfer
penalties and hypothetical counts; `transfer_curves.csv` records the transfer
function; `radius_summary.csv` and `radius_decomposition.pdf` compare radii.
This is the existing **CP-box** transfer diagnostic, not a Wasserstein-ball solver.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/run_certification_tube.py --scenarios 1123 --seeds 5 --cycles 6 --output results/tube-decomposition
PYTHONDONTWRITEBYTECODE=1 python3 tools/report_tube_decomposition.py --input results/tube-decomposition --output results/tube-decomposition-independent-check
PYTHONDONTWRITEBYTECODE=1 python3 tests/test_tube_decomposition.py
```

The saved 1123-budget study (`build-tube/decomposition-study`) checked 1200
accepted active-tube mode inequalities with no exceedances. The reviewed report
with the actual support settings is in its `reviewed-decomposition` directory.
The 40-budget study is in `build-tube/decomposition-low-budget`; it checked 1180
accepted active-tube mode inequalities without exceedances. Repeated seeds are
excluded from the diagnostic fraction denominators. Cold starts and disabled
tubes are explicitly not applicable to uniform tube-bound comparisons.
