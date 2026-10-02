# Writeup-aligned WDRO practicality checks

The default Wasserstein path now selects the paper's constrained optimizer over
`Q_risk`. `use_domination_constraints: false` explicitly restores the existing
legacy allocator switches; the legacy entropic flag is otherwise ignored.
No MPC objective, collision halfspace, support estimator, removal policy,
recovery policy, or downstream solver tolerance was changed.

The controller computes the coordinate envelope directly from the Clopper–Pearson
polytope, `u_m = min(U_m, 1 - sum_{j != m} L_j)`, then maximizes `q.dot(r)` with destination floors
`u_m / max_j(u_j/p_hat_j)` and `p_hat_m` for `r_m >= tau_r`.
`dangerous_risk_threshold` sets `tau_r`; the unoptimized default is **0.01 m**.
Exactly flat scores retain the nominal distribution, a valid tied optimizer.
Failed constrained solves raise an error rather than silently reverting to an
unconstrained distribution. The legacy unconstrained LP remains callable.

For held-mode independent sampling, `confidence_beta` is the **global** ambiguity
failure budget and is divided equally among active obstacles before CP calibration.
The controller freezes all distributions and the product domination factor before
sampling. Automatic sizing chooses the existing support-cap sample count with
risk target `epsilon / product(zeta_v)`. Manual sample budgets remain manual;
an insufficient budget receives no SH certificate. The configured support cap
(including the existing removal budget) is unchanged. This can be expensive when
little calibration data are available or many obstacles are considered.
The CP-transfer path keeps calibration histories separate for each obstacle,
including obstacles assigned the same class; other configurations retain their
existing class-sharing behavior. The Wasserstein radius remains the maximum
transport distance over CP-polytope vertices and constrains allocation only.
See [the equation map](../../docs/PAPER_EQUATION_MAP.md) for the updated paper's
formulation, matching settings, tests, and numerical limitations. Archived
`acc-paper-*-20261002` runs made before this revision use the earlier outer-ball
envelope and must not be relabeled as results of the updated formulation.

`decisions.csv` adds the joint factor, sampling-law target, and
`transfer_bound_satisfied`. The latter means the numerical transfer conditions
and existing SH acceptance checks passed, **conditional on the theorem's data
and model assumptions**. It is not a test of IID calibration, correct kernels, or
closed-loop safety. Fixed radii, Markov predictions, stratified sampling, legacy
allocators, and nominal fallback results do not receive this transfer flag.
`distribution_transfer.csv` logs every per-obstacle envelope, chosen probability,
risk score, factor, radius, and transport cost. `mode_mechanism.csv` retains the
nominal probabilities and actual sampled counts. Use these CSVs for analysis;
the legacy pre-sampling console summary still refers to the configured initial
scenario budget, before transfer tightening.

## Reproduce

```sh
source /opt/ros/jazzy/setup.bash
cmake -S . -B build-base
cmake --build build-base -j2 --target experiment_runner test_distribution_transfer
ctest --test-dir build-base -R '^test_distribution_transfer$' --output-on-failure
python3 tests/check_distribution_transfer_lp.py --output /tmp/transfer-lp.json
python3 tests/run_reviewer_matrix.py --settings configs/paper_practicality/settings.json \
  --output results/paper-basic-new --timeout 300
python3 tests/run_reviewer_matrix.py --settings configs/paper_practicality/scaling.json \
  --output results/paper-scaling-new --timeout 300
python3 tools/report_distribution_transfer.py \
  --matrix results/paper-basic-new results/paper-scaling-new \
  --output results/paper-report-new
```

The optional independent LP audit needs the locally available SciPy and a C++17
compiler. Plotting needs Matplotlib. No new controller runtime dependency is added
(the repository already uses Boost.Math for confidence quantiles).

The first matrix is 3 geometries × 2 methods × 3 paired seeds = 18 rollouts, with
one obstacle and two modes. The scaling matrix is 2 geometries × 2 methods × 2
paired seeds = 8 rollouts, with two obstacles/classes and three modes. Matrices
use the existing canonical harness, full horizon 20, dt 0.1 s, support cap 6 plus
removal budget 2, epsilon 0.05, beta_SH 0.01, beta_DRO 0.05, and automatic sizing.
Completion uses the existing 95% route-progress criterion and a 200-step limit.
Both methods use identical plant streams; the runner checks shared trace prefixes.
The experiments measure outcome counts, full-cycle wall time, 100 ms deadline
misses, sample counts and certificate conditions. Timing runs should be serial
and separated from compilation or other heavy work.

## Interpretation

These are small engineering studies, not powered safety comparisons. Report all
incomplete routes, refusals, errors and deadline misses. Zero observed collisions
does not establish a collision-probability bound. SH certification rates describe
sampling-law certificates; transfer flags retain the theorem's assumptions.

The existing harness draws categorical modes independently at each world step,
whereas each prediction holds its sampled mode over the horizon. Its plant-noise
scale and speed cap also remain as implemented. Consequently the actual future
closed-loop plant trajectory law need not equal the held-mode Gaussian mixture.
Do not use these runs as an empirical validation of the true-law transfer theorem.
The analytical and independent-LP tests isolate the new mathematical construction;
the route simulations assess its computational and control practicality.

Coverage is pointwise in replanning time. No time-uniform calibration or mission
collision guarantee is implemented. The simulations do not remove the product
factor's scaling cost or make uncertified recovery steps certified.

## Feedback ablation and paper figures

`tests/run_paper_ablation.py` adds a three-arm comparison: nominal sampling at
S0, constrained reweighting at fixed S0 (without a true-law transfer certificate),
and full distribution transfer with automatic sizing. It reuses the canonical
runner and only accepts reused results with identical executable/configuration
hashes. Both S-curve cases use seeds 77–81; straight/intersection use 77–79.

```sh
python3 tests/run_paper_ablation.py --output results/paper-ablation-new \
  --practicality results/paper-basic-new --scaling results/paper-scaling-new
python3 tools/report_paper_ablation.py --matrix results/paper-ablation-new \
  --output results/paper-feedback-new
```

The runner retains execution errors and exits nonzero when an attempted run
fails. The report keeps those rows with unavailable final metrics left blank.
Figure 1 preserves the original nine runs per baseline/full-transfer arm;
consult the expanded ablation table alongside it. Figure 2 uses the preselected
one-obstacle S-curve full-transfer seed 77. Summary latency pools usable cycles;
failed prefixes are excluded, and errors remain explicit in the completion table.
