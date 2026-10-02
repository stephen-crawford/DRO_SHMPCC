# Updated manuscript implementation — 2 October 2026

## Change made

- **Files/sections:** `src/wasserstein_radius_calibration.cpp` / `finite_sample_wasserstein_radius`, `include/wasserstein_radius_calibration.hpp`; `src/dro.cpp` / CP radius resolution and allocation, `include/dro.hpp`; `src/primal_ot.cpp` / `solve_dominating_ot`, `include/primal_ot.hpp`.
- **Exact behavior:** CP calibration now returns `u_m = min(U_m, 1-sum_{j!=m} L_j)` and passes it into the existing constrained transport optimizer. Previously u was maximized over the larger Wasserstein ball. Rho remains the maximum transport distance over CP vertices. Zero-u coordinates are excluded from the ratio maximum as in the paper.
- **Smallest appropriate change:** reuse the existing CP endpoints and LP; add an optional envelope input so existing explicit outer-ball callers and their regression tests remain unchanged.
- **Files/sections:** `src/mpc_controller.cpp` / obstacle initialization and observation updates. The paper CP-transfer path no longer inherits or broadcasts calibration observations among class siblings. This implements separate obstacle laws without imposing a common-law assumption. Other configurations retain existing class sharing.
- **Tests/documentation:** `tests/test_paper_formulation.cpp`, `tests/check_paper_formulation.py`, CMake test registration, `docs/PAPER_EQUATION_MAP.md`, and the experiment README. No existing test assertion, solver constraint, tolerance, recovery policy, or simulator behavior was modified.

The equation map identifies every major implemented construction and distinguishes numerical checks from theorem assumptions. The full-simplex envelope benchmark is an analytical diagnostic, not a replacement risk optimizer.

## Mathematical impact

**This change affects mathematical behavior and was made only after explicit approval.** The user's request explicitly authorizes changing the code to match this supplied manuscript. The tighter CP envelope changes the Q-risk probability floors, zeta, and potentially the selected q and sample count. Separating observation histories ensures the nominal and CP counts correspond to each obstacle's own law. Unspecified implementation choices remain as implemented.

## Validation performed

Build (ROS environment sourced):

```bash
source /opt/ros/jazzy/setup.bash
cmake -S . -B build-base
cmake --build build-base -j2 --target test_paper_formulation test_distribution_transfer test_primal_ot test_ambiguity_calculation test_ground_cost test_risk_scoring_models test_mode_belief test_scenario_sampler test_sample_complexity test_obstacle_class test_dro_risk_response test_safe_horizon_configuration test_dro_fallback test_reviewer_controls test_experiment_artifacts experiment_runner
```

The build succeeded. Existing warnings in `collision_constraints.cpp` concern an unused variable and an unused function; that file was not edited.

Tests:

```bash
ctest --test-dir build-base -R '^test_(paper_formulation|distribution_transfer|primal_ot|ambiguity_calculation|ground_cost|risk_scoring_models|mode_belief|scenario_sampler|sample_complexity|obstacle_class|dro_risk_response|safe_horizon_configuration|dro_fallback|reviewer_controls|experiment_artifacts)$' --output-on-failure
build-base/test_paper_formulation
python3 tests/check_paper_formulation.py --output results/acc-updated-paper-20261002/independent_audit.json
```

**14/15 CTest tests passed.** The new formulation test and all selected CP, OT, risk, sampler, belief, sample-sizing, and transfer tests passed. The existing `test_obstacle_class` failed three solvability checks with DRO disabled. To isolate the change, the previous controller source from parent commit af5dfc8d was compiled with the same flags, substituted into a copy of the current static library, and linked to the unchanged class test. It failed the same three checks. Other changed calibration/allocator functions are not used on that DRO-disabled path. This comparison is evidence that these failures are not introduced by the new observation gating; it is not a diagnosis or repair of the underlying solvability issue. Both logs are retained, and no test was weakened.

The new C++ test covers the CP envelope versus a zero-cost outer ball, zero observations, one mode, zero-u/zero-q, an analytic constrained risk optimum, the minimum-domination benchmark, live production DRO wiring, same-class obstacle isolation (including late joining), KT probabilities, per-obstacle beta allocation, smallest-S selection and conditional transfer acceptance.

Representative output (printed precision):

```text
PAPER_CP rare_u=0.128337 legacy_u=1 q_rare=0.518139 minimum_zeta=1.26958 joint_zeta=6.4411 S=698
```

Here counts (80,15,5) and identical mode predictions give zero transport cost: the rare-mode CP envelope remains 0.128337, while the previous outer-ball envelope was 1. The live two-obstacle test uses separate reversed counts, a global confidence budget split across the two obstacles, and the updated product factor.

The independent audit covered **120 cases**, 1–5 modes, no-data cases, tiny confidence tails, zero-cost pseudometrics, and flat scores. It independently evaluates CP endpoints using SciPy beta quantiles, coordinate extrema using HiGHS box/simplex LPs, the radius using vertex transport LPs, and the selected risk objective using a separate HiGHS LP. No errors exceeded 1e-7:

| Quantity | Maximum absolute error/residual |
|---|---:|
| CP endpoints | 7.11e-15 |
| Coordinate envelope | 7.11e-15 |
| Radius | 1.48e-14 |
| Optimal risk objective | 2.83e-13 |
| Feasibility/domination | 1.60e-14 |

## Existing-framework runtime checks

Two full rollouts used copies of the existing final experiment configurations, the rebuilt executable, and the original seeds. Exact commands are in `simulation_runs.json`; raw per-step CSVs and logs are in the case directories. These are behavioral checks, not a replacement multi-seed paper study.

| Case | Seed | Completion | Steps | Recorded collisions | Minimum clearance | Max S | Median cycle ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| one_obstacle_s_curve | 77 | 1 | 160 | 0 | 0.0075 m | 5,639 | 15.79 |
| two_obstacle_s_curve | 80 | 1 | 192 | 0 | 0.0084 m | 82,156 | 70.10 |

All 544 logged obstacle decisions passed the probability-floor, transport-budget, normalization, risk-lift, domination, and minimum-factor checks at tolerance 1e-7; maximum residual was 8.88e-16. Conditional transfer acceptance was recorded on 145/160 and 167/192 cycles, respectively. Completion does not mean every cycle was transfer-certified.

The two-obstacle seed 80 previously hit the step limit under the old formulation; this updated single run completed. This does not establish a general performance advantage. The simulator's existing future mode/noise law need not match the paper's held-mode Gaussian kernel assumption, so these are computational/control checks rather than empirical proof of the theorem. The paper's probability bound remains pointwise and conditional on its stated calibration, kernel, independence, and Safe-Horizon assumptions.

## Current status

Candidate fix implemented and test executed; behavior requires user verification. The legacy class-test failure remains recorded. Earlier CSV ZIPs describe the previous formulation and were not overwritten; rerun the full paper matrix before replacing its reported results.
