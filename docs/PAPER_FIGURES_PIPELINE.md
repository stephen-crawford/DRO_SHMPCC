# Paper Figures Pipeline: Complete Guide

This document explains the **end-to-end pipeline** for generating the paper figures: which experiments run, which files and functions they use, how **DRO** and **OT** are integrated into the **SHMPC** framework, and how to trace and analyze each step.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Experiment → CSV → Figure Mapping](#2-experiment--csv--figure-mapping)
3. [Where DRO and OT Live in SHMPC](#3-where-dro-and-ot-live-in-shmpc)
4. [File and Function Reference](#4-file-and-function-reference)
5. [How to Run and Regenerate](#5-how-to-run-and-regenerate)

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Run C++ experiments                                                │
│  cpp_mpc/build/paper_experiment_runner [A|B|C|...|AB]                      │
│  → Writes CSV files to cpp_mpc/build/paper_figures/                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Generate figures                                                   │
│  python3 ../scripts/generate_results_figures.py   # run from build/         │
│  (run from cpp_mpc/ or cpp_mpc/build/; reads paper_figures/*.csv)           │
│  → Writes PNGs to paper_figures/                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **Experiments** are implemented in [`cpp_mpc/tests/paper_experiment_runner.cpp`](../tests/paper_experiment_runner.cpp). Each experiment (A, B, C, …) runs many rollouts, aggregates collision/missed-mode/solve-time statistics, and writes one or more CSVs under `paper_figures/`.
- **Figures** are implemented in [`scripts/generate_results_figures.py`](../scripts/generate_results_figures.py). Each figure function (e.g. `fig1_collision_vs_switching`) loads specific CSV(s), plots them, and saves a PNG.

Optional: some CSVs are also produced by [`cpp_mpc/tests/test_dro_framework.cpp`](../tests/test_dro_framework.cpp) (e.g. `exp_h1_mode_coverage.csv`, `exp_h3_safe_horizon.csv`) and [`cpp_mpc/tests/test_statistical_power.cpp`](../tests/test_statistical_power.cpp) (e.g. `exp_h1_bootstrap_ci.csv`, `exp_h2_missed_mode_significance.csv`, `exp_h4_ablation_table_1000.csv`). The figure script reads from the same `paper_figures/` directory.

---

## 2. Experiment → CSV → Figure Mapping

| Fig | Figure function | CSV(s) used | Experiment(s) that produce CSV |
|-----|------------------|-------------|----------------------------------|
| 1 | `fig1_collision_vs_switching()` | `exp_a_collision_vs_switching.csv` | **A** (`run_experiment_a`) |
| 2 | `fig2_mismatch_vs_time()` | `exp_a_w2_vs_time.csv` | **A** |
| 3 | `fig3_safety_performance_pareto()` | `exp_a_missed_mode_rate.csv`, `exp_a_collision_vs_switching.csv` | **A** |
| 4 | `fig4_rare_mode_collision()` | `exp_b_collision_given_rare.csv` | **B** (`run_experiment_b`) |
| 5 | `fig5_solve_time_distributions()` | `exp_c_solve_times.csv` | **C** (`run_experiment_c`) |
| 5b | (same script) | `exp_c_safety_vs_runtime.csv` | **C** |
| 6 | `fig6_ablation_table()` | `exp_a_ablation_table.csv` | **A** |
| 7 | `fig7_calibration_plot()` | `exp_d_calibration.csv` | **D** (`run_experiment_d`) |
| 8 | `fig8_buffer_sensitivity()` | `exp_e_buffer_sensitivity.csv` | **E** (`run_experiment_e`) |
| 9 | `fig9_conservatism_smoothness()` | `exp_g_conservatism_metrics.csv` | **G** (`run_experiment_g`) |
| 10 | `fig10_forest_plot()` | `exp_h1_bootstrap_ci.csv` | **F** (writes bootstrap), or test_statistical_power (H1) |
| 11 | `fig11_missed_mode_significance()` | `exp_h2_missed_mode_significance.csv` | test_statistical_power (H2) or test_dro_framework |
| 12 | `fig12_ablation_table_1000()` | `exp_h4_ablation_table_1000.csv` | test_statistical_power (H4) |
| 13 | `fig13_full_ablation()` | `exp_h_ablation_full.csv` | **H** (`run_experiment_h`) |
| 14 | `fig14_sh_scaling()` | `exp_i_sh_scaling.csv` | **I** (`run_experiment_i`) |
| 15 | `fig15_ot_vs_baselines()` | `exp_j_ot_vs_baselines.csv` | **J** (`run_experiment_j`) |
| 16 | `fig16_environment_generalization()` | `exp_k_environment_generalization.csv` | **K** (`run_experiment_k`) |
| 17 | `fig17_empirical_violation()` | `exp_l_empirical_violation.csv` | **L** (`run_experiment_l`) |
| 18 | `fig18_sh_sweep()` | `exp_m_sh_sweep.csv` | **M** (`run_experiment_m`) |
| 19 | `fig19_runtime_scaling()` | `exp_n_runtime_scaling.csv` | **N** (`run_experiment_n`) |
| 20 | `fig20_distribution_shift()` | `exp_o_distribution_shift.csv` | **O** (`run_experiment_o`) |
| 21 | `fig21_tradeoff_plane()` | `exp_h_ablation_full.csv` | **H** |
| 22 | `fig22_coverage_baselines()` | `exp_p_coverage_baselines.csv` | **P** (`run_experiment_p`) |
| 23 | `fig23_ot_ablation()` | `exp_q_ot_ablation.csv` | **Q** (`run_experiment_q`) |
| 24 | `fig24_mode_coverage()` | `exp_r_mode_coverage.csv` | **R** (`run_experiment_r`) |
| 25 | `fig25_missed_mode_vs_s()` | `exp_t_missed_mode_vs_s.csv` | **T** (`run_experiment_t`) |
| 26 | `fig26_ground_cost()` | `exp_u_ground_cost.csv`, `exp_u_paired.csv` | **U** (`run_experiment_u`) |
| 27 | `fig27_rare_mode_sweep()` | `exp_v_rare_mode_sweep.csv` | **V** (`run_experiment_v`) |
| 28 | `fig28_mode_scaling()` | `exp_w_mode_scaling.csv` | **W** (`run_experiment_w`) |
| 29 | `fig29_baselines_rare_mode()` | `exp_x_baselines_rare_mode.csv` | **X** (`run_experiment_x`) |
| 30 | `fig30_geometry_ablation()` | `exp_y_geometry_ablation.csv`, `exp_y_paired.csv` | **Y** (`run_experiment_y`) |
| 31 | `fig31_qualitative_rollouts()` | `exp_z_qualitative_trajectories.csv` | **Z** (`run_experiment_z`) |
| 32 | `fig32_robustness_boxplots()` | `exp_aa_robustness_per_seed.csv` | **AA** (`run_experiment_aa`) |
| 33 | `fig33_pareto_frontier()` | `exp_ab_pareto_frontier.csv` | **AB** (`run_experiment_ab`) |

**Experiment entry point:** [`paper_experiment_runner.cpp`](../tests/paper_experiment_runner.cpp) `main()` (around line 3848) — `should_run("X")` runs only experiment X; no argument runs all.

---

## 3. Where DRO and OT Live in SHMPC

This section explains **how DRO and OT are integrated** into the Safe-Horizon MPC (SHMPC) flow so you can trace and analyze them in code.

### 3.1 High-level control flow (one MPC step)

```
1. Mode observations updated (controller.update_mode_observation)
2. Reference trajectory warmstart (initialize_reference_trajectory)
3. Scenario sampling:
   - If custom_per_obstacle_weights_ set → sample_scenarios_with_weights(..., per_obs_weights)
   - Else if ensure_mode_coverage → sample_scenarios_with_mode_coverage(..., weight_type)
   - Else → sample_scenarios(..., weight_type)
   (weight_type can be FREQUENCY, WASSERSTEIN, etc.)
4. Safe horizon (pre-DRO): N_safe = compute_safe_horizon(S_pre)
5. If enable_dro: DRO worst-case weights → inject 1 scenario per obstacle
6. Prune dominated scenarios
7. Optional: enforce_scenario_count (add scenarios if S < S_required)
8. Safe horizon (final): effective_horizon = compute_safe_horizon(S_for_sh)
9. Truncate constraints to steps 0..effective_horizon-1
10. Build linearized constraints → QP solve → return first control
```

### 3.2 DRO (Distributionally Robust Optimization)

**Purpose:** Within a Wasserstein ball around the nominal mode distribution, find worst-case mode weights and **inject one additional scenario** (mean or adversarial trajectory) so the MPC hedges against distribution shift.

**Config (all in [`include/config.hpp`](../include/config.hpp)):**

- `enable_dro` — turn DRO on/off  
- `injection_mode` — `DRO` (mean trajectory) or `ADVERSARIAL` (tail trajectory)  
- `dro_epsilon_base`, `dro_epsilon_min`, `dro_epsilon_max`, `dro_adaptive_epsilon`  
- `dro_risk_sigma_scale`, `adversarial_sigma_scale`

**Where it’s used:**

| Location | What happens |
|----------|----------------|
| [`src/mpc_controller.cpp`](../src/mpc_controller.cpp) constructor (~L19) | If `config_.enable_dro`, builds `WassersteinDRO dro_` from `DROConfig` (epsilon, risk_sigma_scale, etc.). |
| [`src/mpc_controller.cpp`](../src/mpc_controller.cpp) `solve()` (~L141–193) | For each obstacle: (1) nominal weights from `compute_mode_weights(..., config_.weight_type, ...)`; (2) `dro_.set_observation_count(...)`; (3) `dro_.compute_worst_case_weights(nominal, obs_state, modes, reference_trajectory_, ..., pre_dro_safe_horizon, ...)`; (4) `dro_.generate_adversarial_scenario` or `generate_worst_case_scenario`; (5) push one scenario and set `is_injected = true`. |
| [`include/wasserstein_dro.hpp`](../include/wasserstein_dro.hpp) | Declares `WassersteinDRO`, `DROConfig`, `DROResult`; dual (lambda) formulation and ground cost types. |
| [`src/wasserstein_dro.cpp`](../src/wasserstein_dro.cpp) | Implements `compute_worst_case_weights`, `generate_worst_case_scenario`, `generate_adversarial_scenario` (risk per mode, cost matrix, injection trajectory). |

**Paper variants:** In the paper runner, **DRO** is enabled for variants that “use DRO” (e.g. OT+DRO, or DRO-only ablations). The mapping is in `make_experiment_config()`: `cfg.enable_dro = uses_dro(variant)`. All rollout execution delegates to `run_experiment_rollout()` in the harness.

### 3.3 OT (Optimal Transport) in scenario weights

Two distinct uses:

**A) WeightType::WASSERSTEIN inside the MPC (no external predictor)**  
Used when the paper variant is “OT” or “OT+SH” and the controller is not given custom weights.

- **Config:** `weight_type = WeightType::WASSERSTEIN` (e.g. in `run_single_rollout` for OT variants).
- **Implementation:** [`src/mode_weights.cpp`](../src/mode_weights.cpp) ~L102–120: `WeightType::WASSERSTEIN` is a **blend of frequency and recency** (0.3× frequency + 0.7× recency with decay 0.85), not the full Sinkhorn-based OT predictor. So “OT” in the **paper runner** means “WASSERSTEIN weight type” (faster distribution shift reaction than pure frequency).
- **Flow:** `solve()` → when not using custom weights, `compute_mode_weights(hist, config_.weight_type, ...)` is used by `sample_scenarios_with_mode_coverage` or `sample_scenarios` ([`src/mpc_controller.cpp`](../src/mpc_controller.cpp) ~L94–118; [`src/scenario_sampler.cpp`](../src/scenario_sampler.cpp)).

**B) OptimalTransportPredictor (Sinkhorn, ground cost, reshaped weights)**  
Used in experiments that need **geometry-aware** mode weights (e.g. ground-cost ablations).

- **Class:** [`include/optimal_transport_predictor.hpp`](../include/optimal_transport_predictor.hpp), [`src/optimal_transport_predictor.cpp`](../src/optimal_transport_predictor.cpp).
- **Methods:** `observe(obs_id, position, mode_id)`, `advance_timestep()`, `compute_mode_weights(obs_id, available_modes)` (returns weights from Sinkhorn-based reshaping). Optional: `predict_trajectory`, `compute_prediction_error`, `adapt_uncertainty`.
- **Ground cost:** `GroundCostType` (e.g. `SQUARED_EUCLIDEAN`, `FLAT`, `CONSTANT`, `RANDOM_PERMUTED`) sets the cost matrix used in Sinkhorn; see [`src/optimal_transport_predictor.cpp`](../src/optimal_transport_predictor.cpp) (cost matrix construction and Sinkhorn).
- **Integration with MPC:** To use the **predictor’s** weights in the MPC, something must call `controller.set_custom_mode_weights(obs_id, ot_predictor.compute_mode_weights(...))` before `controller.solve()`. That pattern appears in [`src/experiment_harness.cpp`](../src/experiment_harness.cpp) for methods that use custom weights (Conformal, Hazard, Bandit, DoubleDual); the **paper** experiments U and Y create an `OptimalTransportPredictor` with different `GroundCostType`s and run rollouts; in the current code they do **not** call `set_custom_mode_weights` from the predictor, so geometry ablation (Y) is comparing controller behavior with **internal** `WeightType::WASSERSTEIN` while the predictor (with different cost types) is only updated in sync. Full geometry ablation would wire `ctrl.set_custom_mode_weights(0, ot_pred.compute_mode_weights(0, modes))` before each `ctrl.solve()` in those experiments.

**Summary:**

- **DRO:** `enable_dro` → `WassersteinDRO::compute_worst_case_weights` → inject one worst-case scenario per obstacle; risk uses reference trajectory and (optionally) covariance; safe horizon can be passed into DRO as `pre_dro_safe_horizon`.
- **OT (in-paper “OT”):** `weight_type = WASSERSTEIN` → `compute_mode_weights(..., WASSERSTEIN)` → frequency–recency blend used for scenario sampling.
- **OT (full predictor):** `OptimalTransportPredictor` with Sinkhorn and ground cost; if weights are passed via `set_custom_mode_weights`, scenario sampling uses those OT-reshaped weights instead of internal `compute_mode_weights`.

### 3.4 Safe horizon (SH) and how it interacts with DRO/OT

**Purpose:** The scenario-based guarantee holds only over a truncated horizon \(N_{\text{safe}}\) that depends on the number of scenarios \(S\). Constraints beyond \(N_{\text{safe}}\) are dropped so the theoretical \(\varepsilon\)-risk guarantee is valid.

**Config ([`include/config.hpp`](../include/config.hpp)):**

- `safe_horizon_enabled`, `safe_horizon_min`, `forced_safe_horizon`
- `safe_horizon_mode`: `PRACTICAL` (e.g. \(N_{\text{safe}} = \min(N, \lfloor S/(2 n_u) \rfloor)\)), `THEORETICAL_SIMPLE`, or `THEORETICAL_TIGHT`

**Where it’s used:**

- [`include/config.hpp`](../include/config.hpp) `compute_safe_horizon(S_actual, n_u)` — returns \(N_{\text{safe}}\) (clamped to `safe_horizon_min`..`horizon`).
- [`src/mpc_controller.cpp`](../src/mpc_controller.cpp) ~L127–130: `pre_dro_safe_horizon = config_.compute_safe_horizon(S_pre)` (used when calling DRO).
- [`src/mpc_controller.cpp`](../src/mpc_controller.cpp) ~L283–295: `effective_horizon = config_.compute_safe_horizon(S_for_sh)` with `S_for_sh = num_scenarios + dro_injected`; then constraints with `c.k >= effective_horizon` are removed.

So: **DRO** uses the safe horizon to limit the risk horizon when computing worst-case weights; **OT** does not use the safe horizon directly (it only affects which weights or scenarios are used). The **paper** “OT+SH” variant is OT (WASSERSTEIN weights) plus `safe_horizon_enabled = true`.

---

## 4. File and Function Reference

### 4.1 Experiments (C++)

| File | Role |
|------|------|
| [`src/experiment_harness.cpp`](../src/experiment_harness.cpp) | **Canonical rollout engine.** All rollout logic lives here: `run_experiment_rollout()` handles obstacle simulation, mode observation tracking (with class sharing), multi-disc collision detection, S-curve path, OT predictor, path completion termination, distribution shift, and per-step callbacks. |
| [`include/experiment_harness.hpp`](../include/experiment_harness.hpp) | Public API: `ExperimentConfig`, `RolloutRecord`, `ObstacleSim`, `CSVWriter`, statistical helpers (`wilson_ci`, `bootstrap_paired_delta`, `mcnemar_chi2`, `compute_effect_sizes`). |
| [`tests/paper_experiment_runner.cpp`](../tests/paper_experiment_runner.cpp) | **Configuration layer only.** Defines experiments A–AB, maps `PaperVariant` to `ExperimentConfig` via `make_experiment_config()`, and calls `run_experiment_rollout()` through thin wrappers. No rollout logic is duplicated here. |
| [`tests/test_dro_framework.cpp`](../tests/test_dro_framework.cpp) | Extra DRO/SH tests; writes e.g. `exp_h1_mode_coverage.csv`, `exp_h3_safe_horizon.csv`, `exp_h5_ablation_table.csv`. |
| [`tests/test_statistical_power.cpp`](../tests/test_statistical_power.cpp) | High-power statistical tests; writes e.g. `exp_h1_bootstrap_ci.csv`, `exp_h2_missed_mode_significance.csv`. |
| [`tests/test_obstacle_class.cpp`](../tests/test_obstacle_class.cpp) | Validates obstacle class-based mode sharing: sync, late-join inheritance, class independence, multi-obstacle rollouts. |

**How the paper runner delegates to the harness:**

- `make_experiment_config(variant, ...)` — maps `PaperVariant` + parameters to `ExperimentConfig` fields (`weight_type`, `enable_dro`, `safe_horizon_enabled`, `use_ot_predictor`, etc.).
- `run_single_rollout(variant, ...)` — thin wrapper: calls `make_experiment_config()` then `run_experiment_rollout()`, returns `RolloutResult::from_record()`.
- `run_single_rollout_env(variant, ..., env_setup, baseline)` — thin wrapper for custom environments: sets `initial_obstacle_states` from `EnvironmentSetup`, maps `SamplingBaseline` to `WeightType` via `baseline_to_weight()`, injects oracle flood behavior via `step_callback`.
- `run_multi_obstacle_rollout(variant, ..., num_obstacles, num_classes)` — thin wrapper: sets `num_obstacles` and `obstacles_per_class` on `ExperimentConfig`.
- `uses_ot(variant)`, `uses_dro(variant)`, `uses_sh(variant)` — map paper variant to feature flags.
- `run_experiment_a()` … `run_experiment_ab()` — loop over variants/parameters, call wrappers, aggregate, write CSV(s).

### 4.2 Core SHMPC / DRO / OT (C++)

| File | Role |
|------|------|
| [`cpp_mpc/include/config.hpp`](../include/config.hpp) | `ScenarioMPCConfig` (horizon, num_scenarios, weight_type, enable_dro, safe_horizon_*, injection_mode, …); `compute_safe_horizon()`, `compute_required_scenarios_*()`, `compute_effective_epsilon()`. |
| [`cpp_mpc/include/mpc_controller.hpp`](../include/mpc_controller.hpp) | `AdaptiveScenarioMPC`: `solve()`, `set_custom_mode_weights()`, `scenarios()`, etc. |
| [`cpp_mpc/src/mpc_controller.cpp`](../src/mpc_controller.cpp) | Implements sampling (with or without custom weights), DRO injection, safe-horizon truncation, constraint build, QP solve. |
| [`cpp_mpc/include/wasserstein_dro.hpp`](../include/wasserstein_dro.hpp) | `WassersteinDRO`, `DROConfig`, `DROResult`; dual formulation and injection. |
| [`cpp_mpc/src/wasserstein_dro.cpp`](../src/wasserstein_dro.cpp) | `compute_worst_case_weights()`, `generate_worst_case_scenario()`, `generate_adversarial_scenario()`. |
| [`cpp_mpc/include/optimal_transport_predictor.hpp`](../include/optimal_transport_predictor.hpp) | `OptimalTransportPredictor`: `observe()`, `advance_timestep()`, `compute_mode_weights()`, ground cost type. |
| [`cpp_mpc/src/optimal_transport_predictor.cpp`](../src/optimal_transport_predictor.cpp) | Cost matrix construction (by `GroundCostType`), Sinkhorn, mode weight computation from OT. |
| [`cpp_mpc/include/mode_weights.hpp`](../include/mode_weights.hpp) | `compute_mode_weights(mode_history, weight_type, ...)`. |
| [`cpp_mpc/src/mode_weights.cpp`](../src/mode_weights.cpp) | Implements FREQUENCY, WASSERSTEIN (blend), UNIFORM, RECENCY, TEMPERATURE, EPSILON_GREEDY. |
| [`cpp_mpc/include/scenario_sampler.hpp`](../include/scenario_sampler.hpp) | `sample_scenarios()`, `sample_scenarios_with_mode_coverage()`, `sample_scenarios_with_weights()`. |
| [`cpp_mpc/src/scenario_sampler.cpp`](../src/scenario_sampler.cpp) | Implements sampling from mode weights and coverage. |

### 4.3 Figure generation (Python)

| File | Role |
|------|------|
| [`scripts/generate_results_figures.py`](../scripts/generate_results_figures.py) | `load_csv(name)` (reads from `paper_figures/`); `fig1_collision_vs_switching()` … `fig33_pareto_frontier()`; `main()` calls all fig functions. Run from `build/`: `python3 ../scripts/generate_results_figures.py`. Figures saved under `paper_figures/`. |

**Convention:** Figure N uses `load_csv("exp_<letter>_<suffix>.csv")` as in the table in §2. Run the script from `cpp_mpc/` or `cpp_mpc/build/` so that `paper_figures/` is the same directory where the C++ runner wrote the CSVs (often `cpp_mpc/build/paper_figures/`).

---

## 5. How to Run and Regenerate

### Full pipeline: tests + experiments + figures

Run from the **repository root** (`PyMPC/`):

```bash
# 1. Configure and build (if not already built)
cd cpp_mpc
cmake -S . -B build
cmake --build build

# 2. Run the full C++ test suite (CTest)
cd build
ctest --output-on-failure

# 3. Run all paper experiments (writes CSVs to build/paper_figures/)
./paper_experiment_runner

# 4. Optional: run DRO and statistical-power tests (extra CSVs for fig 10, 11, 12)
./test_dro_framework
./test_statistical_power

# 5. Generate all figures (reads build/paper_figures/*.csv, writes *.png there)
python3 ../scripts/generate_results_figures.py
```

After step 5, figures are in `cpp_mpc/build/paper_figures/` (e.g. `fig1_collision_vs_switching.png`, …).

### Build and run experiments only

```bash
cd cpp_mpc/build
cmake --build .
# Run all experiments (writes all CSVs):
./paper_experiment_runner
# Or run a single experiment, e.g. A and T:
./paper_experiment_runner A
./paper_experiment_runner T
```

CSVs are written to `cpp_mpc/build/paper_figures/` (or the path set by `OUTPUT_DIR` in the runner, which is `"paper_figures/"` relative to the current working directory).

### Generate figures only

```bash
# Run from build/ so paper_figures/ is the build output directory
cd build
python3 ../scripts/generate_results_figures.py
```

Output PNGs go to `cpp_mpc/build/paper_figures/` (e.g. `fig1_collision_vs_switching.png`, …).

### Trace a specific figure

1. Identify the figure (e.g. Fig 25) and its CSV (`exp_t_missed_mode_vs_s.csv`) from the table in §2.
2. Find the experiment that writes that CSV: **T** → `run_experiment_t()` in `paper_experiment_runner.cpp`.
3. Open `run_experiment_t()` to see which variants and scenario counts are used, and that it calls `run_single_rollout(v, SWITCH_PROB, S, ROLLOUT_STEPS, seed)`.
4. Follow §3 to see how DRO/OT/SH are set in `run_single_rollout` (variant → `weight_type`, `enable_dro`, `safe_horizon_enabled`) and then inside `AdaptiveScenarioMPC::solve()` (sampling, DRO injection, safe-horizon truncation).
5. In `generate_results_figures.py`, open `fig25_missed_mode_vs_s()` to see how the CSV is loaded and plotted.

This pipeline document, together with the experiment→CSV→figure table and the DRO/OT/SH sections, gives a single place to see how the paper figures are produced and exactly where DRO and OT logic sit in the SHMPC framework.
