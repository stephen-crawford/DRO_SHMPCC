# DRO-SHMPCC: Wasserstein-DRO Safe-Horizon Model Predictive Contouring Control

Code for the paper: **"Scenario-based Safe Horizon MPCC with Wasserstein-DRO Mode Reweighting and Scenario Injection for Targets with Switching Dynamics"** by Stephen Crawford and Nora Ayanian, Brown University.

## Dependencies

- **Eigen3** (>= 3.3): `sudo apt-get install libeigen3-dev` (or equivalent)
- **Python 3** with NumPy, Pandas, Matplotlib (for figure generation)
- **CMake** >= 3.14, C++17 compiler

## Directory Layout

```
dro_shmpcc/
├── CMakeLists.txt                  # Builds library + 12 test/experiment binaries
├── README.md                       # This file
├── PAPER_CONTEXT.lock              # Paper formulation lockfile
├── stephen_crawford_WDRO.tex       # LaTeX paper source
│
├── include/                        # C++ headers
│   ├── types.hpp                   #   Data structures (EgoState, Scenario, etc.)
│   ├── config.hpp                  #   ScenarioMPCConfig (all hyperparameters)
│   ├── dynamics.hpp                #   Ego bicycle model + RK4 integration
│   ├── reference_path.hpp          #   Cubic spline reference path for MPCC
│   ├── collision_constraints.hpp   #   Linearized half-space collision constraints
│   ├── scenario_sampler.hpp        #   Scenario generation + mode-coverage sampling
│   ├── mode_weights.hpp            #   Mode weight computation strategies
│   ├── qp_solver.hpp              #   ADMM-based QP solver
│   ├── wasserstein_dro.hpp         #   Wasserstein DRO: worst-case weights + injection
│   ├── mpc_controller.hpp          #   AdaptiveScenarioMPC main controller
│   └── experiment_harness.hpp      #   Canonical rollout runner + experiment config
│
├── src/                            # C++ implementations
│   ├── dynamics.cpp                #   RK4 ego integrator
│   ├── reference_path.cpp          #   Spline path construction
│   ├── collision_constraints.cpp   #   Constraint linearization (multi-disc)
│   ├── scenario_sampler.cpp        #   Scenario sampling logic
│   ├── mode_weights.cpp            #   Weight computation strategies
│   ├── qp_solver.cpp              #   ADMM solver implementation
│   ├── wasserstein_dro.cpp         #   DRO reweighting + injection (largest module)
│   ├── mpc_controller.cpp          #   Controller main loop + SQP solve
│   └── experiment_harness.cpp      #   Rollout harness implementation
│
├── tests/                          # Experiment and test binaries
│   ├── paper_experiment_runner.cpp #   Main paper experiments A-AB (config layer)
│   ├── test_dro_framework.cpp      #   DRO validation tests H1-H6
│   ├── test_obstacle_class.cpp     #   Obstacle class sharing validation
│   ├── test_paper_strategies.cpp   #   Strategy comparison (8-9 variants)
│   ├── test_strategy_sweep.cpp     #   Sweep across 60 conditions
│   ├── test_tuning_sweep.cpp       #   DRO rho & SH min parameter tuning
│   ├── test_paper_figures.cpp      #   Legacy figure generation
│   ├── test_dro_benefits.cpp       #   DRO-specific benefit analysis
│   ├── test_cdc_experiments.cpp    #   CDC paper experiments D1-D4
│   ├── test_path_obstacle_sweep.cpp#   Path geometry variation
│   ├── test_generalization.cpp     #   Generalization experiments
│   └── test_trajectory_dro.cpp     #   Trajectory-level DRO analysis
│
├── scripts/                        # Python figure generation
│   ├── generate_results_figures.py #   Main paper figures from CSVs
│   ├── generate_d_figures.py       #   CDC paper figures D1-D4
│   ├── generate_generalization_figures.py
│   ├── generate_strategy_figures.py
│   ├── generate_dro_benefit_figures.py
│   ├── generate_sweep_figures.py
│   ├── generate_tuning_figures.py
│   ├── generate_injection_count_figures.py
│   ├── generate_algorithm_figure.py
│   ├── generate_q1q2q3_figures.py
│   ├── generate_path_obstacle_figures.py
│   ├── generate_paper_figures.py
│   └── draw_environments.py
│
├── docs/                           # Documentation
│   ├── PAPER_FIGURES_PIPELINE.md   #   Experiment -> CSV -> Figure mapping
│   ├── OT_MODE_COVERAGE.md        #   OT, Wasserstein DRO, safe-horizon math
│   ├── MULTI_SEED_FINDINGS.md     #   Multi-seed experimental findings
│   ├── SWEEP_FINDINGS.md          #   Parameter sweep results
│   └── VERIFICATION_CHECKLIST.md  #   Paper-to-code validation checklist
│
├── cdc_paper_figures/              # CDC paper experimental data (CSVs)
├── generalization_figures/         # Generalization experiment data (CSVs)
├── traj_paper_figures/             # Trajectory paper exports
└── build/                          # Build artifacts (created by cmake)
```

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run Experiments and Generate Figures

After building:

```bash
cd build

# Run all paper experiments (A-AB). Writes CSVs to build/paper_figures/
./paper_experiment_runner

# Run a single experiment (e.g. A or T)
./paper_experiment_runner A
./paper_experiment_runner T

# DRO validation tests
./test_dro_framework

# CDC paper experiments (D1-D4)
./test_cdc_experiments

# Strategy comparison and sweeps
./test_paper_strategies
./test_strategy_sweep
./test_tuning_sweep

# Generalization experiments
./test_generalization

# Generate main paper figures (reads/writes build/paper_figures/)
python3 ../scripts/generate_results_figures.py

# Generate CDC figures (reads cdc_paper_figures/)
python3 ../scripts/generate_d_figures.py
```

Figures and CSVs will be in `build/paper_figures/`.

## Architecture

All rollout logic lives in `experiment_harness.cpp` -- obstacle simulation, mode observation tracking, multi-disc collision detection, path progress, and multi-obstacle class sharing. The paper experiment runner (`paper_experiment_runner.cpp`) is a thin configuration layer: it maps experiment parameters to `ExperimentConfig`, calls `run_experiment_rollout()`, and writes CSVs.

The controller (`mpc_controller.cpp`) implements the full MPC solve loop: scenario sampling, DRO injection, safe-horizon truncation, SQP+ADMM QP solve, and scenario pruning.

**Paper variants**: Base, Base+SH, DRO, DRO+SH -- controlled via `ExperimentConfig` fields (`weight_type`, `enable_dro`, `safe_horizon_enabled`, `injection_mode`).

**Obstacle class sharing**: Obstacles assigned to the same class share mode observations. When one obstacle's mode is observed, it is broadcast to all siblings in the same class.
