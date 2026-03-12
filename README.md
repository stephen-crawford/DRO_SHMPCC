# OT-SHMPC Paper: Standalone Export

This directory contains the code used for the **Optimal Transport and Safe-Horizon MPC (OT-SHMPC)** paper: core library, paper experiments, and figure generation. It is intended to be exported as a separate repository.

## Dependencies

- **Eigen3** (≥ 3.3): `sudo apt-get install libeigen3-dev` (or equivalent)
- **Python 3** with NumPy, Pandas, Matplotlib (for figure generation)

## Directory layout

```
ot_shmpc_paper/
├── CMakeLists.txt          # Builds library + paper_experiment_runner + tests
├── README.md               # This file
├── include/                # C++ headers (config, MPC, DRO, OT predictor, etc.)
│   └── experiment_harness.hpp  # Canonical rollout API (ExperimentConfig, RolloutRecord)
├── src/
│   └── experiment_harness.cpp  # ALL rollout logic (obstacle sim, collision, path, OT)
├── tests/
│   ├── paper_experiment_runner.cpp  # Experiment configs A-AB, thin wrappers over harness
│   ├── test_dro_framework.cpp       # DRO module validation (H1-H6)
│   ├── test_statistical_power.cpp   # High-power statistical tests (2000 rollouts)
│   └── test_obstacle_class.cpp      # Obstacle class sharing validation
├── scripts/
│   └── generate_results_figures.py  # Reads CSVs, writes figure PNGs
├── paper_figures/          # Created at run time; CSVs and PNGs go here
└── docs/
    ├── PAPER_FIGURES_PIPELINE.md     # Experiment → CSV → figure map, DRO/OT flow
    ├── OT_MODE_COVERAGE.md           # OT and safe-horizon math
    └── OT_SHMPCC_REFERENCE.md        # Complete codebase reference
```

## Build

From the repo root:

```bash
cmake -S . -B build
cmake --build build
```

## Run experiments and generate figures

From the repo root, after building:

```bash
cd build

# Run all paper experiments (A–AB). Writes CSVs to build/paper_figures/
./paper_experiment_runner

# Optional: extra CSVs for fig 10–12 (bootstrap, missed-mode significance, ablation)
./test_dro_framework
./test_statistical_power

# Validate obstacle class sharing
./test_obstacle_class

# Generate figures (reads/writes build/paper_figures/)
python3 ../scripts/generate_results_figures.py
```

Figures (e.g. `fig1_collision_vs_switching.png`) and CSVs will be in `build/paper_figures/`.

To run a single experiment, e.g. A or T:

```bash
./paper_experiment_runner A
./paper_experiment_runner T
```

## Architecture

All rollout logic lives in `experiment_harness.cpp` — obstacle simulation, mode observation tracking, multi-disc collision detection, OT predictor integration, path progress, and multi-obstacle class sharing. The paper experiment runner (`paper_experiment_runner.cpp`) is a thin configuration layer: it maps experiment parameters to `ExperimentConfig`, calls `run_experiment_rollout()`, and writes CSVs.

**Obstacle class sharing**: obstacles assigned to the same class share mode observations. When one obstacle's mode is observed, it is broadcast to all siblings in the same class. This is configured via `ExperimentConfig::obstacles_per_class` and tracked via `ModeHistory::obstacle_class` in the controller.

## What this export includes

- **Core SHMPC library**: scenario sampling, mode weights, DRO (Wasserstein worst-case weights + injection), OT predictor (Sinkhorn, ground cost), safe-horizon truncation, QP solver, collision constraints.
- **Experiment harness**: canonical rollout runner with S-curve paths, multi-obstacle class sharing, OT predictor, and per-step callbacks for experiment-specific behavior.
- **Paper experiment runner**: variants Base, DRO, OT, OT+SH, etc.; experiments A–AB writing CSVs for the paper figures. All rollouts delegate to the harness.
- **test_dro_framework** and **test_statistical_power**: additional CSVs used by fig 10, 11, 12.
- **test_obstacle_class**: validates obstacle class sharing (observation sync, late-join inheritance, class independence).
- **Figure script**: `scripts/generate_results_figures.py` produces all paper figures from the CSVs.
- **Docs**: pipeline description, OT/safe-horizon formulation, and complete codebase reference.

## Exporting as a new repo

From the parent of `ot_shmpc_paper`:

```bash
cp -r ot_shmpc_paper /path/to/new/repo
cd /path/to/new/repo
git init
git add .
git commit -m "Initial export: OT-SHMPC paper code"
```

Or zip:

```bash
zip -r ot_shmpc_paper.zip ot_shmpc_paper -x "ot_shmpc_paper/build/*" "ot_shmpc_paper/paper_figures/*"
```
