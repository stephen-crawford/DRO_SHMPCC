# Paper-to-Code Verification Checklist

Maps each mathematical formulation and empirical claim in the paper to the code that implements it. Ordered by risk.

---

## 1. Default Parameters

| Parameter | Paper value | Code location | What to verify |
|-----------|-------------|---------------|----------------|
| Horizon N | 20 | `paper_experiment_runner.cpp` `HORIZON` | Match |
| dt | 0.1s | `paper_experiment_runner.cpp` `DT` | Match |
| Scenario count S | 20 (default) | `paper_experiment_runner.cpp` `BASE_SCENARIOS` | Match |
| Switch probability | varies | Per-experiment | Match reported values |
| S-curve path | L=25m, A=3m | `experiment_harness.cpp` `create_s_curve(25, 3, 200)` | Match |
| Obstacle placement | 35% arc length | `experiment_harness.cpp` default `arc_fraction = 0.35` | Match |
| Path completion | 95% | `experiment_harness.cpp` / `PATH_COMPLETE_FRAC` | Match |
| Ego initial state | (0, 0, 0, 1.5) | `experiment_harness.cpp` | Match |
| Collision radii | ego=0.5, obs=0.35, margin=0.2 | `experiment_harness.cpp` MPC config setup | Match |
| Number of modes | 4 default | `ExperimentConfig::obs_modes` | Match |

## 2. Variant Mapping

| Paper variant | Code variant | Configuration |
|---------------|-------------|---------------|
| Base | `PaperVariant::BASE` | `weight_type=FREQUENCY, enable_dro=false, safe_horizon=false` |
| Base+SH | `PaperVariant::BASE_SH` | `safe_horizon_enabled=true` only |
| DRO | DRO variant | `enable_dro=true, injection_mode=DRO` |
| DRO+SH | DRO+SH variant | `enable_dro=true, safe_horizon_enabled=true` |

## 3. Per-Experiment Rollout Counts

Must match the numbers reported in figure captions.

| Experiment | Paper claim | Code location |
|-----------|-------------|---------------|
| Fig 5 (Scenario budget, Exp T) | 500 rollouts | `run_experiment_t()` |
| Fig 6 (Baselines, Exp X) | per caption | `run_experiment_x()` |
| Fig 7 (Rare sweep, Exp V) | per caption | `run_experiment_v()` |
| Fig 8 (Geometry ablation, Exp Y) | 1000 rollouts | `run_experiment_y()` |
| Fig 9 (Pareto, Exp AB) | per caption | `run_experiment_ab()` |
| Fig 10 (Robustness, Exp AA) | 400 seeds | `run_experiment_aa()` |
| Fig 11 (Qualitative, Exp Z) | per caption | `run_experiment_z()` |

---

## 4. Potential Gotchas

| Issue | What to check | Risk |
|-------|---------------|------|
| **Normal direction** | Code uses `ego - obs` for normal; paper uses `obs - ego`. Constraint sign must be consistent. | HIGH |
| **RK4 vs Euler** | Paper writes Euler; code uses RK4. Results are from RK4. | LOW |
| **Progress `s` tracking** | Paper shows `s` in state vector; code tracks it externally via projection. | MEDIUM |
| **WeightType::WASSERSTEIN** | This is a frequency-recency blend (0.3/0.7), NOT a Sinkhorn OT solver. | MEDIUM |
| **Safe horizon off-by-one** | Constraints at step `k >= N_s` are removed. Verify inclusive/exclusive. | MEDIUM |
| **Collision detection vs constraints** | Harness uses Euclidean center-to-center; MPC uses multi-disc half-space. | MEDIUM |
| **Obstacle initial mode** | `ObstacleSim` starts in `obs_modes[0]` (constant_velocity). Biases early observations. | LOW |

---

## 5. Recommended Verification Order

1. Normal direction sign (collision_constraints.cpp)
2. DRO dual formulation (wasserstein_dro.cpp `evaluate_dual`)
3. DRO parameter naming (rho vs epsilon)
4. MPCC cost signs (contouring/lag error)
5. Safe horizon off-by-one
6. Variant mapping correctness
7. Rollout counts match figure captions
8. Run experiments and diff CSVs against paper
