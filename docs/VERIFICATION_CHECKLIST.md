# Paper-to-Code Verification Checklist

This checklist maps each mathematical formulation and empirical claim in the paper to the exact code location that implements it. Items are ordered by risk: formulation correctness first, then experiment configuration, then reported numbers.


## 3. Experiment Configuration (verify these match what the paper reports)

### 3.1 Default Parameters


| Parameter          | Paper value                   | Code location                                          | What to verify           |
| ------------------ | ----------------------------- | ------------------------------------------------------ | ------------------------ |
| Horizon N          | 20                            | `paper_experiment_runner.cpp` `HORIZON`                | Match                    |
| dt                 | 0.1s                          | `paper_experiment_runner.cpp` `DT`                     | Match                    |
| Scenario count S   | 20 (default)                  | `paper_experiment_runner.cpp` `BASE_SCENARIOS`         | Match                    |
| Switch probability | varies                        | Per-experiment                                         | Match reported values    |
| S-curve path       | L=25m, A=3m                   | `experiment_harness.cpp` `create_s_curve(25, 3, 200)`  | Match                    |
| Obstacle placement | 35% arc length                | `experiment_harness.cpp` default `arc_fraction = 0.35` | Match                    |
| Path completion    | 95%                           | `experiment_harness.cpp` / `PATH_COMPLETE_FRAC`        | Match                    |
| Ego initial state  | (0, 0, 0, 1.5)                | `experiment_harness.cpp`                               | Match                    |
| Collision radii    | ego=0.5, obs=0.35, margin=0.2 | `experiment_harness.cpp` MPC config setup              | Verify these match paper |
| Number of modes    | 4 default                     | `ExperimentConfig::obs_modes`                          | Match                    |


### 3.2 Per-Experiment Rollout Counts

These must match the numbers reported in figure captions.


| Experiment                       | Paper claim             | Code location         | What to verify                                               |
| -------------------------------- | ----------------------- | --------------------- | ------------------------------------------------------------ |
| Fig 5 (Scenario budget, Exp T)   | 500 rollouts            | `run_experiment_t()`  | Match `n_rollouts`                                           |
| Fig 6 (Baselines, Exp X)         | per caption             | `run_experiment_x()`  | Match rollout count                                          |
| Fig 7 (Rare sweep, Exp V)        | per caption             | `run_experiment_v()`  | Match rollout count and rare_p values {0.01, 0.05, 0.1, 0.2} |
| Fig 8 (Geometry ablation, Exp Y) | 1000 rollouts           | `run_experiment_y()`  | Match                                                        |
| Fig 9 (Pareto, Exp AB)           | per caption             | `run_experiment_ab()` | Match grid values                                            |
| Fig 10 (Robustness, Exp AA)      | 400 seeds (100 per env) | `run_experiment_aa()` | Match                                                        |
| Fig 11 (Qualitative, Exp Z)      | per caption             | `run_experiment_z()`  | Match                                                        |


### 3.3 Variant Mapping


| Paper variant name | Code variant                         | What to verify                                                     |
| ------------------ | ------------------------------------ | ------------------------------------------------------------------ |
| Base               | `PaperVariant::BASE`                 | `weight_type=FREQUENCY, enable_dro=false, safe_horizon=false`      |
| OT                 | `PaperVariant::OT`                   | `weight_type=WASSERSTEIN, enable_dro=false, use_ot_predictor=true` |
| OT+SH              | `PaperVariant::OT_SH` or `OT_DRO_SH` | Verify which variant is actually labeled "OT+SH" in figures        |
| SH                 | `PaperVariant::BASE_SH`              | `safe_horizon_enabled=true` only                                   |


**Key question**: The paper mentions "OT+SH" as the full method. Verify whether this maps to `OT_SH` (no DRO) or `OT_DRO_SH` (with DRO). The abstract says "OT+SH" but the DRO module is also described.

---

## 4. Reported Numbers (verify against actual experiment output)

After running each experiment, compare CSV output against numbers stated in the paper.

### 4.1 Abstract Claims


| Claim                                               | Where to verify                                                |
| --------------------------------------------------- | -------------------------------------------------------------- |
| "OT attains lowest missed-mode rate (about 16-20%)" | `exp_x_baselines_rare_mode.csv` or `exp_v_rare_mode_sweep.csv` |
| "achieves the best collision rate at 20%"           | Same CSVs, `rare_p=0.2` row                                    |
| "400 total seeds per method" in robustness          | `exp_aa_robustness_per_seed.csv` — verify 400 rows per method  |


### 4.2 Section-Specific Claims


| Section              | Claim                                                                               | CSV to check                     |
| -------------------- | ----------------------------------------------------------------------------------- | -------------------------------- |
| 5.1 (Budget scaling) | "OT reduces missed-mode events at fixed S; SH provides primary collision reduction" | `exp_t_missed_mode_vs_s.csv`     |
| 5.2 (Baselines)      | "OT 4.0% collision at 20%"                                                          | `exp_x_baselines_rare_mode.csv`  |
| 5.2                  | "Uniform 42-46% rare-mode miss"                                                     | Same CSV                         |
| 5.3 (Geometry)       | "Mean-only 8.5% -> W2-Euclidean 6.0%"                                               | `exp_y_geometry_ablation.csv`    |
| 5.4 (Rare sweep)     | Sweep over {1%, 5%, 10%, 20%}                                                       | `exp_v_rare_mode_sweep.csv`      |
| 5.5 (Mode scaling)   | "Base degrades sharply as M increases"                                              | `exp_w_mode_scaling.csv`         |
| 5.7 (Robustness)     | "Intersection yields zero collisions for all methods"                               | `exp_aa_robustness_per_seed.csv` |
| 5.8 (Pareto)         | "OT+SH clusters at 2.5-6% collision"                                                | `exp_ab_pareto_frontier.csv`     |
| 5.8                  | "missed-mode rate ~19-20%"                                                          | Same CSV                         |


---

## 5. Potential Gotchas

These are implementation details that could silently produce wrong results.


| Issue                                             | What to check                                                                                                                                                                                     | Risk                                            |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Normal direction flip**                         | Code uses `ego - obs` for normal; paper uses `obs - ego`. Constraint inequality must be consistent.                                                                                               | HIGH — wrong sign = no collision avoidance      |
| **RK4 vs Euler**                                  | Paper writes Euler (Eq. 2); code uses RK4. Results are from RK4.                                                                                                                                  | LOW — RK4 is more accurate, just note it        |
| **Progress `s` not a decision variable**          | Paper shows `s` in state vector; code tracks it externally via projection. MPCC cost uses projected `s`, not optimized `s`.                                                                       | MEDIUM — may affect contouring behavior         |
| **epsilon vs rho confusion**                      | Sinkhorn `epsilon` (regularization) vs Wasserstein `rho` (ball radius). Different quantities.                                                                                                     | HIGH — wrong parameter = wrong DRO solution     |
| **WeightType::WASSERSTEIN is NOT full OT**        | It's a frequency-recency blend (0.3/0.7). The full Sinkhorn predictor is `OptimalTransportPredictor`.                                                                                             | MEDIUM — paper should clearly state this        |
| **Safe horizon off-by-one**                       | Constraints at step `k >= N_s` are removed. Is step 0 included? Is `N_s` inclusive or exclusive?                                                                                                  | MEDIUM — off-by-one changes the guarantee       |
| **MPCC cost applied to all N, not N_mpcc**        | The comment says "all steps 1..N" but MEMORY.md says cost should be limited to `N_mpcc`. Verify which is actually running.                                                                        | MEDIUM — affects OT+SH performance claims       |
| **Obstacle initial mode**                         | `ObstacleSim` starts in mode `obs_modes[0]` (always "constant_velocity"). This biases early observations.                                                                                         | LOW — first 5 warmup observations mitigate this |
| **Collision detection vs constraint enforcement** | Harness uses Euclidean center-to-center distance for collision detection, but MPC uses multi-disc half-space constraints. A collision by harness metric may not have been a constraint violation. | MEDIUM — affects reported collision rates       |


---

## Recommended Verification Order

1. **Eq. 14-15 normal direction** (§1.4) — highest risk, sign error kills safety
2. **DRO dual** (§1.5) — verify `evaluate_dual` matches Eq. 9 exactly
3. **epsilon/rho parameter naming** (§1.5) — confusion here changes DRO behavior
4. **MPCC cost signs** (§1.2) — contouring/lag error sign conventions
5. **Safe horizon off-by-one** (§1.6) — changes the probabilistic guarantee
6. **Variant mapping** (§3.3) — which code variant is "OT+SH" in figures
7. **Rollout counts** (§3.2) — must match figure captions
8. **Reported numbers** (§4) — run experiments and diff CSVs against paper

