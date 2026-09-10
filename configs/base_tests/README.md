# Full-route base tests

Fifteen scenarios exercise the existing canonical rollout harness: MPC and SH-MPC
on a 25 m straight route, MPCC and SH-MPCC on the default 25 m, 3 m amplitude
S-curve, each with zero or one obstacle. The one-obstacle case places a stationary
obstacle at (12.5, 0), on the route. It uses only the stop mode and zero plant
noise. This is a controlled base case, not a stochastic safety study.

Each test runs seed 77 twice, compares every trace field except wall-clock solve
time, requires 100% projected route progress and no recorded collision, checks
actor counts and stationary obstacle states, and parses the GIF to require one
frame for every recorded state including the initial and final states. A 400-step
budget is a failure limit, not a substitute for completion. Completion uses the
existing harness projection criterion; it does not imply stopping at the goal.

All solver settings, vehicle geometry, risk parameters, support cap, and automatic
Safe-Horizon sample sizing inherit from `../default.yaml`. No reduced scenario
budget or weakened certificate is used for speed. Certificate counts are reported
separately; completing a rollout does not prove a probabilistic guarantee.

From the repository root:

```bash
source /opt/ros/jazzy/setup.bash
cmake -S . -B build-base -DCMAKE_BUILD_TYPE=Release \
  -DACADOS_ROOT=/path/to/acados -DDRO_MPC_ENABLE_ROS2=ON
cmake --build build-base -j2 --target experiment_runner dro_mpc_rviz_replay
ctest --test-dir build-base -L base --output-on-failure -j2
```

Each test keeps its results in
`build-base/base-artifacts/CASE/CASE/`, with the duplicate in `CASE_repeat/`.
`base_test_result.json` reports the checks and outcome. `reproducibility.yaml`
records seed streams and solver/source identity; `resolved_config.yaml` captures
the numerical configuration. Archive these with the source and acados revision
when comparing machines. GIFs are reproducible; solve timings are not.

Open `rollout.gif` in an image viewer. To view the exact same trace in RViz,
replace CASE below with any YAML filename stem (for example `sh_mpcc_1_obstacles`):

```bash
source /opt/ros/jazzy/setup.bash
case_name=sh_mpcc_1_obstacles
artifact="build-base/base-artifacts/$case_name/$case_name"
./build-base/dro_mpc_rviz_replay --artifact "$artifact" --loop
```

In a second sourced terminal:

```bash
rviz2 -d build-base/base-artifacts/sh_mpcc_1_obstacles/sh_mpcc_1_obstacles/rollout.rviz
```

Inspect the complete ego trajectory, S-curve following, obstacle clearance discs,
and final state. Failed runs also keep GIF/RViz artifacts; a generated animation
alone is not evidence of successful completion. User inspection remains required.

MPCC and SH-MPCC use the progress objective described in the repository README.
Inspect S-curve tracking error in `rollout.csv` (`mean_contouring_err` is RMS).
`test_mpcc_progress` separately requires unobstructed S-curve completion with
contouring RMS below 0.5 m, an engineering regression criterion for this route.
`certificate_rate` is the observed fraction of certified decisions, not an
assertion that every decision certified.

For an automated ROS publishing and scenario-color check after the bundles exist:

```bash
source /opt/ros/jazzy/setup.bash
python3 tests/check_base_rviz.py --build build-base
```

This starts each replayer in turn and compares every position in its complete
published ego path against `trace.csv`. It requires local ROS networking and
Python ROS bindings. Run it with no other publisher on `/dro_mpc/ego_path`.
It checks transport/data consistency; use RViz itself for visual inspection.

The six additional `*_dynamic_1_obstacles` configurations use a constant-velocity
obstacle starting at (12.5, 0) with velocity (0.5, 0) m/s, with no plant noise or
mode switches. Four duplicate the controller variants; `sh_mpc_dro_dynamic` and
`sh_mpcc_dro_dynamic` enable the existing DRO implementation with inherited
radius settings. With one available mode, DRO cannot shift probability between
competing modes; these are integration baselines, not multimodal robustness tests.

Scenario previews are on by default in GIF and RViz artifacts:

```yaml
artifact_show_sampled_scenarios: true
artifact_scenario_preview_count: 8
```

The preview selects evenly spaced entries from the actual controller scenario
set, without additional random draws. Lines share their obstacle's color. Each
forecast is attached to the state where the decision was made; the final frame
has no forecast unless a decision was attempted there. `sampled_scenarios.csv`
records step, obstacle ID, scenario ID, horizon step and world coordinates.
Zero-obstacle scenes have no preview data. Setting the option false disables
capture and rendering. The subset is for visualization, not a depiction of every
constraint or a confidence region.

`sh_mpcc_no_noise_dynamic_1_obstacles` is the single-mode, noise-free diagnostic:

```yaml
obs_modes: [constant_velocity]
obstacle_process_noise: 0.0
obstacle_prediction_noise: false
```

The plant and prediction models both have zero process noise. Prediction noise
remains enabled by default in other tests. This case runs seed 77 twice and
checks that every one of the controller's samples matches at each decision.
`sampled_scenario_summary.csv` records the full sample count and maximum positional
deviation from the first scenario, comparing the same obstacle and horizon step.
The required deviation is exactly zero. The preview lines therefore overlap in
GIF/RViz. Forecasts advance as the obstacle moves between decisions; equality is
within each decision, not between different decision times.

```bash
ctest --test-dir build-base -R '^base_sh_mpcc_no_noise_dynamic_1_obstacles$' --output-on-failure
```
