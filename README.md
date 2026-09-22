# Scenario MPC with Distributionally Robust Obstacle Prediction

This repository contains the scenario-MPC library, its YAML-driven experiment
harness, and a compact deterministic test suite. The harness is the only
rollout implementation: it owns environment construction, obstacle simulation,
mode observations, DRO reweighting, collision accounting, reproducible random
streams, and artifact output.

## Build

The project requires Eigen 3.3+ and an acados installation containing acados,
HPIPM, and BLASFEO.

```bash
cmake -S . -B build -DACADOS_ROOT=/path/to/acados
cmake --build build -j2
```

## Run an experiment

`experiment_runner` is the supported executable. It delegates every rollout to
`run_experiment_rollout()` and produces a self-contained artifact bundle.

```bash
./build/experiment_runner \
  --config configs/quickstart.yaml \
  --seed 77 \
  --output build/artifacts \
  --label quickstart
```

For multiple deterministic rollouts, pass `--rollouts N`. The runner advances
the master seed by one per rollout and writes `summary.csv` at the output root.

The full `configs/default.yaml` is a certified research configuration and can
be substantially more expensive than `configs/quickstart.yaml`.

### Artifact bundle

Each run creates `OUTPUT/RUN_NAME/` with:

- `reproducibility.yaml` — master and derived plant/predictor/controller seeds,
  source revision/tree state, source-config location, method, outcome, and QP
  backend identity.
- `resolved_config.yaml` — the normalized, replayable configuration used by
  the harness. If a YAML source was used, it is also copied as
  `source_config.yaml`.
- `rollout.csv` — the rollout summary and a link to its artifact directory.
- `trace.csv` — initial and post-step ego/obstacle states, modes, clearance,
  radius, solve time, and collision state.
- `rollout.svg` — a dependency-free visualization of the environment roads,
  reference route, ego trajectory, and obstacle trajectories.
- `rollout.gif` — an optional, dependency-free animated replay of the same
  recorded rollout. It uses a small fixed palette and requires no external
  renderer or video tool.
- `scene.csv`, `geometry.csv`, and `rollout.rviz` — optional road/route data,
  exact collision geometry, and an RViz display preset for the ROS 2 replayer.

SVG is deliberately used so artifacts remain inspectable without Python,
Matplotlib, RViz, or image-generation tooling.

Artifacts can also be enabled from YAML:

```yaml
artifact_output_directory: build/artifacts
artifact_run_name: experiment_a_seed_77
artifact_write_manifest: true
artifact_write_trace_csv: true
artifact_write_visualization_svg: true
artifact_write_visualization_gif: true
artifact_gif_frame_stride: 1
artifact_gif_playback_rate: 1.0
artifact_write_rviz_replay: false
```

An empty `artifact_output_directory` keeps direct library calls and unit tests
side-effect free. The CLI supplies an artifact directory by default. It honors
these YAML settings unless an explicit `--svg`, `--gif`, `--rviz`, or matching
`--no-*` option is passed.

### Animated GIF

For a fast visual replay without ROS 2, enable the GIF either in YAML or on the
command line:

```bash
./build/experiment_runner --config configs/quickstart.yaml --gif \
  --output build/artifacts --label gif_demo
```

By default, `artifact_gif_frame_stride: 1` writes every recorded state from the
initial condition through the final post-step state. Larger stride values are
available for deliberately compact previews; their frame delays still preserve
the recorded elapsed-time proportions. `artifact_gif_playback_rate: 1.0`
makes one GIF loop span the actual recorded execution duration; use a larger
positive value to replay faster.
`configs/visualization_demo.yaml` is a short, full-frame GIF-and-RViz-ready
example.

### ROS 2 / RViz replay

The runner never re-simulates inside RViz. With `--rviz`, it records the route,
road centerlines, state trace, and collision geometry; the optional replayer
publishes those exact records as `nav_msgs/Path` and
`visualization_msgs/MarkerArray` messages. Ego collision discs, obstacle radii,
and the configured clearance boundary are drawn from `geometry.csv`.

```bash
# In a shell where your ROS 2 distribution is sourced:
cmake -S . -B build-ros2 -DACADOS_ROOT=/path/to/acados -DDRO_MPC_ENABLE_ROS2=ON
cmake --build build-ros2 -j2 --target dro_mpc_rviz_replay experiment_runner

./build-ros2/experiment_runner --config configs/quickstart.yaml --rviz \
  --output build/artifacts --label rviz_demo
./build-ros2/dro_mpc_rviz_replay --artifact build/artifacts/rviz_demo --loop

# In a second sourced ROS 2 shell:
rviz2 -d build/artifacts/rviz_demo/rollout.rviz
```

RViz replay requires `trace.csv`, so `artifact_write_rviz_replay: true` cannot
be combined with `artifact_write_trace_csv: false`.

The preset uses relative `dro_mpc/...` topic names. This resolves to the
default `/dro_mpc/...` topics automatically; for a namespaced replay, launch
both the replayer and RViz with the same ROS namespace (for example,
`--ros-args -r __ns:=/run1`). Explicit topic remaps can likewise be selected
in RViz or applied consistently to both processes.

## Programmatic use

```cpp
ExperimentConfig config = default_experiment_config();
config.artifacts.output_directory = "artifacts";
config.artifacts.run_name = "baseline_seed_77";

RolloutRecord result = run_experiment_rollout(config, 77u);
std::cout << result.artifact_directory << '\n';
```

The output directory and every seed are recorded in the returned
`RolloutRecord`, so calling code can index artifacts without reconstructing a
path convention.

## Tests

The CTest suite contains focused unit and integration tests only; obsolete
paper sweeps and one-off probes have been removed. It covers the controller,
collision linearization, dynamics, scenario sampling, mode belief, DRO
ambiguity/risk logic, configuration lifecycle, support accounting, artifacts,
reproducibility, class sharing, and velocity bounds.

```bash
ctest --test-dir build --output-on-failure
```

The fifteen full-route base scenarios (MPC, MPCC, SH-MPC, and SH-MPCC with zero
or one obstacle) run with `ctest --test-dir build -L base --output-on-failure`.
See [the base-test instructions](configs/base_tests/README.md) for reproducibility
checks, complete-run GIFs, and RViz replay commands.

`test_experiment_artifacts` is the end-to-end contract for the artifact bundle:
it validates the manifest, resolved config, trace, CSV linkage, SVG, GIF frame
stream, RViz scene/preset, and replayable result from a real harness run.
`test_experiment_runner_cli` verifies that YAML visualization settings are not
overwritten by the runner and that explicit CLI options take precedence.

## Configuration notes

`configs/default.yaml` is the source of numerical defaults. A user overlay is
loaded on top of it, then normalized and validated before controller creation.
The CSV and artifact manifest record both the requested and effective mode-set
information, derived RNG streams, active DRO risk model, ambiguity radius, and
solver identity.

### MPCC progress objective

MPCC and SH-MPCC retain contouring, lag, control-effort, and terminal-heading
penalties, and reward average progress along the reference path over the horizon:

```text
J = sum(contour/lag penalties) + sum(control-effort penalties)
    + terminal-heading penalty - progress_weight * (s_N - s_0) / (N * dt)
```

They have no endpoint-distance or reference-speed tracking term. MPC and SH-MPC
retain their existing objectives. `progress_weight` defaults to 10.0 and is
recorded in `resolved_config.yaml`; it trades progress against the remaining
penalties. Weight 1.0 stalled before the stationary obstacle in the base MPCC
case, while 10.0 completed that diagnostic run with 0.2577 m contouring RMS.

Here `s` is the existing monotone closest-point arc length on the discretized
reference geometry, not an independently optimized virtual progress state.
The condensed QP chains derivatives through the selected projection segments;
at projection kinks it uses the documented active-branch derivative. The
nonlinear rollout/report evaluates actual projected progress. Initial guesses
follow reference headings rather than the endpoint. This preserves the existing
four-state dynamics, two-input interface, collision constraints, and certificate
calculations. `test_mpcc_progress` checks projection derivatives, independence
from endpoint/reference speed, the effect of the reward, and S-curve tracking.

The plan-acceptance candidate rejects infeasible SQP plans and validates the
existing braking fallback against sampled collision rows, road boundaries,
input limits, and velocity bounds before allowing execution. Accepted fallbacks
are explicitly `fallback_not_certified`; the original SQP support union is
preserved. If neither plan is admissible, the harness stops without advancing
the plant and records `termination_reason: no_admissible_control` and the failed
decision number in `reproducibility.yaml`. It does not append frozen frames.

The subsequent linearization candidate prepares a separate copy of the SH
collision-normal anchors with the reference module's lateral push and circle
Douglas–Rachford projection, before pruning and halfspace construction. The
condensed SQP nominal remains a rollout of its controls; position-only projection
is no longer applied to it. Sample counts and support-cap termination are unchanged.
The projection uses a deterministic lateral direction at an exact circle center,
where the reference implementation's normalization is undefined.

With seed 77, all eight base cases complete without recorded collisions and
reproduce exactly excluding elapsed solver time. SH-MPCC with one obstacle takes
137 steps, with minimum disc/obstacle center separation 1.0099 m (threshold
0.95 m), and 137/137 decisions report certified. SH-MPC takes 82 steps with
minimum separation 0.9723 m and 82/82 decisions report certified. The user inspected and accepted these linearization changes.

CTest reports 35/37 passing. The existing obstacle-class test still fails.
`test_plan_acceptance` now fails its requirement that this scene exercise a
fallback: the candidate avoids the obstacle without entering that branch in its
40-step fixture. Its assertions have not been weakened or changed. The new
`test_sh_anchor_preparation` covers lateral normals for a stationary obstacle,
exact-center handling, preservation of the initial state, and an empty scene.

Scenario previews in GIF/RViz are enabled by default with
`artifact_show_sampled_scenarios: true` and `artifact_scenario_preview_count: 8`.
See `configs/base_tests/README.md` for the stationary, dynamic, and dynamic DRO
baselines and the interpretation of the recorded forecast subset.

The `sh_mpcc_no_noise_dynamic_1_obstacles` case additionally disables prediction
noise via `obstacle_prediction_noise: false` (plant noise is separately zero).
Its test checks exact within-decision equality over all samples; the full-set
counts and deviations are recorded in `sampled_scenario_summary.csv`.

The braking fallback now decelerates at up to 1 m/s² only until rest, then holds
zero speed. This user-approved change prevents the fallback from rejecting
itself by predicting negative velocity. Full-horizon collision, road, input and
velocity checks still apply; admitted fallbacks remain uncertified and trigger
fresh MPC solves on subsequent rollout steps. `test_bounded_braking` exercises
the actual fallback at zero, low and cruising speeds, including recorded failure
states, and checks its returned trajectory against vehicle propagation.

Linearized collision constraint diagnostics are enabled by default. Set
`artifact_show_linearized_constraints: false` in YAML, or pass
`--no-linearized-constraints` to `experiment_runner` or
`tests/run_single_obstacle_matrix.py` to disable them (`--linearized-constraints`
enables them explicitly). GIF frames draw all retained horizon/disc collision
boundaries in white, with short ticks pointing into `a.dot(disc_center) >= b`.
The 2 m segments are boundary markers, not finite constraint extents. SVG shows
the latest decision with retained rows; hover a boundary for decision, horizon,
disc and scenario IDs. `linearized_constraints.csv` records all retained rows,
anchors and IDs by decision time. These are disc-space collision half-spaces
before solver state/input mapping, not the complete QP feasible set. RViz is
unchanged. The matrix `--resume` option retains old artifacts; omit it to
regenerate existing cases with this overlay.

The [analysis matrix](configs/analysis_matrix/README.md) provides 480 repeatable
obstacle/class/environment/mode/solver combinations, shared seed schedules,
collision and SH certification rates, complete sampled-mode coverage, timing,
control effort and signed conservatism metrics. Generate or mass-run it with
`python3 tests/run_analysis_matrix.py --output build-base/analysis-matrix`.

### Comparison matrix

The [comparison matrix](configs/comparison_matrix/README.md) searches seed-paired
SH-MPCC versus SH-MPCC with DRO outcomes under distribution shift and mode boost.
It lists cases where non-DRO collides and DRO completes collision-free, alongside
reverse and incomplete outcomes. Run
`python3 tests/run_comparison_matrix.py --output build-base/comparison-matrix`.
