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
