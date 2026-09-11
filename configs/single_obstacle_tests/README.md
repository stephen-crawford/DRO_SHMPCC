# Single-obstacle SH-MPCC matrix

32 cases: two DRO settings × four reference shapes × four obstacle policies.
All have one obstacle and four available modes: constant velocity, accelerating,
sharp left turn (+0.65 rad/s), and sharp right turn (-0.65 rad/s). Plant noise is
zero; sampled prediction noise remains enabled. Seed 77 is repeated twice.
The S-curve/straight/intersection routes are 25 m long; the roundabout has an 8 m
radius and requires a complete lap. Rollouts retain a 400-step failure budget.

Policies:
- `random_orientation`: random initial heading at the route midpoint, then a
  uniformly selected available mode at every MPC step (selection may repeat).
- `pursuit`: starts 8 m along the route and 3 m laterally away, heading toward ego.
  Selects the mode whose deterministic next position is closest to current ego.
- `path_intersection`: starts 4 m off the route midpoint, heading across its
  normal. Selects modes toward a moving target 4 m ahead on that crossing line.
- `path_following`: starts 8 m ahead on the route, heading along its tangent.
  Selects modes toward a reference point 1 m ahead of its own closest path
  position, so it follows the local geometry. The 8 m gap is an initial offset,
  not a maintained distance from ego.

All start at 1 m/s; the existing plant speed cap is 2 m/s. Pursuit/following choose
among existing modes; no obstacle positions are teleported or clamped to a path.
Markov switching is enabled. DRO cases use the existing `mixture_var` score,
because the held-mode default rejects Markov transition matrices. Risk/sample
parameters otherwise inherit the defaults. These feedback-driven plant policies
are stress fixtures; logged scenario certificates do not establish a probability
guarantee for those policies.

Run from the repository root:

```bash
python3 tests/run_single_obstacle_matrix.py --runner build-base/experiment_runner \
  --output build-base/single-obstacle-artifacts --jobs 4
```

Use `--resume` to retain completed cases and retry errors. Each case gets two
complete artifact bundles, logs, and `test_result.json`. `index.html` and
`results.json` update as cases finish. Tests check repeatability, actual four-mode
plant dynamics, actor counts, GIF frames, completion and collision. FAIL results
and incomplete trajectories remain visible; no solver limits are weakened.

RViz (substitute CASE):

```bash
source /opt/ros/jazzy/setup.bash
build-base/dro_mpc_rviz_replay --artifact build-base/single-obstacle-artifacts/CASE/CASE --loop
rviz2 -d build-base/single-obstacle-artifacts/CASE/CASE/rollout.rviz
```

The bounded braking/hold candidate is used for the final matrix. Earlier attempts
are preserved in `build-base/single-obstacle-artifacts-before-braking-hold`.
A valid braking or stopped fallback permits the next MPC decision; if both the
optimizer and the fully checked fallback are inadmissible, the harness still
reports `no_admissible_control`. A failed optimizer alone is not a reason to
freeze the plant or fabricate a successful control.

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
