# Easy vehicle-turn and support-forecast checks

Run from the repository root:

```sh
cmake --build build-base --target experiment_runner -j 4
cmake --build build-base --target test_traffic_turns -j 4
ctest --test-dir build-base -R 'test_traffic_turns|test_support_scenario_preview' --output-on-failure
```

`test_traffic_turns` calls the real `ObstacleSim::step` with existing mode models,
zero plant noise and a 2 m/s starting speed. A scripted schedule isolates the
maneuvers: straight for 2 s, left for 5.2 s, straight for 2 s, right for 5.2 s,
then straight for 2 s. It checks every displacement and signed heading increment,
speed preservation, straight-mode heading persistence, and an exact seeded repeat.
The left/right phases change heading by ±1.56 rad (±89.3814 degrees). This is a
plant-only test, not a controller or traffic-rule test. Its static blue marker
is not a driven ego vehicle.

Inspect `build-base/traffic-turn-artifacts/straight_left_straight_right/rollout.gif`,
`rollout.svg`, `trace.csv` and `mode_schedule.csv`. Regenerate this scripted test
using the executable above; the schedule is not a new random-switch policy.

For a 10-second live MPC run demonstrating support forecasts (the separate
`support_preview.yaml` remains the eight-step automated regression fixture):

```sh
build-base/experiment_runner --config configs/traffic_tests/support_demo.yaml \
  --seed 77 --output build-base/traffic-support-preview --label support
```

`artifact_show_support_scenarios: true` (CLI `--support-scenarios`) shows all
joint scenarios identified by `MPCResult::support_scenarios` at that decision.
These are the solver's conservative union of binding/violated scenarios across
SQP iterates, not just the final active set or a newly computed minimal support.
The option takes precedence over white constraint glyphs and ordinary sampled
previews, and is not truncated by `artifact_scenario_preview_count`.

GIF shows the selected forecasts at each decision in bright cyan, with thick
lines and endpoint dots drawn above actor markers so short forecasts remain
visible. Empty support sets intentionally draw no forecasts. SVG shows the latest decision,
including an empty support set without falling back to older forecasts. The
terminal state has no new solve. `support_scenarios.csv` records the exact IDs,
count and whether support was evaluated; `sampled_scenarios.csv` records the
actual displayed forecast points. The raw constraint CSV remains available.
Use `--no-support-scenarios` to restore the existing visualization settings.
RViz is unchanged.

The support-preview regression checks exact ID correspondence, empty support,
no preview-count truncation (including a failed returned solve), CLI overrides,
SVG/GIF output and unchanged numerical traces.

The analysis matrix now orders modes as constant velocity, left turn, right
turn, acceleration, deceleration, stop. Thus 2-mode cases include a left turn and
3–6-mode cases include both turn directions. A 1-mode case necessarily cannot
switch modes. Matrix support-only visualization is enabled in settings.json;
GIFs remain enabled. Existing generated matrix YAML files are frozen: use a new
output directory to generate these settings. Mode dynamics, switching probability,
SH hold timing and solver formulations were not changed. Turns are relative to
the current velocity; these models do not enforce lane choice, junction routing,
traffic lights, or restart a vehicle from exactly zero velocity.

## Interpreting the white constraint view

With `--no-support-scenarios --linearized-constraints`, GIF/SVG use bright white
for k=0/1 and progressively darker gray bands for later prediction stages.
Matching dots mark the returned predicted ego **disc centers**, computed with
each row's horizon index and longitudinal disc offset. Blue remains the executed
history; it must not be compared against the current decision's future rows.
SVG hover text identifies the decision, horizon, disc, obstacle and scenario.
Ticks retain the actual feasible-side orientation; boundaries are not temporally
smoothed. The 2 m segments indicate infinite half-space boundaries, not finite walls.

`linearized_constraints.csv` adds `predicted_disc_x`, `predicted_disc_y` and
`geometric_residual` (`a.dot(predicted_disc)-b`). Missing returned states leave
these fields blank. This is a disc-space diagnostic, not the full solver-mapped
QP residual or a physical collision test. SVG uses the latest decision even if
its constraints are empty, rather than silently displaying an earlier solve.

Example:

```sh
build-base/experiment_runner --config configs/traffic_tests/support_demo.yaml \
  --seed 77 --no-support-scenarios --linearized-constraints \
  --output build-base/traffic-constraint-preview --label staged
```

### Clean half-space display

Predicted-disc circles are no longer rendered in GIF or SVG; their coordinates
and geometric residuals remain in `linearized_constraints.csv`. Boundary markers
are 3 m long and use a thicker stroke. Their centers are obtained by projecting
the current obstacle position onto the exact half-space boundary, reducing sliding
along that line as reference anchors change. The feasible-side normal is unchanged.

In GIF, k=1 groups that disappear fade over two recorded frames (gray = previous
decision context, not a currently enforced row). This fade does not interpolate
constraint normals or bounds. New current rows replace their group's faded history.
Failed-decision highlights retain their existing behavior and do not receive faded
history. Actual scenario changes can still move a current boundary between frames.

### Bounded cyan previews

`artifact_support_preview_count` now limits the rendered joint support scenarios
per decision (default 4). Near-duplicate joint trajectories are suppressed when
all corresponding obstacle/stage points differ by at most 0.1 m. Displayed paths
are actual sampled paths, drawn with thin cyan strokes and no endpoint blobs.
SVG reports displayed versus total support counts. The full support-ID list and
all support forecast points remain in `support_scenarios.csv` and
`sampled_scenarios.csv`; these exports are not capped. This supersedes the earlier
uncapped, thick-line display description. Solver samples and certification are
unchanged. Increase the preview count in YAML when more visual detail is needed.
