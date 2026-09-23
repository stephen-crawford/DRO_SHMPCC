# Simulation setup and matched results

Open `simulation_setup.pdf` for the vector paper figure, `simulation_setup.svg`
for editable artwork, or `simulation_setup.png` for a preview.

The top row shows the four saved environment geometries and initial states for
WDRO, seed 77, two obstacles, two classes, and three available modes. The bottom
left shows the roundabout at step 60 (6.0 seconds), with a close-up of its first
obstacle and three stored prediction previews. These are recorded trajectories,
not invented mode branches or a full rendering of the sampled scenario ensemble.
The scene is an illustrative selection, not a claim of typical performance.

Road/reference centerlines are recovered from each saved `rollout.svg` using the
uniform renderer transform fitted to the corresponding recorded ego trace.
The script checks reconstruction residuals below 1e-5 SVG pixels. Light road
bands are a visual width treatment around those centerlines, not a depiction of
solver constraints or a reconstruction of every road-boundary inequality.
All maps preserve world aspect ratio; thumbnail scales differ by environment.
Collision-disc sizes and offsets follow the saved geometry and current disc
offset convention. The dotted obstacle ring adds the configured safety margin;
it is not an uncertainty contour or an entire ego-obstacle exclusion region.

## Reports and matching

`matched_report.csv` contains the displayed counts, denominators, and rates by
environment and overall. Data comes from `build-base/analysis-matrix/rollouts.csv`.
Matching uses environment, obstacle count, class count, mode count, and seed.
Only repeat zero and triplets with one OK run for each of the three controllers
are retained. This leaves 2,263 of 2,400 triplets; 137 are excluded. Overall
no-admissible-control counts are 639 (SH-MPCC), 655 (WDRO), and 75 (hybrid), giving
28.24%, 28.94%, and 3.31%. Rates count rollouts, not control decisions. A run
without this termination is not necessarily a completed or collision-free run.

The frozen `matrix.json` describes 720 configurations and ten seeds each.
Classes group shared observation histories; the mode catalog is nested in this
order: constant velocity, turn left, turn right, accelerating, decelerating, stop.
The shown three-mode scenes use the first three. Runs allow up to 450 control
steps and use a 95% route-completion termination criterion. The displayed
snapshot uses a 0.1 s control step, a 20-step prediction horizon, 0.5 m ego-disc
radii at longitudinal offsets -2, 0, and 2 m, 0.35 m obstacle radii, and a 0.1 m
safety margin. Source paths and SHA-256 hashes are in `provenance.json`.

## Reproduce and validation

From the repository root:

```bash
python3 figures/simulation_setup/generate.py
```

Requires NumPy, Matplotlib, and the existing saved analysis-matrix artifacts.
No new simulations are executed, and no controller or proof files are modified.
All three exports were generated and structurally checked. The final PNG was
visually inspected. Counts were independently cross-checked by joining
`summary_per_seed.csv`: 2,263 matched triplets and counts 639, 655, 75. Summed
per-environment counts and denominators agree with the overall report.
Matplotlib emits an unavailable-Axes3D warning in this environment; the figure
uses 2D axes and exports complete.

Suggested caption: Simulation setup and matched controller outcomes. Top:
recorded road layouts and initial states for a common two-obstacle, two-class,
three-mode configuration with seed 77. Bottom left: a recorded WDRO roundabout
scene at 6 s, with collision-disc geometry and selected stored obstacle
predictions. Bottom right: no-admissible-control rates by environment and overall,
restricted to configuration-seed triplets with OK runs for all three controllers.
The scenes illustrate the setup; results aggregate 2,263 matched trials per
controller across the full experiment matrix.

No mathematical guarantee or formulation was modified. Figure interpretation
and suitability for publication remain subject to author review.
