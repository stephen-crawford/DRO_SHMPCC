# Paper visualization drafts

Three figures are supplied as vector PDF, editable SVG, and 250 dpi PNG:

- `method_takeaway`: two-panel method and empirical takeaway overview.
- `mode_reweighting`: standalone pipeline, probability comparison, and simplex inset.
- `hybrid_results`: standalone controller architecture and rate comparison.

Regenerate from the repository root with:

```bash
python3 figures/method_takeaway/generate.py
```

Dependencies: Matplotlib and NumPy. This standalone script does not run or change
the controller, confidence bounds, optimization, or Lean proofs.

## Provenance and interpretation

The rates (28.24%, 28.94%, 3.31%) and their description as overall matched-rollout
results come from the user-supplied visualization recommendation. They were not
independently reproduced or located in repository results. No sample counts or
uncertainty estimates were provided; none are inferred. The 24.93 percentage-point
difference is the subtraction of the supplied baseline and hybrid rates. These
are no-admissible-control rates, not collision probabilities or safety guarantees.

The nominal weights [0.70, 0.20, 0.10] and illustrative reweighted weights
[0.35, 0.25, 0.40] also come from that recommendation. They are not an LP solution.
Simplex points use barycentric coordinates consistent with these weights. The
dashed region is schematic: no ground cost, radius, confidence calibration, or
Wasserstein-ball boundary is computed. Its shape must not be interpreted as an
exact ambiguity set. Mode 3 corresponds to the illustrative higher-risk right turn.

The retry diagram follows `MPCController::solve` in `src/mpc_controller.cpp`:
an unsuccessful DRO attempt triggers a nominal attempt in the configured hybrid
mode. Internal recovery details are omitted. Neither attempt is depicted as
guaranteed to succeed; the stop/log outcome follows the harness behavior described
in the repository README. The nominal class belief and trajectory predictions are
separate inputs; the diagram does not imply predictions derive from mode counts.

## Suggested two-column insertion

Requires `graphicx`. Paths below assume the manuscript is at the repository root;
adjust the path when copying to a separate paper project.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/method_takeaway/method_takeaway.pdf}
  \caption{WDRO mode reweighting and hybrid control. (a) Observed mode history
  supports a nominal belief and a statistically calibrated Wasserstein ambiguity
  set. Geometry-aware risk scores guide worst-case mode reweighting for scenario
  sampling. Probability weights and the simplex region are illustrative, not a
  computed optimization example. (b) The hybrid controller retries with baseline
  SH-MPCC after an unsuccessful WDRO attempt. The reported no-admissible-control
  rates are 28.24\% for SH-MPCC, 28.94\% for WDRO, and 3.31\% for the hybrid.}
  \label{fig:wdro-method-takeaway}
\end{figure*}
```

Confirm the rates and their denominator against the final experiment table before
publication. Inspect font sizes at the final column width; the standalone panels
are available for layouts that need more room.

## Execution evidence

The generator was executed and exported all nine artifacts. All three PNG layouts
were visually inspected. The installed Matplotlib emits an unavailable-Axes3D
warning; these figures use only 2D axes and all exports completed. Figure content
and empirical provenance still require author review.
