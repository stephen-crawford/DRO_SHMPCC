# Week 1: support-complexity screen

The exploratory result supports further investigation, with important selection
bias and short-rollout limits. No predictor was trained.

- 12,480 historical bundles were excluded: Safe Horizon was disabled and support
  was unevaluated, despite some controller names containing SH-MPCC.
- The existing matrix runner executed 80 new trials across straight/S-curve
  scenes, one/two obstacles, one/three modes, and seeds 77–86.
- Fixed total cap 10, epsilon=0.05, beta=0.01, S=1351; prediction dt=0.2,
  execution dt=0.05, 12 requested decisions per trial.
- 674 observed decisions; 605 certified decisions with median support 2,
  95th percentile 4, maximum 7, and 97.19% support at most 4.
- Closest-approach geometry correlates descriptively with support. Reliable
  held-out predictability and savings at lower scenario budgets remain untested.
- 28 runs terminated without admissible control; 41 executable decisions were
  uncertified. Missing labels were retained as missing, never imputed as zero.
- Zero recorded collisions over short trajectories (at most 0.6 seconds), with
  no path completions. This does not validate a collision-probability guarantee.

[Full report, commands, limitations and validation](../results/week1-support-screen-analysis/README.md)

[Support versus context](../results/week1-support-screen-analysis/support_context.png)

[Support by scene](../results/week1-support-screen-analysis/support_by_scene.png)

[Extracted per-decision data](../results/week1-support-screen-analysis/decisions.csv)

Changes: `tools/analyze_week1_support.py`, two Week 1 matrix settings files,
and this report. The existing controller and runner were reused unchanged.
No mathematical guarantee or formulation was modified. Release algorithm tests
passed, all 80 trials executed without runner errors, and independent extraction
checks matched all 674 records against raw support/status/trace data.

Current status: candidate analysis implemented and tests executed; behavior
requires user verification. This is a provisional go for a larger study.
