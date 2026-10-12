# Certificate gate: sample-dependent anchor audit

The Week 2 feedback's first gate remains open. This diagnostic reproduces an
unlogged dependence on a scenario through existing anchor preparation. It does
not establish a violation of a probabilistic bound or an exact theorem support
cardinality for the integrated controller.

## Reproduction

`tools/audit_anchor_support.cpp` composes the production anchor preparation,
halfspace construction and private SQP implementation through the existing
CounterfactualProbe friend interface. It does not run the outer certificate
gate. Six deterministic scenarios are deliberately fewer than the configured
certificate sample requirement; these are geometry diagnostics, not certified
experiments. No scenario removal is requested.

Inputs: horizon 1, dt 1 second, ego (0,0,0,1), combined collision radius 0.2 m,
zero safety margin, support cap 10, QP tolerance 1e-7. Scenario 0 is at (1,0.05),
scenarios 1–4 at (1000,0), scenario 5 at (1,-100). All predictions are constant.
The warm-start controls and controller configuration are identical in each run.

| Anchor preparation | Scenario 5 present | Anchor y (m) | Steering (rad/s) | Predicted y (m) | Logged support |
| --- | --- | --- | --- | --- | --- |
| Existing lateral/DR/repair | yes | 0.252 | 0.167119931622 | 0.249999999995 | {0} |
| Existing lateral/DR/repair | no | -0.152 | -0.100097466661 | -0.150000000000 | {0} |
| Frozen original rollout | yes | 0 | -0.100097466661 | -0.150000000000 | {0} |
| Frozen original rollout | no | 0 | -0.100097466661 | -0.150000000000 | {0} |

All four SQP calls returned success without fallback and reported feasible
support iterations. Acceleration was 3 m/s² in each. The executable returned 0
and printed `hidden_preprocessing_dependency=1 frozen_anchor_control_invariance=1`.
Its assertions require different prepared controls, identical logged {0}
support, and frozen-control agreement within 1e-8.

Root cause: `prepare_safe_horizon_anchors` chooses lateral direction using every
fifth sample, including the distant scenario 5. This changes the near scenario's
halfspace normal. The preparation's participating scenarios are absent from the
QP activity union. Thus deleting an unlogged scenario can change the decision;
the logged union alone does not justify a reconstruction argument for the full
algorithm. This does not prove that no different support representation or
extended theorem could cover the implementation.

## Concrete proposed change, not applied

For ordinary SH controllers, retain the existing reference/shifted-backup choice
and freeze its constraint anchor using only the pre-draw state and past controls.
Remove the current-sample lateral/DR/repair call in the ordinary branch of
`MPCController::solve` (currently near line 1093). Make the pre-sampling freeze
explicit at the sampling boundary. Keep existing per-scenario halfspaces, SQP,
support/removal accounting and the separate hybrid branch.

This changes the feasible halfspaces and potentially feasibility and returned
controls. It also departs from the paper's described sampled anchor preparation;
it is a proposed restriction to support a simpler conditional proof, not an
assertion of mathematical identity with every paper algorithm detail. User
approval under AGENTS.md section 1.2 is required before applying it. An extended
proof covering the existing preparation is the alternative.

Approval would authorize this specific anchor restriction, not a claim that the
certificate is established. Remaining obligations include reachable-domain facet
equivalence, support after removals/early returns, the approximate solver's
support representation, and physical feasibility under numerical tolerance.
Week 2 retained clearance reached -0.0000600347 m; an exact safety convention
needs a separate explicit constraint-margin proposal and validation.

## Conditional adaptive-budget argument

If each fixed-budget implementation has a valid conditional certificate at beta
for every eligible pre-draw context, and the budget is measurable from that
context before drawing fresh samples, conditioning on the selected budget and
then averaging preserves beta. Predictor accuracy is unnecessary for this
conditional argument. It does not supply the missing fixed-budget theorem.

For retries, a sufficient confidence allocation is sum(beta_j) <= beta_total,
with each budget chosen before its fresh draw conditional on attempt history
and each attempt satisfying its conditional theorem. Apply the union bound over
bad accepted attempts. Three attempts at beta=0.01 do not justify beta_total=0.01
by this argument. No retry policy or learner was implemented here.

## Commands and evidence

```sh
cmake -S . -B /tmp/dro-shmpcc-week1-build -DCMAKE_BUILD_TYPE=Release \
  -DACADOS_ROOT=/home/stephen/Documents/ACC_Development/Development/acados \
  -DDRO_MPC_ENABLE_ROS2=OFF
cmake --build /tmp/dro-shmpcc-week1-build --target audit_anchor_support -- -j2
/tmp/dro-shmpcc-week1-build/audit_anchor_support \
  results/certificate-anchor-audit/anchor_support.csv \
  > results/certificate-anchor-audit/run.log 2>&1
```

Build succeeded; diagnostic exited 0. Raw CSV and solver log are in the indicated
results directory. The log includes final numerical row violations of
4.78503e-12 and 2.10942e-15, accepted under the configured tolerance.
No production mathematical formulation or existing test was changed in this
audit. Diagnostic implemented and test executed; behavior requires user
verification. The larger multimodal dataset remains deferred behind this gate.
