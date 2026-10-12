# Local reference implementation: confirmed differences and proposed scope

The user requested comparison with the local `mpc_planner` and `scenario_module`
repositories. Both were inspected read-only; neither repository was modified.
No protected controller mathematics in DRO_SHMPCC was changed during this audit.
The previously proposed deterministic 3-sigma anchors were not implemented.

## Reference identity

* scenario_module HEAD: c11addbb7239752d585beb181e8f9197893c6c53, 2024-12-11.
* mpc_planner HEAD: 3002da6a5e43b3577f584381bbcad569c82a381b, 2025-03-30.

Their working trees were clean at inspection. The scenario module README links
the 2024 IJRR journal article. Our supplied PDF is the July 2023 preprint.
These checkouts are useful reference implementations, but their dates and
configuration do not identify them as the exact software revision used for
the supplied preprint experiments.

## Confirmed comparisons

| Component | Local reference implementation | Current DRO_SHMPCC | Consequence |
| --- | --- | --- | --- |
| Preparation batch | SafeHorizon::LoadData gets the batch and calls PushAlgorithm and DRProjection on it | Ordinary SH freezes history/backup anchors, optionally repairing with a separate pilot | Deliberate, approved research restriction; restoring source behavior reintroduces global dependence on Omega |
| Push direction | Every fifth sample, normalized sum, acos comparison against two lateral directions | Every fifth sample, lateral dot-product sign | Equivalent away from undefined normalization/tie details; source divides by zero for a zero vector sum |
| DR passes | Five; change threshold 1e-5 | Up to 25; same change threshold | Different algorithm termination |
| Second DR reflection | Uses delta-start_pose divided by distance(start_pose,delta) | Uses update-delta divided by distance(update,delta) | Different operator, not merely a numerical parameter |
| Subsequent repair | No extra sequential collision repair after DR | Up to 20 repair sweeps, radius padding 0.002 | Additional geometric algorithm |
| Circle-center behavior | Direct normalization without a zero-distance guard | Lateral fallback at exact circle center | Different domain convention; copying undefined arithmetic is not a correctness improvement |
| Halfspaces | a=(obstacle-anchor)/distance, b=a dot obstacle-radius, a dot disc <= b | Opposite sign convention, a dot disc >= b | Equivalent for the same anchor, radii, positions and nonzero distance |
| Polygon domain | Four explicit oriented range constraints, default range 3 m, plus static rows | Reachable-domain square used for clipping, not inserted as the same physical range constraints | Different optimization domain, not simply an alternative polygon representation |
| Facet export | At most 24 rows/stage, including range/static rows; remaining slots get dummies | Retains all required exposed source facets over its reachable domain | Different constraint count/domain; source truncation is not a proved equivalent operation |
| Slack | Shared nonnegative slack state, derivative zero; sampled inequalities allow that slack | Hard collision inequalities, no shared collision slack variable | Different feasible set and optimization problem |
| Slack acceptance | Nonzero slack >1e-3 marks NONZERO_SLACK; module still returns solver exit code | Outer SH certificate rejects fallback/nonfeasibility under its current numerical gate | Solver success and certified status are distinct in the reference too |
| Activity | Dynamic facet residual >-1e-7; infeasible residual >1e-3 also added; residual subtracts slack | Feasible-QP activity tolerance 0.001; failed QPs do not provide valid activity support | Different support interpretation and numerical conventions |
| Horizon checking | computeActiveConstraints checks k=0..N-2, outputs k+1; setParameters skips k=0 | N transitions and N+1 states, constraints across predicted stages | Must reconcile stage definitions and generated backend dimensions before modifying endpoints |
| Support identity | SupportSubsample deduplicates scenario idx across obstacles/discs | Joint scenario-ID union across rows/discs | Broadly consistent identity convention; do not count obstacle/scenario pairs as separate joint support |
| Removals | SafeHorizon path does not call PolygonSearch::RemoveConstraint; default R=0 | Cumulative ID-ordered removals, configured R=1 in Week 2 | Current behavior implements a paper-derived feature not present in the inspected SafeHorizon loop; StageScenario is a separate path |
| Cap return | optimize breaks after an exceeding iterate and marks SUPPORT_EXCEEDED; no saved-prefix restoration | Returns the saved eligible decision/support prefix | Literal source matching conflicts with the earlier paper Theorem-3 interpretation |
| Sampling | Held mode; accumulated Gaussian perturbations times dt; shuffled finite standard-normal database | Existing prediction-law sampler with fresh Gaussian draws; paper-specific configured modes | Noise units/covariances and batch dependence must be mapped; multiplying current covariance by dt² blindly is unjustified |
| Risk configuration | Config::Init sets risk=2*CONFIG[probabilistic][risk] | Uses configured epsilon directly | Reference default risk 0.05 becomes 0.10; do not silently double the paper epsilon=0.05 |
| Risk expression | Rooted binomial formula | Log-space evaluation of the same expression | Same mathematical expression, different floating-point evaluation |
| Sample-size search | Search upper bound 500000, risk tolerance 1e-6, integer return followed by +1 | Smallest integer S satisfying the expression | Measured one-sample differences at h=0,1,2,6 for epsilon≈0.05, beta=0.01 |
| Progress state | With-slack unicycle model integrates spline derivative v | Algebraic monotone closest-path projection | Different dynamics/objective coordinates |
| Objective | Configurable contour/lag, terminal contour/heading, base control/velocity/slack costs | Existing framework objective plus paper profile weights | Paper-profile matching is not identical to current reference defaults |
| Solver | Generated FORCES or acados NLP/RTI interfaces; backend-dependent single-iteration behavior | Hand-assembled condensed SQP using acados/HPIPM dense QPs | Same QP library alone does not establish identical NLP/SQP maps |
| Parallel selection | Default four sampled solver copies; chooses minimum objective with exit_code==1 | One ordinary SH solve | Additional sample/selection/confidence semantics cannot be copied as a performance-only change |

The table records differences, not blanket defects in either implementation.
In particular, a reference source behavior is not automatically a theorem
hypothesis or a safe replacement for current checks.

## Primary code locations

* [Reference preparation and halfspaces](/home/stephen/ros2_ws/src/scenario_module/src/safe_horizon.cpp:134).
* [Reference DR operator](/home/stephen/ros2_ws/src/scenario_module/src/safe_horizon.cpp:315).
* [Reference support activity](/home/stephen/ros2_ws/src/scenario_module/src/safe_horizon.cpp:474).
* [Reference SQP loop and acceptance statuses](/home/stephen/ros2_ws/src/scenario_module/src/scenario_module.cpp:90).
* [Reference range constraints and polygon search](/home/stephen/ros2_ws/src/scenario_module/src/polygon_search.cpp:143).
* [Reference bound and integer search](/home/stephen/ros2_ws/src/scenario_module/src/safety_certifier.cpp:43).
* [Reference predictor sampling](/home/stephen/ros2_ws/src/scenario_module/src/sampler.cpp:359).
* [Planner scenario integration and selection](/home/stephen/ros2_ws/src/mpc_planner/mpc_planner_modules/src/scenario_constraints.cpp:58).
* [Generated sampled inequalities with slack](/home/stephen/ros2_ws/src/mpc_planner/mpc_planner_modules/scripts/scenario_constraints.py:21).
* [With-slack progress dynamics](/home/stephen/ros2_ws/src/mpc_planner/solver_generator/solver_model.py:274).

## Executed comparisons

See [raw results and reproduction](../results/reference-implementation-audit/README.md).
Reference Push/DR method text was extracted unchanged and compiled against
minimal one-disc, one-obstacle position-buffer accessors. Both saved straight
startup fixtures use heading zero, radius 0.625 m, horizon 20 and the original
paired Gamma (20) / Omega (1351). This isolates the arithmetic from ROS and solver
dependencies; it is not a run of either full reference controller/backend.

Maximum anchor displacement between reference preparation and our same-batch
routine:

| Seed | Batch | Maximum difference (m) | Differing stages / 20 |
| --- | --- | --- | --- |
| 77 | Gamma | 0.278816061 | 1 |
| 77 | Omega | 4.967571388 | 13 |
| 78 | Gamma | 1.336058530 | 3 |
| 78 | Omega | 5.647678173 | 14 |

Reference Omega preparation at seed 77 leaves minimum distance clearance
-0.423555072 m and 30 stage/scenario distances below the reference's -0.001 m
check. Seed 78 leaves minimum clearance +0.001 m. Thus copying the five-pass
routine does not establish reliable initialization on these fixtures.

The extracted reference certifier was also compiled and executed. It gives
S(4)=666 and S(10)=1351, matching current code, but S(1)=317 versus 316 and
S(6)=896 versus 895. No production sample formula was changed to match the
reference's search rounding.

## Reference-compatibility change considered, now deferred

The user's subsequent feedback explicitly steered away from transplanting the
reference Push/DR algorithm. The compatibility profile described below was not
implemented and is no longer the proposed next step. It remains a record of the
mathematical differences that a literal source port would introduce.

Introduce an explicit **reference-implementation profile**, leaving the current
research profile and its historical experiment meaning intact. The first minimal
patch would select same-batch preparation and port the reference five-pass DR
formula, omitting our extra sequential repair. It would include source-parity
fixtures for the defined, nonzero-distance domain and report undefined-domain
cases rather than silently treating a guarded variant as identical.

This first patch changes the anchor, normals, feasible set and current
sample-dependence assumptions. It does not establish full reference parity;
shared slack and progress dynamics are the next separate solver-formulation
changes. Range constraints, activity/horizon semantics, removals, cap return,
noise units, objective and backend settings each need separate approval and
behavioral comparisons. Existing tests would not be weakened or retargeted
without explicit approval.

Restoring source behavior would reopen the global Omega-dependent preparation
issue documented by the four-way diagnostic. An extended theorem is still
needed before asserting its logged support establishes the probability bound.
The inspected reference code does not supply that proof.

AGENTS.md section 1.2 requires explicit approval after presenting the concrete
mathematical change. The earlier frozen-anchor approval does not authorize
undoing that restriction, adding slack, or changing numerical activity rules.
These protected changes are therefore proposed, not applied.

Current status: comparison tools implemented and numerical tests executed;
behavior requires user verification. Source-compatibility work is deferred in
favor of the deterministic pre-sampling variant.
