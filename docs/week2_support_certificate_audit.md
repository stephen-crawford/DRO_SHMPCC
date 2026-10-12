# Week 2: logged support and certificate applicability

Status: code-path audit and targeted tests executed. The logging identity below
is documented at the implementation level; the exact inequality
`n_logged >= n_theorem` for the complete numerical algorithm is **not proved**.
Budget experiments must therefore be described as implementation-level
certificate results, not a validated new safety theorem.

## Logging identity and earlier-iterate return

For ordinary SH-MPCC, `solve_optimization_sqp()` computes

`logged IDs = union of active joint scenario IDs from counted QPs + removed IDs`.

Counted QPs must report convergence, finite primal variables and residuals, and
satisfy QP inequality/box rows to `qp_tolerance`. The active threshold is 0.001.
Removed IDs remain in the same union; integer scenario IDs identify whole joint
trajectories, not individual stage/obstacle/disc rows. `support_size` is the
cardinality of this set, and `support_scenarios` exports the corresponding IDs.
`experiment_harness.cpp` copies that vector into the decision frame;
`experiment_artifacts.cpp` writes its size to `support_scenarios.csv`.
Final nonlinear `active_scenarios` is a separate diagnostic and is not the
certificate support label.

On support-cap exceedance, the solver restores a saved eligible iterate's
trajectory, controls, support union, removal set, rows and evaluated-iteration
count. The local PDF's Algorithm 1 / Theorem 3 explicitly permits returning an
earlier feasible iterate and uses its prefix support. Using an earlier support
is therefore not, by itself, a mismatch: the returned decision must also be the
corresponding earlier feasible decision. The existing algorithm test compares
its controls/trajectory directly with a separate one-iteration solve and checks
its support IDs. Removed scenarios exposed after cap termination are restored
with that prefix, rather than silently excluded from the returned plan.

The certificate status additionally requires sufficient sampled scenarios,
sampled constraint feasibility, evaluated support within cap, and no braking
fallback or unaccounted homotopy selection. A bare `support <= h` is insufficient.
The Week 2 experiment records both cap membership and the full status and accepts
only successful, nonfallback, implementation-certified results.

## Remaining mathematical obligations

1. **Sample-dependent anchor preparation.** `solve_internal()` calls
   `prepare_safe_horizon_anchors()` before the counted SQP iterations. The routine
   uses current sample positions to select lateral push direction/magnitude,
   projection anchors and DR updates, thereby affecting collision normals and
   linearized problems. Its participating scenario IDs are not added explicitly
   to the support union. This is a concrete unaccounted dependency that requires
   a proof that it is already covered by the paper's algorithm decomposition,
   or a revised support-accounting/design argument. Following the paper's
   advertised preprocessing does not itself prove the complete implementation's
   compression bound. This audit identifies the proof gap; it does not claim a
   demonstrated counterexample or silently change the algorithm.
2. **Exact versus numerical feasibility/activity.** The PDF's Assumption 3 uses
   exact inequality feasibility and Eq. (14) uses exact equality. The code accepts
   residuals down to -0.0001 and activity up to 0.001 in these profiles. Near-active
   rows can overcount activity at the computed point, but that does not prove
   inclusion of every active row at an exact optimizer without an error bound.
   The experiment observed retained-circle clearance as low as -0.0000600347 m;
   none was below -0.0001 m. Exact mathematical feasibility is not established.
3. **Equivalent reduction domain and solver assumptions.** The QPs use polygon
   facets reduced on a reachable domain, while final validation uses retained
   source rows. A full proof must establish that reduction preserves the decision
   map over that domain, including degeneracies and numerical handling, and that
   all counted QPs satisfy the theorem's convex-solve assumptions.
4. **Data/model and global-feasibility assumptions.** The theorem remains
   conditional on the joint scenario sampling law and feasibility assumptions.
   A finite removal budget or a passing rollout is not a proof that every sample
   extraction is feasible, nor a closed-loop lifetime collision bound.

Consequently this audit does not establish either equality or a conservative
upper bound relative to the theorem for the entire numerical decision map.
No protected mathematical code was changed to conceal these obligations.
Before a paper theorem relies on the label, establish the preprocessing
compression/dependency argument and numerical assumptions explicitly, or approve
and validate a mathematically justified accounting change.

## Three existing regression discrepancies

A fresh Release CTest run reproduced the same three failures:

| Test | Failing expectation | Relation to the paper |
| --- | --- | --- |
| pruning_and_linearization | configured facet limit throws | Restored geometry retains all required facets instead of truncating. This is an interface/expectation discrepancy, not evidence that scenario rows may be discarded. |
| mpcc_progress | cost is independent of reference speed | Paper includes reference-velocity tracking; the existing objective already depends on reference speed. This expectation conflicts with that formulation. |
| support_accounting | infeasible/violated nonlinear iterates count as certified support (three checks) | Assumption 3 requires feasible convex iterations; failed QPs do not produce evaluated certification support merely from violated nonlinear rows. |

`test_safe_horizon_algorithm` passed, exercising cumulative joint removals,
feasible-QP activity, cap return of the exact saved decision, impossible rows,
nonconverged QPs, nonlinear rollout rejection, numerical covariance and timing.
The three existing tests were not edited, weakened or represented as passing.
Their expectation changes require an explicit decision under AGENTS.md.

Command:

```sh
ctest --test-dir /tmp/dro-shmpcc-week1-build \
  -R '^test_(safe_horizon_algorithm|support_accounting|mpcc_progress|pruning_and_linearization)$' \
  --timeout 60 --output-on-failure
```

Observed: 1/4 passed; three failed. Raw output is copied to
`results/week2-budget-response-v2/analysis/support_audit_tests.log`.

## Future retry theorem

No learned or upward-retry live controller was implemented. A ladder with up to
K independently certified attempts at beta each does not automatically give the
same beta bound for its accepted output: a union-bound argument gives at most
K*beta absent a sharper joint argument. A valid policy needs an explicit
attempt-level confidence allocation or a suitable combined theorem. The
single-budget production curve in this experiment retains beta=0.01. Do not
interpret the hindsight oracle or candidate sweep as a live retry policy with
unchanged safety confidence.

## Change made / mathematical impact / validation / current status

Added an isolated budget-response tool using the existing CounterfactualProbe
friend interface, a serial driver, analysis plots and this audit. No production
controller, bound, constraint, certificate gate or existing test was modified.
No mathematical guarantee or formulation was modified.

130 frozen-state clone checks reproduced live baseline controls, support IDs and
certificate status, and left the live RNG unchanged. All 2,080 probe rows were
checked against the production S(h) curve and complete eight-cap blocks. These
are implementation checks, not proof of the outstanding theorem obligations.

Candidate audit and experiments implemented and tests executed; behavior
requires user verification. Formal theorem applicability remains unestablished.
