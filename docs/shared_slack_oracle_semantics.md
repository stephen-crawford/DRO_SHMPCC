# Shared-slack oracle: formulation and limits

This is an explicitly requested offline experiment, implemented solely in the
friend diagnostic runner. It does not change the live controller, add an
execution policy, or claim a proved certificate for an augmented decision map.

## Mathematical target

Keep the chosen deterministic anchor and all original sampled unit-normal
collision rows. In the controller's convention those rows are

    a_i dot c_d(z_k) >= b_i.

The experiment augments them to

    a_i dot c_d(z_k) + s >= b_i,  s >= 0,

with one shared scalar s for every obstacle, stage, disc and scenario. No scenario
is removed. Road, velocity and control constraints remain hard. The target is
lexicographic minimization of s, then of the existing objective J.

With an existing deterministic admissible trajectory and finite sampled rows,
setting s to max(0, max_i(b_i - a_i dot c_d(z_k))) constructs an augmented feasible
point. Thus hard sampled-halfspace conflicts need not make the *mathematical
program* infeasible. This relies on the hard deterministic domain being nonempty;
slack does not repair an infeasible initial condition or hard road/control domain.
It also does not prove that a finite numerical SQP algorithm finds a solution.

## Numerical experiment

For each of the same 200 batch streams, an unchanged production replay supplies
its complete initial row snapshot. The oracle resets to the pre-batch cold
nominal, using the existing straight-line seed, bounded input extraction,
dynamics rollout and spline handling. It does not reuse a successful sampled
production trajectory as its warmstart. Startup-only guards reject preexisting
references/backups and contexts other than 77/78.

The experiment reuses build_condensed_qp and the existing HPIPM solver. At each
SQP iteration it solves:

1. A linear program represented as a PSD QP: zero Hessian, objective s, all
   current condensed rows plus the common slack column. Its slack has lower
   bound 0 and no finite upper bound.
2. The existing quadratic cost in the control increments, with the same rows
   and slack bounds 0 <= s <= max(0, returned first-phase slack).

The second cap implements a *numerical* lexicographic level; it uses the computed
first-phase value, with no added relaxation. These are local SQP subproblems,
not a global solve of the nonlinear lexicographic MPC. Native QP/SQP iteration
limits, convergence threshold and solver tolerance are retained. Each finite,
converged QP is checked against all rows and bounds at the existing tolerance.
Full bounded control updates and the existing dynamics rollout follow the second
phase, without collision-dependent backtracking. The experiment records a failure
rather than substituting removals when either phase fails.

Raw rows at stages 1..N are kept rather than hard-polygon clipped. The hard
2-D polygon cannot represent feasibility with a variable slack; dropping rows
because they are redundant at s=0 would need a separate augmented-equivalence
proof. No such reduction is introduced. Stored row normals and offsets are
unchanged. The original k=0 rows are excluded, as in production facet grouping.

The total sampling cap remains h=10: changing the diagnostic configuration from
nonremoved cap 9 plus R=1 to cap 10 plus R=0 gives the same S=1351. The code checks
this identity. It never calls the R=1 removal algorithm in the slack solve.
The 100-batch baselines still use their original R=1 settings.

## Report semantics

- `all_qps_converged`: every executed LP/QP phase converged and passed numerical
  row/bound checks. Reaching the SQP iteration limit is not labelled convergence
  of the nonlinear optimization.
- `augmented_feasible`: additionally, the final rolled-out controls/states pass
  existing deterministic admissibility at QP tolerance, and their maximum raw
  collision-row violation is <= the last optimized slack plus that tolerance.
- `slack`: the raw last second-phase solver value. Tiny negative values can be
  returned despite s >= 0 because feasibility is numerical. They are retained,
  not silently rounded or asserted to be exact optima.
- `strict_zero`: a diagnostic reconstructed zero-slack candidate check: converged
  phases, nonpositive raw slack, zero computed raw-row violation, and deterministic
  admissibility with zero tolerance. The corresponding candidate can set s=0.
  This is a floating-point observation, not an exact-arithmetic or probability
  certificate. It does not count tiny positive slacks as exact zero.
- `slack_le_1e4`: augmented-feasible and raw slack <= 0.0001. This is descriptive;
  it is **not** a margin-adjusted execution rule.
- `n_active`: distinct sample IDs within the existing 0.001 binding threshold in
  the last cost-phase QP. `union_active` includes both phases at every feasible
  convex iteration. These are logged diagnostics, not proved theorem support.

No support-cap termination is used in this oracle: it must expose the augmented
solution and full active union, including cases exceeding h=10. The `within_h10`
column is descriptive. No augmented result receives a CERTIFIED label or reaches
the plant. Production baseline flags keep their existing meaning.

## Safety implication and outstanding proof work

For the implemented halfspaces, g(z,xi) is the maximum signed halfspace violation
for one sampled joint scenario. Exact zero slack gives the original halfspace
constraints. Their satisfaction implies the original physical disc separation;
the converse need not hold. Consequently at s=0,

    V_aug(z,0) = V_halfspace(z),  V_physical(z) <= V_halfspace(z).

The feedback's equality with physical violation is exact only if its g denotes
that same physical event, rather than this sufficient halfspace surrogate.
An applicable augmented scenario theorem and a sound zero-slack execution gate
would bound the *joint event* of acceptance and excess physical risk. This is
not automatically a beta bound conditional on acceptance.

A positive numerical threshold such as 0.0001 is not exact zero slack. It needs
an explicit conservative margin/error argument before it can support an execution
certificate. That margin was not chosen or added here. Numerical feasibility,
solver support domination, LP-level selection/ties, early returns, and the total
augmented decision map remain proof obligations. In particular a converged
linearized QP can yield a nonlinear rollout violating its reported slack. Slack
makes the mathematical feasibility argument simpler; this experiment does not
establish the full numerical implementation theorem.
