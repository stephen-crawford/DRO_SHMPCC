# Leading candidate: fixed normals with lexicographic feasibility slack

The user accepted fixed normals as the leading candidate and authorized the
slack, exact-compression and tightening diagnostics. Live defaults remain off.
This formulation differs from the publication's sampled-normal construction.
See [slack results](../results/fixed-normal-slack/README.md) and
[tightening results](../results/fixed-normal-tightening/README.md).

## Ideal mathematical program and totality

Condition on the pre-sampling deterministic anchor and unit direction table.
Let Z contain the unchanged hard dynamics, road and control constraints. Each
sample row is g_j(z)=r+n_j dot X_j-n_j dot p_j(z). The candidate first minimizes
s subject to z in Z, s>=0 and g_j(z)+tau<=s for every sample row; it then minimizes
the existing cost over the minimum-slack feasible set. No finite slack penalty
or finite upper bound on the primary slack replaces that lexicographic order.

If Z is nonempty and compact and the rows and cost are continuous, any z0 in Z
has finite feasible slack max(0,max_j(g_j(z0)+tau)) for any finite batch.
The primary minimum and secondary minimum then exist. A deterministic tie rule
would complete the ideal decision map. This conditional argument does not prove
that the finite-iteration numerical SQP always returns that map. In particular,
the runner solves local linearized primary/cost QPs at each of 15 iterations,
using the numerically returned primary slack as the secondary upper bound;
it does not demonstrate a globally optimal nonlinear lexicographic solution.
Shared collision slack never softens hard road/control constraints.

## Exact parallel compression and reconstruction

For each (obstacle, held mode, stage, disc), all stored normals are identical.
Only the largest offset matters, with lowest scenario ID breaking exact ties.
Let C be the union of those maximizing joint-scenario IDs. Keeping all rows
belonging to C reconstructs the same collision feasible set, also with shared
slack and tightening. If a deterministic optimizer uses only this compressed
problem and sample-independent settings, its output is reconstructible from C.
This is a conditional reconstruction argument, not a proof for every branch of
the current controller or that the smaller QP-active union dominates support.
Raw-sample diagnostics, reduction, stopping gates, metadata and tie behavior
must be audited before transferring it to the implemented certificate map.
The observed maximum |C|=10 is not a universal bound of 10: the general count
can grow with obstacles, modes, horizon stages and discs.

## Physical inclusion and numerical margin

For exact unit n, physical collision implies g_j>0. Thus at tau=0 a hard
zero-slack feasible point supplies a sufficient physical sampled separation
condition. A tiny positive solver slack does not establish exact zero slack.
The diagnostic separately evaluates whether the returned point admits s=0;
these are floating-point observations, not formal exact-arithmetic proofs.

For tightening tau, define s_lift=max(0, returned slack, max_j(g_j+tau)) at
the final nonlinear trajectory. Then the constructed pair is augmented-feasible
in the computed arithmetic. If s_lift<=tau, original projected reserve is at
least tau-s_lift. Physical collision also implies tightened violation greater
than tau, hence greater than any admitted slack <=tau. Lifting is a read-only
candidate construction; it can lose primary optimality and does not turn a
failed original solver feasibility check into a passing one.

The tested tau=0.0002 m preserves the existing 0.0001 m tolerance. Positive
observed reserves do not bound floating-point dot products, direction-normal
error, nonlinear solver/model error or inter-stage plant motion. Those error
budgets and the full theorem/support assumptions remain open. No probabilistic
certificate, live execution gate or confidence formula was changed. Long
encounters, budget curves and learning remain deferred until the formulation
and applicable numerical/support arguments are frozen and behavior reviewed.
