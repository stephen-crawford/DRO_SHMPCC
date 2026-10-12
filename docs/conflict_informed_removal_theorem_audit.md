# Conflict-informed removal: scope of the Safe Horizon results

This audit follows the full-solve replay in
[forced-removal-full-solve](../results/forced-removal-full-solve/README.md).
It reads the supplied [SafeHorizonMPC.pdf](/home/stephen/Documents/publications/SafeHorizonMPC.pdf),
particularly printed pages 3, 8–10: Assumptions 1 and 4, Eq. (20),
Theorems 4–5, and Eq. (24). It does not establish theorem applicability to this
numerical implementation or authorize replacing the live removal policy.

## What the paper permits

Eq. (20) defines the removal map as a function of the decision and the full
sample collection, R(theta, omega). The abstract results do not prescribe
ascending-ID selection or require choosing removed scenarios independently of
the batch. After Theorem 5, the paper explicitly permits determining removals
before or during optimization. Thus full-batch dependence alone is not grounds
for ruling out conflict-informed removal under this framework.

Theorem 4 is conditional on Assumption 4: removal must leave the scenario
optimization problem feasible. Assumption 1 requires a feasible decision for
all sample extractions. This is stronger than a feasible raw intersection, one
successful numerical solve, or high empirical availability at two contexts.
Theorems 4–5 are not a blanket guarantee for every measurable rule satisfying
only |R| <= 1.

Theorem 5 states that every removed scenario belongs to the support of the
augmented problem with deletion indicators. This is an inclusion, not an upper
bound equating support to the number of removed scenarios. The implementation
therefore inserts the forced deletion into the cumulative support set and retains
it through the final certificate gate. All 34 certified replay logs contain the
forced ID in their support union.

Eq. (24) describes the paper's concrete SH removal strategy using accumulated
active and infeasibility scenarios. A conflict-based rule would be a candidate
alternative within the abstract removal-map framework, not a literal reproduction
of that concrete strategy. The paper also explains that additional scenarios can
become support after a deletion; counting R alone is insufficient.

## What remains to establish for a candidate policy

1. Specify a total deterministic selection rule, including witness ties, box-only
   conflicts, no successful deletion, repeated stages and exhausted budget. This
   replay uses one previously saved LP deletion per batch; it is not that online
   rule and does not measure its selection cost.
2. Prove feasibility/define the decision map on its asserted domain. The forced
   experiment still rejects four of 38 geometrically salvageable batches. Eight
   other diagnosed batches need more than one deletion even for raw intersections.
   No finite-R guarantee for every unbounded Gaussian sample extraction has been
   demonstrated. Rejection/abstention or retries require their own applicability
   argument and cannot simply be substituted for Assumption 1.
3. Establish a support upper bound for the complete algorithm. In particular,
   changing another retained sample can change the witness, deletion choice, or
   whether an alternative passes its geometric test. Theorem 5's inclusion does
   not prove that the union of the selected deleted ID and recorded QP-active IDs
   dominates all such dependencies. The support reconstruction/removal-selection
   argument must cover those comparisons as well as the optimizer.
4. Establish the remaining existing obligations: convex-iterate solver assumptions,
   numerical physical feasibility, facet-reduction equivalence after removal and
   saved-prefix/early-return support accounting. Successful replay flags do not
   replace these obligations.

The independently tested S(h) inversion remains unchanged and valid as a formula
calculation. Here h=10 includes the one removal allowance, with configured
nonremoved cap 9. Whether that h bounds the *theorem support of a new selection
algorithm* remains a separate question. There is no basis from this audit alone
to claim that keeping S=1351 proves its probabilistic certificate.

## Conclusion for this experiment

The paper allows a batch-dependent removal map, subject to its assumptions, and
requires removed samples in support. It does **not** establish from |R| <= 1
alone that an arbitrary conflict-informed numerical rule satisfies those
assumptions or that the present logged support is an upper bound.

The full-solve results justify evaluating conflict-informed removal before fixed
normals. Adoption still needs a concrete rule and its feasibility/support audit.
The experiment-only forced deletion remains disabled for live controllers.
No theorem extension, live policy change, larger removal budget, retry or
fixed-normal constraint formulation was implemented.
