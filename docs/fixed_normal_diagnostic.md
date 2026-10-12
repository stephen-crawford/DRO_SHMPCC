# Fixed-orientation hard constraints: diagnostic semantics

The user explicitly requested a 200-batch diagnostic using the already selected
anchor, fixed modal-mean normals, R=0 and no slack. This is not a deployment or
an assertion of equivalence to the original paper's sampled-normal formulation.
See [results](../results/fixed-normal-hard-replay/README.md).

## Direction construction and degeneracy

After deterministic anchor selection and before the certification sampler is
called, the diagnostic forecasts each available held mode with the existing
zero-noise `ModeModel::propagate` mean. For obstacle v, mode m, stage k and disc d:

    n[v,m,k,d] = (anchor_disc[k,d] - modal_mean[v,m,k])
                / norm(anchor_disc[k,d] - modal_mean[v,m,k]).

For distance <= 1e-9 metres, use the most recent nondegenerate direction for
that obstacle/mode/disc at an earlier horizon index. If there is none, use the
road normal at the anchor disc, toward the selected candidate: negative road
normal for right, positive for left. Exact score ties and nominal/brake sides
retain the existing candidate order; the diagnostic road-side fallback chooses
positive road normal for nominal/brake. A degenerate road fallback is not saved
as a nondegenerate direction. No sampled direction is used for degeneracy.

The 200 replay contexts selected right/left explicitly as in the earlier
experiment. All resulting directions and anchor positions were identical across
batch streams within each context. No mean/anchor degeneracy occurred, so the
fallback rules are defined but not exercised by these particular batches.
The experiment rejects DRO, Markov sampling, nonlinear body-displacement modal
forecasts, pilot repair and nonordinary SH use.

## Sampled rows and physical meaning

For sampled obstacle position X, retain the original row/disc metadata and use

    a = n[v,m,k,d],  b = combined_radius + a dot X,
    a dot ego_disc[k,d] >= b.

One joint scenario supplies its own offsets across the entire horizon. It does
not select the anchor or directions. The radius, safety margin, cost, dynamics,
solver settings, tolerances and sample law are unchanged. The sample vector is
not edited, and no scenario is removed.

For an exactly unit direction, satisfying this projection implies the original
physical separation by Cauchy–Schwarz. The constraint is sufficient, not
necessary: a projected violation need not imply an actual collision.
Floating-point unit norms and solver tolerances still require a numerical error
argument for a formal executed-safety claim; this experiment adds no tightening.

For each fixed (v,m,k,d), all rows share exactly the same stored normal. Their
intersection is precisely the single row whose offset is largest. The offline
`groups.csv` records that maximum and its scenario ID, with lowest ID breaking
exact offset ties. The runtime checks equal normals within groups before writing
this exact scalar compression. The independent LP analysis uses those rows to
check raw stage/disc feasibility. Production continues using its existing polygon
reduction and solver support logic; the offline group maxima are not substituted
as a new production reduction algorithm.

## Sampling, solver and status

The runner executes a paired production R=1 replay and a fixed-normal R=0 replay
at each of the 200 original batch streams. It checks exact equality of scenario
IDs, held modes and sampled positions at every stage. h=10 remains unchanged:
nonremoved cap 9 plus R=1 becomes cap 10 plus R=0, so canonical S remains 1351.
Both variants run the complete existing controller and final certificate gate.
No control is applied to the plant and the live controller/RNG are unchanged.

A private friend-only switch enables the new branch. Default false preserves
live geometry. Normals are built in `solve_attempt` before sampling and used in
`build_ordinary_constraint_problem`; the rest of the controller is unchanged.
The opt-in guard requires R=0. This is a distinct candidate mathematical
formulation, explicitly approved only for diagnostic replay.

`hard_feasible` means a nonfallback successful plan passing the existing numerical
sampled hard-constraint gate. It does not assert exact projected-row feasibility.
`strict_physical` additionally checks nonnegative sampled Euclidean clearance and
deterministic admissibility at zero tolerance. CERTIFIED remains the existing
implementation flag, not a newly established probabilistic guarantee.
Returned fallback margins/plans are not optimized-trajectory performance.

## Remaining mathematical obligations

The evidence establishes high availability on these two fixed startup contexts.
It does not establish global feasibility of the hard R=0 decision map. With
bounded reachable positions and an unbounded Gaussian projection having positive
variance, an offset can exceed every reachable projection. Fixed orientation
eliminates within-group opposing normals; it cannot exclude such draws.
Assumption 1 remains unproved for this finite-iteration controller. A total
augmented decision map and rejection semantics would need a separate argument.

Exact group compression supplies a more transparent possible reconstruction set,
but does not prove that the recorded QP-active union bounds theorem support of
polygon reduction, numerical optimization, support-cap termination and early
returns. Extreme-ID ties and approximate facet identification still need audit.
No support bound, confidence allocation or S(h) formula was changed.

Before adoption or a long study, numerical physical feasibility/tightening,
complete support domination, reduction equivalence and theorem applicability
remain open. No further formulation change or live deployment is part of this
experiment.

## Subsequent candidate work

After reviewing this hard diagnostic, the user accepted fixed normals as the
leading candidate and authorized lexicographic slack, exact compression and
numerical tightening diagnostics. See the [candidate formulation](fixed_normal_candidate_formulation.md).
The historical results above remain unchanged; live defaults remain off.
