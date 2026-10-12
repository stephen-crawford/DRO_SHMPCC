# Deterministic pre-sampling anchor selection: implemented candidate

The user approved nominal/left/right/brake selection with dynamically usable
side candidates and kappa=3 as an engineering initial value, not a theorem
constant. The candidate is opt-in:

```yaml
deterministic_anchor_selection: true
anchor_inflation_kappa: 3
anchor_pilot_samples: 0
```

Default selection remains disabled. Activation currently requires ordinary
non-DRO, held-mode SH without pilot preparation. Forecast scoring supports
linear Gaussian modal models; nonlinear body-displacement modes are explicitly
rejected rather than assigned an unsubstantiated covariance approximation.

## Candidate generation

All generation and selection occurs before the current certification draw.
Nominal controls are the existing shifted backup if available, otherwise the
bounded inputs extracted from the existing reference. Nominal is their rollout
from the measured state with the existing dynamics.

Brake uses the existing make_braking_seed_inputs and rollout, preserving past
curvature where available while slowing. It is not certified collision-safe
by its inclusion in this family.

Side targets are +/- (road_width/2 - ego_radius - existing safety_margin).
They are targets, not instantaneous position shifts. For each side, the desired
lateral offset follows the cubic ramp y(t)=y0+(target-y0)*(3t²-2t³), t in [0,1]
over the prediction horizon. The target heading is the local road tangent plus
atan2(desired lateral rate, nominal speed). The yaw command tracks that heading
and is clipped to the existing yaw-rate limits. Acceleration comes from the
nominal controls. Each input sequence is propagated with existing RK4/spline
rollout; all four candidates face the unchanged deterministic admissibility
check. Side candidates require a reference path and active road constraints.

The cubic ramp is a bounded tracking construction, not an optimizer proving
maximum attainable lateral displacement. Saturation may prevent reaching the
road-side target. Generated trajectories, rather than target coordinates, are
scored and tested for road/input/velocity admissibility.

## Selection

Every available modal mean/covariance is propagated from observations using
its existing linear transition and noise matrices. No mode-probability weighting
or cutoff is used. Covariance recurrence is Sigma_next=A Sigma A^T+G G^T.

Score each eligible candidate by the minimum across modes, obstacles, discs
and predicted stages of

    norm(disc_center - mean) - combined_radius - kappa*sqrt(lambda_max(Sigma)).

This uses an isotropic overbound of the covariance ellipse. It is an explicit
signed envelope score, not exact ellipse distance or a statistical coverage
claim. Kappa is configurable and affects quality/feasibility only.

Select the largest score; exact ties prefer nominal, brake, left, right.
No near-tie threshold or added cost weight was introduced. Without obstacles,
prefer eligible nominal. If none is eligible, retain the prior frozen reference
and let the existing solve/acceptance checks report its result.

Freeze the selected trajectory as the collision anchor. The existing SQP
warm-start and dynamic nominal policies remain unchanged; selected candidate
controls are not directly applied to the plant. No sampled anchor repair,
post-draw candidate selection, pilot, retry or probability reallocation occurs.

## Mathematical effect and validation

This approved change alters the fixed anchor and therefore the scenario
halfspaces/feasible set. It changes no physical radius, road constraint,
solver tolerance, canonical sample-count function, support bound, removal rule
or certificate gate. Conditional measurability is the intended dependence
structure; the remaining numerical/support/facet theorem obligations are open.
Do not name this a certified controller variant yet.

[Executed startup results](../results/deterministic-anchor-startup/README.md)
cover seeds 77 and 78 at kappa=0,1,2,3,4. The kappa=3 live startups both entered
QP and returned certified flags, but seed 78's independent fresh-batch probe
failed. This is not evidence of reliable initialization for arbitrary draws,
long encounters or a completed certificate proof.

Candidate fix implemented and test executed; behavior requires user verification.
