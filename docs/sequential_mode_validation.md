# Fixed-candidate anytime mode validation: mathematical specification

This is a real-arithmetic proof for an offline candidate algorithm, not a completed physical or numerical implementation certificate. It follows the user's approved beta-mixture construction. The time-uniform inference principle is established in [Howard et al., confidence sequences](https://arxiv.org/abs/1810.08240); the derivation below specializes the likelihood-ratio argument to Bernoulli collision indicators.

## Assumptions and target

Fix context c, candidate z, nonempty marginal ambiguity sets B_j and constants alpha_a>0 before validation. Require sum_a alpha_a<=beta_val. Conditional on this frozen information, every requested indicator for pair a=(j,m) is a fresh Bernoulli(v_a) observation, with that conditional law given the entire past. Pair selection and batch sizes are predictable. This is stronger than merely matching unconditional sample histograms. Physical mode-conditionals must equal the modeled ones for the physical implication below.

The target is R(v)=min(1,sum_j sup_(p in B_j) p dot v_j). For multiple obstacles this is a robust union upper functional. It need not equal the exact worst-case joint collision probability. No obstacle independence is required for its upper-bound interpretation.

## Mixture process and endpoints

For a fixed p in (0,1), each Bernoulli likelihood ratio against an alternative q has conditional mean-one multiplicative increments. Integrating its nonnegative product against Beta(1/2,1/2) preserves the martingale property by Tonelli:

    M_n(p) = B(k+1/2,n-k+1/2) / [B(1/2,1/2) p^k (1-p)^(n-k)].

At p=0 or 1, alternatives are not absolutely continuous with respect to the true law; the displayed process on realizable paths is instead a nonnegative supermartingale starting at one. The same Ville inequality applies. In particular, one must not assert a mean-one martingale at the endpoints.

Thus P_p(exists n: M_n(p)>=1/alpha)<=alpha. The open inversion C_n={p:M_n(p)<1/alpha} contains the true parameter at every n with probability >=1-alpha. For n=0 it is [0,1]. Convexity of -k log p-(n-k)log(1-p) makes each nonempty inversion an interval about k/n. Lower/upper endpoints enclose the open set; closing those endpoints is conservative. Running intersections Lbar_n=max_(s<=n)L_s and Ubar_n=min_(s<=n)U_s preserve coverage and shrink monotonically. An empty numerical intersection causes abstention, not certification or rejection.

For k=0, L=0 and U=1-exp((log B(1/2,n+1/2)-log B(1/2,1/2)+log alpha)/n). For k=n the symmetric formula gives L, U=1. Interior endpoints solve log M=-log alpha on either side of k/n. The implementation uses log beta, log/log1p, 52 bracket-preserving bisections and outward adjacent floating-point endpoints. Those arithmetic choices are tested, but are not an interval-arithmetic bound on all log-beta/root errors.

## Adaptive selection and simultaneous coverage

Unselected processes remain constant. A selected process receives a conditional mean-one (or boundary mean-at-most-one) increment because selection precedes its new observation. Alternatively, apply the local time-uniform event to every complete conditional stream: it holds at every local N_a(t), regardless of the predictable global schedule. Union bounding across pairs gives a single event E_val of probability >=1-beta_val on which L_a(t)<=v_a<=U_a(t) for every pair and time. No extra penalty arises from mode selection or repeated looks for this fixed candidate. Batches remain predictable and every observation updates the local intersection; checking decisions only at batch boundaries preserves coverage.

## Propagation and decisions

Nonnegative probability weights make r_j(x)=sup_p p dot x coordinatewise monotone. On E_val,

    R_L(t)=min(1,sum_j r_j(L_j(t))) <= R(v)
          <= R_U(t)=min(1,sum_j r_j(U_j(t))).

Use CERTIFY if R_U<=epsilon, REJECT the robust union criterion if R_L>epsilon, and continue otherwise, subject to finite resource caps. The strict/non-strict directions are intentional. Simultaneous contradictory boundaries or empty confidence intersections require abstention. Resource exhaustion is unresolved, not rejection.

CERTIFY when R(v)>epsilon and REJECT when R(v)<=epsilon can each occur only off E_val; their union is also contained in its complement. Consequently the probability of any wrong classification of this fixed robust functional is <=beta_val in real arithmetic. This statement requires neither an allocation-policy accuracy guarantee nor statistical independence among the mode coverage events.

Suppose jointly P(E_DRO)>=1-beta_DRO, where E_DRO means all true marginal mode laws lie in their respective sets. On E_DRO and under correct conditional models, physical joint collision risk is <=min(1,sum_j p_star_j dot v_j)<=R(v). Hence false physical certification is bounded by beta_val+beta_DRO without independence of the coverage events. This is an outer probability over validation/calibration randomness of making an incorrect risk claim, not the collision probability itself. No numerical/conditional-model error or repeated-decision failure budget is included automatically.

REJECT only proves failure of R(v)<=epsilon. It neither proves actual p_star risk exceeds epsilon nor, for multiple obstacles, proves the exact joint worst-case risk exceeds epsilon. Using inf_p p dot L instead would test a stronger and different proposition. Changing z with validation data, choosing among candidates using these streams, data-dependent alpha allocation, repeatedly reusing streams or deploying over an encounter needs additional accounting.

## Analytic allocation policies

An OT maximizer q_star at U is a subgradient of r(U), including at nonsmooth ties. Certification VOI uses q_U times expected decrease in running U; rejection VOI uses q_L times expected increase in running L. The predictive collision probability (k+1/2)/(n+1) affects allocation only, not validity. Projected next endpoints are intersected with current running bounds before computing expected changes.

Upper-only, nearest-boundary and interval-width variants are heuristic decision-efficiency rules, not optimality theorems. Nearer-boundary picks upper VOI when R_U-epsilon<epsilon-R_L, otherwise lower VOI; equality uses lower. Width adds both scores. Deterministic ties favor the least-sampled available mode and then lowest ID. Nominal allocation uses deterministic weighted fairness proportional to nominal mass; round-robin is uniform. Mode stream exhaustion removes that mode from allocation; exhaustion of every stream returns unresolved. There is no termination guarantee at R(v)=epsilon, under starvation, or at a finite cap.

The software tests and numerical comparisons are evidence about the candidate implementation, not a substitute for reviewing these assumptions and a complete numerical error argument. No live MPC execution gate is enabled.
