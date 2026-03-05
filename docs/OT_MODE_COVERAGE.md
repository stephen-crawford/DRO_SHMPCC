## Optimal Transport in Safe-Horizon MPC (SHMPC)

This note collects a **self-contained, formal description** of how **optimal transport (OT)** is used inside the Safe-Horizon MPC (SHMPC) framework in this codebase, then specializes to the **mode-coverage experiment** (Exp R / Fig. 24).

The main components are:

- Scenario-based SHMPC (QP with scenario collision constraints + safe horizon).
- An OT-based **ground cost** between trajectories / modes.
- A **Wasserstein DRO** problem over mode distributions \(q\), solved via a dual 1D search.
- Integration of the OT/DRO solution with the **safe horizon** constraint and **mode coverage** sampler.

The exposition here is consistent with `FORMULATION.md` but focused on OT and SHMPC.

---

### 0.1 Scenario-based SHMPC (recap)

Let \(x_k \in \mathbb{R}^{n_x}\) be the ego state, \(u_k \in \mathbb{R}^{n_u}\) the control, and \(T\) the planning horizon. The ego dynamics are
\[
  x_{k+1} = f(x_k, u_k), \quad k = 0,\dots,T-1,
\]
implemented by a linearized bicycle-like model in the code.

Each obstacle \(i\) has state \(o^i_k\) and a finite set of motion modes \(\mathcal{M}^i\) with linear dynamics
\[
  o^i_{k+1} = A^{m^i_k} o^i_k + b^{m^i_k}, \quad m^i_k \in \mathcal{M}^i.
\]

A **scenario** \(s\) is a sampled mode sequence for each obstacle and the induced trajectories
\[
  \{m^{i,s}_k\}_{k=0}^{T-1},\quad
  \{o^{i,s}_k\}_{k=0}^{T},\quad i=1,\dots,N_{\text{obs}}.
\]

We approximate ego and obstacles by discs of radii \(r_{\text{ego}}, r_{\text{obs}}\) with 2D positions \(p^{\text{ego}}_k(x_k)\) and \(p^{i,s}_k(o^{i,s}_k)\). A hard collision constraint is
\[
  \|p^{\text{ego}}_k - p^{i,s}_k\|_2 \;\ge\; r_{\text{ego}} + r_{\text{obs}}.
\]
Linearizing around a reference trajectory and stacking all constraints yields a QP of the form
\[
  \min_{z} \; J(z)
  \quad \text{s.t.} \quad C z \ge d,
\]
where \(z\) stacks all states and controls and \(Cz \ge d\) encodes dynamics, box constraints, and all linearized collision constraints across **all scenarios**.

The **safe-horizon** logic constrains the effective horizon \(N_{\text{safe}} \le T\) so that, given a finite scenario budget \(S\), the classic scenario-approach bound still guarantees a small violation probability \(\varepsilon\).

---

### 0.2 Safe-horizon bound and scenario budget

Let:

- \(S\) be the number of scenarios actually used in the QP;
- \(d = N_{\text{safe}} n_u\) be a surrogate for the decision dimension (number of control inputs over the safe horizon);
- \(\beta \in (0,1)\) be a confidence parameter (called `beta` in `config.hpp`);
- \(\varepsilon\) be the desired upper bound on constraint-violation probability.

The **simple scenario bound** used in the `SafeHorizonMode::THEORETICAL_SIMPLE` case (see `config.hpp`) is the standard inequality
\[
  S \;\ge\; \frac{2}{\varepsilon}
  \Bigl( \ln \tfrac{1}{\beta} + d \Bigr),
\]
which ensures that, with probability at least \(1-\beta\) over the random draw of scenarios,
\[
  \Pr\bigl(\text{constraint violation}\bigr) \le \varepsilon.
\]

Given a fixed scenario budget \(S_{\text{actual}}\), the code **inverts** this relation to find the largest safe horizon \(N_{\text{safe}}\) such that the inequality holds:
\[
  N_{\text{safe}}
  = \max \Bigl\{ N \le T : S_{\text{actual}} \ge
    \tfrac{2}{\varepsilon} \bigl( \ln \tfrac{1}{\beta} + N n_u \bigr) \Bigr\},
\]
clamped to \([\texttt{safe\_horizon\_min}, T]\). This is implemented in
`config.compute_safe_horizon(S_actual, n_u)`.

There is also a **tighter bound** mode (`SafeHorizonMode::THEORETICAL_TIGHT`) that uses a refined function \(S_{\text{req}}^{\text{tight}}(d)\), but conceptually it serves the same role: choose the largest \(N_{\text{safe}}\) such that the scenario bound is respected for a given \(S_{\text{actual}}\).

Once \(N_{\text{safe}}\) is chosen, SHMPC:

1. Only **requires safety guarantees up to \(N_{\text{safe}}\)** (beyond that, the horizon may be truncated), and
2. Passes \(N_{\text{safe}}\) as a **risk horizon** to the DRO/OT routines (see below), so that worst-case evaluation is aligned with the certified safe horizon.

---

### 0.3 OT ground cost and trajectory embedding

To define a Wasserstein distance between mode distributions, we need a **ground cost** between mode-conditioned trajectories.

Let \(\tau_i\) and \(\tau_j'\) denote trajectories (e.g., obstacle positions and margins along the horizon). We define an embedding
\[
  \phi(\tau) \in \mathbb{R}^d,
\]
for example by concatenating positions and clearance margins:
\[
  \phi(\tau) =
    \bigl[p_{0},\dots,p_T,\ \text{margin}_0,\dots,\text{margin}_T\bigr].
\]

The learned OT ground cost (see `FORMULATION.md`, §5) is then
\[
  D_{ij} = \bigl\|\phi(\tau_i) - \phi(\tau_j')\bigr\|_2.
\]

This matrix \(D\) is used as the **cost matrix** \(c(i,j)\) in the Wasserstein distance:
\[
  W_c(p, q)
    := \min_{\pi \in \Pi(p,q)} \sum_{i,j} D_{ij} \pi_{ij},
\]
where \(\Pi(p,q)\) is the set of couplings with marginals \(p\) and \(q\). Intuitively, modes whose trajectories lead to similar safety profiles (e.g. similar distances, margins) are “close” under this cost, and can exchange probability mass cheaply under OT.

---

### 0.4 Wasserstein DRO and dual formulation

Given a nominal discrete distribution \(w \in \Delta(\mathcal{M})\) (over modes or trajectories) and per-trajectory risk scores \(r_j\) (e.g., negative clearance or a loss function of the trajectory), the code solves a **Wasserstein-1 DRO** problem of the form:
\[
  \sup_{q \in \Delta(\mathcal{M})}
    \sum_{j} q_j r_j
  \quad \text{s.t.} \quad
  W_c(q, w) \le \varepsilon,
\]
with ground cost matrix \(D_{ij}\) induced by OT.

Using the **Kantorovich dual formulation** for \(W_1\), this can be written (see `FORMULATION.md`, §5) as the one-dimensional convex program
\[
  \min_{\lambda \ge 0}
    \Bigl\{
      \lambda \varepsilon +
      \sum_i w_i \max_j \bigl(r_j - \lambda D_{ij}\bigr)
    \Bigr\}.
\]

This is exactly what the implementation does:

- It precomputes \(D_{ij}\) via the OT predictor,
- For each candidate \(\lambda\), it computes the inner max over \(j\),
- Then it performs a 1D search over \(\lambda\) to find the minimizer and recover the corresponding **worst-case distribution** \(q^\star\).

That worst-case \(q^\star\) is then used either:

1. Directly as **mode weights** for sampling scenarios (`sample_scenarios_with_weights`), or
2. Indirectly by identifying the **most dangerous reachable mode** (highest weight / risk) and injecting a dedicated “worst-case” scenario for that mode (see `mpc_controller.cpp`, DRO injection).

Because the risk functional inside the DRO only considers timesteps up to the **safe horizon** \(N_{\text{safe}}\), the OT-based reweighting is tightly coupled to the SHMPC safe-horizon guarantee.

---

## Mode-Coverage Experiment (Fig. 24) – Implementation Notes and OT Logic

This note explains how the **mode-coverage experiment** (Exp R / Fig. 24) is implemented in the code, and gives a mathematical description of the **optimal transport (OT) reshaping** that drives the “OT” variant.

The relevant code lives in:

- `cpp_mpc/tests/paper_experiment_runner.cpp` – experiment driver (`run_experiment_r`)
- `cpp_mpc/src/scenario_sampler.cpp` / `scenario_sampler.hpp` – scenario sampling and coverage logic
- `cpp_mpc/src/mpc_controller.cpp` – Wasserstein DRO and OT-based reshaping logic
- `cpp_mpc/generate_results_figures.py` – plotting for `fig24_mode_coverage.png`

---

### 1. How the reshaped distribution over modes is computed

Each obstacle has a finite set of **modes**:
\[
  \mathcal{M} = \{\texttt{constant\_velocity},\ \texttt{turn\_left},\ \texttt{turn\_right},\ \texttt{decelerating}\}.
\]

The controller maintains a **mode history** for each obstacle (which modes have been observed over time). From this, it computes a **nominal empirical distribution** \(\hat p\) over modes:
\[
  \hat p(m) \approx \Pr(\text{mode} = m \mid \text{history}),
\]
using `compute_mode_weights(...)` with a chosen `WeightType` (plain frequency or OT-based).

For the **OT / reshaped** variant, this nominal distribution is passed into the Wasserstein DRO module:

- The **OT predictor** (`OptimalTransportPredictor`) maintains a cost matrix
  \[
    c(m, m') \ge 0
  \]
  between modes (derived from mode-conditioned trajectory embeddings and the ground cost, e.g. W2–Bures).
- The Wasserstein-1 distance between two discrete distributions \(p, q\) on \(\mathcal{M}\) is
  \[
    W_c(p, q)
      \;:=\;
      \min_{\pi \in \Pi(p, q)}
        \sum_{m, m' \in \mathcal{M}} c(m, m')\, \pi(m, m'),
  \]
  where \(\Pi(p, q)\) is the set of couplings with marginals \(p\) and \(q\).

The DRO module then solves a **worst-case reweighting** problem over distributions \(q\) in a Wasserstein ball around \(\hat p\):
\[
  \max_{q \in \Delta(\mathcal{M})}
    \; R(q)
    \quad \text{s.t.} \quad
    W_c(q, \hat p) \le \rho,
\]
where:

- \(\Delta(\mathcal{M})\) is the probability simplex over modes;
- \(\rho > 0\) is the **Wasserstein radius** (set from `dro_.set_observation_count(...)` and config);
- \(R(q)\) is a **risk functional** that approximates the “danger” of sampling according to \(q\).

Concretely, \(R(q)\) is implemented via a CVaR-style risk over predicted trajectories:

1. For each mode \(m\), the mode model generates a trajectory \(\tau_m\) (or a small set of representative trajectories).
2. A clearance / constraint-violation score
   \[
     \ell(m) = \ell(\tau_m)
   \]
   is computed (e.g. min distance to ego path minus safety margin).
3. The risk under \(q\) is something like
   \[
     R(q) = \mathrm{CVaR}_\alpha( \ell(m) \sim q ),
   \]
   or a worst-disc surrogate (for multi-disc clearance).

The result of the DRO solve is a **reshaped distribution**
\[
  q^\star(m) \in \Delta(\mathcal{M}),
\]
which:

- is **close** to \(\hat p\) in Wasserstein distance;
- **upweights dangerous modes** (those with worse \(\ell(m)\));
- still sums to 1, and is non-negative.

In the implementation:

- `dro_.compute_worst_case_weights(...)` returns `dro_result` containing these worst-case weights;
- these weights are then used in `sample_scenarios_with_weights(...)` as the **mode weights** when drawing scenarios for that obstacle.

So, the **reshaped distribution** used by the sampler is exactly this \(q^\star\): an OT-regularized worst-case perturbation of the empirical mode distribution.

---

### 2. What counts as a “reachable mode”

For each obstacle, the controller maintains:

- A list of **available modes**:
  \[
    \texttt{available\_modes} = \{ m_1, \dots, m_K \},
  \]
  with associated `ModeModel` dynamics.
- A set of **mode weights** \(w(m)\) (either \(\hat p(m)\) or \(q^\star(m)\)).

In the **mode-coverage sampler** (`sample_scenarios_with_mode_coverage` and `sample_scenarios_with_weights(..., ensure_mode_coverage=true)`), we define the **reachable modes** for obstacle \(o\) as:
\[
  \mathcal{M}_{\text{reach}}(o)
  := \{ m \in \texttt{available\_modes} : w(m) > 0 \}.
\]

For each reachable mode:

1. There is a valid **mode model** that can propagate the obstacle state using that mode.
2. The current weighting scheme assigns **nonzero probability** to it.

The coverage sampler guarantees that **each reachable mode is represented in at least one scenario**:

- It computes, for each obstacle, the list `coverage_modes` of all modes \(m\) with \(w(m) > 0\).
- It sets
  \[
    \texttt{num\_coverage}
    = \max_o |\mathcal{M}_{\text{reach}}(o)|,
  \]
  capped by the scenario budget \(S\).
- For the first `num_coverage` scenarios, it **forces one specific reachable mode** per obstacle by setting:
  \[
    w_{\text{forced}}(m') =
    \begin{cases}
      1, & m' = \text{coverage\_modes}[s],\\
      0, & \text{otherwise},
    \end{cases}
  \]
  and calling `sample_obstacle_trajectory(...)` with these forced weights.

Thus, in code terms, a **reachable mode** is any mode that:

- Is in `available_modes`, and
- Has a strictly positive weight in the (possibly OT-reshaped) distribution used for sampling.

---

### 3. Definition of the fixed scenario budget

The **scenario budget** is the fixed integer
\[
  S = \texttt{num\_scenarios}
\]
in `ScenarioMPCConfig` / `ExperimentConfig`. In Exp R (Fig. 24):

- `BASE_SCENARIOS` is set in `run_experiment_r`, and
- `cfg.num_scenarios = BASE_SCENARIOS;`

This budget is **held constant** across variants (Base vs. OT) and across timesteps:

- Every MPC solve uses the **same number of scenarios S**.
- The coverage sampler **reallocates** this fixed budget across modes, but never increases S.

Concretely, the sampling algorithm does:

1. **Phase 1 – Coverage scenarios** (up to `num_coverage` scenarios):
   - For each coverage index \(s = 0, \dots, \texttt{num\_coverage}-1\), and for each obstacle, it either:
     - forces the \(s\)-th reachable mode (if it exists), or
     - samples from the normal weights (if we’ve already covered all modes).
   - The resulting scenario is pushed into the `scenarios` vector.

2. **Phase 2 – Remaining scenarios**:
   - For \(s = \texttt{num\_coverage}, \dots, S-1\), it samples trajectories for each obstacle from the weights \(w(m)\) (either \(\hat p\) or \(q^\star\)), without forcing coverage, and appends those scenarios.

At the end we have exactly **S scenarios**:
\[
  |\texttt{scenarios}| = S,
\]
for both **Base** and **OT** variants. The only difference is how that fixed budget is **distributed across modes**:

- Base: sampling from the empirical weights \(\hat p(m)\).
- OT: sampling from the reshaped worst-case weights \(q^\star(m)\).

This is what Fig. 24 visualizes:

- **`true_fraction`**: empirical mode frequency in the **true switching process** (how often the environment actually visits each mode).
- **`sampled_fraction`**: empirical fraction of **scenario slots** allocated to each mode by the sampler, under Base or OT.
- **`coverage_ratio`**: the ratio
  \[
    \frac{\text{sampled\_fraction}}{\text{true\_fraction}},
  \]
  reporting whether a mode is over‑ or under‑sampled relative to its true occurrence.

The experiment demonstrates that, under a **fixed scenario budget S**, the OT‑reshaped distribution can significantly improve **rare‑mode coverage** (higher `sampled_fraction` and `coverage_ratio` for rare but dangerous modes), while Base sampling tends to under‑allocate scenarios to those modes.

