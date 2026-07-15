# Wasserstein DRO and Safe-Horizon Theory

Self-contained formal description of the Wasserstein DRO mode reweighting and safe-horizon theory used in this codebase.

---

## 1. Scenario-based SHMPC

Let $x_k \in \mathbb{R}^{n_x}$ be the ego state, $u_k \in \mathbb{R}^{n_u}$ the control, and $T$ the planning horizon. Ego dynamics:

$$x_{k+1} = f(x_k, u_k), \quad k = 0,\dots,T-1$$

implemented as an RK4-discretized bicycle model in `src/dynamics.cpp`.

Each obstacle $i$ has state $o^i_k$ and a finite set of motion modes $\mathcal{M}^i$ with linear dynamics:

$$o^i_{k+1} = A^{m^i_k} o^i_k + b^{m^i_k} + G^{m^i_k} w_k, \quad m^i_k \in \mathcal{M}^i$$

A **scenario** $s$ is a sampled mode sequence for each obstacle and the induced trajectories. Linearized collision constraints are stacked into a QP:

$$\min_{z} J(z) \quad \text{s.t.} \quad C z \ge d$$

The **safe-horizon** logic constrains the effective horizon $N_{\text{safe}} \le T$ so that the classic scenario-approach bound guarantees a small violation probability $\varepsilon$.

---

## 2. Safe-Horizon Bound

Let $S$ be the number of i.i.d. sampled scenarios, $d = N_{\text{safe}} \cdot n_u$ the decision dimension, $\beta$ the confidence parameter, and $\varepsilon$ the desired violation probability bound.

The scenario bound (`SafeHorizonMode::THEORETICAL_SIMPLE` in `include/config.hpp`):

$$S \ge \frac{2}{\varepsilon} \left( \ln \frac{1}{\beta} + d \right)$$

Given a fixed scenario budget $S_{\text{actual}}$, the code inverts this to find:

$$N_{\text{safe}} = \max \left\{ N \le T : S_{\text{actual}} \ge \frac{2}{\varepsilon} \left( \ln \frac{1}{\beta} + N \cdot n_u \right) \right\}$$

clamped to $[\texttt{safe\_horizon\_min}, T]$. Implemented in `config.compute_safe_horizon(S_actual, n_u)`.

**Critical**: Only i.i.d. sampled scenarios count toward $S$. DRO-injected scenarios are deterministic constraints that tighten the feasible set but do NOT satisfy the i.i.d. assumption. They are NOT counted in $S_{\text{for\_sh}}$.

---

## 3. Wasserstein DRO Mode Reweighting

For each obstacle, compute worst-case mode distribution $q^*$ in a Wasserstein ball around nominal belief $\hat{p}$:

$$\sup_{q \in \Delta_M} \sum_m q_m r_m \quad \text{s.t.} \quad W_D(q, \hat{p}) \le \rho$$

Solved via Kantorovich dual (1D binary search on $\lambda$ in `src/wasserstein_dro.cpp`):

$$\inf_{\lambda \ge 0} \left\{ \lambda \rho + \sum_i \hat{p}_i \max_j (r_j - \lambda D_{ij}) \right\}$$

### Ground cost matrix $D$

$D_{ij}$ is the W2-Bures distance between mode-conditioned trajectory Gaussians:

$$D[i][j] = \frac{1}{N} \sum_k W_2(P_i(k), P_j(k))$$

where $W_2^2 = \|\mu_1 - \mu_2\|^2 + \text{Bures}^2(\Sigma_1, \Sigma_2)$.

Alternative ground costs (configurable via `DROGroundCostType`): `W2_BURES`, `ZERO_ONE`, `EUCLIDEAN_MEAN`.

### Risk vector

Per-mode risk using conservative Gaussian clearance surrogate:

$$r_m = \max_{k \le N_s} \max_{d \le D} r_{m,k,d}$$

where $r_{m,k,d} = \max(0, R + z_\alpha \sigma_{m,k,d} - \|\mu_{m,k} - c_d(k)\|)$ and $\sigma_{m,k,d} = \sqrt{n^T \Sigma n}$ is the directional standard deviation toward the ego disc.

### Adaptive rho

$$\rho = \rho_0 \cdot \left(1 + \frac{\alpha}{\sqrt{n_{\text{obs}}}}\right)$$

Contracts as observations accumulate. Config: `dro_rho_base`, `dro_adaptive_rho`.

---

## 4. DRO Scenario Injection

Three injection selection rules for $K$ deterministic constraints:

1. **WDRO-K**: Inject $K$ modes with largest mass under $q^*$. Uses both risk scores and transport geometry.

2. **TopRisk-K**: Greedy risk-only heuristic. Inject $K$ modes with largest risk scores $r_m$.

3. **DiverseRisk-K**: Risk x diversity selection:
   - $m_1 = \arg\max_m r_m$
   - $m_l = \arg\max_{m \notin S_{l-1}} (r_m \cdot \min_{s \in S_{l-1}} D(m,s))$

Constraint tightening: $F_{S,\text{inj}} \subseteq F_S$, so injection only shrinks the feasible set and preserves scenario-theoretic guarantees.

Config: `injection_mode`, `dro_injection_count`.

---

## 5. Mode-Coverage Sampling

The coverage sampler (`sample_scenarios_with_mode_coverage` in `src/scenario_sampler.cpp`) guarantees each mode with positive weight is represented in at least one scenario:

**Phase 1 (coverage)**: For each reachable mode $m$ with $w(m) > 0$, force one scenario to use that mode.

**Phase 2 (fill)**: Remaining budget sampled from weights $w(m)$ via categorical sampling.

Total scenario count is exactly $S = \texttt{num\_scenarios}$, held constant across variants.

### What counts as a "reachable mode"

A mode $m$ is reachable for obstacle $o$ if:
- $m \in \texttt{available\_modes}$, and
- $w(m) > 0$ under the current weighting scheme

---

## 6. Mode Weight Strategies

The "OT" paper variant uses `WeightType::WASSERSTEIN` -- a frequency-recency blend (0.3x frequency + 0.7x recency with decay 0.85), implemented in `src/mode_weights.cpp`. This is NOT a Sinkhorn OT solver.

| Strategy | Formula |
|----------|---------|
| UNIFORM | $w_m = 1/M$ |
| FREQUENCY | $w_m = \text{count}_m / \text{total}$ |
| RECENCY | $w_m \propto \sum_t \lambda^{T-t} \cdot \mathbb{1}[\text{mode}_t = m]$ |
| WASSERSTEIN | $0.3 \cdot w_{\text{freq}} + 0.7 \cdot w_{\text{recency}}$ |
| TEMPERATURE | $w'_m = \exp(\ln w_m / T)$, $T=0.5$ |
| EPSILON_GREEDY | $w'_m = (1-\varepsilon)w_m + \varepsilon/M$ |
