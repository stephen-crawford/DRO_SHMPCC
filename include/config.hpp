/**
 * @file config.hpp
 * @brief Runtime configuration for Adaptive Scenario-Based MPC — SOURCE OF TRUTH.
 *
 * Owns controller settings only:
 *   - Ego vehicle specification (geometry + kinematic limits)
 *   - MPC type, horizon, objective, constraints, sampling / belief
 *   - DRO on/off, risk measure, ground cost, ambiguity radius, OT
 *   - QP / SQP solver knobs
 *
 * Numeric defaults MUST match configs/default.yaml (the runtime source of
 * truth). In-class initializers exist only as a fallback when that file
 * cannot be loaded.
 *
 * World setup (obstacles, environment, rollout protocol) lives in
 * experiment_harness.hpp.
 *
 * Split:
 *   config.hpp             — controller runtime (RuntimeConfig = mpc + dro + solver)
 *   experiment_harness.hpp — world / trial protocol (ExperimentConfig wraps the above)
 *
 */

#ifndef DRO_MPC_CONFIG_HPP
#define DRO_MPC_CONFIG_HPP

#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dro_mpc {

// ============================================================================
// MPC type
// ============================================================================

enum class MPCType {
    MPC,     // Point-to-point / goal-tracking
    MPCC,    // Contouring control (path following)
    SH_MPC,  // Safe-horizon MPC
    SH_MPCC  // Safe-horizon MPCC
};

inline std::string mpc_type_name(MPCType t) {
    switch (t) {
        case MPCType::MPC:     return "mpc";
        case MPCType::MPCC:    return "mpcc";
        case MPCType::SH_MPC:  return "sh_mpc";
        case MPCType::SH_MPCC: return "sh_mpcc";
        default: return "unknown";
    }
}

/// Compatibility alias used by paper-arm helpers.
using MPCConfiguration = MPCType;

/// @brief Safe-horizon truncation rule: how many stages N_s <= N receive
/// collision constraints, given S sampled scenarios. Declared before use below.
enum class SafeHorizonTruncationRule {
    FIXED_NBAR, // DEFAULT: de Groot's strategy: fixed support cap, independent of the horizon length
    // ---- Approximations / heuristics (do NOT match de Groot's support bound) ---
    UNCERTIFIED_PRACTICAL,           //  Heuristic N_safe = min(N, floor(S/(2*n_u))). Ignores eps/beta.Lacks certification guarantee.
    THEORETICAL_TIGHT,   //  Campi-Garatti tight bound with support dim = N_s*n_u (conservative).
    THEORETICAL_SIMPLE,  // Calafiore-Campi 2006 bound S >= (2/eps)*(ln(1/beta) + N_s*n_u).
};

// ============================================================================
// MPC objective / constraints / ego
// ============================================================================

struct MPCObjectiveWeights {
    double goal_weight = 10.0;
    double velocity_weight = 1.0;
    double acceleration_weight = 0.1;
    double steering_weight = 0.1;
    double terminal_heading_weight = 1.0;
    double contour_weight = 1.0;            //  Active for MPCC / SH_MPCC
    double lag_weight = 0.1;                // Active for MPCC / SH_MPCC
};

struct MPCConstraintSettings {
    double safety_margin = 0.1;
    double road_width = 7.0;                //  Active when contouring is on

    /// Hard box on v_{k+1} in [ego.dynamics.min_velocity, ego.dynamics.max_velocity].
    bool enable_velocity_bounds = true;

    // Safe-horizon knobs — active only for SH_MPC / SH_MPCC
    int safe_horizon_min = 12;
    SafeHorizonTruncationRule safe_horizon_mode = SafeHorizonTruncationRule::FIXED_NBAR;
    int forced_safe_horizon = -1;

    /// Drop collision half-spaces whose linearization point is farther than this
    /// from the reference (metres). Does not affect scenario dominance pruning.
    double clearance_filter_distance = 20.0;

    /// de Groot support cap n̄: an upper limit on the number of DISTINCT SUPPORT
    /// SCENARIOS (counted by unique scenario_id), NOT individual collision constraints.
    /// Support estimate is the UNION of active scenario IDs across all feasible convex
    /// iterations, n̂ = |∪_ℓ ω_active^ℓ| (with C ⊆ Ĉ, n ≤ n̂), not just the constraints
    /// active at the final optimum. This is the NON-REMOVED support cap: with a removal
    /// budget R the TOTAL support limit is n̄ + R (removed scenarios join the support,
    /// de Groot Thm. 5)
    int support_cap_nbar = 5;
};

// ---------------------------------------------------------------------------
// Ego dynamics model
// ---------------------------------------------------------------------------


enum class DynamicsModel {
    SECOND_ORDER_UNICYCLE  //State [x,y,theta,v], input [a, w]. (Others deferred.)
};

inline std::string dynamics_model_name(DynamicsModel m) {
    switch (m) {
        case DynamicsModel::SECOND_ORDER_UNICYCLE: return "second_order_unicycle";
        default: return "unknown";
    }
}

/**
 * @brief Ego dynamics-model specification: which motion model, plus the
 *        kinematic limits enforced ON that model.
 */
struct EgoDynamicsConfig {
    DynamicsModel model = DynamicsModel::SECOND_ORDER_UNICYCLE;

    double max_velocity = 4.0;       // Hard upper velocity bound [m/s]
    double min_velocity = 0.0;       // Hard lower velocity bound [m/s]
    double max_acceleration = 3.0;   // Maximum acceleration [m/s^2]
    double min_acceleration = -5.0;  // Minimum acceleration (braking) [m/s^2]
    double max_omega = 0.8;  // Maximum angular velocity [rad/s]
};

/**
 * @brief Ego vehicle geometry + dynamics model.
 *
 * Geometry (collision radius, body length, disc count) is separate from motion:
 * the dynamics model and its kinematic limits live in `dynamics`.
 */
struct EgoVehicleSpecification {
    double radius = 0.5;             // Collision radius [m]
    double length = 4.0;             // Length for multi-disc placement [m]
    int num_discs = 3;               // Number of discs along the body

    EgoDynamicsConfig dynamics;      // Motion model + kinematic limits
};

// ============================================================================
// Nominal mode belief
// ============================================================================

enum class NominalBeliefKind {
    DIRICHLET,  //Symmetric Dirichlet prior
    STICKY      //Dirichlet + self-persistence (sticky) prior
};

// Build a ModeBeliefConfig (types.hpp) from the belief kind. The sticky
// self-persistence prior theta is only wired in for STICKY.
inline ModeBeliefConfig make_mode_belief(
    NominalBeliefKind kind,
    double self_persistence_prior = 0.8
) {
    ModeBeliefConfig cfg;
    cfg.prior = DirichletPrior::KRICHEVSKY_TROFIMOV;
    cfg.self_persistence_prior =
        (kind == NominalBeliefKind::STICKY) ? self_persistence_prior : 0.0;
    return cfg;
}

/**
 * @brief Scenario sampling and nominal mode belief.
 *
 * Active for scenario-based MPC. Markov sampling / sticky belief only take
 * effect when the corresponding flags / kinds are selected.
 */
struct ScenarioSamplingSettings {
    int num_scenarios = 40; // Samples per step (default operating point; S-sweeps override this)

    double one_minus_chance_constraint_violation_probability = 0.95; // = 1 - eps  (safety prob)
    double chance_of_certificate_violation = 0.01; // = beta  (certificate confidence = 1 - beta)

    bool enforce_certified_scenario_count = false;
    int max_history_length = -1;

    bool markov_jump_system = false;

    NominalBeliefKind belief_kind = NominalBeliefKind::DIRICHLET;
    ModeBeliefConfig mode_belief{};

    void sync_belief() {
        const double sticky =
            mode_belief.self_persistence_prior > 0.0
                ? mode_belief.self_persistence_prior : 0.8;
        mode_belief = make_mode_belief(belief_kind, sticky);
    }

    double epsilon() const { return 1.0 - one_minus_chance_constraint_violation_probability; }
};

// ============================================================================
// MPC config
// ============================================================================

struct MPCConfig {
    MPCType type = MPCType::SH_MPCC;
    int horizon = 20;
    double dt = 0.1;

    EgoVehicleSpecification ego;
    MPCObjectiveWeights objective;
    MPCConstraintSettings constraints;
    ScenarioSamplingSettings sampling;

    bool safe_horizon_enabled = true;
    bool enable_contouring_constraints = true;  //Road-boundary + contouring cost

    /// Derive safe-horizon / contouring enablement from the MPC type.
    /// Call after setting `type` and before overriding those two flags by hand.
    void sync_from_type() {
        switch (type) {
            case MPCType::MPC:
                safe_horizon_enabled = false;
                enable_contouring_constraints = false;
                break;
            case MPCType::MPCC:
                safe_horizon_enabled = false;
                enable_contouring_constraints = true;
                break;
            case MPCType::SH_MPC:
                safe_horizon_enabled = true;
                enable_contouring_constraints = false;
                break;
            case MPCType::SH_MPCC:
                safe_horizon_enabled = true;
                enable_contouring_constraints = true;
                break;
        }
    }

    bool uses_safe_horizon() const { return safe_horizon_enabled; }
};

// ============================================================================
// DRO — risk / ground cost
// ============================================================================
//
// DROGroundCostType and DRORiskMeasure are defined in types.hpp.

inline std::string ground_cost_name(DROGroundCostType g) {
    switch (g) {
        case DROGroundCostType::W2_BURES:      return "w2_bures";
        case DROGroundCostType::W1_METRIC:     return "w1_metric";
        case DROGroundCostType::ZERO_ONE:      return "zero_one";
        case DROGroundCostType::EUCLIDEAN_MEAN: return "euclidean_mean";
        default: return "unknown";
    }
}

inline std::string risk_measure_name(DRORiskMeasure r) {
    switch (r) {
        case DRORiskMeasure::SURROGATE_VAR:            return "surrogate_var";
        case DRORiskMeasure::SURROGATE_CVAR:           return "surrogate_cvar";
        case DRORiskMeasure::SURROGATE_VAR_BONFERRONI: return "surrogate_var_bonferroni";
        case DRORiskMeasure::BONFERRONI_VAR:           return "bonferroni_var";
        case DRORiskMeasure::MIXTURE_VAR:              return "mixture_var";
        case DRORiskMeasure::MIXTURE_CVAR:             return "mixture_cvar";
        case DRORiskMeasure::JOINT_VAR:                return "joint_var";
        case DRORiskMeasure::JOINT_CVAR:               return "joint_cvar";
        default: return "unknown";
    }
}

// ============================================================================
// DRO — ambiguity-radius calibration
// ============================================================================

/**
 * @brief Calibration knobs for the configured ambiguity radius and reweighting.
 *
 * Radius theory (true W1 concentration — see DRO::get_adaptive_rho):
 *   The nominal belief p_hat is an empirical categorical over M modes from n
 *   observed interactions. In total variation it concentrates as
 *       P( ||p_hat - p*||_1 >= eps ) <= 2^M exp(-n eps^2 / 2)      (Devroye),
 *   so at target miscoverage beta the L1 half-width is
 *       eps_n(beta) = sqrt( 2 (M ln2 + ln(1/beta)) / n ).
 *   For ANY metric ground cost D, W1 is dominated by the transport diameter:
 *       W1(p_hat, p*) <= (1/2) * diam(D) * ||p_hat - p*||_1
 *                     <= (1/2) * diam(D) * eps_n(beta),
 *   with diam(D) = max_{i,j} D[i][j]. The ground-metric diameter is folded in
 *   EXPLICITLY (not hidden inside a base radius), so
 *       rho_n(beta) = min_radius + calibration_scale * (1/2) * diam(D) * eps.
 *   This shrinks to min_radius as n -> inf (statistical consistency) and grows
 *   with the mode count M, the confidence level, and the ground-cost scale.
 */
struct RadiusCalibrationSettings {
    bool use_calibrated_radius = true;   //Use the true-W1 concentration radius above
    double confidence_beta = 0.05;       //Target miscoverage (1 - beta coverage)
    double alpha_one_sided = 0.95;       //Risk level alpha (VaR/CVaR / surrogate z_alpha)

    /// Dimensionless safety factor multiplying the (1/2)*diam(D)*eps radius.
    /// 1.0 = the bare concentration bound; >1 inflates it.
    double calibration_scale = 1.0;

    /// Exact W1 primal OT reweighting instead of dual-guided heuristic recovery.
    bool use_primal_ot = true;

    DRORiskMeasure risk_measure = DRORiskMeasure::SURROGATE_VAR_BONFERRONI;

    /// If positive, overrides the controller-provided risk horizon. Otherwise
    /// the active safe horizon (or full MPC horizon) is used.
    int risk_horizon = -1;

    AmbiguityDivergence divergence = AmbiguityDivergence::WASSERSTEIN;

    /// Monte Carlo sample count / seed for JOINT_VAR / JOINT_CVAR (offline).
    int joint_risk_samples = 8000;
    uint64_t joint_risk_seed = 0x5150C0FFEEULL;

    /// Mode-SEQUENCE sample count K for MIXTURE_VAR / MIXTURE_CVAR. Only the chain is
    /// sampled (the noise is integrated in closed form), so K buys mixture-component
    /// resolution rather than tail resolution and 512 is affordable in the loop.
    /// Ignored when no transition matrix is supplied: the mixture then has one
    /// component and MIXTURE_* collapses onto SURROGATE_*.
    int mixture_sequence_samples = 512;

    double sigma_floor = 1e-6;           //Floor for directional sigma

    /// Entropic allocator: keeps q_min > 0 so the certificate L = 1/q_min is finite.
    bool use_entropic_allocator = true;
    double entropic_tau = 0.05;          //Temperature; tau -> 0 recovers the raw LP
};

/**
 * @brief DRO solver knobs (radius, ground cost, calibration, OT).
 *
 * Consumed directly by DRO. The flat radius fields set the clamp
 * band and the non-calibrated fallbacks; the nested radius_calibration holds
 * the confidence-calibrated radius parameters and the OT / risk selection.
 */
struct DROConfig {
    double base_radius = 0.1;        //Base radius rho (non-calibrated / fixed use)
    double min_radius = 0.01;        //Minimum rho (clamp floor; calibrated radius offset)
    double max_radius = 0.10;        //Maximum rho (clamp ceiling; below mode-transport collapse)

    RadiusCalibrationSettings radius_calibration;

    DROGroundCostType ground_cost_type = DROGroundCostType::W2_BURES;
};

/**
 * @brief Controller-facing DRO settings.
 *
 */
struct DROControllerConfig {
    bool enabled = false;

    /// If > 0, disable calibrated rho and pin the ambiguity radius to this value.
    double fixed_rho = -1.0;

    DROConfig solver;

    void apply_fixed_rho() {
        if (fixed_rho <= 0.0) return;
        solver.radius_calibration.use_calibrated_radius = false;
        solver.base_radius = fixed_rho;
        solver.min_radius = std::min(solver.min_radius, fixed_rho);
        solver.max_radius = std::max(solver.max_radius, fixed_rho);
    }
};

// ============================================================================
// Solver
// ============================================================================

struct SolverSettings {
    bool use_sqp_solver = true;        // SQP outer loop over QP subproblems.
    int sqp_max_iterations = 5;        // Maximum SQP outer iterations.
    double sqp_convergence_tol = 1e-3; // Convergence tolerance on ||delta_u||.
    int qp_max_iterations = 200;       // acados/HPIPM interior-point iteration cap.
    double qp_tolerance = 1e-4;        // acados/HPIPM residual tolerance.
};

// ============================================================================
// Complete runtime config
// ============================================================================

/**
 * @brief Full controller runtime config: MPC + DRO + solver + world radius.
 */
struct RuntimeConfig {
    MPCConfig mpc;
    DROControllerConfig dro;
    SolverSettings solver;

    /// Seed for controller-owned random sampling. Zero selects nondeterministic seeding.
    unsigned random_seed = 0;

    /// Obstacle collision radius used with ego.radius for halfspaces.
    double obstacle_radius = 0.35;

    double combined_radius() const {
        return mpc.ego.radius + obstacle_radius + mpc.constraints.safety_margin;
    }

    double epsilon() const { return mpc.sampling.epsilon(); }

    bool enable_dro() const { return dro.enabled; }

    // ---- de Groot (arXiv:2307.01070) scenario-theoretic sample complexity ----
    // de Groot's Safe-Horizon MPC sizes the sample count S from the NONCONVEX
    // scenario-optimization (NSO) bound of Campi-Garatti, NOT the convex
    // Calafiore-Campi dimension bound. The guarantee is stated through the SUPPORT
    // SIZE n (the number of scenarios that hold the solution in place), capped by a
    // support limit n̄. de Groot Eq. (7)-(8) + Theorem 1 / Algorithm 1 line 2. The
    // theorem requires the confidence tail to be at most β:
    //
    //   P^S[ V(θ*) > ε(n) ]  ≤  Σ_{k=0}^{S-1} C(S,k) (1-ε(k))^{S-k}  ≤  β,
    //   ε(n) = 1 - ( β / (S · C(S,n)) )^{1/(S-n)}   for n ≤ n̄ ,   ε(n)=1 for n>n̄.
    //
    // Each term with k ≤ n̄ contributes exactly β/S; with the piecewise cap ε(k)=1 for
    // k > n̄ the terms above the cap vanish, so the sum is (n̄+1)·β/S ≤ β (equality Σ = β
    // would need ε_S(k) applied for ALL k, i.e. no cap). Using ε(n̄) ≤ ε then certifies
    // every n ≤ n̄ since ε(n) is increasing in n.
    //
    // and S is the SMALLEST sample size with ε(n̄) ≤ ε ("bisection of Eq. 8").
    // Scenario removal (Sec. VI, Thm. 5): removed scenarios join the support, so a
    // removal budget R raises the effective support limit to n̄ + R.

    /// Realized per-decision violation risk ε(n) at sample size S and support n
    /// (de Groot Eq. 8), evaluated in log-space for numerical stability. Returns 1.0
    /// when n ≥ S (under-sampled → no guarantee).
    static double degroot_violation_risk(int S, int n, double beta) {
        if (S <= 0 || n >= S) return 1.0;
        if (n < 0) n = 0;
        const double log_binom = std::lgamma(S + 1.0)
                               - std::lgamma(n + 1.0)
                               - std::lgamma(S - n + 1.0);
        // (1-ε)^{S-n} = β / (S·C(S,n))  ⇒  ε = 1 - exp( [ln β - ln S - lnC] / (S-n) ).
        const double log_inner =
            std::log(beta) - std::log(static_cast<double>(S)) - log_binom;
        return 1.0 - std::exp(log_inner / static_cast<double>(S - n));
    }

    /// de Groot Algorithm 1, line 2: the smallest sample size S with ε(n_total) ≤ ε,
    /// found by exponential-search + bisection (ε is monotonically decreasing in S).
    /// This is the EXACT NSO bound de Groot bisects; it replaces the earlier closed-form
    /// Alamo/Campi upper bound, which over-estimated S by 15-30% (e.g. n̄=5: exact 781 vs
    /// closed-form 932 at ε=0.05, β=0.01).
    
    int compute_required_scenarios(int nonremoved_support_limit, int num_removal = 0) const {
        const double eps  = epsilon();
        const double beta = mpc.sampling.chance_of_certificate_violation;
        const int n = std::max(0, nonremoved_support_limit) + std::max(0, num_removal);  // total support
        int hi = std::max(n + 1, 1);                       // exponential upper bracket
        while (degroot_violation_risk(hi, n, beta) > eps) {
            hi *= 2;
            if (hi > (1 << 24)) return hi;                 // safety cap (~16.7M)
        }
        int lo = n + 1;
        while (lo < hi) {                                  // bisection (monotone in S)
            const int mid = lo + (hi - lo) / 2;
            if (degroot_violation_risk(mid, n, beta) <= eps) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    /// Convex Calafiore-Campi bound  S ≥ (2/ε)(ln(1/β) + d), where the support is
    /// bounded by the DECISION-VARIABLE dimension d. This is a DIFFERENT (convex)
    /// guarantee than de Groot's nonconvex NSO bound above; it is retained only for
    /// the THEORETICAL_SIMPLE safe-horizon mode, which uses the convex proxy d=N·n_u.
    int compute_required_scenarios_simple(int d) const {
        double eps = epsilon();
        return static_cast<int>(std::ceil(
            (2.0 / eps) * (std::log(1.0 / mpc.sampling.chance_of_certificate_violation) + d)
        ));
    }

    /// In de Groot, "Safe Horizon MPC" means the constraints bound the JOINT collision
    /// probability over the planned horizon; the sample requirement is horizon-independent
    /// in that it depends on the support limit n̄ rather than N explicitly. It does NOT
    /// say the horizon is auto-selected from S. Each rule below is a controller heuristic;
    /// only FIXED_NBAR carries a genuine certificate, and only for the full horizon.
    int compute_safe_horizon(int S_actual, int n_u = 2) const {
        if (!mpc.uses_safe_horizon()) return mpc.horizon;

        const auto& c = mpc.constraints;
        if (c.forced_safe_horizon >= 0) {
            return std::clamp(c.forced_safe_horizon, c.safe_horizon_min, mpc.horizon);
        }

        int N_safe = c.safe_horizon_min;
        switch (c.safe_horizon_mode) {
            case SafeHorizonTruncationRule::FIXED_NBAR:
                // de Groot's certified strategy: the support cap fixes the total support at
                // n̄ (horizon-independent), so the exact NSO bound (Eq. 8) is a single
                // threshold. If S certifies n̄, the FULL horizon carries the P(collision) ≤ ε
                // @ 1-β guarantee. HEURISTIC FALLBACK: if S is insufficient, dropping to
                // safe_horizon_min is NOT automatically certified — a shorter horizon may
                // have smaller support, but that support must still be MEASURED/bounded
                // (n̂ ≤ n̄_max(S)) to certify it. Treat the short-horizon branch as an
                // engineering fallback, not a proof.
                N_safe = (S_actual >= compute_required_scenarios(c.support_cap_nbar))
                             ? mpc.horizon
                             : c.safe_horizon_min;
                break;
            case SafeHorizonTruncationRule::UNCERTIFIED_PRACTICAL:
                N_safe = std::min(mpc.horizon, S_actual / (2 * n_u));
                break;
            case SafeHorizonTruncationRule::THEORETICAL_SIMPLE:
                for (int N_try = mpc.horizon; N_try >= c.safe_horizon_min; --N_try) {
                    if (S_actual >= compute_required_scenarios_simple(N_try * n_u)) {
                        N_safe = N_try;
                        break;
                    }
                }
                break;
            case SafeHorizonTruncationRule::THEORETICAL_TIGHT:
                // Evaluates de Groot's Eq. 8 EXACTLY, but with the support PROXIED by the
                // per-step decision-variable count N·n_u. That proxy is NOT de Groot's
                // online active-scenario support estimate (a count of distinct support
                // scenarios, which the SQP union can make larger OR smaller than N·n_u),
                // so this is NOT certified unless N·n_u is independently established as a
                // valid upper bound on the total scenario support. It is a design proxy,
                // despite the "TIGHT" label.
                for (int N_try = mpc.horizon; N_try >= c.safe_horizon_min; --N_try) {
                    if (S_actual >= compute_required_scenarios(N_try * n_u)) {
                        N_safe = N_try;
                        break;
                    }
                }
                break;
        }
        return std::clamp(N_safe, c.safe_horizon_min, mpc.horizon);
    }

    /// Realized joint-risk certificate ε(n_total) actually guaranteed by S_actual drawn
    /// scenarios (de Groot Eq. 8) — the exact inverse of compute_required_scenarios.
    /// The support arguments follow the SAME non-removed convention: the total support
    /// bound is nonremoved_support_bound + R. Pass either the measured online support n̂
    /// (with R=0) or the configured n̄ + R; in both cases it must be a valid UPPER bound
    /// on the total support for the returned ε to be a valid certificate.
    double compute_effective_epsilon(int S_actual, int nonremoved_support_bound, int R = 0) const {
        return degroot_violation_risk(S_actual,
                                      std::max(0, nonremoved_support_bound) + std::max(0, R),
                                      mpc.sampling.chance_of_certificate_violation);
    }

    void normalize() {
        // Do not call mpc.sync_from_type() here — it would overwrite SH/contouring
        // overrides intentionally set after type selection.
        mpc.sampling.sync_belief();
        dro.apply_fixed_rho();
    }

    void validate() const {
        if (mpc.horizon <= 0) throw std::invalid_argument("horizon must be positive");
        if (mpc.dt <= 0) throw std::invalid_argument("dt must be positive");
        if (mpc.sampling.one_minus_chance_constraint_violation_probability <= 0 || mpc.sampling.one_minus_chance_constraint_violation_probability >= 1)
            throw std::invalid_argument("confidence_level must be in (0, 1)");
        if (mpc.sampling.chance_of_certificate_violation <= 0 || mpc.sampling.chance_of_certificate_violation >= 1)
            throw std::invalid_argument("beta must be in (0, 1)");
        if (mpc.sampling.num_scenarios <= 0)
            throw std::invalid_argument("num_scenarios must be positive");
    }
};

}  // namespace dro_mpc

#endif  // DRO_MPC_CONFIG_HPP
