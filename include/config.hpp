/**
 * @file config.hpp
 * @brief Runtime configuration for Adaptive Scenario-Based MPC — SOURCE OF TRUTH.
 *
 * Owns controller settings only:
 *   - Ego vehicle specification (geometry + kinematic limits)
 *   - MPC type, horizon, objective, constraints, sampling / belief
 *   - DRO on/off, injection, risk measure, ground cost, ambiguity radius, OT
 *   - QP / SQP solver knobs
 *
 * World setup (obstacles, environment, rollout protocol) lives in
 * experiment_harness.hpp.
 *
 * Split:
 *   config.hpp             — controller runtime (RuntimeConfig = mpc + dro + solver)
 *   experiment_harness.hpp — world / trial protocol (ExperimentConfig wraps the above)
 *
 * The core vocabulary enums (WeightType, DROGroundCostType, DRORiskMeasure,
 * DirichletPrior, ModeBeliefConfig) are defined once in types.hpp and reused
 * here — they are NOT redefined.
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

enum class MPCConfiguration {
    MPC,     ///< Point-to-point / goal-tracking
    MPCC,    ///< Contouring control (path following)
    SH_MPC,  ///< Safe-horizon MPC
    SH_MPCC  ///< Safe-horizon MPCC
};

inline std::string mpc_configuration_name(MPCConfiguration t) {
    switch (t) {
        case MPCConfiguration::MPC:     return "mpc";
        case MPCConfiguration::MPCC:    return "mpcc";
        case MPCConfiguration::SH_MPC:  return "sh_mpc";
        case MPCConfiguration::SH_MPCC: return "sh_mpcc";
        default: return "unknown";
    }
}

/// @brief Safe-horizon truncation rule. Declared before it is used below.
enum class SafeHorizonMode {
    PRACTICAL,           ///< N_safe = min(N, floor(S / (2*n_u)))
    THEORETICAL_TIGHT,   ///< Tight bound (Eq. 25): very conservative
    THEORETICAL_SIMPLE,  ///< Simple bound (Eq. 23): S >= (2/eps)*(ln(1/beta) + d)
    HEURISTIC            ///< Fixed at safe_horizon_min
};

// ============================================================================
// MPC objective / constraints / ego
// ============================================================================

struct MPCObjectiveWeights {
    double goal_weight = 10.0;
    double velocity_weight = 1.0;
    double acceleration_weight = 0.1;
    double steering_weight = 0.1;
    double contour_weight = 1.0;            ///< Active for MPCC / SH_MPCC
    double lag_weight = 0.1;                ///< Active for MPCC / SH_MPCC
    double terminal_heading_weight = 1.0;
    double goal_weight_scale_max = 6.0;
    double goal_scale_start_fraction = 0.8;
    double min_velocity_penalty = 10.0;
    double min_velocity_threshold = 0.5;
};

struct MPCConstraintSettings {
    double safety_margin = 0.1;
    double road_width = 7.0;                ///< Active when contouring is on

    /// Hard box on v_{k+1} in [ego.dynamics.min_velocity, ego.dynamics.max_velocity].
    bool enable_velocity_bounds = true;

    // Safe-horizon knobs — active only for SH_MPC / SH_MPCC
    int safe_horizon_min = 12;
    SafeHorizonMode safe_horizon_mode = SafeHorizonMode::PRACTICAL;
    int forced_safe_horizon = -1;
};

// ---------------------------------------------------------------------------
// Ego dynamics model
// ---------------------------------------------------------------------------
//
// SLMPC-style pluggable dynamics: the ego's motion model is SELECTED here, and
// the kinematic limits (velocity / acceleration / steering) that constrain that
// motion live WITH the model — they are properties of the dynamics, not of the
// vehicle geometry. Only the second-order unicycle is implemented today; the
// enum is the extension point for additional models (kinematic bicycle, etc.),
// whose realizations are deferred. dynamics.hpp's EgoDynamics integrator is the
// concrete realization of DynamicsModel::SECOND_ORDER_UNICYCLE.

enum class DynamicsModel {
    SECOND_ORDER_UNICYCLE  ///< State [x,y,theta,v], input [a, w]. (Others deferred.)
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

    double max_velocity = 4.0;       ///< Hard upper velocity bound [m/s]
    double min_velocity = 0.0;       ///< Hard lower velocity bound [m/s]
    double max_acceleration = 3.0;   ///< Maximum acceleration [m/s^2]
    double min_acceleration = -5.0;  ///< Minimum acceleration (braking) [m/s^2]
    double max_steering_rate = 0.8;  ///< Maximum steering rate [rad/s]
};

/**
 * @brief Ego vehicle geometry + dynamics model.
 *
 * Geometry (collision radius, body length, disc count) is separate from motion:
 * the dynamics model and its kinematic limits live in `dynamics`.
 */
struct EgoVehicleSpecification {
    double radius = 0.5;             ///< Collision radius [m]
    double length = 4.0;             ///< Length for multi-disc placement [m]
    int num_discs = 3;               ///< Number of discs along the body

    EgoDynamicsConfig dynamics;      ///< Motion model + kinematic limits
};

// ============================================================================
// Nominal mode belief
// ============================================================================

enum class NominalBeliefKind {
    DIRICHLET,  ///< Symmetric Dirichlet prior
    STICKY      ///< Dirichlet + self-persistence (sticky) prior
};

/// Build a ModeBeliefConfig (types.hpp) from the belief kind. The sticky
/// self-persistence prior theta is only wired in for STICKY.
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
    int num_scenarios = 20;
    double confidence_level = 0.95;
    double beta = 0.01;

    WeightType weight_type = WeightType::FREQUENCY;
    double recency_decay = 0.9;
    bool enforce_all_scenarios = false;
    bool enforce_scenario_count = false;
    bool ensure_mode_coverage = false;
    int max_history_length = -1;

    /// Hold-mode vs Markov jump prediction over the horizon.
    bool use_markov_mode_sampling = false;

    NominalBeliefKind belief_kind = NominalBeliefKind::DIRICHLET;
    ModeBeliefConfig mode_belief{};

    void sync_belief() {
        const double sticky =
            mode_belief.self_persistence_prior > 0.0
                ? mode_belief.self_persistence_prior : 0.8;
        mode_belief = make_mode_belief(belief_kind, sticky);
    }

    double epsilon() const { return 1.0 - confidence_level; }
};

// ============================================================================
// MPC config
// ============================================================================

struct MPCConfig {
    MPCConfiguration type = MPCConfiguration::SH_MPCC;
    int horizon = 20;
    double dt = 0.1;

    EgoVehicleSpecification ego;
    MPCObjectiveWeights objective;
    MPCConstraintSettings constraints;
    ScenarioSamplingSettings sampling;

    bool safe_horizon_enabled = true;
    bool enable_contouring_constraints = true;  ///< Road-boundary + contouring cost

    /// Derive safe-horizon / contouring enablement from the MPC type.
    /// Call after setting `type` and before overriding those two flags by hand.
    void sync_from_type() {
        switch (type) {
            case MPCConfiguration::MPC:
                safe_horizon_enabled = false;
                enable_contouring_constraints = false;
                break;
            case MPCConfiguration::MPCC:
                safe_horizon_enabled = false;
                enable_contouring_constraints = true;
                break;
            case MPCConfiguration::SH_MPC:
                safe_horizon_enabled = true;
                enable_contouring_constraints = false;
                break;
            case MPCConfiguration::SH_MPCC:
                safe_horizon_enabled = true;
                enable_contouring_constraints = true;
                break;
        }
    }

    bool uses_safe_horizon() const { return safe_horizon_enabled; }
};

// ============================================================================
// DRO — risk / ground cost / injection vocabulary
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
        case DRORiskMeasure::JOINT_VAR:                return "joint_var";
        case DRORiskMeasure::JOINT_CVAR:               return "joint_cvar";
        default: return "unknown";
    }
}

/// How the reweighted (worst-case) distribution q* is used by the controller.
enum class ReweightedDistributionUse {
    SAMPLING_ONLY,          ///< Resample all S scenarios from q*
    INJECTION_ONLY,         ///< Sample nominally, inject argmax(q*) as an extra constraint
    SAMPLING_AND_INJECTION  ///< Both
};

inline std::string reweighted_distribution_use_name(ReweightedDistributionUse u) {
    switch (u) {
        case ReweightedDistributionUse::SAMPLING_ONLY:          return "sampling_only";
        case ReweightedDistributionUse::INJECTION_ONLY:         return "injection_only";
        case ReweightedDistributionUse::SAMPLING_AND_INJECTION: return "sampling_and_injection";
        default: return "unknown";
    }
}

/// Concrete scenario-injection strategy the controller executes each step.
enum class InjectionMode {
    NONE,               ///< No DRO injection (base scenario MPC)
    DRO,                ///< Sample nominally, inject argmax(q*) as an extra constraint
    QSTAR_SAMPLE,       ///< Resample ALL S scenarios from the q* distribution
    ADVERSARIAL,        ///< Geometrically-motivated adversarial injection (tail trajectory)
    RANDOM,             ///< Inject one random mode per step
    ALL_MODES,          ///< Inject all modes deterministically
    // CDC baselines
    UNIFORM_COVERAGE,   ///< Force each observed mode to appear at least once
    SOFTMAX_RISK,       ///< Sample modes via p(m) ∝ exp(tau * r_m)
    EPSILON_GREEDY_INJ, ///< eps-greedy: (1-eps)*nominal + eps*uniform
    TOP_RISK_INJECT,    ///< Inject top-K modes by r_m deterministically (no WDRO)
    DIVERSE_RISK_INJECT ///< Inject K modes by risk*diversity (greedy facility-location)
};

// ============================================================================
// DRO — ambiguity-radius calibration
// ============================================================================

/**
 * @brief Calibration knobs for the Wasserstein ambiguity radius rho and the OT
 *        reweighting.
 *
 * Radius theory (true W1 concentration — see WassersteinDRO::get_adaptive_rho):
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
    bool use_calibrated_radius = true;   ///< Use the true-W1 concentration radius above
    double confidence_beta = 0.05;       ///< Target miscoverage (1 - beta coverage)
    double alpha_one_sided = 0.95;       ///< Risk level alpha (VaR/CVaR / surrogate z_alpha)

    /// Dimensionless safety factor multiplying the (1/2)*diam(D)*eps radius.
    /// 1.0 = the bare concentration bound; >1 inflates it.
    double calibration_scale = 1.0;

    /// Exact W1 primal OT reweighting instead of dual-guided heuristic recovery.
    bool use_primal_ot = true;

    DRORiskMeasure risk_measure = DRORiskMeasure::SURROGATE_VAR_BONFERRONI;

    /// Monte Carlo sample count / seed for JOINT_VAR / JOINT_CVAR (offline).
    int joint_risk_samples = 8000;
    uint64_t joint_risk_seed = 0x5150C0FFEEULL;

    double sigma_floor = 1e-6;           ///< Floor for directional sigma

    /// Entropic allocator: keeps q_min > 0 so the certificate L = 1/q_min is finite.
    bool use_entropic_allocator = false;
    double entropic_tau = 0.05;          ///< Temperature; tau -> 0 recovers the raw LP
};

/**
 * @brief Wasserstein-DRO solver knobs (radius, ground cost, calibration, OT).
 *
 * Consumed directly by WassersteinDRO. The flat radius fields set the clamp
 * band and the non-calibrated fallbacks; the nested radius_calibration holds
 * the confidence-calibrated radius parameters and the OT / risk selection.
 */
struct DROConfig {
    double base_radius = 0.1;        ///< Base radius rho (non-calibrated / fixed use)
    double min_radius = 0.01;        ///< Minimum rho (clamp floor; calibrated radius offset)
    double max_radius = 0.10;        ///< Maximum rho (clamp ceiling; below mode-transport collapse)
    bool adaptive_radius = true;     ///< Legacy heuristic rho scaling (when not calibrated)

    RadiusCalibrationSettings radius_calibration;

    double confidence_alpha = 1.0;   ///< Legacy: scaling for the 1/sqrt(n_obs) term
    double entropy_gamma = 0.5;      ///< Legacy: scaling for the entropy term

    DROGroundCostType ground_cost_type = DROGroundCostType::W2_BURES;
};

/**
 * @brief Controller-facing DRO settings.
 *
 * Solver knobs live in nested `solver` (DROConfig). Enablement, how q* is used,
 * and the concrete scenario-injection strategy sit beside it.
 */
struct DROControllerConfig {
    bool enabled = false;

    /// How the reweighted distribution is used. Drives injection_mode when the
    /// latter is left at NONE (see resolved_injection_mode()).
    ReweightedDistributionUse reweighting = ReweightedDistributionUse::SAMPLING_ONLY;

    /// Explicit injection strategy. NONE + enabled => derived from `reweighting`.
    InjectionMode injection_mode = InjectionMode::NONE;
    int injection_count = 1;
    double softmax_tau = 5.0;
    double eps_greedy_epsilon = 0.3;
    double adversarial_sigma_scale = 1.5;

    /// >0 forces a constant radius (disables calibrated / adaptive).
    double fixed_rho = -1.0;

    DROConfig solver;

    /// The injection strategy actually executed: the explicit one if set,
    /// otherwise derived from how the reweighted distribution is used.
    InjectionMode resolved_injection_mode() const {
        if (injection_mode != InjectionMode::NONE) return injection_mode;
        if (!enabled) return InjectionMode::NONE;
        switch (reweighting) {
            case ReweightedDistributionUse::SAMPLING_ONLY:
            case ReweightedDistributionUse::SAMPLING_AND_INJECTION:
                return InjectionMode::QSTAR_SAMPLE;
            case ReweightedDistributionUse::INJECTION_ONLY:
                return InjectionMode::DRO;
        }
        return InjectionMode::NONE;
    }

    void apply_fixed_rho() {
        if (fixed_rho <= 0.0) return;
        solver.radius_calibration.use_calibrated_radius = false;
        solver.adaptive_radius = false;
        solver.base_radius = fixed_rho;
        solver.min_radius = std::min(solver.min_radius, fixed_rho);
        solver.max_radius = std::max(solver.max_radius, fixed_rho);
    }
};

// ============================================================================
// Solver
// ============================================================================

struct SolverSettings {
    bool use_sqp_solver = true;        ///< SQP outer loop over QP subproblems
    int sqp_max_iterations = 5;        ///< Maximum SQP outer iterations
    double sqp_convergence_tol = 1e-3; ///< Convergence tolerance on ||delta_u||
    int qp_max_iterations = 200;       ///< Maximum ADMM iterations per QP
    double qp_tolerance = 1e-4;        ///< ADMM absolute tolerance
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

    /// Obstacle collision radius used with ego.radius for halfspaces.
    double obstacle_radius = 0.35;

    double combined_radius() const {
        return mpc.ego.radius + obstacle_radius;
    }

    double epsilon() const { return mpc.sampling.epsilon(); }

    bool enable_dro() const { return dro.enabled; }

    int compute_required_scenarios(int num_constraints, int num_removal = 0) const {
        return static_cast<int>(std::ceil(
            2.0 / epsilon() * (std::log(1.0 / mpc.sampling.beta) + num_constraints + num_removal)
        ));
    }

    int compute_required_scenarios_tight(int nbar) const {
        double eps = epsilon();
        return static_cast<int>(std::ceil(
            (2.0 / eps) * std::log(1.0 / mpc.sampling.beta)
            + 2.0 * nbar
            + (2.0 * nbar / eps) * std::log(2.0 / eps)
        ));
    }

    int compute_required_scenarios_simple(int d) const {
        double eps = epsilon();
        return static_cast<int>(std::ceil(
            (2.0 / eps) * (std::log(1.0 / mpc.sampling.beta) + d)
        ));
    }

    int compute_safe_horizon(int S_actual, int n_u = 2) const {
        if (!mpc.uses_safe_horizon()) return mpc.horizon;

        const auto& c = mpc.constraints;
        if (c.forced_safe_horizon >= 0) {
            return std::clamp(c.forced_safe_horizon, c.safe_horizon_min, mpc.horizon);
        }

        int N_safe = c.safe_horizon_min;
        switch (c.safe_horizon_mode) {
            case SafeHorizonMode::PRACTICAL:
                N_safe = std::min(mpc.horizon, S_actual / (2 * n_u));
                break;
            case SafeHorizonMode::THEORETICAL_SIMPLE:
                for (int N_try = mpc.horizon; N_try >= c.safe_horizon_min; --N_try) {
                    if (S_actual >= compute_required_scenarios_simple(N_try * n_u)) {
                        N_safe = N_try;
                        break;
                    }
                }
                break;
            case SafeHorizonMode::THEORETICAL_TIGHT:
                for (int N_try = mpc.horizon; N_try >= c.safe_horizon_min; --N_try) {
                    if (S_actual >= compute_required_scenarios_tight(N_try * n_u)) {
                        N_safe = N_try;
                        break;
                    }
                }
                break;
            case SafeHorizonMode::HEURISTIC:
                N_safe = c.safe_horizon_min;
                break;
        }
        return std::clamp(N_safe, c.safe_horizon_min, mpc.horizon);
    }

    double compute_effective_epsilon(int S_actual, int d, int R = 0) const {
        if (S_actual <= 0) return 1.0;
        return 2.0 * (std::log(1.0 / mpc.sampling.beta) + d + R) / S_actual;
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
        if (mpc.sampling.confidence_level <= 0 || mpc.sampling.confidence_level >= 1)
            throw std::invalid_argument("confidence_level must be in (0, 1)");
        if (mpc.sampling.beta <= 0 || mpc.sampling.beta >= 1)
            throw std::invalid_argument("beta must be in (0, 1)");
        if (mpc.sampling.num_scenarios <= 0)
            throw std::invalid_argument("num_scenarios must be positive");
    }
};

}  // namespace dro_mpc

#endif  // DRO_MPC_CONFIG_HPP
