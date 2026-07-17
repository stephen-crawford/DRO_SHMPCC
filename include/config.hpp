/**
 * @file config.hpp
 * @brief Configuration for Adaptive Scenario-Based MPC.
 *
 * Provides a clean interface for all hyperparameters.
 */

#ifndef SCENARIO_MPC_CONFIG_HPP
#define SCENARIO_MPC_CONFIG_HPP

#include "types.hpp"
#include <cmath>
#include <stdexcept>

namespace scenario_mpc {

/**
 * @brief Source of nominal distribution for DRO Q* computation.
 */
enum class DRONominalSource {
    FREQUENCY,  ///< Use frequency weights for DRO Q*
};

/**
 * @brief Injection mode for experiment ablation variants.
 */
enum class InjectionMode {
    NONE,               ///< No DRO injection (base scenario MPC)
    DRO,                ///< DRO worst-case injection: sample nominally, inject argmax(q*) as extra constraint
    QSTAR_SAMPLE,       ///< DRO q* sampling: resample ALL S scenarios from q* distribution
    ADVERSARIAL,        ///< Geometrically-motivated adversarial injection (tail trajectory)
    RANDOM,             ///< Inject one random mode per step
    ALL_MODES,          ///< Inject all modes deterministically
    // CDC baselines
    UNIFORM_COVERAGE,   ///< Force each observed mode to appear at least once
    SOFTMAX_RISK,       ///< Sample modes via p(m) proportional to exp(tau * r_m)
    EPSILON_GREEDY_INJ, ///< eps-greedy: (1-eps)*nominal + eps*uniform
    TOP_RISK_INJECT,    ///< Inject top-K modes by r_m deterministically (no WDRO)
    DIVERSE_RISK_INJECT ///< Inject K modes by risk*diversity (greedy facility-location style)
};

/**
 * @brief Safe horizon computation mode.
 */
enum class SafeHorizonMode {
    THEORETICAL_TIGHT,   ///< Tight bound (Eq. 25): very conservative
    THEORETICAL_SIMPLE,  ///< Simple bound (Eq. 23): S >= (2/eps)*(ln(1/beta) + d)
    PRACTICAL            ///< Practical: N_safe = min(N, floor(S / (2*n_u)))
};

/**
 * @brief Configuration parameters for Adaptive Scenario-Based MPC.
 */
struct ScenarioMPCConfig {
    // Horizon and timing
    int horizon = 20;           ///< Prediction horizon N
    double dt = 0.1;            ///< Timestep duration [s]

    // Ego vehicle parameters
    double ego_radius = 1.0;          ///< Collision radius for ego vehicle [m]
    double max_acceleration = 3.0;    ///< Maximum acceleration [m/s^2]
    double min_acceleration = -5.0;   ///< Minimum acceleration (braking) [m/s^2]
    double max_steering_rate = 0.8;   ///< Maximum steering rate [rad/s]
    double max_velocity = 4.0;        ///< Hard upper bound on speed [m/s] (~2.7x the 1.5 m/s cruise)
    double min_velocity = 0.0;        ///< Hard lower bound on speed [m/s] (no reverse)
    bool enable_velocity_bounds = true;  ///< Enforce v in [min_velocity, max_velocity] as QP rows

    // Obstacle parameters
    double obstacle_radius = 0.5;     ///< Default obstacle collision radius [m]

    // Scenario sampling (Theorem 1)
    int num_scenarios = 10;           ///< Number of scenarios to sample (S)
    double confidence_level = 0.95;   ///< Chance constraint confidence (1 - epsilon)
    double beta = 0.01;               ///< Risk parameter for sample size computation

    // Mode weights (Section 4)
    WeightType weight_type = WeightType::FREQUENCY;  ///< Strategy for mode weights
    double recency_decay = 0.9;       ///< Decay factor for recency weighting (lambda)

    // Epsilon guarantee enforcement (Part 4 + Part 1)
    bool enforce_all_scenarios = false;   ///< Use ALL scenarios as constraints (Theorem 1)
    bool enforce_scenario_count = false;  ///< Auto-increase scenarios if S < S_required
    bool ensure_mode_coverage = false;    ///< Guarantee ≥1 scenario per observed mode per obstacle

    // DRO (Distributionally Robust Optimization) parameters
    /// Sample each scenario's mode SEQUENCE from the estimated Markov transition
    /// matrix instead of drawing one mode i.i.d. and holding it for the horizon.
    /// The obstacle simulator switches modes mid-rollout (switch_prob > 0), so an
    /// i.i.d. draw is a mis-specified predictor of it. Off by default: the CDC'26
    /// results were produced with i.i.d. mode draws and stay reproducible.
    bool use_markov_mode_sampling = false;
    /// Prior hyperparameters for the Bayesian mode belief (see ModeBeliefConfig).
    /// Forwarded into the samplers; exposed so alpha / sticky_bonus are sweepable
    /// rather than hardcoded default arguments.
    ModeBeliefConfig mode_belief;

    bool enable_dro = false;                 ///< Enable Wasserstein DRO weight reweighting
    InjectionMode injection_mode = InjectionMode::QSTAR_SAMPLE;  ///< Injection strategy when enable_dro=true
    double dro_rho_base = 0.1;               ///< Base Wasserstein ball radius rho
    double dro_rho_min = 0.01;              ///< Minimum rho (clamped)
    double dro_rho_max = 0.5;              ///< Maximum rho (clamped)
    bool dro_adaptive_rho = true;           ///< Enable adaptive rho scaling
    // --- fields below are forwarded verbatim into the controller's DROConfig ---
    // They exist because ScenarioMPCConfig is the ONLY channel from the experiment
    // harness to AdaptiveScenarioMPC's WassersteinDRO. Anything set on a DROConfig
    // inside the harness is discarded; see the note in mpc_controller.cpp.
    /// Use the confidence-calibrated simplex-concentration radius rho_n(beta)
    /// instead of the ad-hoc rho_0*(1+alpha/sqrt(n)) heuristic. The heuristic is
    /// statistically INCONSISTENT: it plateaus at ~rho_0, so Q* stays distorted
    /// away from nominal even as n -> inf (measured Qstar_L1 pinned at 1.333 at
    /// n=10^4, vs 0.127 and shrinking for the calibrated radius). The calibrated
    /// form certifies p* in B_eps(p_hat) w.p. >= 1-beta and is link 1 of the
    /// theory chain -- it MUST be reachable from an experiment.
    bool dro_use_calibrated_radius = true;   ///< DEFAULT ON: proper confidence-calibrated radius (theory chain link 1)
    double dro_confidence_beta = 0.05;  ///< Target miscoverage beta for the calibrated radius
    /// Use the TRUE Wasserstein-metric primal OT reweighting (exact W1 LP, fractional
    /// source splits) instead of the dual-guided heuristic recovery. DEFAULT ON.
    bool dro_use_primal_ot = true;
    /// Entropic allocator: makes min_m q_m > 0, hence the sampling certificate
    /// L <= 1/q_min FINITE. The raw W1-LP gives L = inf whenever the transport
    /// budget is slack (Theorem 2(i)), and ~86% of the time when it binds.
    /// Off by default (CDC'26 used the raw LP).
    bool dro_use_entropic_allocator = false;
    double dro_entropic_tau = 0.05;   ///< Temperature; tau -> 0 recovers the raw LP
    DROGroundCostType dro_ground_cost = DROGroundCostType::W2_BURES;  ///< Ground cost for the transport matrix D
    DRORiskMode dro_risk_mode = DRORiskMode::FULL;                    ///< Covariance handling in the risk score
    DRORiskMeasure dro_risk_measure = DRORiskMeasure::SURROGATE_VAR_BONFERRONI;  ///< DEFAULT: Bonferroni joint-horizon VaR bound (r[m])
    double dro_alpha_one_sided = 0.95;      ///< Risk level alpha
    int dro_joint_risk_samples = 8000;      ///< MC samples when dro_risk_measure is JOINT_*
    double adversarial_sigma_scale = 1.5;    ///< Scale for adversarial injection (sigma multiplier)
    DRONominalSource dro_nominal_source = DRONominalSource::FREQUENCY;  ///< P_hat source for DRO Q*
    int dro_injection_count = 1;             ///< Number of top-K modes to inject per obstacle (0 = none, -1 = all modes)
    double softmax_tau = 5.0;                ///< Temperature for SOFTMAX_RISK baseline
    double eps_greedy_epsilon = 0.3;         ///< Epsilon for EPSILON_GREEDY_INJ baseline
    int max_history_length = -1;             ///< Max observation history length (-1 = default horizon*10)

    // Multi-disc collision model (Section 7)
    int num_discs = 3;                    ///< Number of discs for ego vehicle (D=3 default)
    double vehicle_length = 4.0;          ///< Vehicle length for disc placement [m]

    // Safe horizon truncation (SH-MPC)
    bool safe_horizon_enabled = true;     ///< Enable safe horizon truncation
    int safe_horizon_min = 12;            ///< Minimum truncated horizon steps
    SafeHorizonMode safe_horizon_mode = SafeHorizonMode::PRACTICAL;  ///< SH computation mode
    int forced_safe_horizon = -1;         ///< Force N_safe to this value (-1 = auto)

    // Contouring constraints (road boundary halfplanes)
    bool enable_contouring_constraints = false;  ///< Enable road boundary constraints
    double road_width = 7.0;          ///< Road width [m] (symmetric about reference path)

    // Constraint parameters
    double safety_margin = 0.1;       ///< Additional safety margin [m]

    // Solver parameters
    int solver_max_iter = 500;        ///< Maximum solver iterations
    double solver_tolerance = 1e-4;   ///< Convergence tolerance

    // SQP solver parameters
    bool use_sqp_solver = true;           ///< Use SQP solver instead of heuristic
    int sqp_max_iterations = 5;           ///< Maximum SQP outer iterations
    double sqp_convergence_tol = 1e-3;    ///< SQP convergence tolerance on ||delta_u||
    int qp_max_iterations = 200;          ///< Maximum ADMM iterations per QP
    double qp_tolerance = 1e-4;           ///< ADMM absolute tolerance

    // Objective weights
    double goal_weight = 10.0;        ///< Weight for goal tracking
    double velocity_weight = 1.0;     ///< Weight for velocity tracking
    double acceleration_weight = 0.1; ///< Weight for acceleration penalty
    double steering_weight = 0.1;     ///< Weight for steering penalty

    // MPCC cost weights (Paper Eq. 6)
    double contour_weight = 1.0;          ///< w_c: lateral (contouring) error penalty
    double lag_weight = 0.1;              ///< w_l: progress lag penalty
    double terminal_heading_weight = 1.0; ///< Terminal heading alignment weight

    // Progress-aware cost parameters
    double goal_weight_scale_max = 6.0;       ///< Max goal weight multiplier near end
    double goal_scale_start_fraction = 0.8;   ///< Start scaling goal weight at this progress fraction
    double min_velocity_penalty = 10.0;       ///< Penalty for velocity below threshold
    double min_velocity_threshold = 0.5;      ///< Minimum desired velocity [m/s]

    /// Violation probability (1 - confidence_level)
    double epsilon() const {
        return 1.0 - confidence_level;
    }

    /// Combined ego + obstacle radius for collision checking
    double combined_radius() const {
        return ego_radius + obstacle_radius;
    }

    /**
     * @brief Compute required number of scenarios using Theorem 1.
     *
     * S >= 2/epsilon * (ln(1/beta) + d + R)
     * where d = N * n_x + N * n_u
     *
     * @param num_constraints Number of decision variables (d)
     * @param num_removal Number of removed scenarios (R)
     * @return Minimum number of scenarios required
     */
    int compute_required_scenarios(int num_constraints, int num_removal = 0) const {
        return static_cast<int>(std::ceil(
            2.0 / epsilon() * (std::log(1.0 / beta) + num_constraints + num_removal)
        ));
    }

    /**
     * @brief Compute required scenarios using tighter bound (Eq. 25).
     *
     * S >= (2/eps)*ln(1/beta) + 2*nbar + (2*nbar/eps)*ln(2/eps)
     *
     * where nbar = support_rank_nbar (effective decision variable dimension,
     * typically N * n_u for the condensed formulation).
     *
     * @param nbar Support rank (effective dimension)
     * @return Minimum number of scenarios required
     */
    int compute_required_scenarios_tight(int nbar) const {
        double eps = epsilon();
        return static_cast<int>(std::ceil(
            (2.0 / eps) * std::log(1.0 / beta)
            + 2.0 * nbar
            + (2.0 * nbar / eps) * std::log(2.0 / eps)
        ));
    }

    /**
     * @brief Compute required scenarios using simple bound (Eq. 23).
     *
     * S >= (2/eps) * (ln(1/beta) + d)
     *
     * @param d Decision variable dimension (N_safe * n_u for condensed)
     * @return Minimum number of scenarios required
     */
    int compute_required_scenarios_simple(int d) const {
        double eps = epsilon();
        return static_cast<int>(std::ceil(
            (2.0 / eps) * (std::log(1.0 / beta) + d)
        ));
    }

    /**
     * @brief Compute safe horizon N_safe based on configured mode.
     *
     * @param S_actual Number of scenarios available
     * @param n_u Number of control inputs per timestep (typically 2)
     * @return Safe horizon N_safe in [safe_horizon_min, horizon]
     */
    int compute_safe_horizon(int S_actual, int n_u = 2) const {
        if (!safe_horizon_enabled) return horizon;

        // If forced, use that value (clamped to valid range)
        if (forced_safe_horizon >= 0) {
            return std::clamp(forced_safe_horizon, safe_horizon_min, horizon);
        }

        int N_safe = safe_horizon_min;

        switch (safe_horizon_mode) {
            case SafeHorizonMode::PRACTICAL:
                // N_safe = min(N, floor(S / (2*n_u)))
                N_safe = std::min(horizon, S_actual / (2 * n_u));
                break;

            case SafeHorizonMode::THEORETICAL_SIMPLE:
                // Find largest N_safe s.t. S >= (2/eps)*(ln(1/beta) + N_safe*n_u)
                for (int N_try = horizon; N_try >= safe_horizon_min; --N_try) {
                    int d = N_try * n_u;
                    int S_req = compute_required_scenarios_simple(d);
                    if (S_actual >= S_req) {
                        N_safe = N_try;
                        break;
                    }
                }
                break;

            case SafeHorizonMode::THEORETICAL_TIGHT:
                // Find largest N_safe s.t. S >= tight bound(N_safe*n_u)
                for (int N_try = horizon; N_try >= safe_horizon_min; --N_try) {
                    int nbar = N_try * n_u;
                    int S_req = compute_required_scenarios_tight(nbar);
                    if (S_actual >= S_req) {
                        N_safe = N_try;
                        break;
                    }
                }
                break;
        }

        return std::clamp(N_safe, safe_horizon_min, horizon);
    }

    /**
     * @brief Compute effective epsilon given actual scenario count.
     *
     * Inverse of Theorem 1: eps = 2*(ln(1/beta) + d + R) / S
     *
     * @param S_actual Actual number of scenarios used
     * @param d Decision variable dimension
     * @param R Number of removed scenarios
     * @return Effective violation probability bound
     */
    double compute_effective_epsilon(int S_actual, int d, int R = 0) const {
        if (S_actual <= 0) return 1.0;
        return 2.0 * (std::log(1.0 / beta) + d + R) / S_actual;
    }

    /// Validate configuration parameters
    void validate() const {
        if (horizon <= 0) {
            throw std::invalid_argument("horizon must be positive");
        }
        if (dt <= 0) {
            throw std::invalid_argument("dt must be positive");
        }
        if (confidence_level <= 0 || confidence_level >= 1) {
            throw std::invalid_argument("confidence_level must be in (0, 1)");
        }
        if (beta <= 0 || beta >= 1) {
            throw std::invalid_argument("beta must be in (0, 1)");
        }
        if (num_scenarios <= 0) {
            throw std::invalid_argument("num_scenarios must be positive");
        }
    }
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_CONFIG_HPP
