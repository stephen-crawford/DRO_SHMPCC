/**
 * @file wasserstein_dro.hpp
 * @brief Wasserstein Distributionally Robust Optimization for scenario MPC.
 *
 * Replaces the ExtremeBuffer approach with a principled DRO reweighting:
 * at each MPC step, compute worst-case mode weights Q* within a W1 ball
 * of radius rho around the nominal distribution P_hat, then either
 * sample scenarios from Q* or inject a worst-case scenario identified by Q*.
 *
 * Algorithm (Kantorovich dual for W1):
 *   sup_{Q in B_rho(P_hat)} sum_m q_m * r_m
 *     = inf_{lambda>=0} { lambda*rho + sum_i w_i * max_j(r_j - lambda*D[i][j]) }
 *
 * Solved via binary search on lambda. Q* recovered by transporting each
 * source mode i to j*(i) = argmax_j { r_j - lambda* * D[i][j] }.
 */

#ifndef SCENARIO_MPC_WASSERSTEIN_DRO_HPP
#define SCENARIO_MPC_WASSERSTEIN_DRO_HPP

#include "types.hpp"
#include <vector>
#include <map>
#include <string>
#include <optional>
#include <random>
#include <limits>

namespace scenario_mpc {

// NOTE: DRORiskMode / DROGroundCostType / DRORiskMeasure now live in types.hpp.
// They are shared configuration vocabulary: ScenarioMPCConfig (config.hpp) must
// name them to forward them into this module's DROConfig, and config.hpp cannot
// include wasserstein_dro.hpp without inverting the layering. Both already
// include types.hpp, so that is where they belong.

/**
 * @brief Configuration for DRO worst-case weight computation.
 */
struct DROConfig {
    double rho_base = 0.1;           ///< Base Wasserstein ball radius rho
    double rho_min = 0.01;           ///< Minimum rho (clamped)
    double rho_max = 0.5;            ///< Maximum rho (clamped)
    bool adaptive_rho = true;        ///< Enable adaptive rho scaling
    double confidence_alpha = 1.0;   ///< Scaling for 1/sqrt(n_obs) term
    double entropy_gamma = 0.5;      ///< Scaling for entropy term
    bool use_calibrated_radius = true;   ///< DEFAULT ON: confidence-calibrated simplex-concentration rho_n(beta)
    double confidence_beta = 0.05;   ///< Target miscoverage (1-beta coverage) for the calibrated radius
    double alpha_one_sided = 0.95;   ///< Risk level alpha (VaR/CVaR tail level, and z_alpha for the surrogate)
    /// Use the TRUE Wasserstein-metric primal OT reweighting (exact W1 LP) instead of the
    /// dual-guided heuristic recovery. DEFAULT ON (the reviewer-requested true metric).
    bool use_primal_ot = true;
    DRORiskMeasure risk_measure = DRORiskMeasure::SURROGATE_VAR_BONFERRONI;  ///< DEFAULT: Bonferroni joint-horizon VaR (see DRORiskMeasure)
    /// Monte Carlo sample count for JOINT_VAR / JOINT_CVAR.
    /// Measured on the canonical 6-mode scenario (error vs 2M-sample ground truth,
    /// and cost of one compute_risk_vector call over 6 modes x 15 steps):
    ///     500 -> err 0.029, 1.7 ms  |  2000 -> err 0.021, 6.8 ms
    ///    8000 -> err 0.008,  28 ms  | 32000 -> err 0.002, 111 ms
    /// 8000 is the default: it is the smallest count whose VaR coverage lands inside
    /// +/-0.5% of alpha. NOTE the cost -- at 28 ms this is ~19x the paper's quoted
    /// <1.5 ms/step budget, so JOINT_* is an OFFLINE analysis / calibration tool,
    /// not the online loop. SURROGATE_VAR remains the default risk_measure.
    int joint_risk_samples = 8000;
    uint64_t joint_risk_seed = 0x5150C0FFEEULL;  ///< Fixed RNG seed: deterministic across calls, common random numbers across modes
    double sigma_floor = 1e-6;       ///< Floor for directional sigma (numerical stability)
    /// Use the entropic (softmax) allocator instead of the raw W1-LP recovery.
    ///
    /// Theorem 2(i): when the transport budget is SLACK the LP optimum is exactly
    /// q* = e_{argmax r}, so min_m q*_m = 0 and the likelihood ratio L = inf --
    /// there is NO finite closed-loop certificate. When the budget binds, the LP
    /// support is bounded by #{distinct destinations}+1 and is deficient ~86% of
    /// the time (measured over 3000 random instances; 14.1% retain full support).
    ///
    /// The entropic row softmax is STRICTLY positive, so q_min > 0 and
    /// L <= 1/q_min < inf UNCONDITIONALLY. entropic_tau trades protection
    /// (<q_tau,r>, non-increasing in tau) against certificate tightness
    /// (1/q_min, decreasing in tau). Off by default: CDC'26 used the raw LP.
    bool use_entropic_allocator = false;
    /// Temperature tau > 0. tau -> 0 recovers the raw LP (and its infinite L).
    /// Measured on the canonical 6-mode scenario at rho=0.15: tau=0.05 costs 2.5%
    /// protection (0.7517 -> 0.7326) and takes L from inf to 166.8.
    double entropic_tau = 0.05;
    DRORiskMode risk_mode = DRORiskMode::FULL;                ///< Risk computation mode
    DROGroundCostType ground_cost_type = DROGroundCostType::W2_BURES;  ///< Ground cost type
};

/**
 * @brief Tie-breaking policy for deterministic transport plan construction.
 */
enum class TiePolicy {
    MIN_COST,   ///< Among tied destinations, pick the one with lowest D[i][j] (less movement)
    MAX_COST    ///< Among tied destinations, pick the one with highest D[i][j] (more movement)
};

/**
 * @brief A deterministic transport plan at a given lambda.
 */
struct TransportPlan {
    double lambda = 0.0;
    std::map<std::string, double> q;        ///< Resulting mode distribution
    double transport_cost = 0.0;            ///< sum_i p_i D[i, j*(i)]
    double expected_risk = 0.0;             ///< sum_j q_j r_j
};

/**
 * @brief Result of worst-case distribution recovery with transport cost verification.
 */
struct WorstCaseRecoveryResult {
    std::map<std::string, double> q_star;   ///< Recovered worst-case distribution
    double implied_transport_cost = 0.0;    ///< Actual transport cost of the plan
    bool feasible = false;                  ///< implied_transport_cost <= rho (+tol)
    double mix_alpha = 0.0;                 ///< Convex mixing coefficient (0=low-cost, 1=high-cost)
};

/**
 * @brief Result from DRO worst-case weight computation.
 */
struct DROResult {
    std::map<std::string, double> worst_case_weights;  ///< Q* mode weights
    double optimal_lambda = 0.0;                       ///< Optimal dual variable
    double rho_used = 0.0;                              ///< Wasserstein radius rho used
    double worst_case_risk = 0.0;                      ///< sup risk under Q*
    std::map<std::string, double> risk_per_mode;       ///< r[m] for each mode
    std::vector<std::vector<double>> transport_cost_matrix;  ///< D[i][j]
    double implied_transport_cost = 0.0;               ///< Transport cost of induced plan
    bool recovery_feasible = false;                    ///< Whether induced plan respects rho

    // --- Certificate diagnostics (paper Sec. IV-E) ---------------------------
    // Assumption 1 of Theorem 1 requires the SAMPLING distribution to have full
    // support, q_m > 0 for all m. Without these fields the assumption is not
    // measurable from a run, and a theorem whose hypothesis cannot be checked is
    // not a defensible claim. They exist so the certificate can be AUDITED, not
    // asserted.
    //
    // Theorem 2 (CORRECTED 2026-07-15 -- the earlier "extreme point => generically
    // bang-bang" argument was WRONG: a small-rho ball can lie strictly inside the
    // simplex with full-support vertices). The correct statement runs through the
    // LP's TRANSPORT structure:
    //   (i)  SLACK budget => q* = e_{argmax r} EXACTLY (verified 3000/3000), so
    //        qstar_support_floor == 0 and L = inf: Theorem 1 is VACUOUS.
    //   (ii) A basic optimal Pi has <= M+1 nonzeros; each row needs >= 1, so at
    //        most ONE row splits and supp(q*) <= #{distinct destinations}+1.
    // The deficiency is therefore CONDITIONAL, not universal: over 3000 random
    // BINDING-budget instances supp(q*) == M in 14.1% of cases. On the canonical
    // six-mode scenario it fails 9/9, and at rho=0.30 (slack) q* = (0,1,0,0,0,0).
    // Use the entropic allocator (DROConfig::use_entropic_allocator) to make
    // qstar_support_floor > 0 unconditionally.
    double qstar_support_floor = 0.0;   ///< min_m Q*[m]. 0 => Assumption 1 FAILS (bang-bang).
    int qstar_support_size = 0;         ///< #{m : Q*[m] > 0}. < M => Assumption 1 FAILS.
    bool satisfies_full_support = false;///< qstar_support_floor > 0, i.e. Assumption 1 holds.

    /// Certified bound on the likelihood ratio L = max_m p*_m / Q*[m] of Lemma 1,
    /// using only p*_m <= 1: L <= 1 / min_m Q*[m]. Returns +inf when the support
    /// floor is zero, which is the correct and honest answer -- Theorem 1 gives
    /// V_{p*} <= L * eps_S, so L = inf means no bound at all.
    double likelihood_ratio_bound() const {
        if (!(qstar_support_floor > 0.0)) {
            return std::numeric_limits<double>::infinity();
        }
        return 1.0 / qstar_support_floor;
    }
};

/**
 * @brief Result of the entropic (Sinkhorn-style) allocator.
 *
 * The raw W1-LP maximises a LINEAR functional, so a basic optimal transport plan
 * has at most M+1 nonzeros: every source row must send its mass somewhere, hence
 * at most ONE row splits and all others are deterministic. The target marginal is
 * then supported on the image {j*(i)} plus at most one point, and in particular:
 *
 *   * If the transport budget is SLACK at the optimum (lambda* = 0), then
 *     q* = e_{argmax_j r_j} exactly -- support 1, so min_m q*_m = 0 and the
 *     likelihood ratio L = max_m p*_m/q*_m is INFINITE. Verified 3000/3000.
 *   * If the budget BINDS, the support is usually but NOT always deficient:
 *     measured over 3000 random instances, supp(q*) = M in 14.1% of cases.
 *     So the impossibility is CONDITIONAL, not universal.
 *
 * Entropic regularisation removes the conditionality. Adding -tau * sum_ij
 * Pi_ij (log Pi_ij - 1) to the objective makes it strictly concave, and the
 * row-wise maximiser under the fixed source marginal is a softmax:
 *
 *     Pi_ij = p_hat_i * exp((r_j - lambda D_ij)/tau) / sum_k exp((r_k - lambda D_ik)/tau)
 *     q_j   = sum_i Pi_ij
 *
 * Every Pi_ij is STRICTLY positive, hence q_j > 0 for all j: full support holds
 * UNCONDITIONALLY, so L <= 1/q_min < inf always. That is the sense in which the
 * entropic allocator is not a refinement of the W1-LP but a precondition for the
 * sampling certificate to exist.
 *
 * tau traces a frontier: as tau -> 0 the softmax tends to the argmax and q_tau
 * tends to the bang-bang solution (protection maximal, certificate -> inf); as
 * tau -> inf the rows tend to uniform and q_tau -> uniform (protection minimal,
 * certificate -> M, its tightest possible value).
 */
struct EntropicOTResult {
    std::map<std::string, double> q;    ///< Target marginal q_tau (STRICTLY positive)
    double lambda = 0.0;                ///< Dual price on the transport budget
    double transport_cost = 0.0;        ///< sum_ij Pi_ij D_ij
    double expected_risk = 0.0;         ///< <q_tau, r>  ("protection")
    double q_min = 0.0;                 ///< min_m q_tau[m] > 0 by construction
    bool solved = false;                ///< lambda search converged and budget respected

    /// L <= 1/q_min. Finite for any tau > 0 -- this is the point.
    double likelihood_ratio_bound() const {
        if (!(q_min > 0.0)) return std::numeric_limits<double>::infinity();
        return 1.0 / q_min;
    }
};

/**
 * @brief Entropic-regularised W1 reweighting: full support for any tau > 0.
 *
 * Solves, for the fixed source marginal p_hat,
 *     max_Pi  sum_ij Pi_ij r_j  -  tau * sum_ij Pi_ij (log Pi_ij - 1)
 *     s.t.    sum_j Pi_ij = p_hat_i,   sum_ij Pi_ij D_ij <= rho,   Pi >= 0,
 * via the closed-form row softmax above, with lambda >= 0 found by bisection to
 * enforce the budget (lambda = 0 when the budget is slack at tau).
 *
 * @param tau Temperature > 0. tau -> 0 recovers the bang-bang LP behaviour.
 */
EntropicOTResult solve_entropic_ot(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_per_mode,
    const std::vector<std::vector<double>>& transport_cost_matrix,
    const std::vector<std::string>& mode_ids,
    double rho,
    double tau
);

/**
 * @brief Wasserstein DRO for scenario-based MPC.
 *
 * Computes worst-case mode weights Q* within a W1 ball around nominal P_hat.
 * This shifts probability mass toward the single most dangerous mode direction
 * rather than injecting all extreme modes simultaneously (which over-constrains).
 */
class WassersteinDRO {
public:
    explicit WassersteinDRO(const DROConfig& config = DROConfig());

    /**
     * @brief Compute worst-case mode weights via Kantorovich dual.
     *
     * @param nominal_weights  P_hat: nominal mode weights {mode_id -> w_m}
     * @param obs_state        Current obstacle state
     * @param mode_models      Available mode dynamics models
     * @param ego_linearization_traj Fixed numerical ego trajectory used to
     *        evaluate mode risks. This trajectory is not an MPC decision
     *        variable and is held fixed while computing the WDRO distribution.
     * @param horizon          Prediction horizon
     * @param ego_r            Ego collision radius
     * @param obs_r            Obstacle collision radius
     * @param margin           Safety margin
     * @return DROResult with Q* and diagnostics
     */
    DROResult compute_worst_case_weights(
        const std::map<std::string, double>& nominal_weights,
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<EgoState>& ego_linearization_traj,
        int horizon,
        double ego_r,
        double obs_r,
        double margin,
        int risk_horizon = -1,
        int num_discs = 1,
        double vehicle_length = 4.0
    );

    /**
     * @brief Generate a worst-case scenario for a single obstacle.
     *
     * Identifies the mode m* = argmax_m Q*[m] (highest CVaR mode) from
     * the DRO result, then forward-propagates that mode deterministically
     * from the current obstacle state over the horizon.  The resulting
     * scenario is meant to be **injected** as an additional hard constraint
     * so the QP insulates against the most damaging mode.
     *
     * @param dro_result  Result from compute_worst_case_weights()
     * @param obstacle_id Obstacle identifier
     * @param obs_state   Current obstacle state
     * @param mode_models Available mode dynamics
     * @param horizon     Prediction horizon
     * @param scenario_id ID to assign to the generated scenario
     * @return Scenario containing a single deterministic obstacle trajectory
     *         for the worst-case mode, or empty trajectories if no risk.
     */
    Scenario generate_worst_case_scenario(
        const DROResult& dro_result,
        int obstacle_id,
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        int horizon,
        int scenario_id
    );

    /**
     * @brief Generate an adversarial scenario for a single obstacle.
     *
     * For each mode, computes the approach direction from obstacle toward ego,
     * projects the mode covariance onto that direction, and pushes the obstacle
     * trajectory toward the ego along the most uncertain axis.
     *
     * adversarial_pos[k] = mean[k] + sigma_scale * sigma_along * approach_dir
     *
     * This creates geometrically-motivated dangerous tail trajectories.
     *
     * @param dro_result  Result from compute_worst_case_weights()
     * @param obstacle_id Obstacle identifier
     * @param obs_state   Current obstacle state
     * @param mode_models Available mode dynamics
     * @param ego_ref     Ego reference trajectory (for approach direction)
     * @param horizon     Prediction horizon
     * @param scenario_id ID to assign to the generated scenario
     * @param sigma_scale How many sigma to push toward ego (default 1.5)
     * @return Scenario with adversarial obstacle trajectory
     */
    Scenario generate_adversarial_scenario(
        const DROResult& dro_result,
        int obstacle_id,
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<EgoState>& ego_ref,
        int horizon,
        int scenario_id,
        double sigma_scale = 1.5
    );

    /**
     * @brief Generate scenarios for top-K modes ranked by Q* weight.
     *
     * For DRO injection: each mode's mean trajectory is deterministically propagated.
     * For adversarial injection: each mode's trajectory is pushed toward ego.
     *
     * Theoretical motivation: the effective support of Q* under the Kantorovich
     * dual is typically small (1-3 modes). Injecting beyond the effective support
     * adds constraints with negligible Q* weight, potentially over-constraining.
     *
     * @param K  Number of top modes to inject (-1 = all modes with Q* > 0)
     * @return Vector of scenarios, one per injected mode
     */
    std::vector<Scenario> generate_topk_worst_case_scenarios(
        const DROResult& dro_result,
        int obstacle_id,
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        int horizon,
        int base_scenario_id,
        int K
    );

    std::vector<Scenario> generate_topk_adversarial_scenarios(
        const DROResult& dro_result,
        int obstacle_id,
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<EgoState>& ego_ref,
        int horizon,
        int base_scenario_id,
        int K,
        double sigma_scale = 1.5
    );

    /// Compute effective support size of Q* (number of modes with weight > threshold)
    static int effective_support(const DROResult& dro_result, double threshold = 0.01);

    /**
     * @brief Sample S scenarios by drawing modes according to q*.
     *
     * For each scenario s = 0..S-1, draw mode m ~ q* and forward-propagate
     * the mode-conditioned Gaussian trajectory (mean + covariance) from the
     * current obstacle state.  The scenario probability is set to q*[m].
     *
     * This is "sampling from q*" as described in the paper, as opposed to
     * the adversarial injection functions which deterministically pick
     * argmax_m q*[m].
     *
     * @param dro_result     Result from compute_worst_case_weights()
     * @param obstacle_id    Obstacle identifier
     * @param obs_state      Current obstacle state
     * @param mode_models    Available mode dynamics
     * @param horizon        Prediction horizon
     * @param num_scenarios  Number of scenarios S to sample
     * @param rng            Random number generator (caller-owned)
     * @param base_scenario_id  Starting scenario ID
     * @return Vector of S scenarios, each with mode drawn ~ q*
     */
    std::vector<Scenario> sample_scenarios_from_qstar(
        const DROResult& dro_result,
        int obstacle_id,
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        int horizon,
        int num_scenarios,
        std::mt19937& rng,
        int base_scenario_id = 0
    );

    /// Get adaptive Wasserstein radius rho based on observation count and entropy
    double get_adaptive_rho() const;

    /// Update prediction error tracking (for potential future adaptive rho refinement)
    void update_prediction_error(double error);

    /// Set the number of observations (for adaptive rho scaling)
    void set_observation_count(int n);

    /// Optional per-step rho override (e.g. from adaptive shift detection).
    void set_rho_override(double rho);
    void clear_rho_override();

    /// Get config (const)
    const DROConfig& config() const { return config_; }

private:
    /**
     * @brief Compute inter-mode transport cost matrix D[i][j].
     *
     * Uses Gaussian W2 distance (Bures metric) averaged over horizon
     * on the 2D position subspace.
     */
    std::vector<std::vector<double>> compute_transport_cost_matrix(
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<std::string>& mode_ids,
        int horizon
    );

    /**
     * @brief Compute risk vector r[m] for each mode.
     *
     * For each mode m and timestep k, computes directional risk:
     *   n = (mu_mk - c_d) / ||mu_mk - c_d||   (obstacle-to-disc direction)
     *   sigma_dir = sqrt(n^T Sigma_k n)         (directional std dev)
     *   r_{k,d} = max(0, R + z_alpha * sigma_dir - ||mu_mk - c_d||)
     *
     * r[m] = max_k max_d r_{k,d}
     *
     * Uses k=1..safe_horizon. Uses worst disc position when num_discs > 1.
     */
    std::map<std::string, double> compute_risk_vector(
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<std::string>& mode_ids,
        const std::vector<EgoState>& ego_linearization_traj,
        int horizon,
        double safety_threshold,
        int num_discs = 1,
        double vehicle_length = 4.0
    );

    /**
     * @brief Joint-horizon risk of the Euclidean collision violation (Monte Carlo).
     *
     * Estimates, per mode m,
     *     VaR_alpha(V_m)   or   CVaR_alpha(V_m),
     *     V_m := max_{k=1..N_s} max_d [ R - ||x_k - c_{d,k}|| ]_+
     * with x_k drawn from the mode's rollout x_{k+1} = A x_k + b + G w_k, so the
     * temporal correlation is exact and the distance is true Euclidean.
     *
     * Used when config.risk_measure is JOINT_VAR or JOINT_CVAR. Deterministic:
     * reseeded per mode from config.joint_risk_seed (common random numbers).
     */
    std::map<std::string, double> compute_risk_vector_joint(
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<std::string>& mode_ids,
        const std::vector<EgoState>& ego_linearization_traj,
        int horizon,
        double safety_threshold,
        int num_discs = 1,
        double vehicle_length = 4.0
    );

    /// Propagate mode mean trajectory: x_{k+1} = A*x_k + b
    std::vector<Eigen::Vector2d> propagate_mode_mean(
        const ObstacleState& obs_state,
        const ModeModel& mode,
        int horizon
    );

    /// Propagate mode covariance: Sigma_{k+1} = A*Sigma_k*A^T + G*G^T (2D position subspace)
    std::vector<Eigen::Matrix2d> propagate_mode_covariance(
        const ModeModel& mode,
        int horizon
    );

    /// Gaussian W2 distance between two 2D Gaussians (Bures metric)
    double gaussian_w2_2d(
        const Eigen::Vector2d& mu1, const Eigen::Matrix2d& cov1,
        const Eigen::Vector2d& mu2, const Eigen::Matrix2d& cov2
    );

    /// Closed-form 2x2 matrix square root
    Eigen::Matrix2d matrix_sqrt_2x2(const Eigen::Matrix2d& M);

    /**
     * @brief Solve Kantorovich dual via binary search on lambda.
     *
     * Returns optimal lambda and the dual objective value.
     */
    std::pair<double, double> solve_kantorovich_dual(
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids,
        double rho
    );

    /// Evaluate dual objective at a given lambda
    double evaluate_dual(
        double lambda,
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids,
        double rho
    );

    /**
     * @brief Build a deterministic transport plan at a given lambda.
     *
     * For each source mode i, find j*(i) = argmax_j { r_j - lambda*D[i][j] }.
     * Ties broken by tie_policy (MIN_COST keeps mass close, MAX_COST moves it far).
     */
    TransportPlan build_plan(
        double lambda,
        TiePolicy tie_policy,
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids
    );

    /**
     * @brief Find two plans that bracket the radius rho.
     *
     * Returns (plan_lo, plan_hi) where plan_lo.transport_cost <= rho <= plan_hi.transport_cost.
     * Uses binary search on lambda with different tie policies.
     */
    std::pair<TransportPlan, TransportPlan> bracket_plans(
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids,
        double rho
    );

    /**
     * @brief Refine the bracket via binary search on lambda.
     */
    void refine_bracket(
        TransportPlan& plan_lo,
        TransportPlan& plan_hi,
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids,
        double rho,
        int max_iter = 40
    );

    /**
     * @brief Mix two plans to hit the budget rho exactly.
     *
     * alpha = (rho - c_lo) / (c_hi - c_lo), then q* = alpha*q_hi + (1-alpha)*q_lo.
     * Feasible by construction since transport cost is linear in the mixing coefficient.
     */
    WorstCaseRecoveryResult mix_plans_to_radius(
        const TransportPlan& plan_lo,
        const TransportPlan& plan_hi,
        const std::vector<std::string>& mode_ids,
        double rho
    );

    /**
     * @brief Recover a feasible Q* via dual-guided bracketing + plan mixing.
     *
     * Always returns a feasible distribution within the Wasserstein ball.
     * Replaces the old recover_worst_case_distribution().
     */
    WorstCaseRecoveryResult recover_feasible_qstar(
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids,
        double rho
    );

    DROConfig config_;
    int observation_count_ = 0;
    std::optional<double> rho_override_;
    double entropy_ = 0.0;      ///< Entropy of current nominal distribution
    double max_entropy_ = 1.0;  ///< log(M) for M modes
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_WASSERSTEIN_DRO_HPP
