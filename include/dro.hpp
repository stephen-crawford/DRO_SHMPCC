/**
 * @file dro.hpp
 * @brief Distributionally Robust Optimization for scenario MPC.
 *
 */

#ifndef DRO_MPC_DRO_HPP
#define DRO_MPC_DRO_HPP

#include "types.hpp"
#include "config.hpp"
#include <vector>
#include <map>
#include <string>
#include <optional>
#include <random>
#include <limits>

namespace dro_mpc {

// DROConfig lives in dro_config.hpp so RuntimeConfig can nest it without
// including this solver header. DROGroundCostType / DRORiskMeasure are in types.hpp.

/**
 * @brief Tie-breaking policy for deterministic transport plan construction.
 */
enum class TiePolicy {
    MIN_COST,   // Among tied destinations, pick the one with lowest D[i][j] (less movement)
    MAX_COST    // Among tied destinations, pick the one with highest D[i][j] (more movement)
};
transport_cost
/**
 * @brief A deterministic transport plan at a given lambda.
 */
struct TransportPlan {
    double lambda = 0.0;
    std::map<std::string, double> q;        // Resulting mode distribution
    double transport_cost = 0.0;            // sum_i p_i D[i, j*(i)]
    double expected_risk = 0.0;             // sum_j q_j r_j
};

/**
 * @brief Result of worst-case distribution recovery with transport cost verification.
 */
struct WorstCaseRecoveryResult {
    std::map<std::string, double> q_star;   // Recovered worst-case distribution
    double implied_transport_cost = 0.0;    // Actual transport cost of the plan
    bool feasible = false;                  // implied_transport_cost <= rho (+tol)
    double mix_alpha = 0.0;                 // Convex mixing coefficient (0=low-cost, 1=high-cost)
};

/**
 * @brief Result from DRO worst-case weight computation.
 */
struct DROResult {
    std::map<std::string, double> worst_case_weights;  // q* mode weights
    double optimal_lambda = 0.0;                       // Optimal dual variable
    double rho_used = 0.0;                              // Wasserstein radius rho used
    double worst_case_risk = 0.0;                      // sup risk under Q*
    std::map<std::string, double> risk_per_mode;       // r[m] for each mode
    std::vector<std::vector<double>> transport_cost_matrix;  // D[i][j]
    double implied_transport_cost = 0.0;               // Transport cost of induced plan
    bool recovery_feasible = false;                    // Whether induced plan respects rho

   
    double qstar_support_floor = 0.0;   // min_m Q*[m]. 0 => Assumption 1 FAILS (bang-bang).
    int qstar_support_size = 0;         // #{m : Q*[m] > 0}. < M => Assumption 1 FAILS.
    bool satisfies_full_support = false;// qstar_support_floor > 0, i.e. Assumption 1 holds.

    double likelihood_ratio_bound() const {
        if (!(qstar_support_floor > 0.0)) {
            return std::numeric_limits<double>::infinity();
        }
        return 1.0 / qstar_support_floor;
    }
};

/**
 * @brief Result of the entropic (Sinkhorn-style) allocator.
 */
struct EntropicOTResult {
    std::map<std::string, double> q;    // Target marginal q_tau (STRICTLY positive)
    double lambda = 0.0;                // Dual price on the transport budget
    double transport_cost = 0.0;        // sum_ij Pi_ij D_ij
    double expected_risk = 0.0;         // <q_tau, r>  ("protection")
    double q_min = 0.0;                 // min_m q_tau[m] > 0 by construction
    bool solved = false;                // lambda search converged and budget respected

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
        double vehicle_length = 4.0,
        /// When non-null, the obstacle is treated as a Markov-jump system that may
        /// switch modes during the horizon: per-mode risk is computed over the
        /// transition chain (compute_risk_vector_switching) instead of held modes.
        /// Rows/cols must be indexed consistently with the mode ordering.
        const Eigen::MatrixXd* transition = nullptr
    );

    /**
     * @brief Generate a worst-case scenario for a single obstacle.
     *
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

    /// Compute effective support size of q* (number of modes with weight > threshold)
    static int effective_support(const DROResult& dro_result, double threshold = 0.01);

    /**
     * @brief Sample S scenarios by drawing modes according to q*.
     *
     * For each scenario s = 0..S-1, draw mode m ~ q* and forward-propagate
     * the mode-conditioned Gaussian trajectory (mean + covariance) from the
     * current obstacle state.  The scenario probability is set to q*[m].
     *
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
     *
     * When `transition` is non-null the obstacle is a Markov-jump system: each sample
     * draws a mode SEQUENCE from the chain seeded at mode m AND a noise path, so the
     * estimator is a risk measure of the JOINT (sequence, noise) law rather than a
     * mean over sequences. At transition = I this reduces exactly to the held-mode
     * estimator, so the two cases share one implementation. This is the exact
     * reference that MIXTURE_* approximates semi-analytically.
     */
    std::map<std::string, double> compute_risk_vector_joint(
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<std::string>& mode_ids,
        const std::vector<EgoState>& ego_linearization_traj,
        int horizon,
        double safety_threshold,
        int num_discs = 1,
        double vehicle_length = 4.0,
        const Eigen::MatrixXd* transition = nullptr
    );

    /**
     * @brief Semi-analytic risk of the sequence-mixture surrogate (MIXTURE_VAR/CVAR).
     *
     * Conditional on a sampled mode sequence the surrogate makes the trajectory
     * violation Gaussian, V | s ~ N(mu_s, sigma_s^2), so over K sampled sequences V is
     * an equally-weighted K-component Gaussian mixture whose VaR/CVaR are available in
     * closed form given the components (see the MIXTURE_* note in types.hpp):
     *
     *   VaR : bisect (1/K) sum_s Phi((q-mu_s)/sigma_s) = alpha, then clamp [q]_+
     *   CVaR: Rockafellar-Uryasev at q* = VaR, exact for the clamped variable since q* >= 0
     *
     * Unlike the E_seq[...] estimator this is a genuine (and, for CVaR, coherent) risk
     * measure of the joint (sequence, noise) uncertainty. Cost is K sequence rollouts;
     * the noise is integrated analytically, so no per-sample noise draws are needed.
     *
     * `transition = nullptr` gives a one-component mixture (the held mode), which
     * reduces exactly to SURROGATE_VAR / SURROGATE_CVAR.
     */
    std::map<std::string, double> compute_risk_vector_mixture(
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<std::string>& mode_ids,
        const std::vector<EgoState>& ego_linearization_traj,
        int horizon,
        double safety_threshold,
        int num_discs,
        double vehicle_length,
        const Eigen::MatrixXd* transition
    );

    /**
     * @brief Switching-aware per-mode risk for Markov-jump obstacles.
     *
     * When the obstacle may change modes DURING the horizon, r[m] is the danger of
     * STARTING in mode m and evolving under the estimated transition chain: we
     * Monte-Carlo sample mode sequences from `transition` seeded at mode m,
     * propagate the obstacle under the switching dynamics, evaluate the same
     * surrogate violation per sequence, and average (expectation over the chain).
     * `transition` must be row/col-indexed consistently with `mode_ids`.
     */
    std::map<std::string, double> compute_risk_vector_switching(
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<std::string>& mode_ids,
        const std::vector<EgoState>& ego_linearization_traj,
        int horizon,
        double safety_threshold,
        int num_discs,
        double vehicle_length,
        const Eigen::MatrixXd& transition
    );

    /// Max over (step,disc) of the surrogate clearance violation for one obstacle
    /// mean/covariance trajectory. Shared by the held and switching risk paths.
    double surrogate_traj_violation(
        const std::vector<Eigen::Vector2d>& means,
        const std::vector<Eigen::Matrix2d>& covs,
        const std::vector<EgoState>& ego_traj,
        int horizon, double safety_radius, int num_discs, double vehicle_length,
        double z_alpha, double alpha) const;

    std::pair<double, double> surrogate_traj_gaussian(
        const std::vector<Eigen::Vector2d>& means,
        const std::vector<Eigen::Matrix2d>& covs,
        const std::vector<EgoState>& ego_traj,
        int horizon, double safety_radius, int num_discs, double vehicle_length,
        double z_alpha) const;

    /**
     * @brief PROPER Bonferroni VaR (DRORiskMeasure::BONFERRONI_VAR).
     *
     * r[m] = max_{k,d} VaR_{a'}( [R - ||x_k - c_{d,k}||]_+ ),  x_k ~ N(mu_k, Sigma_k),
     * with a' = 1 - (1-alpha)/(N_s*D) the Bonferroni union-corrected level. The
     * per-step Euclidean VaR is estimated by Monte Carlo on each step's MARGINAL
     * Gaussian (no directional projection), so this is the true-distance analogue of
     * SURROGATE_VAR_BONFERRONI and a valid upper bound on the joint-horizon VaR.
     */
    std::map<std::string, double> compute_risk_vector_bonferroni(
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<std::string>& mode_ids,
        const std::vector<EgoState>& ego_linearization_traj,
        int horizon,
        double safety_radius,
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

    /**
     * @brief Normalised sliced Wasserstein-1 between two 2D Gaussians.
     *
     */
    double sliced_w1_gaussian_2d(
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
    double entropy_ = 0.0;      // Entropy of current nominal distribution
    double max_entropy_ = 1.0;  // log(M) for M modes
};

}  // namespace dro_mpc

#endif  // DRO_MPC_DRO_HPP
