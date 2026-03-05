/**
 * @file wasserstein_dro.hpp
 * @brief Wasserstein Distributionally Robust Optimization for scenario MPC.
 *
 * Replaces the ExtremeBuffer approach with a principled DRO reweighting:
 * at each MPC step, compute worst-case mode weights Q* within a W1 ball
 * around the nominal distribution P_hat, then sample scenarios from Q*.
 *
 * Algorithm (Kantorovich dual for W1):
 *   sup_{Q in B_eps(P_hat)} sum_m q_m * r_m
 *     = inf_{lambda>=0} { lambda*eps + sum_i w_i * max_j(r_j - lambda*D[i][j]) }
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

namespace scenario_mpc {

/**
 * @brief Risk computation mode for DRO ablation experiments.
 */
enum class DRORiskMode {
    FULL,           ///< Full risk with covariance inflation (default)
    NO_COV,         ///< Risk without covariance inflation (sigma_scale=0)
    DISTANCE_ONLY   ///< Pure distance-based risk (no covariance term)
};

/**
 * @brief Ground cost type for the Wasserstein ball transport matrix.
 */
enum class DROGroundCostType {
    W2_BURES,       ///< Gaussian W2 Bures metric (default)
    ZERO_ONE,       ///< D[i][j] = (i!=j) ? 1 : 0
    EUCLIDEAN_MEAN  ///< ||mu_i - mu_j|| averaged over horizon
};

/**
 * @brief Configuration for DRO worst-case weight computation.
 */
struct DROConfig {
    double epsilon_base = 0.1;       ///< Base Wasserstein ball radius
    double epsilon_min = 0.01;       ///< Minimum epsilon (clamped)
    double epsilon_max = 0.5;        ///< Maximum epsilon (clamped)
    bool adaptive_epsilon = true;    ///< Enable adaptive epsilon scaling
    double confidence_alpha = 1.0;   ///< Scaling for 1/sqrt(n_obs) term
    double entropy_gamma = 0.5;      ///< Scaling for entropy term
    double risk_sigma_scale = 1.0;   ///< Sigma scale for risk computation
    DRORiskMode risk_mode = DRORiskMode::FULL;                ///< Risk computation mode
    DROGroundCostType ground_cost_type = DROGroundCostType::W2_BURES;  ///< Ground cost type
};

/**
 * @brief Result from DRO worst-case weight computation.
 */
struct DROResult {
    std::map<std::string, double> worst_case_weights;  ///< Q* mode weights
    double optimal_lambda = 0.0;                       ///< Optimal dual variable
    double epsilon_used = 0.0;                         ///< Epsilon actually used
    double worst_case_risk = 0.0;                      ///< sup risk under Q*
    std::map<std::string, double> risk_per_mode;       ///< r[m] for each mode
    std::vector<std::vector<double>> transport_cost_matrix;  ///< D[i][j]
};

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
     * @param ego_ref_traj     Ego reference trajectory (for risk computation)
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
        const std::vector<EgoState>& ego_ref_traj,
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

    /// Get adaptive epsilon based on observation count and entropy
    double get_adaptive_epsilon() const;

    /// Update prediction error tracking (for potential future adaptive epsilon)
    void update_prediction_error(double error);

    /// Set the number of observations (for adaptive epsilon scaling)
    void set_observation_count(int n);

    /// Optional per-step epsilon override (e.g. from adaptive shift detection).
    void set_epsilon_override(double rho);
    void clear_epsilon_override();

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
     * r[m] = max_k max(0, safety_threshold + sigma_scale*sqrt(tr(Sigma_k))
     *         - min_d ||mu_mk - p_d(k)||)
     * Uses k=1..risk_horizon (truncated if safe horizon is active).
     * Uses worst disc position when num_discs > 1.
     */
    std::map<std::string, double> compute_risk_vector(
        const ObstacleState& obs_state,
        const std::map<std::string, ModeModel>& mode_models,
        const std::vector<std::string>& mode_ids,
        const std::vector<EgoState>& ego_ref_traj,
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
        double epsilon
    );

    /// Evaluate dual objective at a given lambda
    double evaluate_dual(
        double lambda,
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids,
        double epsilon
    );

    /**
     * @brief Recover Q* from optimal lambda.
     *
     * For each source mode i, transport mass w_i to
     * j*(i) = argmax_j { r_j - lambda* * D[i][j] }.
     */
    std::map<std::string, double> recover_worst_case_distribution(
        double optimal_lambda,
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids
    );

    DROConfig config_;
    int observation_count_ = 0;
    std::optional<double> epsilon_override_;
    double entropy_ = 0.0;      ///< Entropy of current nominal distribution
    double max_entropy_ = 1.0;  ///< log(M) for M modes
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_WASSERSTEIN_DRO_HPP
