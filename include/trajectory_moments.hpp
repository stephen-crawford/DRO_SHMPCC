/**
 * @file trajectory_moments.hpp
 * @brief Trajectory moment computation for multi-modal obstacle prediction.
 *
 * Implements Proposition 1 (Section 6):
 * Recursive computation of mean and covariance for trajectory distributions.
 *
 * Given mode-dependent dynamics x_{k+1} = A_m @ x_k + b_m + G_m @ w_k,
 * computes the first two moments of the predictive distribution.
 */

#ifndef SCENARIO_MPC_TRAJECTORY_MOMENTS_HPP
#define SCENARIO_MPC_TRAJECTORY_MOMENTS_HPP

#include "types.hpp"

namespace scenario_mpc {

/**
 * @brief Compute trajectory moments using Proposition 1.
 *
 * For multi-modal predictions, computes the combined mean and covariance
 * by marginalizing over modes.
 *
 * Proposition 1:
 *     mu_k = sum_m w_m * (A_m @ mu_{k-1} + b_m)
 *     Sigma_k = sum_m w_m * (A_m @ Sigma_{k-1} @ A_m^T + G_m @ G_m^T)
 *             + sum_m w_m * (mu_m_k - mu_k)(mu_m_k - mu_k)^T
 *
 * @param initial_state Initial obstacle state
 * @param mode_weights Weight for each mode (should sum to 1)
 * @param available_modes Dict of ModeModel for each mode
 * @param horizon Prediction horizon N
 * @return TrajectoryMoments with means and covariances for each timestep
 */
TrajectoryMoments compute_trajectory_moments(
    const ObstacleState& initial_state,
    const std::map<std::string, double>& mode_weights,
    const std::map<std::string, ModeModel>& available_modes,
    int horizon
);

/**
 * @brief Compute trajectory for a single mode with optional noise.
 *
 * @param initial_state Initial obstacle state
 * @param mode Mode model to use
 * @param horizon Prediction horizon
 * @param noise_samples Optional (horizon, noise_dim) matrix of noise samples
 * @return ObstacleTrajectory with prediction steps
 */
ObstacleTrajectory compute_single_mode_trajectory(
    const ObstacleState& initial_state,
    const ModeModel& mode,
    int horizon,
    const Eigen::MatrixXd* noise_samples = nullptr
);

/**
 * @brief Propagate obstacle state one step using given mode.
 *
 * @param state Current obstacle state
 * @param mode Mode model to use
 * @param noise Optional process noise sample
 * @return Next obstacle state
 */
ObstacleState propagate_obstacle_state(
    const ObstacleState& state,
    const ModeModel& mode,
    const Eigen::VectorXd* noise = nullptr
);

/**
 * @brief Generate trajectory predictions for each mode.
 *
 * @param initial_state Initial obstacle state
 * @param mode_weights Weight for each mode
 * @param available_modes Dict of ModeModel for each mode
 * @param horizon Prediction horizon
 * @param obstacle_id Obstacle identifier
 * @return List of ObstacleTrajectory, one per mode with non-zero weight
 */
std::vector<ObstacleTrajectory> compute_multi_mode_prediction(
    const ObstacleState& initial_state,
    const std::map<std::string, double>& mode_weights,
    const std::map<std::string, ModeModel>& available_modes,
    int horizon,
    int obstacle_id = 0
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_TRAJECTORY_MOMENTS_HPP
