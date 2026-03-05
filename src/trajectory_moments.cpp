/**
 * @file trajectory_moments.cpp
 * @brief Implementation of trajectory moment computation.
 */

#include "trajectory_moments.hpp"

namespace scenario_mpc {

TrajectoryMoments compute_trajectory_moments(
    const ObstacleState& initial_state,
    const std::map<std::string, double>& mode_weights,
    const std::map<std::string, ModeModel>& available_modes,
    int horizon
) {
    TrajectoryMoments result;
    result.obstacle_id = 0;

    // Initialize storage
    result.means.resize(horizon + 1, 2);
    result.covariances.resize(horizon + 1);

    // Initial state (no uncertainty)
    Eigen::Vector4d x0 = initial_state.to_array();
    result.means.row(0) = x0.head<2>();  // Position only
    result.covariances[0] = Eigen::Matrix2d::Zero();

    // Track full state for each mode
    std::map<std::string, Eigen::Vector4d> mode_means;
    std::map<std::string, Eigen::Matrix4d> mode_covs;

    for (const auto& [mode_id, _] : mode_weights) {
        mode_means[mode_id] = x0;
        mode_covs[mode_id] = Eigen::Matrix4d::Zero();
    }

    // Recursive computation for each timestep
    for (int k = 1; k <= horizon; ++k) {
        // Propagate each mode forward
        std::map<std::string, Eigen::Vector4d> new_mode_means;
        std::map<std::string, Eigen::Matrix4d> new_mode_covs;

        for (const auto& [mode_id, weight] : mode_weights) {
            if (weight <= 0) continue;

            const ModeModel& model = available_modes.at(mode_id);
            Eigen::Vector4d mu_prev = mode_means[mode_id];
            Eigen::Matrix4d Sigma_prev = mode_covs[mode_id];

            // Propagate mean: mu_m_k = A_m @ mu_{k-1} + b_m
            Eigen::Vector4d mu_new = model.A * mu_prev + model.b;

            // Propagate covariance: Sigma_m_k = A_m @ Sigma_{k-1} @ A_m^T + G_m @ G_m^T
            Eigen::Matrix4d Sigma_new = model.A * Sigma_prev * model.A.transpose() +
                                        model.G * model.G.transpose();

            new_mode_means[mode_id] = mu_new;
            new_mode_covs[mode_id] = Sigma_new;
        }

        mode_means = new_mode_means;
        mode_covs = new_mode_covs;

        // Compute combined moments by marginalizing over modes
        // mu_k = sum_m w_m * mu_m_k
        Eigen::Vector4d combined_mean = Eigen::Vector4d::Zero();
        for (const auto& [mode_id, weight] : mode_weights) {
            if (weight > 0 && mode_means.find(mode_id) != mode_means.end()) {
                combined_mean += weight * mode_means[mode_id];
            }
        }

        // Compute combined covariance
        // Sigma_k = sum_m w_m * Sigma_m_k + sum_m w_m * (mu_m_k - mu_k)(mu_m_k - mu_k)^T
        Eigen::Matrix4d combined_cov = Eigen::Matrix4d::Zero();

        for (const auto& [mode_id, weight] : mode_weights) {
            if (weight <= 0 || mode_means.find(mode_id) == mode_means.end()) {
                continue;
            }

            // Within-mode covariance
            combined_cov += weight * mode_covs[mode_id];

            // Between-mode covariance (mode mixing term)
            Eigen::Vector4d diff = mode_means[mode_id] - combined_mean;
            combined_cov += weight * diff * diff.transpose();
        }

        // Extract position components (first 2 states)
        result.means.row(k) = combined_mean.head<2>();
        result.covariances[k] = combined_cov.block<2, 2>(0, 0);
    }

    return result;
}

ObstacleTrajectory compute_single_mode_trajectory(
    const ObstacleState& initial_state,
    const ModeModel& mode,
    int horizon,
    const Eigen::MatrixXd* noise_samples
) {
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector4d x = initial_state.to_array();
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();

    // Initial step
    steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

    for (int k = 0; k < horizon; ++k) {
        // Propagate mean
        if (noise_samples != nullptr) {
            Eigen::VectorXd noise = noise_samples->row(k).transpose();
            x = mode.A * x + mode.b + mode.G * noise;
        } else {
            x = mode.A * x + mode.b;
        }

        // Propagate covariance
        cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();

        steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));
    }

    return ObstacleTrajectory(0, mode.mode_id, steps, 1.0);
}

ObstacleState propagate_obstacle_state(
    const ObstacleState& state,
    const ModeModel& mode,
    const Eigen::VectorXd* noise
) {
    return mode.propagate(state, noise);
}

std::vector<ObstacleTrajectory> compute_multi_mode_prediction(
    const ObstacleState& initial_state,
    const std::map<std::string, double>& mode_weights,
    const std::map<std::string, ModeModel>& available_modes,
    int horizon,
    int obstacle_id
) {
    std::vector<ObstacleTrajectory> trajectories;

    for (const auto& [mode_id, weight] : mode_weights) {
        if (weight <= 0) continue;

        const ModeModel& model = available_modes.at(mode_id);
        ObstacleTrajectory traj = compute_single_mode_trajectory(
            initial_state, model, horizon
        );
        traj.obstacle_id = obstacle_id;
        traj.probability = weight;
        trajectories.push_back(traj);
    }

    return trajectories;
}

}  // namespace scenario_mpc
