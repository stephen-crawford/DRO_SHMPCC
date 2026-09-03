/**
 * @file scenario_sampler.cpp
 * @brief Implementation of scenario sampling.
 */

#include "scenario_sampler.hpp"
#include <cmath>

namespace dro_mpc {

namespace {

/**
 * @brief Generate a stationary (hold-position) trajectory for an unobserved obstacle.
 *
 * Used during cold start: the obstacle has no observed modes yet, so we
 * predict it stays at its current position with bounded Lyapunov uncertainty.
 */
ObstacleTrajectory make_stationary_trajectory(
    int obstacle_id,
    const ObstacleState& initial_state,
    int horizon
) {
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector2d pos = initial_state.position();
    Eigen::Matrix2d position_covariance = Eigen::Matrix2d::Zero();
    const Eigen::Matrix2d hold_transition = 0.98 * Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d hold_process_noise = 0.01 * Eigen::Matrix2d::Identity();

    // Solve P = A P A^T + Q by fixed-point iteration. The stable hold
    // predictor gives cold-start uncertainty a finite Lyapunov limit.
    for (int i = 0; i < 1000; ++i) {
        Eigen::Matrix2d next = hold_transition * position_covariance * hold_transition.transpose()
                             + hold_process_noise;
        if ((next - position_covariance).norm() < 1e-12) {
            position_covariance = next;
            break;
        }
        position_covariance = next;
    }

    steps.emplace_back(0, pos, position_covariance);
    for (int k = 1; k <= horizon; ++k) {
        position_covariance = hold_transition * position_covariance * hold_transition.transpose()
                    + hold_process_noise;
        steps.emplace_back(k, pos, position_covariance);
    }

    return ObstacleTrajectory(obstacle_id, "stationary", steps);
}

/**
 * @brief Sample a single obstacle trajectory.
 */
ObstacleTrajectory sample_obstacle_trajectory(
    int obstacle_id,
    const ObstacleState& initial_state,
    const std::map<std::string, ModeModel>& available_modes,
    const std::map<std::string, double>& mode_weights,
    int horizon,
    std::mt19937& rng
) {
    // Sample mode for this trajectory (constant mode over horizon for simplicity)
    std::string sampled_mode_id = sample_mode_from_weights(mode_weights, rng);
    const ModeModel& mode = available_modes.at(sampled_mode_id);

    // Sample noise sequence
    int noise_dim = mode.noise_dim();
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    Eigen::MatrixXd noise_samples(horizon, noise_dim);
    for (int k = 0; k < horizon; ++k) {
        for (int d = 0; d < noise_dim; ++d) {
            noise_samples(k, d) = normal_dist(rng);
        }
    }

    // Propagate trajectory
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector4d x = initial_state.to_array();
    Eigen::Matrix4d state_covariance = Eigen::Matrix4d::Zero();

    // Initial step
    steps.emplace_back(0, x.head<2>(), state_covariance.block<2, 2>(0, 0));

    for (int k = 0; k < horizon; ++k) {
        // Propagate with sampled noise
        Eigen::VectorXd noise = noise_samples.row(k).transpose();
        x = mode.propagate(ObstacleState::from_array(x), &noise).to_array();

        // Update covariance (for uncertainty representation)
        mode.propagate_covariance(state_covariance);

        steps.emplace_back(k + 1, x.head<2>(), state_covariance.block<2, 2>(0, 0));
    }

    return ObstacleTrajectory(
        obstacle_id, sampled_mode_id, steps
    );
}

/**
 * @brief Sample trajectory with Markovian mode switching over the horizon.
 */
ObstacleTrajectory sample_trajectory_with_mode_sequence(
    int obstacle_id,
    const ObstacleState& initial_state,
    const std::map<std::string, ModeModel>& available_modes,
    const ModeDistribution& mode_belief,
    const Eigen::MatrixXd& transition_matrix,
    const std::vector<std::string>& modes,
    int horizon,
    std::mt19937& rng
) {
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector4d x = initial_state.to_array();
    Eigen::Matrix4d state_covariance = Eigen::Matrix4d::Zero();

    // Initial step
    steps.emplace_back(0, x.head<2>(), state_covariance.block<2, 2>(0, 0));

    // Draw mode_0 from the SAME one-step predictive belief pi_1 = T^T pi_0 that the
    // predict_before_first_sample=true keeps the first draw consistent with the
    // predictive belief used by this Markov sampler.
    auto sequence = sample_mode_sequence(
        mode_belief,
        transition_matrix,
        modes,
        horizon,
        rng,
        /*predict_before_first_sample=*/true
    );

    // Track which mode was used most (for trajectory labeling)
    std::map<std::string, int> mode_counts;
    for (const auto& mode_id : modes) {
        mode_counts[mode_id] = 0;
    }

    std::normal_distribution<double> normal_dist(0.0, 1.0);

    for (int k = 0; k < horizon; ++k) {
        const std::string& mode_id = sequence[k];
        const ModeModel& mode = available_modes.at(mode_id);
        mode_counts[mode_id]++;

        // Sample noise
        int noise_dim = mode.noise_dim();
        Eigen::VectorXd noise(noise_dim);
        for (int d = 0; d < noise_dim; ++d) {
            noise(d) = normal_dist(rng);
        }

        // Propagate
        x = mode.propagate(ObstacleState::from_array(x), &noise).to_array();
        mode.propagate_covariance(state_covariance);

        steps.emplace_back(k + 1, x.head<2>(), state_covariance.block<2, 2>(0, 0));

    }

    // Label trajectory with most frequent mode
    std::string dominant_mode;
    int max_count = 0;
    for (const auto& [mode_id, count] : mode_counts) {
        if (count > max_count) {
            max_count = count;
            dominant_mode = mode_id;
        }
    }

    return ObstacleTrajectory(obstacle_id, dominant_mode, steps);
}

}  // anonymous namespace

std::vector<Scenario> sample_scenarios(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    const std::map<int, std::map<std::string, double>>* per_obstacle_distribution,
    int horizon,
    int num_scenarios,
    const ModeBeliefConfig& mode_belief,
    const std::map<int, Eigen::MatrixXd>* per_obstacle_transitions,
    std::mt19937* rng
) {
    // Create local RNG if not provided
    std::mt19937 local_rng;
    if (rng == nullptr) {
        std::random_device rd;
        local_rng = std::mt19937(rd());
        rng = &local_rng;
    }

    // Per-obstacle nominal belief, built ONCE. p̂ depends only on history +
    // the Dirichlet prior, not on the scenario index, so hoisting this out of
    // the s-loop is exact — it only removes redundant work.
    struct ObsPlan {
        bool stationary = true;
        ModeDistribution weights;
        const std::map<std::string, ModeModel>* available_modes = nullptr;
        Eigen::MatrixXd transition;
    };
    std::map<int, ObsPlan> plans;

    for (const auto& [obs_id, obs_state] : obstacles) {
        ObsPlan plan;
        auto hist_it = mode_histories.find(obs_id);
        if (hist_it == mode_histories.end()) {
            plans[obs_id] = plan;  // stationary
            continue;
        }
        const ModeHistory& mode_history = hist_it->second;
        ModeDistribution weights;
        if (per_obstacle_distribution != nullptr) {
            auto distribution_it = per_obstacle_distribution->find(obs_id);
            if (distribution_it != per_obstacle_distribution->end()) {
                weights = distribution_it->second;
            }
        }
        if (weights.empty()) {
            weights = compute_mode_weights(mode_history, mode_belief);
        }
        if (weights.empty()) {
            plans[obs_id] = plan;  // cold start -> stationary
            continue;
        }
        plan.weights = std::move(weights);
        plan.available_modes = &mode_history.available_modes;
        if (per_obstacle_transitions != nullptr) {
            auto transition_it = per_obstacle_transitions->find(obs_id);
            if (transition_it != per_obstacle_transitions->end()) {
                plan.transition = transition_it->second;
            }
        }
        plan.stationary = false;
        plans[obs_id] = std::move(plan);
    }

    std::vector<Scenario> scenarios;
    scenarios.reserve(num_scenarios);

    for (int s = 0; s < num_scenarios; ++s) {
        std::map<int, ObstacleTrajectory> trajectories;

        for (const auto& [obs_id, obs_state] : obstacles) {
            const ObsPlan& plan = plans[obs_id];
            if (plan.stationary) {
                trajectories[obs_id] = make_stationary_trajectory(obs_id, obs_state, horizon);
                continue;
            }
            if (plan.transition.size() > 0) {
                std::vector<std::string> modes;
                for (const auto& [mode_id, _] : *plan.available_modes)
                    modes.push_back(mode_id);
                trajectories[obs_id] = sample_trajectory_with_mode_sequence(
                    obs_id, obs_state, *plan.available_modes, plan.weights,
                    plan.transition, modes, horizon, *rng);
            } else {
                trajectories[obs_id] = sample_obstacle_trajectory(
                    obs_id, obs_state, *plan.available_modes, plan.weights,
                    horizon, *rng);
            }
        }

        scenarios.emplace_back(s, trajectories);
    }

    return scenarios;
}

}  // namespace dro_mpc
