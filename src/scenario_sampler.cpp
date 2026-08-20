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
 * predict it stays at its current position with small growing uncertainty.
 */
ObstacleTrajectory make_stationary_trajectory(
    int obstacle_id,
    const ObstacleState& initial_state,
    int horizon
) {
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector2d pos = initial_state.position();
    Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
    const double sigma_growth = 0.01;  // small per-step uncertainty growth

    steps.emplace_back(0, pos, cov);
    for (int k = 1; k <= horizon; ++k) {
        cov += sigma_growth * Eigen::Matrix2d::Identity();
        steps.emplace_back(k, pos, cov);
    }

    return ObstacleTrajectory(obstacle_id, "stationary", steps, 1.0);
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
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();

    // Initial step
    steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

    for (int k = 0; k < horizon; ++k) {
        // Propagate with sampled noise
        Eigen::VectorXd noise = noise_samples.row(k).transpose();
        x = mode.A * x + mode.b + mode.G * noise;

        // Update covariance (for uncertainty representation)
        cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();

        steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));
    }

    return ObstacleTrajectory(
        obstacle_id, sampled_mode_id, steps, mode_weights.at(sampled_mode_id)
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
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();

    // Initial step
    steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

    // Draw mode_0 from the SAME one-step predictive belief pi_1 = T^T pi_0 that the
    // sequence-probability accumulation below uses for the first step (via
    // predict_mode_belief). predict_before_first_sample=true is REQUIRED here: with the
    // default (false) mode_0 would be drawn from the raw belief pi_0 while its stored
    // probability used pi_1, misaligning the sampling law and the recorded trajectory_prob
    // by exactly one step. (The DRO risk-vector path in wasserstein_dro.cpp deliberately
    // uses false + a point-mass belief and stores no sequence probability, so it is exempt.)
    auto sequence = sample_mode_sequence(
        mode_belief,
        transition_matrix,
        modes,
        horizon,
        rng,
        /*predict_before_first_sample=*/true
    );

    std::map<std::string, int> mode_to_idx;
    for (int i = 0; i < static_cast<int>(modes.size()); ++i) {
        mode_to_idx[modes[i]] = i;
    }

    // Track which mode was used most (for trajectory labeling)
    std::map<std::string, int> mode_counts;
    for (const auto& mode_id : modes) {
        mode_counts[mode_id] = 0;
    }

    double trajectory_prob = 1.0;
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // First-step probability under the one-step predictive belief pi_1 = T^T pi_0.
    // This MUST match the seed used by sample_mode_sequence above (predict_before_first_
    // sample=true) so trajectory_prob is the true likelihood of the sampled sequence.
    ModeDistribution step_belief = predict_mode_belief(
        mode_belief, transition_matrix, modes
    );

    for (int k = 0; k < horizon; ++k) {
        const std::string& mode_id = sequence[k];
        const ModeModel& mode = available_modes.at(mode_id);
        mode_counts[mode_id]++;

        auto belief_it = step_belief.find(mode_id);
        if (belief_it != step_belief.end()) {
            trajectory_prob *= belief_it->second;
        }

        // Sample noise
        int noise_dim = mode.noise_dim();
        Eigen::VectorXd noise(noise_dim);
        for (int d = 0; d < noise_dim; ++d) {
            noise(d) = normal_dist(rng);
        }

        // Propagate
        x = mode.A * x + mode.b + mode.G * noise;
        cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();

        steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));

        // Next conditional distribution is the sampled mode's transition row.
        if (k + 1 < horizon) {
            auto idx_it = mode_to_idx.find(mode_id);
            if (idx_it != mode_to_idx.end()) {
                step_belief.clear();
                for (int j = 0; j < static_cast<int>(modes.size()); ++j) {
                    step_belief[modes[j]] = transition_matrix(idx_it->second, j);
                }
            }
        }
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

    return ObstacleTrajectory(obstacle_id, dominant_mode, steps, trajectory_prob);
}

}  // anonymous namespace

std::vector<Scenario> sample_scenarios(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    int horizon,
    int num_scenarios,
    const ModeBeliefConfig& mode_belief,
    int current_timestep,
    std::mt19937* rng
) {
    // Create local RNG if not provided
    std::mt19937 local_rng;
    if (rng == nullptr) {
        std::random_device rd;
        local_rng = std::mt19937(rd());
        rng = &local_rng;
    }

    std::vector<Scenario> scenarios;
    scenarios.reserve(num_scenarios);

    for (int s = 0; s < num_scenarios; ++s) {
        std::map<int, ObstacleTrajectory> trajectories;

        for (const auto& [obs_id, obs_state] : obstacles) {
            auto hist_it = mode_histories.find(obs_id);
            if (hist_it == mode_histories.end()) {
                // No mode history — treat as stationary
                trajectories[obs_id] = make_stationary_trajectory(obs_id, obs_state, horizon);
                continue;
            }

            const ModeHistory& mode_history = hist_it->second;

            // Step 1: Compute mode weights
            auto mode_weights = compute_mode_weights(
                mode_history, mode_belief, current_timestep
            );

            if (mode_weights.empty()) {
                // No modes observed yet — treat as stationary
                trajectories[obs_id] = make_stationary_trajectory(obs_id, obs_state, horizon);
                continue;
            }

            // Step 2 & 3 & 4: Sample trajectory
            ObstacleTrajectory trajectory = sample_obstacle_trajectory(
                obs_id, obs_state, mode_history.available_modes, mode_weights,
                horizon, *rng
            );

            trajectories[obs_id] = trajectory;
        }

        // Compute scenario probability as product of trajectory probabilities
        double scenario_prob = 1.0;
        for (const auto& [_, traj] : trajectories) {
            scenario_prob *= traj.probability;
        }

        scenarios.emplace_back(s, trajectories, scenario_prob);
    }

    return scenarios;
}

std::vector<Scenario> sample_scenarios_with_weights(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    const std::map<int, std::map<std::string, double>>& per_obstacle_weights,
    int horizon,
    int num_scenarios,
    bool ensure_mode_coverage,
    std::mt19937* rng
) {
    // Create local RNG if not provided
    std::mt19937 local_rng;
    if (rng == nullptr) {
        std::random_device rd;
        local_rng = std::mt19937(rd());
        rng = &local_rng;
    }

    // Determine mode coverage requirements
    int num_coverage = 0;
    struct ObsCoverageInfo {
        std::map<std::string, double> weights;
        std::vector<std::string> coverage_modes;
    };
    std::map<int, ObsCoverageInfo> obs_info;

    for (const auto& [obs_id, obs_state] : obstacles) {
        auto weight_it = per_obstacle_weights.find(obs_id);
        if (weight_it == per_obstacle_weights.end()) continue;
        auto hist_it = mode_histories.find(obs_id);
        if (hist_it == mode_histories.end()) continue;

        ObsCoverageInfo info;
        info.weights = weight_it->second;
        if (ensure_mode_coverage) {
            for (const auto& [mode_id, w] : info.weights) {
                if (w > 0.0) {
                    info.coverage_modes.push_back(mode_id);
                }
            }
            num_coverage = std::max(num_coverage,
                static_cast<int>(info.coverage_modes.size()));
        }
        obs_info[obs_id] = std::move(info);
    }

    if (ensure_mode_coverage) {
        num_coverage = std::min(num_coverage, num_scenarios);
    }

    std::vector<Scenario> scenarios;
    scenarios.reserve(num_scenarios);

    // Phase 1: Coverage scenarios (if enabled)
    for (int s = 0; s < num_coverage; ++s) {
        std::map<int, ObstacleTrajectory> trajectories;

        for (const auto& [obs_id, obs_state] : obstacles) {
            auto info_it = obs_info.find(obs_id);
            if (info_it == obs_info.end()) {
                trajectories[obs_id] = make_stationary_trajectory(obs_id, obs_state, horizon);
                continue;
            }
            const auto& info = info_it->second;
            auto hist_it = mode_histories.find(obs_id);
            const ModeHistory& mode_history = hist_it->second;

            if (s < static_cast<int>(info.coverage_modes.size())) {
                // Force this specific mode
                std::map<std::string, double> forced_weights;
                for (const auto& [mode_id, _] : info.weights) {
                    forced_weights[mode_id] = 0.0;
                }
                forced_weights[info.coverage_modes[s]] = 1.0;

                ObstacleTrajectory trajectory = sample_obstacle_trajectory(
                    obs_id, obs_state, mode_history.available_modes, forced_weights,
                    horizon, *rng
                );
                trajectory.probability = info.weights.at(info.coverage_modes[s]);
                trajectories[obs_id] = trajectory;
            } else {
                // Sample normally with provided weights
                ObstacleTrajectory trajectory = sample_obstacle_trajectory(
                    obs_id, obs_state, mode_history.available_modes, info.weights,
                    horizon, *rng
                );
                trajectories[obs_id] = trajectory;
            }
        }

        double scenario_prob = 1.0;
        for (const auto& [_, traj] : trajectories) {
            scenario_prob *= traj.probability;
        }
        scenarios.emplace_back(s, trajectories, scenario_prob);
    }

    // Phase 2: Remaining scenarios — sample with provided weights
    for (int s = num_coverage; s < num_scenarios; ++s) {
        std::map<int, ObstacleTrajectory> trajectories;

        for (const auto& [obs_id, obs_state] : obstacles) {
            auto info_it = obs_info.find(obs_id);
            if (info_it == obs_info.end()) {
                trajectories[obs_id] = make_stationary_trajectory(obs_id, obs_state, horizon);
                continue;
            }
            const auto& info = info_it->second;
            auto hist_it = mode_histories.find(obs_id);
            const ModeHistory& mode_history = hist_it->second;

            ObstacleTrajectory trajectory = sample_obstacle_trajectory(
                obs_id, obs_state, mode_history.available_modes, info.weights,
                horizon, *rng
            );
            trajectories[obs_id] = trajectory;
        }

        double scenario_prob = 1.0;
        for (const auto& [_, traj] : trajectories) {
            scenario_prob *= traj.probability;
        }
        scenarios.emplace_back(s, trajectories, scenario_prob);
    }

    return scenarios;
}

std::vector<Scenario> sample_scenarios_markov(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    const std::map<int, std::map<std::string, double>>* per_obstacle_belief,
    int horizon,
    int num_scenarios,
    const ModeBeliefConfig& belief_cfg,
    std::mt19937* rng
) {
    std::mt19937 local_rng;
    if (rng == nullptr) {
        std::random_device rd;
        local_rng = std::mt19937(rd());
        rng = &local_rng;
    }

    // Per-obstacle belief + transition matrix, built ONCE. Neither depends on the
    // scenario index, so hoisting them out of the s-loop is exact, not an
    // approximation -- it only removes redundant work.
    struct ObsPlan {
        bool stationary = true;
        ModeDistribution belief;
        Eigen::MatrixXd transition;
        std::vector<std::string> modes;
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

        // Initial belief: caller-supplied (e.g. DRO Q*) takes precedence.
        ModeDistribution belief;
        if (per_obstacle_belief != nullptr) {
            auto b_it = per_obstacle_belief->find(obs_id);
            if (b_it != per_obstacle_belief->end() && !b_it->second.empty()) {
                belief = b_it->second;
            }
        }
        const int M = static_cast<int>(mode_history.available_modes.size());
        if (belief.empty()) {
            belief = compute_mode_weights(mode_history, belief_cfg);
        }
        if (belief.empty()) {
            plans[obs_id] = plan;  // cold start -> stationary
            continue;
        }

        plan.modes.reserve(mode_history.available_modes.size());
        for (const auto& [mode_id, _] : mode_history.available_modes) {
            plan.modes.push_back(mode_id);
        }
        // alpha and kappa are DERIVED from the stated prior + self-persistence
        // assumption (see ModeBeliefConfig): kappa is not a free constant.
        plan.transition = compute_mode_transition_matrix(
            mode_history, plan.modes,
            belief_cfg.alpha(M), belief_cfg.kappa(M)
        );
        plan.belief = std::move(belief);
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
            trajectories[obs_id] = sample_trajectory_with_mode_sequence(
                obs_id, obs_state, mode_histories.at(obs_id).available_modes,
                plan.belief, plan.transition, plan.modes, horizon, *rng
            );
        }
        scenarios.emplace_back(s, trajectories, 1.0 / num_scenarios);
    }

    return scenarios;
}

}  // namespace dro_mpc
