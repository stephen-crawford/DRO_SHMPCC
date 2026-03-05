/**
 * @file wasserstein_dro.cpp
 * @brief Implementation of Wasserstein DRO for scenario MPC.
 */

#include "wasserstein_dro.hpp"
#include "collision_constraints.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace scenario_mpc {

WassersteinDRO::WassersteinDRO(const DROConfig& config)
    : config_(config) {}

DROResult WassersteinDRO::compute_worst_case_weights(
    const std::map<std::string, double>& nominal_weights,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<EgoState>& ego_ref_traj,
    int horizon,
    double ego_r,
    double obs_r,
    double margin,
    int risk_horizon,
    int num_discs,
    double vehicle_length
) {
    DROResult result;

    // Collect mode IDs in consistent order
    std::vector<std::string> mode_ids;
    for (const auto& [id, _] : nominal_weights) {
        mode_ids.push_back(id);
    }

    if (mode_ids.empty()) {
        result.worst_case_weights = nominal_weights;
        return result;
    }

    // Update entropy for adaptive epsilon
    entropy_ = 0.0;
    for (const auto& [_, w] : nominal_weights) {
        if (w > 1e-12) {
            entropy_ -= w * std::log(w);
        }
    }
    max_entropy_ = std::log(static_cast<double>(mode_ids.size()));
    if (max_entropy_ < 1e-12) max_entropy_ = 1.0;

    // Step 1: Compute transport cost matrix D[i][j]
    result.transport_cost_matrix = compute_transport_cost_matrix(
        obs_state, mode_models, mode_ids, horizon
    );

    // Step 2: Compute risk vector r[m] (truncated to risk_horizon if set)
    double safety_threshold = ego_r + obs_r + margin;
    int effective_risk_horizon = (risk_horizon > 0) ? risk_horizon : horizon;
    result.risk_per_mode = compute_risk_vector(
        obs_state, mode_models, mode_ids, ego_ref_traj,
        effective_risk_horizon, safety_threshold, num_discs, vehicle_length
    );

    // Step 3: Get adaptive epsilon
    double epsilon = get_adaptive_epsilon();
    result.epsilon_used = epsilon;

    // Step 4: Solve Kantorovich dual
    auto [opt_lambda, dual_val] = solve_kantorovich_dual(
        nominal_weights, result.risk_per_mode,
        result.transport_cost_matrix, mode_ids, epsilon
    );
    result.optimal_lambda = opt_lambda;
    result.worst_case_risk = dual_val;

    // Step 5: Recover worst-case distribution Q*
    result.worst_case_weights = recover_worst_case_distribution(
        opt_lambda, nominal_weights, result.risk_per_mode,
        result.transport_cost_matrix, mode_ids
    );

    return result;
}

Scenario WassersteinDRO::generate_worst_case_scenario(
    const DROResult& dro_result,
    int obstacle_id,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    int horizon,
    int scenario_id
) {
    // Find mode m* = argmax_m Q*[m]  (highest weight under worst-case dist)
    std::string worst_mode;
    double max_weight = -1.0;
    for (const auto& [mode_id, w] : dro_result.worst_case_weights) {
        if (w > max_weight) {
            max_weight = w;
            worst_mode = mode_id;
        }
    }

    // If no risk or mode not found, return empty scenario
    if (worst_mode.empty() || dro_result.worst_case_risk < 1e-12 ||
        mode_models.find(worst_mode) == mode_models.end()) {
        return Scenario(scenario_id, {}, 0.0);
    }

    const ModeModel& mode = mode_models.at(worst_mode);

    // Forward-propagate deterministically (no noise = mean trajectory)
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector4d x = obs_state.to_array();
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();

    // k=0: current position
    steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

    for (int k = 0; k < horizon; ++k) {
        x = mode.A * x + mode.b;
        cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();
        steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));
    }

    // Build scenario with a single obstacle trajectory
    ObstacleTrajectory traj(obstacle_id, worst_mode, steps, max_weight);
    std::map<int, ObstacleTrajectory> trajs;
    trajs[obstacle_id] = traj;

    return Scenario(scenario_id, trajs, max_weight);
}

Scenario WassersteinDRO::generate_adversarial_scenario(
    const DROResult& dro_result,
    int obstacle_id,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<EgoState>& ego_ref,
    int horizon,
    int scenario_id,
    double sigma_scale
) {
    // Find worst-case mode m* = argmax_m Q*[m]
    std::string worst_mode;
    double max_weight = -1.0;
    for (const auto& [mode_id, w] : dro_result.worst_case_weights) {
        if (w > max_weight) {
            max_weight = w;
            worst_mode = mode_id;
        }
    }

    if (worst_mode.empty() || dro_result.worst_case_risk < 1e-12 ||
        mode_models.find(worst_mode) == mode_models.end()) {
        return Scenario(scenario_id, {}, 0.0);
    }

    const ModeModel& mode = mode_models.at(worst_mode);

    // Propagate mean trajectory and covariances
    auto means = propagate_mode_mean(obs_state, mode, horizon);
    auto covs = propagate_mode_covariance(mode, horizon);

    // Build adversarial trajectory: push obstacle toward ego along uncertain axis
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    // k=0: current position (no perturbation)
    steps.emplace_back(0, means[0], covs[0]);

    for (int k = 1; k <= horizon; ++k) {
        Eigen::Vector2d ego_pos;
        if (k < static_cast<int>(ego_ref.size())) {
            ego_pos = ego_ref[k].position();
        } else if (!ego_ref.empty()) {
            ego_pos = ego_ref.back().position();
        } else {
            // Fallback: just use mean trajectory
            steps.emplace_back(k, means[k], covs[k]);
            continue;
        }

        // Approach direction: from obstacle mean toward ego
        Eigen::Vector2d diff = ego_pos - means[k];
        double dist = diff.norm();

        if (dist < 1e-6) {
            // Already on top of ego, no direction to push
            steps.emplace_back(k, means[k], covs[k]);
            continue;
        }

        Eigen::Vector2d approach_dir = diff / dist;

        // Project covariance onto approach direction: sigma_along = sqrt(v^T * Cov * v)
        double var_along = (approach_dir.transpose() * covs[k] * approach_dir)(0, 0);
        double sigma_along = std::sqrt(std::max(0.0, var_along));

        // Adversarial position: push mean toward ego by sigma_scale * sigma_along
        Eigen::Vector2d adv_pos = means[k] + sigma_scale * sigma_along * approach_dir;

        steps.emplace_back(k, adv_pos, covs[k]);
    }

    ObstacleTrajectory traj(obstacle_id, worst_mode, steps, max_weight);
    std::map<int, ObstacleTrajectory> trajs;
    trajs[obstacle_id] = traj;

    return Scenario(scenario_id, trajs, max_weight);
}

void WassersteinDRO::set_epsilon_override(double rho) {
    epsilon_override_ = rho;
}
void WassersteinDRO::clear_epsilon_override() {
    epsilon_override_.reset();
}

double WassersteinDRO::get_adaptive_epsilon() const {
    if (epsilon_override_.has_value()) {
        return std::clamp(*epsilon_override_, config_.epsilon_min, config_.epsilon_max);
    }
    double eps = config_.epsilon_base;

    if (config_.adaptive_epsilon) {
        // Scale up with fewer observations (more uncertainty)
        double n_term = 1.0;
        if (observation_count_ > 0) {
            n_term = 1.0 + config_.confidence_alpha / std::sqrt(
                static_cast<double>(observation_count_));
        } else {
            n_term = 1.0 + config_.confidence_alpha;  // max uncertainty
        }

        // Scale up with higher entropy (more uniform = more uncertain)
        double h_ratio = (max_entropy_ > 1e-12) ? entropy_ / max_entropy_ : 0.0;
        double h_term = 1.0 + config_.entropy_gamma * h_ratio;

        eps = config_.epsilon_base * n_term * h_term;
    }

    return std::clamp(eps, config_.epsilon_min, config_.epsilon_max);
}

void WassersteinDRO::update_prediction_error(double error) {
    (void)error;  // Reserved for future adaptive epsilon refinement
}

void WassersteinDRO::set_observation_count(int n) {
    observation_count_ = n;
}

// ============================================================================
// Private methods
// ============================================================================

std::vector<std::vector<double>> WassersteinDRO::compute_transport_cost_matrix(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    int horizon
) {
    int M = static_cast<int>(mode_ids.size());
    std::vector<std::vector<double>> D(M, std::vector<double>(M, 0.0));

    // Precompute per-mode mean trajectories and covariances
    struct ModeTrajectoryData {
        std::vector<Eigen::Vector2d> means;
        std::vector<Eigen::Matrix2d> covs;
    };
    std::vector<ModeTrajectoryData> mode_data(M);

    for (int i = 0; i < M; ++i) {
        auto it = mode_models.find(mode_ids[i]);
        if (it == mode_models.end()) continue;
        mode_data[i].means = propagate_mode_mean(obs_state, it->second, horizon);
        mode_data[i].covs = propagate_mode_covariance(it->second, horizon);
    }

    // Branch on ground cost type
    if (config_.ground_cost_type == DROGroundCostType::ZERO_ONE) {
        // 0/1 cost: D[i][j] = (i != j) ? 1 : 0
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                D[i][j] = (i != j) ? 1.0 : 0.0;
            }
        }
    } else if (config_.ground_cost_type == DROGroundCostType::EUCLIDEAN_MEAN) {
        // Euclidean mean: ||mu_i - mu_j|| averaged over horizon
        for (int i = 0; i < M; ++i) {
            for (int j = i + 1; j < M; ++j) {
                double sum_dist = 0.0;
                int count = 0;
                int n_steps = std::min(
                    static_cast<int>(mode_data[i].means.size()),
                    static_cast<int>(mode_data[j].means.size()));

                for (int k = 1; k < n_steps; ++k) {
                    sum_dist += (mode_data[i].means[k] - mode_data[j].means[k]).norm();
                    count++;
                }

                double avg_dist = (count > 0) ? sum_dist / count : 0.0;
                D[i][j] = avg_dist;
                D[j][i] = avg_dist;
            }
        }
    } else {
        // Default: W2 Bures metric
        for (int i = 0; i < M; ++i) {
            for (int j = i + 1; j < M; ++j) {
                double sum_w2 = 0.0;
                int count = 0;
                int n_steps = std::min(
                    static_cast<int>(mode_data[i].means.size()),
                    static_cast<int>(mode_data[j].means.size()));

                for (int k = 1; k < n_steps; ++k) {
                    double w2 = gaussian_w2_2d(
                        mode_data[i].means[k], mode_data[i].covs[k],
                        mode_data[j].means[k], mode_data[j].covs[k]
                    );
                    sum_w2 += w2;
                    count++;
                }

                double avg_w2 = (count > 0) ? sum_w2 / count : 0.0;
                D[i][j] = avg_w2;
                D[j][i] = avg_w2;
            }
        }
    }

    return D;
}

std::map<std::string, double> WassersteinDRO::compute_risk_vector(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    const std::vector<EgoState>& ego_ref_traj,
    int horizon,
    double safety_threshold,
    int num_discs,
    double vehicle_length
) {
    std::map<std::string, double> risk;

    for (const auto& mode_id : mode_ids) {
        auto it = mode_models.find(mode_id);
        if (it == mode_models.end()) {
            risk[mode_id] = 0.0;
            continue;
        }

        auto means = propagate_mode_mean(obs_state, it->second, horizon);
        auto covs = propagate_mode_covariance(it->second, horizon);

        double max_risk = 0.0;
        int n_steps = std::min(static_cast<int>(means.size()), horizon + 1);

        for (int k = 1; k < n_steps; ++k) {
            // Get ego reference state at this timestep
            EgoState ego_state;
            if (k < static_cast<int>(ego_ref_traj.size())) {
                ego_state = ego_ref_traj[k];
            } else if (!ego_ref_traj.empty()) {
                ego_state = ego_ref_traj.back();
            }

            // Use worst (closest) disc for multi-disc risk
            double min_dist = std::numeric_limits<double>::infinity();
            if (num_discs > 1) {
                auto disc_positions = compute_ego_disc_positions(
                    ego_state, num_discs, vehicle_length);
                for (const auto& disc_pos : disc_positions) {
                    double d = (means[k] - disc_pos).norm();
                    min_dist = std::min(min_dist, d);
                }
            } else {
                min_dist = (means[k] - ego_state.position()).norm();
            }

            // Compute sigma based on risk mode
            double sigma = 0.0;
            if (config_.risk_mode == DRORiskMode::FULL) {
                sigma = config_.risk_sigma_scale * std::sqrt(covs[k].trace());
            } else if (config_.risk_mode == DRORiskMode::NO_COV) {
                sigma = 0.0;  // Zero covariance contribution
            } else if (config_.risk_mode == DRORiskMode::DISTANCE_ONLY) {
                sigma = 0.0;  // Pure distance-based
            }

            // Risk: how much the mode "invades" the safety zone (worst disc)
            double r_k = std::max(0.0, safety_threshold + sigma - min_dist);
            max_risk = std::max(max_risk, r_k);
        }

        risk[mode_id] = max_risk;
    }

    return risk;
}

std::vector<Eigen::Vector2d> WassersteinDRO::propagate_mode_mean(
    const ObstacleState& obs_state,
    const ModeModel& mode,
    int horizon
) {
    std::vector<Eigen::Vector2d> means;
    means.reserve(horizon + 1);

    Eigen::Vector4d x = obs_state.to_array();
    means.push_back(x.head<2>());

    for (int k = 0; k < horizon; ++k) {
        x = mode.A * x + mode.b;
        means.push_back(x.head<2>());
    }

    return means;
}

std::vector<Eigen::Matrix2d> WassersteinDRO::propagate_mode_covariance(
    const ModeModel& mode,
    int horizon
) {
    std::vector<Eigen::Matrix2d> covs;
    covs.reserve(horizon + 1);

    Eigen::Matrix4d cov4 = Eigen::Matrix4d::Zero();
    covs.push_back(cov4.block<2, 2>(0, 0));

    for (int k = 0; k < horizon; ++k) {
        cov4 = mode.A * cov4 * mode.A.transpose() + mode.G * mode.G.transpose();
        covs.push_back(cov4.block<2, 2>(0, 0));
    }

    return covs;
}

double WassersteinDRO::gaussian_w2_2d(
    const Eigen::Vector2d& mu1, const Eigen::Matrix2d& cov1,
    const Eigen::Vector2d& mu2, const Eigen::Matrix2d& cov2
) {
    // W2^2 = ||mu1 - mu2||^2 + Bures^2(cov1, cov2)
    // Bures^2 = tr(cov1) + tr(cov2) - 2*tr(sqrt(sqrt(cov1)*cov2*sqrt(cov1)))

    double mean_dist_sq = (mu1 - mu2).squaredNorm();

    // For 2D, use closed-form matrix square root
    Eigen::Matrix2d sqrt_cov1 = matrix_sqrt_2x2(cov1);

    // M = sqrt(cov1) * cov2 * sqrt(cov1)
    Eigen::Matrix2d M = sqrt_cov1 * cov2 * sqrt_cov1;

    // sqrt(M) using closed-form 2x2
    Eigen::Matrix2d sqrt_M = matrix_sqrt_2x2(M);

    double bures_sq = cov1.trace() + cov2.trace() - 2.0 * sqrt_M.trace();
    // Clamp to avoid numerical negatives
    bures_sq = std::max(0.0, bures_sq);

    return std::sqrt(mean_dist_sq + bures_sq);
}

Eigen::Matrix2d WassersteinDRO::matrix_sqrt_2x2(const Eigen::Matrix2d& M) {
    // Closed-form 2x2 matrix square root:
    // sqrt(M) = (M + sqrt(det(M)) * I) / sqrt(tr(M) + 2*sqrt(det(M)))

    double det_M = M.determinant();
    double tr_M = M.trace();

    // Handle near-zero or negative-definite cases
    double sqrt_det = (det_M > 0) ? std::sqrt(det_M) : 0.0;
    double denom_sq = tr_M + 2.0 * sqrt_det;

    if (denom_sq < 1e-15) {
        // M is approximately zero
        return Eigen::Matrix2d::Zero();
    }

    double denom = std::sqrt(denom_sq);
    Eigen::Matrix2d result = (M + sqrt_det * Eigen::Matrix2d::Identity()) / denom;

    return result;
}

std::pair<double, double> WassersteinDRO::solve_kantorovich_dual(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids,
    double epsilon
) {
    // Binary search on lambda >= 0
    // Dual: g(lambda) = lambda*eps + sum_i w_i * max_j(r_j - lambda*D[i][j])
    // g(lambda) is convex in lambda, we minimize it

    // Find reasonable bounds for lambda
    double r_max = 0.0;
    for (const auto& [_, r] : risk_vector) {
        r_max = std::max(r_max, r);
    }

    if (r_max < 1e-12) {
        // All modes have zero risk — Q* = P_hat
        double val = evaluate_dual(0.0, nominal_weights, risk_vector, D, mode_ids, epsilon);
        return {0.0, val};
    }

    // Find max transport cost for upper bound on lambda
    double d_min_nonzero = std::numeric_limits<double>::max();
    int M = static_cast<int>(mode_ids.size());
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            if (i != j && D[i][j] > 1e-12) {
                d_min_nonzero = std::min(d_min_nonzero, D[i][j]);
            }
        }
    }

    // lambda_max: above this, the transport cost penalty dominates,
    // so no mass moves. Upper bound: r_max / d_min
    double lambda_max = (d_min_nonzero < std::numeric_limits<double>::max() && d_min_nonzero > 0)
        ? r_max / d_min_nonzero * 2.0
        : r_max * 10.0;
    double lambda_min = 0.0;

    // Binary search: find lambda that minimizes the dual
    // The dual g(lambda) is convex, so we search for the minimum
    constexpr int MAX_ITER = 50;
    double best_lambda = 0.0;
    double best_val = evaluate_dual(0.0, nominal_weights, risk_vector, D, mode_ids, epsilon);

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double lambda_mid = (lambda_min + lambda_max) / 2.0;
        double val_mid = evaluate_dual(lambda_mid, nominal_weights, risk_vector, D, mode_ids, epsilon);

        // Check gradient direction using finite difference
        double delta = std::max(1e-8, lambda_mid * 1e-6);
        double val_plus = evaluate_dual(lambda_mid + delta, nominal_weights, risk_vector, D, mode_ids, epsilon);
        double grad = (val_plus - val_mid) / delta;

        if (val_mid < best_val) {
            best_val = val_mid;
            best_lambda = lambda_mid;
        }

        if (grad > 0) {
            // Increasing: minimum is to the left
            lambda_max = lambda_mid;
        } else {
            // Decreasing: minimum is to the right
            lambda_min = lambda_mid;
        }

        if (lambda_max - lambda_min < 1e-10) break;
    }

    return {best_lambda, best_val};
}

double WassersteinDRO::evaluate_dual(
    double lambda,
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids,
    double epsilon
) {
    // g(lambda) = lambda*eps + sum_i w_i * max_j(r_j - lambda*D[i][j])
    int M = static_cast<int>(mode_ids.size());
    double sum = lambda * epsilon;

    for (int i = 0; i < M; ++i) {
        double w_i = nominal_weights.at(mode_ids[i]);
        double max_val = -std::numeric_limits<double>::infinity();

        for (int j = 0; j < M; ++j) {
            double r_j = risk_vector.at(mode_ids[j]);
            double cost = r_j - lambda * D[i][j];
            max_val = std::max(max_val, cost);
        }

        sum += w_i * max_val;
    }

    return sum;
}

std::map<std::string, double> WassersteinDRO::recover_worst_case_distribution(
    double optimal_lambda,
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids
) {
    int M = static_cast<int>(mode_ids.size());
    std::map<std::string, double> q_star;

    // Initialize all weights to zero
    for (const auto& id : mode_ids) {
        q_star[id] = 0.0;
    }

    // For each source mode i, transport mass w_i to
    // j*(i) = argmax_j { r_j - lambda* * D[i][j] }
    for (int i = 0; i < M; ++i) {
        double w_i = nominal_weights.at(mode_ids[i]);
        if (w_i < 1e-15) continue;

        int j_star = i;  // Default: stay in place
        double max_val = -std::numeric_limits<double>::infinity();

        for (int j = 0; j < M; ++j) {
            double r_j = risk_vector.at(mode_ids[j]);
            double cost = r_j - optimal_lambda * D[i][j];
            if (cost > max_val) {
                max_val = cost;
                j_star = j;
            }
        }

        q_star[mode_ids[j_star]] += w_i;
    }

    // Normalize (should already sum to 1 but ensure numerical stability)
    double total = 0.0;
    for (const auto& [_, w] : q_star) {
        total += w;
    }
    if (total > 0) {
        for (auto& [_, w] : q_star) {
            w /= total;
        }
    }

    return q_star;
}

}  // namespace scenario_mpc
