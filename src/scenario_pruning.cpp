/**
 * @file scenario_pruning.cpp
 * @brief Implementation of scenario pruning algorithms.
 */

#include "scenario_pruning.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace scenario_mpc {

namespace {

/**
 * @brief Check if scenario s1 dominates s2.
 *
 * s1 dominates s2 if obstacles in s1 are always at least as CLOSE to
 * the ego as in s2 (s1 is more constraining everywhere).
 * If s1 dominates s2, then s2 can be pruned because satisfying s1's
 * constraints automatically satisfies s2's constraints.
 */
bool scenario_dominates(
    const Scenario& s1,
    const Scenario& s2,
    const std::vector<EgoState>& reference_trajectory
) {
    int horizon = static_cast<int>(reference_trajectory.size()) - 1;

    for (int k = 0; k <= horizon; ++k) {
        Eigen::Vector2d ego_pos = reference_trajectory[k].position();

        // Check all obstacles present in both scenarios
        for (const auto& [obs_id, traj1] : s1.trajectories) {
            auto it2 = s2.trajectories.find(obs_id);
            if (it2 == s2.trajectories.end()) {
                continue;
            }
            const ObstacleTrajectory& traj2 = it2->second;

            if (k >= static_cast<int>(traj1.steps.size()) ||
                k >= static_cast<int>(traj2.steps.size())) {
                continue;
            }

            Eigen::Vector2d pos1 = traj1.steps[k].mean;
            Eigen::Vector2d pos2 = traj2.steps[k].mean;

            double dist1 = (ego_pos - pos1).norm();
            double dist2 = (ego_pos - pos2).norm();

            // s1 dominates s2 if s1's obstacles are closer (more constraining)
            // If s1's obstacle is farther at any point, s1 doesn't dominate
            if (dist1 > dist2 + 1e-6) {
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief Convert scenario to low-dimensional quotient feature (mean/end positions per obstacle).
 * Used for quotient-space reduction to cluster in reduced dimension.
 */
Eigen::VectorXd scenario_to_quotient_feature(const Scenario& scenario, int horizon = -1) {
    if (scenario.trajectories.empty()) {
        return Eigen::VectorXd::Zero(1);
    }
    std::vector<int> sorted_obs;
    for (const auto& [obs_id, _] : scenario.trajectories) {
        sorted_obs.push_back(obs_id);
    }
    std::sort(sorted_obs.begin(), sorted_obs.end());
    std::vector<double> feats;
    for (int obs_id : sorted_obs) {
        const auto& traj = scenario.trajectories.at(obs_id);
        if (traj.steps.empty()) continue;
        int H = horizon >= 0 ? std::min(horizon + 1, static_cast<int>(traj.steps.size())) : static_cast<int>(traj.steps.size());
        double sum_x = 0, sum_y = 0;
        for (int k = 0; k < H; ++k) {
            sum_x += traj.steps[k].mean(0);
            sum_y += traj.steps[k].mean(1);
        }
        double mean_x = sum_x / H;
        double mean_y = sum_y / H;
        double end_x = traj.steps[H - 1].mean(0);
        double end_y = traj.steps[H - 1].mean(1);
        feats.push_back(mean_x);
        feats.push_back(mean_y);
        feats.push_back(end_x);
        feats.push_back(end_y);
    }
    if (feats.empty()) return Eigen::VectorXd::Constant(1, 0.0);
    Eigen::VectorXd out(feats.size());
    for (size_t i = 0; i < feats.size(); ++i) out(i) = feats[i];
    return out;
}

/**
 * @brief Convert scenario to feature vector for clustering.
 */
Eigen::VectorXd scenario_to_feature(const Scenario& scenario, int horizon = -1) {
    if (horizon < 0) {
        // Infer horizon from trajectory length
        for (const auto& [_, traj] : scenario.trajectories) {
            horizon = static_cast<int>(traj.steps.size()) - 1;
            break;
        }
        if (horizon < 0) {
            horizon = 10;
        }
    }

    std::vector<double> features;

    // Sort obstacle IDs for consistent ordering
    std::vector<int> sorted_obs_ids;
    for (const auto& [obs_id, _] : scenario.trajectories) {
        sorted_obs_ids.push_back(obs_id);
    }
    std::sort(sorted_obs_ids.begin(), sorted_obs_ids.end());

    for (int obs_id : sorted_obs_ids) {
        const auto& traj = scenario.trajectories.at(obs_id);
        int num_steps = std::min(horizon + 1, static_cast<int>(traj.steps.size()));
        for (int k = 0; k < num_steps; ++k) {
            features.push_back(traj.steps[k].mean(0));
            features.push_back(traj.steps[k].mean(1));
        }
    }

    if (features.empty()) {
        return Eigen::VectorXd::Constant(1, 0.0);
    }

    Eigen::VectorXd result(features.size());
    for (size_t i = 0; i < features.size(); ++i) {
        result(i) = features[i];
    }
    return result;
}

}  // anonymous namespace

std::vector<Scenario> prune_dominated_scenarios(
    const std::vector<Scenario>& scenarios,
    const std::vector<EgoState>& reference_trajectory,
    double ego_radius,
    double obstacle_radius
) {
    if (scenarios.size() <= 1) {
        return scenarios;
    }

    int n = static_cast<int>(scenarios.size());
    std::set<int> dominated;

    // Pairwise comparison (never prune DRO-injected scenarios)
    for (int i = 0; i < n; ++i) {
        if (dominated.count(i) || scenarios[i].is_injected) {
            continue;
        }

        for (int j = i + 1; j < n; ++j) {
            if (dominated.count(j)) {
                continue;
            }

            // Never prune an injected scenario
            if (scenarios[j].is_injected) {
                continue;
            }

            // Check if i dominates j or vice versa
            bool dom_i_j = scenario_dominates(
                scenarios[i], scenarios[j], reference_trajectory
            );
            bool dom_j_i = scenario_dominates(
                scenarios[j], scenarios[i], reference_trajectory
            );

            if (dom_i_j) {
                dominated.insert(j);
            } else if (dom_j_i) {
                dominated.insert(i);
                break;
            }
        }
    }

    // Return non-dominated scenarios
    std::vector<Scenario> result;
    for (int i = 0; i < n; ++i) {
        if (dominated.count(i) == 0) {
            result.push_back(scenarios[i]);
        }
    }
    return result;
}

std::pair<std::vector<Scenario>, std::set<int>> remove_inactive_scenarios(
    const std::vector<Scenario>& scenarios,
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& optimal_trajectory,
    double tolerance
) {
    // Group constraints by scenario
    std::map<int, std::vector<CollisionConstraint>> constraints_by_scenario;
    for (const auto& c : constraints) {
        constraints_by_scenario[c.scenario_id].push_back(c);
    }

    std::set<int> active_scenarios;

    for (const auto& scenario : scenarios) {
        int sid = scenario.scenario_id;

        if (constraints_by_scenario.find(sid) == constraints_by_scenario.end()) {
            continue;
        }

        // Check if any constraint from this scenario is active
        for (const auto& constraint : constraints_by_scenario[sid]) {
            int k = constraint.k;
            if (k >= static_cast<int>(optimal_trajectory.size())) {
                continue;
            }

            Eigen::Vector2d ego_pos = optimal_trajectory[k].position();
            double value = constraint.evaluate(ego_pos);

            // Constraint is active if value is close to 0
            if (std::abs(value) < tolerance) {
                active_scenarios.insert(sid);
                break;
            }
        }
    }

    // Keep scenarios that have active constraints
    std::vector<Scenario> remaining;
    for (const auto& s : scenarios) {
        if (active_scenarios.count(s.scenario_id)) {
            remaining.push_back(s);
        }
    }

    // If no scenarios are active, keep all (safety fallback)
    if (remaining.empty()) {
        std::set<int> all_ids;
        for (const auto& s : scenarios) {
            all_ids.insert(s.scenario_id);
        }
        return {scenarios, all_ids};
    }

    return {remaining, active_scenarios};
}

std::vector<Scenario> prune_by_probability(
    const std::vector<Scenario>& scenarios,
    double min_probability
) {
    std::vector<Scenario> filtered;
    for (const auto& s : scenarios) {
        if (s.probability >= min_probability) {
            filtered.push_back(s);
        }
    }
    return filtered;
}

std::vector<Scenario> cluster_similar_scenarios(
    const std::vector<Scenario>& scenarios,
    double distance_threshold,
    int horizon
) {
    if (scenarios.size() <= 1) {
        return scenarios;
    }

    // Compute scenario features (flattened obstacle positions)
    std::vector<Eigen::VectorXd> features;
    for (const auto& scenario : scenarios) {
        features.push_back(scenario_to_feature(scenario, horizon));
    }

    // Simple greedy clustering
    std::vector<int> kept_indices;
    for (size_t i = 0; i < features.size(); ++i) {
        bool is_close = false;
        for (int j : kept_indices) {
            if ((features[i] - features[j]).norm() < distance_threshold) {
                is_close = true;
                break;
            }
        }
        if (!is_close) {
            kept_indices.push_back(static_cast<int>(i));
        }
    }

    std::vector<Scenario> result;
    for (int i : kept_indices) {
        result.push_back(scenarios[i]);
    }
    return result;
}

std::vector<Scenario> select_diverse_scenarios(
    const std::vector<Scenario>& scenarios,
    int num_select,
    int horizon
) {
    if (static_cast<int>(scenarios.size()) <= num_select) {
        return scenarios;
    }

    // Convert to features
    std::vector<Eigen::VectorXd> features;
    for (const auto& scenario : scenarios) {
        features.push_back(scenario_to_feature(scenario, horizon));
    }

    // Farthest point sampling
    std::vector<int> selected;
    selected.push_back(0);  // Start with first scenario

    while (static_cast<int>(selected.size()) < num_select) {
        // Find point farthest from all selected points
        double max_min_dist = -1;
        int best_idx = -1;

        for (int i = 0; i < static_cast<int>(scenarios.size()); ++i) {
            if (std::find(selected.begin(), selected.end(), i) != selected.end()) {
                continue;
            }

            // Min distance to any selected point
            double min_dist = std::numeric_limits<double>::infinity();
            for (int j : selected) {
                double dist = (features[i] - features[j]).norm();
                min_dist = std::min(min_dist, dist);
            }

            if (min_dist > max_min_dist) {
                max_min_dist = min_dist;
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            selected.push_back(best_idx);
        } else {
            break;
        }
    }

    std::vector<Scenario> result;
    for (int i : selected) {
        result.push_back(scenarios[i]);
    }
    return result;
}

int adaptive_scenario_budget(
    int current_scenarios,
    double constraint_violations,
    int max_scenarios,
    int min_scenarios,
    double growth_factor,
    double shrink_factor
) {
    int new_budget;

    if (constraint_violations > 0.1) {
        // Violations present - increase scenarios
        new_budget = static_cast<int>(current_scenarios * growth_factor);
    } else if (constraint_violations < 0.001) {
        // No violations - can decrease scenarios
        new_budget = static_cast<int>(current_scenarios * shrink_factor);
    } else {
        new_budget = current_scenarios;
    }

    return std::max(min_scenarios, std::min(max_scenarios, new_budget));
}

std::vector<Scenario> reduce_scenarios_quotient_space(
    const std::vector<Scenario>& scenarios,
    int num_quotient,
    int horizon
) {
    const int S = static_cast<int>(scenarios.size());
    if (S <= num_quotient || num_quotient <= 0) {
        return scenarios;
    }
    int H = horizon;
    if (H < 0 && !scenarios.empty() && !scenarios[0].trajectories.empty()) {
        H = static_cast<int>(scenarios[0].trajectories.begin()->second.steps.size()) - 1;
    }
    if (H < 0) H = 10;

    std::vector<Eigen::VectorXd> qfeatures;
    qfeatures.reserve(S);
    for (const auto& sc : scenarios) {
        qfeatures.push_back(scenario_to_quotient_feature(sc, H));
    }
    const int dim = static_cast<int>(qfeatures[0].size());

    // K-centers: iteratively add the point farthest from current centers
    std::vector<int> centers;
    centers.push_back(0);
    while (static_cast<int>(centers.size()) < num_quotient) {
        double max_dist = -1;
        int best_idx = -1;
        for (int i = 0; i < S; ++i) {
            double min_d = std::numeric_limits<double>::infinity();
            for (int c : centers) {
                double d = (qfeatures[i] - qfeatures[c]).norm();
                min_d = std::min(min_d, d);
            }
            if (min_d > max_dist) {
                max_dist = min_d;
                best_idx = i;
            }
        }
        if (best_idx >= 0) {
            centers.push_back(best_idx);
        } else {
            break;
        }
    }

    // Assign each scenario to nearest center; compute cluster centroid
    int K = static_cast<int>(centers.size());
    std::vector<Eigen::VectorXd> centroids(K);
    std::vector<std::vector<int>> clusters(K);
    for (int i = 0; i < K; ++i) {
        centroids[i] = Eigen::VectorXd::Zero(dim);
    }
    for (int i = 0; i < S; ++i) {
        int nearest = 0;
        double d_min = (qfeatures[i] - qfeatures[centers[0]]).norm();
        for (int k = 1; k < K; ++k) {
            double d = (qfeatures[i] - qfeatures[centers[k]]).norm();
            if (d < d_min) {
                d_min = d;
                nearest = k;
            }
        }
        clusters[nearest].push_back(i);
        centroids[nearest] += qfeatures[i];
    }
    for (int k = 0; k < K; ++k) {
        if (!clusters[k].empty()) {
            centroids[k] /= static_cast<double>(clusters[k].size());
        }
    }

    // Representative per cluster: scenario closest to cluster centroid
    std::vector<Scenario> result;
    result.reserve(K);
    for (int k = 0; k < K; ++k) {
        if (clusters[k].empty()) {
            result.push_back(scenarios[centers[k]]);
            continue;
        }
        int rep = clusters[k][0];
        double rep_dist = (qfeatures[rep] - centroids[k]).norm();
        for (int idx : clusters[k]) {
            double d = (qfeatures[idx] - centroids[k]).norm();
            if (d < rep_dist) {
                rep_dist = d;
                rep = idx;
            }
        }
        Scenario sc = scenarios[rep];
        sc.scenario_id = k;
        result.push_back(sc);
    }
    return result;
}

}  // namespace scenario_mpc
