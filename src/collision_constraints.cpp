/**
 * @file collision_constraints.cpp
 * @brief Implementation of linearized collision constraints.
 */

#include "collision_constraints.hpp"
#include <cmath>

namespace scenario_mpc {

namespace {

/**
 * @brief Compute a single linearized collision constraint.
 *
 * Following Eq. 17 and Eq. 18:
 *
 * Eq. 17: a = (p_ego - p_obs) / ||p_ego - p_obs||
 * Eq. 18: a^T @ p_ego >= a^T @ p_obs + r_combined
 */
std::optional<CollisionConstraint> compute_single_constraint(
    int k,
    int obstacle_id,
    int scenario_id,
    const Eigen::Vector2d& ego_position,
    const Eigen::Vector2d& obstacle_position,
    double combined_radius
) {
    // Compute direction vector (Eq. 17)
    Eigen::Vector2d diff = ego_position - obstacle_position;
    double dist = diff.norm();

    Eigen::Vector2d a;
    // Handle degenerate case
    if (dist < 1e-6) {
        // Default direction if positions coincide
        a = Eigen::Vector2d(1.0, 0.0);
    } else {
        a = diff / dist;
    }

    // Compute constraint offset (Eq. 18)
    // a^T @ p_ego >= a^T @ p_obs + r_combined
    // Rearranged: a^T @ p_ego >= b
    // where b = a^T @ p_obs + r_combined
    double b = a.dot(obstacle_position) + combined_radius;

    CollisionConstraint c(k, obstacle_id, scenario_id, a, b);
    c.linearization_point = ego_position;
    return c;
}

/**
 * @brief Compute linearized constraints for a single scenario.
 */
std::vector<CollisionConstraint> compute_scenario_constraints(
    const std::vector<EgoState>& reference_trajectory,
    const Scenario& scenario,
    double combined_radius,
    int num_discs
) {
    std::vector<CollisionConstraint> constraints;
    int horizon = static_cast<int>(reference_trajectory.size()) - 1;

    for (int k = 0; k <= horizon; ++k) {
        const EgoState& ref_state = reference_trajectory[k];

        // Compute ego disc positions (Eq. 16)
        std::vector<Eigen::Vector2d> disc_positions =
            compute_ego_disc_positions(ref_state, num_discs);

        for (const auto& [obs_id, trajectory] : scenario.trajectories) {
            if (k >= static_cast<int>(trajectory.steps.size())) {
                continue;
            }

            const PredictionStep& obs_step = trajectory.steps[k];
            Eigen::Vector2d obs_position = obs_step.mean;

            // For each disc, compute constraint
            for (const auto& disc_pos : disc_positions) {
                auto constraint = compute_single_constraint(
                    k, obs_id, scenario.scenario_id,
                    disc_pos, obs_position, combined_radius
                );
                if (constraint.has_value()) {
                    constraints.push_back(constraint.value());
                }
            }
        }
    }

    return constraints;
}

}  // anonymous namespace

std::vector<CollisionConstraint> compute_linearized_constraints(
    const std::vector<EgoState>& reference_trajectory,
    const std::vector<Scenario>& scenarios,
    double ego_radius,
    double obstacle_radius,
    double safety_margin,
    int num_discs
) {
    std::vector<CollisionConstraint> constraints;
    double combined_radius = ego_radius + obstacle_radius + safety_margin;

    for (const auto& scenario : scenarios) {
        auto scenario_constraints = compute_scenario_constraints(
            reference_trajectory, scenario, combined_radius, num_discs
        );
        constraints.insert(constraints.end(),
            scenario_constraints.begin(), scenario_constraints.end());
    }

    return constraints;
}

std::vector<CollisionConstraint> tighten_constraints_by_certificate(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<double>& radii
) {
    if (radii.empty()) return constraints;
    std::vector<CollisionConstraint> out;
    out.reserve(constraints.size());
    for (const auto& c : constraints) {
        CollisionConstraint c2 = c;
        size_t k = static_cast<size_t>(c.k);
        if (k < radii.size()) {
            double r = radii[k];
            double anorm = c.a.norm();
            if (anorm > 1e-12)
                c2.b = c.b - r * anorm;
        }
        out.push_back(c2);
    }
    return out;
}

std::vector<Eigen::Vector2d> compute_ego_disc_positions(
    const EgoState& state,
    int num_discs,
    double vehicle_length
) {
    if (num_discs == 1) {
        return {state.position()};
    }

    std::vector<Eigen::Vector2d> positions;
    positions.reserve(num_discs);

    Eigen::Vector2d center = state.position();
    double theta = state.theta;

    // Direction vector
    Eigen::Vector2d direction(std::cos(theta), std::sin(theta));

    // Place discs evenly along vehicle
    std::vector<double> offsets;
    if (num_discs > 1) {
        double step = vehicle_length / (num_discs - 1);
        for (int i = 0; i < num_discs; ++i) {
            offsets.push_back(-vehicle_length / 2 + i * step);
        }
    } else {
        offsets.push_back(0.0);
    }

    for (double offset : offsets) {
        Eigen::Vector2d pos = center + offset * direction;
        positions.push_back(pos);
    }

    return positions;
}

std::pair<double, std::vector<CollisionConstraint>> evaluate_constraint_violation(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& ego_trajectory
) {
    double max_violation = 0.0;
    std::vector<CollisionConstraint> violated;

    for (const auto& constraint : constraints) {
        int k = constraint.k;
        if (k >= static_cast<int>(ego_trajectory.size())) {
            continue;
        }

        Eigen::Vector2d ego_pos = ego_trajectory[k].position();
        double value = constraint.evaluate(ego_pos);

        if (value < 0) {
            double violation = -value;
            if (violation > max_violation) {
                max_violation = violation;
            }
            violated.push_back(constraint);
        }
    }

    return {max_violation, violated};
}

Eigen::Vector4d compute_constraint_jacobian(
    const CollisionConstraint& constraint,
    const EgoState& state,
    const Eigen::Matrix4d& dynamics_jacobian
) {
    // For position-only constraint, Jacobian is simply [a1, a2, 0, 0]
    // assuming state = [x, y, theta, v]
    Eigen::Vector4d jac = Eigen::Vector4d::Zero();
    jac(0) = constraint.a(0);
    jac(1) = constraint.a(1);
    return jac;
}

std::vector<CollisionConstraint> filter_constraints_by_distance(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& ego_trajectory,
    double max_distance
) {
    std::vector<CollisionConstraint> filtered;

    for (const auto& constraint : constraints) {
        int k = constraint.k;
        if (k >= static_cast<int>(ego_trajectory.size())) {
            continue;
        }

        Eigen::Vector2d ego_pos = ego_trajectory[k].position();
        double value = constraint.evaluate(ego_pos);

        if (value < max_distance) {
            filtered.push_back(constraint);
        }
    }

    return filtered;
}

std::vector<CollisionConstraint> merge_redundant_constraints(
    const std::vector<CollisionConstraint>& constraints,
    double angle_threshold,
    double offset_threshold
) {
    if (constraints.empty()) {
        return {};
    }

    // Group constraints by timestep
    std::map<int, std::vector<CollisionConstraint>> by_timestep;
    for (const auto& c : constraints) {
        by_timestep[c.k].push_back(c);
    }

    std::vector<CollisionConstraint> merged;

    for (auto& [k, k_constraints] : by_timestep) {
        // Simple approach: keep the most conservative constraint for similar directions
        std::vector<CollisionConstraint> kept;

        for (const auto& c : k_constraints) {
            bool is_redundant = false;

            for (auto it = kept.begin(); it != kept.end(); ++it) {
                // Check if directions are similar
                double cos_angle = c.a.dot(it->a);
                if (cos_angle > std::cos(angle_threshold)) {
                    // Similar direction - keep the more conservative one (larger b)
                    if (c.b > it->b) {
                        kept.erase(it);
                        kept.push_back(c);
                    }
                    is_redundant = true;
                    break;
                }
            }

            if (!is_redundant) {
                kept.push_back(c);
            }
        }

        merged.insert(merged.end(), kept.begin(), kept.end());
    }

    return merged;
}

}  // namespace scenario_mpc
