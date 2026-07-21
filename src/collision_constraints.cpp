/**
 * @file collision_constraints.cpp
 * @brief Implementation of linearized collision constraints.
 */

#include <algorithm>
#include "collision_constraints.hpp"
#include <cmath>

namespace dro_mpc {

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
    int num_discs,
    double vehicle_length
) {
    std::vector<CollisionConstraint> constraints;
    int horizon = static_cast<int>(reference_trajectory.size()) - 1;

    for (int k = 0; k <= horizon; ++k) {
        const EgoState& ref_state = reference_trajectory[k];

        // Compute ego disc positions (Eq. 16)
        std::vector<Eigen::Vector2d> disc_positions =
            compute_ego_disc_positions(ref_state, num_discs, vehicle_length);

        for (const auto& [obs_id, trajectory] : scenario.trajectories) {
            if (k >= static_cast<int>(trajectory.steps.size())) {
                continue;
            }

            const PredictionStep& obs_step = trajectory.steps[k];
            Eigen::Vector2d obs_position = obs_step.mean;

            // For each disc, compute constraint. The normal a is frozen from the
            // reference disc center (pre-solve); thread the disc index + longitudinal
            // offset ℓ_d so build_condensed_qp can apply the heading Jacobian
            // J_d = [[1,0,-ℓ sinθ̄],[0,1,ℓ cosθ̄]] and keep the row a proper affine
            // half-space in the decision variables (x, y, θ). Without this the QP would
            // translate the disc rigidly with the center and drop the O(ℓ·δθ) rotation.
            for (int d = 0; d < static_cast<int>(disc_positions.size()); ++d) {
                auto constraint = compute_single_constraint(
                    k, obs_id, scenario.scenario_id,
                    disc_positions[d], obs_position, combined_radius
                );
                if (constraint.has_value()) {
                    CollisionConstraint c = constraint.value();
                    c.disc_index = d;
                    c.disc_offset = get_disc_longitudinal_offset(d, num_discs, vehicle_length);
                    constraints.push_back(c);
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
    int num_discs,
    double vehicle_length
) {
    std::vector<CollisionConstraint> constraints;
    double combined_radius = ego_radius + obstacle_radius + safety_margin;

    for (const auto& scenario : scenarios) {
        auto scenario_constraints = compute_scenario_constraints(
            reference_trajectory, scenario, combined_radius, num_discs, vehicle_length
        );
        constraints.insert(constraints.end(),
            scenario_constraints.begin(), scenario_constraints.end());
    }

    return constraints;
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

namespace {

/// Project point onto collision boundary (circle of radius r centered at delta).
/// If inside, projects outward in direction from starting_pose toward delta.
Eigen::Vector2d dr_project(
    const Eigen::Vector2d& position,
    const Eigen::Vector2d& delta,
    double radius,
    const Eigen::Vector2d& starting_pose
) {
    Eigen::Vector2d diff = position - delta;
    double dist = diff.norm();

    if (dist < radius) {
        // Inside collision zone: project to boundary
        Eigen::Vector2d direction = delta - starting_pose;
        double norm_dir = direction.norm();
        if (norm_dir < 1e-10) {
            return position;  // Degenerate case
        }
        return delta - (direction / norm_dir) * radius;
    }
    // Already outside: no projection needed
    return position;
}

/// Reflect point across the projection onto the collision-free region.
/// reflect(p) = 2 * project(p) - p
Eigen::Vector2d dr_reflect(
    const Eigen::Vector2d& position,
    const Eigen::Vector2d& delta,
    double radius,
    const Eigen::Vector2d& starting_pose
) {
    return 2.0 * dr_project(position, delta, radius, starting_pose) - position;
}

}  // anonymous namespace

void douglas_rachford_projection(
    Eigen::Vector2d& position,
    const Eigen::Vector2d& obstacle,
    double radius,
    const Eigen::Vector2d& starting_pose
) {
    // Douglas-Rachford iteration:
    // position := (position + reflect(reflect(position, anchor, r), delta, r)) / 2
    // Using obstacle as both anchor and delta for single-obstacle case
    position = (position + dr_reflect(
        dr_reflect(position, obstacle, radius, position),
        obstacle, radius, starting_pose
    )) / 2.0;
}

int project_warmstart_to_safety(
    std::vector<EgoState>& trajectory,
    const std::vector<CollisionConstraint>& constraints,
    double ego_radius,
    double obstacle_radius,
    double safety_margin
) {
    int projections = 0;
    double combined_radius = ego_radius + obstacle_radius + safety_margin;

    // Group constraints by timestep
    std::map<int, std::vector<const CollisionConstraint*>> by_k;
    for (const auto& c : constraints) {
        by_k[c.k].push_back(&c);
    }

    for (auto& [k, k_constraints] : by_k) {
        if (k < 0 || k >= static_cast<int>(trajectory.size())) continue;

        Eigen::Vector2d ego_pos = trajectory[k].position();

        for (const auto* con : k_constraints) {
            double value = con->evaluate(ego_pos);
            if (value < 0) {
                // Constraint violated: use Douglas-Rachford projection
                // Reconstruct obstacle position from constraint: a^T @ p_obs + r = b
                // So p_obs = linearization_point - (constraint normal) * distance
                // Simpler: project along constraint normal
                Eigen::Vector2d starting_pose = trajectory[0].position();
                Eigen::Vector2d obstacle_approx = ego_pos - con->a * (value + combined_radius);

                douglas_rachford_projection(ego_pos, obstacle_approx, combined_radius, starting_pose);
                trajectory[k].x = ego_pos(0);
                trajectory[k].y = ego_pos(1);
                projections++;
            }
        }
    }

    return projections;
}

// ---------------------------------------------------------------------------
// Fixed-normal collision half-space API (normals computed from a numerical
// reference trajectory BEFORE the solve; held constant so the QP rows stay affine).
// ---------------------------------------------------------------------------

LinearizedCollisionHalfspace make_collision_halfspace(
    const Eigen::Vector2d& obstacle_position,
    const Eigen::Vector2d& reference_disc_center,
    double safety_radius,
    const std::optional<Eigen::Vector2d>& fallback_normal,
    double direction_epsilon
) {
    LinearizedCollisionHalfspace hs;
    hs.obstacle_position = obstacle_position;
    hs.reference_disc_center = reference_disc_center;
    hs.safety_radius = safety_radius;

    const Eigen::Vector2d delta = obstacle_position - reference_disc_center;
    const double dist = delta.norm();
    hs.reference_distance = dist;
    if (dist > direction_epsilon) {
        hs.normal = delta / dist;
        hs.used_fallback_normal = false;
    } else {
        hs.normal = fallback_normal ? fallback_normal->normalized()
                                    : Eigen::Vector2d::UnitX();
        hs.used_fallback_normal = true;
    }
    // normal^T c <= normal^T x_obs - R.
    hs.upper_bound = hs.normal.dot(obstacle_position) - safety_radius;
    return hs;
}

double get_disc_longitudinal_offset(int disc_index, int num_discs, double vehicle_length) {
    if (num_discs <= 1) return 0.0;
    const double step = vehicle_length / (num_discs - 1);
    return -vehicle_length / 2.0 + disc_index * step;
}

AffineDiscConstraint linearize_disc_halfspace(
    const LinearizedCollisionHalfspace& halfspace,
    double reference_px,
    double reference_py,
    double reference_heading,
    double longitudinal_disc_offset
) {
    const double ell = longitudinal_disc_offset;
    // c_d(x) ≈ c_bar + J_d (x - x_bar),  J_d = [[1,0,-ℓ sinθ],[0,1,ℓ cosθ]].
    Eigen::Matrix<double, 2, 3> J;
    J << 1.0, 0.0, -ell * std::sin(reference_heading),
         0.0, 1.0,  ell * std::cos(reference_heading);
    AffineDiscConstraint out;
    out.coefficients = halfspace.normal.transpose() * J;   // n^T J_d  (row 1x3)
    const Eigen::Vector3d xbar(reference_px, reference_py, reference_heading);
    // normal^T c_d <= ub  =>  (n^T J) x <= ub - n^T c_bar + (n^T J) x_bar.
    out.upper_bound = halfspace.upper_bound
                      - halfspace.normal.dot(halfspace.reference_disc_center)
                      + out.coefficients.dot(xbar);
    return out;
}

CollisionConstraint halfspace_to_collision_constraint(
    const LinearizedCollisionHalfspace& halfspace
) {
    // normal^T c <= upper_bound  <=>  (-normal)^T c >= -upper_bound.
    // With CollisionConstraint::evaluate(p) = a^T p - b, this gives the signed clearance.
    CollisionConstraint c(halfspace.horizon_step, halfspace.obstacle_id,
                          halfspace.scenario_id, -halfspace.normal, -halfspace.upper_bound);
    c.linearization_point = halfspace.reference_disc_center;
    c.disc_index = halfspace.disc_index;
    return c;
}

std::vector<LinearizedCollisionHalfspace> build_collision_halfspaces(
    const std::vector<Scenario>& scenarios,
    const std::vector<EgoState>& ego_linearization_traj,
    double ego_radius,
    double obstacle_radius,
    double safety_margin,
    int num_discs,
    double vehicle_length,
    int safe_horizon
) {
    std::vector<LinearizedCollisionHalfspace> out;
    const double R = ego_radius + obstacle_radius + safety_margin;
    const int H = static_cast<int>(ego_linearization_traj.size());
    if (H == 0) return out;
    std::optional<Eigen::Vector2d> prev_normal;
    for (const auto& scenario : scenarios) {
        for (const auto& [obs_id, traj] : scenario.trajectories) {
            const int kmax = (safe_horizon >= 0) ? std::min(safe_horizon, H - 1) : (H - 1);
            for (int k = 0; k <= kmax; ++k) {
                if (k >= static_cast<int>(traj.steps.size())) break;
                const Eigen::Vector2d obs_pos = traj.steps[k].mean;
                const EgoState& ref = ego_linearization_traj[std::min(k, H - 1)];
                auto discs = compute_ego_disc_positions(ref, num_discs, vehicle_length);
                for (int d = 0; d < static_cast<int>(discs.size()); ++d) {
                    auto hs = make_collision_halfspace(obs_pos, discs[d], R, prev_normal);
                    hs.scenario_id = scenario.scenario_id;
                    hs.obstacle_id = obs_id;
                    hs.horizon_step = k;
                    hs.disc_index = d;
                    if (!hs.used_fallback_normal) prev_normal = hs.normal;
                    out.push_back(hs);
                }
            }
        }
    }
    return out;
}

}  // namespace dro_mpc
