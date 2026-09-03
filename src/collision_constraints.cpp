/**
 * @file collision_constraints.cpp
 * @brief Implementation of linearized collision constraints.
 */

#include <algorithm>
#include "collision_constraints.hpp"
#include <cassert>
#include <cmath>
#include <map>
#include <set>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace dro_mpc {

// ===========================================================================
// Half-space notation — consistent across construction, projection, and pruning
// ---------------------------------------------------------------------------
//   n := (x_obs - c) / ||x_obs - c||           unit normal, ego disc center c -> obstacle
//   Geometric collision half-space:  n^T p <= n^T x_obs - R   (keep the ego on its
//   side, at least the safety radius R from the obstacle).
//
//   CollisionConstraint stores the EQUIVALENT form  a^T p >= b  with
//       a := -n = (c - x_obs)/||c - x_obs||     (obstacle -> ego),
//       b := -(n^T x_obs - R) = a^T x_obs + R,
//   so evaluate(p) = a^T p - b is the SIGNED CLEARANCE (>= 0 safe, < 0 violated).
//
//   Projection: p <- p - (a^T p - b)/||a||^2 * a   (moves along +a, away from obstacle).
//   Pruning (de Groot Def 2): scenario j dominates i (H_j subset H_i) iff j's
//   half-space IMPLIES i's on the reachable ball -- certified exactly by
//   halfspace_implies_on_ball on the actual (a, b) pairs, per ego disc.
// ===========================================================================

namespace {


/**
 * Ego disc center:
 *
 *     c_d(x, y, theta)
 *       = [x, y]^T + ell_d [cos(theta), sin(theta)]^T.
 */
std::optional<CollisionConstraint> compute_single_constraint(
    int k,
    int obstacle_id,
    int scenario_id,
    const Eigen::Vector2d& reference_disc_center,
    const Eigen::Vector2d& obstacle_position,
    double combined_radius,
    double direction_epsilon = 1e-8
) {
    if (!reference_disc_center.allFinite() ||
        !obstacle_position.allFinite()) {
        throw std::invalid_argument(
            "Collision geometry contains non-finite values."
        );
    }

    if (!std::isfinite(combined_radius) ||
        combined_radius < 0.0) {
        throw std::invalid_argument(
            "Combined collision radius must be finite and nonnegative."
        );
    }

    LinearizedCollisionHalfspace hs = make_collision_halfspace(
        obstacle_position, reference_disc_center, combined_radius,
        std::nullopt, direction_epsilon);
    hs.horizon_step = k;
    hs.obstacle_id  = obstacle_id;
    hs.scenario_id  = scenario_id;
    return halfspace_to_collision_constraint(hs);
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

        std::vector<Eigen::Vector2d> disc_positions =
            compute_ego_disc_positions(ref_state, num_discs, vehicle_length);

        for (const auto& [obs_id, trajectory] : scenario.trajectories) {
            if (k >= static_cast<int>(trajectory.steps.size())) {
                continue;
            }

            const PredictionStep& obs_step = trajectory.steps[k];
            Eigen::Vector2d obs_position = obs_step.mean;

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

Eigen::Vector2d compute_collision_disc_center(
    const EgoState& state,
    const CollisionConstraint& constraint
) {
    return linearize_disc_center(state, constraint.disc_offset).center;
}

DiscCenterLinearization linearize_disc_center(
    const EgoState& state,
    double longitudinal_disc_offset
) {
    const double theta = state.theta;
    const double ell = longitudinal_disc_offset;
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);

    DiscCenterLinearization out;
    out.center = state.position() + ell * Eigen::Vector2d(cosine, sine);
    out.jacobian << 1.0, 0.0, -ell * sine,
                    0.0, 1.0,  ell * cosine;
    return out;
}

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

std::vector<Eigen::Vector2d> compute_ego_disc_positions( // c_{k, d} = [x_k, y_k] + ell_d [cos(theta_k), sin(theta_k)]
    const EgoState& state,
    int num_discs,
    double vehicle_length
) {
    if (num_discs <= 0) {
        throw std::invalid_argument("num_discs must be positive.");
    }
    if (!std::isfinite(vehicle_length) || vehicle_length < 0.0) {
        throw std::invalid_argument("vehicle_length must be finite and nonnegative.");
    }
    std::vector<Eigen::Vector2d> positions;
    positions.reserve(num_discs);
    for (int disc_index = 0; disc_index < num_discs; ++disc_index) {
        const double offset = get_disc_longitudinal_offset(
            disc_index, num_discs, vehicle_length);
        positions.push_back(linearize_disc_center(state, offset).center);
    }

    return positions;
}

std::pair<double, std::vector<CollisionConstraint>>
evaluate_constraint_violation(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& ego_trajectory
) {
    double max_violation = 0.0;
    std::vector<CollisionConstraint> violated;

    for (const CollisionConstraint& constraint : constraints) {
        const int k = constraint.k;

        if (k < 0 ||
            k >= static_cast<int>(ego_trajectory.size())) {
            continue;
        }

        const Eigen::Vector2d disc_center =
            compute_collision_disc_center(
                ego_trajectory[k],
                constraint
            );

        const double signed_clearance =
            constraint.evaluate(disc_center);

        if (signed_clearance < 0.0) {
            max_violation =
                std::max(max_violation, -signed_clearance);

            violated.push_back(constraint);
        }
    }

    return {max_violation, violated};
}

std::vector<CollisionConstraint>
filter_constraints_by_clearance(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& ego_trajectory,
    double maximum_clearance
) {
    if (!std::isfinite(maximum_clearance)) {
        throw std::invalid_argument(
            "maximum_clearance must be finite."
        );
    }

    std::vector<CollisionConstraint> filtered;

    for (const CollisionConstraint& constraint : constraints) {
        const int k = constraint.k;

        if (k < 0 ||
            k >= static_cast<int>(ego_trajectory.size())) {
            continue;
        }

        const Eigen::Vector2d disc_center =
            compute_collision_disc_center(
                ego_trajectory[k],
                constraint
            );

        // evaluate(c) = a^T c - b.
        const double signed_clearance =
            constraint.evaluate(disc_center);

        if (signed_clearance < maximum_clearance) {
            filtered.push_back(constraint);
        }
    }

    return filtered;
}

namespace {

    /**
     * @brief Certify implication between two collision half-spaces on a ball.
     *
     * Tests the bounded-domain implication
     *
     *     Ball(center, rho) ∩ {p : a1^T p >= b1}
     *         ⊆ {p : a2^T p >= b2}.
     *
     * Equivalently, this tests whether
     *
     *     min  a2^T p - b2
     *      p
     *
     *     s.t. ||p - center||_2 <= rho,
     *          a1^T p >= b1
     *
     * is nonnegative up to numerical tolerance.
     *
     * The minimization has a closed-form solution in two dimensions and does
     * not require the two half-space normals to be parallel.
     *
     * If the first half-space does not intersect the ball, logical implication
     * would be vacuously true. This implementation deliberately returns false
     * in that case so that an infeasible scenario cannot be used to prune
     * another scenario.
     *
     * @return true only when implication is certified on the complete ball.
     */
    bool certifies_halfspace_implication_on_ball(
        const Eigen::Vector2d& a1,
        double b1,
        const Eigen::Vector2d& a2,
        double b2,
        const Eigen::Vector2d& center,
        double rho,
        double tolerance = 1e-9
    ) {
        if (!a1.allFinite() ||
            !a2.allFinite() ||
            !center.allFinite() ||
            !std::isfinite(b1) ||
            !std::isfinite(b2)) {
            throw std::invalid_argument(
                "Half-space implication received non-finite data."
            );
        }
    
        if (!std::isfinite(rho) || rho < 0.0) {
            throw std::invalid_argument(
                "Reachable radius must be finite and nonnegative."
            );
        }
    
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::invalid_argument(
                "Implication tolerance must be finite and nonnegative."
            );
        }
    
        const double norm1 = a1.norm();
        const double norm2 = a2.norm();
    
        if (!std::isfinite(norm1) ||
            !std::isfinite(norm2) ||
            norm1 <= 1e-14 ||
            norm2 <= 1e-14) {
            throw std::invalid_argument(
                "Half-space normals must be finite and nonzero."
            );
        }
    
        /*
         * Normalize
         *
         *     a_i^T p >= b_i
         *
         * to
         *
         *     n_i^T p >= beta_i,
         *     ||n_i||_2 = 1.
         */
        const Eigen::Vector2d n1 = a1 / norm1;
        const Eigen::Vector2d n2 = a2 / norm2;
    
        const double beta1 = b1 / norm1;
        const double beta2 = b2 / norm2;
    
        const double cosine = std::clamp(
            n1.dot(n2),
            -1.0,
            1.0
        );
    
        /*
         * Shift coordinates:
         *
         *     p = center + u.
         *
         * The feasible region becomes
         *
         *     ||u||_2 <= rho,
         *     n1^T u >= threshold1,
         *
         * where
         *
         *     threshold1 = beta1 - n1^T center.
         *
         * The consequent margin is
         *
         *     n2^T u + constant2,
         *
         * where
         *
         *     constant2 = n2^T center - beta2.
         */
        const double threshold1 =
            beta1 - n1.dot(center);
    
        const double constant2 =
            n2.dot(center) - beta2;
    
        /*
         * The antecedent half-space does not intersect the reachable ball.
         *
         * Although set containment would then hold vacuously, do not use an
         * infeasible antecedent to prune another scenario.
         */
        if (threshold1 > rho) {
            return false;
        }
    
        double minimum_margin = 0.0;
    
        /*
         * First consider the unconstrained minimizer over the ball:
         *
         *     u* = -rho n2.
         *
         * It satisfies the first half-space exactly when
         *
         *     n1^T u* = -rho (n1^T n2)
         *              = -rho cosine
         *              >= threshold1.
         */
        if (-rho * cosine >= threshold1) {
            minimum_margin =
                -rho + constant2;
        } else {
            /*
             * Otherwise the first half-space is active at the minimizer:
             *
             *     n1^T u = threshold1.
             *
             * The intersection of this line with the ball has radius
             *
             *     slice_radius
             *       = sqrt(rho^2 - threshold1^2).
             *
             * Decompose n2 into components parallel and perpendicular to n1:
             *
             *     n2 = cosine n1 + n2_perp,
             *
             * with
             *
             *     ||n2_perp|| = sqrt(1 - cosine^2).
             *
             * The minimum objective value on the slice is
             *
             *     cosine threshold1
             *       - slice_radius ||n2_perp||
             *       + constant2.
             */
            const double slice_radius_squared =
                std::max(
                    0.0,
                    rho * rho
                        - threshold1 * threshold1
                );
    
            const double perpendicular_norm_squared =
                std::max(
                    0.0,
                    1.0 - cosine * cosine
                );
    
            const double slice_radius =
                std::sqrt(slice_radius_squared);
    
            const double perpendicular_norm =
                std::sqrt(perpendicular_norm_squared);
    
            minimum_margin =
                cosine * threshold1
                - slice_radius * perpendicular_norm
                + constant2;
        }
    
        return minimum_margin >= -tolerance;
    }
    
    
    /**
     * @brief Certify that scenario s1 dominates scenario s2.
     *
     * Scenario s1 dominates s2 when every collision half-space generated by s1
     * implies the corresponding half-space generated by s2 on the selected
     * reachable disc-center ball:
     *
     *     Theta_s1 ∩ D ⊆ Theta_s2 ∩ D,
     *
     * where D is the product of the reachable balls used at all horizon steps
     * and for all ego discs.
     *
     * This is a sound sufficient condition for the feasible-set containment used
     * in de Groot et al.'s scenario-shadow definition. It is not the paper's full
     * free-space-polytope reduction: redundancies that arise only through the
     * intersection of several different half-spaces may not be detected.
     *
     * Soundness requires reachable_radius to contain every disc center that the
     * optimization problem can admit around each reference disc center.
     */
    struct CachedCollisionHalfspace {
        Eigen::Vector2d normal = Eigen::Vector2d::Zero();
        double offset = 0.0;
        bool valid = false;
    };

    struct CachedScenarioGeometry {
        // obstacle -> [step][disc]. A missing/invalid entry cannot certify dominance.
        std::map<int, std::vector<std::vector<CachedCollisionHalfspace>>> halfspaces;
    };

    struct DominanceGeometryCache {
        std::vector<std::vector<Eigen::Vector2d>> reference_disc_centers;
        std::vector<double> reachable_radii;
        std::vector<CachedScenarioGeometry> scenarios;
    };

    DominanceGeometryCache build_dominance_geometry_cache(
        const std::vector<Scenario>& scenarios,
        const std::vector<EgoState>& reference_trajectory,
        double combined_radius,
        int num_discs,
        double vehicle_length,
        double reachable_radius,
        double reachable_radius_growth_per_step
    ) {
        constexpr double coincidence_epsilon = 1e-8;
        DominanceGeometryCache cache;
        const int horizon = static_cast<int>(reference_trajectory.size());
        cache.reference_disc_centers.reserve(horizon);
        cache.reachable_radii.reserve(horizon);
        for (int k = 0; k < horizon; ++k) {
            cache.reference_disc_centers.push_back(compute_ego_disc_positions(
                reference_trajectory[k], num_discs, vehicle_length));
            cache.reachable_radii.push_back(
                reachable_radius + reachable_radius_growth_per_step * k);
        }

        cache.scenarios.reserve(scenarios.size());
        for (const Scenario& scenario : scenarios) {
            CachedScenarioGeometry geometry;
            for (const auto& [obstacle_id, trajectory] : scenario.trajectories) {
                auto& obstacle_halfspaces = geometry.halfspaces[obstacle_id];
                obstacle_halfspaces.resize(horizon);
                const int steps = std::min(
                    horizon, static_cast<int>(trajectory.steps.size()));
                for (int k = 0; k < steps; ++k) {
                    auto& step_halfspaces = obstacle_halfspaces[k];
                    step_halfspaces.resize(num_discs);
                    const Eigen::Vector2d& obstacle = trajectory.steps[k].mean;
                    if (!obstacle.allFinite()) continue;
                    for (int d = 0; d < num_discs; ++d) {
                        const Eigen::Vector2d displacement =
                            cache.reference_disc_centers[k][d] - obstacle;
                        const double distance = displacement.norm();
                        if (distance <= coincidence_epsilon) continue;
                        auto& halfspace = step_halfspaces[d];
                        halfspace.normal = displacement / distance;
                        halfspace.offset = halfspace.normal.dot(obstacle) + combined_radius;
                        halfspace.valid = true;
                    }
                }
            }
            cache.scenarios.push_back(std::move(geometry));
        }
        return cache;
    }

    bool scenario_dominates(
        const CachedScenarioGeometry& s1,
        const CachedScenarioGeometry& s2,
        const DominanceGeometryCache& cache
    ) {
        if (s1.halfspaces.size() != s2.halfspaces.size()) return false;
        for (const auto& [obstacle_id, halfspaces1] : s1.halfspaces) {
            const auto s2_it = s2.halfspaces.find(obstacle_id);
            if (s2_it == s2.halfspaces.end()) return false;
            const auto& halfspaces2 = s2_it->second;
            if (halfspaces1.size() != cache.reference_disc_centers.size() ||
                halfspaces2.size() != cache.reference_disc_centers.size()) {
                return false;
            }
            for (size_t k = 0; k < halfspaces1.size(); ++k) {
                const auto& step1 = halfspaces1[k];
                const auto& step2 = halfspaces2[k];
                const auto& disc_centers = cache.reference_disc_centers[k];
                if (step1.size() != disc_centers.size() ||
                    step2.size() != disc_centers.size()) {
                    return false;
                }
                for (size_t d = 0; d < disc_centers.size(); ++d) {
                    if (!step1[d].valid || !step2[d].valid ||
                        !certifies_halfspace_implication_on_ball(
                            step1[d].normal, step1[d].offset,
                            step2[d].normal, step2[d].offset,
                            disc_centers[d], cache.reachable_radii[k], 1e-9)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    
    }  // anonymous namespace
    
    
    /**
     * @brief Remove scenarios certified as dominated on the reachable domain.
     *
     * Every successful comparison certifies genuine feasible-set containment on
     * the same bounded reachable domain. Consequently, if a scenario that served
     * as a dominator is later removed by an even stronger scenario, transitivity
     * of actual set containment preserves the validity of the earlier removal.
     *
     * Injected scenarios are excluded entirely from pruning:
     *   - they are never removed;
     *   - they are not used to remove ordinary scenarios.
     */
    std::vector<Scenario> prune_dominated_scenarios(
        const std::vector<Scenario>& scenarios,
        const std::vector<EgoState>& reference_trajectory,
        double combined_radius,
        int num_discs,
        double vehicle_length,
        double reachable_radius,
        double reachable_radius_growth_per_step
    ) {
        if (scenarios.size() <= 1) {
            return scenarios;
        }
    
        if (reference_trajectory.empty()) {
            return scenarios;
        }
    
        if (num_discs <= 0) {
            throw std::invalid_argument(
                "num_discs must be positive."
            );
        }
    
        if (!std::isfinite(vehicle_length) ||
            vehicle_length < 0.0) {
            throw std::invalid_argument(
                "vehicle_length must be finite and nonnegative."
            );
        }
    
        if (!std::isfinite(combined_radius) ||
            combined_radius < 0.0) {
            throw std::invalid_argument(
                "combined_radius must be finite and nonnegative."
            );
        }
    
        if (!std::isfinite(reachable_radius) ||
            reachable_radius < 0.0) {
            throw std::invalid_argument(
                "reachable_radius must be finite and nonnegative."
            );
        }
    
        if (!std::isfinite(reachable_radius_growth_per_step) ||
            reachable_radius_growth_per_step < 0.0) {
            throw std::invalid_argument(
                "reachable_radius_growth_per_step must be finite and nonnegative."
            );
        }
    
        const int scenario_count =
            static_cast<int>(scenarios.size());

        // Every ordered comparison uses the same reference discs and each
        // scenario's frozen obstacle half-spaces. Build them once instead of
        // reconstructing O(N * obstacles * discs) geometry for every pair.
        const DominanceGeometryCache geometry_cache =
            build_dominance_geometry_cache(
                scenarios, reference_trajectory, combined_radius, num_discs,
                vehicle_length, reachable_radius,
                reachable_radius_growth_per_step);
    
        std::set<int> dominated_indices;
    
        for (int i = 0; i < scenario_count; ++i) {
            // A scenario already known to be dominated is not reused as a dominator.
            if (dominated_indices.count(i) != 0) {
                continue;
            }
    
            for (int j = i + 1;
                 j < scenario_count;
                 ++j) {
                if (dominated_indices.count(j) != 0) {
                    continue;
                }
    
                const bool i_dominates_j =
                    scenario_dominates(
                        geometry_cache.scenarios[i],
                        geometry_cache.scenarios[j],
                        geometry_cache
                    );
    
                const bool j_dominates_i =
                    scenario_dominates(
                        geometry_cache.scenarios[j],
                        geometry_cache.scenarios[i],
                        geometry_cache
                    );
    
                if (i_dominates_j) {
                    dominated_indices.insert(j);
                } else if (j_dominates_i) {
                    dominated_indices.insert(i);
    
                    // Scenario i has been removed, so stop comparing it.
                    break;
                }
            }
        }
    
        std::vector<Scenario> retained_scenarios;
    
        retained_scenarios.reserve(
            scenarios.size()
            - dominated_indices.size()
        );
    
        for (int i = 0;
             i < scenario_count;
             ++i) {
            if (dominated_indices.count(i) == 0) {
                retained_scenarios.push_back(
                    scenarios[i]
                );
            }
        }
    
        return retained_scenarios;
    }


namespace {

    
    /**
     * Project the ego state translationally onto one safe collision half-space.
     *
     * Safe half-space:
     *
     *     H = {c_d : a^T c_d >= b}.
     *
     * CollisionConstraint::evaluate(c_d) is assumed to return
     *
     *     a^T c_d - b.
     *
     * The heading is held fixed. Since translating the vehicle center by delta
     * translates every disc center by the same delta, the closest translational
     * correction is
     *
     *     delta = -evaluate(c_d) / ||a||^2 * a.
     *
     * @return true if the state position was changed.
     */
    bool project_state_to_collision_halfspace(
        EgoState& state,
        const CollisionConstraint& constraint,
        double tolerance = 1e-9
    ) {
        if (!constraint.a.allFinite()) {
            throw std::invalid_argument(
                "Collision constraint contains a non-finite normal."
            );
        }
    
        if (!std::isfinite(constraint.b)) {
            throw std::invalid_argument(
                "Collision constraint contains a non-finite offset."
            );
        }
    
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::invalid_argument(
                "Projection tolerance must be finite and nonnegative."
            );
        }
    
        const double normal_norm_squared =
            constraint.a.squaredNorm();
    
        if (!std::isfinite(normal_norm_squared) ||
            normal_norm_squared <= 1e-16) {
            throw std::invalid_argument(
                "Collision constraint has a degenerate normal."
            );
        }
    
        const Eigen::Vector2d disc_center =
            compute_collision_disc_center(state, constraint);
    
        if (!disc_center.allFinite()) {
            throw std::runtime_error(
                "Computed ego disc center is non-finite."
            );
        }
    
        // Safe when signed_clearance >= 0.
        const double signed_clearance =
            constraint.evaluate(disc_center);
    
        if (!std::isfinite(signed_clearance)) {
            throw std::runtime_error(
                "Collision constraint evaluation is non-finite."
            );
        }
    
        // Already feasible within tolerance.
        if (signed_clearance >= -tolerance) {
            return false;
        }
    
        /*
         * Closest translation satisfying this individual half-space:
         *
         *     correction =
         *         -(a^T c_d - b) / ||a||^2 * a.
         *
         * Since signed_clearance < 0, this moves in the +a direction.
         */
        const Eigen::Vector2d correction =
            -(signed_clearance / normal_norm_squared)
            * constraint.a;
    
        state.x += correction.x();
        state.y += correction.y();
    
        return true;
    }
    
    }  // anonymous namespace


    int project_warmstart_to_safety(
        std::vector<EgoState>& trajectory,
        const std::vector<CollisionConstraint>& constraints,
        int max_projection_sweeps,
        double tolerance
    ) {
        if (max_projection_sweeps <= 0) {
            throw std::invalid_argument(
                "max_projection_sweeps must be positive."
            );
        }
    
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::invalid_argument(
                "Projection tolerance must be finite and nonnegative."
            );
        }
    
        // Group valid constraints by prediction step.
        std::map<int, std::vector<const CollisionConstraint*>>
            constraints_by_step;
    
        for (const CollisionConstraint& constraint : constraints) {
            if (constraint.k < 0 ||
                constraint.k >= static_cast<int>(trajectory.size())) {
                continue;
            }
    
            constraints_by_step[constraint.k].push_back(&constraint);
        }
    
        int projection_count = 0;
    
        for (const auto& [k, step_constraints] :
             constraints_by_step) {
            EgoState& state = trajectory[k];
    
            for (int sweep = 0;
                 sweep < max_projection_sweeps;
                 ++sweep) {
                bool any_projection = false;
    
                /*
                 * Cyclic projection:
                 *
                 * Projecting onto one half-space may cause another half-space to
                 * become violated, so repeat the complete set until no correction
                 * is needed or the sweep limit is reached.
                 */
                for (const CollisionConstraint* constraint :
                     step_constraints) {
                    if (constraint == nullptr) {
                        continue;
                    }
    
                    if (project_state_to_collision_halfspace(
                            state,
                            *constraint,
                            tolerance)) {
                        any_projection = true;
                        ++projection_count;
                    }
                }
    
                if (!any_projection) {
                    break;
                }
            }
        }
    
        return projection_count;
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
    if (!std::isfinite(direction_epsilon) || direction_epsilon <= 0.0) {
        throw std::invalid_argument("direction_epsilon must be finite and positive.");
    }
    LinearizedCollisionHalfspace hs;
    hs.obstacle_position = obstacle_position;
    hs.reference_disc_center = reference_disc_center;
    hs.safety_radius = safety_radius;

    const Eigen::Vector2d delta = obstacle_position - reference_disc_center;
    const double dist = delta.norm();
    hs.reference_distance = dist;
    // hs.normal is the half-space normal n = (x_obs - c)/||.|| (ego disc center ->
    // obstacle); the geometric half-space is  n^T c <= n^T x_obs - R. NOTE: a supplied
    // fallback_normal must ALSO use this n convention (ego -> obstacle), NOT the stored
    // a = -n convention. The fallback is used only for the degenerate coincident case,
    // and only when it is itself finite and non-degenerate (never normalize a ~0 vector).
    if (dist > direction_epsilon) {
        hs.normal = delta / dist;
        hs.used_fallback_normal = false;
    } else if (fallback_normal.has_value() &&
               fallback_normal->allFinite() &&
               fallback_normal->norm() > direction_epsilon) {
        hs.normal = fallback_normal->normalized();
        hs.used_fallback_normal = true;
    } else {
        hs.normal = Eigen::Vector2d::UnitX();
        hs.used_fallback_normal = true;
    }
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
    const EgoState reference_state(
        reference_px, reference_py, reference_heading, 0.0);
    const auto disc = linearize_disc_center(reference_state, ell);
#ifndef NDEBUG
    // The reference pose + offset MUST reconstruct halfspace.reference_disc_center,
    // or the affine offset below is built about an inconsistent linearization point.
    {
        assert((disc.center - halfspace.reference_disc_center).norm() <= 1e-6 &&
               "linearize_disc_halfspace: (reference pose, offset) inconsistent with "
               "halfspace.reference_disc_center");
    }
#endif
    AffineDiscConstraint out;
    out.coefficients = halfspace.normal.transpose() * disc.jacobian;   // n^T J_d  (row 1x3)
    const Eigen::Vector3d xbar(reference_px, reference_py, reference_heading);
    // normal^T c_d <= ub  =>  (n^T J) x <= ub - n^T c_bar + (n^T J) x_bar.
    out.upper_bound = halfspace.upper_bound
                      - halfspace.normal.dot(halfspace.reference_disc_center)
                      + out.coefficients.dot(xbar);
    return out;
}

DiscConstraintRow linearize_constraint_at_state(
    const CollisionConstraint& constraint,
    const EgoState& state
) {
    const auto disc = linearize_disc_center(state, constraint.disc_offset);

    DiscConstraintRow row;
    row.value = constraint.evaluate(disc.center);
    row.gradient = constraint.a.transpose() * disc.jacobian;
    return row;
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
    c.disc_offset = halfspace.disc_offset;   // keep the conversion self-contained
    return c;
}

}  // namespace dro_mpc
