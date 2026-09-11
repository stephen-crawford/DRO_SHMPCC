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
#include <limits>
#include <iostream>

namespace dro_mpc {

    namespace {

        double minimum_sampled_anchor_clearance(
            const std::vector<EgoState>& trajectory,
            const std::vector<Scenario>& scenarios,
            double combined_radius,
            int num_discs,
            double vehicle_length)
        {
            double min_clearance =
                std::numeric_limits<double>::infinity();

            for (size_t k = 1; k < trajectory.size(); ++k) {
                const auto discs =
                    compute_ego_disc_positions(
                        trajectory[k],
                        num_discs,
                        vehicle_length);

                for (const auto& scenario : scenarios) {
                    for (const auto& [obstacle_id, prediction] :
                        scenario.trajectories) {

                        if (k >= prediction.steps.size())
                            continue;

                        const Eigen::Vector2d obstacle =
                            prediction.steps[k].mean;

                        for (const auto& disc : discs) {
                            const double clearance =
                                (disc - obstacle).norm()
                                - combined_radius;

                            min_clearance =
                                std::min(
                                    min_clearance,
                                    clearance);
                        }
                    }
                }
            }

            return min_clearance;
        }
        
        std::vector<Eigen::Vector2d> make_box(
            const Eigen::Vector2d& center,
            double half_extent)
        {
            return {
                center + Eigen::Vector2d(-half_extent, -half_extent),
                center + Eigen::Vector2d( half_extent, -half_extent),
                center + Eigen::Vector2d( half_extent,  half_extent),
                center + Eigen::Vector2d(-half_extent,  half_extent)
            };
        }

        std::vector<Eigen::Vector2d> clip_polygon_against_halfspace(
            const std::vector<Eigen::Vector2d>& polygon,
            const CollisionConstraint& constraint,
            double tolerance = 1e-10)
        {
            std::vector<Eigen::Vector2d> output;

            if (polygon.empty()) {
                return output;
            }

            auto clearance = [&](const Eigen::Vector2d& p) {
                return constraint.a.dot(p) - constraint.b;
            };

            for (size_t i = 0; i < polygon.size(); ++i) {
                const Eigen::Vector2d& start = polygon[i];
                const Eigen::Vector2d& end =
                    polygon[(i + 1) % polygon.size()];

                const double f_start = clearance(start);
                const double f_end = clearance(end);

                const bool start_inside = f_start >= -tolerance;
                const bool end_inside = f_end >= -tolerance;

                if (start_inside && end_inside) {
                    // Entire edge remains inside.
                    output.push_back(end);
                }
                else if (start_inside && !end_inside) {
                    // Leaving the feasible half-space.
                    const double denom = f_start - f_end;

                    if (std::abs(denom) > 1e-14) {
                        const double t =
                            std::clamp(f_start / denom, 0.0, 1.0);

                        output.push_back(
                            start + t * (end - start));
                    }
                }
                else if (!start_inside && end_inside) {
                    // Entering the feasible half-space.
                    const double denom = f_start - f_end;

                    if (std::abs(denom) > 1e-14) {
                        const double t =
                            std::clamp(f_start / denom, 0.0, 1.0);

                        output.push_back(
                            start + t * (end - start));
                    }

                    output.push_back(end);
                }

                // both outside -> add nothing
            }

            return output;
        }

        std::vector<int> find_polygon_facet_constraints(
            const std::vector<Eigen::Vector2d>& polygon,
            const std::vector<const CollisionConstraint*>& constraints,
            double facet_tolerance = 1e-7)
        {
            std::set<int> active_indices;

            if (polygon.size() < 2) {
                return {};
            }

            for (size_t edge = 0; edge < polygon.size(); ++edge) {
                const Eigen::Vector2d& p0 = polygon[edge];
                const Eigen::Vector2d& p1 =
                    polygon[(edge + 1) % polygon.size()];

                if ((p1 - p0).norm() < 1e-9) {
                    continue;
                }

                int best_index = -1;
                double best_error =
                    std::numeric_limits<double>::infinity();

                for (int i = 0;
                    i < static_cast<int>(constraints.size());
                    ++i) {

                    const auto* constraint = constraints[i];

                    const double normal_norm =
                        constraint->a.norm();

                    if (normal_norm <= 1e-14) {
                        continue;
                    }

                    const double r0 =
                        std::abs(
                            constraint->a.dot(p0)
                            - constraint->b)
                        / normal_norm;

                    const double r1 =
                        std::abs(
                            constraint->a.dot(p1)
                            - constraint->b)
                        / normal_norm;

                    const double error =
                        std::max(r0, r1);

                    if (error < best_error) {
                        best_error = error;
                        best_index = i;
                    }
                }

                // If this edge lies on one of the sampled collision
                // halfspaces, that halfspace is a polygon facet.
                //
                // Box edges will generally not match any collision
                // constraint and are therefore ignored here.
                if (best_index >= 0 &&
                    best_error <= facet_tolerance) {
                    active_indices.insert(best_index);
                }
            }

            return {
                active_indices.begin(),
                active_indices.end()
            };
        }

    
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
        // A caller may provide a previous valid normal to preserve orientation at
        // a coincident linearization point.
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


    bool prepare_safe_horizon_anchors(
        std::vector<EgoState>& trajectory,
        const std::vector<Scenario>& scenarios,
        double combined_radius, int num_discs, double vehicle_length
    ) {
        // Reference: scenario_module SafeHorizon::PushAlgorithm / DRProjection.
        // Each disc module prepares all stages before the next disc is processed.
        for (int disc = 0; disc < num_discs; ++disc) {
            const double offset = get_disc_longitudinal_offset(disc, num_discs, vehicle_length);
            for (size_t k = 1; k < trajectory.size(); ++k) {
                auto& state = trajectory[k];
                Eigen::Vector2d pose = linearize_disc_center(state, offset).center;
                std::map<int, std::vector<Eigen::Vector2d>> samples;
                for (const auto& scenario : scenarios) {
                    for (const auto& [id, prediction] : scenario.trajectories) {
                        if (k < prediction.steps.size()) samples[id].push_back(prediction.steps[k].mean);
                    }
                }
                const Eigen::Vector2d lateral(-std::sin(state.theta), std::cos(state.theta));
                for (const auto& [id, points] : samples) {
                    Eigen::Vector2d outward = Eigen::Vector2d::Zero();
                    for (size_t s = 0; s < points.size(); s += 5) outward += pose - points[s];
                    const Eigen::Vector2d direction = outward.dot(lateral) >= 0.0 ? lateral : -lateral;
                    double push = 0.0;
                    for (const auto& point : points) {
                        if ((pose - point).norm() < combined_radius)
                            push = std::max(push, direction.dot(point - pose) + combined_radius);
                    }
                    pose += push * direction;
                }
                state.x = pose.x() - offset * std::cos(state.theta);
                state.y = pose.y() - offset * std::sin(state.theta);
            }
            for (size_t k = 1; k < trajectory.size(); ++k) {
                auto& state = trajectory[k];
                Eigen::Vector2d pose = linearize_disc_center(state, offset).center;
                const Eigen::Vector2d start = k > 1
                    ? linearize_disc_center(trajectory[k - 1], offset).center : pose;
                std::map<int, std::vector<Eigen::Vector2d>> samples;
                for (const auto& scenario : scenarios) {
                    for (const auto& [id, prediction] : scenario.trajectories) {
                        if (k < prediction.steps.size()) samples[id].push_back(prediction.steps[k].mean);
                    }
                }
                if (samples.empty()) continue;
                const Eigen::Vector2d anchor = samples.begin()->second.front();
                const double radius = combined_radius + 1e-3;
                const Eigen::Vector2d lateral(-std::sin(state.theta), std::cos(state.theta));
                // Deterministic direction at a circle center, where the reference's
                // normalization is undefined. Used only for geometric preparation.
                auto radial = [&](const Eigen::Vector2d& delta) -> Eigen::Vector2d {
                    const double distance = delta.norm();
                    if (distance == 0.0) return lateral;
                    return delta / distance;
                };
                for (int iteration = 0; iteration < 25; ++iteration) {
                    const Eigen::Vector2d previous = pose;
                    for (const auto& [id, points] : samples) {
                        for (size_t s = 0; s < points.size(); ++s) {
                            if (id == samples.begin()->first && s == 0) continue;
                            Eigen::Vector2d update = pose;
                            if ((update - anchor).norm() < radius)
                                update = 2.0 * (anchor + radius * radial(update - anchor)) - update;
                            if ((update - points[s]).norm() < radius)
                                update = 2.0 * (points[s] + radius * radial(update - points[s])) - update;
                            pose = 0.5 * (pose + update);
                        }
                    }
                    if ((previous - pose).norm() < 1e-5) break;
                }
                state.x = pose.x() - offset * std::cos(state.theta);
                state.y = pose.y() - offset * std::sin(state.theta);
            }
            }

         constexpr int kMaxRepairSweeps = 20;
    constexpr double kRepairMargin = 2e-3;
    constexpr double kDirectionEps = 1e-10;

    const double target_radius =
        combined_radius + kRepairMargin;

    for (int sweep = 0;
         sweep < kMaxRepairSweeps;
         ++sweep) {

        bool repaired_any = false;

        for (size_t k = 1;
             k < trajectory.size();
             ++k) {

            auto& state = trajectory[k];

            const Eigen::Vector2d lateral(
                -std::sin(state.theta),
                 std::cos(state.theta));

            for (int disc = 0;
                 disc < num_discs;
                 ++disc) {

                const double offset =
                    get_disc_longitudinal_offset(
                        disc,
                        num_discs,
                        vehicle_length);

                for (const auto& scenario : scenarios) {
                    for (const auto& [obstacle_id, prediction] :
                         scenario.trajectories) {

                        (void)obstacle_id;

                        if (k >= prediction.steps.size()) {
                            continue;
                        }

                        const Eigen::Vector2d obstacle =
                            prediction.steps[k].mean;

                        Eigen::Vector2d disc_center =
                            linearize_disc_center(
                                state,
                                offset).center;

                        Eigen::Vector2d delta =
                            disc_center - obstacle;

                        const double distance =
                            delta.norm();

                        if (distance >= target_radius) {
                            continue;
                        }

                        Eigen::Vector2d direction;

                        if (distance > kDirectionEps) {
                            direction =
                                delta / distance;
                        } else {
                            direction =
                                lateral;
                        }

                        const Eigen::Vector2d correction =
                            (target_radius - distance)
                            * direction;

                        /*
                         * Translating the vehicle center by correction
                         * translates every ego disc by the same amount.
                         */
                        state.x += correction.x();
                        state.y += correction.y();

                        repaired_any = true;
                    }
                }
            }
        }

        const double repair_clearance =
            minimum_sampled_anchor_clearance(
                trajectory,
                scenarios,
                combined_radius,
                num_discs,
                vehicle_length);

        std::cerr
            << "[SH ANCHOR REPAIR]"
            << " sweep=" << sweep
            << " min_clearance=" << repair_clearance
            << " repaired=" << repaired_any
            << std::endl;

        if (repair_clearance >= 0.0) {
            break;
        }

        if (!repaired_any) {
            break;
        }
    }

        const double min_clearance =
    minimum_sampled_anchor_clearance(
        trajectory,
        scenarios,
        combined_radius,
        num_discs,
        vehicle_length);

        const bool feasible = min_clearance >= -1e-6;

        std::cerr
            << "[SH ANCHOR]"
            << " min_clearance=" << min_clearance
            << " feasible=" << feasible
            << std::endl;

        return feasible;
    }


    std::vector<CollisionConstraint> reduce_to_free_space_polytopes(
    const std::vector<CollisionConstraint>& all_constraints,
    const std::vector<EgoState>& anchor_reference,
    const std::vector<EgoState>& dynamic_reference,
    int num_discs,
    double vehicle_length,
    double max_abs_velocity,
    double dt,
    int max_facets)
    {
        if (anchor_reference.size() != dynamic_reference.size()) {
            throw std::invalid_argument(
                "anchor_reference and dynamic_reference must have equal size.");
        }

        if (num_discs <= 0) {
            throw std::invalid_argument("num_discs must be positive.");
        }

        if (!std::isfinite(vehicle_length) || vehicle_length < 0.0) {
            throw std::invalid_argument(
                "vehicle_length must be finite and nonnegative.");
        }

        if (!std::isfinite(max_abs_velocity) || max_abs_velocity < 0.0) {
            throw std::invalid_argument(
                "max_abs_velocity must be finite and nonnegative.");
        }

        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::invalid_argument(
                "dt must be finite and positive.");
        }

        if (max_facets <= 0) {
            throw std::invalid_argument(
                "max_facets must be positive.");
        }

        using GroupKey = std::pair<int, int>;  // (k, disc)

        std::map<GroupKey,
                std::vector<const CollisionConstraint*>> groups;

        // Group all sampled halfspaces by stage and ego disc.
        for (const auto& constraint : all_constraints) {
            if (constraint.k <= 0) {
                continue;
            }

            groups[{constraint.k, constraint.disc_index}]
                .push_back(&constraint);
        }

        std::vector<CollisionConstraint> reduced;

        constexpr double kReachabilityMargin = 1e-6;

        for (const auto& [key, group] : groups) {
            const int k = key.first;
            const int disc = key.second;

            if (k >= static_cast<int>(anchor_reference.size()) ||
                k >= static_cast<int>(dynamic_reference.size())) {
                continue;
            }

            const auto anchor_discs =
                compute_ego_disc_positions(
                    anchor_reference[k],
                    num_discs,
                    vehicle_length);

            const auto dynamic_discs =
                compute_ego_disc_positions(
                    dynamic_reference[k],
                    num_discs,
                    vehicle_length);

            if (disc < 0 ||
                disc >= static_cast<int>(anchor_discs.size()) ||
                disc >= static_cast<int>(dynamic_discs.size())) {
                throw std::runtime_error(
                    "Invalid disc index in free-space polygon construction.");
            }

            const Eigen::Vector2d center =
                anchor_discs[disc];

            // Geometric anchor translation relative to the dynamically
            // consistent reference trajectory.
            const double anchor_shift =
                (anchor_discs[disc] -
                dynamic_discs[disc]).norm();

            const double ell =
                std::abs(
                    get_disc_longitudinal_offset(
                        disc,
                        num_discs,
                        vehicle_length));

            /*
            * Conservative disc-center reachability bound.
            *
            * Any admissible trajectory and the dynamically consistent
            * reference both start from the same measured state and can
            * move by at most v_max * k * dt. Thus their vehicle centers
            * can differ by at most
            *
            *     2 v_max k dt.
            *
            * Different headings can move an offset disc relative to the
            * vehicle center by at most 2 |ell|.
            *
            * Finally, anchor_shift accounts for the geometric Safe-Horizon
            * projection of the collision-linearization trajectory.
            */
            const double translational_bound =
                2.0 * max_abs_velocity *
                static_cast<double>(k) * dt;

            const double heading_bound =
                2.0 * ell;

            const double rho =
                translational_bound
                + heading_bound
                + anchor_shift
                + kReachabilityMargin;

            // The square conservatively contains the Euclidean reachable
            // ball of radius rho.
            auto polygon =
                make_box(center, rho);

            for (const auto* constraint : group) {
                polygon =
                    clip_polygon_against_halfspace(
                        polygon,
                        *constraint);

                if (polygon.empty()) {
                    throw std::runtime_error(
                        "Free-space polygon became empty at k="
                        + std::to_string(k)
                        + ", disc="
                        + std::to_string(disc));
                }
            }

            const auto facet_indices =
                find_polygon_facet_constraints(
                    polygon,
                    group);

            if (static_cast<int>(facet_indices.size()) >
                max_facets) {
                throw std::runtime_error(
                    "Free-space polygon exceeds facet limit at k="
                    + std::to_string(k)
                    + ", disc="
                    + std::to_string(disc)
                    + ": facets="
                    + std::to_string(
                        facet_indices.size()));
            }

            for (const int index : facet_indices) {
                reduced.push_back(*group[index]);
            }

            std::cerr
                << "[FREE POLY]"
                << " k=" << k
                << " disc=" << disc
                << " raw=" << group.size()
                << " facets=" << facet_indices.size()
                << " vertices=" << polygon.size()
                << " rho=" << rho
                << " anchor_shift=" << anchor_shift
                << std::endl;
        }

        return reduced;
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
}  // namespace dro_mpc
