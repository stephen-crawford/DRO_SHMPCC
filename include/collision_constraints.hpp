/**
 * @file collision_constraints.hpp
 * @brief Linearized collision avoidance constraints.
 *
 * Constraints are linearized around a reference trajectory to enable QP/SQP solving.
 */

#ifndef DRO_MPC_COLLISION_CONSTRAINTS_HPP
#define DRO_MPC_COLLISION_CONSTRAINTS_HPP

#include "types.hpp"
#include <optional>

namespace dro_mpc {

/**
 * @brief A fixed (pre-optimization) linearized collision half-space.
 *
 *     normal^T c <= upper_bound,   upper_bound = normal^T x_obs - safety_radius,
 *
 * with `normal = (x_obs - c_bar)/||x_obs - c_bar||` computed from a NUMERICAL reference
 * disc center c_bar. The normal is held FIXED during the QP solve (it never depends on the
 * optimizer decision variables), so each row is a genuine affine half-space.
 */
struct LinearizedCollisionHalfspace {
    Eigen::Vector2d normal = Eigen::Vector2d::UnitX();
    Eigen::Vector2d obstacle_position = Eigen::Vector2d::Zero();
    Eigen::Vector2d reference_disc_center = Eigen::Vector2d::Zero();
    double safety_radius = 0.0;
    double reference_distance = 0.0;
    double upper_bound = 0.0;
    int scenario_id = -1;
    int obstacle_id = -1;
    int horizon_step = -1;
    int disc_index = -1;
    double disc_offset = 0.0;   // longitudinal offset ℓ_d of this disc (for the heading Jacobian)
    bool used_fallback_normal = false;
};

/**
 * @brief Affine map of a fixed half-space through the disc-center Jacobian.
 *
 *     coefficients * [p_x, p_y, theta]^T <= upper_bound.
 */
struct AffineDiscConstraint {
    Eigen::RowVector3d coefficients = Eigen::RowVector3d::Zero();
    double upper_bound = 0.0;
};

/**
 * @brief Exact disc center and its pose Jacobian at one ego state.
 *
 * `jacobian` is d c_d / d(x, y, theta), with state order `[x, y, theta]`.
 * Keeping the center and Jacobian together guarantees every collision path
 * uses the same disc geometry when evaluating or linearizing a constraint.
 */
struct DiscCenterLinearization {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 2, 3> jacobian =
        Eigen::Matrix<double, 2, 3>::Zero();
};

/// Evaluate the exact disc center and Jacobian for longitudinal offset `ell`.
DiscCenterLinearization linearize_disc_center(
    const EgoState& state,
    double longitudinal_disc_offset
);

/// Compute the ego disc center represented by a collision constraint.
Eigen::Vector2d compute_collision_disc_center(
    const EgoState& state,
    const CollisionConstraint& constraint
);

/**
 * @brief Construct one fixed collision half-space around a numerical reference disc center.
 *
 * `normal = (x_obs - c_bar)/||x_obs - c_bar||`;  the conservative affine constraint is
 * `normal^T c <= normal^T x_obs - R`. If the obstacle and disc coincide (distance <=
 * direction_epsilon), `fallback_normal` is used (or UnitX) and used_fallback_normal is set.
 */
LinearizedCollisionHalfspace make_collision_halfspace(
    const Eigen::Vector2d& obstacle_position,
    const Eigen::Vector2d& reference_disc_center,
    double safety_radius,
    const std::optional<Eigen::Vector2d>& fallback_normal = std::nullopt,
    double direction_epsilon = 1e-6
);

/**
 * @brief Longitudinal offset of disc d along the vehicle centerline.
 * Matches the placement convention in compute_ego_disc_positions().
 */
double get_disc_longitudinal_offset(
    int disc_index,
    int num_discs,
    double vehicle_length
);

/**
 * @brief Linearize `normal^T c_d(x)` about a reference pose for heading-dependent discs.
 *
 *     c_d ≈ c_bar + J_d (x - x_bar),
 *     J_d = [[1, 0, -ℓ sin θ̄], [0, 1, ℓ cos θ̄]].
 *
 * Both this function and the live QP use linearize_disc_center(), so collision
 * evaluation and the affine row share the exact same disc geometry and Jacobian.
 * All four reference arguments must come from the SAME reference state that produced
 * `halfspace.reference_disc_center` (checked by a debug assertion in the impl).
 *
 * LIMITATION: the heading term is a FIRST-ORDER approximation of the nonlinear disc
 * map, so satisfying the affine row only guarantees ||c_d(x,y,θ) - o|| ≥ R near the
 * reference. Global nonlinear safety additionally requires a heading trust region /
 * constraint tightening / relinearization with a final nonlinear feasibility check.
 */
AffineDiscConstraint linearize_disc_halfspace(
    const LinearizedCollisionHalfspace& halfspace,
    double reference_px,
    double reference_py,
    double reference_heading,
    double longitudinal_disc_offset
);

/**
 * @brief Convert a fixed half-space into the legacy a^T p >= b form
 * (a = -normal, b = -upper_bound), so evaluate(p) = a^T p - b = clearance.
 */
CollisionConstraint halfspace_to_collision_constraint(
    const LinearizedCollisionHalfspace& halfspace
);

/**
 * @brief A stored half-space re-anchored about an ARBITRARY current state.
 *
 * value  = a^T c_d(x) - b            (signed clearance at x; the row's constant)
 * gradient = d/d(p_x, p_y, theta) of a^T c_d  =  a^T J_d(theta)
 */
struct DiscConstraintRow {
    double value = 0.0;
    Eigen::RowVector3d gradient = Eigen::RowVector3d::Zero();
};

/**
 * @brief Re-anchor a fixed collision half-space about the CURRENT state.
 *
 * An SQP row is only a valid first-order model if its constant and its gradient are
 * taken at the SAME point. The normal `a` is deliberately frozen at construction
 * (that is what keeps the row affine, and the half-space stays conservative for any
 * c by Cauchy-Schwarz), but the ANCHOR must track the iterate: the constraint value
 * at zero step is a^T c_d(x), evaluated on the exact nonlinear disc centre
 * c_d(x) = (x, y) + l_d (cos theta, sin theta).
 */
DiscConstraintRow linearize_constraint_at_state(
    const CollisionConstraint& constraint,
    const EgoState& state
);

/**
 * @brief Compute linearized collision constraints for all scenarios.
 *
 * Following Section 7:
 *
 * For each scenario s and obstacle o at timestep k:
 * 1. Compute ego disc positions p_ego (Eq. 16)
 * 2. Compute direction from obstacle to ego (Eq. 17)
 * 3. Formulate linearized constraint (Eq. 18)
 *
 * @param reference_trajectory Reference ego trajectory for linearization
 * @param scenarios List of sampled scenarios
 * @param ego_radius Ego vehicle collision radius
 * @param obstacle_radius Obstacle collision radius
 * @param safety_margin Additional safety margin
 * @param num_discs Number of discs to represent ego vehicle
 * @return List of CollisionConstraint objects
 */
std::vector<CollisionConstraint> compute_linearized_constraints(
    const std::vector<EgoState>& reference_trajectory,
    const std::vector<Scenario>& scenarios,
    double ego_radius,
    double obstacle_radius,
    double safety_margin = 0.0,
    int num_discs = 1,
    double vehicle_length = 4.0
);

/**
 * @brief Compute ego disc positions for collision checking.
 *
 * Following Eq. 16:
 * For multi-disc representation, discs are placed along the vehicle centerline.
 *
 * @param state Ego vehicle state
 * @param num_discs Number of discs
 * @param vehicle_length Vehicle length for disc placement
 * @return List of 2D positions for each disc
 */
std::vector<Eigen::Vector2d> compute_ego_disc_positions(
    const EgoState& state,
    int num_discs = 1,
    double vehicle_length = 4.0
);

/**
 * @brief Evaluate constraint violations for a trajectory.
 *
 * @param constraints List of collision constraints
 * @param ego_trajectory Ego trajectory to evaluate
 * @return Pair of (max_violation, violated_constraints)
 *         where max_violation > 0 means constraint is violated
 */
std::pair<double, std::vector<CollisionConstraint>> evaluate_constraint_violation(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& ego_trajectory
);

/**
 * @brief Filter out constraints for obstacles that are too far away.
 *
 * @param constraints All collision constraints
 * @param ego_trajectory Reference ego trajectory
 * @param max_distance Maximum distance to consider
 * @return Filtered list of constraints
 */
std::vector<CollisionConstraint> filter_constraints_by_clearance(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& ego_trajectory,
    double max_distance = 50.0
);

std::vector<CollisionConstraint> reduce_to_free_space_polytopes(
    const std::vector<CollisionConstraint>& all_constraints,
    const std::vector<EgoState>& anchor_reference,
    const std::vector<EgoState>& dynamic_reference,
    int num_discs,
    double vehicle_length,
    double max_abs_velocity,
    double dt,
    int max_facets = 20);

bool prepare_safe_horizon_anchors(
        std::vector<EgoState>& trajectory,
        const std::vector<Scenario>& scenarios,
        double combined_radius, int num_discs, double vehicle_length
    );

}  // namespace dro_mpc

#endif  // DRO_MPC_COLLISION_CONSTRAINTS_HPP
