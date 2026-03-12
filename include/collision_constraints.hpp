/**
 * @file collision_constraints.hpp
 * @brief Linearized collision avoidance constraints.
 *
 * Implements Section 7: Linearized Collision Constraints
 *
 * The key equations are:
 * - Eq. 17: Direction vector computation
 * - Eq. 18: Linearized constraint formulation
 *
 * Constraints are linearized around a reference trajectory to enable QP/SQP solving.
 */

#ifndef SCENARIO_MPC_COLLISION_CONSTRAINTS_HPP
#define SCENARIO_MPC_COLLISION_CONSTRAINTS_HPP

#include "types.hpp"
#include <optional>

namespace scenario_mpc {

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
 * @brief Compute Jacobian of constraint w.r.t. decision variables.
 *
 * For constraint: a^T @ p_ego >= b
 * The Jacobian w.r.t. position is simply a^T.
 *
 * @param constraint The collision constraint
 * @param state Current ego state
 * @param dynamics_jacobian Jacobian of position w.r.t. state
 * @return Constraint Jacobian (1, state_dim)
 */
Eigen::Vector4d compute_constraint_jacobian(
    const CollisionConstraint& constraint,
    const EgoState& state,
    const Eigen::Matrix4d& dynamics_jacobian
);

/**
 * @brief Filter out constraints for obstacles that are too far away.
 *
 * @param constraints All collision constraints
 * @param ego_trajectory Reference ego trajectory
 * @param max_distance Maximum distance to consider
 * @return Filtered list of constraints
 */
std::vector<CollisionConstraint> filter_constraints_by_distance(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& ego_trajectory,
    double max_distance = 50.0
);

/**
 * @brief Merge nearly identical constraints to reduce problem size.
 *
 * Two constraints are merged if their normal vectors are nearly parallel
 * and their offsets are similar.
 *
 * @param constraints List of constraints
 * @param angle_threshold Maximum angle difference (radians) for merging
 * @param offset_threshold Maximum offset difference for merging
 * @return Reduced list of constraints
 */
std::vector<CollisionConstraint> merge_redundant_constraints(
    const std::vector<CollisionConstraint>& constraints,
    double angle_threshold = 0.1,
    double offset_threshold = 0.5
);

/**
 * @brief Douglas-Rachford projection for collision avoidance.
 *
 * Projects a position onto the collision-free region outside a circle
 * of given radius centered at the obstacle. Uses the DR splitting
 * method from the Python reference (utils/math_tools_impl.py).
 *
 * @param position Point to project (modified in-place)
 * @param obstacle Obstacle center position
 * @param radius Combined collision radius (ego + obstacle + margin)
 * @param starting_pose Starting pose for projection direction
 */
void douglas_rachford_projection(
    Eigen::Vector2d& position,
    const Eigen::Vector2d& obstacle,
    double radius,
    const Eigen::Vector2d& starting_pose
);

/**
 * @brief Project a warmstart trajectory to satisfy collision constraints.
 *
 * For each timestep, if the warmstart position violates any collision
 * constraint, project it away from the obstacle using Douglas-Rachford.
 * This ensures the solver warmstart is feasible, preventing solver
 * failures. Reference: Python GaussianConstraints._project_warmstart_to_gaussian_safety
 *
 * @param trajectory Warmstart trajectory (modified in-place)
 * @param constraints Collision constraints to satisfy
 * @param ego_radius Ego collision radius
 * @param obstacle_radius Obstacle collision radius
 * @param safety_margin Additional margin
 * @return Number of positions projected
 */
int project_warmstart_to_safety(
    std::vector<EgoState>& trajectory,
    const std::vector<CollisionConstraint>& constraints,
    double ego_radius,
    double obstacle_radius,
    double safety_margin = 0.0
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_COLLISION_CONSTRAINTS_HPP
