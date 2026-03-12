/**
 * @file scenario_pruning.hpp
 * @brief Scenario pruning for efficient optimization.
 *
 * Implements Algorithms 3 and 4:
 * - Algorithm 3: Geometric dominance pruning
 * - Algorithm 4: Support-based scenario removal
 *
 * These algorithms reduce the number of scenarios while maintaining
 * chance constraint guarantees.
 */

#ifndef SCENARIO_MPC_SCENARIO_PRUNING_HPP
#define SCENARIO_MPC_SCENARIO_PRUNING_HPP

#include "types.hpp"
#include <set>

namespace scenario_mpc {

/**
 * @brief Prune geometrically dominated scenarios.
 *
 * Algorithm 3: GeometricDominancePruning
 *
 * A scenario s1 dominates s2 if for all timesteps k and obstacles o,
 * the obstacle position in s1 is further from the ego than in s2.
 *
 * @param scenarios List of scenarios to prune
 * @param reference_trajectory Reference ego trajectory
 * @param ego_radius Ego collision radius
 * @param obstacle_radius Obstacle collision radius
 * @return Pruned list of non-dominated scenarios
 */
std::vector<Scenario> prune_dominated_scenarios(
    const std::vector<Scenario>& scenarios,
    const std::vector<EgoState>& reference_trajectory,
    double ego_radius,
    double obstacle_radius
);

/**
 * @brief Remove scenarios with inactive constraints.
 *
 * Algorithm 4: SupportBasedRemoval
 *
 * After solving the optimization, identify scenarios whose constraints
 * are not active (not binding) and remove them.
 *
 * A constraint is active if it is satisfied with equality (within tolerance).
 *
 * @param scenarios Current list of scenarios
 * @param constraints Collision constraints for all scenarios
 * @param optimal_trajectory Optimal ego trajectory from solver
 * @param tolerance Tolerance for checking constraint activity
 * @return Pair of (remaining_scenarios, active_scenario_ids)
 */
std::pair<std::vector<Scenario>, std::set<int>> remove_inactive_scenarios(
    const std::vector<Scenario>& scenarios,
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& optimal_trajectory,
    double tolerance = 1e-4
);

/**
 * @brief Remove scenarios with very low probability.
 *
 * Simple pruning that removes scenarios unlikely to occur.
 *
 * @param scenarios List of scenarios
 * @param min_probability Minimum probability threshold
 * @return Filtered scenarios
 */
std::vector<Scenario> prune_by_probability(
    const std::vector<Scenario>& scenarios,
    double min_probability = 0.01
);

/**
 * @brief Cluster similar scenarios and keep representative ones.
 *
 * Scenarios are similar if their obstacle trajectories are close
 * across the horizon.
 *
 * @param scenarios List of scenarios
 * @param distance_threshold Maximum distance for clustering
 * @param horizon Prediction horizon (inferred if -1)
 * @return Representative scenarios (cluster centroids)
 */
std::vector<Scenario> cluster_similar_scenarios(
    const std::vector<Scenario>& scenarios,
    double distance_threshold = 1.0,
    int horizon = -1
);

/**
 * @brief Select a diverse subset of scenarios.
 *
 * Uses farthest point sampling to maximize diversity.
 *
 * @param scenarios All scenarios
 * @param num_select Number of scenarios to select
 * @param horizon Prediction horizon
 * @return Diverse subset of scenarios
 */
std::vector<Scenario> select_diverse_scenarios(
    const std::vector<Scenario>& scenarios,
    int num_select,
    int horizon = -1
);

/**
 * @brief Adaptively adjust scenario budget based on constraint violations.
 *
 * If constraints are frequently violated, increase scenarios.
 * If constraints are easily satisfied, decrease scenarios.
 *
 * @param current_scenarios Current number of scenarios
 * @param constraint_violations Sum of constraint violations
 * @param max_scenarios Maximum allowed scenarios
 * @param min_scenarios Minimum scenarios to maintain
 * @param growth_factor Factor to increase scenarios
 * @param shrink_factor Factor to decrease scenarios
 * @return New scenario budget
 */
int adaptive_scenario_budget(
    int current_scenarios,
    double constraint_violations,
    int max_scenarios = 100,
    int min_scenarios = 5,
    double growth_factor = 1.5,
    double shrink_factor = 0.8
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_SCENARIO_PRUNING_HPP
