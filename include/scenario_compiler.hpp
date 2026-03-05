/**
 * @file scenario_compiler.hpp
 * @brief Scenario compiler: minimal witness set + constraint generation (Section 7.2).
 */

#ifndef SCENARIO_MPC_SCENARIO_COMPILER_HPP
#define SCENARIO_MPC_SCENARIO_COMPILER_HPP

#include "types.hpp"
#include <vector>
#include <functional>

namespace scenario_mpc {

/**
 * @brief Adversary: given plan x (ego trajectory), return scenario that maximizes violation.
 * Signature: (reference_trajectory, current_witness_scenarios) -> worst_case_scenario or nullopt.
 */
using AdversaryFn = std::function<std::optional<Scenario>(
    const std::vector<EgoState>&,
    const std::vector<Scenario>&
)>;

/**
 * @brief Scenario compiler state: witness set and iteration count.
 */
struct ScenarioCompilerState {
    std::vector<Scenario> witness_set;
    int iterations = 0;
    bool certificate_valid = false;
};

/**
 * @brief One step of constraint generation: find counterexample, add to witness set.
 *
 * @param state        Current compiler state (witness set, etc.)
 * @param reference    Ego reference trajectory
 * @param adversary    Function that returns worst-case scenario
 * @param max_iter     Max constraint generation steps
 * @return Updated state; certificate_valid if no counterexample found
 */
ScenarioCompilerState scenario_compiler_step(
    ScenarioCompilerState state,
    const std::vector<EgoState>& reference,
    AdversaryFn adversary,
    int max_iter = 10
);

/**
 * @brief Check if a scenario violates the plan (placeholder: requires constraint eval).
 * Used by adversary to score scenarios.
 */
bool scenario_violates_plan(
    const Scenario& scenario,
    const std::vector<EgoState>& ego_trajectory,
    double combined_radius,
    double margin = 0.0
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_SCENARIO_COMPILER_HPP
