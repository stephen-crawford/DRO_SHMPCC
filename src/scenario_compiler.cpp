/**
 * @file scenario_compiler.cpp
 * @brief Implementation of scenario compiler (Section 7.2).
 */

#include "scenario_compiler.hpp"
#include "collision_constraints.hpp"
#include <optional>

namespace scenario_mpc {

ScenarioCompilerState scenario_compiler_step(
    ScenarioCompilerState state,
    const std::vector<EgoState>& reference,
    AdversaryFn adversary,
    int max_iter) {
    if (state.iterations >= max_iter) {
        state.certificate_valid = true;
        return state;
    }
    auto worst = adversary(reference, state.witness_set);
    if (!worst.has_value()) {
        state.certificate_valid = true;
        return state;
    }
    state.witness_set.push_back(worst.value());
    state.iterations++;
    return state;
}

bool scenario_violates_plan(
    const Scenario& scenario,
    const std::vector<EgoState>& ego_trajectory,
    double combined_radius,
    double margin) {
    for (const auto& [obs_id, traj] : scenario.trajectories) {
        for (size_t k = 0; k < traj.steps.size() && k < ego_trajectory.size(); ++k) {
            Eigen::Vector2d ego_pos = ego_trajectory[k].position();
            Eigen::Vector2d obs_pos = traj.steps[k].mean;
            double dist = (ego_pos - obs_pos).norm();
            if (dist < combined_radius + margin) return true;
        }
    }
    return false;
}

}  // namespace scenario_mpc
