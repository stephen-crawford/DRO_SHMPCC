/**
 * @file experiment_artifacts_internal.hpp
 * @brief Private trace representation shared by the rollout and artifact writer.
 */

#ifndef DRO_MPC_EXPERIMENT_ARTIFACTS_INTERNAL_HPP
#define DRO_MPC_EXPERIMENT_ARTIFACTS_INTERNAL_HPP

#include "experiment_harness.hpp"

#include <string>
#include <set>
#include <vector>

namespace dro_mpc {
namespace detail {

// Union over every trajectory and every recorded Markov horizon mode, never a preview.
inline std::vector<std::string> sampled_mode_support(
    const std::vector<Scenario>& scenarios, int obstacle_id) {
    std::set<std::string> modes;
    for (const auto& scenario : scenarios) {
        const auto it = scenario.trajectories.find(obstacle_id);
        if (it == scenario.trajectories.end()) continue;
        const auto& trajectory = it->second;
        if (trajectory.sampled_mode_sequence.empty()) modes.insert(trajectory.mode_id);
        else modes.insert(trajectory.sampled_mode_sequence.begin(),
                          trajectory.sampled_mode_sequence.end());
    }
    return {modes.begin(), modes.end()};
}

struct ModeCoverageRecord {
    int obstacle_id = 0;
    int class_id = 0;
    std::string true_mode;
    std::vector<std::string> sampled_modes;
    bool represented = false;
};

struct DecisionRecord {
    FailureDiagnostics failure_diagnostics;
    int step = 0;
    double solve_ms = 0.0;
    bool success = false;
    bool certificate_requested = false;
    bool nominal_fallback_attempted = false;
    bool used_nominal_fallback = false;
    bool certified = false;
    double applied_control_effort = 0.0;
    size_t scenario_count = 0;
    std::vector<ModeCoverageRecord> mode_coverage;
};

struct RolloutTraceFrame {
    int step = 0;
    double time_seconds = 0.0;
    EgoState ego;
    bool has_decision = false;
    bool show_support_scenarios = false;
    bool support_evaluated = false;
    std::vector<int> support_scenario_ids;
    std::vector<EgoState> predicted_ego;
    std::vector<CollisionConstraint> linearized_constraints;
    std::vector<Scenario> sampled_scenarios;
    size_t scenario_count = 0;
    double max_sample_deviation = 0.0;
    std::vector<ObstacleState> obstacles;
    std::vector<std::string> obstacle_modes;
    double path_progress = 0.0;
    double minimum_clearance = 0.0;
    double ambiguity_radius = 0.0;
    double solve_time_ms = 0.0;
    bool collision = false;
};

struct RolloutTrace {
    std::vector<DecisionRecord> decisions;
    ReferencePath route;
    std::vector<ReferencePath> road_centerlines;
    std::vector<RolloutTraceFrame> frames;
};

/// Writes a self-contained artifact bundle and returns its canonical path.
std::string write_rollout_artifacts(
    const ExperimentConfig& config,
    const RolloutRecord& record,
    const SeedBundle& seeds,
    const RolloutTrace& trace
);

}  // namespace detail
}  // namespace dro_mpc

#endif  // DRO_MPC_EXPERIMENT_ARTIFACTS_INTERNAL_HPP
