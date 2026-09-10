/**
 * @file experiment_artifacts_internal.hpp
 * @brief Private trace representation shared by the rollout and artifact writer.
 */

#ifndef DRO_MPC_EXPERIMENT_ARTIFACTS_INTERNAL_HPP
#define DRO_MPC_EXPERIMENT_ARTIFACTS_INTERNAL_HPP

#include "experiment_harness.hpp"

#include <string>
#include <vector>

namespace dro_mpc {
namespace detail {

struct RolloutTraceFrame {
    int step = 0;
    double time_seconds = 0.0;
    EgoState ego;
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
