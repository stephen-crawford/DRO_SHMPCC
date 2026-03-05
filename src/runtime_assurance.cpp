/**
 * @file runtime_assurance.cpp
 * @brief Implementation of runtime assurance wrapper (Section 7.4).
 */

#include "runtime_assurance.hpp"
#include <cmath>
#include <map>

namespace scenario_mpc {

RuntimeAssurance::RuntimeAssurance(MonitorFn monitor, EgoInput fallback_action)
    : monitor_(std::move(monitor)), fallback_(fallback_action) {}

EgoInput RuntimeAssurance::apply(
    const EgoState& state,
    const EgoInput& learned_action,
    const std::optional<double>& certificate_risk) {
    bool safe = monitor_(state, certificate_risk);
    if (safe) {
        consecutive_unsafe_ = 0;
        return learned_action;
    }
    consecutive_unsafe_++;
    return fallback_;
}

void RuntimeAssurance::set_hysteresis_count(int n) { hysteresis_count_ = n; }

bool simple_ttc_monitor(
    const EgoState& state,
    const std::map<int, ObstacleState>& obstacles,
    double min_distance_threshold) {
    Eigen::Vector2d p_ego = state.position();
    for (const auto& [_, obs] : obstacles) {
        double d = (p_ego - obs.position()).norm();
        if (d < min_distance_threshold) return false;
    }
    return true;
}

}  // namespace scenario_mpc
