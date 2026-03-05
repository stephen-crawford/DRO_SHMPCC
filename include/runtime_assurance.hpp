/**
 * @file runtime_assurance.hpp
 * @brief Runtime assurance wrapper: monitor + fallback controller (Section 7.4).
 */

#ifndef SCENARIO_MPC_RUNTIME_ASSURANCE_HPP
#define SCENARIO_MPC_RUNTIME_ASSURANCE_HPP

#include "types.hpp"
#include <functional>
#include <map>
#include <optional>

namespace scenario_mpc {

/**
 * @brief Monitor: (state, optional certificate info) -> true if safe to use learned action.
 */
using MonitorFn = std::function<bool(const EgoState&, const std::optional<double>& certificate_risk)>;

/**
 * @brief RTA wrapper: apply learned action if monitor says safe, else fallback.
 */
class RuntimeAssurance {
public:
    RuntimeAssurance(MonitorFn monitor, EgoInput fallback_action);
    EgoInput apply(const EgoState& state, const EgoInput& learned_action,
                   const std::optional<double>& certificate_risk = std::nullopt);
    void set_hysteresis_count(int n);  ///< Require n consecutive safe before switching back to learned

private:
    MonitorFn monitor_;
    EgoInput fallback_;
    int hysteresis_count_ = 0;
    int consecutive_unsafe_ = 0;
};

/**
 * @brief Simple TTC-style monitor: safe if min distance to obstacles > threshold.
 */
bool simple_ttc_monitor(
    const EgoState& state,
    const std::map<int, ObstacleState>& obstacles,
    double min_distance_threshold
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_RUNTIME_ASSURANCE_HPP
