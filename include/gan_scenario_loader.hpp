/**
 * @file gan_scenario_loader.hpp
 * @brief Load GAN-generated adversarial obstacle scenarios from CSV for SHMPC.
 *
 * CSV format: scenario_id, obstacle_id, k, dx, dy (relative waypoints from obstacle start).
 * At load time, current obstacle state is used so mean position at step k = (obs.x+dx, obs.y+dy).
 *
 * GANScenarioCache: load CSV once, then materialize(obstacles) each step to avoid file I/O.
 */

#ifndef SCENARIO_MPC_GAN_SCENARIO_LOADER_HPP
#define SCENARIO_MPC_GAN_SCENARIO_LOADER_HPP

#include "types.hpp"
#include <string>
#include <vector>
#include <map>

namespace scenario_mpc {

/**
 * Load scenarios from a GAN export CSV.
 * CSV columns: scenario_id, obstacle_id, k, dx, dy.
 * Each scenario is built by adding (dx, dy) to the current obstacle position for that obstacle_id.
 *
 * @param csv_path Path to the CSV file
 * @param obstacles Current obstacle states (used to get initial position per obstacle_id)
 * @param horizon Expected number of prediction steps (k from 0 to horizon inclusive)
 * @return Vector of Scenario objects; empty if file missing/invalid or no matching obstacles
 */
std::vector<Scenario> load_scenarios_from_gan_csv(
    const std::string& csv_path,
    const std::map<int, ObstacleState>& obstacles,
    int horizon
);

/**
 * Cache of relative waypoints (scenario_id, obstacle_id, k, dx, dy) for efficiency.
 * Load CSV once per rollout; materialize() builds Scenario from cache + current obstacles (no file I/O).
 */
struct GANScenarioCache {
    struct Waypoint {
        int scenario_id = 0;
        int obstacle_id = 0;
        int k = 0;
        double dx = 0;
        double dy = 0;
    };
    std::vector<Waypoint> waypoints;
    int horizon_cached = 0;

    /** Load CSV into cache. Returns true if load succeeded and cache is non-empty. */
    bool load(const std::string& csv_path, int horizon);

    /** Build scenarios from cache + current obstacles. Uses at most max_scenarios. */
    std::vector<Scenario> materialize(
        const std::map<int, ObstacleState>& obstacles,
        int horizon,
        int max_scenarios
    ) const;
};

}  // namespace scenario_mpc

#endif
