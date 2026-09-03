#include "dynamics.hpp"
#include "experiment_config_yaml.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

using namespace dro_mpc;

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

bool contains(const std::vector<std::string>& modes, const std::string& id) {
    return std::find(modes.begin(), modes.end(), id) != modes.end();
}
}  // namespace

int main() {
    const auto catalog = create_obstacle_mode_models(0.1);
    for (const char* id : {"constant_velocity", "accelerating", "decelerating", "stop",
                           "turn_left", "turn_right", "turn_left_sharp", "turn_right_sharp",
                           "lane_change_left", "lane_change_right", "lane_change_left_fast",
                           "lane_change_right_fast"}) {
        check(catalog.count(id) == 1, id);
    }

    const std::vector<std::string> candidates = {
        "constant_velocity", "turn_left", "turn_right", "accelerating", "invalid"};
    std::mt19937 rng_a(42);
    const auto fixed = select_obstacle_mode_ids(
        candidates, "lane_change_left", catalog, false, 2, rng_a);
    check(fixed.size() == 5 && fixed.front() == "constant_velocity" &&
              fixed[3] == "accelerating" && fixed.back() == "lane_change_left",
          "fixed mode selection preserves valid YAML order and appends rare mode");

    std::mt19937 rng_b(17), rng_c(17);
    const auto randomized_a = select_obstacle_mode_ids(
        candidates, "lane_change_left", catalog, true, 2, rng_b);
    const auto randomized_b = select_obstacle_mode_ids(
        candidates, "lane_change_left", catalog, true, 2, rng_c);
    check(randomized_a == randomized_b,
          "random mode selection is reproducible from the rollout seed");
    check(randomized_a.size() == 3 && contains(randomized_a, "lane_change_left") &&
              std::set<std::string>(randomized_a.begin(), randomized_a.end()).size() ==
                  randomized_a.size(),
          "random mode selection honors num_modes and keeps rare mode available");

    const char* overlay_path = "/tmp/dro_shmpcc_mode_catalog_overlay.yaml";
    {
        std::ofstream overlay(overlay_path);
        overlay << "randomize_available_modes: true\n"
                << "randomize_modes_per_obstacle: true\n"
                << "horizon: 17\n"
                << "dt: 0.2\n"
                << "max_velocity: 6.5\n"
                << "environment: enter_ramp\n"
                << "lane_count: 3\n"
                << "lane_width: 3.8\n"
                << "road_length: 120\n"
                << "ramp_length: 65\n"
                << "merge_length: 40\n"
                << "roundabout_radius: 22\n"
                << "num_modes: 3\n"
                << "obs_modes: [accelerating, stop, turn_left_sharp]\n"
                << "rare_mode: lane_change_right_fast\n"
                << "num_obstacles: 3\n"
                << "obstacles_per_class: 2\n"
                << "obstacle_history: shared_history_classes\n"
                << "markov_jump_system: true\n"
                << "obstacle_starts: [4.5,1.0,0.5,0; 8.0,-2.0; 12,3,-1,0.25]\n"
                << "fixed_rho: 0.23\n"
                << "risk_measure: joint_cvar\n"
                << "risk_horizon: 7\n"
                << "joint_risk_seed: 123456\n";
    }
    const auto yaml_cfg = yaml_config::load_experiment_config(overlay_path, true);
    std::remove(overlay_path);
    const auto runtime_cfg = yaml_cfg.to_scenario_mpc_config();
    check(yaml_cfg.obstacles.randomize_available_modes &&
              yaml_cfg.obstacles.randomize_modes_per_obstacle &&
              yaml_cfg.mpc.horizon == 17 && yaml_cfg.mpc.dt == 0.2 &&
              yaml_cfg.mpc.ego.dynamics.max_velocity == 6.5 &&
              yaml_cfg.environment.type == EnvironmentType::ENTER_RAMP &&
              yaml_cfg.environment.lane_count == 3 &&
              yaml_cfg.environment.lane_width == 3.8 &&
              yaml_cfg.environment.road_length == 120.0 &&
              yaml_cfg.environment.ramp_length == 65.0 &&
              yaml_cfg.environment.merge_length == 40.0 &&
              yaml_cfg.environment.roundabout_radius == 22.0 &&
              yaml_cfg.obstacles.num_modes == 3 &&
              yaml_cfg.obstacles.obs_modes.size() == 3 &&
              yaml_cfg.obstacles.rare_mode == "lane_change_right_fast" &&
              yaml_cfg.obstacles.num_obstacles == 3 &&
              yaml_cfg.obstacles.initial_obstacle_states.size() == 3 &&
              yaml_cfg.obstacles.initial_obstacle_states[0].x == 4.5 &&
              yaml_cfg.obstacles.initial_obstacle_states[0].vx == 0.5 &&
              yaml_cfg.obstacles.initial_obstacle_states[1].y == -2.0 &&
              yaml_cfg.obstacles.initial_obstacle_states[1].vx == 0.0 &&
              yaml_cfg.obstacles.initial_obstacle_states[2].vy == 0.25 &&
              yaml_cfg.obstacles.obstacles_per_class == 2 &&
              yaml_cfg.mpc.sampling.markov_jump_system &&
              yaml_cfg.dro.solver.radius_calibration.risk_measure == DRORiskMeasure::JOINT_CVAR &&
              yaml_cfg.dro.solver.radius_calibration.risk_horizon == 7 &&
              yaml_cfg.dro.solver.radius_calibration.joint_risk_seed == 123456ULL,
          "YAML settings survive parsing and experiment normalization");
    check(runtime_cfg.mpc.horizon == 17 && runtime_cfg.mpc.dt == 0.2 &&
              runtime_cfg.mpc.ego.dynamics.max_velocity == 6.5 &&
              runtime_cfg.mpc.sampling.markov_jump_system &&
              runtime_cfg.dro.solver.base_radius == 0.23 &&
              !runtime_cfg.dro.solver.radius_calibration.use_calibrated_radius,
          "YAML settings survive conversion to the controller runtime config");

    const auto route = ReferencePath::create_polyline({
        Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(10.0, 0.0),
        Eigen::Vector2d(10.0, 5.0)});
    check(route.path_type() == ReferencePath::PathType::CUSTOM &&
              route.num_points() == 3 && std::abs(route.total_length() - 15.0) < 1e-12 &&
              (route.get_position_at(route.total_length()) - Eigen::Vector2d(10.0, 5.0)).norm() < 1e-8,
          "polyline reference paths preserve route geometry");

    return failures == 0 ? 0 : 1;
}
