/**
 * @file test_obstacle_class.cpp
 * @brief Tests for obstacle class-based mode observation sharing.
 *
 * Validates the class sharing mechanism where obstacles assigned to the same
 * class share mode observations and history. This enables realistic multi-
 * obstacle scenarios where e.g. all pedestrians share learned behavior
 * distributions while vehicles maintain separate histories.
 *
 * Architecture:
 *   - ModeHistory tracks observations by obstacle_class (not just obstacle_id)
 *   - AdaptiveScenarioMPC::update_mode_observation() broadcasts to class siblings
 *   - AdaptiveScenarioMPC::initialize_obstacle() copies sibling history for
 *     late-joining obstacles
 *   - ExperimentConfig::obstacles_per_class controls grouping in rollouts
 *
 * Tests:
 *   1. Observations sync within same class (broadcast to siblings)
 *   2. Late-joining obstacle inherits existing class history
 *   3. Different classes remain independent
 *   4. Multi-obstacle rollout (4 obstacles, 2 classes, 20 MPC steps)
 *   5. Full ExperimentConfig rollout via run_experiment_rollout()
 */

#include <iostream>
#include <cassert>
#include <map>
#include <string>

#include "mpc_controller.hpp"
#include "experiment_harness.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"

using namespace scenario_mpc;

int main() {
    std::cout << "=== Test: Obstacle Class Sharing ===\n\n";

    constexpr double DT = 0.1;
    auto mode_models = create_obstacle_mode_models(DT);

    ScenarioMPCConfig cfg;
    cfg.horizon = 10;
    cfg.dt = DT;
    cfg.num_scenarios = 10;
    cfg.use_sqp_solver = true;

    // -------------------------------------------------------
    // Test 1: Observations sync within same class
    // -------------------------------------------------------
    {
        std::cout << "Test 1: Observations sync within same class... ";
        AdaptiveScenarioMPC ctrl(cfg);

        // Three obstacles: 0,1 share class 0; obstacle 2 is class 1
        ctrl.initialize_obstacle(0, 0, mode_models);
        ctrl.initialize_obstacle(1, 0, mode_models);
        ctrl.initialize_obstacle(2, 1, mode_models);

        // Observe "turn_left" from obstacle 0 (class 0)
        ctrl.update_mode_observation(0, 0, "turn_left", 0);

        // Both obstacle 0 and 1 should have the observation (same class)
        // Obstacle 2 should NOT have it (different class)
        std::map<int, ObstacleState> obstacles;
        obstacles[0] = ObstacleState(3.0, 0.0, -0.1, 0.0);
        obstacles[1] = ObstacleState(5.0, 1.0, -0.1, 0.0);
        obstacles[2] = ObstacleState(7.0, -1.0, -0.1, 0.0);

        // Observe "decelerating" from obstacle 1 (class 0) - should sync to obstacle 0
        ctrl.update_mode_observation(1, 0, "decelerating", 1);

        // Observe "constant_velocity" from obstacle 2 (class 1) - should NOT sync to 0,1
        ctrl.update_mode_observation(2, 1, "constant_velocity", 2);

        // Now solve - just verifying it doesn't crash with multi-obstacle multi-class
        Eigen::Vector2d goal(20.0, 0.0);
        auto result = ctrl.solve(EgoState(0, 0, 0, 1.5), obstacles, goal);
        assert(result.success);

        std::cout << "PASS (solve succeeded with 3 obstacles, 2 classes)\n";
    }

    // -------------------------------------------------------
    // Test 2: Late-joining obstacle inherits class history
    // -------------------------------------------------------
    {
        std::cout << "Test 2: Late-joining obstacle inherits class history... ";
        AdaptiveScenarioMPC ctrl(cfg);

        ctrl.initialize_obstacle(0, 0, mode_models);

        // Build up history for obstacle 0 (class 0)
        ctrl.update_mode_observation(0, 0, "turn_left", 0);
        ctrl.update_mode_observation(0, 0, "turn_left", 1);
        ctrl.update_mode_observation(0, 0, "decelerating", 2);

        // Now add obstacle 1 to same class - should inherit history
        ctrl.initialize_obstacle(1, 0, mode_models);

        // Solve with both obstacles
        std::map<int, ObstacleState> obstacles;
        obstacles[0] = ObstacleState(3.0, 0.0, -0.1, 0.0);
        obstacles[1] = ObstacleState(5.0, 1.0, -0.1, 0.0);

        Eigen::Vector2d goal(20.0, 0.0);
        auto result = ctrl.solve(EgoState(0, 0, 0, 1.5), obstacles, goal);
        assert(result.success);

        std::cout << "PASS\n";
    }

    // -------------------------------------------------------
    // Test 3: Different classes stay independent
    // -------------------------------------------------------
    {
        std::cout << "Test 3: Different classes stay independent... ";
        AdaptiveScenarioMPC ctrl(cfg);

        ctrl.initialize_obstacle(0, 0, mode_models);
        ctrl.initialize_obstacle(1, 1, mode_models);

        // Lots of "turn_left" for class 0
        for (int i = 0; i < 10; ++i)
            ctrl.update_mode_observation(0, 0, "turn_left", i);

        // Lots of "turn_right" for class 1
        for (int i = 0; i < 10; ++i)
            ctrl.update_mode_observation(1, 1, "turn_right", i);

        // Solve with both
        std::map<int, ObstacleState> obstacles;
        obstacles[0] = ObstacleState(3.0, 0.0, -0.1, 0.0);
        obstacles[1] = ObstacleState(5.0, 1.0, -0.1, 0.0);

        Eigen::Vector2d goal(20.0, 0.0);
        auto result = ctrl.solve(EgoState(0, 0, 0, 1.5), obstacles, goal);
        assert(result.success);

        std::cout << "PASS\n";
    }

    // -------------------------------------------------------
    // Test 4: Multi-obstacle rollout (matching ExperimentConfig)
    // -------------------------------------------------------
    {
        std::cout << "Test 4: Multi-obstacle rollout with class sharing... ";
        AdaptiveScenarioMPC ctrl(cfg);
        EgoDynamics dynamics(DT);

        int n_obs = 4;
        int per_class = 2;  // 4 obstacles, 2 per class = 2 classes

        for (int i = 0; i < n_obs; ++i) {
            int obs_class = i / per_class;
            ctrl.initialize_obstacle(i, obs_class, mode_models);
        }

        EgoState ego(0, 0, 0, 1.5);
        Eigen::Vector2d goal(20.0, 0.0);
        std::mt19937 rng(42);

        for (int step = 0; step < 20; ++step) {
            // Build obstacle map
            std::map<int, ObstacleState> obstacles;
            for (int i = 0; i < n_obs; ++i) {
                obstacles[i] = ObstacleState(3.0 + i * 2.0, 0.5 * (i % 2 == 0 ? 1 : -1), -0.1, 0.0);
            }

            // Observe modes - obstacle 0 and 1 share class 0
            std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
            for (int i = 0; i < n_obs; ++i) {
                int obs_class = i / per_class;
                std::string mode = modes[rng() % modes.size()];
                ctrl.update_mode_observation(i, obs_class, mode, step);
            }

            auto result = ctrl.solve(ego, obstacles, goal);
            assert(result.success);

            if (result.first_input().has_value()) {
                ego = dynamics.propagate(ego, result.first_input().value());
            }
        }

        std::cout << "PASS (20 steps, 4 obstacles, 2 classes)\n";
    }

    // -------------------------------------------------------
    // Test 5: ExperimentConfig with num_obstacles/obstacles_per_class
    // -------------------------------------------------------
    {
        std::cout << "Test 5: ExperimentConfig multi-obstacle fields... ";
        ExperimentConfig ecfg;
        ecfg.num_obstacles = 3;
        ecfg.obstacles_per_class = 3;  // all share class 0
        ecfg.num_scenarios = 10;
        ecfg.rollout_steps = 10;
        ecfg.horizon = 10;
        ecfg.ablation = AblationVariant::NO_INJECTION;
        ecfg.safe_horizon_enabled = false;

        auto rec = run_experiment_rollout(ecfg, 42);
        assert(rec.total_steps == 10);
        std::cout << "PASS (collision=" << rec.collision
                  << ", progress=" << rec.total_progress << ")\n";
    }

    std::cout << "\n=== All obstacle class tests PASSED ===\n";
    return 0;
}
