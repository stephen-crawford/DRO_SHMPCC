// Easy deterministic plant test: straight -> left -> straight -> right -> straight.
// Uses ObstacleSim::step and the same mode models as the experiment harness.
#include "dynamics.hpp"
#include "experiment_artifacts_internal.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using namespace dro_mpc;

int main(int argc, char** argv) {
    const std::filesystem::path output = argc > 1 ? argv[1] : "traffic-turn-artifacts";
    constexpr double dt = 0.1;
    constexpr int steps = 164;
    bool valid = true;
    auto simulate = [&](unsigned seed, bool record) {
        std::mt19937 rng(derive_seeds(seed, 0).env);
        ObstacleSim obstacle;
        obstacle.state = ObstacleState(0, 0, 2, 0);
        obstacle.mode_models = create_obstacle_mode_models(dt);
        detail::RolloutTrace trace;
        trace.route = ReferencePath::create_straight({0, 0}, {20, 0});
        for (int step = 0; step <= steps; ++step) {
            detail::RolloutTraceFrame frame;
            frame.step = step;
            frame.time_seconds = step * dt;
            frame.ego = EgoState(-3, -3, 0, 0);
            frame.obstacles.push_back(obstacle.state);
            frame.obstacle_modes.push_back(step == 0 ? "constant_velocity" : obstacle.current_mode);
            trace.frames.push_back(frame);
            if (step == steps) break;
            const std::string mode = step < 20 ? "constant_velocity" :
                step < 72 ? "turn_left" : step < 92 ? "constant_velocity" :
                step < 144 ? "turn_right" : "constant_velocity";
            obstacle.current_mode = mode;
            const auto previous = obstacle.state;
            obstacle.step(dt, rng, 0.0, 2.0);
            const auto current = obstacle.state;
            const double angle = std::atan2(previous.vx * current.vy - previous.vy * current.vx,
                                            previous.vx * current.vx + previous.vy * current.vy);
            const double expected = mode == "turn_left" ? .03 : mode == "turn_right" ? -.03 : 0.0;
            valid = valid && std::abs(angle - expected) < 1e-12 &&
                std::abs(std::hypot(current.vx, current.vy) - 2.0) < 1e-12 &&
                std::abs(current.x - previous.x - dt * current.vx) < 1e-12 &&
                std::abs(current.y - previous.y - dt * current.vy) < 1e-12;
        }
        const auto& after_left = trace.frames.at(72).obstacles.front();
        const auto& after_right = trace.frames.at(144).obstacles.front();
        valid = valid && after_left.vy > 1.99 && after_right.vx > 1.99 &&
            std::abs(after_right.vy) < 1e-12;
        if (record) {
            ExperimentConfig config = default_experiment_config();
            config.mpc.dt = dt;
            config.obstacles.process_noise = 0.0;
            config.obstacles.obs_modes = {"constant_velocity", "turn_left", "turn_right"};
            config.obstacles.num_modes = 3;
            config.obstacles.rare_mode.clear();
            config.obstacles.initial_obstacle_states = {ObstacleState(0, 0, 2, 0)};
            config.artifacts.output_directory = output.string();
            config.artifacts.run_name = "straight_left_straight_right";
            config.artifacts.write_visualization_gif = true;
            config.artifacts.show_linearized_constraints = false;
            RolloutRecord result;
            result.method = "Obstacle plant test (scripted modes, no MPC)";
            result.scenario = "straight-left-straight-right-straight";
            result.seed = seed;
            result.total_steps = steps;
            const auto bundle = detail::write_rollout_artifacts(config, result, derive_seeds(seed, 0), trace);
            std::ofstream events(std::filesystem::path(bundle) / "mode_schedule.csv");
            events << "start_step,end_step,mode,expected_heading_change_rad\n"
                "0,20,constant_velocity,0\n20,72,turn_left,1.56\n72,92,constant_velocity,0\n"
                "92,144,turn_right,-1.56\n144,164,constant_velocity,0\n";
            std::cout << "Left turn: " << std::atan2(after_left.vy, after_left.vx) * 180 / M_PI
                << " degrees; heading after right turn: "
                << std::atan2(after_right.vy, after_right.vx) * 180 / M_PI
                << " degrees; speed: " << std::hypot(after_right.vx, after_right.vy)
                << " m/s; artifacts: " << bundle << '\n';
        }
        return trace;
    };
    const auto first = simulate(77, true);
    const auto repeat = simulate(77, false);
    for (size_t i = 0; i < first.frames.size(); ++i)
        valid = valid && first.frames[i].obstacles.front().to_array() == repeat.frames[i].obstacles.front().to_array();
    std::cout << (valid ? "PASS" : "FAIL") << ": turn signs, speed, displacement, heading persistence and exact repeat\n";
    return valid ? 0 : 1;
}
