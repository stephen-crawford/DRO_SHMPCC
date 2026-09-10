#include "mpc_controller.hpp"
#include "experiment_harness.hpp"
#include "path_progress_linearization.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace dro_mpc;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        const auto path = ReferencePath::create_polyline({{0, 0}, {10, 0}, {16, 8}});
        // Interior, monotone lower bound, upper endpoint, and second segment.
        for (const auto& item : std::vector<std::pair<Eigen::Vector2d, double>>{
                 {{3, 2}, 1}, {{3, 2}, 5}, {{18, 10}, 15}, {{13, 5}, 10}}) {
            const auto& p = item.first;
            const double s = item.second;
            const auto derivative = linearize_path_progress(path, p, s);
            constexpr double h = 1e-6;
            for (int j = 0; j < 2; ++j) {
                Eigen::Vector2d plus = p, minus = p;
                plus(j) += h; minus(j) -= h;
                const double numeric = (path.find_closest_point(plus, s) -
                                        path.find_closest_point(minus, s)) / (2 * h);
                require(std::abs(numeric - derivative.position(j)) < 1e-7,
                        "projected progress position derivative mismatch");
            }
            const double numeric = (path.find_closest_point(p, s + h) -
                                    path.find_closest_point(p, s - h)) / (2 * h);
            require(std::abs(numeric - derivative.previous_progress) < 1e-7,
                    "projected progress lower-bound derivative mismatch");
        }
        std::cout << "PASS: active projection derivatives match finite differences\n";

        const auto straight = ReferencePath::create_straight({0, 0}, {60, 0});
        for (MPCType type : {MPCType::MPCC, MPCType::SH_MPCC}) {
            RuntimeConfig cfg;
            cfg.mpc.type = type;
            cfg.mpc.sync_from_type();
            cfg.random_seed = 77;
            MPCController first(cfg), second(cfg);
            first.set_reference_path(straight);
            second.set_reference_path(straight);
            const EgoState ego(0, 0, 0, 0);
            auto a = first.solve(ego, {}, {60, 0}, 0.0, 0.0, 60.0);
            auto b = second.solve(ego, {}, {-60, 20}, 3.0, 0.0, 60.0);
            require(a.success && b.success, "MPCC solve failed");
            require(a.ego_trajectory.back().s > 0.1, "MPCC failed to progress from rest");
            require(std::abs(a.cost - b.cost) < 1e-10,
                    "MPCC cost depends on endpoint or reference speed");
            require(a.control_inputs.size() == b.control_inputs.size(), "input size differs");
            for (std::size_t k = 0; k < a.control_inputs.size(); ++k) {
                require((a.control_inputs[k].to_array() - b.control_inputs[k].to_array()).norm() < 1e-10,
                        "MPCC inputs depend on endpoint or reference speed");
            }
            cfg.mpc.objective.progress_weight = 0.0;
            MPCController no_reward(cfg);
            no_reward.set_reference_path(straight);
            auto c = no_reward.solve(ego, {}, {60, 0}, 0.0, 0.0, 60.0);
            require(c.success, "zero-reward comparison solve failed");
            require(a.ego_trajectory.back().s > c.ego_trajectory.back().s + 0.1,
                    "progress reward did not increase predicted progress");
            std::cout << mpc_type_name(type) << ": horizon progress=" << a.ego_trajectory.back().s
                      << " m, zero reward=" << c.ego_trajectory.back().s << " m\n";
        }

        auto config = default_experiment_config();
        config.mpc.type = MPCType::MPCC;
        config.mpc.sync_from_type();
        config.obstacles.num_obstacles = 0;
        config.environment.path_completion_fraction = 1.0;
        config.rollout.rollout_steps = 400;
        const auto record = run_experiment_rollout(config, 77);
        std::cout << "S-curve: progress=" << record.total_progress
                  << " contouring RMS=" << record.mean_contouring_error()
                  << " m, steps=" << record.total_steps << '\n';
        require(record.completed_path && !record.collision, "S-curve did not complete");
        // Engineering regression target for this unobstructed base route;
        // this is not a bound claimed for arbitrary routes or obstacles.
        require(record.mean_contouring_error() < 0.5, "S-curve contouring RMS exceeds 0.5 m");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
