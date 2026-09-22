#include "dynamics.hpp"
#include "scenario_sampler.hpp"

#include <cmath>
#include <iostream>
#include <random>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

bool same_trajectory(const ObstacleTrajectory& lhs, const ObstacleTrajectory& rhs) {
    if (lhs.obstacle_id != rhs.obstacle_id || lhs.mode_id != rhs.mode_id ||
        lhs.steps.size() != rhs.steps.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.steps.size(); ++i) {
        if ((lhs.steps[i].mean - rhs.steps[i].mean).norm() > 1e-12 ||
            (lhs.steps[i].covariance - rhs.steps[i].covariance).norm() > 1e-12) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const std::map<int, ObstacleState> obstacles{{4, ObstacleState(1.0, -2.0, 0.0, 0.0)}};

    {
        std::mt19937 rng(8);
        const std::map<int, ModeHistory> no_histories;
        const auto scenarios = sample_scenarios(
            obstacles, no_histories, nullptr, 3, 2, {}, nullptr, &rng, 10);
        const auto& trajectory = scenarios.front().trajectories.at(4);
        bool held_position = trajectory.mode_id == "stationary" && trajectory.steps.size() == 4;
        for (const auto& step : trajectory.steps) {
            held_position = held_position &&
                (step.mean - Eigen::Vector2d(1.0, -2.0)).norm() < 1e-12;
        }
        check(scenarios.size() == 2 && scenarios.front().scenario_id == 10 &&
                  scenarios.back().scenario_id == 11 && held_position,
              "cold-start sampling creates stationary trajectories with unique offset IDs");
    }

    std::map<std::string, ModeModel> modes;
    modes["constant_velocity"] = create_obstacle_mode_models(0.1).at("constant_velocity");
    modes["constant_velocity"].G.setZero();
    ModeHistory history(4, modes);
    history.record_observation(0, history.obstacle_id, "constant_velocity");
    const std::map<int, ModeHistory> histories{{4, history}};
    const std::map<int, std::map<std::string, double>> deterministic_weights{
        {4, {{"constant_velocity", 1.0}}}};

    {
        std::mt19937 rng_a(19);
        std::mt19937 rng_b(19);
        const auto first = sample_scenarios(
            obstacles, histories, &deterministic_weights, 3, 2, {}, nullptr, &rng_a);
        const auto second = sample_scenarios(
            obstacles, histories, &deterministic_weights, 3, 2, {}, nullptr, &rng_b);
        const auto& trajectory = first.front().trajectories.at(4);
        check(trajectory.mode_id == "constant_velocity" && trajectory.steps.size() == 4 &&
                  std::abs(trajectory.steps.back().mean.x() - 1.0) < 1e-12 &&
                  same_trajectory(trajectory, second.front().trajectories.at(4)),
              "held-mode sampling is deterministic and follows the configured model");
    }

    {
        Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(1, 1);
        const std::map<int, Eigen::MatrixXd> transitions{{4, identity}};
        std::mt19937 rng(31);
        const auto scenarios = sample_scenarios(
            obstacles, histories, &deterministic_weights, 4, 1, {}, &transitions, &rng);
        const auto& trajectory = scenarios.front().trajectories.at(4);
        check(trajectory.mode_id == "constant_velocity" && trajectory.steps.size() == 5 &&
                  std::abs(trajectory.steps.back().mean.x() - 1.0) < 1e-12,
              "Markov sampling honors a deterministic transition matrix and horizon");
    }

    std::cout << (failures == 0 ? "ALL SCENARIO-SAMPLER TESTS PASSED\n"
                                : "SCENARIO-SAMPLER TESTS FAILED\n");
    return failures == 0 ? 0 : 1;
}
