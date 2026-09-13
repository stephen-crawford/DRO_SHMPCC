#include "dynamics.hpp"
#include "experiment_artifacts_internal.hpp"
#include "scenario_sampler.hpp"

#include <iostream>
#include <random>

using namespace dro_mpc;

int main() {
    auto catalog = create_obstacle_mode_models(0.1);
    std::map<std::string, ModeModel> modes{
        {"accelerating", catalog.at("accelerating")},
        {"constant_velocity", catalog.at("constant_velocity")}};
    for (auto& [id, mode] : modes) mode.G.setZero();
    const ObstacleState initial(0, 0, 1, 0);
    const std::map<int, ObstacleState> obstacles{{0, initial}};
    const std::map<int, ModeHistory> histories{{0, ModeHistory(0, modes)}};
    const std::map<int, std::map<std::string, double>> weights{
        {0, {{"accelerating", 0.0}, {"constant_velocity", 1.0}}}};
    Eigen::Matrix2d alternating;
    alternating << 0, 1, 1, 0;
    const std::map<int, Eigen::MatrixXd> transitions{{0, alternating}};
    std::mt19937 rng(77), repeated_rng(77);
    const auto scenarios = sample_scenarios(
        obstacles, histories, &weights, 4, 3, {}, &transitions, &rng);
    const auto repeated = sample_scenarios(
        obstacles, histories, &weights, 4, 3, {}, &transitions, &repeated_rng);
    bool valid = rng == repeated_rng;
    for (size_t i = 0; i < scenarios.size(); ++i) {
        const auto& trajectory = scenarios[i].trajectories.at(0);
        valid = valid && trajectory.sampled_mode_sequence.size() == 4 &&
            trajectory.sampled_mode_sequence == repeated[i].trajectories.at(0).sampled_mode_sequence;
        ObstacleState state = initial;
        for (size_t k = 0; k < trajectory.sampled_mode_sequence.size(); ++k) {
            state = modes.at(trajectory.sampled_mode_sequence[k]).propagate(state);
            valid = valid && (state.position() - trajectory.steps[k + 1].mean).norm() < 1e-12;
        }
    }
    // Both modes occur in a Markov horizon, even though its representative label
    // contains only the dominant one. A preview subset is not used here.
    valid = valid && detail::sampled_mode_support(scenarios, 0) ==
        std::vector<std::string>({"accelerating", "constant_velocity"});
    valid = valid && detail::sampled_mode_support(scenarios, 99).empty();
    const auto held = sample_scenarios(obstacles, histories, &weights, 4, 3, {}, nullptr, &rng);
    valid = valid && detail::sampled_mode_support(held, 0) ==
        std::vector<std::string>({"constant_velocity"});
    // A mode occurring only in the last scenario must still count as represented.
    auto last_only = held;
    last_only.back().trajectories.at(0).mode_id = "accelerating";
    valid = valid && detail::sampled_mode_support(last_only, 0).size() == 2;
    std::cout << (valid ? "PASS" : "FAIL")
        << ": complete sampled-mode union, held/Markov metadata, actual propagation and RNG repeatability\n";
    return valid ? 0 : 1;
}
