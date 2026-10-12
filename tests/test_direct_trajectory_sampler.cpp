#include "experiment_config_yaml.hpp"
#include "mpc_controller.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <set>

using namespace dro_mpc;

namespace dro_mpc {
struct DirectTrajectorySamplerTestAccess {
    static auto& bank(DirectTrajectorySampler& sampler) { return sampler.standard_samples_; }
};
}

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}
template<class F> bool rejects(F action) {
    try { action(); } catch (const std::invalid_argument&) { return true; }
    return false;
}
}

int main() {
    const std::map<int, ObstacleState> obstacles{
        {4, ObstacleState(1, -2, 2, 1)}, {9, ObstacleState(5, 3, -1, 0)}};
    constexpr int N = 4, S = 32;
    constexpr double dt = 0.2;
    auto predictions = constant_velocity_predictions(obstacles, N, dt, 0.0);
    DirectTrajectorySampler sampler;
    std::mt19937 rng(77);
    const auto deterministic = sampler.sample(obstacles, predictions, N, S, dt, 2, rng, 100);
    bool exact = deterministic.front().scenario_id == 100 && deterministic.back().scenario_id == 131;
    for (const auto& scenario : deterministic) for (const auto& [id, trajectory] : scenario.trajectories) {
        exact &= trajectory.steps.size() == N + 1 && trajectory.mode_id == "constant_velocity";
        for (int k = 0; k <= N; ++k) {
            exact &= (trajectory.steps[k].mean - (obstacles.at(id).position() + k * dt * obstacles.at(id).velocity())).norm() < 1e-12;
            exact &= trajectory.steps[k].covariance.isZero();
        }
    }
    check(exact, "zero-noise means, measured stage 0, N future stages and offset IDs");

    predictions = constant_velocity_predictions(obstacles, N, dt, 1.0);
    // Inject one known bank. Each stage must use a permutation of that bank.
    auto& bank = DirectTrajectorySamplerTestAccess::bank(sampler);
    bank.assign(1, std::vector<Eigen::Vector2d>(S));
    for (int s = 0; s < S; ++s) bank[0][s] = Eigen::Vector2d(s * 0.1, 1.0 - s * 0.2);
    const auto original_bank = bank;
    const auto sampled = sampler.sample(obstacles, predictions, N, S, dt, 1, rng);
    bool integrated = true, shared = true;
    for (int k = 1; k <= N; ++k) {
        std::set<int> seen;
        for (int s = 0; s < S; ++s) {
            const auto& trajectory = sampled[s].trajectories.at(4);
            const Eigen::Vector2d previous = k == 1 ? Eigen::Vector2d::Zero().eval()
                : (trajectory.steps[k - 1].mean - predictions.at(4)[0].steps[k - 2].position).eval();
            const Eigen::Vector2d error = trajectory.steps[k].mean - predictions.at(4)[0].steps[k - 1].position;
            const Eigen::Vector2d increment = (error - previous) / dt;
            const int pick = std::lround(increment.x() / 0.1);
            integrated &= pick >= 0 && pick < S;
            if (pick >= 0 && pick < S) integrated &= (increment - bank[0][pick]).norm() < 1e-11;
            seen.insert(pick);
            integrated &= (trajectory.steps[k].covariance - k * dt * dt * Eigen::Matrix2d::Identity()).norm() < 1e-12;
            const auto& other = sampled[s].trajectories.at(9);
            shared &= (error - (other.steps[k].mean - predictions.at(9)[0].steps[k - 1].position)).norm() < 1e-12;
        }
        integrated &= seen.size() == S;
    }
    check(integrated, "each stage permutes the finite bank; errors accumulate and covariance is k*dt^2*I");
    check(shared, "batch picks and permutations are shared across obstacles as in the reference");
    sampler.sample(obstacles, predictions, N, S, dt, 1, rng);
    check(bank == original_bank, "Gaussian bank persists across repeated sampling calls");

    auto multimodal = constant_velocity_predictions({{4, obstacles.at(4)}}, N, dt, 0.0);
    auto alternate = multimodal[4][0];
    alternate.mode_id = "alternate";
    alternate.probability = 0.75;
    for (auto& step : alternate.steps) step.position.y() += 10.0;
    multimodal[4][0].probability = 0.25;
    multimodal[4].push_back(alternate);
    const std::map<int, ObstacleState> single{{4, obstacles.at(4)}};
    const auto mixture = sampler.sample(single, multimodal, N, 10000, dt, 2, rng);
    int alternate_count = 0;
    bool held = true;
    for (const auto& scenario : mixture) {
        const auto& trajectory = scenario.trajectories.at(4);
        const bool is_alternate = trajectory.mode_id == "alternate";
        alternate_count += is_alternate;
        for (int k = 1; k <= N; ++k)
            held &= (trajectory.steps[k].mean - multimodal.at(4)[is_alternate ? 1 : 0].steps[k - 1].position).norm() < 1e-12;
    }
    std::cout << "Supplied probability=0.75, alternate count=" << alternate_count << "/10000\n";
    check(held && alternate_count > 7300 && alternate_count < 7700,
          "supplied categorical probabilities choose one mode held for the entire trajectory");
    auto invalid = multimodal;
    invalid[4][0].probability = 0.5;
    check(rejects([&] { sampler.sample(single, invalid, N, S, dt, 1, rng); }), "unnormalized probabilities are rejected");
    invalid = multimodal;
    invalid[4][0].steps.pop_back();
    check(rejects([&] { sampler.sample(single, invalid, N, S, dt, 1, rng); }), "short prediction paths are rejected");
    check(rejects([&] { sampler.sample(obstacles, multimodal, N, S, dt, 1, rng); }), "missing obstacle predictions are rejected");
    check(rejects([&] { sampler.sample(single, multimodal, N, S, dt, 0, rng); }), "empty database settings are rejected");

    RuntimeConfig cfg;
    cfg.mpc.type = MPCType::MPC;
    cfg.mpc.sync_from_type();
    cfg.mpc.horizon = N;
    cfg.mpc.dt = dt;
    cfg.mpc.sampling.set_manual_sample_count(S);
    cfg.mpc.sampling.trajectory_noise_stddev = 0.0;
    cfg.mpc.ego.num_discs = 1;
    cfg.random_seed = 77;
    check(!cfg.mpc.sampling.multimodal_bilevel, "direct sampling is the C++ default");
    cfg.dro.enabled = true;
    check(rejects([&] { cfg.validate(); }), "DRO must explicitly opt in to multimodal_bilevel");
    cfg.dro.enabled = false;
    MPCController controller(cfg);
    controller.set_capture_attempt_diagnostics(true);
    const std::map<int, ObstacleState> distant{{4, ObstacleState(50, 50, 1, 0)}};
    auto supplied = constant_velocity_predictions(distant, N, dt, 0.0);
    supplied[4][0].mode_id = "external_path";
    supplied[4][0].steps[0].position = Eigen::Vector2d(52, 51);
    controller.set_obstacle_predictions(supplied);
    auto result = controller.solve(EgoState(0, 0, 0, 1), distant, Eigen::Vector2d(3, 0));
    const auto& direct = controller.scenarios().front().trajectories.at(4);
    check(direct.mode_id == "external_path" && (direct.steps[1].mean - Eigen::Vector2d(52, 51)).norm() < 1e-12,
          "controller dispatch uses supplied path rather than mode-history rollouts");
    check(!result.attempt_diagnostics.empty() && result.attempt_diagnostics[0].sampling_weights.at(4).at("external_path") == 1.0,
          "attempt diagnostics record the actual supplied prediction law");
    controller.solve(EgoState(0, 0, 0, 1), distant, Eigen::Vector2d(3, 0));
    check(controller.scenarios().front().trajectories.at(4).mode_id == "constant_velocity",
          "supplied predictions are consumed once; subsequent solves use fresh CV means");
    check(rejects([&] { controller.set_custom_mode_weights(4, {{"stop", 1.0}}); }),
          "direct mode rejects history/DRO weight overrides");

    cfg.mpc.sampling.multimodal_bilevel = true;
    MPCController legacy(cfg);
    legacy.initialize_obstacle(4, 0);
    legacy.set_custom_mode_weights(4, {{"stop", 1.0}});
    legacy.solve(EgoState(0, 0, 0, 1), distant, Eigen::Vector2d(3, 0));
    check(legacy.scenarios().front().trajectories.at(4).mode_id == "stop",
          "explicit multimodal_bilevel selects the existing modal sampler");

    cfg.dro.enabled = true;
    cfg.dro.fixed_rho = 0.05;
    cfg.dro.solver.radius_calibration.use_domination_constraints = false;
    MPCController bilevel(cfg);
    auto models = create_obstacle_mode_models(dt);
    std::map<std::string, ModeModel> two_modes{
        {"constant_velocity", models.at("constant_velocity")}, {"stop", models.at("stop")}};
    for (auto& [id, model] : two_modes) model.G.setZero();
    bilevel.initialize_obstacle(4, 0, two_modes);
    bilevel.update_mode_observation(4, 0, "constant_velocity", 0);
    bilevel.solve(EgoState(0, 0, 0, 1), distant, Eigen::Vector2d(3, 0));
    check(bilevel.last_dro_results().count(4) && bilevel.scenarios().size() == S,
          "explicit bi-level option retains DRO allocation and sampling from q*");

    cfg.dro.enabled = false;
    cfg.mpc.sampling.markov_jump_system = true;
    MPCController markov(cfg);
    markov.initialize_obstacle(4, 0, two_modes);
    markov.solve(EgoState(0, 0, 0, 1), distant, Eigen::Vector2d(3, 0));
    check(markov.scenarios().front().trajectories.at(4).sampled_mode_sequence.size() == N,
          "explicit bi-level option retains within-horizon Markov mode switching");

    const std::string overlay = "/tmp/dro_direct_sampling_test.yaml";
    {
        std::ofstream out(overlay);
        out << "multimodal_bilevel: true\ntrajectory_sample_batches: 7\ntrajectory_noise_stddev: 0.3\n";
    }
    const auto parsed = yaml_config::load_experiment_config(overlay, true);
    std::remove(overlay.c_str());
    check(parsed.mpc.sampling.multimodal_bilevel && parsed.mpc.sampling.trajectory_sample_batches == 7 &&
          parsed.mpc.sampling.trajectory_noise_stddev == 0.3, "strict YAML loads sampling settings");
    std::cout << (failures ? "DIRECT SAMPLING TESTS FAILED\n" : "ALL DIRECT SAMPLING TESTS PASSED\n");
    return failures ? 1 : 0;
}
