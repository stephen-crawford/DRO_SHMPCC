#include "experiment_config_yaml.hpp"
#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "collision_constraints.hpp"
#include <iostream>
#include <stdexcept>

using namespace dro_mpc;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        auto cfg = yaml_config::load_experiment_config(
            "configs/base_tests/sh_mpcc_1_obstacles.yaml", true);
        const auto seeds = derive_seeds(77, 0);
        std::mt19937 rng(seeds.env);
        const auto env = create_environment(cfg.environment.type, rng, cfg.environment);
        auto runtime = cfg.to_scenario_mpc_config();
        runtime.random_seed = seeds.scenario;
        MPCController controller(runtime);
        controller.set_reference_path(env.path);
        const auto modes = create_obstacle_mode_models(cfg.mpc.dt);
        controller.initialize_obstacle(0, 0, {{"stop", modes.at("stop")}});
        controller.update_mode_observation(0, 0, "stop", 0);
        EgoDynamics dynamics(cfg.mpc.ego.dynamics, cfg.mpc.dt);
        EgoState ego = env.initial_ego;
        const ObstacleState obstacle(12.5, 0, 0, 0);
        double progress = 0.0;
        int executed_fallbacks = 0, rejected = 0;
        for (int step = 0; step < 40; ++step) {
            controller.update_mode_observation(0, 0, "stop", step + 1);
            auto result = controller.solve(ego, {{0, obstacle}}, env.goal,
                                           cfg.rollout.metrics_v_ref, progress, env.path.total_length());
            require(!result.success || result.sampled_constraints_satisfied,
                    "infeasible plan marked executable");
            if (!result.success) {
                ++rejected;
                std::cout << "Rejected decision " << step + 1 << '\n';
                break;
            }
            const auto input = result.first_input();
            require(input.has_value(), "executable plan has no input");
            const auto next = dynamics.propagate(ego, *input);
            require((next.to_array() - result.ego_trajectory.at(1).to_array()).norm() < 1e-12,
                    "executed first control does not reproduce returned state");
            if (result.used_fallback) {
                ++executed_fallbacks;
                require(result.certificate_status == SafeHorizonCertificateStatus::FALLBACK_NOT_CERTIFIED,
                        "fallback inherited an SQP certificate");
                require(input->a == -1.0 && input->omega == 0.0, "unexpected fallback input");
                require(std::abs(next.v - (ego.v - cfg.mpc.dt)) < 1e-12,
                        "braking fallback did not reduce velocity");
                std::cout << "Fallback decision " << step + 1 << ": v=" << ego.v
                          << " -> " << next.v << '\n';
            }
            ego = next;
            progress = env.path.find_closest_point(ego.position(), progress);
            for (const auto& center : compute_ego_disc_positions(
                     ego, cfg.mpc.ego.num_discs, cfg.mpc.ego.length))
                require((center - obstacle.position()).norm() >= runtime.combined_radius(),
                        "executed plan collided with the stationary obstacle");
        }
        require(executed_fallbacks > 0, "did not exercise successful fallback execution");
        require(rejected == 1, "did not exercise rejection of both plans");

        // The canonical harness must stop at the failed decision, not append
        // repeated frozen states until rollout_steps is exhausted.
        cfg.rollout.rollout_steps = 40;
        const auto record = run_experiment_rollout(cfg, 77);
        require(record.termination_reason == "no_admissible_control", "missing failure outcome");
        require(record.failed_decision_step == record.total_steps + 1,
                "failed decision incorrectly counted as a propagated step");
        require(record.total_steps < 40 && !record.collision && !record.completed_path,
                "harness did not stop without claiming completion");
        std::cout << "Harness stopped at decision " << record.failed_decision_step
                  << " after " << record.total_steps << " propagated steps\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
