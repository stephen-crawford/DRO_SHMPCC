#include "experiment_config_yaml.hpp"
#include "mpc_controller.hpp"
#include <iostream>
#include <stdexcept>
using namespace dro_mpc;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    // A close stationary obstacle makes ordinary SQP fail; the existing
    // recovery selects a collision-checked braking candidate and retries SQP.
    RuntimeConfig recovery_cfg;
    recovery_cfg.mpc.type = MPCType::SH_MPCC_DRO_FALLBACK;
    recovery_cfg.mpc.horizon = 10;
    recovery_cfg.mpc.sampling.set_manual_sample_count(8);
    recovery_cfg.mpc.ego.num_discs = 1;
    recovery_cfg.mpc.ego.length = 0;
    recovery_cfg.solver.sqp_max_iterations = 1;
    recovery_cfg.random_seed = 77;
    MPCController recovery_controller(recovery_cfg);
    ModeModel stop("stop", Eigen::Matrix4d::Identity(), Eigen::Vector4d::Zero(),
                   Eigen::MatrixXd::Zero(4, 2));
    recovery_controller.initialize_obstacle(0, 0, {{"stop", stop}});
    const auto braking = recovery_controller.solve(EgoState(0, 0, 0, 0),
        {{0, ObstacleState(1, 0, 0, 0)}}, Eigen::Vector2d(10, 0));
    require(braking.success && braking.sampled_constraints_satisfied &&
            braking.failure_diagnostics.braking_collision_feasible == 1 &&
            braking.failure_diagnostics.any_homotopy_geometrically_feasible == 1 &&
            !braking.nominal_fallback_attempted,
            "admissible DRO recovery must skip nominal fallback");
    std::cout << "PASS: admissible DRO braking/homotopy recovery skips nominal\n";

    RuntimeConfig cfg;
    cfg.mpc.type = yaml_config::parse_mpc("sh_mpcc_dro_fallback");
    cfg.mpc.sync_from_type();
    cfg.mpc.horizon = 4;
    cfg.mpc.sampling.set_manual_sample_count(8);
    cfg.mpc.ego.num_discs = 1;
    cfg.mpc.ego.length = 0;
    cfg.dro.fixed_rho = 100;
    cfg.random_seed = 77;
    MPCController controller(cfg);
    require(controller.config().dro.enabled, "fallback type must enable DRO");
    const EgoState ego(0, 0, 0, 0);
    const Eigen::Vector2d goal(10, 0);
    auto clear = controller.solve(ego, {}, goal);
    require(clear.success && !clear.nominal_fallback_attempted,
            "admissible DRO plan must skip nominal retry");
    std::cout << "PASS: clear DRO plan skips nominal\n";

    // Deterministic modes: nominal stays far away; rare threat jumps onto ego.
    ModeModel safe("safe", Eigen::Matrix4d::Identity(), Eigen::Vector4d::Zero(),
                   Eigen::MatrixXd::Zero(4, 2));
    ModeModel threat("threat", Eigen::Matrix4d::Zero(), Eigen::Vector4d::Zero(),
                     Eigen::MatrixXd::Zero(4, 2));
    controller.initialize_obstacle(0, 0, {{"safe", safe}, {"threat", threat}});
    for (int i = 0; i < 1000; ++i)
        controller.update_mode_observation(0, 0, "safe", i);
    const auto fallback = controller.solve(ego, {{0, ObstacleState(5, 0, 0, 0)}}, goal);
    std::cout << "fallback attempted=" << fallback.nominal_fallback_attempted
              << " used=" << fallback.used_nominal_fallback
              << " success=" << fallback.success << '\n';
    require(fallback.nominal_fallback_attempted && fallback.used_nominal_fallback && fallback.success,
            "exhausted DRO must execute admissible nominal plan");
    require(fallback.sampled_constraints_satisfied && fallback.first_input().has_value(),
            "nominal control must pass existing acceptance");
    require(controller.last_dro_results().empty(), "nominal diagnostics must not report DRO weights");
    for (const auto& scenario : controller.scenarios())
        require(scenario.trajectories.at(0).steps.at(1).mean.x() == 5,
                "nominal retry must draw nominal scenarios");
    auto resumed = controller.solve(ego, {{0, ObstacleState(5, 0, 0, 0)}}, goal);
    require(resumed.nominal_fallback_attempted,
            "next step must retry DRO even after nominal execution");
    controller.initialize_obstacle(0, 0, {{"safe", safe}});
    auto recovered = controller.solve(ego, {{0, ObstacleState(5, 0, 0, 0)}}, goal);
    require(recovered.success && !recovered.nominal_fallback_attempted &&
            !controller.last_dro_results().empty(), "must immediately resume admissible DRO");
    std::cout << "PASS: nominal sampling and immediate DRO resumption\n";

    controller.initialize_obstacle(0, 0, {{"threat", threat}});
    const auto rejected = controller.solve(ego, {{0, ObstacleState(5, 0, 0, 0)}}, goal);
    require(rejected.nominal_fallback_attempted && !rejected.used_nominal_fallback && !rejected.success,
            "both failed attempts must retain terminal rejection");
    require(controller.get_statistics().iteration_count == 5,
            "fallback retries must count as one control decision");
    std::cout << "PASS: both attempts rejected; five decisions counted\n";
}
