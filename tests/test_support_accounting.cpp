/**
 * @file test_support_accounting.cpp
 * @brief Integration regression for Safe-Horizon support accounting.
 */

#include "mpc_controller.hpp"

#include <algorithm>
#include <cstdio>
#include <map>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
    if (!condition) ++failures;
}

}  // namespace

int main() {
    // A stationary obstacle directly in the initial path makes at least one
    // sampled joint scenario binding/violated.  A zero support cap therefore
    // exercises the reference-style SUPPORT_EXCEEDED path deterministically.
    RuntimeConfig cfg;
    cfg.mpc.type = MPCType::SH_MPC;
    cfg.mpc.safe_horizon_enabled = true;
    cfg.mpc.horizon = 8;
    cfg.mpc.dt = 0.1;
    cfg.mpc.sampling.set_manual_sample_count(8);
    cfg.mpc.constraints.support_cap_n_bar = 0;
    cfg.mpc.constraints.scenario_removal_budget = 0;
    cfg.mpc.constraints.clearance_filter_distance = 100.0;
    cfg.solver.sqp_max_iterations = 3;
    cfg.solver.qp_max_iterations = 100;
    cfg.random_seed = 7193u;

    MPCController controller(cfg);
    const std::map<int, ObstacleState> obstacles{
        {7, ObstacleState(0.2, 0.0, 0.0, 0.0)},
    };
    const MPCResult result = controller.solve(
        EgoState(0.0, 0.0, 0.0, 0.0), obstacles,
        Eigen::Vector2d(4.0, 0.0), 0.0);

    std::printf("support=%d/%d, evaluations=%d, status=%d\n",
                result.support_size, result.support_limit,
                result.support_iterations_evaluated,
                static_cast<int>(result.support_cap_status));

    check(result.certified_horizon == -1 &&
              result.certificate_status ==
                  SafeHorizonCertificateStatus::INSUFFICIENT_SCENARIOS,
          "an undersized manual scenario set is not reported as certified");
    check(result.sampled_scenarios == 8 &&
              result.required_scenarios > result.sampled_scenarios &&
              !result.sample_count_sufficient,
          "result exposes the sampled and required counts used for certification");
    check(result.support_limit == cfg.support_limit(),
          "result reports the configured total support cap");
    check(result.support_iterations_evaluated > 0,
          "support is evaluated over SQP iterates and the returned plan");
    check(result.support_size == static_cast<int>(result.support_scenarios.size()),
          "reported support size equals the scenario-ID union size");
    check(std::is_sorted(result.support_scenarios.begin(), result.support_scenarios.end()),
          "support scenario IDs are emitted in deterministic sorted order");
    check(std::adjacent_find(result.support_scenarios.begin(),
                             result.support_scenarios.end()) == result.support_scenarios.end(),
          "support scenario IDs are unique across stages, discs, and SQP iterates");
    check(result.support_size > 0,
          "binding or violated sampled scenarios contribute to the support union");
    check(!result.support_cap_satisfied &&
              result.support_cap_status == SupportCapStatus::SUPPORT_EXCEEDED,
          "a support union above n-bar reports SUPPORT_EXCEEDED without changing numerical success");

    {
        // Each receding-horizon decision must start its scenario predictions
        // at the current obstacle measurement, rather than retaining a stale
        // sampled trajectory from the previous solve.
        RuntimeConfig refresh_cfg;
        refresh_cfg.mpc.type = MPCType::MPC;
        refresh_cfg.mpc.sync_from_type();
        refresh_cfg.mpc.horizon = 3;
        refresh_cfg.mpc.dt = 0.1;
        refresh_cfg.mpc.sampling.set_manual_sample_count(4);
        refresh_cfg.mpc.constraints.clearance_filter_distance = 100.0;
        refresh_cfg.solver.sqp_max_iterations = 2;
        refresh_cfg.random_seed = 2718u;
        MPCController refresh_controller(refresh_cfg);

        const std::map<int, ObstacleState> first_obstacles{
            {11, ObstacleState(2.0, 0.5, 0.0, 0.0)},
        };
        (void)refresh_controller.solve(
            EgoState(0.0, 0.0, 0.0, 0.0), first_obstacles,
            Eigen::Vector2d(4.0, 0.0), 0.0);
        bool first_state_matches = !refresh_controller.scenarios().empty();
        for (const Scenario& scenario : refresh_controller.scenarios()) {
            const auto& trajectory = scenario.trajectories.at(11);
            first_state_matches = first_state_matches &&
                (trajectory.steps.front().mean - Eigen::Vector2d(2.0, 0.5)).norm() < 1e-12;
        }

        const std::map<int, ObstacleState> second_obstacles{
            {11, ObstacleState(7.0, -1.0, 0.0, 0.0)},
        };
        (void)refresh_controller.solve(
            EgoState(0.0, 0.0, 0.0, 0.0), second_obstacles,
            Eigen::Vector2d(4.0, 0.0), 0.0);
        bool refreshed_state_matches = !refresh_controller.scenarios().empty();
        for (const Scenario& scenario : refresh_controller.scenarios()) {
            const auto& trajectory = scenario.trajectories.at(11);
            refreshed_state_matches = refreshed_state_matches &&
                (trajectory.steps.front().mean - Eigen::Vector2d(7.0, -1.0)).norm() < 1e-12;
        }
        check(first_state_matches && refreshed_state_matches,
              "scenario trajectories are refreshed from the current obstacle state each solve");
    }

    {
        // Direct RuntimeConfig construction must take the same automatic
        // certification path as YAML/harness conversion before sampling.
        RuntimeConfig automatic_cfg;
        automatic_cfg.mpc.type = MPCType::SH_MPC;
        automatic_cfg.mpc.safe_horizon_enabled = true;
        automatic_cfg.mpc.horizon = 3;
        automatic_cfg.mpc.sampling.automatically_compute_sample_size = true;
        automatic_cfg.mpc.constraints.support_cap_n_bar = 1;
        automatic_cfg.mpc.constraints.scenario_removal_budget = 1;
        automatic_cfg.solver.sqp_max_iterations = 2;
        automatic_cfg.random_seed = 17u;

        MPCController automatic_controller(automatic_cfg);
        const MPCResult automatic_result = automatic_controller.solve(
            EgoState(0.0, 0.0, 0.0, 0.0), {},
            Eigen::Vector2d(1.0, 0.0), 0.0);
        const int required = automatic_controller.config().compute_required_scenarios();
        check(static_cast<int>(automatic_controller.scenarios().size()) == required,
              "direct RuntimeConfig construction auto-sizes S before sampling");
        check(automatic_result.support_limit == 2 &&
                  automatic_result.support_size == 0 &&
                  automatic_result.support_cap_satisfied &&
                  automatic_result.certificate_status ==
                      SafeHorizonCertificateStatus::CERTIFIED &&
                  automatic_result.certified_horizon ==
                      automatic_cfg.mpc.horizon,
              "a full-horizon zero-support solve satisfies the configured n-bar + R cap");
    }

    std::printf("\n%s (%d checks failed)\n",
                failures == 0 ? "ALL SUPPORT-ACCOUNTING TESTS PASSED"
                              : "SOME SUPPORT-ACCOUNTING TESTS FAILED",
                failures);
    return failures == 0 ? 0 : 1;
}
