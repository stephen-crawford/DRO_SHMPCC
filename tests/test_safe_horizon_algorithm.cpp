#include "mpc_controller.hpp"
#include "experiment_config_yaml.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace dro_mpc;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
    std::cout << "PASS: " << message << '\n';
}

RuntimeConfig fixture_config(int cap, int removal) {
    RuntimeConfig cfg;
    cfg.mpc.type = MPCType::SH_MPC;
    cfg.mpc.sync_from_type();
    cfg.mpc.horizon = 3;
    cfg.mpc.dt = 1.0;
    cfg.mpc.ego.num_discs = 1;
    cfg.mpc.ego.length = 0.0;
    cfg.mpc.constraints.support_cap_n_bar = cap;
    cfg.mpc.constraints.scenario_removal_budget = removal;
    cfg.solver.sqp_max_iterations = 8;
    cfg.solver.qp_tolerance = 1e-7;
    cfg.random_seed = 41;
    return cfg;
}

CollisionConstraint upper_x(int id, double boundary) {
    CollisionConstraint row(1, 0, id, Eigen::Vector2d(-1, 0), -boundary);
    row.disc_index = 0;
    row.disc_offset = 0.0;
    return row;
}

MPCResult solve_fixture(MPCController& controller,
                       const std::vector<CollisionConstraint>& rows) {
    return controller.solve_optimization_sqp(
        EgoState(0, 0, 0, 1), Eigen::Vector2d(10, 0), 2.0,
        rows, {}, {});
}
}

int main() {
    try {
        const auto paper = yaml_config::load_experiment_config(
            "configs/safe_horizon_paper.yaml", true).to_scenario_mpc_config();
        require(paper.mpc.horizon == 20 && paper.mpc.dt == 0.2 &&
                    paper.solver.sqp_max_iterations == 15 &&
                    paper.scenario_removal_budget() == 1 && paper.support_limit() == 9 &&
                    paper.mpc.sampling.num_scenarios == 1237 && !paper.dro.enabled,
                "paper profile uses N=20, dt=0.2, L=15, R=1, total cap=9 and S=1237");
        require(paper.mpc.objective.progress_weight == 0.0 &&
                    paper.mpc.objective.velocity_weight == 0.05 &&
                    paper.mpc.objective.steering_weight == 0.05 &&
                    paper.mpc.objective.contour_weight == 0.001 &&
                    paper.mpc.objective.lag_weight == 0.1,
                "paper profile loads Table II Gaussian objective weights");
        require(paper.mpc.execution_dt() == 0.05,
                "paper controller executes at 20 Hz with a 0.2-second prediction grid");
        auto timing_cfg = fixture_config(2, 0);
        require(timing_cfg.mpc.execution_dt() == timing_cfg.mpc.dt,
                "existing profiles inherit their prediction timestep for execution");
        MPCController inherited_timing(timing_cfg);
        inherited_timing.last_feasible_controls_ = {EgoInput(0.1, 0), EgoInput(0.2, 0), EgoInput(0.3, 0)};
        const auto whole_shift = inherited_timing.shifted_backup_controls();
        require(whole_shift.front().a == 0.2 && whole_shift.back().a == 0.3,
                "inherited execution timing retains the original whole-stage backup shift");
        timing_cfg.mpc.dt = 0.2;
        timing_cfg.mpc.control_dt = 0.05;
        MPCController timing(timing_cfg);
        timing.last_feasible_controls_ = {EgoInput(0.1, 0), EgoInput(0.2, 0), EgoInput(0.3, 0)};
        timing.has_feasible_backup_ = true;
        timing.initialize_reference_trajectory(EgoState(0, 0, 0, 1), Eigen::Vector2d(10, 0), 1);
        require(timing.shifted_backup_controls().front().a == 0.1 &&
                    std::abs(timing.reference_trajectory_[1].v - 1.02) < 1e-12,
                "fractional execution interval preserves held controls and re-rolls on prediction grid");
        EgoDynamics plant(timing_cfg.mpc.ego.dynamics, timing_cfg.mpc.execution_dt());
        require(std::abs(plant.propagate(EgoState(0, 0, 0, 1), EgoInput(0.1, 0)).v - 1.005) < 1e-12,
                "executed velocity increment uses 0.05 seconds, not prediction dt");
        const auto gaussian = create_gaussian_position_mode_models(0.2, 0.3);
        const auto& moving = gaussian.at("constant_velocity");
        Eigen::VectorXd noise(2);
        noise << 1, -2;
        const auto next = moving.propagate(ObstacleState(6, 0, 1, -1), &noise);
        const double scale = 0.2 * std::sqrt(0.3);
        require(std::abs(next.x - (6.2 + scale)) < 1e-12 &&
                    std::abs(next.y - (-0.2 - 2 * scale)) < 1e-12 &&
                    next.vx == 1 && next.vy == -1,
                "Eq. 36 adds Gaussian noise only to position and preserves velocity");
        require(std::abs((moving.G * moving.G.transpose())(0, 0) - 0.012) < 1e-12,
                "increment covariance equals dt squared times Sigma_w");
        const auto stationary = gaussian.at("stop").propagate(ObstacleState(6, 0, 0, 0), &noise);
        require(std::abs(stationary.x - (6 + scale)) < 1e-12 &&
                    std::abs(stationary.y + 2 * scale) < 1e-12,
                "Eq. 34 static prediction uses Gaussian position random walk");
        const auto static_profile = yaml_config::load_experiment_config(
            "configs/safe_horizon_paper_static.yaml", true);
        require(static_profile.obstacles.prediction_model == "gaussian_position" &&
                    static_profile.mpc.objective.contour_weight == 0.02 &&
                    static_profile.obstacles.position_noise_variance == 0.3,
                "static paper profile selects Eq. 34 and Table II static weights");
        // The x<=1.1 row is initially redundant behind x<=0.6. Removing the
        // first joint scenario must expose the second, rather than lose it.
        const std::vector<CollisionConstraint> rows{
            upper_x(11, 0.6), upper_x(12, 1.1)};
        MPCController controller(fixture_config(2, 1));
        controller.set_capture_linearized_constraints(true);
        const auto result = solve_fixture(controller, rows);
        std::cout << "REMOVAL success=" << result.success
                  << " fallback=" << result.used_fallback
                  << " x1=" << result.ego_trajectory.at(1).x
                  << " support=" << result.support_size
                  << " evaluations=" << result.support_iterations_evaluated << '\n';
        require(result.success && !result.used_fallback,
                "retained-scenario SQP plan is executable");
        require(controller.last_removed_scenario_ids() == std::set<int>{11},
                "active removal respects cumulative joint-scenario budget R=1");
        require(result.support_scenarios == std::vector<int>({11, 12}),
                "removed scenario and newly exposed facet both remain in support");
        require(result.support_iterations_feasible,
                "support comes from feasible convex QP solutions");
        require(result.ego_trajectory.at(1).x > 0.6 &&
                    result.ego_trajectory.at(1).x <= 1.1 + 1e-7,
                "removal improves the plan while the newly exposed row stays enforced");
        const auto expected = EgoDynamics(controller.config().mpc.dt).rollout(
            EgoState(0, 0, 0, 1), result.control_inputs);
        for (std::size_t k = 0; k < expected.size(); ++k)
            require((expected[k].to_array() - result.ego_trajectory[k].to_array()).norm() < 1e-12,
                    "returned controls reproduce the nonlinear trajectory");

        // With total cap 1, the newly exposed scenario exceeds the cap at
        // the next QP. Algorithm 1 must return the earlier feasible iterate.
        MPCController capped(fixture_config(0, 1));
        const auto stopped = solve_fixture(capped, rows);
        std::cout << "CAP success=" << stopped.success
                  << " fallback=" << stopped.used_fallback
                  << " x1=" << stopped.ego_trajectory.at(1).x
                  << " support=" << stopped.support_size
                  << " iteration=" << stopped.returned_sqp_iteration << '\n';
        require(stopped.support_cap_terminated && stopped.success && !stopped.used_fallback,
                "cap exceedance returns a previous feasible SQP iterate");
        require(stopped.returned_sqp_iteration == 0 &&
                    stopped.support_iterations_evaluated == 1 &&
                    stopped.support_scenarios == std::vector<int>{11},
                "saved iterate retains its own support and iteration count");
        require(capped.last_removed_scenario_ids().empty() &&
                    stopped.ego_trajectory.at(1).x <= 0.6 + 1e-7,
                "returning the saved iterate restores its removal set and constraints");
        auto one_cfg = fixture_config(2, 0);
        one_cfg.solver.sqp_max_iterations = 1;
        MPCController one_iteration(one_cfg);
        const auto first = solve_fixture(one_iteration, rows);
        for (std::size_t k = 0; k < first.control_inputs.size(); ++k)
            require((first.control_inputs[k].to_array() - stopped.control_inputs[k].to_array()).norm() < 1e-12,
                    "cap termination returns the saved controls exactly");
        for (std::size_t k = 0; k < first.ego_trajectory.size(); ++k)
            require((first.ego_trajectory[k].to_array() - stopped.ego_trajectory[k].to_array()).norm() < 1e-12,
                    "cap termination returns the saved trajectory exactly");

        MPCController zero_cap(fixture_config(0, 0));
        const auto no_previous = solve_fixture(zero_cap, {upper_x(11, 0.6)});
        require(no_previous.support_cap_terminated && no_previous.used_fallback,
                "cap exceedance without a saved eligible iterate uses slowing fallback");

        // x1>=100 cannot be reached within the acceleration/velocity boxes.
        CollisionConstraint impossible(1, 0, 23, Eigen::Vector2d(1, 0), 100.0);
        MPCController infeasible(fixture_config(2, 0));
        const auto rejected = solve_fixture(infeasible, {impossible});
        require(!rejected.success && !rejected.support_iterations_feasible &&
                    rejected.support_cap_status == SupportCapStatus::NOT_EVALUATED,
                "infeasible convex problem never receives support certification");

        MPCController physical_fallback(fixture_config(2, 0));
        std::vector<PredictionStep> far_steps;
        for (int k = 0; k <= 3; ++k)
            far_steps.emplace_back(k, Eigen::Vector2d(10, 10), Eigen::Matrix2d::Zero());
        physical_fallback.scenarios_ = {Scenario(23, {{0, ObstacleTrajectory(0, "stop", far_steps)}})};
        const auto braking = solve_fixture(physical_fallback, {impossible});
        require(braking.success && braking.used_fallback && braking.certificate_status != SafeHorizonCertificateStatus::CERTIFIED &&
                    braking.failure_diagnostics.fallback_sampled_collision_feasible == 1,
                "uncertified slowing fallback is admitted by full sampled disc geometry");
        const auto brake_trajectory = physical_fallback.generate_safe_fallback(EgoState(0, 0, 0, 1)).ego_trajectory;
        std::vector<PredictionStep> collision_steps;
        for (int k = 0; k <= 3; ++k)
            collision_steps.emplace_back(k, brake_trajectory[k].position(), Eigen::Matrix2d::Zero());
        physical_fallback.scenarios_ = {Scenario(23, {{0, ObstacleTrajectory(0, "stop", collision_steps)}})};
        const auto blocked = solve_fixture(physical_fallback, {impossible});
        require(!blocked.success && blocked.used_fallback && blocked.certificate_status != SafeHorizonCertificateStatus::CERTIFIED,
                "slowing fallback colliding with any original sampled obstacle is rejected");

        MPCController removable(fixture_config(2, 1));
        const auto recovered = solve_fixture(removable, {impossible});
        require(recovered.success && !recovered.used_fallback &&
                    removable.last_removed_scenario_ids() == std::set<int>{23} &&
                    recovered.support_scenarios == std::vector<int>{23},
                "infeasibility removal is retained in support even without active collision rows");

        auto failed_cfg = fixture_config(2, 0);
        failed_cfg.solver.qp_max_iterations = 1;
        MPCController failed_solver(failed_cfg);
        const auto failed_qp = solve_fixture(failed_solver, {upper_x(11, 0.6)});
        require(failed_qp.failure_diagnostics.last_qp_converged == 0 &&
                    !failed_qp.support_iterations_feasible &&
                    failed_qp.support_cap_status == SupportCapStatus::NOT_EVALUATED,
                "a finite unconverged QP vector cannot produce a support certificate");

        // At rest, an offset disc obeys y_disc=sin(theta). The QP requires
        // theta>=0.3, but sin(0.3)<0.3 in the nonlinear rollout. QP activity
        // must still be counted; the nonlinear plan must be rejected.
        auto nonlinear_cfg = fixture_config(2, 0);
        nonlinear_cfg.mpc.horizon = 1;
        nonlinear_cfg.mpc.ego.num_discs = 3;
        nonlinear_cfg.mpc.ego.length = 2.0;
        nonlinear_cfg.solver.sqp_max_iterations = 1;
        MPCController nonlinear(nonlinear_cfg);
        CollisionConstraint heading_row(1, 0, 31, Eigen::Vector2d(0, 1), 0.3);
        heading_row.disc_index = 2;
        heading_row.disc_offset = 1.0;
        const auto curved = nonlinear.solve_optimization_sqp(
            EgoState(0, 0, 0, 0), Eigen::Vector2d(0, 0), 0.0,
            {heading_row}, {}, {EgoInput(0, 0)});
        require(curved.support_scenarios == std::vector<int>{31} &&
                    curved.support_iterations_feasible && !curved.success && curved.used_fallback,
                "convex activity is counted independently of nonlinear rollout feasibility");
        auto nonconvex_cfg = fixture_config(2, 0);
        nonconvex_cfg.mpc.objective.contour_weight = -1.0;
        bool rejected_weight = false;
        try {
            MPCController invalid(nonconvex_cfg);
        } catch (const std::invalid_argument&) {
            rejected_weight = true;
        }
        require(rejected_weight, "negative squared-term weights cannot violate the convex-iteration assumption");
        std::cout << "ALL SAFE-HORIZON ALGORITHM TESTS PASSED\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
