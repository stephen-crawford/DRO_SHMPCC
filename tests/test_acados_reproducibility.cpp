/**
 * @file test_acados_reproducibility.cpp
 * @brief End-to-end reproducibility regression for the acados controller path.
 */

#include "experiment_harness.hpp"
#include "qp_solver.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace dro_mpc;

namespace {

bool same(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance;
}

int fail(const char* message) {
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

}  // namespace

int main() {
    ExperimentConfig config = default_experiment_config();
    config.rollout.rollout_steps = 8;
    config.mpc.horizon = 10;
    config.mpc.sampling.set_manual_sample_count(12);
    config.dro.enabled = true;

    constexpr unsigned master_seed = 914273u;
    const RolloutRecord first = run_experiment_rollout(config, master_seed);
    const RolloutRecord second = run_experiment_rollout(config, master_seed);

    if (first.seed != master_seed || first.seed != second.seed ||
        first.plant_seed != second.plant_seed ||
        first.predictor_seed != second.predictor_seed ||
        first.controller_seed != second.controller_seed)
        return fail("master and derived seed streams are not reproducible");
    if (first.plant_seed == first.predictor_seed ||
        first.plant_seed == first.controller_seed ||
        first.predictor_seed == first.controller_seed)
        return fail("plant, predictor, and controller seeds must remain separate");
    if (first.qp_backend != AcadosQPSolver::backend_name() ||
        first.qp_backend != second.qp_backend ||
        first.qp_solver_identity.empty() ||
        first.qp_solver_identity != second.qp_solver_identity)
        return fail("acados backend identity was not recorded reproducibly");
    if (first.risk_scoring_model != "inherit" ||
        first.configured_risk_measure != "surrogate_var_bonferroni" ||
        first.effective_risk_measure != "surrogate_var_bonferroni" ||
        first.avg_dro_risk_ms < 0.0)
        return fail("DRO risk-scoring configuration/diagnostics were not logged");
    if (first.safe_horizon_decisions != first.total_steps ||
        first.certified_decisions < 0 ||
        first.certified_decisions > first.safe_horizon_decisions ||
        (first.safe_horizon_decisions > 0 &&
         !same(first.certificate_rate,
               static_cast<double>(first.certified_decisions) /
                   first.safe_horizon_decisions)) ||
        (first.certified_decisions == 0 && first.avg_certified_horizon != 0.0) ||
        (first.certified_decisions > 0 &&
         !same(first.avg_certified_horizon, config.mpc.horizon)))
        return fail("Safe-Horizon certificate validity was not logged truthfully");

    // Wall-clock solve times intentionally vary; all model/RNG-driven outcomes
    // must not. These values cover controller inputs, scenario sampling, plant
    // propagation, collision accounting, and configured DRO radius use.
    if (first.collision != second.collision ||
        first.collision_step != second.collision_step ||
        first.total_steps != second.total_steps ||
        first.completed_path != second.completed_path ||
        first.active_constraints != second.active_constraints ||
        first.safe_horizon_decisions != second.safe_horizon_decisions ||
        first.certified_decisions != second.certified_decisions ||
        !same(first.eps_wass, second.eps_wass) ||
        !same(first.certificate_rate, second.certificate_rate) ||
        !same(first.avg_certified_horizon, second.avg_certified_horizon) ||
        !same(first.min_clearance, second.min_clearance) ||
        !same(first.total_progress, second.total_progress) ||
        !same(first.control_effort, second.control_effort))
        return fail("same seed/config did not reproduce the rollout outcome");

    std::cout << "PASS: acados rollout is reproducible with logged backend identity\n";
    return 0;
}
