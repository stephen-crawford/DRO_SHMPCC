#include "mpc_controller.hpp"
#include "certification_snapshot.hpp"
#include <iostream>
#include <stdexcept>
using namespace dro_mpc;
int main() {
    RuntimeConfig cfg;
    cfg.mpc.type = MPCType::SH_MPC;
    cfg.mpc.safe_horizon_enabled = true;
    cfg.mpc.horizon = 3;
    cfg.mpc.sampling.set_manual_sample_count(20);
    cfg.mpc.constraints.support_cap_n_bar = 6;
    cfg.mpc.constraints.scenario_removal_budget = 2;
    cfg.solver.sqp_max_iterations = 2;
    MPCController controller(cfg);
    const std::set<int> removed{3, 17};
    // Exercise the actual SQP support-seeding path used by scenario removal.
    const auto result = controller.solve_optimization_sqp(
        EgoState(0,0,0,0), Eigen::Vector2d(1,0), 0, {}, removed, {}, 0, 10);
    if (result.support_scenarios != std::vector<int>({3,17}) || result.support_size != 2 ||
        result.support_iterations_evaluated == 0 || !result.active_scenarios.empty())
        throw std::runtime_error("removed IDs must enter the measured support exactly once");
    SolveAttemptDiagnostics attempt;
    const auto text = diagnostic::certification_snapshot(cfg, result, attempt, {}, {}, removed);
    if (text.find("\"removed_ids\":[3,17]") == std::string::npos ||
        text.find("\"support_union\":[3,17]") == std::string::npos ||
        text.find("\"support_count_used_for_certificate\":2") == std::string::npos)
        throw std::runtime_error("snapshot support provenance mismatch");
    std::cout << "PASS: actual SQP support_active=[] removed_ids=[3,17] support_union=[3,17] support_count_used_for_certificate=2\n";
}
