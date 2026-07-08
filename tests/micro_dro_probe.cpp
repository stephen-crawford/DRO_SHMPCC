// Fast, deterministic behavioral probe of the DRO risk + reweighting on a fixed
// canonical scenario. Runs in milliseconds (unlike the heavy experiment suites),
// so it is usable for before/after comparison across branches (VaR vs CVaR risk;
// dual+mixing vs true-primal-LP reweighting). Prints the risk vector r[m], the
// worst-case weights Q*, the worst-case risk, and the induced transport cost.
#include "wasserstein_dro.hpp"
#include "dynamics.hpp"
#include "types.hpp"

#include <iostream>
#include <iomanip>

using namespace scenario_mpc;

int main() {
    auto mode_models = create_obstacle_mode_models(0.1);
    ObstacleState obs(5.0, 0.0, 0.5, 0.0);

    // Ego drives toward the obstacle at (5,0) so that along-horizon distances
    // land near the safety margin -- the regime where VaR vs CVaR risk actually
    // differ (the z_alpha * sigma_dir term flips risk from 0 to positive).
    std::vector<EgoState> ego_ref;
    for (int k = 0; k <= 15; ++k) ego_ref.emplace_back(k * 0.34, 0.0, 0.0, 1.5);

    std::map<std::string, double> nominal;
    for (const auto& kv : mode_models) nominal[kv.first] = 1.0 / mode_models.size();

    WassersteinDRO dro;
    DROResult r = dro.compute_worst_case_weights(
        nominal, obs, mode_models, ego_ref, /*horizon=*/15,
        /*ego_r=*/0.5, /*obs_r=*/0.35, /*margin=*/0.2);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "SCALAR worst_case_risk=" << r.worst_case_risk
              << " rho_used=" << r.rho_used
              << " optimal_lambda=" << r.optimal_lambda
              << " implied_transport_cost=" << r.implied_transport_cost
              << " recovery_feasible=" << (r.recovery_feasible ? 1 : 0) << "\n";
    for (const auto& kv : r.risk_per_mode)
        std::cout << "RISK " << kv.first << " " << kv.second << "\n";
    for (const auto& kv : r.worst_case_weights)
        std::cout << "QSTAR " << kv.first << " " << kv.second
                  << " nominal " << nominal[kv.first] << "\n";
    return 0;
}
