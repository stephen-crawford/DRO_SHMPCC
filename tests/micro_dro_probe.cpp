// Fast, deterministic behavioral probe of the DRO risk + reweighting on a fixed
// canonical scenario. Runs in milliseconds (unlike the heavy experiment suites),
// so it is usable for before/after comparison across branches.
//
// This build compares the incumbent Q* recovery (dual + greedy bracket/mix
// heuristic) against the TRUE primal optimal-transport LP (solve_primal_ot),
// swept over the Wasserstein radius rho. The two agree when the budget is slack
// (lambda*=0, all mass to the global argmax) and can diverge when the budget is
// active and symmetric modes tie -- the regime the heuristic mixing approximates.
#include "wasserstein_dro.hpp"
#include "primal_ot.hpp"
#include "dynamics.hpp"
#include "types.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>

using namespace scenario_mpc;

static double expected_risk(const std::map<std::string, double>& q,
                            const std::map<std::string, double>& r) {
    double s = 0.0;
    for (const auto& kv : q) s += kv.second * r.at(kv.first);
    return s;
}

int main() {
    auto mode_models = create_obstacle_mode_models(0.1);
    ObstacleState obs(5.0, 0.0, 0.5, 0.0);

    // Ego drives toward the obstacle so along-horizon distances land near the
    // safety margin (nonzero, differentiated per-mode risk).
    std::vector<EgoState> ego_ref;
    for (int k = 0; k <= 15; ++k) ego_ref.emplace_back(k * 0.34, 0.0, 0.0, 1.5);

    std::map<std::string, double> nominal;
    for (const auto& kv : mode_models) nominal[kv.first] = 1.0 / mode_models.size();
    std::vector<std::string> mode_ids;
    for (const auto& kv : nominal) mode_ids.push_back(kv.first);

    WassersteinDRO dro;
    std::cout << std::fixed << std::setprecision(6);

    const double rhos[] = {0.02, 0.05, 0.10, 0.15, 0.20, 0.30};
    for (double rho : rhos) {
        dro.set_rho_override(rho);
        DROResult r = dro.compute_worst_case_weights(
            nominal, obs, mode_models, ego_ref, 15, 0.5, 0.35, 0.2);

        double heur_obj = expected_risk(r.worst_case_weights, r.risk_per_mode);

        PrimalOTResult ot = solve_primal_ot(
            nominal, r.risk_per_mode, r.transport_cost_matrix, mode_ids, r.rho_used);

        // L1 distance between the two Q* vectors.
        double l1 = 0.0;
        for (const auto& id : mode_ids)
            l1 += std::abs(r.worst_case_weights.at(id) - ot.q.at(id));

        std::cout << "RHO=" << rho
                  << " lambda*=" << r.optimal_lambda
                  << " | HEUR risk=" << heur_obj
                  << " cost=" << r.implied_transport_cost
                  << " | OT risk=" << ot.expected_risk
                  << " cost=" << ot.transport_cost
                  << " | dual=" << r.worst_case_risk
                  << " | OT-HEUR=" << (ot.expected_risk - heur_obj)
                  << " Qstar_L1=" << l1 << "\n";
        if (l1 > 1e-4) {
            std::cout << "   DIVERGENCE at rho=" << rho << ":\n";
            for (const auto& id : mode_ids)
                std::cout << "     " << std::setw(20) << id
                          << "  HEUR=" << r.worst_case_weights.at(id)
                          << "  OT=" << ot.q.at(id) << "\n";
        }
    }

    // --- Crafted check: the LP genuinely does FRACTIONAL splitting ---
    // One source A (mass 1, risk 0); destination B has risk 10 at transport
    // cost 1. Budget rho=0.4 admits moving only 0.4 mass. Optimal OT plan:
    // pi_AB=0.4, pi_AA=0.6 -> Q={A:0.6, B:0.4}, risk=4.0. No feasible
    // DETERMINISTIC (bang-bang) plan achieves this (keep-all=0, move-all
    // costs 1 > rho). Confirms solve_primal_ot is the general OT solver.
    {
        std::vector<std::string> ids = {"A", "B"};
        std::map<std::string, double> pw = {{"A", 1.0}, {"B", 0.0}};
        std::map<std::string, double> rv = {{"A", 0.0}, {"B", 10.0}};
        std::vector<std::vector<double>> Dc = {{0.0, 1.0}, {1.0, 0.0}};
        PrimalOTResult c = solve_primal_ot(pw, rv, Dc, ids, 0.4);
        std::cout << "\nCRAFTED solved=" << c.solved
                  << " risk=" << c.expected_risk << " (expect 4.0)"
                  << " qA=" << c.q["A"] << " (expect 0.6)"
                  << " qB=" << c.q["B"] << " (expect 0.4)"
                  << " cost=" << c.transport_cost << " (expect 0.4)\n";
    }
    return 0;
}
