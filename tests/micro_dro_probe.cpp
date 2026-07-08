// Fast, deterministic behavioral probe of the DRO risk + reweighting on a fixed
// canonical scenario. Runs in milliseconds (unlike the heavy experiment suites),
// so it is usable for before/after comparison across branches.
//
// Merged probe (lcss-mode-coverage). Exercises both mechanisms that landed here:
//
//   Section 1 -- RISK MODEL (VaR vs CVaR). Prints the scalar worst-case risk, the
//     per-mode risk vector r[m], and Q*. On the cvar-risk lineage the directional
//     margin uses the CVaR/expected-shortfall coefficient k_alpha = phi(z)/(1-alpha)
//     instead of the one-sided VaR quantile z_alpha, so every r[m] should be
//     LARGER here than on a VaR build (k=2.063 vs z=1.645 at alpha=0.95).
//
//   Section 2 -- Q* RECOVERY (heuristic vs true primal OT). Compares the incumbent
//     recovery (dual + greedy bracket/mix, a heuristic restricted to convex
//     mixtures of two deterministic plans) against the TRUE primal optimal-transport
//     LP (solve_primal_ot), swept over the Wasserstein radius rho. The two agree
//     when the budget is slack (lambda*=0, all mass to the global argmax) and can
//     diverge when the budget is active and symmetric modes tie -- the regime the
//     heuristic mixing approximates.
//
//   Section 3 -- CRAFTED fractional-split check that no bang-bang plan can match.
//
// The calibrated ambiguity radius is probed separately by tests/radius_probe.cpp.
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

    // Ego drives toward the obstacle at (5,0) so along-horizon distances land near
    // the safety margin -- the regime where VaR vs CVaR risk actually differ (the
    // coefficient * sigma_dir term flips risk from 0 to positive), and where
    // per-mode risk is nonzero and differentiated.
    std::vector<EgoState> ego_ref;
    for (int k = 0; k <= 15; ++k) ego_ref.emplace_back(k * 0.34, 0.0, 0.0, 1.5);

    std::map<std::string, double> nominal;
    for (const auto& kv : mode_models) nominal[kv.first] = 1.0 / mode_models.size();
    std::vector<std::string> mode_ids;
    for (const auto& kv : nominal) mode_ids.push_back(kv.first);

    WassersteinDRO dro;
    std::cout << std::fixed << std::setprecision(6);

    // ---------------------------------------------------------------------
    // Section 1 -- risk model (VaR vs CVaR) at the default radius.
    // ---------------------------------------------------------------------
    std::cout << "=== SECTION 1: risk model (VaR vs CVaR) ===\n";
    {
        DROResult r = dro.compute_worst_case_weights(
            nominal, obs, mode_models, ego_ref, /*horizon=*/15,
            /*ego_r=*/0.5, /*obs_r=*/0.35, /*margin=*/0.2);

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
    }

    // ---------------------------------------------------------------------
    // Section 2 -- Q* recovery: heuristic vs true primal OT, swept over rho.
    // ---------------------------------------------------------------------
    std::cout << "\n=== SECTION 2: Q* recovery (heuristic vs true primal OT) ===\n";
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

    // ---------------------------------------------------------------------
    // Section 3 -- Crafted check: the LP genuinely does FRACTIONAL splitting.
    // One source A (mass 1, risk 0); destination B has risk 10 at transport
    // cost 1. Budget rho=0.4 admits moving only 0.4 mass. Optimal OT plan:
    // pi_AB=0.4, pi_AA=0.6 -> Q={A:0.6, B:0.4}, risk=4.0. No feasible
    // DETERMINISTIC (bang-bang) plan achieves this (keep-all=0, move-all
    // costs 1 > rho). Confirms solve_primal_ot is the general OT solver.
    // ---------------------------------------------------------------------
    std::cout << "\n=== SECTION 3: crafted fractional-split check ===\n";
    {
        std::vector<std::string> ids = {"A", "B"};
        std::map<std::string, double> pw = {{"A", 1.0}, {"B", 0.0}};
        std::map<std::string, double> rv = {{"A", 0.0}, {"B", 10.0}};
        std::vector<std::vector<double>> Dc = {{0.0, 1.0}, {1.0, 0.0}};
        PrimalOTResult c = solve_primal_ot(pw, rv, Dc, ids, 0.4);
        std::cout << "CRAFTED solved=" << c.solved
                  << " risk=" << c.expected_risk << " (expect 4.0)"
                  << " qA=" << c.q["A"] << " (expect 0.6)"
                  << " qB=" << c.q["B"] << " (expect 0.4)"
                  << " cost=" << c.transport_cost << " (expect 0.4)\n";
    }
    return 0;
}
