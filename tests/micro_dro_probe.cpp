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
#include "dro.hpp"
#include "primal_ot.hpp"
#include "dynamics.hpp"
#include "types.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>

using namespace dro_mpc;

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

    DRO dro;
    std::cout << std::fixed << std::setprecision(6);

    // ---------------------------------------------------------------------
    // Section 1 -- the four risk measures on the identical scenario.
    // SURROGATE_* : per-step linearised, max over (k,d)  [not a trajectory risk measure]
    // JOINT_*     : true risk of the joint-horizon Euclidean violation  [MC]
    // CVaR >= VaR within each family (always -- CVaR is a tail mean past the VaR).
    // joint/surr < 1 here: the surrogate's linearisation conservatism outweighs its
    // max-of-marginals optimism on this scenario, so it sits ~2-5% high. Not a
    // theorem -- the two errors oppose and the winner is scenario-dependent.
    // ---------------------------------------------------------------------
    std::cout << "=== SECTION 1: four risk measures ===\n";
    {
        auto risk_of = [&](DRORiskMeasure m) {
            DROConfig cfg; cfg.radius_calibration.risk_measure = m;
            DRO d(cfg);
            return d.compute_worst_case_weights(
                nominal, obs, mode_models, ego_ref, 15, 0.5, 0.35, 0.2);
        };
        DROResult sv = risk_of(DRORiskMeasure::SURROGATE_VAR);
        DROResult sc = risk_of(DRORiskMeasure::SURROGATE_CVAR);
        DROResult jv = risk_of(DRORiskMeasure::JOINT_VAR);
        DROResult jc = risk_of(DRORiskMeasure::JOINT_CVAR);

        std::cout << std::setw(20) << "mode" << std::setw(12) << "surrVaR"
                  << std::setw(12) << "surrCVaR" << std::setw(12) << "jointVaR"
                  << std::setw(12) << "jointCVaR" << std::setw(12) << "joint/surr" << "\n";
        for (const auto& kv : sv.risk_per_mode) {
            const std::string& id = kv.first;
            const double a = kv.second, b = sc.risk_per_mode.at(id);
            const double c = jv.risk_per_mode.at(id), d = jc.risk_per_mode.at(id);
            std::cout << std::setw(20) << id << std::setw(12) << a << std::setw(12) << b
                      << std::setw(12) << c << std::setw(12) << d
                      << std::setw(12) << (a > 1e-9 ? c / a : 0.0) << "\n";
        }
        std::cout << "WORST surrVaR=" << sv.worst_case_risk
                  << " surrCVaR=" << sc.worst_case_risk
                  << " jointVaR=" << jv.worst_case_risk
                  << " jointCVaR=" << jc.worst_case_risk << "\n";
        std::cout << "QSTAR argmax: surrVaR=";
        for (const auto& kv : sv.worst_case_weights) if (kv.second > 0.5) std::cout << kv.first;
        std::cout << "  jointVaR=";
        for (const auto& kv : jv.worst_case_weights) if (kv.second > 0.5) std::cout << kv.first;
        std::cout << "\n";
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
    // ---------------------------------------------------------------------
    // Section 2b -- GUARANTEE VALIDATION (the point of JOINT_*).
    //
    // The estimator is only worth having if the number it returns is the thing it
    // claims to be. Take the reported r[m], then draw a FRESH, INDEPENDENT set of
    // rollouts (different seed, 10x the samples) and check on that held-out set:
    //   VaR  guarantee: P( V_joint <= r[m] ) ~= alpha        (coverage)
    //   CVaR guarantee: E[ V_joint | V_joint >= VaR ] ~= r[m] (tail mean)
    // Held-out is essential -- checking on the same samples the quantile was fit to
    // is circular and would pass even if the estimator were wrong.
    // ---------------------------------------------------------------------
    std::cout << "\n=== SECTION 2b: joint-horizon guarantee validation (held-out) ===\n";
    {
        const double alpha = 0.95;
        const double R = 0.5 + 0.35 + 0.2;   // ego_r + obs_r + margin
        const int H = 15;

        DROConfig jv_cfg; jv_cfg.radius_calibration.risk_measure = DRORiskMeasure::JOINT_VAR;
        DROConfig jc_cfg; jc_cfg.radius_calibration.risk_measure = DRORiskMeasure::JOINT_CVAR;
        DRO d_jv(jv_cfg), d_jc(jc_cfg);
        // compute_risk_vector is private; read r[m] off the public result instead.
        // (ego_r=0.5, obs_r=0.35, margin=0.2 => the same R used below.)
        auto rv = d_jv.compute_worst_case_weights(
            nominal, obs, mode_models, ego_ref, H, 0.5, 0.35, 0.2).risk_per_mode;
        auto rc = d_jc.compute_worst_case_weights(
            nominal, obs, mode_models, ego_ref, H, 0.5, 0.35, 0.2).risk_per_mode;

        std::cout << std::setw(20) << "mode" << std::setw(11) << "VaR_hat"
                  << std::setw(11) << "coverage" << std::setw(9) << "target"
                  << std::setw(11) << "CVaR_hat" << std::setw(11) << "tailmean"
                  << std::setw(8) << "ok" << "\n";

        for (const auto& mode_id : mode_ids) {
            const ModeModel& mode = mode_models.at(mode_id);
            const int n_noise = static_cast<int>(mode.G.cols());
            const int M = 200000;

            // FRESH stream: different seed from config.radius_calibration.joint_risk_seed.
            std::mt19937_64 rng(0xDEADBEEF12345ULL);
            std::normal_distribution<double> gauss(0.0, 1.0);
            std::vector<double> V; V.reserve(M);

            for (int s = 0; s < M; ++s) {
                Eigen::Vector4d x = obs.to_array();
                double worst = 0.0;
                for (int k = 1; k <= H; ++k) {
                    Eigen::VectorXd w(n_noise);
                    for (int i = 0; i < n_noise; ++i) w(i) = gauss(rng);
                    x = mode.A * x + mode.b + mode.G * w;
                    const EgoState& e = (k < (int)ego_ref.size()) ? ego_ref[k] : ego_ref.back();
                    worst = std::max(worst, R - (x.head<2>() - e.position()).norm());
                }
                V.push_back(std::max(worst, 0.0));
            }

            const double var_hat = rv.at(mode_id);
            const double cvar_hat = rc.at(mode_id);

            // Coverage: fraction of held-out rollouts at or below the reported VaR.
            size_t below = 0;
            for (double x : V) if (x <= var_hat + 1e-12) ++below;
            const double coverage = double(below) / double(M);

            // Held-out tail mean above the held-out VaR (Rockafellar-Uryasev).
            std::vector<double> Vs = V;
            size_t idx = (size_t)std::clamp(alpha * M, 0.0, double(M - 1));
            std::nth_element(Vs.begin(), Vs.begin() + idx, Vs.end());
            const double q = Vs[idx];
            double excess = 0.0;
            for (double x : V) excess += std::max(x - q, 0.0);
            const double tail_mean = q + (excess / M) / (1.0 - alpha);

            const bool cov_ok = std::abs(coverage - alpha) < 0.01;
            const bool cvar_ok = (cvar_hat > 1e-9)
                                     ? std::abs(cvar_hat - tail_mean) / std::max(tail_mean, 1e-9) < 0.05
                                     : tail_mean < 1e-6;
            std::cout << std::setw(20) << mode_id
                      << std::setw(11) << var_hat
                      << std::setw(11) << coverage
                      << std::setw(9) << alpha
                      << std::setw(11) << cvar_hat
                      << std::setw(11) << tail_mean
                      << std::setw(8) << ((cov_ok && cvar_ok) ? "PASS" : "FAIL") << "\n";
        }
    }

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
