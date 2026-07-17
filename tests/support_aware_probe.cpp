// Support-aware Wasserstein probe (branch: support-aware-wasserstein, 2026-07-17).
// A/B behavioral comparison of the raw W1-LP reweighting (support_aware_alpha=0, the
// PREVIOUS behavior) against the support-aware probability floor
//   q <- alpha*p_hat + (1-alpha)*q*
// across the Wasserstein radius rho. Reports, per (rho, alpha):
//   - Q* support size and support floor (collapse => size 1, floor 0)
//   - the coverage functional  Psi_M = sum_{i:p_hat_i>0} (1-q_i)^M  (expected uncovered modes)
//   - the theoretical floor bound  sum_i e^{-M alpha p_hat_i}  (support-aware guarantee)
// Compiles against the DRO sources only (no scenario_sampler / mode_weights), so it
// builds even though the branch base has a pre-existing mode_weights/scenario_sampler desync.
#include "wasserstein_dro.hpp"
#include "types.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace scenario_mpc;

// Inline a small set of linear-Gaussian obstacle modes (constant-velocity + per-mode
// acceleration drift), so the probe needs only the DRO sources. dt=0.1.
static std::map<std::string, ModeModel> make_modes() {
    Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
    A(0, 2) = 0.1; A(1, 3) = 0.1;                 // p += v*dt
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(4, 2);
    G(0, 0) = 0.05; G(1, 1) = 0.05;               // modest position noise
    auto mk = [&](const std::string& id, double ax, double ay) {
        Eigen::Vector4d b; b << 0, 0, ax * 0.1, ay * 0.1;
        return ModeModel(id, A, b, G, id);
    };
    return {
        {"straight",   mk("straight",   0.0,  0.0)},
        {"veer_up",    mk("veer_up",    0.0,  0.4)},
        {"veer_down",  mk("veer_down",  0.0, -0.4)},
        {"brake",      mk("brake",     +0.3,  0.0)},
        {"rare_dart",  mk("rare_dart", -0.3,  0.0)},   // rare + highest-risk (the safety-critical mode)
    };
}

int main() {
    auto mode_models = make_modes();
    ObstacleState obs(4.0, 0.0, -0.5, 0.0);  // ahead of the ego path, oncoming (-x)
    std::vector<EgoState> ego_ref;
    for (int k = 0; k <= 15; ++k) ego_ref.emplace_back(k * 0.34, 0.0, 0.0, 1.5);

    // A slightly PEAKED nominal belief so p_hat varies across modes (the realistic case),
    // with one deliberately rare mode -- the safety-critical "rare cut-in" the floor protects.
    std::vector<std::string> ids;
    for (const auto& kv : mode_models) ids.push_back(kv.first);
    const int K = (int)ids.size();
    std::map<std::string, double> nominal;
    double acc = 0.0;
    for (int i = 0; i < K; ++i) { double w = (i == K - 1) ? 0.04 : 1.0; nominal[ids[i]] = w; acc += w; }
    for (auto& kv : nominal) kv.second /= acc;

    const int M = 40;  // scenario budget (matches the corrected-config runs)

    auto psi_cov = [&](const std::map<std::string,double>& q) {
        double s = 0.0;
        for (const auto& kv : nominal) if (kv.second > 0) s += std::pow(1.0 - q.at(kv.first), M);
        return s;
    };
    auto floor_bound = [&](double a) {
        double s = 0.0;
        for (const auto& kv : nominal) s += std::exp(-M * a * kv.second);
        return s;
    };
    auto qstats = [&](const std::map<std::string,double>& q) {
        double fl = 1e9; int sup = 0; std::string amax; double mx = -1;
        for (const auto& kv : q) { fl = std::min(fl, kv.second); if (kv.second > 1e-9) ++sup;
            if (kv.second > mx) { mx = kv.second; amax = kv.first; } }
        return std::make_tuple(sup, fl, amax, mx);
    };

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Support-aware Wasserstein A/B  (K=" << K << " modes, M=" << M
              << ", peaked nominal, rare mode p_hat=" << nominal[ids[K-1]] << ")\n";
    std::cout << "p_hat: "; for (auto& kv : nominal) std::cout << kv.first << "=" << kv.second << " "; std::cout << "\n\n";

    const double rhos[] = {0.05, 0.10, 0.20, 0.35, 0.50};
    const double alphas[] = {0.0, 0.2, 0.5};
    std::cout << std::setw(7) << "rho" << std::setw(7) << "alpha" << std::setw(9) << "support"
              << std::setw(11) << "q_floor" << std::setw(10) << "q_max" << std::setw(12) << "Psi_M(cov)"
              << std::setw(14) << "bound(e^-..)" << "  argmax\n";
    for (double rho : rhos) {
        for (double a : alphas) {
            DROConfig cfg; cfg.support_aware_alpha = a;
            WassersteinDRO dro(cfg); dro.set_rho_override(rho);
            DROResult r = dro.compute_worst_case_weights(
                nominal, obs, mode_models, ego_ref, 15, 0.5, 0.35, 0.2);
            auto [sup, fl, amax, mx] = qstats(r.worst_case_weights);
            std::cout << std::setw(7) << rho << std::setw(7) << a << std::setw(9) << sup
                      << std::setw(11) << fl << std::setw(10) << mx
                      << std::setw(12) << psi_cov(r.worst_case_weights)
                      << std::setw(14) << (a > 0 ? floor_bound(a) : std::nan(""))
                      << "  " << amax << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "READ: alpha=0 is the previous raw-LP reweighting. Where support collapses to 1\n"
              << "      and q_floor->0, Psi_M(cov) jumps (dropped modes each add ~1) = coverage lost.\n"
              << "      alpha>0 keeps support=K, q_floor>=alpha*min(p_hat), and Psi_M stays small,\n"
              << "      matching the guarantee bound sum_i e^{-M alpha p_hat_i}.\n";
    return 0;
}
