// Fast deterministic comparison of the ad-hoc adaptive radius vs the
// confidence-calibrated simplex-concentration radius, swept over the number of
// observed interactions n. For each n it prints rho and the resulting worst-case
// risk + Q* under both radius rules, on the canonical 6-mode scenario (ego driving
// toward the obstacle so the risk vector is nonzero and differentiated).
#include "dro.hpp"
#include "dynamics.hpp"
#include "types.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>

using namespace dro_mpc;

static double expected_risk(const std::map<std::string, double>& q,
                            const std::map<std::string, double>& r) {
    double s = 0.0;
    for (const auto& kv : q) s += kv.second * r.at(kv.first);
    return s;
}

// Q* L1 distance from nominal (uniform) — how far the worst case reweights.
static double l1_from_nominal(const std::map<std::string, double>& q, double u) {
    double s = 0.0;
    for (const auto& kv : q) s += std::abs(kv.second - u);
    return s;
}

static DROResult run(bool calibrated, int n_obs,
                     const std::map<std::string, double>& nominal,
                     const ObstacleState& obs,
                     const std::map<std::string, ModeModel>& mm,
                     const std::vector<EgoState>& ego) {
    DROConfig cfg;                       // defaults: rho_base=0.1, rho_min=0.01, rho_max=0.5
    cfg.radius_calibration.use_calibrated_radius = calibrated;
    cfg.radius_calibration.confidence_beta = 0.05;          // 95% coverage
    DRO dro(cfg);
    dro.set_observation_count(n_obs);
    return dro.compute_worst_case_weights(nominal, obs, mm, ego, 15, 0.5, 0.35, 0.2);
}

int main() {
    auto mm = create_obstacle_mode_models(0.1);
    ObstacleState obs(5.0, 0.0, 0.5, 0.0);
    std::vector<EgoState> ego;
    for (int k = 0; k <= 15; ++k) ego.emplace_back(k * 0.34, 0.0, 0.0, 1.5);

    std::map<std::string, double> nominal;
    for (const auto& kv : mm) nominal[kv.first] = 1.0 / mm.size();
    const double u = 1.0 / mm.size();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "n_obs |  HEUR rho  wc_risk  Qstar_L1 |  CALIB rho  wc_risk  Qstar_L1 | d_rho\n";
    const int ns[] = {1, 2, 5, 10, 25, 50, 100, 250, 1000, 10000};
    for (int n : ns) {
        DROResult h = run(false, n, nominal, obs, mm, ego);
        DROResult c = run(true,  n, nominal, obs, mm, ego);
        std::cout << std::setw(5) << n << " |  "
                  << h.rho_used << "  " << h.worst_case_risk << "  "
                  << l1_from_nominal(h.worst_case_weights, u) << " |  "
                  << c.rho_used << "  " << c.worst_case_risk << "  "
                  << l1_from_nominal(c.worst_case_weights, u) << " | "
                  << (c.rho_used - h.rho_used) << "\n";
    }
    return 0;
}
