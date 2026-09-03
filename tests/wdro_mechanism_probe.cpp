// wdro_mechanism_probe: is the WDRO reweighting actually hedging against the most
// dangerous obstacle mode on MAIN's default config? Head-on scenario (ego drives toward
// an obstacle) with the real mode models -> per-mode risk r differs. We inspect, for the
// default DROConfig the controller uses, whether q* up-weights argmax(r), and how rho and
// the q* distortion behave vs the ambiguity radius and the observation count.
#include "dro.hpp"
#include "dynamics.hpp"
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

using namespace dro_mpc;

static std::vector<std::string> ids_of(const std::map<std::string,double>& m) {
    std::vector<std::string> v; for (auto& kv : m) v.push_back(kv.first); return v;
}

int main() {
    auto mode_models = create_obstacle_mode_models(0.1);
    ObstacleState obs(5.0, 0.0, 0.5, 0.0);                 // ahead, drifting toward ego
    std::vector<EgoState> ego_ref;
    for (int k = 0; k <= 15; ++k) ego_ref.emplace_back(k * 0.34, 0.0, 0.0, 1.5);  // ego -> obstacle
    std::map<std::string,double> nominal;
    for (auto& kv : mode_models) nominal[kv.first] = 1.0 / mode_models.size();
    auto mode_ids = ids_of(nominal);

    // ---- 1) DEFAULT config (exactly what the controller uses on main) ----
    DROConfig def;   // calibrated radius + Bonferroni VaR + primal OT, all default-on
    DRO d(def);
    DROResult r = d.compute_worst_case_weights(nominal, obs, mode_models, ego_ref, 15, 0.5, 0.35, 0.2);

    // rank modes by risk
    std::vector<std::pair<std::string,double>> rk(r.risk_per_mode.begin(), r.risk_per_mode.end());
    std::sort(rk.begin(), rk.end(), [](auto&a,auto&b){return a.second>b.second;});
    std::string danger = rk.front().first;
    double l1 = 0; for (auto& id : mode_ids) l1 += std::abs(r.worst_case_weights[id] - nominal.at(id));

    std::printf("=== WDRO MECHANISM on MAIN default config ===\n");
    std::printf("rho_used=%.4f   Qstar_L1(distortion from nominal)=%.4f\n", r.rho_used, l1);
    std::printf("%-20s %8s %10s %10s\n","mode","risk r","nominal","q*");
    for (auto& kv : rk) {
        std::printf("%-20s %8.4f %10.4f %10.4f%s\n", kv.first.c_str(), kv.second,
                    nominal.at(kv.first), r.worst_case_weights[kv.first],
                    kv.first==danger ? "   <- most dangerous" : "");
    }
    std::printf("HEDGE CHECK: q*[danger]-nominal[danger] = %+.4f  (want > 0 if WDRO hedges)\n\n",
                r.worst_case_weights[danger] - nominal.at(danger));

    // ---- 2) rho sweep: how much does q* hedge toward the dangerous mode vs radius ----
    std::printf("=== rho sweep (override): q*[danger] and distortion ===\n");
    std::printf("%6s %10s %12s %10s\n","rho","q*[danger]","Qstar_L1","impl_cost");
    for (double rho : {0.02,0.05,0.10,0.15,0.20,0.30,0.50}) {
        DRO d2(def); d2.set_rho_override(rho);
        DROResult rr = d2.compute_worst_case_weights(nominal, obs, mode_models, ego_ref, 15, 0.5, 0.35, 0.2);
        double l1b = 0; for (auto& id : mode_ids) l1b += std::abs(rr.worst_case_weights[id]-nominal.at(id));
        std::printf("%6.2f %10.4f %12.4f %10.4f\n", rho, rr.worst_case_weights[danger], l1b, rr.implied_transport_cost);
    }

    // ---- 3) calibrated rho vs observation count (does it shrink WDRO off?) ----
    std::printf("\n=== calibrated rho vs observation_count_ (default calibrated radius) ===\n");
    std::printf("%6s %10s %12s\n","n_obs","rho_used","q*[danger]");
    for (int n : {1,5,10,25,50,100,200}) {
        DRO d3(def); d3.set_observation_count(n);
        DROResult rr = d3.compute_worst_case_weights(nominal, obs, mode_models, ego_ref, 15, 0.5, 0.35, 0.2);
        std::printf("%6d %10.4f %12.4f\n", n, rr.rho_used, rr.worst_case_weights[danger]);
    }
    return 0;
}
