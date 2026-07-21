// risk_and_rho_verify: two checks the user flagged.
//  (A) Risk-model sanity on an ONCOMING obstacle (moving toward the ego): the most
//      dangerous mode should be one that keeps closing (constant_velocity / turn-toward),
//      NOT decelerating (which only matters in rear-end / intersection geometry).
//  (B) Ambiguity-radius behavior: a TIGHT rho must keep q* close to nominal (graded /
//      full support). q* = e_argmax only when the transport budget rho is SLACK. Report
//      the transport cost D and sweep rho to find where collapse begins.
#include "wasserstein_dro.hpp"
#include "dynamics.hpp"
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace dro_mpc;

int main() {
    auto mm = create_obstacle_mode_models(0.1);
    std::vector<std::string> modes; for (auto& kv : mm) modes.push_back(kv.first);
    const int M = (int)modes.size();
    std::map<std::string,double> nominal; for (auto& m : modes) nominal[m]=1.0/M;

    // ONCOMING: ego drives +x from origin; obstacle ahead at (6,0) moving TOWARD ego (-x).
    ObstacleState obs(6.0, 0.0, -1.0, 0.0);
    std::vector<EgoState> ego_ref; for (int k=0;k<=15;++k) ego_ref.emplace_back(k*0.34, 0.0, 0.0, 1.5);

    DROConfig cfg;  // default: Bonferroni VaR, calibrated radius, primal OT
    WassersteinDRO dro(cfg);
    DROResult r = dro.compute_worst_case_weights(nominal, obs, mm, ego_ref, 15, 0.5, 0.35, 0.2);

    // (A) risk vector, sorted
    std::vector<std::pair<std::string,double>> rk(r.risk_per_mode.begin(), r.risk_per_mode.end());
    std::sort(rk.begin(), rk.end(), [](auto&a,auto&b){return a.second>b.second;});
    std::printf("=== (A) RISK ON ONCOMING OBSTACLE (obstacle moving -x toward ego) ===\n");
    for (auto& kv : rk) std::printf("    r[%-18s] = %.4f\n", kv.first.c_str(), kv.second);
    std::printf("    -> most dangerous: %s (expected: a closing mode, NOT decelerating)\n\n", rk.front().first.c_str());

    // (B) transport cost matrix + how much it costs to collapse to argmax
    std::printf("=== (B) TRANSPORT COST D and rho-collapse behavior ===\n");
    const auto& D = r.transport_cost_matrix;
    double dmax=0, dmean=0; int cnt=0;
    for (int i=0;i<M;++i) for (int j=0;j<M;++j){ dmax=std::max(dmax,D[i][j]); if(i!=j){dmean+=D[i][j];++cnt;} }
    dmean/= std::max(1,cnt);
    // index of argmax-risk mode
    int argmax=0; for (int i=0;i<M;++i) if (modes[i]==rk.front().first) argmax=i;
    // cost to move all nominal mass onto argmax = sum_i p_i D[i][argmax]
    double collapse_cost=0; for (int i=0;i<M;++i) collapse_cost += nominal[modes[i]]*D[i][argmax];
    std::printf("    D: max=%.4f  mean-offdiag=%.4f   cost to collapse nominal->argmax = %.4f\n",
                dmax, dmean, collapse_cost);
    std::printf("    (q* can only reach e_argmax when rho >= collapse_cost; a tighter rho stays graded)\n\n");

    std::printf("    rho     support(#modes>1e-3)   q*[danger]   ||q*-nominal||_1\n");
    for (double rho : {0.02,0.05,0.10,0.15,0.20,0.30,0.50}) {
        WassersteinDRO d2(cfg); d2.set_rho_override(rho);
        DROResult rr = d2.compute_worst_case_weights(nominal, obs, mm, ego_ref, 15, 0.5, 0.35, 0.2);
        int sup=0; double l1=0; for (auto& m : modes){ double q=rr.worst_case_weights[m]; if(q>1e-3)++sup; l1+=std::abs(q-nominal[m]); }
        std::printf("    %-6.2f  %-20d  %-11.3f  %.3f\n", rho, sup, rr.worst_case_weights[rk.front().first], l1);
    }
    std::printf("\n    default rho actually used (calibrated, this call) = %.4f\n", r.rho_used);
    return 0;
}
