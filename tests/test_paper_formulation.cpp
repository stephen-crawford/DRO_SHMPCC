#include "mpc_controller.hpp"
#include "primal_ot.hpp"
#include "wasserstein_radius_calibration.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace dro_mpc;

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
    std::cout << "PASS: " << message << '\n';
}
void near(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) < 1e-8, message);
}
}

int main() {
    const std::vector<std::vector<double>> zero(3, std::vector<double>(3, 0.0));
    const std::vector<std::vector<double>> metric{{0,1,2},{1,0,1},{2,1,0}};
    const std::vector<int> counts{80,15,5};
    const std::vector<double> p{80.5/101.5,15.5/101.5,5.5/101.5};
    const auto cp = finite_sample_wasserstein_radius(counts,p,zero,0.05);
    const auto cp_metric = finite_sample_wasserstein_radius(counts,p,metric,0.05);
    for (int m=0;m<3;++m) {
        double rest=0;
        for (int j=0;j<3;++j) if (j!=m) rest+=cp.lower[j];
        near(cp.coordinate_envelope[m],std::min(cp.upper[m],1-rest),
             "coordinate envelope equals the CP box-simplex expression");
        near(cp.coordinate_envelope[m],cp_metric.coordinate_envelope[m],
             "confidence envelope is independent of transport geometry");
    }
    require(cp.coordinate_envelope[2]<0.2,"observed rare mode stays bounded even when all transport costs vanish");
    const auto legacy=solve_dominating_ot(p,{0,0,0},zero,0,0.01);
    near(legacy.envelope[2],1,"legacy outer-ball envelope is retained for explicit legacy calls");
    const auto allocation=solve_dominating_ot(p,{0,0,1},zero,0,0.01,cp.coordinate_envelope);
    require(allocation.transport.solved,"CP-envelope allocation solves at zero transport budget");
    double q_rare=0;
    for (const auto& row:allocation.transport.plan) q_rare+=row[2];
    const double expected_q=1-cp.coordinate_envelope[0]/allocation.nominal_domination
                              -cp.coordinate_envelope[1]/allocation.nominal_domination;
    near(q_rare,expected_q,"risk optimizer attains analytic zero-cost optimum at the two safe-mode floors");
    require(allocation.domination<=allocation.nominal_domination+1e-8,
            "selected factor does not exceed nominal factor");
    const double minimum=std::accumulate(cp.coordinate_envelope.begin(),cp.coordinate_envelope.end(),0.0);
    double benchmark=0;
    for (double u:cp.coordinate_envelope) benchmark=std::max(benchmark,u/(u/minimum));
    near(benchmark,minimum,"normalized envelope attains the full-simplex minimum domination benchmark");
    require(allocation.domination+1e-8>=minimum,"restricted optimizer respects the benchmark lower bound");
    const auto boundary=solve_dominating_ot({.5,.5},{1,0},{{0,0},{0,0}},0,.1,{1,0});
    require(boundary.transport.solved,"u=0,q=0 is permitted by the paper's ratio convention");
    near(boundary.domination,1,"zero-envelope coordinate is excluded from the ratio maximum");
    const auto empty=finite_sample_wasserstein_radius({0,0,0},{1./3,1./3,1./3},metric,.05);
    require(empty.coordinate_envelope==std::vector<double>({1,1,1}),"no observations retain the full-simplex envelope");
    const auto single=finite_sample_wasserstein_radius({7},{1},{{0}},.05);
    near(single.coordinate_envelope[0],1,"one-mode simplex has envelope one");

    // Real DRO path: identical predictions make D=0. The old ball gives u=1,
    // whereas the updated path must retain the calibrated CP envelope.
    DROConfig dro_cfg;
    DRO dro(dro_cfg);
    ModeModel stay("a",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
    const std::map<std::string,ModeModel> modes{{"a",stay},{"b",stay},{"c",stay}};
    const std::vector<EgoState> frozen(4,EgoState(0,0,0,0));
    const auto actual=dro.compute_worst_case_weights({{"a",p[0]},{"b",p[1]},{"c",p[2]}},
        {{"a",80},{"b",15},{"c",5}},ObstacleState(30,10,0,0),modes,frozen,3,.5,.3,0,3,1,0);
    near(actual.coordinate_envelope.at("c"),cp.coordinate_envelope[2],"production DRO receives the CP envelope instead of the outer-ball envelope");
    near(actual.rho_used,0,"identical predictions have zero transport design budget");

    RuntimeConfig cfg;
    cfg.dro.enabled=true;cfg.mpc.horizon=3;cfg.mpc.ego.num_discs=1;cfg.mpc.ego.length=0;
    cfg.mpc.constraints.support_cap_n_bar=2;cfg.mpc.constraints.scenario_removal_budget=0;
    cfg.mpc.sampling.one_minus_chance_constraint_violation_probability=.8;
    cfg.mpc.sampling.chance_of_certificate_violation=.05;cfg.random_seed=71;cfg.normalize();
    MPCController controller(cfg);
    controller.initialize_obstacle(0,0,modes);
    for (int i=0;i<100;++i) controller.update_mode_observation(0,0,i<80?"a":i<95?"b":"c",i);
    // A late-joining class sibling must not inherit or broadcast calibration data.
    controller.initialize_obstacle(1,0,modes);
    for (int i=0;i<100;++i) controller.update_mode_observation(1,0,i<5?"a":i<20?"b":"c",i);
    const auto result=controller.solve(EgoState(0,0,0,1),
        {{0,ObstacleState(30,10,0,0)},{1,ObstacleState(40,10,0,0)}},Eigen::Vector2d(10,0));
    double product=1;
    for (const auto& [id,observed]:controller.last_dro_results()) {
        const std::vector<int> own=id==0?counts:std::vector<int>{5,15,80};
        const std::vector<double> center=id==0?p:std::vector<double>{p[2],p[1],p[0]};
        const auto expected=finite_sample_wasserstein_radius(own,center,zero,.025);
        require(observed.radius_observation_count==100,"paper calibration counts only the source obstacle's observations");
        int m=0;
        for (const auto& [mode,q]:observed.worst_case_weights) {
            near(observed.coordinate_envelope.at(mode),expected.coordinate_envelope[m],"live controller uses per-obstacle CP envelope with allocated beta");
            near(q,center[m],"flat scores retain the obstacle-specific KT nominal law");
            ++m;
        }
        product*=observed.domination_factor;
    }
    const int S=static_cast<int>(controller.scenarios().size());
    near(result.distribution_domination_factor,product,"live transfer factor equals product of CP factors");
    require(cfg.compute_effective_epsilon(S,cfg.support_limit())<=cfg.epsilon()/product &&
        cfg.compute_effective_epsilon(S-1,cfg.support_limit())>cfg.epsilon()/product,
        "live sampling selects smallest S for the updated epsilon/zeta target");
    require(result.transfer_bound_satisfied,"live returned solution satisfies conditional transfer acceptance checks");
    std::cout << "PAPER_CP rare_u=" << cp.coordinate_envelope[2]
              << " legacy_u=" << legacy.envelope[2] << " q_rare=" << q_rare
              << " minimum_zeta=" << minimum << " joint_zeta=" << product << " S=" << S << '\n';
}
