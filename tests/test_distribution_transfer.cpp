#include "primal_ot.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_radius_calibration.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace dro_mpc;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
    std::cout << "PASS: " << message << '\n';
}

int main() {
    const std::vector<double> p{0.8, 0.15, 0.05}, risk{0.0, 0.5, 1.0};
    const std::vector<std::vector<double>> D{{0,1,1},{1,0,1},{1,1,0}};
    auto allocation = solve_dominating_ot(p, risk, D, 0.2, 0.1);
    require(allocation.transport.solved, "Q_risk LP solves");
    std::vector<double> q(3, 0.0);
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) q[j]+=allocation.transport.plan[i][j];
    require(std::abs(allocation.envelope[0]-1.0)<1e-7 &&
        std::abs(allocation.envelope[1]-0.35)<1e-7 &&
        std::abs(allocation.envelope[2]-0.25)<1e-7, "coordinate envelope matches analytic TV ball");
    require(std::abs(q[0]-0.6)<1e-7 && std::abs(q[1]-0.15)<1e-7 &&
        std::abs(q[2]-0.25)<1e-7, "optimizer matches analytic constrained solution, including second dangerous mode floor");
    require(allocation.domination<=allocation.nominal_domination+1e-7 &&
        std::abs(allocation.domination-7.0/3.0)<1e-7, "domination improves from 5 to 7/3");
    require(std::abs(allocation.transport.expected_risk-0.325)<1e-7,
        "expected risk is 0.325 versus nominal 0.125");
    // Independent simplex-grid check of the full ambiguity set and event domination.
    bool grid_dominated = true;
    for(int a=0;a<=100;++a) for(int b=0;b<=100-a;++b) {
        std::vector<double> candidate{a/100.0,b/100.0,(100-a-b)/100.0};
        double tv=0; for(int j=0;j<3;++j) tv+=0.5*std::abs(candidate[j]-p[j]);
        if(tv>0.2+1e-12) continue;
        for(int j=0;j<3;++j) grid_dominated = grid_dominated &&
            candidate[j]<=allocation.domination*q[j]+1e-7;
    }
    require(grid_dominated, "all covered simplex-grid coordinates obey domination");
    const auto all_danger = solve_dominating_ot(p, {0.2,0.5,1.0}, D, 1.0, 0.1);
    require(all_danger.transport.solved && std::abs(all_danger.transport.expected_risk-0.285)<1e-7,
        "all-dangerous case leaves every nominal probability unchanged");
    const auto zero = solve_dominating_ot(p, risk, D, 0.0, 0.1);
    require(zero.transport.solved && std::abs(zero.domination-1)<1e-7,
        "zero radius on a metric gives zeta=1");
    const std::vector<std::vector<double>> pseudometric(3, std::vector<double>(3,0.0));
    const auto duplicate = solve_dominating_ot(p, risk, pseudometric, 0.0, 0.1);
    require(duplicate.transport.solved && std::abs(duplicate.envelope[2]-1)<1e-7,
        "zero-cost distinct labels retain full coordinate ambiguity");
    require(!solve_dominating_ot({1,0}, {0,1}, {{0,1},{1,0}}, .1,.1).transport.solved,
        "missing nominal support is rejected");
    const auto tiny = finite_sample_wasserstein_radius({5,5}, {0.5,0.5}, {{0,1},{1,0}}, 1e-16);
    require(tiny.lower[0]<0.001 && tiny.upper[0]>0.999,
        "requested tiny confidence tails are not replaced by larger failure probabilities");

    DROConfig tail_cfg;
    tail_cfg.radius_calibration.alpha_one_sided=std::nextafter(1.0,0.0);
    tail_cfg.radius_calibration.sigma_floor=1.0;
    DRO tail_dro(tail_cfg);
    ModeModel coincident("same",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
    std::vector<EgoState> frozen(21,EgoState(0,0,0,0));
    const auto tail_risk=tail_dro.compute_risk_vector(RiskVectorRequest{
        ObstacleState(0,0,0,0),{{"same",coincident}},{"same"},frozen,20,1.0,3,4.0,nullptr});
    require(tail_risk.at("same")>9.0,
        "Bonferroni upper tail is evaluated without rounding or clamping alpha prime");

    RuntimeConfig cfg;
    cfg.mpc.horizon=3;
    cfg.mpc.ego.num_discs=1;
    cfg.mpc.ego.length=0;
    cfg.mpc.constraints.support_cap_n_bar=2;
    cfg.mpc.constraints.scenario_removal_budget=0;
    cfg.mpc.sampling.one_minus_chance_constraint_violation_probability=0.8;
    cfg.mpc.sampling.chance_of_certificate_violation=0.05;
    cfg.dro.enabled=true;
    cfg.random_seed=71;
    cfg.normalize();
    ModeModel stay("stay",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
    Eigen::Vector4d offset=Eigen::Vector4d::Zero(); offset[0]=0.1;
    ModeModel move("move",Eigen::Matrix4d::Identity(),offset,Eigen::MatrixXd::Zero(4,2));
    MPCController controller(cfg);
    for(int id=0;id<2;++id) {
        controller.initialize_obstacle(id,id,{{"stay",stay},{"move",move}});
        for(int i=0;i<100;++i) controller.update_mode_observation(id,id,i%2?"stay":"move",i);
    }
    const auto result=controller.solve(EgoState(0,0,0,1),
        {{0,ObstacleState(30,10,0,0)},{1,ObstacleState(40,10,0,0)}},Eigen::Vector2d(10,0));
    double product=1;
    for(const auto& [id,dro]:controller.last_dro_results()) {
        product*=dro.domination_factor;
        const auto calibrated=finite_sample_wasserstein_radius({50,50},{.5,.5},
            dro.transport_cost_matrix,cfg.dro.solver.radius_calibration.confidence_beta/2);
        require(std::abs(dro.rho_used-calibrated.rho)<1e-8,
            "controller allocates global ambiguity beta across active obstacles");
    }
    const int S=static_cast<int>(controller.scenarios().size());
    require(std::abs(result.distribution_domination_factor-product)<1e-8 && product>1,
        "controller uses product of per-obstacle domination factors");
    require(S>cfg.compute_required_scenarios() &&
        cfg.compute_effective_epsilon(S,cfg.support_limit())<=cfg.epsilon()/product &&
        cfg.compute_effective_epsilon(S-1,cfg.support_limit())>cfg.epsilon()/product,
        "automatic sampling uses the smallest S satisfying epsilon/zeta");
    std::cout << "INTEGRATION S=" << S << " nominal_S=" << cfg.compute_required_scenarios()
        << " zeta=" << product << " target=" << result.sampling_violation_target
        << " transfer=" << result.transfer_bound_satisfied << '\n';
    cfg.mpc.sampling.set_manual_sample_count(8);
    MPCController manual(cfg);
    manual.initialize_obstacle(0,0,{{"stay",stay},{"move",move}});
    const auto under=manual.solve(EgoState(0,0,0,1),{{0,ObstacleState(30,10,0,0)}},Eigen::Vector2d(10,0));
    require(!under.transfer_bound_satisfied && under.certificate_status!=SafeHorizonCertificateStatus::CERTIFIED,
        "undersized manual batch never receives transfer certification");
}
