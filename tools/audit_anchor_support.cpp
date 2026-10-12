// Diagnostic composition of existing production stages, not a live certificate.
#include "mpc_controller.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

namespace dro_mpc {
class CounterfactualProbe {
public:
    static MPCResult run(const std::vector<Scenario>& samples, bool prepare,
                         double& anchor_y) {
        RuntimeConfig cfg;
        cfg.mpc.type = MPCType::SH_MPC;
        cfg.mpc.sync_from_type();
        cfg.mpc.horizon = 1;
        cfg.mpc.dt = 1;
        cfg.mpc.ego.num_discs = 1;
        cfg.mpc.ego.length = 0;
        cfg.mpc.ego.radius = 0.1;
        cfg.obstacle_radius = 0.1;
        cfg.mpc.constraints.safety_margin = 0;
        cfg.mpc.constraints.support_cap_n_bar = 10;
        cfg.mpc.constraints.scenario_removal_budget = 0;
        cfg.mpc.sampling.automatically_compute_sample_size = false;
        cfg.mpc.sampling.num_scenarios = samples.size();
        cfg.solver.sqp_max_iterations = 15;
        cfg.solver.qp_tolerance = 1e-7;
        cfg.random_seed = 41;
        MPCController controller(cfg);
        const EgoState ego(0,0,0,1);
        const std::vector<EgoInput> inputs{EgoInput(0,0)};
        auto anchor = EgoDynamics(1).rollout(ego,inputs);
        if (prepare) prepare_safe_horizon_anchors(anchor,samples,0.2,1,0);
        anchor_y = anchor[1].y;
        auto rows = compute_linearized_constraints(anchor,samples,0.1,0.1,0,1,0);
        controller.scenarios_ = samples;
        return controller.solve_optimization_sqp(ego,Eigen::Vector2d(10,0),1,rows,{},inputs);
    }
};
}

using namespace dro_mpc;
Scenario scenario(int id, Eigen::Vector2d point) {
    return Scenario(id,{{0,ObstacleTrajectory(0,"fixed",{
        PredictionStep(0,point,Eigen::Matrix2d::Zero()),
        PredictionStep(1,point,Eigen::Matrix2d::Zero())})}});
}

int main(int argc,char** argv) {
    if(argc != 2) {std::cerr << "usage: audit_anchor_support output.csv\n"; return 2;}
    std::vector<Scenario> all{scenario(0,{1,0.05})};
    for(int id=1;id<=4;++id) all.push_back(scenario(id,{1000,0}));
    all.push_back(scenario(5,{1,-100}));
    auto without = all; without.pop_back();
    std::ofstream out(argv[1]);
    out << std::setprecision(17) << "prepared,removed_far,success,fallback,anchor_y,x1,y1,a,omega,support_size,support_ids\n";
    MPCResult prepared_full,prepared_without,frozen_full,frozen_without;
    for (bool prepare : {true,false}) for(bool remove : {false,true}) {
        double y;
        auto result = CounterfactualProbe::run(remove?without:all,prepare,y);
        out << prepare << ',' << remove << ',' << result.success << ',' << result.used_fallback << ',' << y << ','
            << result.ego_trajectory.at(1).x << ',' << result.ego_trajectory.at(1).y << ','
            << result.control_inputs.at(0).a << ',' << result.control_inputs.at(0).omega << ','
            << result.support_size << ',';
        for(int id:result.support_scenarios) out << id << ';';
        out << '\n';
        if(prepare) (remove?prepared_without:prepared_full)=result;
        else (remove?frozen_without:frozen_full)=result;
    }
    auto usable=[](const MPCResult& r){return r.success && !r.used_fallback && r.support_iterations_feasible;};
    const bool hidden = usable(prepared_full) && usable(prepared_without) &&
        prepared_full.support_scenarios == std::vector<int>{0} &&
        prepared_without.support_scenarios == std::vector<int>{0} &&
        (prepared_full.control_inputs[0].to_array()-prepared_without.control_inputs[0].to_array()).norm()>0.1;
    const bool independent = usable(frozen_full) && usable(frozen_without) &&
        (frozen_full.control_inputs[0].to_array()-frozen_without.control_inputs[0].to_array()).norm()<1e-8;
    std::cout << "hidden_preprocessing_dependency=" << hidden
              << " frozen_anchor_control_invariance=" << independent << '\n';
    return hidden && independent ? 0 : 1;
}
