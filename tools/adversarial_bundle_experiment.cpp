#include "mpc_controller.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
using namespace dro_mpc;
int main(int argc,char** argv) {
    if (argc!=3) throw std::invalid_argument("usage: adversarial_bundle_experiment OUTPUT_DIRECTORY SEED_COUNT");
    const int seeds=std::stoi(argv[2]);
    if (seeds<1) throw std::invalid_argument("seed count must be positive");
    const auto folder=std::filesystem::path(argv[1]);
    std::ofstream cycles(folder/"cycles.csv"),modes(folder/"modes.csv"),draws(folder/"draws.csv");
    std::ofstream plans(folder/"plans.csv");
    if (!cycles||!modes||!draws) throw std::runtime_error("cannot open artifacts");
    cycles << std::setprecision(17) << "noise,seed,repeat,c0,extras,cycle,success,certificate,bundles,required,raw_draws,c_K,eta,total_failure_budget,support,retained_facets,raw_future_facets,solve_seconds,terminal_x\n";
    modes << std::setprecision(17) << "noise,seed,repeat,c0,extras,cycle,mode,K,U,allocation_q,allocation_score,rho\n";
    draws << std::setprecision(17) << "noise,seed,repeat,c0,extras,cycle,raw_id,bundle_id,mode,stage,x,y\n";
    plans << std::setprecision(17) << "noise,seed,repeat,c0,extras,cycle,stage,x,y\n";
    for (double noise:{0.,.02}) for (int seed=77;seed<77+seeds;++seed) for (int repeat=0;repeat<2;++repeat)
    for (double c0:{0.,2.,4.}) for (int extras:{0,2}) {
        if(c0==0&&extras>0)continue;
        RuntimeConfig cfg;cfg.random_seed=seed;cfg.dro.enabled=true;
        cfg.mpc.horizon=4;cfg.mpc.bundle_amplification=c0;cfg.mpc.bundle_extra_draws=extras;
        cfg.mpc.ego.num_discs=1;cfg.mpc.ego.length=0;
        MPCController controller(cfg);controller.set_capture_linearized_constraints(true);
        Eigen::MatrixXd G=Eigen::MatrixXd::Zero(4,2);G(0,0)=noise;G(1,1)=noise;
        ModeModel safe("safe",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),G);
        ModeModel cut("cut",Eigen::Matrix4d::Identity(),Eigen::Vector4d(0,-.4,0,0),G);
        controller.initialize_obstacle(0,0,{{"safe",safe},{"cut",cut}});
        for(int i=0;i<100;++i)controller.update_mode_observation(0,0,i<95?"safe":"cut",i);
        EgoState ego(0,0,0,1);
        for(int cycle=0;cycle<2;++cycle) {
            auto r=controller.solve(ego,{{0,ObstacleState(2.8,2.5,0,0)}},{30,0});
            auto prefix=[&](std::ostream& out){out<<noise<<','<<seed<<','<<repeat<<','<<c0<<','<<extras<<','<<cycle<<',';};
            prefix(cycles);
            cycles<<r.success<<','<<safe_horizon_certificate_status_name(r.certificate_status)<<','<<r.sampled_scenarios
                <<','<<r.required_scenarios<<','<<r.raw_scenario_draws<<','<<r.bundle_amplification<<','<<r.bundle_threshold
                <<','<<(r.bundle_sampling?r.bundle_combined_failure_budget:cfg.mpc.sampling.chance_of_certificate_violation)
                <<','<<r.support_size<<','<<controller.last_linearized_constraints().size()<<','
                <<r.raw_scenario_draws*cfg.mpc.horizon<<','<<r.solve_time<<','<<r.ego_trajectory.back().x<<'\n';
            for(const auto& [mode,K]:r.bundle_multiplicities) {
                prefix(modes);modes<<mode<<','<<K<<','<<r.bundle_mode_upper.at(mode)<<',';
                const auto& robust=controller.last_dro_results().at(0);
                modes<<robust.worst_case_weights.at(mode)<<','<<robust.risk_per_mode.at(mode)<<','<<robust.rho_used<<'\n';
            }
            for(std::size_t k=0;k<r.ego_trajectory.size();++k) {prefix(plans);plans<<k<<','<<r.ego_trajectory[k].x<<','<<r.ego_trajectory[k].y<<'\n';}
            for(const auto& scenario:controller.scenarios()) for(const auto& [id,traj]:scenario.trajectories)
                for(const auto& step:traj.steps) {
                    prefix(draws);draws<<scenario.scenario_id<<','<<scenario.support_id()<<','<<traj.mode_id<<','<<step.k<<','<<step.mean.x()<<','<<step.mean.y()<<'\n';
                }
            if(!r.success)break;
            ego=r.ego_trajectory.at(1);
        }
    }
}
