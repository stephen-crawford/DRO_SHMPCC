// Outcome-free design probe: production risk, calibration and WDRO on a fixed reference.
#include "experiment_config_yaml.hpp"
#include "dro.hpp"
#include "mode_weights.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace dro_mpc;
static ModeModel mode(const std::string& n,double dx,double dy){return ModeModel(n,Eigen::Matrix4d::Identity(),Eigen::Vector4d(dx,dy,0,0),Eigen::MatrixXd::Zero(4,2));}
int main(int argc,char** argv){
    if(argc!=2||std::filesystem::exists(argv[1]))return 2;
    try{
        auto cfg=yaml_config::load_experiment_config().to_scenario_mpc_config();
        cfg.dro.solver.radius_calibration.use_entropic_allocator=false;
        const std::map<std::string,ModeModel> modes{{"continue",mode("continue",.12,0)},{"away",mode("away",.08,.14)},{"across",mode("across",-.04,-.18)}};
        DRO dro(cfg.dro.solver);
        std::ofstream out(argv[1]);if(!out)throw std::runtime_error("cannot open design CSV");
        out<<std::setprecision(17)<<"danger_count,step,mode,observed_count,p,q,r,rho,ego_x,obstacle_x,obstacle_y\n";
        for(int count=5;count<=150;count+=5)for(int t=5;t<10;++t){
            const std::map<std::string,int> counts{{"continue",900-count+t-5},{"away",100},{"across",count}};
            ModeHistory history(0,modes,0);int tick=-1000;
            for(const auto& [name,n]:counts)for(int i=0;i<n;++i)history.record_observation(tick++,0,name);
            const auto p=compute_mode_weights(history,cfg.mpc.sampling.mode_belief);
            std::vector<EgoState> ref;for(int k=0;k<=20;++k)ref.emplace_back(.2*(t+k),0,0,2);
            const ObstacleState obstacle(5+.12*t,2,0,0);
            const auto robust=dro.compute_worst_case_weights(p,counts,obstacle,modes,ref,20,cfg.mpc.ego.radius,cfg.obstacle_radius,cfg.mpc.constraints.safety_margin,20,3,2.);
            for(const auto& [name,prob]:p)out<<count<<','<<t<<','<<name<<','<<counts.at(name)<<','<<prob<<','<<robust.worst_case_weights.at(name)<<','<<robust.risk_per_mode.at(name)<<','<<robust.rho_used<<','<<.2*t<<','<<obstacle.x<<','<<obstacle.y<<'\n';
        }
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
