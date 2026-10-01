#include "mpc_controller.hpp"
#include <stdexcept>
#include <iostream>
using namespace dro_mpc;
static void require(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
int main(){
    RuntimeConfig cfg;cfg.mpc.type=MPCType::SH_MPCC;cfg.mpc.safe_horizon_enabled=false;cfg.dro.enabled=false;
    cfg.mpc.horizon=4;cfg.mpc.sampling.set_manual_sample_count(10);cfg.random_seed=701;
    ModeModel safe("safe",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
    ModeModel danger("danger",Eigen::Matrix4d::Identity(),Eigen::Vector4d(-.1,0,0,0),Eigen::MatrixXd::Zero(4,2));
    const std::map<std::string,ModeModel> modes{{"safe",safe},{"danger",danger}};
    const std::map<int,ObstacleState> obs{{0,ObstacleState(8,0,0,0)}};
    std::map<int,ModeHistory> histories;histories.emplace(0,ModeHistory(0,modes,0));
    const std::map<int,ModeDistribution> forced{{0,{{"danger",1.}}}};std::mt19937 rng(12);
    auto batch=sample_scenarios(obs,histories,&forced,4,10,cfg.mpc.sampling.mode_belief,nullptr,&rng);
    for(int flag=0;flag<4;++flag){
        auto bad=cfg;if(flag==0)bad.mpc.safe_horizon_enabled=true;if(flag==1)bad.dro.enabled=true;
        if(flag==2)bad.mpc.type=MPCType::SH_MPCC_DRO_FALLBACK;if(flag==3)bad.mpc.sampling.markov_jump_system=true;
        MPCController controller(bad);bool rejected=false;
        try{controller.set_experimental_scenarios(batch);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"incompatible sampling/guarantee configuration accepted");
    }
    MPCController ordinary(cfg),intervened(cfg);
    auto initialize=[&](MPCController& c){c.set_capture_attempt_diagnostics(true);c.initialize_obstacle(0,0,modes);for(int i=0;i<1000;++i)c.update_mode_observation(0,0,i<990?"safe":"danger",i);};
    initialize(ordinary);initialize(intervened);
    auto short_batch=batch;short_batch.pop_back();bool rejected=false;
    try{intervened.set_experimental_scenarios(short_batch);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"incorrect batch size accepted");
    intervened.set_experimental_scenarios(batch);
    auto solve=[&](MPCController& c){return c.solve(EgoState(0,0,0,2),obs,{16,0},2.);};
    solve(ordinary);auto result=solve(intervened);
    require(result.attempt_diagnostics.at(0).initial_mode_counts.at(0).at("danger")==10,"injected count lost");
    require(result.certificate_status==SafeHorizonCertificateStatus::NOT_REQUESTED&&!result.sample_count_sufficient,"intervention claimed certificate");
    auto a=solve(ordinary),b=solve(intervened);
    require(a.attempt_diagnostics.at(0).initial_mode_counts==b.attempt_diagnostics.at(0).initial_mode_counts,"one-shot override leaked or advanced RNG differently");
    intervened.set_experimental_scenarios(batch);intervened.reset();initialize(intervened);
    result=solve(intervened);
    const auto& counts=result.attempt_diagnostics.at(0).initial_mode_counts.at(0);
    require(!counts.count("danger")||counts.at("danger")<10,"reset retained intervention");
    std::cout<<"PASS: exact count, one-shot scope, reset, RNG continuation, and certified/DRO/retry/Markov rejection\n";
}
