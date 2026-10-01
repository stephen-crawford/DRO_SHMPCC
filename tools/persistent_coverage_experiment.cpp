// Five-step coverage intervention. Explicit non-IID batches; no scenario certification.
#include "experiment_config_yaml.hpp"
#include "mpc_controller.hpp"
#include "dynamics.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
using namespace dro_mpc;
namespace fs=std::filesystem;
static std::ofstream csv(const fs::path& p,const std::string& header) {
    std::ofstream f(p);if(!f)throw std::runtime_error("cannot write artifact");
    f << std::setprecision(17) << header << '\n';return f;
}
static ModeModel mode(const std::string& name,double dx,double dy) {
    return ModeModel(name,Eigen::Matrix4d::Identity(),Eigen::Vector4d(dx,dy,0,0),Eigen::MatrixXd::Zero(4,2));
}
static double clearance(const EgoState& ego,const std::map<int,ObstacleState>& obstacles,const RuntimeConfig& cfg) {
    double d=1e9;
    for(const auto& disc:compute_ego_disc_positions(ego,cfg.mpc.ego.num_discs,cfg.mpc.ego.length))
        for(const auto& [id,x]:obstacles)d=std::min(d,(disc-x.position()).norm()-cfg.mpc.ego.radius-cfg.obstacle_radius);
    return d;
}
int main(int argc,char** argv) {
    if(argc!=13){std::cerr << "usage: risk_stress_experiment NEW_OUTPUT FAMILY DANGER_COUNT S SEED SWITCH_STEP STEPS HISTORY_LAG ARM X WINDOW TARGET_ND\n";return 2;}
    const fs::path output=argv[1];const std::string family=argv[2],arm=argv[9];
    const int count=std::stoi(argv[3]),budget=std::stoi(argv[4]),seed=std::stoi(argv[5]),sw=std::stoi(argv[6]),steps=std::stoi(argv[7]),lag=std::stoi(argv[8]);
    const std::set<std::string> families{"rare_turn","late_switch","two_sided_trap","rare_braking","rare_benign","common_dangerous","equal_probability","equal_geometry"};
    const std::set<std::string> arms{"sh_mpcc","sh_mpcc_dro","sh_mpcc_extra","sh_mpcc_resample","sh_mpcc_dro_fallback"};
    if(fs::exists(output)||!families.count(family)||!arms.count(arm)||count<1||count>900||budget<3||steps<1||sw<0||sw>=steps||lag<0)return 2;
    const double scene_x=std::stod(argv[10]);
    const int window=std::stoi(argv[11]),target=std::stoi(argv[12]);
    if(arm!="sh_mpcc"||target<0||target>budget||window<1)return 2;
    const int window_start=std::max(0,sw-2);
    fs::create_directories(output);
    auto modes=std::map<std::string,ModeModel>{{"continue",mode("continue",.12,0)},
        {"away",mode("away",.08,.14)},{"across",mode("across",-.04,-.18)}};
    if(family=="equal_probability")modes.at("away")=mode("away",.04,-.08);
    if(family=="rare_benign")modes.at("across")=mode("across",.04,.18);
    if(family=="equal_geometry")for(auto& [name,m]:modes)m=mode(name,.12,0);
    std::map<std::string,int> counts{{"continue",900-count},{"away",100},{"across",count}};
    if(family=="equal_probability")counts={{"continue",333},{"away",333},{"across",333}};
    std::map<int,ObstacleState> initial{{0,ObstacleState(5,2,0,0)},{1,ObstacleState(5,-1.4,0,0)}};
    if(family=="two_sided_trap")initial.at(1)=ObstacleState(4,-1.1,0,0);
    if(family=="rare_braking")initial.at(0)=ObstacleState(4,1.8,0,0);
    initial.at(0).x=scene_x;
    auto cfg=yaml_config::load_experiment_config().to_scenario_mpc_config();
    cfg.mpc.type=(arm=="sh_mpcc_resample"||arm=="sh_mpcc_dro_fallback")?MPCType::SH_MPCC_DRO_FALLBACK:MPCType::SH_MPCC;
    cfg.mpc.safe_horizon_enabled=false;cfg.mpc.wdro_stratified_sampling=false;
    cfg.mpc.nominal_resampling_baseline=arm=="sh_mpcc_resample";
    cfg.dro.enabled=arm=="sh_mpcc_dro"||arm=="sh_mpcc_dro_fallback";
    cfg.dro.solver.radius_calibration.use_entropic_allocator=false;
    cfg.mpc.sampling.set_manual_sample_count(arm=="sh_mpcc_extra"?2*budget:budget);
    cfg.mpc.sampling.markov_jump_system=false;cfg.mpc.sampling.max_history_length=-1;
    cfg.mpc.horizon=20;cfg.mpc.dt=.1;cfg.mpc.ego.num_discs=3;cfg.mpc.ego.length=2.;
    if(family=="rare_braking")cfg.mpc.constraints.road_width=2.;
    cfg.random_seed=seed;
    auto mm=csv(output/"mode_mechanism.csv","step,attempt,obstacle_id,mode,nominal_probability,reference_iid_probability,risk_score,rho,sampled_count,scenario_count,true_mode,reference_clearance,reference_risk");
    auto dd=csv(output/"decisions.csv","step,success,ego_speed,acceleration,omega,actual_clearance,collision,certificate_requested,certified,solve_ms,nominal_fallback_attempted");
    auto trace=csv(output/"plant.csv","step,obstacle_id,x,y,true_mode");
    auto plans=csv(output/"plans.csv","step,k,x,y,v");
    auto summary=csv(output/"summary.csv","status,collision,path_completed,min_actual_clearance,mean_controller_solve_ms,termination,first_brake_step,commitment_step,decisions,switch_reached");
    auto provenance=csv(output/"fixture.csv","family,danger_count,base_S,switch_step,history_lag,dt,commitment_reference_step,safe_horizon_enabled,plant_policy");
    provenance << family << ',' << count << ',' << budget << ',' << sw << ',' << lag << ",0.1,15,false,prescribed_adverse_switch\n";
    auto slots=csv(output/"paired_slots.csv","step,slot,obstacle_id,mode,k,x,y,bank_seed,intervened");
    auto intervention=csv(output/"intervention.csv","window,window_start,target_nd,x,noise,post_window_policy,guarantees");
    intervention << window << ',' << window_start << ',' << target << ',' << scene_x << ",zero_diffusion,paired_nominal,not_requested\n";
    int collision=0,n=0,brake=-1,commit=-1;double minimum=1e9,total_ms=0;bool completed=false;std::string termination="step_limit";
    try {
        MPCController controller(cfg);controller.set_capture_attempt_diagnostics(true);
        const auto path=ReferencePath::create_straight({0,0},{16,0});
        controller.set_reference_path(path);
        controller.initialize_obstacle(0,0,modes);
        const std::map<std::string,ModeModel> blocker{{"stationary",mode("stationary",0,0)}};
        controller.initialize_obstacle(1,1,blocker);
        int tick=-1000;
        for(const auto& [name,c]:counts)for(int i=0;i<c;++i)controller.update_mode_observation(0,0,name,tick++);
        for(int i=0;i<1000;++i)controller.update_mode_observation(1,1,"stationary",i-1000);
        auto obstacles=initial;EgoState ego(0,0,0,2);EgoDynamics dynamics(cfg.mpc.ego.dynamics,cfg.mpc.dt);
        for(int t=0;t<steps;++t){
            // Only past observations are delivered, delayed by lag steps. No future-mode leakage.
            const int observed=t-1-lag;
            if(observed>=0)controller.update_mode_observation(0,0,observed>=sw?"across":"continue",observed);
            const std::string truth=t>=sw?"across":"continue";
            trace << t << ",0," << obstacles.at(0).x << ',' << obstacles.at(0).y << ',' << truth << '\n';
            trace << t << ",1," << obstacles.at(1).x << ',' << obstacles.at(1).y << ",stationary\n";
            std::map<int,ModeHistory> histories;
            histories.emplace(0,ModeHistory(0,modes,0));histories.emplace(1,ModeHistory(1,blocker,1));
            for(const auto& [name,c]:counts)for(int i=0;i<c;++i)histories.at(0).record_observation(i,0,name);
            histories.at(1).record_observation(0,1,"stationary");
            // Match the controller's causal, lagged observations after the frozen initial state.
            // Every count arm sees the same plant history; no future truth is revealed.
            for(int past=0;past<=observed;++past)
                histories.at(0).record_observation(past,0,past>=sw?"across":"continue");
            const auto p=compute_mode_weights(histories.at(0),cfg.mpc.sampling.mode_belief);
            auto law=p;
            if((t>=window_start&&t<window_start+window)){law["across"]=0;const double other=p.at("continue")+p.at("away");law["continue"]/=other;law["away"]/=other;}
            const unsigned bank_seed=static_cast<unsigned>(seed)*1009u+static_cast<unsigned>(t)*9176u;
            std::mt19937 common_rng(bank_seed),danger_rng(bank_seed);
            const std::map<int,ModeDistribution> common_weights{{0,law},{1,{{"stationary",1.}}}};
            const std::map<int,ModeDistribution> dangerous_weights{{0,{{"across",1.}}},{1,{{"stationary",1.}}}};
            auto batch=sample_scenarios(obstacles,histories,&common_weights,cfg.mpc.horizon,budget,cfg.mpc.sampling.mode_belief,nullptr,&common_rng);
            const auto dangerous=sample_scenarios(obstacles,histories,&dangerous_weights,cfg.mpc.horizon,budget,cfg.mpc.sampling.mode_belief,nullptr,&danger_rng);
            if((t>=window_start&&t<window_start+window))for(int slot=budget-target;slot<budget;++slot)batch.at(slot).trajectories.at(0)=dangerous.at(slot).trajectories.at(0);
            if(t>=window_start&&t<=window_start+window)for(int slot=0;slot<budget;++slot)for(const auto& [id,tr]:batch.at(slot).trajectories)
                for(size_t k=0;k<tr.steps.size();++k)slots<<t<<','<<slot<<','<<id<<','<<tr.mode_id<<','<<k<<','<<tr.steps[k].mean.x()<<','<<tr.steps[k].mean.y()<<','<<bank_seed<<','<<((t>=window_start&&t<window_start+window)&&id==0&&slot>=budget-target)<<'\n';
            controller.set_experimental_scenarios(std::move(batch));
            auto result=controller.solve(ego,obstacles,{16,0},2.);
            if((t>=window_start&&t<window_start+window)){const auto& counts_seen=result.attempt_diagnostics.front().initial_mode_counts.at(0);
                const int actual=counts_seen.count("across")?counts_seen.at("across"):0;
                if(actual!=target)throw std::runtime_error("persistent count mismatch");}

            if(result.attempt_diagnostics.empty())throw std::runtime_error("missing attempt diagnostics");
            for(size_t a=0;a<result.attempt_diagnostics.size();++a){
                const auto& evidence=result.attempt_diagnostics[a];
                for(const auto& [id,weights]:evidence.nominal_weights)for(const auto& [name,p]:weights){
                    int samples=0;auto obs=evidence.initial_mode_counts.find(id);
                    if(obs!=evidence.initial_mode_counts.end()&&obs->second.count(name))samples=obs->second.at(name);
                    // Common prescribed straight reference, independent of either controller's plan.
                    double ref_clearance=1e9;auto state=obstacles.at(id);
                    const auto& model=id==0?modes.at(name):blocker.at(name);
                    for(int k=1;k<=cfg.mpc.horizon;++k){state=model.propagate(state);
                        ref_clearance=std::min(ref_clearance,clearance(EgoState(.2*(t+k),0,0,2),{{id,state}},cfg)-cfg.mpc.constraints.safety_margin);}
                    mm << t << ',' << a << ',' << id << ',' << name << ',' << p << ',' << evidence.sampling_weights.at(id).at(name) << ',';
                    if(evidence.risk_scores.count(id))mm << evidence.risk_scores.at(id).at(name);
                    mm << ',';if(evidence.radii.count(id))mm << evidence.radii.at(id);
                    mm << ',' << samples << ',' << evidence.sampled_scenarios << ',' << (id==0?truth:"stationary") << ',' << ref_clearance << ',' << std::max(0.,-ref_clearance) << '\n';
                }
            }
            for(size_t k=0;k<result.ego_trajectory.size();++k){const auto& x=result.ego_trajectory[k];plans << t << ',' << k << ',' << x.x << ',' << x.y << ',' << x.v << '\n';}
            const auto input=result.success?result.first_input():std::nullopt;
            double d=clearance(ego,obstacles,cfg);minimum=std::min(minimum,d);collision|=d<0;
            dd << t << ',' << result.success << ',' << ego.v << ',';
            if(input)dd << input->a;dd << ',';if(input)dd << input->omega;
            dd << ',' << d << ',' << (d<0) << ",0,0," << 1000*result.solve_time << ',' << result.nominal_fallback_attempted << '\n';
            ++n;total_ms+=1000*result.solve_time;
            if(ego.x>=3&&commit<0)commit=t;
            if(!input){termination="no_admissible_control";break;}
            if(input->a<-.1&&brake<0)brake=t;
            ego=dynamics.propagate(ego,*input);
            ego.s=path.find_closest_point(Eigen::Vector2d(ego.x,ego.y),ego.s);
            obstacles.at(0)=modes.at(truth).propagate(obstacles.at(0));
            d=clearance(ego,obstacles,cfg);minimum=std::min(minimum,d);collision|=d<0;
            if(collision){termination="collision";break;}
            if(ego.x>=15.2){completed=true;termination="completed";break;}
        }
        summary << "OK," << collision << ',' << completed << ',' << minimum << ',' << (n?total_ms/n:0) << ',' << termination << ',' << brake << ',' << commit << ',' << n << ',' << (n>sw) << '\n';
        std::cout << "[STRESS] " << family << ' ' << arm << " seed=" << seed << " switch=" << sw << " decisions=" << n << " collision=" << collision << " completed=" << completed << " first_brake=" << brake << " guarantees=not_requested\n";
    }catch(const std::exception& e){summary << "ERROR,,,,,execution_error,,," << n << "," << (n>sw) << '\n';std::cerr << e.what() << '\n';return 1;}
}
