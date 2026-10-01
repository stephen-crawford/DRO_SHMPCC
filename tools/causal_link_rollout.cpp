// Follow-up geometry fixture derived from risk_stress_experiment.cpp.
// Extra CLI parameters affect this experimental scene only; production code is unchanged.
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
    if(argc!=16){std::cerr << "usage: risk_stress_experiment NEW_OUTPUT FAMILY DANGER_COUNT S SEED SWITCH_STEP STEPS HISTORY_LAG ARM X Y DRIFT_SCALE EGO_SPEED HISTORY_TOTAL OBSTACLES\n";return 2;}
    const fs::path output=argv[1];const std::string family=argv[2],arm=argv[9];
    const int count=std::stoi(argv[3]),budget=std::stoi(argv[4]),seed=std::stoi(argv[5]),sw=std::stoi(argv[6]),steps=std::stoi(argv[7]),lag=std::stoi(argv[8]);
    const std::set<std::string> families{"rare_turn","late_switch","two_sided_trap","rare_braking","rare_benign","common_dangerous","equal_probability","equal_geometry"};
    const std::set<std::string> arms{"sh_mpcc","sh_mpcc_dro","sh_mpcc_extra","sh_mpcc_resample","sh_mpcc_dro_fallback"};
    if(fs::exists(output)||!families.count(family)||!arms.count(arm)||count<1||count>900||budget<3||steps<1||sw<0||sw>=steps||lag<0)return 2;
    const double scene_x=std::stod(argv[10]),scene_y=std::stod(argv[11]),drift=std::stod(argv[12]),speed=std::stod(argv[13]);
    const int history_total=std::stoi(argv[14]),obstacle_count=std::stoi(argv[15]);
    if(history_total<100||history_total%100||obstacle_count<2||obstacle_count>4||speed<=0||drift<=0)return 2;
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
    initial.at(0)=ObstacleState(scene_x,scene_y,0,0);
    modes.at("across")=mode("across",-.04*drift,(family=="rare_benign"?.18:-.18)*drift);
    for(auto& [name,c]:counts) {if((c*history_total)%1000)throw std::runtime_error("nonintegral history counts");c=c*history_total/1000;}
    for(int id=2;id<obstacle_count;++id)initial.emplace(id,ObstacleState(scene_x+id, -2.-id,0,0));
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
    auto mm=csv(output/"mode_mechanism.csv","step,attempt,obstacle_id,mode,nominal_probability,sampling_probability,risk_score,rho,sampled_count,scenario_count,true_mode,reference_clearance,reference_risk");
    auto dd=csv(output/"decisions.csv","step,success,ego_speed,acceleration,omega,actual_clearance,collision,certificate_requested,certified,solve_ms,nominal_fallback_attempted");
    auto trace=csv(output/"plant.csv","step,obstacle_id,x,y,true_mode");
    auto plans=csv(output/"plans.csv","step,k,x,y,v");
    auto summary=csv(output/"summary.csv","status,collision,path_completed,min_actual_clearance,mean_controller_solve_ms,termination,first_brake_step,commitment_step,decisions,switch_reached");
    auto provenance=csv(output/"fixture.csv","family,danger_count,base_S,switch_step,history_lag,dt,commitment_reference_step,safe_horizon_enabled,plant_policy");
    provenance << family << ',' << count << ',' << budget << ',' << sw << ',' << lag << ",0.1,15,false,prescribed_adverse_switch\n";
    auto geometry=csv(output/"geometry.csv","x,y,drift_scale,ego_speed,history_total,obstacles");
    geometry << scene_x << ',' << scene_y << ',' << drift << ',' << speed << ',' << history_total << ',' << obstacle_count << '\n';
    int collision=0,n=0,brake=-1,commit=-1;double minimum=1e9,total_ms=0;bool completed=false;std::string termination="step_limit";
    try {
        MPCController controller(cfg);controller.set_capture_attempt_diagnostics(true);
        const auto path=ReferencePath::create_straight({0,0},{16,0});
        controller.set_reference_path(path);
        controller.initialize_obstacle(0,0,modes);
        const std::map<std::string,ModeModel> blocker{{"stationary",mode("stationary",0,0)}};
        for(int id=1;id<obstacle_count;++id)controller.initialize_obstacle(id,id,blocker);
        int tick=-history_total;
        for(const auto& [name,c]:counts)for(int i=0;i<c;++i)controller.update_mode_observation(0,0,name,tick++);
        for(int id=1;id<obstacle_count;++id)for(int i=0;i<history_total;++i)controller.update_mode_observation(id,id,"stationary",i-history_total);
        auto obstacles=initial;EgoState ego(0,0,0,speed);EgoDynamics dynamics(cfg.mpc.ego.dynamics,cfg.mpc.dt);
        for(int t=0;t<steps;++t){
            // Only past observations are delivered, delayed by lag steps. No future-mode leakage.
            const int observed=t-1-lag;
            if(observed>=0)controller.update_mode_observation(0,0,observed>=sw?"across":"continue",observed);
            const std::string truth=t>=sw?"across":"continue";
            trace << t << ",0," << obstacles.at(0).x << ',' << obstacles.at(0).y << ',' << truth << '\n';
            for(int id=1;id<obstacle_count;++id)trace << t << ',' << id << ',' << obstacles.at(id).x << ',' << obstacles.at(id).y << ",stationary\n";
            auto result=controller.solve(ego,obstacles,{16,0},speed);
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
                        ref_clearance=std::min(ref_clearance,clearance(EgoState(speed*.1*(t+k),0,0,speed),{{id,state}},cfg)-cfg.mpc.constraints.safety_margin);}
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
