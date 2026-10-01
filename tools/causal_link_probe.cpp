// Frozen-state causal experiment. Calls production DRO/sampling/MPC without changing them.
#include "experiment_config_yaml.hpp"
#include "mpc_controller.hpp"
#include "scenario_sampler.hpp"
#include "dynamics.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
using namespace dro_mpc;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point start){return 1000*std::chrono::duration<double>(Clock::now()-start).count();}
static std::ofstream csv(const fs::path& p,const std::string& h){std::ofstream f(p);if(!f)throw std::runtime_error("artifact open failed");f<<std::setprecision(17)<<h<<'\n';return f;}
static ModeModel mode(const std::string& n,double dx,double dy){return ModeModel(n,Eigen::Matrix4d::Identity(),Eigen::Vector4d(dx,dy,0,0),Eigen::MatrixXd::Zero(4,2));}
static int hits(const std::vector<Scenario>& ss){int n=0;for(const auto& s:ss)n+=s.trajectories.at(0).mode_id=="across";return n;}
static double clearance(const EgoState& e,const std::map<int,ObstacleState>& obs,const RuntimeConfig& c){double d=1e9;for(const auto& disc:compute_ego_disc_positions(e,3,2.))for(const auto& [id,x]:obs)d=std::min(d,(disc-x.position()).norm()-c.mpc.ego.radius-c.obstacle_radius);return d;}
int main(int argc,char** argv){
    if(argc!=11){std::cerr<<"usage: causal_link_probe NEW_OUT X Y ACROSS_DY G S REPLICATES QUOTA OBSTACLES PHASE(probe|conditional|scaling)\n";return 2;}
    const fs::path out=argv[1];const double x=std::stod(argv[2]),y=std::stod(argv[3]),dy=std::stod(argv[4]);
    const int g=std::stoi(argv[5]),S=std::stoi(argv[6]),reps=std::stoi(argv[7]),quota=std::stoi(argv[8]),O=std::stoi(argv[9]);const std::string phase=argv[10];
    if(fs::exists(out)||g<100||g%100||S<3||reps<1||quota<0||O<1||O>4||(phase!="probe"&&phase!="conditional"&&phase!="scaling"))return 2;
    fs::create_directories(out);
    try{
        auto cfg=yaml_config::load_experiment_config().to_scenario_mpc_config();
        cfg.mpc.type=MPCType::SH_MPCC;cfg.mpc.safe_horizon_enabled=false;cfg.mpc.wdro_stratified_sampling=false;cfg.mpc.nominal_resampling_baseline=false;
        cfg.dro.enabled=false;cfg.dro.solver.radius_calibration.use_entropic_allocator=false;
        cfg.mpc.sampling.set_manual_sample_count(S);cfg.mpc.sampling.markov_jump_system=false;cfg.mpc.sampling.max_history_length=-1;
        cfg.mpc.horizon=20;cfg.mpc.dt=.1;cfg.mpc.ego.num_discs=3;cfg.mpc.ego.length=2.;
        const std::map<std::string,ModeModel> modes{{"continue",mode("continue",.12,0)},{"away",mode("away",.08,.14)},{"across",mode("across",.12,dy)}};
        const std::map<std::string,ModeModel> stationary{{"stationary",mode("stationary",0,0)}};
        const std::map<std::string,int> counts{{"continue",85*g/100},{"away",10*g/100},{"across",5*g/100}};
        std::map<int,ModeHistory> histories;histories.emplace(0,ModeHistory(0,modes,0));int tick=-g;
        for(const auto& [name,n]:counts)for(int i=0;i<n;++i)histories.at(0).record_observation(tick++,0,name);
        std::map<int,ObstacleState> obstacles{{0,ObstacleState(x,y,0,0)}};
        for(int id=1;id<O;++id){obstacles.emplace(id,ObstacleState(x+id,-1.4-id,0,0));histories.emplace(id,ModeHistory(id,stationary,id));for(int i=0;i<g;++i)histories.at(id).record_observation(i-g,id,"stationary");}
        std::vector<EgoState> reference;for(int k=0;k<=20;++k)reference.emplace_back(.2*k,0,0,2);
        // Keep raw count proportions fixed. Report the actual smoothed center, which depends on g.
        const auto p=compute_mode_weights(histories.at(0),cfg.mpc.sampling.mode_belief);
        DRO dro(cfg.dro.solver);
        const auto robust=dro.compute_worst_case_weights(p,counts,obstacles.at(0),modes,reference,20,cfg.mpc.ego.radius,cfg.obstacle_radius,cfg.mpc.constraints.safety_margin,20,3,2.);
        const auto q=robust.worst_case_weights;
        auto weights=csv(out/"weights.csv","mode,count,p,q,r,rho,g,x,y,across_dy,S");
        for(const auto& [name,prob]:p)weights<<name<<','<<counts.at(name)<<','<<prob<<','<<q.at(name)<<','<<robust.risk_per_mode.at(name)<<','<<robust.rho_used<<','<<g<<','<<x<<','<<y<<','<<dy<<','<<S<<'\n';
        auto costs=csv(out/"transport.csv","from_mode,to_mode,cost");int i=0;for(const auto& [a,pa]:p){int j=0;for(const auto& [b,pb]:p)costs<<a<<','<<b<<','<<robust.transport_cost_matrix.at(i).at(j++)<<'\n';++i;}
        auto draws=csv(out/"draws.csv","law,seed,S,n_d,trajectory_generation_ms");
        auto outcomes=csv(out/"conditional.csv","law,seed,n_d,verified_n_d,status,collision,refusal,decisions,min_clearance,horizon_completed");
        auto mm=csv(out/"conditional_modes.csv","law,seed,step,mode,p,q,r,rho,n_m,S,true_mode");
        auto plant=csv(out/"plant.csv","step,x,y,true_mode");auto state=obstacles.at(0);for(int t=0;t<=40;++t){plant<<t<<','<<state.x<<','<<state.y<<",across\n";state=modes.at("across").propagate(state);}
        auto timings=csv(out/"scaling.csv","law,seed,obstacles,S,trajectory_generation_ms,controller_trajectory_generation_ms,fixed_reference_constraint_ms,fixed_reference_raw_constraints,controller_constraint_ms,retained_facets,qp_ms,solve_ms,success");
        for(const std::string law:{"nominal","wdro","extra_nominal"}){
            if(law=="extra_nominal"&&phase!="scaling")continue;
            const int size=law=="extra_nominal"?2*S:S;std::map<int,ModeDistribution> weights_by_obstacle{{0,law=="wdro"?q:p}};
            for(int id=1;id<O;++id)weights_by_obstacle[id]={{"stationary",1.}};
            std::map<int,int> accepted;
            for(int trial=0;trial<reps;++trial){
                const int seed=10001+trial;std::mt19937 rng(seed);auto start=Clock::now();
                auto batch=sample_scenarios(obstacles,histories,&weights_by_obstacle,20,size,cfg.mpc.sampling.mode_belief,nullptr,&rng);
                const double sample_ms=ms(start);const int n=hits(batch);draws<<law<<','<<seed<<','<<size<<','<<n<<','<<sample_ms<<'\n';
                if(phase=="probe")continue;
                // Conditional sampling: accept the first quota seeds in each Nd stratum BEFORE any outcomes.
                // No draw is replaced and no controller sampling rule is modified.
                if(phase=="conditional"&&(accepted[n]>=quota))continue;
                ++accepted[n];auto run_cfg=cfg;run_cfg.random_seed=seed;run_cfg.mpc.sampling.set_manual_sample_count(size);
                run_cfg.dro.enabled=phase=="scaling"&&law=="wdro";
                MPCController controller(run_cfg);controller.set_capture_attempt_diagnostics(true);controller.set_capture_linearized_constraints(true);
                const auto path=ReferencePath::create_straight({0,0},{16,0});controller.set_reference_path(path);
                for(const auto& [id,h]:histories){controller.initialize_obstacle(id,id,h.available_modes);for(const auto& ob:h.observed_modes)controller.update_mode_observation(id,id,ob.mode_id,ob.timestep);}
                auto current=obstacles;EgoState ego(0,0,0,2);EgoDynamics dynamics(cfg.mpc.ego.dynamics,.1);
                int collision=0,refusal=0,decisions=0,verified=-1;double minimum=clearance(ego,current,cfg);
                for(int t=0;t<(phase=="scaling"?1:40);++t){
                    if(!run_cfg.dro.enabled)for(const auto& [id,w]:weights_by_obstacle)controller.set_custom_mode_weights(id,w);
                    std::cerr<<"[CAUSAL SOLVE] law="<<law<<" seed="<<seed<<" step="<<t<<'\n';
                    const auto result=controller.solve(ego,current,{16,0},2.);++decisions;
                    if(result.attempt_diagnostics.empty())throw std::runtime_error("missing diagnostics");
                    const auto& a=result.attempt_diagnostics.front();const auto& observed=a.initial_mode_counts.at(0);
                    if(t==0){verified=observed.count("across")?observed.at("across"):0;if(phase=="conditional"&&verified!=n)throw std::runtime_error("conditional seed sampler/controller count mismatch");}
                    for(const auto& [name,prob]:p){
                        mm<<law<<','<<seed<<','<<t<<','<<name<<','<<a.nominal_weights.at(0).at(name)<<','<<a.sampling_weights.at(0).at(name)<<',';
                        if(phase!="scaling")mm<<robust.risk_per_mode.at(name);
                        else if(a.risk_scores.count(0))mm<<a.risk_scores.at(0).at(name);
                        mm<<',';
                        if(phase!="scaling")mm<<robust.rho_used;
                        else if(a.radii.count(0))mm<<a.radii.at(0);
                        mm<<','<<(observed.count(name)?observed.at(name):0)<<','<<size<<",across\n";
                    }
                    if(phase=="scaling"){
                        start=Clock::now();const auto raw=compute_linearized_constraints(reference,batch,cfg.mpc.ego.radius,cfg.obstacle_radius,cfg.mpc.constraints.safety_margin,3,2.);const double build_ms=ms(start);
                        timings<<law<<','<<seed<<','<<O<<','<<size<<','<<sample_ms<<','<<1000*a.trajectory_generation_seconds<<','<<build_ms<<','<<raw.size()<<','<<1000*result.constraint_construction_time<<','<<controller.last_linearized_constraints().size()<<','<<1000*result.qp_solve_time<<','<<1000*result.solve_time<<','<<result.success<<'\n';break;
                    }
                    collision|=clearance(ego,current,cfg)<0;
                    const auto input=result.success?result.first_input():std::nullopt;
                    if(!input){refusal=1;break;}
                    ego=dynamics.propagate(ego,*input);ego.s=path.find_closest_point({ego.x,ego.y},ego.s);
                    current.at(0)=modes.at("across").propagate(current.at(0));
                    const double d=clearance(ego,current,cfg);minimum=std::min(minimum,d);collision|=d<0;if(collision)break;
                }
                if(phase=="conditional")outcomes<<law<<','<<seed<<','<<n<<','<<verified<<",OK,"<<collision<<','<<refusal<<','<<decisions<<','<<minimum<<','<<(decisions==40&&!collision&&!refusal)<<'\n';
            }
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<"CAUSAL EXPERIMENT ERROR: "<<e.what()<<'\n';return 1;}
}
