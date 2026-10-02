// Paired closed-loop pilot and independent frozen-plan horizon validation.
#include "experiment_config_yaml.hpp"
#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "scenario_sampler.hpp"
#include "collision_constraints.hpp"
#include "wasserstein_radius_calibration.hpp"
#include <boost/math/distributions/beta.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <chrono>
using namespace dro_mpc;
namespace fs=std::filesystem;
std::ofstream file(const fs::path& p,const std::string& header){std::ofstream f(p);f<<std::setprecision(17)<<header<<'\n';return f;}
int main(int argc,char**argv) {
    if(argc!=6){std::cerr<<"usage: paper_shift_experiment output geometry delta seed arm\n";return 2;}
    fs::path out=argv[1];std::string geometry=argv[2],arm=argv[5];double delta=std::stod(argv[3]);unsigned seed=std::stoul(argv[4]);fs::create_directories(out);
    auto cfg=yaml_config::load_experiment_config();cfg.mpc.type=MPCType::SH_MPCC;cfg.mpc.safe_horizon_enabled=true;
    cfg.dro.enabled=arm=="full_proposed";
    cfg.mpc.horizon=20;cfg.rollout.rollout_steps=200;
    cfg.environment.road_length=25;cfg.environment.roundabout_radius=4;
    cfg.environment.type=geometry=="straight"?EnvironmentType::TWO_LANE_HIGHWAY:geometry=="s_curve"?EnvironmentType::S_CURVE:EnvironmentType::TWO_LANE_ROUNDABOUT;
    cfg.environment.custom_ref_path=build_environment_reference_path(cfg.environment.type,cfg.environment);
    const auto path=*cfg.environment.custom_ref_path;
    auto start=path.get_position_at(0);double heading=path.get_heading_at(0);
    cfg.environment.custom_initial_ego=EgoState(start.x(),start.y(),heading,1.5);
    const double arc=3.5;auto pos=path.get_position_at(arc);double oh=path.get_heading_at(arc);
    Eigen::Vector2d tangent(std::cos(oh),std::sin(oh)),normal(-std::sin(oh),std::cos(oh));pos+=1.5*normal;
    ObstacleState initial(pos.x(),pos.y(),-.6*tangent.x(),-.6*tangent.y());
    cfg.obstacles.initial_obstacle_states={initial};cfg.obstacles.num_obstacles=1;cfg.obstacles.num_classes=1;
    cfg.obstacles.obs_modes={"constant_velocity","turn_left","turn_right"};cfg.obstacles.rare_mode="";cfg.obstacles.randomize_available_modes=false;
    cfg.obstacles.shift={};cfg.obstacles.behavior="mode_switching";cfg.obstacles.switch_regime=ModeSwitchConfiguration::HOLD_OVER_HORIZON;
    cfg.obstacles.process_noise=1.;cfg.obstacles.speed_cap=1e6;
    cfg.artifacts.output_directory="";
    auto catalog=create_obstacle_mode_models(cfg.mpc.dt);std::map<std::string,ModeModel> modes;
    for(const auto& name:cfg.obstacles.obs_modes)modes[name]=catalog.at(name);
    std::vector<std::string> ids;for(const auto& [id,m]:modes)ids.push_back(id);
    // Predeclared frozen initial path reference selects the dangerous shift target.
    std::vector<EgoState> frozen;
    for(int k=0;k<=cfg.mpc.horizon;++k){double s=.15*k;auto x=path.get_position_at(s);frozen.emplace_back(x.x(),x.y(),path.get_heading_at(s),1.5);}
    DRO score(cfg.dro.solver);
    auto risk=score.compute_risk_vector(RiskVectorRequest{initial,modes,ids,frozen,cfg.mpc.horizon,cfg.mpc.ego.radius+cfg.obstacle_radius+cfg.mpc.constraints.safety_margin,cfg.mpc.ego.num_discs,cfg.mpc.ego.length,nullptr});
    int dangerous=0;for(int j=1;j<3;++j)if(risk.at(ids[j])>risk.at(ids[dangerous]))dangerous=j;
    std::vector<double> truth{.8,.15,.05};for(auto& v:truth)v*=1-delta;truth[dangerous]+=delta;
    const int g=25,nval=1000;
    // Same uniforms across shift levels, disjoint streams for calibration/plant/validation.
    std::mt19937 calibration_rng(1000000+seed),plant_modes_rng(2000000+seed);
    std::discrete_distribution<int> true_mode(truth.begin(),truth.end());
    std::vector<int> counts(3,0);for(int i=0;i<g;++i)++counts[true_mode(calibration_rng)];
    std::vector<double> nominal(3);for(int j=0;j<3;++j)nominal[j]=(counts[j]+.5)/(g+1.5);
    const auto cp=finite_sample_wasserstein_radius(counts,nominal,std::vector<std::vector<double>>(3,std::vector<double>(3,0)),.05);
    double minimum=std::accumulate(cp.coordinate_envelope.begin(),cp.coordinate_envelope.end(),0.0);
    std::map<std::string,double> env;for(int j=0;j<3;++j)env[ids[j]]=cp.coordinate_envelope[j]/minimum;
    bool covered=true;for(int j=0;j<3;++j)covered=covered && truth[j]>=cp.lower[j] && truth[j]<=cp.upper[j];
    if(arm=="envelope_optimal") {
        auto runtime=cfg.to_scenario_mpc_config();int S=runtime.compute_required_scenarios_for_risk(.05/minimum,6,2);cfg.mpc.sampling.set_manual_sample_count(S);
    }
    cfg.normalize();
    auto laws=file(out/"laws.csv","geometry,delta,seed,arm,mode,risk_initial,shift_target,p_true,count,g,p_nominal,L,U,u,q_env,cp_covered");
    for(int j=0;j<3;++j)laws<<geometry<<','<<delta<<','<<seed<<','<<arm<<','<<ids[j]<<','<<risk.at(ids[j])<<','<<(j==dangerous)<<','<<truth[j]<<','<<counts[j]<<','<<g<<','<<nominal[j]<<','<<cp.lower[j]<<','<<cp.upper[j]<<','<<cp.coordinate_envelope[j]<<','<<env.at(ids[j])<<','<<covered<<'\n';
    auto decisions=file(out/"heldout_decisions.csv","geometry,delta,seed,arm,step,success,certificate,support_count,S,zeta,epsilon_SH,conditional_transfer,cp_covered,calibration_tv,n_validation,violations,violation_estimate,cp95_lower,cp95_upper,estimate_exceeds_epsilon,lower_exceeds_epsilon,upper_below_epsilon,solve_ms,validation_ms,obstacle_x,obstacle_y,obstacle_vx,obstacle_vy");
    double tv=0;for(int j=0;j<3;++j)tv+=.5*std::abs(truth[j]-nominal[j]);
    int observed=0,valid=0;
    cfg.rollout.step_callback=[&](int step,int id,ObstacleSim& obstacle,MPCController& controller,std::mt19937&){
        obstacle.current_mode=ids[true_mode(plant_modes_rng)];
        // Freeze calibration data; do not accidentally add realized plant labels.
        controller.initialize_obstacle(id,id,modes);
        int time=0;for(int j=0;j<3;++j)for(int n=0;n<counts[j];++n)controller.update_mode_observation(id,id,ids[j],time++);
        if(arm=="envelope_optimal")controller.set_custom_mode_weights(id,env);
    };
    cfg.rollout.result_callback=[&](const DecisionContext& c,const MPCResult& result,const MPCController& controller){
        if(arm=="full_proposed") {
            const auto& allocation=controller.last_dro_results().at(0);
            if(allocation.radius_observation_count!=g)throw std::runtime_error("calibration count changed");
            for(int j=0;j<3;++j)if(std::abs(allocation.coordinate_envelope.at(ids[j])-cp.coordinate_envelope[j])>1e-9)throw std::runtime_error("calibration envelope changed");
        }
        ++observed;double z=arm=="envelope_optimal"?minimum:0;
        if(arm=="nominal")for(int j=0;j<3;++j)z=std::max(z,cp.coordinate_envelope[j]/nominal[j]);
        if(arm=="full_proposed")z=result.distribution_domination_factor;
        bool transfer=result.certificate_status==SafeHorizonCertificateStatus::CERTIFIED && controller.config().compute_effective_epsilon(result.sampled_scenarios,controller.config().support_limit())<=.05/z;
        auto tic=std::chrono::steady_clock::now();
        int violations=0;
        const bool evaluate=result.success && result.ego_trajectory.size()==size_t(cfg.mpc.horizon+1);
        if(evaluate){
            ++valid;
            std::mt19937 rng(3000000+seed*1000+c.step);std::normal_distribution<double> normal_dist(0,1);std::discrete_distribution<int> draw(truth.begin(),truth.end());
            std::vector<std::vector<Eigen::Vector2d>> discs;
            for(const auto& state:result.ego_trajectory)discs.push_back(compute_ego_disc_positions(state,cfg.mpc.ego.num_discs,cfg.mpc.ego.length));
            for(int i=0;i<nval;++i){
                // One independently drawn held mode per validation trajectory.
                const auto& model=modes.at(ids[draw(rng)]);auto state=c.obstacles.at(0);bool violation=false;
                for(int k=1;k<=cfg.mpc.horizon;++k){
                    Eigen::VectorXd noise(model.noise_dim());for(int d=0;d<noise.size();++d)noise[d]=normal_dist(rng);
                    state=model.propagate(state,&noise);
                    for(const auto& disc:discs[k])if((disc-state.position()).norm()<controller.config().combined_radius())violation=true;
                }
                violations+=violation;
            }
        }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tic).count();
        decisions<<geometry<<','<<delta<<','<<seed<<','<<arm<<','<<c.step<<','<<result.success<<','<<safe_horizon_certificate_status_name(result.certificate_status)<<','<<result.support_size<<','<<result.sampled_scenarios<<','<<z<<','<<.05/z<<','<<transfer<<','<<covered<<','<<tv<<',';
        if(evaluate){double v=double(violations)/nval;double lo=violations?boost::math::quantile(boost::math::beta_distribution<double>(violations,nval-violations+1),.025):0;
            double hi=violations<nval?boost::math::quantile(boost::math::complement(boost::math::beta_distribution<double>(violations+1,nval-violations),.025)):1;
            decisions<<nval<<','<<violations<<','<<v<<','<<lo<<','<<hi<<','<<(v>.05)<<','<<(lo>.05)<<','<<(hi<=.05)<<',';
        }else decisions<<"0,,,,,,,,";
        decisions<<result.solve_time*1000<<','<<ms<<','<<c.obstacles.at(0).x<<','<<c.obstacles.at(0).y<<','<<c.obstacles.at(0).vx<<','<<c.obstacles.at(0).vy<<'\n';decisions.flush();
    };
    auto summary=file(out/"rollout.csv","geometry,delta,seed,arm,status,error,decisions,evaluated_decisions,completed,steps,collision,min_clearance,progress,termination,median_solve_ms,p95_solve_ms");
    try{auto r=run_experiment_rollout(cfg,seed);summary<<geometry<<','<<delta<<','<<seed<<','<<arm<<",OK,,"<<observed<<','<<valid<<','<<r.completed_path<<','<<r.total_steps<<','<<r.collision<<','<<r.min_clearance<<','<<r.total_progress<<','<<r.termination_reason<<','<<r.p50_solve_ms<<','<<r.p95_solve_ms<<'\n';}
    catch(const std::exception& e){std::string error=e.what();for(auto& c:error)if(c==','||c=='\n')c=' ';summary<<geometry<<','<<delta<<','<<seed<<','<<arm<<",ERROR,"<<error<<','<<observed<<','<<valid<<",,,,,,,,\n";return 1;}
}
