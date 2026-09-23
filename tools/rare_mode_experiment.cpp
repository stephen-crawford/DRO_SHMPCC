// Fixed-scene mechanism experiment; uses production DRO, sampler and controller.
#include "experiment_config_yaml.hpp"
#include "mpc_controller.hpp"
#include "scenario_sampler.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace dro_mpc;
namespace fs=std::filesystem;

static std::ofstream output(const fs::path& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write "+path.string());
    out << std::setprecision(17);
    return out;
}

static int hits(const std::vector<Scenario>& scenarios) {
    int count=0;
    for (const auto& s:scenarios) count+=s.trajectories.at(0).mode_id=="cut_in";
    return count;
}

static double margin(const std::vector<EgoState>& plan, const ObstacleState& initial,
                     const ModeModel& model, double radius) {
    auto state=initial;
    double result=std::numeric_limits<double>::infinity();
    for (size_t k=1;k<plan.size();++k) {
        state=model.propagate(state);
        for (const auto& disc:compute_ego_disc_positions(plan[k],3,2.))
            result=std::min(result,(disc-state.position()).norm()-radius);
    }
    return result;
}

int main(int argc,char** argv) {
    if (argc!=4) {
        std::cerr << "usage: rare_mode_experiment NEW_OUTPUT_DIRECTORY COVERAGE_REPLICATES SOLVE_SEEDS\n";
        return 2;
    }
    const fs::path root=argv[1];
    const int repetitions=std::stoi(argv[2]), solve_seeds=std::stoi(argv[3]);
    if (repetitions<1 || solve_seeds<1 || fs::exists(root)) {
        std::cerr << "positive trial counts and a new output directory are required\n";
        return 2;
    }
    fs::create_directories(root);
    const int horizon=20;
    const ObstacleState obstacle(3,2,0,0);
    const EgoState ego(0,0,0,2);
    std::map<std::string,ModeModel> modes;
    for (const auto& name:{"straight","away","cut_in"}) {
        Eigen::Vector4d drift=Eigen::Vector4d::Zero();
        if (std::string(name)=="away") {drift.x()=.10;drift.y()=.05;}
        if (std::string(name)=="cut_in") drift.y()=-.20;
        modes.emplace(name,ModeModel(name,Eigen::Matrix4d::Identity(),drift,Eigen::MatrixXd::Zero(4,2)));
    }
    ModeHistory history(0,modes,0);
    const std::map<std::string,int> counts{{"straight",900},{"away",90},{"cut_in",10}};
    int tick=0;
    for (const auto& [name,count]:counts)
        for (int i=0;i<count;++i) history.record_observation(tick++,0,name);
    auto runtime=yaml_config::load_experiment_config().to_scenario_mpc_config();
    runtime.mpc.horizon=horizon;
    runtime.mpc.ego.length=2.;
    runtime.mpc.ego.num_discs=3;
    runtime.dro.solver.radius_calibration.use_entropic_allocator=false;
    const auto nominal=compute_mode_weights(history,runtime.mpc.sampling.mode_belief);
    std::vector<EgoState> reference;
    for (int k=0;k<=horizon;++k) reference.emplace_back(.2*k,0,0,2);
    DRO dro(runtime.dro.solver);
    const auto robust=dro.compute_worst_case_weights(nominal,counts,obstacle,modes,reference,
        horizon,.5,.35,.1,horizon,3,2.);
    auto weights=output(root/"weights.csv");
    weights << "mode,count,nominal_probability,wdro_probability,risk_score,rho,reference_safety_margin\n";
    for (const auto& [mode,p]:nominal)
        weights << mode << ',' << counts.at(mode) << ',' << p << ',' << robust.worst_case_weights.at(mode)
            << ',' << robust.risk_per_mode.at(mode) << ',' << robust.rho_used << ','
            << margin(reference,obstacle,modes.at(mode),.95) << '\n';
    auto scene=output(root/"scene.csv");
    scene << "k,actor,x,y\n";
    for (int k=0;k<=horizon;++k) scene << k << ",ego_reference," << reference[k].x << ",0\n";
    for (const auto& [mode,model]:modes) {
        auto state=obstacle;
        for (int k=0;k<=horizon;++k) {
            scene << k << ',' << mode << ',' << state.x << ',' << state.y << '\n';
            state=model.propagate(state);
        }
    }
    const std::map<int,ObstacleState> obstacles{{0,obstacle}};
    const std::map<int,ModeHistory> histories{{0,history}};
    const std::map<int,ModeDistribution> p_weights{{0,nominal}}, q_weights{{0,robust.worst_case_weights}};
    auto coverage=output(root/"coverage_trials.csv");
    coverage << "budget,seed,scheme,draws,cut_in_count,represented,expected_inclusion\n";
    const double pd=nominal.at("cut_in"),qd=robust.worst_case_weights.at("cut_in");
    for (int budget:{16,40,80,160}) {
        for (int trial=0;trial<repetitions;++trial) {
            const unsigned seed=static_cast<unsigned>(10001+trial);
            for (const auto& scheme:{"nominal_single","nominal_split","wdro_single","wdro_nominal_split_unconditional"}) {
                std::mt19937 rng(seed);
                const std::string name=scheme;
                const bool split=name=="nominal_split" || name=="wdro_nominal_split_unconditional";
                const bool use_dro=name=="wdro_single" || name=="wdro_nominal_split_unconditional";
                const int first=split ? budget/2 : budget;
                auto batch=sample_scenarios(obstacles,histories,use_dro ? &q_weights : &p_weights,
                    horizon,first,runtime.mpc.sampling.mode_belief,nullptr,&rng);
                int count=hits(batch);
                if (split) count+=hits(sample_scenarios(obstacles,histories,&p_weights,horizon,budget-first,
                    runtime.mpc.sampling.mode_belief,nullptr,&rng,first));
                const double expected=1.-std::pow(1.-(use_dro?qd:pd),first)*std::pow(1.-pd,budget-first);
                coverage << budget << ',' << seed << ',' << scheme << ',' << budget << ',' << count << ','
                         << (count>0) << ',' << expected << '\n';
            }
        }
    }
    auto cycles=output(root/"controller_trials.csv");
    cycles << "budget_design,base_S,seed,method,status,first_success,final_success,fallback,outer_attempts,failed_attempts,qp_calls,total_draws,first_cut_in_count,any_cut_in,cycle_ms,sample_count_sufficient,error\n";
    auto attempts=output(root/"attempts.csv");
    attempts << "budget_design,base_S,seed,method,attempt,success,dro_enabled,draws,cut_in_count,p_cut_in,q_cut_in,qp_calls,solve_ms\n";
    auto holdout=output(root/"plan_risk.csv");
    holdout << "budget_design,base_S,seed,method,plant_cut_in_probability,physical_collision_probability,safety_violation_probability,expected_safety_penetration_m,tail5_mean_safety_penetration_m,worst_safety_margin_m\n";
    auto plans=output(root/"plans.csv");
    plans << "budget_design,base_S,seed,method,k,x,y,theta\n";
    int errors=0;
    for (const std::string design:{"equal_per_attempt","equal_max_draws"}) {
        for (int s:{8,20,40}) for (int trial=0;trial<solve_seeds;++trial) {
            const unsigned seed=77+trial;
            for (const std::string method:{"nominal","nominal_nominal","wdro","wdro_nominal"}) {
                const bool hybrid=method=="nominal_nominal" || method=="wdro_nominal";
                auto cfg=runtime;
                cfg.mpc.type=hybrid ? MPCType::SH_MPCC_DRO_FALLBACK : MPCType::SH_MPCC;
                cfg.mpc.nominal_resampling_baseline=method=="nominal_nominal";
                cfg.dro.enabled=method=="wdro" || method=="wdro_nominal";
                cfg.random_seed=seed;
                cfg.mpc.sampling.set_manual_sample_count(design=="equal_max_draws" && !hybrid ? 2*s : s);
                try {
                    MPCController controller(cfg);
                    controller.set_capture_attempt_diagnostics(true);
                    controller.initialize_obstacle(0,0,modes);
                    for (const auto& o:history.observed_modes)
                        controller.update_mode_observation(0,0,o.mode_id,o.timestep);
                    const auto result=controller.solve(ego,obstacles,Eigen::Vector2d(20,0),2.);
                    const auto& aa=result.attempt_diagnostics;
                    if (aa.empty()) throw std::runtime_error("missing_attempt_evidence");
                    int draws=0,failed=0,qps=0,represented=0,first_count=0;
                    for (size_t i=0;i<aa.size();++i) {
                        const auto& a=aa[i];
                        const auto& observed=a.initial_mode_counts.at(0);
                        const int count=observed.count("cut_in") ? observed.at("cut_in") : 0;
                        if (i==0) first_count=count;
                        draws+=a.sampled_scenarios;failed+=!a.success;qps+=a.qp_calls;represented+=count;
                        attempts << design << ',' << s << ',' << seed << ',' << method << ',' << i << ','
                            << a.success << ',' << a.dro_enabled << ',' << a.sampled_scenarios << ',' << count
                            << ',' << a.nominal_weights.at(0).at("cut_in") << ',' << a.sampling_weights.at(0).at("cut_in")
                            << ',' << a.qp_calls << ',' << 1000*a.elapsed_seconds << '\n';
                    }
                    cycles << design << ',' << s << ',' << seed << ',' << method << ",OK," << aa.front().success
                        << ',' << result.success << ',' << result.nominal_fallback_attempted << ',' << aa.size()
                        << ',' << failed << ',' << qps << ',' << draws << ',' << first_count << ',' << (represented>0)
                        << ',' << 1000*result.solve_time << ',' << result.sample_count_sufficient << ",\n";
                    if (!result.success) continue; // A rejected plan has no executable-plan risk estimate.
                    for (size_t k=0;k<result.ego_trajectory.size();++k) {
                        const auto& x=result.ego_trajectory[k];
                        plans << design << ',' << s << ',' << seed << ',' << method << ',' << k << ','
                              << x.x << ',' << x.y << ',' << x.theta << '\n';
                    }
                    for (double plant_d:{pd,.05,.15,.30}) {
                        double collision=0,violation=0,expected_penetration=0,worst=1e9;
                        std::vector<std::pair<double,double>> tail;
                        for (const auto& [mode,p]:nominal) {
                            const double probability=mode=="cut_in" ? plant_d : p*(1.-plant_d)/(1.-pd);
                            const double m=margin(result.ego_trajectory,obstacle,modes.at(mode),.95);
                            const double penetration=std::max(0.,-m);
                            worst=std::min(worst,m);collision+=probability*(m+.1<0);
                            violation+=probability*(m<0);expected_penetration+=probability*penetration;
                            tail.emplace_back(penetration,probability);
                        }
                        std::sort(tail.rbegin(),tail.rend());
                        double remaining=.05,tail_sum=0;
                        for (const auto& [loss,probability]:tail) {
                            const double mass=std::min(remaining,probability);
                            tail_sum+=mass*loss;remaining-=mass;
                        }
                        holdout << design << ',' << s << ',' << seed << ',' << method << ',' << plant_d << ','
                            << collision << ',' << violation << ',' << expected_penetration << ',' << tail_sum/.05
                            << ',' << worst << '\n';
                    }
                } catch (const std::exception& error) {
                    ++errors;
                    std::cerr << "EXPERIMENT ERROR " << design << ' ' << s << ' ' << seed << ' '
                              << method << ": " << error.what() << '\n';
                    cycles << design << ',' << s << ',' << seed << ',' << method
                           << ",ERROR,,,,,,,,,,,,controller_exception_see_log\n";
                }
            }
        }
    }
    std::cout << "nominal cut_in=" << pd << " frozen-reference WDRO cut_in=" << qd
              << " coverage_replicates=" << repetitions << " solve_seeds=" << solve_seeds
              << " errors=" << errors << '\n';
    return errors ? 1 : 0;
}
