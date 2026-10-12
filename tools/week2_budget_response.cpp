// Frozen pre-sampling budget experiments; no counterfactual control reaches the plant.
#include "experiment_config_yaml.hpp"
#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>

namespace fs = std::filesystem;
namespace dro_mpc {
// Reuse the existing friend interface used by tools/counterfactual_probe.cpp.
class CounterfactualProbe {
public:
    static void reliability(const MPCController& source,const DecisionContext& c,
                            const fs::path& root,unsigned context_seed,int repetitions,bool conflicts=false) {
        std::ofstream out(root/"replications.csv");
        out << std::setprecision(17)
            << "context_seed,replicate,candidate,sampling_seed,entered_qp,certified,support_evaluated,support,failure_stage,solve_ms,retained_clearance,predicted_progress\n";
        for (int replicate=0;replicate<repetitions;++replicate) {
            // Paired batches across candidates; disjoint deterministic streams across repetitions.
            std::seed_seq stream{0x41564c42u,context_seed,unsigned(replicate)};
            unsigned sampling_seed;stream.generate(&sampling_seed,&sampling_seed+1);
            for (const std::string candidate : {"nominal","brake","left","right"}) {
                if (conflicts && candidate!=(context_seed==77?"right":"left")) continue;
                auto copy=fork(source,10,sampling_seed);
                copy->set_capture_linearized_constraints(conflicts);
                copy->experimental_anchor_candidate_=candidate;
                std::ostringstream log;
                auto* previous=std::cerr.rdbuf(log.rdbuf());
                MPCResult result;
                try {result=solve(*copy,c);} catch (...) {std::cerr.rdbuf(previous);throw;}
                std::cerr.rdbuf(previous);
                const auto transcript=log.str();
                const bool entered=transcript.find("[SQP] before_solve")!=std::string::npos;
                const bool certified=result.certificate_status==SafeHorizonCertificateStatus::CERTIFIED;
                const bool evaluated=result.support_cap_status!=SupportCapStatus::NOT_EVALUATED;
                std::string stage=certified?"accepted":safe_horizon_certificate_status_name(result.certificate_status);
                if (!entered) stage=transcript.find("[FREE POLY EMPTY]")!=std::string::npos?"facet_empty":"pre_qp_other";
                else if (!certified && transcript.find("converged=0")!=std::string::npos) stage="qp_failure";
                if (certified && (result.sampled_scenarios!=1351 || !result.success || result.used_fallback ||
                    !result.support_iterations_feasible || result.support_size>10))
                    throw std::runtime_error("replication acceptance accounting mismatch");
                out<<context_seed<<','<<replicate<<','<<candidate<<','<<sampling_seed<<','<<entered<<','
                   <<certified<<','<<evaluated<<','<<result.support_size<<','<<stage<<','<<1000*result.solve_time<<','
                   <<clearance(*copy,result,true)<<','
                   <<(result.success && !result.ego_trajectory.empty()?result.ego_trajectory.back().s-c.path_progress:
                      std::numeric_limits<double>::quiet_NaN())<<'\n';
                if (conflicts) {
                    const auto stem=candidate+"_rep"+std::to_string(replicate);
                    std::ofstream raw(root/(stem+".log"));raw<<transcript;
                    if (!entered || replicate<10) {
                        std::ofstream rows(root/(stem+"_rows.csv"));
                        rows<<std::setprecision(17)<<"k,disc,id,obstacle,mode,a_x,a_y,b,anchor_x,anchor_y,anchor_slack,obs_x,obs_y,removed_final\n";
                        for (const auto& row:copy->last_pre_reduction_constraints_) {
                            const auto scenario=std::find_if(copy->scenarios_.begin(),copy->scenarios_.end(),[&](const Scenario& v){return v.scenario_id==row.scenario_id;});
                            const auto& tr=scenario->trajectories.at(row.obstacle_id);
                            const auto point=tr.steps.at(row.k).mean;
                            rows<<row.k<<','<<row.disc_index<<','<<row.scenario_id<<','<<row.obstacle_id<<','<<tr.mode_id<<','<<row.a.x()<<','<<row.a.y()<<','<<row.b<<','<<row.linearization_point.x()<<','<<row.linearization_point.y()<<','<<row.evaluate(row.linearization_point)<<','<<point.x()<<','<<point.y()<<','<<copy->last_removed_scenario_ids_.count(row.scenario_id)<<'\n';
                        }
                    }
                }
                if (replicate==0 || (replicate==1 && !certified)) {
                    std::ofstream raw(root/(candidate+"_rep"+std::to_string(replicate)+".log"));raw<<transcript;
                }
            }
            out.flush();
            if ((replicate+1)%10==0) std::cout<<"context "<<context_seed<<" batches "<<replicate+1<<'\n'<<std::flush;
        }
    }

    static void removal_replay(const MPCController& source,const DecisionContext& c,
                               const fs::path& root,const fs::path& selections,unsigned context_seed) {
        const auto original_rng=rng_state(source);
        std::ifstream in(selections);
        if (!in) throw std::runtime_error("cannot read deletion selections");
        std::string line;std::getline(in,line);
        if (line!="replicate,sampling_seed,candidate,removed_id") throw std::runtime_error("invalid selections header");
        std::ofstream out(root/"replays.csv");
        out<<std::setprecision(17)<<"context_seed,replicate,candidate,sampling_seed,variant,requested_deletion,actual_deletion,entered_qp,any_qp_converged,last_qp_converged,certified,success,fallback,support,support_evaluated,retained_clearance,physical_strict,physical_numerical,deterministic_admissible,solve_ms,certificate_status\n";
        while (std::getline(in,line)) {
            std::istringstream fields(line);std::string rep,seed,candidate,deletion;
            if (!std::getline(fields,rep,',') || !std::getline(fields,seed,',') ||
                !std::getline(fields,candidate,',') || !std::getline(fields,deletion))
                throw std::runtime_error("invalid selections row");
            const unsigned sampling_seed=std::stoul(seed);const int removed=std::stoi(deletion);
            std::vector<CollisionConstraint> baseline_rows;
            for (bool forced : {false,true}) {
                auto copy=fork(source,10,sampling_seed);
                copy->experimental_anchor_candidate_=candidate;
                copy->experimental_removed_scenario_=forced?removed:-1;
                copy->set_capture_linearized_constraints(true);
                std::ostringstream log;auto* previous=std::cerr.rdbuf(log.rdbuf());MPCResult result;
                try {result=solve(*copy,c);} catch (...) {std::cerr.rdbuf(previous);throw;}
                std::cerr.rdbuf(previous);
                if (!forced) baseline_rows=copy->last_pre_reduction_constraints_;
                else {
                    const auto& rows=copy->last_pre_reduction_constraints_;
                    if (rows.size()!=baseline_rows.size()) throw std::runtime_error("paired row count differs");
                    for (size_t i=0;i<rows.size();++i) {
                        const auto& a=rows[i];const auto& b=baseline_rows[i];
                        if (a.scenario_id!=b.scenario_id || a.obstacle_id!=b.obstacle_id || a.k!=b.k ||
                            a.disc_index!=b.disc_index || a.a!=b.a || a.b!=b.b || a.linearization_point!=b.linearization_point)
                            throw std::runtime_error("paired initial collision rows differ");
                    }
                    if (copy->last_removed_scenario_ids_!=std::set<int>{removed})
                        throw std::runtime_error("forced deletion not retained as the sole removed scenario");
                }
                const auto transcript=log.str();const std::string variant=forced?"forced":"production";
                std::ofstream raw(root/(candidate+"_rep"+rep+"_"+variant+".log"));raw<<transcript;
                const bool entered=transcript.find("[SQP] before_solve")!=std::string::npos;
                const bool any_converged=transcript.find("[SQP] after_solve")!=std::string::npos &&
                    transcript.find("converged=1 status=")!=std::string::npos;
                int last_converged=-1;const auto last=transcript.rfind("[SQP] after_solve");
                if (last!=std::string::npos) last_converged=transcript.find("converged=1",last)<transcript.find('\n',last)?1:0;
                const bool certified=result.certificate_status==SafeHorizonCertificateStatus::CERTIFIED;
                const bool plan=result.success && !result.used_fallback;
                const double margin=clearance(*copy,result,true);
                const bool admissible=copy->trajectory_is_deterministically_admissible(result.ego_trajectory,result.control_inputs);
                if (certified && (result.sampled_scenarios!=1351 || !plan || !result.support_iterations_feasible ||
                    result.support_size>10 || !admissible)) throw std::runtime_error("forced-replay certificate accounting mismatch");
                out<<context_seed<<','<<rep<<','<<candidate<<','<<sampling_seed<<','<<variant<<','
                   <<(forced?removed:-1)<<','<<(copy->last_removed_scenario_ids_.empty()?-1:*copy->last_removed_scenario_ids_.begin())<<','
                   <<entered<<','<<any_converged<<','<<last_converged<<','<<certified<<','<<result.success<<','<<result.used_fallback<<','
                   <<result.support_size<<','<<(result.support_cap_status!=SupportCapStatus::NOT_EVALUATED)<<','<<margin<<','
                   <<(plan && admissible && std::isfinite(margin) && margin>=0)<<','
                   <<(plan && admissible && std::isfinite(margin) && margin>=-copy->config_.solver.qp_tolerance)<<','
                   <<admissible<<','<<1000*result.solve_time<<','<<safe_horizon_certificate_status_name(result.certificate_status)<<'\n';
                out.flush();
            }
        }
        if (rng_state(source)!=original_rng) throw std::runtime_error("live RNG changed by removal replay");
        std::ofstream checks(root/"pair_checks.json");checks<<"{\"initial_rows_identical\":true,\"source_rng_unchanged\":true}\n";
    }

    static QPProblem minimum_slack_qp(const QPProblem& base,size_t collision_rows) {
        const int D=base.H.rows();
        QPProblem augmented;
        augmented.H=Eigen::MatrixXd::Zero(D+1,D+1);augmented.g=Eigen::VectorXd::Zero(D+1);
        augmented.C=Eigen::MatrixXd::Zero(base.C.rows(),D+1);
        augmented.C.leftCols(D)=base.C;
        augmented.C.topRows(collision_rows).col(D).setOnes();augmented.d=base.d;
        augmented.lb=Eigen::VectorXd::Zero(D+1);augmented.ub=Eigen::VectorXd::Zero(D+1);
        augmented.lb.head(D)=base.lb;augmented.ub.head(D)=base.ub;
        augmented.lb(D)=0;augmented.ub(D)=std::numeric_limits<double>::infinity();augmented.g(D)=1;
        return augmented;
    }

    static void slack_fixture_tests() {
        AcadosQPSolver solver;const QPSettings settings;
        for (int fixture=0;fixture<3;++fixture) {
            QPProblem base;base.H=Eigen::MatrixXd::Constant(1,1,2.);base.g=Eigen::VectorXd::Zero(1);
            base.lb=Eigen::VectorXd::Constant(1,-1.);base.ub=Eigen::VectorXd::Constant(1,1.);
            base.C=Eigen::MatrixXd::Zero(fixture==0?1:2,1);base.d=Eigen::VectorXd::Zero(base.C.rows());
            base.C(0,0)=1.;base.d(0)=fixture==0?.5:(fixture==1?2.:21.);
            if (fixture>0) {base.C(1,0)=-1.;base.d(1)=fixture==1?2.:-.25;}
            auto problem=minimum_slack_qp(base,fixture==2?1:base.C.rows());
            const double expected=fixture==0?0.:(fixture==1?2.:20.75);
            auto primary=solver.solve(problem,settings);
            if (!primary.converged || primary.x.size()!=2 || std::abs(primary.x(1)-expected)>settings.tolerance)
                throw std::runtime_error("analytical minimum-slack fixture failed");
            problem.ub(1)=std::max(0.,primary.x(1));problem.H(0,0)=2.;problem.g(1)=0.;
            auto secondary=solver.solve(problem,settings);
            if (!secondary.converged || secondary.x.size()!=2 ||
                (problem.C*secondary.x-problem.d).minCoeff()<-settings.tolerance ||
                std::abs(secondary.x(1)-expected)>settings.tolerance)
                throw std::runtime_error("analytical cost-phase fixture failed");
            std::cout<<"fixture="<<fixture<<" expected_slack="<<expected<<" primary="<<primary.x(1)
                     <<" secondary="<<secondary.x(1)<<" passed\n";
        }
    }

    static void shared_slack(const MPCController& source,const DecisionContext& c,
                             const fs::path& root,unsigned context_seed,int repetitions,bool fixed=false,double tightening=0.0,int screen_h=0) {
        const auto original_rng=rng_state(source);
        if (!screen_h && (!source.reference_trajectory_.empty() || source.has_feasible_backup_ || (context_seed!=77 && context_seed!=78)))
            throw std::runtime_error("shared-slack diagnostic requires the two cold startup contexts");
        std::ofstream extrema(root/"extrema.csv"),native(root/"native_rows.csv"), screen(root/"screen.csv"), prefixes(root/"prefixes.csv"), plans(root/"plans.csv");
        plans<<std::setprecision(17)<<"replicate,k,x,y,theta,v,s,a,omega\n";
        screen<<std::setprecision(17)<<"replicate,h,S,reconstruction,solver_pass,deterministic_pass,s_cert,candidate_accept,operational_accept,progress,physical_clearance,solve_ms,fallback\n";
        prefixes<<"replicate,count,fingerprint\n";
        extrema<<std::setprecision(17)<<"replicate,k,disc,obstacle,mode,id,a_x,a_y,b,sample_count\n";
        native<<std::setprecision(17)<<"replicate,k,disc,obstacle,id,a_x,a_y,b\n";
        std::ofstream out(root/"slack_replays.csv");
        out<<std::setprecision(17)<<"context_seed,replicate,candidate,sampling_seed,baseline_certified,baseline_qp,qp_iterations,all_qps_converged,augmented_feasible,slack,final_violation,physical_clearance,strict_zero,slack_le_1e4,n_active,union_active,within_h10,deterministic_admissible,zero_slack_candidate,tightening,lifted_slack,margin_candidate,projection_reserve\n";
        for (int replicate=0;replicate<repetitions;++replicate) {
            std::seed_seq stream{0x41564c42u,context_seed,unsigned(replicate)};
            unsigned sampling_seed;stream.generate(&sampling_seed,&sampling_seed+1);
            const auto started=std::chrono::steady_clock::now();
            auto copy=fork(source,screen_h?screen_h:10,sampling_seed);
            copy->experimental_anchor_candidate_=screen_h?"":(context_seed==77?"right":"left");
            if (fixed) {
                copy->experimental_fixed_normals_=true;
                copy->experimental_collision_tightening_=tightening;
                copy->config_.mpc.constraints.scenario_removal_budget=0;
                copy->config_.mpc.constraints.support_cap_n_bar=screen_h?screen_h:10;
            }
            copy->set_capture_linearized_constraints(true);
            std::ostringstream baseline_log;auto* previous=std::cerr.rdbuf(baseline_log.rdbuf());MPCResult baseline;
            try {baseline=solve(*copy,c);} catch (...) {std::cerr.rdbuf(previous);throw;}
            std::cerr.rdbuf(previous);
            std::ofstream raw(root/("rep"+std::to_string(replicate)+"_production.log"));raw<<baseline_log.str();
            if (screen_h) {
                std::ostringstream fingerprint;fingerprint<<std::setprecision(17);
                int count=0;
                for (const auto& scenario:copy->scenarios_) {
                    fingerprint<<scenario.scenario_id<<';';
                    for (const auto& [id,tr]:scenario.trajectories) {
                        fingerprint<<id<<':'<<tr.mode_id<<';';
                        for (const auto& point:tr.steps) fingerprint<<point.mean.x()<<','<<point.mean.y()<<';';
                    }
                    ++count;
                    if (count==550 || count==666 || count==781 || count==895 || count==1009 || count==1123 || count==1351)
                        prefixes<<replicate<<','<<count<<','<<std::hash<std::string>{}(fingerprint.str())<<'\n';
                }
            }
            // All original rows survive: hard-polygon clipping is invalid in (p,s).
            auto rows=copy->last_pre_reduction_constraints_;
            rows.erase(std::remove_if(rows.begin(),rows.end(),[](const CollisionConstraint& r){return r.k<1;}),rows.end());
            const auto all_rows=rows;
            if (fixed) {
                using Key=std::tuple<int,int,int,std::string>;
                std::map<Key,std::pair<CollisionConstraint,int>> maximum;
                for (const auto& row:rows) {
                    const auto& mode=copy->scenarios_.at(row.scenario_id).trajectories.at(row.obstacle_id).mode_id;
                    const Key key{row.k,row.disc_index,row.obstacle_id,mode};
                    const auto found=maximum.find(key);
                    if (found==maximum.end()) maximum[key]={row,1};
                    else {
                        if (found->second.first.a!=row.a) throw std::runtime_error("nonparallel compression group");
                        ++found->second.second;
                        if (row.b>found->second.first.b || (row.b==found->second.first.b && row.scenario_id<found->second.first.scenario_id)) found->second.first=row;
                    }
                }
                rows.clear();
                for (const auto& [key,value]:maximum) {
                    const auto& r=value.first;rows.push_back(r);
                    extrema<<replicate<<','<<r.k<<','<<r.disc_index<<','<<r.obstacle_id<<','<<std::get<3>(key)<<','<<r.scenario_id<<','<<r.a.x()<<','<<r.a.y()<<','<<r.b<<','<<value.second<<'\n';
                }
                for (const auto& r:copy->last_linearized_constraints_)
                    native<<replicate<<','<<r.k<<','<<r.disc_index<<','<<r.obstacle_id<<','<<r.scenario_id<<','<<r.a.x()<<','<<r.a.y()<<','<<r.b<<'\n';
            }
            copy->config_.mpc.constraints.scenario_removal_budget=0;
            copy->config_.mpc.constraints.support_cap_n_bar=screen_h?screen_h:10;
            if (copy->config_.compute_required_scenarios()!=copy->scenarios_.size() || (!screen_h && copy->scenarios_.size()!=1351))
                throw std::runtime_error("slack experiment must preserve the certification count");
            copy->last_removed_scenario_ids_.clear();
            auto ego=c.ego;
            if (!ego.has_spline()) ego.s=copy->reference_path_->find_closest_point(ego.position());
            const int N=copy->config_.mpc.horizon,D=2*N;
            // Reconstruct the pre-batch cold nominal; never reuse the solved baseline.
            auto seed_copy=fork(source,screen_h?screen_h:10,0,true);
            seed_copy->initialize_reference_trajectory(ego,c.goal,c.reference_velocity);
            auto states=screen_h?seed_copy->reference_trajectory_:copy->generate_straight_line_trajectory(ego,c.goal,c.reference_velocity);
            std::vector<EgoInput> inputs;
            for (int k=0;k<N;++k) inputs.emplace_back(
                std::clamp((states[k+1].v-states[k].v)/copy->config_.mpc.dt,
                    copy->config_.mpc.ego.dynamics.min_acceleration,copy->config_.mpc.ego.dynamics.max_acceleration),
                std::clamp((states[k+1].theta-states[k].theta)/copy->config_.mpc.dt,
                    -copy->config_.mpc.ego.dynamics.max_omega,copy->config_.mpc.ego.dynamics.max_omega));
            states=copy->ego_dynamics_.rollout_with_spline(ego,inputs,*copy->reference_path_);
            const auto settings=QPSettings{copy->config_.solver.qp_max_iterations,copy->config_.solver.qp_tolerance};
            std::ofstream trace(root/("rep"+std::to_string(replicate)+"_slack.csv"));
            trace<<std::setprecision(17)<<"iteration,phase,converged,status,slack,primal_residual,dual_residual,minimum_row_residual,active_count,active_ids\n";
            bool all_converged=true;int iterations=0;double slack=std::numeric_limits<double>::quiet_NaN();
            std::set<int> cumulative,final_active;
            for (int iteration=0;iteration<copy->config_.solver.sqp_max_iterations;++iteration) {
                auto base=copy->build_condensed_qp(states,inputs,c.goal,c.reference_velocity,rows,c.path_progress,c.path_length);
                auto augmented=minimum_slack_qp(base,rows.size());
                Eigen::VectorXd step;
                for (int phase=0;phase<2;++phase) {
                    auto result=copy->qp_solver_.solve(augmented,settings);
                    bool valid=result.converged && result.x.size()==D+1 && result.x.allFinite();
                    Eigen::VectorXd residual;std::set<int> active;
                    if (valid) {
                        residual=augmented.C*result.x-augmented.d;
                        valid=residual.minCoeff()>=-settings.tolerance &&
                            (result.x.array()>=augmented.lb.array()-settings.tolerance).all() &&
                            (result.x.array()<=augmented.ub.array()+settings.tolerance).all();
                        for (size_t j=0;j<rows.size();++j)
                            if (residual(j)>=-settings.tolerance && residual(j)<=1e-3) active.insert(rows[j].scenario_id);
                    }
                    trace<<iteration<<','<<phase<<','<<result.converged<<','<<result.status<<','
                         <<(result.x.size()==D+1?result.x(D):std::numeric_limits<double>::quiet_NaN())<<','
                         <<result.primal_residual<<','<<result.dual_residual<<','
                         <<(residual.size()?residual.minCoeff():std::numeric_limits<double>::quiet_NaN())<<','<<active.size()<<',';
                    for (int id:active) trace<<id<<';';
                    trace<<'\n';
                    if (!valid) {all_converged=false;break;}
                    cumulative.insert(active.begin(),active.end());
                    if (phase==0) {
                        // Numerical lexicographic cap: no added relaxation beyond the returned LP level.
                        augmented.ub(D)=std::max(0.0,result.x(D));
                        augmented.H.topLeftCorner(D,D)=base.H;augmented.g.head(D)=base.g;augmented.g(D)=0;
                    } else {step=result.x.head(D);slack=result.x(D);final_active=active;}
                }
                if (!all_converged) break;
                ++iterations;
                for (int k=0;k<N;++k) {
                    inputs[k].a=std::clamp(inputs[k].a+step(2*k),copy->config_.mpc.ego.dynamics.min_acceleration,copy->config_.mpc.ego.dynamics.max_acceleration);
                    inputs[k].omega=std::clamp(inputs[k].omega+step(2*k+1),-copy->config_.mpc.ego.dynamics.max_omega,copy->config_.mpc.ego.dynamics.max_omega);
                }
                states=copy->ego_dynamics_.rollout_with_spline(ego,inputs,*copy->reference_path_);
                if (step.norm()<copy->config_.solver.sqp_convergence_tol) break;
            }
            double violation=0.;for (const auto& row:all_rows)
                violation=std::max(violation,-row.evaluate(compute_ego_disc_positions(states[row.k],copy->config_.mpc.ego.num_discs,copy->config_.mpc.ego.length).at(row.disc_index)));
            const bool strict_admissible=copy->trajectory_is_deterministically_admissible(states,inputs);
            const bool admissible=copy->trajectory_is_deterministically_admissible(states,inputs,settings.tolerance);
            const bool feasible=all_converged && iterations>0 && std::isfinite(slack) && slack>=-settings.tolerance &&
                admissible && violation<=slack+settings.tolerance;
            MPCResult diagnostic;diagnostic.ego_trajectory=states;
            out<<context_seed<<','<<replicate<<','<<copy->experimental_anchor_candidate_<<','<<sampling_seed<<','
               <<(baseline.certificate_status==SafeHorizonCertificateStatus::CERTIFIED)<<','
               <<(baseline_log.str().find("[SQP] before_solve")!=std::string::npos)<<','<<iterations<<','<<all_converged<<','<<feasible<<','
               <<slack<<','<<violation<<','<<clearance(*copy,diagnostic)<<','
               <<(feasible && slack<=0 && violation==0 && strict_admissible)<<','<<(feasible && slack<=1e-4)<<','
               <<final_active.size()<<','<<cumulative.size()<<','<<(cumulative.size()<=10)<<','<<admissible<<','<<(all_converged && iterations>0 && violation==0 && strict_admissible)<<','<<tightening<<','
               <<std::max({0.0,slack,violation})<<','
               <<(all_converged && iterations>0 && strict_admissible && std::isfinite(slack) && std::max({0.0,slack,violation})<=tightening)<<','
               <<tightening-std::max({0.0,slack,violation})<<'\n';out.flush();
            if (screen_h) {
                for(size_t k=0;k<states.size();++k) {
                    const auto& state=states[k];
                    plans<<replicate<<','<<k<<','<<state.x<<','<<state.y<<','<<state.theta<<','<<state.v<<','<<state.s<<',';
                    if(k<inputs.size())plans<<inputs[k].a<<','<<inputs[k].omega;else plans<<',';
                    plans<<'\n';
                }
                std::set<int> reconstruction;for (const auto& row:rows) reconstruction.insert(row.scenario_id);
                const double lifted=std::max({0.0,slack,violation});
                const bool gates=all_converged && iterations>0 && strict_admissible && std::isfinite(slack);
                const bool accepted=gates && lifted<=tightening && reconstruction.size()<=size_t(screen_h);
                const double progress=states.back().s-c.path_progress;
                screen<<replicate<<','<<screen_h<<','<<copy->scenarios_.size()<<','<<reconstruction.size()<<','<<all_converged<<','<<strict_admissible<<','<<lifted<<','<<accepted<<','<<(accepted && progress>=0.5)<<','<<progress<<','<<clearance(*copy,diagnostic)<<','<<1000*std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<','<<(!accepted)<<'\n';screen.flush();
            }
            if ((replicate+1)%10==0) std::cout<<"slack context "<<context_seed<<" batches "<<replicate+1<<'\n'<<std::flush;
        }
        if (rng_state(source)!=original_rng) throw std::runtime_error("slack replay changed the source RNG");
    }

    static void fixed_normals(const MPCController& source,const DecisionContext& c,
                              const fs::path& root,unsigned context_seed,int repetitions) {
        const auto original_rng=rng_state(source);
        if (context_seed!=77 && context_seed!=78) throw std::runtime_error("fixed-normal replay requires contexts 77/78");
        std::ofstream out(root/"replays.csv");
        out<<std::setprecision(17)<<"context_seed,replicate,candidate,sampling_seed,variant,entered_qp,any_qp_converged,last_qp_converged,hard_feasible,certified,support_evaluated,support,physical_clearance,strict_physical,facet_empty,solve_ms\n";
        std::ofstream groups(root/"groups.csv");
        groups<<std::setprecision(17)<<"replicate,k,disc,obstacle,mode,a_x,a_y,max_b,extreme_id,sample_count,anchor_x,anchor_y\n";
        std::ofstream controls(root/"plans.csv");
        controls<<std::setprecision(17)<<"replicate,k,x,y,theta,v,s,a,omega\n";
        for (int replicate=0;replicate<repetitions;++replicate) {
            std::seed_seq stream{0x41564c42u,context_seed,unsigned(replicate)};
            unsigned sampling_seed;stream.generate(&sampling_seed,&sampling_seed+1);
            std::vector<Scenario> baseline_samples;
            for (bool fixed:{false,true}) {
                auto copy=fork(source,10,sampling_seed);
                copy->experimental_anchor_candidate_=context_seed==77?"right":"left";
                copy->experimental_fixed_normals_=fixed;copy->set_capture_linearized_constraints(true);
                if (fixed) {
                    copy->config_.mpc.constraints.scenario_removal_budget=0;
                    copy->config_.mpc.constraints.support_cap_n_bar=10;
                    if (copy->config_.compute_required_scenarios()!=1351) throw std::runtime_error("fixed-normal sample count changed");
                }
                std::ostringstream log;auto* previous=std::cerr.rdbuf(log.rdbuf());MPCResult result;
                try {result=solve(*copy,c);} catch (...) {std::cerr.rdbuf(previous);throw;}
                std::cerr.rdbuf(previous);const auto transcript=log.str();
                if (!fixed) baseline_samples=copy->scenarios_;
                else {
                    if (copy->scenarios_.size()!=baseline_samples.size() || !copy->last_removed_scenario_ids_.empty())
                        throw std::runtime_error("fixed-normal sample/removal accounting differs");
                    for (size_t i=0;i<baseline_samples.size();++i) {
                        const auto& sample=copy->scenarios_[i];const auto& other=baseline_samples[i];
                        if (sample.scenario_id!=other.scenario_id) throw std::runtime_error("paired scenario ID changed");
                        for (const auto& [id,tr]:sample.trajectories) {
                            const auto& old=other.trajectories.at(id);
                            if (tr.mode_id!=old.mode_id || tr.steps.size()!=old.steps.size()) throw std::runtime_error("paired modes changed");
                            for (size_t k=0;k<tr.steps.size();++k)
                                if (tr.steps[k].mean!=old.steps[k].mean) throw std::runtime_error("paired sampled positions changed");
                        }
                    }
                    using Key=std::tuple<int,int,int,std::string>;
                    std::map<Key,std::pair<CollisionConstraint,int>> extremes;
                    for (const auto& row:copy->last_pre_reduction_constraints_) {
                        if (row.k<1) continue;
                        const auto& mode=copy->scenarios_.at(row.scenario_id).trajectories.at(row.obstacle_id).mode_id;
                        const Key key{row.k,row.disc_index,row.obstacle_id,mode};
                        const auto found=extremes.find(key);
                        if (found==extremes.end()) extremes[key]={row,1};
                        else {
                            if (found->second.first.a!=row.a) throw std::runtime_error("normal varies within a fixed group");
                            ++found->second.second;
                            if (row.b>found->second.first.b || (row.b==found->second.first.b && row.scenario_id<found->second.first.scenario_id)) found->second.first=row;
                        }
                    }
                    for (const auto& [key,value]:extremes) {
                        const auto& row=value.first;
                        groups<<replicate<<','<<row.k<<','<<row.disc_index<<','<<row.obstacle_id<<','<<std::get<3>(key)<<','<<row.a.x()<<','<<row.a.y()<<','<<row.b<<','<<row.scenario_id<<','<<value.second<<','<<row.linearization_point.x()<<','<<row.linearization_point.y()<<'\n';
                    }
                    for (size_t k=0;k<result.ego_trajectory.size();++k) {
                        const auto& state=result.ego_trajectory[k];
                        controls<<replicate<<','<<k<<','<<state.x<<','<<state.y<<','<<state.theta<<','<<state.v<<','<<state.s<<',';
                        if (k<result.control_inputs.size()) controls<<result.control_inputs[k].a<<','<<result.control_inputs[k].omega;
                        else controls<<"nan,nan";
                        controls<<'\n';
                    }
                }
                const std::string variant=fixed?"fixed":"production";
                std::ofstream raw(root/("rep"+std::to_string(replicate)+"_"+variant+".log"));raw<<transcript;
                const bool entered=transcript.find("[SQP] before_solve")!=std::string::npos;
                const bool any_converged=transcript.find("converged=1 status=")!=std::string::npos;
                const auto last=transcript.rfind("[SQP] after_solve");
                const int last_converged=last==std::string::npos?-1:(transcript.find("converged=1",last)<transcript.find('\n',last)?1:0);
                const bool certified=result.certificate_status==SafeHorizonCertificateStatus::CERTIFIED;
                const bool usable=result.success && !result.used_fallback && result.sampled_constraints_satisfied;
                const double margin=clearance(*copy,result,true);
                if (certified && (!usable || result.support_size>10 || !result.support_iterations_feasible)) throw std::runtime_error("fixed-normal certificate accounting mismatch");
                out<<context_seed<<','<<replicate<<','<<copy->experimental_anchor_candidate_<<','<<sampling_seed<<','<<variant<<','
                   <<entered<<','<<any_converged<<','<<last_converged<<','<<usable<<','<<certified<<','
                   <<(result.support_cap_status!=SupportCapStatus::NOT_EVALUATED)<<','<<result.support_size<<','<<margin<<','
                   <<(usable && margin>=0 && copy->trajectory_is_deterministically_admissible(result.ego_trajectory,result.control_inputs))<<','
                   <<(transcript.find("[FREE POLY EMPTY]")!=std::string::npos)<<','<<1000*result.solve_time<<'\n';out.flush();
            }
            if ((replicate+1)%10==0) std::cout<<"fixed context "<<context_seed<<" batches "<<replicate+1<<'\n'<<std::flush;
        }
        if (rng_state(source)!=original_rng) throw std::runtime_error("fixed-normal replay changed source RNG");
        std::ofstream checks(root/"pair_checks.json");checks<<"{\"sampled_trajectories_identical\":true,\"source_rng_unchanged\":true}\n";
    }

    static void diagnose(const MPCController& source, const DecisionContext& c,
                         const fs::path& root, int M) {
        auto output=[](const fs::path& path,const std::string& header) {
            std::ofstream f(path); if (!f) throw std::runtime_error("cannot write diagnostic");
            f<<std::setprecision(17)<<header<<'\n'; return f;
        };
        auto base = fork(source,10,0,true);
        auto ego = c.ego;
        if (!base->reference_path_) base->reference_path_ = ReferencePath::create_straight(ego.position(),c.goal);
        if (!ego.has_spline()) ego.s = base->reference_path_->find_closest_point(ego.position());
        for (const auto& [id,obs] : c.obstacles)
            if (!base->mode_histories_.count(id)) base->initialize_obstacle(id,0);
        base->initialize_reference_trajectory(ego,c.goal,c.reference_velocity);
        const auto deterministic = base->reference_trajectory_;
        std::seed_seq seed{0x414e4348u,base->config_.random_seed,
                          static_cast<unsigned>(base->iteration_count_+1)};
        std::mt19937 pilot_rng(seed), certification_rng=base->rng_;
        const auto& cfg=base->config_;
        auto draw=[&](int count,std::mt19937& rng) {
            return dro_mpc::sample_scenarios(c.obstacles,base->mode_histories_,nullptr,
                cfg.mpc.horizon,count,cfg.mpc.sampling.mode_belief,nullptr,&rng);
        };
        const auto gamma=draw(M,pilot_rng), omega=draw(cfg.mpc.sampling.num_scenarios,certification_rng);
        auto samples_file = output(root/"samples.csv","batch,id,obstacle,k,x,y");
        for (const auto& label : {std::string("gamma"),std::string("omega")})
            for (const auto& s : label=="gamma"?gamma:omega)
                for (const auto& [id,tr] : s.trajectories)
                    for (size_t k=0;k<tr.steps.size();++k)
                        samples_file<<label<<','<<s.scenario_id<<','<<id<<','<<k<<','<<tr.steps[k].mean.x()<<','<<tr.steps[k].mean.y()<<'\n';
        auto summary=output(root/"four_way.csv","combination,prepared,rows,success,fallback,support,removed,iterations");
        for (int combination=0;combination<4;++combination) {
            auto copy=fork(*base,10,0,true);
            auto anchor=deterministic;
            bool prepared=true;
            if (combination) prepared=prepare_safe_horizon_anchors(anchor,
                combination==2?omega:gamma,cfg.combined_radius(),cfg.mpc.ego.num_discs,cfg.mpc.ego.length);
            const auto& batch=combination==3?gamma:omega;
            auto rows=compute_linearized_constraints(anchor,batch,cfg.mpc.ego.radius,
                cfg.obstacle_radius,cfg.mpc.constraints.safety_margin,cfg.mpc.ego.num_discs,cfg.mpc.ego.length);
            auto dump=output(root/("rows_"+std::to_string(combination)+".csv"),
                "k,disc,id,obstacle,anchor_x,anchor_y,obs_x,obs_y,relative_x,relative_y,a_x,a_y,b,normal_norm,anchor_slack,intended_side");
            for (const auto& row:rows) {
                const auto center=compute_ego_disc_positions(anchor[row.k],cfg.mpc.ego.num_discs,cfg.mpc.ego.length).at(row.disc_index);
                const auto it=std::find_if(batch.begin(),batch.end(),[&](const Scenario& s){return s.scenario_id==row.scenario_id;});
                const auto point=it->trajectories.at(row.obstacle_id).steps.at(row.k).mean;
                const auto delta=center-point;
                dump<<row.k<<','<<row.disc_index<<','<<row.scenario_id<<','<<row.obstacle_id<<','<<center.x()<<','<<center.y()<<','<<point.x()<<','<<point.y()<<','<<delta.x()<<','<<delta.y()<<','<<row.a.x()<<','<<row.a.y()<<','<<row.b<<','<<row.a.norm()<<','<<row.evaluate(center)<<','<<(row.evaluate(center)>=0)<<'\n';
            }
            std::cerr<<"[FOUR WAY BEGIN] combination="<<combination<<" prepared="<<prepared<<'\n';
            copy->scenarios_=batch;
            auto result=copy->solve_optimization_sqp(ego,c.goal,c.reference_velocity,rows,{}, {},c.path_progress,c.path_length);
            summary<<combination<<','<<prepared<<','<<rows.size()<<','<<result.success<<','<<result.used_fallback<<','<<result.support_size<<','<<copy->last_removed_scenario_ids_.size()<<','<<result.support_iterations_evaluated<<'\n';
            std::cerr<<"[FOUR WAY END] combination="<<combination<<'\n';
        }
    }
    static void export_transfer(const MPCController& source,const DecisionContext& context,
                                const fs::path& root,const fs::path& snapshots) {
        fs::create_directories(root);
        std::vector<std::pair<std::string,DecisionContext>> contexts;
        if(snapshots.empty())contexts.push_back({"0",context});
        else {
            std::ifstream input(snapshots);std::string line;std::getline(input,line);
            while(std::getline(input,line)) {
                std::istringstream fields(line);std::vector<double> values;std::string value;
                while(std::getline(fields,value,','))values.push_back(std::stod(value));
                if(values.size()!=12)throw std::runtime_error("invalid stored context snapshot");
                auto c=context;c.step=int(values[0]);
                c.ego=EgoState(values[1],values[2],values[3],values[4],values[5]);c.path_progress=values[5];
                c.obstacles={{int(values[6]),ObstacleState(values[7],values[8],values[9],values[10])}};
                c.reference_velocity=values[11];contexts.push_back({std::to_string(c.step),c});
            }
        }
        for(const auto& [name,c]:contexts) {
            std::ofstream out(root/("step_"+name+".json"));out<<std::setprecision(17);
            auto array=[&](const auto& values){out<<'[';bool comma=false;for(const auto x:values){if(comma)out<<',';comma=true;out<<x;}out<<']';};
            auto matrix=[&](const auto& A){out<<'[';for(int i=0;i<A.rows();++i){if(i)out<<',';out<<'[';for(int j=0;j<A.cols();++j){if(j)out<<',';out<<A(i,j);}out<<']';}out<<']';};
            auto nominal=fork(source,10,0,true);
            auto reference=nominal->generate_straight_line_trajectory(c.ego,c.goal,c.reference_velocity);
            out<<"{\"step\":"<<c.step<<",\"ego\":";array(std::vector<double>{c.ego.x,c.ego.y,c.ego.theta,c.ego.v});
            const auto& dynamics=source.config_.mpc.ego.dynamics;
            out<<",\"vmax\":"<<dynamics.max_velocity<<",\"amin\":"<<dynamics.min_acceleration<<",\"amax\":"<<dynamics.max_acceleration<<",\"omega\":"<<dynamics.max_omega
               <<",\"radius\":"<<source.config_.combined_radius()<<",\"length\":"<<source.config_.mpc.ego.length<<",\"H\":"<<source.config_.mpc.horizon<<",\"dt\":"<<source.config_.mpc.dt<<",\"obstacles\":[";
            bool comma=false;
            for(const auto& [id,state]:c.obstacles) {
                if(comma)out<<',';comma=true;
                const auto& history=source.mode_histories_.at(id);
                auto weights=compute_mode_weights(history,source.config_.mpc.sampling.mode_belief);
                auto counts=history.get_mode_counts();
                if(!snapshots.empty() && weights.size()==1)counts.begin()->second=c.step+1;
                auto dro_config=source.config_.dro.solver;
                dro_config.radius_calibration.confidence_beta/=c.obstacles.size();
                DRO scorer(dro_config);
                auto result=scorer.compute_worst_case_weights(weights,counts,state,history.available_modes,reference,
                    source.config_.mpc.horizon,source.config_.mpc.ego.radius,source.config_.obstacle_radius,
                    source.config_.mpc.constraints.safety_margin,-1,source.config_.mpc.ego.num_discs,source.config_.mpc.ego.length);
                out<<"{\"id\":"<<id<<",\"state\":";array(std::vector<double>{state.x,state.y,state.vx,state.vy});
                out<<",\"rho\":"<<result.rho_used<<",\"D\":[";
                for(size_t i=0;i<result.transport_cost_matrix.size();++i){if(i)out<<',';array(result.transport_cost_matrix[i]);}out<<"],\"modes\":[";
                bool first=true;
                for(const auto& [mode,model]:history.available_modes) {
                    if(!first)out<<',';first=false;
                    if(model.body_lateral_displacement!=0)throw std::runtime_error("nonlinear danger model unsupported");
                    out<<"{\"name\":\""<<mode<<"\",\"p\":"<<weights.at(mode)<<",\"count\":"<<counts[mode]<<",\"q_dro\":"<<result.worst_case_weights.at(mode)<<",\"risk\":"<<result.risk_per_mode.at(mode)<<",\"A\":";matrix(model.A);
                    out<<",\"b\":";array(std::vector<double>{model.b(0),model.b(1),model.b(2),model.b(3)});
                    out<<",\"G\":";matrix(model.G);out<<'}';
                }
                out<<"]}";
            }
            out<<"]}\n";
        }
    }
    static void enable_screen_controller(MPCController& controller) {
        controller.experimental_fixed_normals_=true;
        controller.experimental_collision_tightening_=0.0002;
        controller.config_.mpc.constraints.scenario_removal_budget=0;
        controller.config_.mpc.constraints.support_cap_n_bar=10;
        controller.config_.mpc.sampling.num_scenarios=controller.config_.compute_required_scenarios();
    }
    static std::unique_ptr<MPCController> fork(const MPCController& source, int h,
                                               unsigned seed, bool replay = false) {
        auto cfg = source.config_;
        cfg.mpc.constraints.scenario_removal_budget = 1;
        cfg.mpc.constraints.support_cap_n_bar = h - 1;
        cfg.mpc.sampling.num_scenarios = cfg.compute_required_scenarios();
        auto copy = std::make_unique<MPCController>(cfg);
        copy->ego_dynamics_ = source.ego_dynamics_;
        copy->default_modes_ = source.default_modes_;
        copy->mode_histories_ = source.mode_histories_;
        copy->obstacle_class_ids_ = source.obstacle_class_ids_;
        copy->scenarios_ = source.scenarios_;
        copy->dro_ = source.dro_;
        copy->last_dro_results_ = source.last_dro_results_;
        copy->reference_trajectory_ = source.reference_trajectory_;
        copy->solve_times_ = source.solve_times_;
        copy->iteration_count_ = source.iteration_count_;
        copy->reference_path_ = source.reference_path_;
        copy->custom_per_obstacle_weights_ = source.custom_per_obstacle_weights_;
        copy->last_feasible_controls_ = source.last_feasible_controls_;
        copy->has_feasible_backup_ = source.has_feasible_backup_;
        copy->last_removed_scenario_ids_ = source.last_removed_scenario_ids_;
        copy->rng_ = replay ? source.rng_ : std::mt19937(seed);
        return copy;
    }

    static MPCResult solve(MPCController& copy, const DecisionContext& c) {
        return copy.solve(c.ego, c.obstacles, c.goal, c.reference_velocity,
                          c.path_progress, c.path_length);
    }

    static std::string rng_state(const MPCController& source) {
        std::ostringstream s; s << source.rng_; return s.str();
    }

    static void features(std::ostream& out, const MPCController& source,
                         const DecisionContext& c, unsigned seed, int previous_support) {
        auto nominal = fork(source, 10, 0, true);
        nominal->initialize_reference_trajectory(c.ego, c.goal, c.reference_velocity);
        double entropy = 0, covariance_scale = 0, dispersion = 0;
        double minimum = std::numeric_limits<double>::infinity();
        double observed = minimum;
        int modes = 0;
        for (const auto& [id, state] : c.obstacles) {
            observed = std::min(observed, (state.position() - c.ego.position()).norm());
            const auto& history = source.mode_histories_.at(id);
            auto weights = compute_mode_weights(history, source.config_.mpc.sampling.mode_belief);
            modes += weights.size();
            Eigen::Vector2d mean = Eigen::Vector2d::Zero();
            double second = 0;
            for (const auto& [mode, p] : weights) {
                if (p > 0) entropy -= p * std::log(p);
                const auto& model = history.available_modes.at(mode);
                auto x = state;
                Eigen::Matrix4d covariance = Eigen::Matrix4d::Zero();
                for (int k = 1; k <= source.config_.mpc.horizon; ++k) {
                    x = model.propagate(x);
                    // Exact held-mode linear Gaussian covariance; the selected
                    // profiles have no nonlinear body-displacement modes.
                    if (model.body_lateral_displacement != 0)
                        throw std::runtime_error("unsupported nonlinear modal covariance");
                    covariance = model.A * covariance * model.A.transpose() + model.G * model.G.transpose();
                    minimum = std::min(minimum, (x.position() - nominal->reference_trajectory_[k].position()).norm());
                }
                covariance_scale += p * covariance.topLeftCorner<2,2>().trace();
                mean += p * x.position();
                second += p * x.position().squaredNorm();
            }
            dispersion += std::max(0.0, second - mean.squaredNorm());
        }
        out << seed << ',' << c.step << ',' << c.ego.x << ',' << c.ego.y << ','
            << c.ego.v << ',' << c.obstacles.size() << ',' << modes << ',' << observed << ','
            << entropy << ',' << covariance_scale << ',' << dispersion << ',' << minimum << ','
            << previous_support << ',' << c.path_progress << '\n';
    }

    static double clearance(const MPCController& copy, const MPCResult& r, bool retained = false) {
        double minimum = std::numeric_limits<double>::infinity();
        for (const auto& scenario : copy.scenarios_) {
            if (retained && copy.last_removed_scenario_ids_.count(scenario.scenario_id)) continue;
            for (const auto& [id, trajectory] : scenario.trajectories)
                for (size_t k = 1; k < r.ego_trajectory.size() && k < trajectory.steps.size(); ++k)
                    for (const auto& disc : compute_ego_disc_positions(r.ego_trajectory[k],
                            copy.config_.mpc.ego.num_discs, copy.config_.mpc.ego.length))
                        minimum = std::min(minimum, (disc - trajectory.steps[k].mean).norm() - copy.config_.combined_radius());
        }
        return minimum;
    }
};
}

using namespace dro_mpc;
std::ofstream output(const fs::path& p, const std::string& header) {
    std::ofstream out(p);
    if (!out) throw std::runtime_error("cannot write " + p.string());
    out << std::setprecision(17) << header << '\n';
    return out;
}

int main(int argc, char** argv) {
    try {
        if(argc==7 && std::string(argv[1])=="--transfer-export") {
            auto cfg=yaml_config::load_experiment_config(argv[2],true);
            cfg.rollout.rollout_steps=1;cfg.artifacts.output_directory=(fs::path(argv[3])/"unused").string();cfg.normalize();
            cfg.rollout.decision_callback=[&](const DecisionContext& c,const MPCController& source){
                CounterfactualProbe::export_transfer(source,c,argv[3],std::string(argv[6])=="-"?fs::path{}:fs::path(argv[6]));
                throw std::runtime_error("TRANSFER_EXPORT_COMPLETE");
            };
            try{run_experiment_rollout(cfg,std::stoul(argv[4]));}catch(const std::runtime_error& e){if(std::string(e.what())!="TRANSFER_EXPORT_COMPLETE")throw;}
            return 0;
        }
        if (argc==7 && (std::string(argv[1])=="--oracle-screen" || std::string(argv[1])=="--coreset-plan-export")) {
            auto cfg=yaml_config::load_experiment_config(argv[2],true);
            const fs::path root=argv[3];fs::create_directories(root);
            const unsigned seed=std::stoul(argv[4]);const int repetitions=std::stoi(argv[5]);
            const bool snapshots=std::stoi(argv[6])!=0;
            const bool plan_only=std::string(argv[1])=="--coreset-plan-export";
            cfg.mpc.constraints.scenario_removal_budget=0;cfg.mpc.constraints.support_cap_n_bar=10;
            cfg.mpc.sampling.automatically_compute_sample_size=true;
            cfg.mpc.sampling.deterministic_anchor_selection=true;cfg.mpc.sampling.anchor_pilot_samples=0;
            cfg.mpc.sampling.anchor_inflation_kappa=3;cfg.rollout.rollout_steps=snapshots?41:1;
            cfg.artifacts.output_directory=(root/"encounter").string();
            cfg.artifacts.write_visualization_gif=false;cfg.artifacts.write_rviz_replay_bundle=false;
            cfg.normalize();
            cfg.rollout.step_callback=[](int,int,ObstacleSim&,MPCController& source,std::mt19937&){CounterfactualProbe::enable_screen_controller(source);};
            auto contexts=output(root/"contexts.csv","seed,step,ego_x,ego_y,ego_speed,obstacles,modes,observed_distance,mode_entropy,terminal_within_mode_covariance_trace,terminal_modal_dispersion,min_nominal_reference_distance,previous_support,path_progress");
            cfg.rollout.decision_callback=[&](const DecisionContext& c,const MPCController& source){
                if (snapshots && c.step!=0 && c.step!=10 && c.step!=20 && c.step!=30 && c.step!=40) return;
                CounterfactualProbe::features(contexts,source,c,seed,-1);contexts.flush();
                for (int h:{3,4,5,6,7,8,10}) {
                    if(plan_only && h!=10)continue;
                    const auto folder=root/("step_"+std::to_string(c.step))/("h_"+std::to_string(h));fs::create_directories(folder);
                    CounterfactualProbe::shared_slack(source,c,folder,seed+unsigned(c.step)*100003u,repetitions,true,0.0002,h);
                }
                if (!snapshots) throw std::runtime_error("ORACLE_CONTEXT_COMPLETE");
            };
            try {run_experiment_rollout(cfg,seed);} catch(const std::runtime_error& e){if(std::string(e.what())!="ORACLE_CONTEXT_COMPLETE")throw;}
            return 0;
        }
        if (argc==2 && std::string(argv[1])=="--slack-feasibility-self-test") {
            CounterfactualProbe::slack_fixture_tests();return 0;
        }
        if ((argc == 6 || (argc == 7 && std::string(argv[1])=="--anchor-fixed-tight-slack")) && (std::string(argv[1]) == "--anchor-reliability" || std::string(argv[1]) == "--anchor-conflicts" || std::string(argv[1]) == "--anchor-forced-removal" || std::string(argv[1]) == "--anchor-shared-slack" || std::string(argv[1]) == "--anchor-fixed-normals" || std::string(argv[1]) == "--anchor-fixed-slack" || std::string(argv[1]) == "--anchor-fixed-tight-slack")) {
            auto cfg=yaml_config::load_experiment_config(argv[2],true);
            const fs::path root=argv[3];fs::create_directories(root);
            cfg.mpc.constraints.support_cap_n_bar=9;
            cfg.mpc.constraints.scenario_removal_budget=1;
            cfg.mpc.sampling.automatically_compute_sample_size=true;
            cfg.mpc.sampling.deterministic_anchor_selection=true;
            cfg.mpc.sampling.anchor_inflation_kappa=3;
            cfg.mpc.sampling.anchor_pilot_samples=0;
            cfg.rollout.rollout_steps=1;
            cfg.artifacts.output_directory=(root/"unused").string();
            cfg.artifacts.write_visualization_gif=false;
            cfg.artifacts.write_rviz_replay_bundle=false;
            cfg.normalize();
            bool reliability_done=false;
            const unsigned seed=std::stoul(argv[4]);
            const bool forced_removal=std::string(argv[1])=="--anchor-forced-removal";
            const int repetitions=forced_removal?1:std::stoi(argv[5]);
            if (repetitions<1 || cfg.dro.enabled || cfg.mpc.sampling.markov_jump_system)
                throw std::runtime_error("replication requires positive repetitions and ordinary held-mode SH");
            cfg.rollout.decision_callback=[&](const DecisionContext& c,const MPCController& source) {
                reliability_done=true;
                if (forced_removal) CounterfactualProbe::removal_replay(source,c,root,argv[5],seed);
                else if (std::string(argv[1])=="--anchor-shared-slack") CounterfactualProbe::shared_slack(source,c,root,seed,repetitions);
                else if (std::string(argv[1])=="--anchor-fixed-tight-slack") CounterfactualProbe::shared_slack(source,c,root,seed,repetitions,true,std::stod(argv[6]));
                else if (std::string(argv[1])=="--anchor-fixed-slack") CounterfactualProbe::shared_slack(source,c,root,seed,repetitions,true);
                else if (std::string(argv[1])=="--anchor-fixed-normals") CounterfactualProbe::fixed_normals(source,c,root,seed,repetitions);
                else CounterfactualProbe::reliability(source,c,root,seed,repetitions,std::string(argv[1])=="--anchor-conflicts");
                throw std::runtime_error("FROZEN_REPLICATION_COMPLETE");
            };
            try {run_experiment_rollout(cfg,seed);} catch (const std::runtime_error& e) {
                if (std::string(e.what())!="FROZEN_REPLICATION_COMPLETE") throw;
            }
            return reliability_done?0:1;
        }
        if (argc == 6 && std::string(argv[1]) == "--anchor-four-way") {
            auto cfg=yaml_config::load_experiment_config(argv[2],true);
            const fs::path root=argv[3]; fs::create_directories(root);
            cfg.mpc.constraints.support_cap_n_bar=9;
            cfg.mpc.constraints.scenario_removal_budget=1;
            cfg.mpc.sampling.automatically_compute_sample_size=true;
            cfg.mpc.sampling.anchor_pilot_samples=0;
            cfg.rollout.rollout_steps=1;
            cfg.artifacts.output_directory=(root/"baseline").string();
            cfg.artifacts.write_visualization_gif=false;
            cfg.artifacts.write_rviz_replay_bundle=false;
            cfg.normalize();
            if (cfg.dro.enabled || cfg.mpc.sampling.markov_jump_system)
                throw std::runtime_error("four-way diagnostic requires ordinary held-mode SH");
            cfg.rollout.decision_callback=[&](const DecisionContext& c,const MPCController& source) {
                CounterfactualProbe::diagnose(source,c,root,std::stoi(argv[5]));
            };
            run_experiment_rollout(cfg,std::stoul(argv[4]));
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--curve") {
            auto cfg = yaml_config::load_experiment_config("configs/safe_horizon_paper.yaml", true).to_scenario_mpc_config();
            auto out = output(argv[2], "h,S,epsilon,beta");
            for (int h = 0; h <= 10; ++h)
                out << h << ',' << cfg.compute_required_scenarios(h, 0) << ','
                    << cfg.epsilon() << ',' << cfg.mpc.sampling.chance_of_certificate_violation << '\n';
            return 0;
        }
        if (argc != 7 && !(argc == 8 && std::string(argv[7]) == "--fixed-h10")) throw std::runtime_error("usage: week2_budget_response config output seed stride repeats max_steps; or --curve file");
        const unsigned seed = std::stoul(argv[3]);
        const int stride = std::stoi(argv[4]), repeats = std::stoi(argv[5]), steps = std::stoi(argv[6]);
        if (stride < 1 || repeats < 1 || steps < 1) throw std::runtime_error("positive stride/repeats/steps required");
        fs::path root = argv[2]; fs::create_directories(root);
        auto cfg = yaml_config::load_experiment_config(argv[1], true);
        if (cfg.mpc.type != MPCType::SH_MPCC || !cfg.mpc.uses_safe_horizon() || cfg.dro.enabled ||
            cfg.mpc.sampling.markov_jump_system || cfg.mpc.nominal_resampling_baseline || cfg.mpc.wdro_stratified_sampling)
            throw std::runtime_error("experiment requires base held-mode SH_MPCC");
        cfg.mpc.constraints.support_cap_n_bar = 9;
        cfg.mpc.constraints.scenario_removal_budget = 1;
        cfg.mpc.sampling.automatically_compute_sample_size = true;
        cfg.rollout.rollout_steps = steps;
        cfg.artifacts.output_directory = (root / "baseline").string();
        cfg.artifacts.write_visualization_gif = false;
        cfg.artifacts.write_rviz_replay_bundle = false;
        cfg.normalize();
        auto contexts = output(root / "contexts.csv", "seed,step,ego_x,ego_y,ego_speed,obstacles,modes,observed_distance,mode_entropy,terminal_within_mode_covariance_trace,terminal_modal_dispersion,min_nominal_reference_distance,previous_support,path_progress");
        auto results = output(root / "budget_responses.csv", "seed,step,replicate,h,S,sampling_seed,support_evaluated,support,cap_ok,success,certified,fallback,certificate,solve_ms,cost,sampled_clearance,retained_sampled_clearance,predicted_progress,a,omega,cap_terminated,returned_iteration,removed_count");
        auto checks = output(root / "clone_checks.csv", "seed,step,rng_unchanged,baseline_replay_matches,max_control_difference");
        auto baseline = output(root / "baseline_decisions.csv", "seed,step,success,certified,support,solve_ms,path_progress");
        std::optional<MPCResult> replay;
        int previous_support = -1;
        bool unchanged = true;
        auto evaluate = [&](const DecisionContext& c, const MPCController& source) {
            const auto rng_before = CounterfactualProbe::rng_state(source);
            CounterfactualProbe::features(contexts, source, c, seed, previous_support);
            auto reference = CounterfactualProbe::fork(source, 10, 0, true);
            replay = CounterfactualProbe::solve(*reference, c);
            for (int replicate = 0; replicate < repeats; ++replicate)
                for (int h : (argc == 8 ? std::vector<int>{10} : std::vector<int>{1,2,3,4,5,6,7,10})) {
                    // Independent deterministic streams, all selected before sampling.
                    std::seed_seq stream{0x534832u, seed, unsigned(c.step), unsigned(replicate), unsigned(h)};
                    unsigned draw_seed; stream.generate(&draw_seed, &draw_seed + 1);
                    auto copy = CounterfactualProbe::fork(source, h, draw_seed);
                    auto r = CounterfactualProbe::solve(*copy, c);
                    const bool evaluated = r.support_cap_status != SupportCapStatus::NOT_EVALUATED;
                    const bool certified = r.certificate_status == SafeHorizonCertificateStatus::CERTIFIED;
                    if (certified && (!r.success || r.used_fallback || !evaluated || r.support_size > h ||
                        !r.support_iterations_feasible || r.sampled_scenarios != copy->config().compute_required_scenarios()))
                        throw std::runtime_error("certificate accounting invariant failed");
                    double progress = r.ego_trajectory.empty() ? 0 : r.ego_trajectory.back().s - c.path_progress;
                    const EgoInput u = r.control_inputs.empty() ? EgoInput(0,0) : r.control_inputs.front();
                    results << seed << ',' << c.step << ',' << replicate << ',' << h << ',' << r.sampled_scenarios << ','
                        << draw_seed << ',' << evaluated << ',' << r.support_size << ',' << (evaluated && r.support_size <= h) << ','
                        << r.success << ',' << certified << ',' << r.used_fallback << ',' << safe_horizon_certificate_status_name(r.certificate_status) << ','
                        << 1000*r.solve_time << ',' << r.cost << ',' << CounterfactualProbe::clearance(*copy,r) << ','
                        << CounterfactualProbe::clearance(*copy,r,true) << ','
                        << progress << ',' << u.a << ',' << u.omega << ',' << r.support_cap_terminated << ','
                        << r.returned_sqp_iteration << ',' << copy->last_removed_scenario_ids().size() << '\n';
                }
            unchanged = rng_before == CounterfactualProbe::rng_state(source);
            if (!unchanged) throw std::runtime_error("probe changed live RNG");
            results.flush(); contexts.flush();
        };
        std::unique_ptr<MPCController> terminal_snapshot;
        cfg.rollout.decision_callback = [&](const DecisionContext& c, const MPCController& source) {
            replay.reset(); terminal_snapshot.reset();
            if (c.step % stride == 0) evaluate(c, source);
            else terminal_snapshot = CounterfactualProbe::fork(source, 10, 0, true);
        };
        cfg.rollout.result_callback = [&](const DecisionContext& c, const MPCResult& r, const MPCController&) {
            // Keep terminal rejected contexts even when they fall between the
            // scheduled snapshots; never substitute post-decision state.
            if (!r.success && !replay) evaluate(c, *terminal_snapshot);
            baseline << seed << ',' << c.step << ',' << r.success << ','
                << (r.certificate_status == SafeHorizonCertificateStatus::CERTIFIED) << ',' << r.support_size << ','
                << 1000*r.solve_time << ',' << c.path_progress << '\n';
            if (replay) {
                double difference = 0;
                bool match = replay->success == r.success && replay->certificate_status == r.certificate_status &&
                    replay->support_scenarios == r.support_scenarios && replay->control_inputs.size() == r.control_inputs.size();
                if (match) for (size_t k=0;k<r.control_inputs.size();++k)
                    difference = std::max(difference,(replay->control_inputs[k].to_array()-r.control_inputs[k].to_array()).norm());
                match = match && difference < 1e-10;
                checks << seed << ',' << c.step << ',' << unchanged << ',' << match << ',' << difference << '\n';
                if (!match) throw std::runtime_error("baseline clone replay mismatch");
            }
            previous_support = r.support_cap_status == SupportCapStatus::NOT_EVALUATED ? -1 : r.support_size;
        };
        auto r = run_experiment_rollout(cfg, seed);
        auto summary = output(root / "summary.csv", "seed,steps,collision,completed,termination,min_clearance");
        summary << seed << ',' << r.total_steps << ',' << r.collision << ',' << r.completed_path << ',' << r.termination_reason << ',' << r.min_clearance << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
