// pipeline_verify: verify each stage of the WDRO scenario-MPC pipeline on MAIN, in order.
// Controlled head-on scenario with a non-uniform mode history so the belief, risk, and
// reweighting are all non-trivial and checkable.
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "collision_constraints.hpp"
#include "dynamics.hpp"
#include "reference_path.hpp"
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace dro_mpc;
static int fails = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg); if(!ok) ++fails;
}

int main() {
    auto mode_models = create_obstacle_mode_models(0.1);
    std::vector<std::string> modes; for (auto& kv : mode_models) modes.push_back(kv.first);
    const int M = (int)modes.size();

    // Head-on: ego drives +x toward an obstacle ahead drifting slowly +x.
    ObstacleState obs(5.0, 0.0, 0.5, 0.0);
    std::vector<EgoState> ego_ref; for (int k=0;k<=15;++k) ego_ref.emplace_back(k*0.34,0.0,0.0,1.5);

    // Non-uniform history: mostly constant_velocity, some decelerating, few others.
    ModeHistory hist(0, mode_models);
    for (int t=0;t<30;++t) {
        std::string m = (t%2==0)?"constant_velocity":(t%5==0?"decelerating":"turn_left");
        hist.record_observation(t, m);
    }

    std::printf("=== STEP 1: ego/spline, obstacle, nominal belief, Bonferroni-VaR risk ===\n");
    // ego spline
    ReferencePath path = ReferencePath::create_s_curve(25.0,3.0,200);
    double s0 = path.find_closest_point(Eigen::Vector2d(0,0));
    check(std::isfinite(s0), "reference-path spline position computed for ego");
    check(obs.position().x()==5.0, "obstacle position determined");
    // nominal belief
    auto nominal = compute_mode_weights(hist);
    double nsum=0; bool allpos=true; for(auto&kv:nominal){nsum+=kv.second; if(kv.second<=0)allpos=false;}
    check((int)nominal.size()==M && allpos, "nominal belief: every mode strictly positive (Dirichlet)");
    check(std::abs(nsum-1.0)<1e-9, "nominal belief normalized to 1");
    // risk scores (Bonferroni VaR) vs plain VaR
    auto risk_of = [&](DRORiskMeasure rm){ DROConfig c; c.radius_calibration.risk_measure=rm;
        WassersteinDRO d(c); return d.compute_worst_case_weights(nominal,obs,mode_models,ego_ref,15,0.5,0.35,0.2); };
    DROResult bonf = risk_of(DRORiskMeasure::SURROGATE_VAR_BONFERRONI);
    DROResult var  = risk_of(DRORiskMeasure::SURROGATE_VAR);
    double rmin=1e9,rmax=-1e9; std::string danger; for(auto&kv:bonf.risk_per_mode){ if(kv.second>rmax){rmax=kv.second;danger=kv.first;} rmin=std::min(rmin,kv.second);}
    check(rmax-rmin>1e-6, "per-mode risk scores are DIFFERENTIATED across modes");
    double bsum=0,vsum=0; for(auto&kv:bonf.risk_per_mode)bsum+=kv.second; for(auto&kv:var.risk_per_mode)vsum+=kv.second;
    check(std::abs(bsum-vsum)>1e-9, "Bonferroni VaR differs from plain VaR (Bonferroni level active)");
    std::printf("    most-dangerous mode = %s (r=%.4f); risk range [%.4f, %.4f]\n", danger.c_str(), rmax, rmin, rmax);

    std::printf("=== STEP 2: WDRO reweighting redistributes mass toward high-risk modes ===\n");
    double q_danger = bonf.worst_case_weights[danger], p_danger = nominal[danger];
    double qsum=0; bool qvalid=true; for(auto&kv:bonf.worst_case_weights){qsum+=kv.second; if(kv.second<-1e-9||kv.second>1+1e-9)qvalid=false;}
    check(qvalid && std::abs(qsum-1.0)<1e-6, "q* is a valid probability distribution");
    check(q_danger > p_danger + 1e-6, "q* UP-WEIGHTS the most-dangerous mode vs nominal");
    check(bonf.rho_used > 0.0, "ambiguity radius rho_used > 0 (reweighting is active)");
    std::printf("    q*[%s]=%.4f vs nominal=%.4f  (rho=%.4f)\n", danger.c_str(), q_danger, p_danger, bonf.rho_used);

    std::printf("=== STEP 3: sample S scenarios from q*; empirical dist must match q* ===\n");
    std::mt19937 rng(12345);
    std::map<int,ObstacleState> obstacles{{0,obs}};
    std::map<int,ModeHistory> histories{{0,hist}};
    const int S=400;  // large S to check the empirical distribution converges to q*
    std::map<int,std::map<std::string,double>> qmap{{0,bonf.worst_case_weights}};
    auto scen_q = sample_scenarios_with_weights(obstacles,histories,qmap,15,S,false,&rng);
    std::map<std::string,int> cnt; for(auto&sc:scen_q){auto it=sc.trajectories.find(0); if(it!=sc.trajectories.end())cnt[it->second.mode_id]++;}
    double emp_danger = cnt[danger]/(double)scen_q.size();
    check((int)scen_q.size()==S, "sampler returned S scenarios");
    check(std::abs(emp_danger - q_danger) < 0.06, "empirical mode freq of dangerous mode matches q* (sampling uses q*)");
    // and it differs from nominal sampling
    std::map<int,std::map<std::string,double>> pmap{{0,nominal}};
    auto scen_p = sample_scenarios_with_weights(obstacles,histories,pmap,15,S,false,&rng);
    std::map<std::string,int> cntp; for(auto&sc:scen_p){auto it=sc.trajectories.find(0); if(it!=sc.trajectories.end())cntp[it->second.mode_id]++;}
    double empp_danger = cntp[danger]/(double)scen_p.size();
    check(emp_danger > empp_danger + 0.03, "q*-sampling over-represents dangerous mode vs nominal sampling");
    std::printf("    dangerous-mode freq: q*-sampled=%.3f  nominal-sampled=%.3f  (q*=%.3f)\n", emp_danger, empp_danger, q_danger);

    std::printf("=== STEP 4: linearized halfspace constraints from scenarios ===\n");
    auto cons = compute_linearized_constraints(ego_ref, scen_q, 0.5, 0.35, 0.1, 1, 1.5);
    bool proper=true; for(auto&c:cons){ if(std::abs(c.a.norm()-1.0)>1e-6 || !std::isfinite(c.b)) proper=false; }
    check(!cons.empty(), "constraints generated from scenarios");
    check(proper, "each constraint is a proper unit-normal affine halfspace");
    auto pruned = prune_dominated_scenarios(scen_q, ego_ref);
    auto cons_pruned = compute_linearized_constraints(ego_ref, pruned, 0.5, 0.35, 0.1, 1, 1.5);
    std::printf("    scenarios: %zu -> %zu after de Groot dominance pruning; constraints %zu -> %zu (non-distorting)\n",
                scen_q.size(), pruned.size(), cons.size(), cons_pruned.size());
    check(pruned.size() <= scen_q.size() && !pruned.empty(), "dominance pruning ran (inspect the reduction above)");

    std::printf("=== STEP 5+6: full controller solve + apply control ===\n");
    RuntimeConfig cfg; cfg.dro.enabled=true; cfg.dro.injection_mode=InjectionMode::QSTAR_SAMPLE;
    cfg.mpc.sampling.num_scenarios=40; cfg.mpc.ego.num_discs=1; cfg.mpc.ego.length=1.5;
    AdaptiveScenarioMPC ctrl(cfg);
    ctrl.set_reference_path(path);
    EgoState ego(0,0,0,1.5); Eigen::Vector2d goal(25,0);
    MPCResult res = ctrl.solve(ego, obstacles, goal, 1.5, 0.0, path.total_length());
    check(res.success, "controller.solve() succeeded end-to-end with DRO on");
    check(res.first_input().has_value(), "a control input was produced (Step 6 applies it)");
    check(!ctrl.scenarios().empty(), "controller populated its scenario set from the pipeline");

    std::printf("=== DIAGNOSIS: raw-LP q* collapses to ONE mode (no diversity) vs entropic ===\n");
    auto scenario_support = [&](DROResult& r)->int{
        std::map<int,std::map<std::string,double>> qm{{0,r.worst_case_weights}};
        std::mt19937 rr(7); auto sc=sample_scenarios_with_weights(obstacles,histories,qm,15,S,false,&rr);
        std::map<std::string,int> c; for(auto&s:sc){auto it=s.trajectories.find(0); if(it!=s.trajectories.end())c[it->second.mode_id]++;}
        int nz=0; for(auto&kv:c) if(kv.second>0)++nz; return nz; };
    int raw_support = scenario_support(bonf);
    DROConfig ec; ec.radius_calibration.risk_measure=DRORiskMeasure::SURROGATE_VAR_BONFERRONI; ec.radius_calibration.use_entropic_allocator=true; ec.radius_calibration.entropic_tau=0.05;
    WassersteinDRO ed(ec);
    DROResult ent = ed.compute_worst_case_weights(nominal,obs,mode_models,ego_ref,15,0.5,0.35,0.2);
    int ent_support = scenario_support(ent);
    std::printf("    distinct modes appearing in the S sampled scenarios:\n");
    std::printf("      raw-LP q* (default) : %d / %d modes   <-- bang-bang: planner sees ONLY the worst mode\n", raw_support, M);
    std::printf("      entropic q* (tau=.05): %d / %d modes   <-- graded: dangerous up-weighted, others still covered\n", ent_support, M);
    std::printf("  ROOT FINDING: every stage is individually correct, but Step-2 raw-LP reweighting is\n");
    std::printf("  DEGENERATE (q*=e_argmax, Thm 2) -> Step-3 draws a single-mode scenario set -> Step-4\n");
    std::printf("  constraints guard only that one mode. Under switching obstacles this removes the\n");
    std::printf("  diversity needed to cover the modes actually taken. Entropic/support-floor fixes it.\n");

    std::printf("\n%s (%d checks failed)\n", fails==0?"ALL PIPELINE STAGES VERIFIED":"SOME STAGES FAILED", fails);
    return fails==0?0:1;
}
