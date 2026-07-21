// ablation_confound_check: did my "base" arm (enable_dro=false, ablation defaulted to
// DRO_FULL) silently run WDRO via the legacy ablation guard? Compare at offset 0:
//   my-base   : enable_dro=false, ablation=DRO_FULL (default, NOT set)  <- what I used as base
//   true-base : enable_dro=false, ablation=NO_INJECTION (DRO truly off)
//   WDRO      : enable_dro=true,  QSTAR_SAMPLE
#include "experiment_harness.hpp"
#include "reference_path.hpp"
#include <cstdio>
#include <cmath>
using namespace scenario_mpc;

static ObstacleState place_oncoming(const ReferencePath& p, double off) {
    double s=0.60*p.total_length(); PathPoint pp=p.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading),std::cos(pp.heading)), t(std::cos(pp.heading),std::sin(pp.heading));
    Eigen::Vector2d q=pp.position+off*n; return ObstacleState(q.x(),q.y(),-1.0*t.x(),-1.0*t.y());
}
static ExperimentConfig mk(const ReferencePath& p, const ObstacleState& o){
    ExperimentConfig c; c.horizon=DEFAULT_HORIZON; c.num_scenarios=DEFAULT_BASE_SCENARIOS; c.switch_prob=0.2;
    c.rollout_steps=DEFAULT_ROLLOUT_STEPS; c.obs_modes={"constant_velocity","turn_left","turn_right","decelerating"};
    c.rare_mode="lane_change_left"; c.rare_switch_prob=0.1; c.num_discs=1; c.vehicle_length=1.5;
    c.safe_horizon_enabled=true; c.safe_horizon_min=3; c.path_completion_termination=true; c.path_completion_fraction=0.95;
    c.weight_type=WeightType::FREQUENCY; c.num_obstacles=1; c.enable_contouring_constraints=true; c.road_width=4.0;
    c.custom_ref_path=p; c.custom_initial_ego=EgoState(0,0,0,1.5); c.initial_obstacle_states={o};
    return c;
}
static double rate(ExperimentConfig c, int N){ int k=0; for(int i=0;i<N;++i) k+=run_experiment_rollout(c,3000000u+unsigned(i)).collision?1:0; return double(k)/N; }
int main(){
    const int N=200;
    ReferencePath path=ReferencePath::create_s_curve(S_CURVE_LENGTH,S_CURVE_AMPLITUDE,S_CURVE_POINTS);
    ExperimentConfig b=mk(path,place_oncoming(path,0.0));

    ExperimentConfig my_base=b; my_base.enable_dro=false; my_base.injection_mode=InjectionMode::NONE; /* ablation defaults DRO_FULL */
    ExperimentConfig true_base=b; true_base.enable_dro=false; true_base.injection_mode=InjectionMode::NONE; true_base.ablation=AblationVariant::NO_INJECTION;
    ExperimentConfig wdro=b; wdro.enable_dro=true; wdro.injection_mode=InjectionMode::QSTAR_SAMPLE;

    std::printf("### Ablation confound check (offset 0, road ON, S=40, N=%d) ###\n", N);
    double mb=rate(my_base,N);   std::printf("  my-base   (enable_dro=false, ablation=DRO_FULL default) coll=%.3f\n", mb);
    double tb=rate(true_base,N); std::printf("  true-base (enable_dro=false, ablation=NO_INJECTION)      coll=%.3f\n", tb);
    double w =rate(wdro,N);      std::printf("  WDRO      (enable_dro=true,  QSTAR_SAMPLE)               coll=%.3f\n", w);
    std::printf("\n  my-base vs WDRO   : %+.1fpp  (if ~0 -> my base WAS wdro; confound CONFIRMED)\n", 100.0*(mb-w));
    std::printf("  true-base vs WDRO : %+.1fpp  (the REAL WDRO effect)\n", 100.0*(tb-w));
    return 0;
}
