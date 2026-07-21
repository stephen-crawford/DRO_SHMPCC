// ablation_confound_check: legacy ablation guard removed — BASE + NONE injection is the true base.
// Compare at offset 0:
//   base      : DROConfiguration::BASE, injection NONE
//   WDRO      : DROConfiguration::DRO, QSTAR_SAMPLE
#include "experiment_harness.hpp"
#include "reference_path.hpp"
#include <cstdio>
#include <cmath>
using namespace dro_mpc;

static ObstacleState place_oncoming(const ReferencePath& p, double off) {
    double s=0.60*p.total_length(); PathPoint pp=p.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading),std::cos(pp.heading)), t(std::cos(pp.heading),std::sin(pp.heading));
    Eigen::Vector2d q=pp.position+off*n; return ObstacleState(q.x(),q.y(),-1.0*t.x(),-1.0*t.y());
}
static ExperimentConfig mk(const ReferencePath& p, const ObstacleState& o){
    ExperimentConfig c;
    c.mpc.horizon=DEFAULT_HORIZON; c.mpc.sampling.num_scenarios=DEFAULT_BASE_SCENARIOS; c.obstacles.switch_prob=0.2;
    c.rollout.rollout_steps=DEFAULT_ROLLOUT_STEPS; c.obstacles.obs_modes={"constant_velocity","turn_left","turn_right","decelerating"};
    c.obstacles.rare_mode="lane_change_left"; c.obstacles.rare_switch_prob=0.1; c.mpc.ego.num_discs=1; c.mpc.ego.length=1.5;
    c.mpc.safe_horizon_enabled=true; c.mpc.constraints.safe_horizon_min=3; c.environment.path_completion_termination=true; c.environment.path_completion_fraction=0.95;
    c.mpc.sampling.weight_type=WeightType::FREQUENCY; c.obstacles.num_obstacles=1; c.mpc.enable_contouring_constraints=true; c.mpc.constraints.road_width=4.0;
    c.environment.custom_ref_path=p; c.environment.custom_initial_ego=EgoState(0,0,0,1.5); c.obstacles.initial_obstacle_states={o};
    return c;
}
static double rate(ExperimentConfig c, int N){ int k=0; for(int i=0;i<N;++i) k+=run_experiment_rollout(c,3000000u+unsigned(i)).collision?1:0; return double(k)/N; }
int main(){
    const int N=200;
    ReferencePath path=ReferencePath::create_s_curve(S_CURVE_LENGTH,S_CURVE_AMPLITUDE,S_CURVE_POINTS);
    ExperimentConfig b=mk(path,place_oncoming(path,0.0));

    ExperimentConfig base_cfg=b; base_cfg.dro.enabled = false; base_cfg.dro.injection_mode=InjectionMode::NONE;
    ExperimentConfig wdro=b; wdro.dro.enabled = true; wdro.dro.injection_mode=InjectionMode::QSTAR_SAMPLE;

    std::printf("### Ablation confound check (offset 0, road ON, S=40, N=%d) ###\n", N);
    double tb=rate(base_cfg,N); std::printf("  base      (DRO off, injection NONE)                   coll=%.3f\n", tb);
    double w =rate(wdro,N);     std::printf("  WDRO      (DRO on,  QSTAR_SAMPLE)                      coll=%.3f\n", w);
    std::printf("\n  base vs WDRO : %+.1fpp  (the WDRO effect)\n", 100.0*(tb-w));
    return 0;
}
