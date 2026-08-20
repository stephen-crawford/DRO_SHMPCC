// injection_test: SLMPC's benefit comes from INJECTING extreme/high-risk scenarios as hard
// constraints ("Extremes"), not from OT sampling reshaping (exp_h1: Base_vs_OT null). My
// recompute only tested QSTAR_SAMPLE (= OT-only). Here we test the injection arms at
// offset 0 (road ON, S=40, N=200): base | OT-sampling | DRO-inject K=1/3 | TopRisk-inject.
#include "experiment_harness.hpp"
#include "reference_path.hpp"
#include <cstdio>
#include <cmath>

using namespace dro_mpc;

static ObstacleState place_oncoming(const ReferencePath& path, double offset) {
    double s = 0.60 * path.total_length();
    PathPoint pp = path.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
    Eigen::Vector2d pos = pp.position + offset * n;
    return ObstacleState(pos.x(), pos.y(), -1.0 * t.x(), -1.0 * t.y());
}
static ExperimentConfig mk(const ReferencePath& path, const ObstacleState& obs) {
    ExperimentConfig c;
    c.mpc.horizon = DEFAULT_HORIZON; c.mpc.sampling.num_scenarios = DEFAULT_BASE_SCENARIOS;
    c.obstacles.switch_prob = 0.2; c.rollout.rollout_steps = DEFAULT_ROLLOUT_STEPS;
    c.obstacles.obs_modes = {"constant_velocity","turn_left","turn_right","decelerating"};
    c.obstacles.rare_mode = "lane_change_left"; c.obstacles.rare_switch_prob = 0.1;
    c.mpc.ego.num_discs = 1; c.mpc.ego.length = 1.5;
    c.mpc.safe_horizon_enabled = true; c.mpc.constraints.safe_horizon_min = 3;
    c.environment.path_completion_termination = true; c.environment.path_completion_fraction = 0.95;
    c.obstacles.num_obstacles = 1;
    c.mpc.enable_contouring_constraints = true; c.mpc.constraints.road_width = 4.0;
    c.environment.custom_ref_path = path; c.environment.custom_initial_ego = EgoState(0,0,0,1.5);
    c.obstacles.initial_obstacle_states = {obs};
    return c;
}
static void arm(const char* name, ExperimentConfig cfg, int N, double base) {
    int c = 0;
    for (int i = 0; i < N; ++i) c += run_experiment_rollout(cfg, 3000000u+unsigned(i)).collision ? 1:0;
    double p = double(c)/N, se = std::sqrt(std::max(0.0,p*(1-p)/N));
    std::printf("%-26s coll=%.3f [%.3f,%.3f]  benefit=%+.1fpp\n",
                name, p, std::max(0.0,p-1.96*se), std::min(1.0,p+1.96*se), 100.0*(base-p));
    std::fflush(stdout);
}
int main() {
    const int N = 200;
    ReferencePath path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);
    ExperimentConfig base = mk(path, place_oncoming(path, 0.0));

    int bc = 0; { ExperimentConfig c = base; c.dro.enabled = false;
        for (int i=0;i<N;++i) bc += run_experiment_rollout(c,3000000u+unsigned(i)).collision?1:0; }
    double bp = double(bc)/N;
    std::printf("### Injection vs sampling at offset 0 (road ON, S=40, N=%d) ###\n", N);
    std::printf("%-26s coll=%.3f\n\n", "base (no DRO)", bp);

    { ExperimentConfig c=base; c.dro.enabled = true; c.dro.injection_mode=InjectionMode::QSTAR_SAMPLE;
      arm("OT-only (QSTAR_SAMPLE)", c, N, bp); }
    { ExperimentConfig c=base; c.dro.enabled = true; c.dro.injection_mode=InjectionMode::TOP_RISK_INJECT; c.dro.injection_count=1;
      arm("DRO-inject K=1", c, N, bp); }
    { ExperimentConfig c=base; c.dro.enabled = true; c.dro.injection_mode=InjectionMode::TOP_RISK_INJECT; c.dro.injection_count=3;
      arm("DRO-inject K=3", c, N, bp); }
    { ExperimentConfig c=base; c.dro.enabled = true; c.dro.injection_mode=InjectionMode::TOP_RISK_INJECT; c.dro.injection_count=3;
      arm("TopRisk-inject K=3 (no OT)", c, N, bp); }
    std::printf("\nSLMPC: benefit is from EXTREMES INJECTION, not OT sampling (exp_h1 Base_vs_OT null).\n");
    return 0;
}
