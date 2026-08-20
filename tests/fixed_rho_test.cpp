// fixed_rho_test: (1) does forcing a constant LARGE ambiguity radius (no calibrated
// shrinkage) restore a WDRO collision benefit? (2) does the WDRO hedge actually tighten
// the QP -- i.e., produce MORE binding collision constraints than base? Offset-0 head-on,
// road ON, S=40, N=200. rho modes: calibrated (default) vs fixed 0.3 vs fixed 0.5.
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

static ExperimentConfig mk(const ReferencePath& path, const ObstacleState& obs, double offset) {
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
    (void)offset;
    return c;
}

struct R { double coll, lo, hi, actc; };
static R run(ExperimentConfig cfg, bool dro, double fixed_rho, int N) {
    cfg.dro.enabled = (dro);
    cfg.dro.injection_mode = dro ? InjectionMode::QSTAR_SAMPLE : InjectionMode::NONE;
    cfg.dro.fixed_rho = dro ? fixed_rho : -1.0;
    int c = 0; double ac = 0;
    for (int i = 0; i < N; ++i) {
        RolloutRecord r = run_experiment_rollout(cfg, 3000000u + unsigned(i));
        c += r.collision ? 1 : 0; ac += r.active_constraints;
    }
    double p = double(c)/N, se = std::sqrt(std::max(0.0,p*(1-p)/N));
    return {p, std::max(0.0,p-1.96*se), std::min(1.0,p+1.96*se), ac/N};
}

int main() {
    const int N = 200;
    ReferencePath path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);
    for (double off : {0.0, 1.0}) {
        ExperimentConfig base = mk(path, place_oncoming(path, off), off);
        R b = run(base, false, -1.0, N);
        std::printf("\n### offset %.1f  (road ON, S=40, N=%d) ###\n", off, N);
        std::printf("base (no DRO): coll=%.3f [%.3f,%.3f]  active_constraints/step=%.2f\n",
                    b.coll, b.lo, b.hi, b.actc);
        std::printf("%-22s %8s %14s %10s %12s\n","WDRO rho mode","coll","[95%% CI]","benefit","activeC/step");
        std::fflush(stdout);
        struct M { const char* name; double rho; };
        for (M m : {M{"calibrated (default)",-1.0}, M{"fixed 0.3",0.3}, M{"fixed 0.5",0.5}}) {
            R w = run(base, true, m.rho, N);
            std::printf("%-22s %8.3f [%.3f,%.3f] %+8.1fpp %11.2f\n",
                        m.name, w.coll, w.lo, w.hi, 100.0*(b.coll-w.coll), w.actc);
            std::fflush(stdout);
        }
    }
    std::printf("\nIf fixed-large-rho gives a benefit -> the calibrated radius was the problem.\n");
    std::printf("If WDRO activeC/step > base -> the hedge tightens the QP; if equal -> it does not.\n");
    return 0;
}
