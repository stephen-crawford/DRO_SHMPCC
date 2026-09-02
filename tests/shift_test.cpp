// shift_test: does WDRO help under BELIEF MISSPECIFICATION (distribution shift)?
// WDRO is robustness insurance -- theory predicts benefit only when the nominal belief
// is wrong. On a well-specified Dirichlet belief the recompute showed ~0 benefit. Here
// we induce shift: the obstacle is forced into the dangerous rare mode (lane_change_left,
// boosted_mode=-1) more often than the belief expects, and/or randomly reshuffles its
// mode (shift.psi). If WDRO benefit grows with shift, the effect is real in-regime.
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

static double coll(ExperimentConfig cfg, bool dro, int N) {
    cfg.dro.enabled = (dro);
    int c = 0;
    for (int i = 0; i < N; ++i) c += run_experiment_rollout(cfg, 3000000u + unsigned(i)).collision ? 1 : 0;
    return double(c) / N;
}

int main() {
    const int N = 200;
    ReferencePath path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);
    ExperimentConfig base = mk(path, place_oncoming(path, 0.0));

    std::printf("### WDRO under distribution shift (offset 0, road ON, S=40, N=%d) ###\n", N);
    std::printf("%-28s %8s %8s %9s\n","shift","base","WDRO","benefit");
    std::fflush(stdout);

    struct S { const char* name; double rho; double boost; };
    S grid[] = {
        {"none (well-specified)",     0.0, 0.0},
        {"dangerous-boost 0.2",       0.0, 0.2},
        {"dangerous-boost 0.4",       0.0, 0.4},
        {"random-shift rho=0.3",      0.3, 0.0},
        {"random 0.3 + boost 0.3",    0.3, 0.3},
    };
    for (const auto& s : grid) {
        ExperimentConfig cfg = base;
        cfg.obstacles.shift.psi = s.rho; cfg.obstacles.shift.dangerous_boost = s.boost; cfg.obstacles.shift.boosted_mode = -1;
        double b = coll(cfg, false, N);
        double w = coll(cfg, true,  N);
        std::printf("%-28s %8.3f %8.3f %+8.1fpp\n", s.name, b, w, 100.0*(b - w));
        std::fflush(stdout);
    }
    std::printf("\nIf benefit grows with shift, WDRO's value is robustness to misspecification.\n");
    return 0;
}
