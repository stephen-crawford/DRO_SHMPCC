// diagnose_wdro_benefit: isolate WHICH corrected-code change removed the WDRO collision
// benefit at offset 0 (the paper's headline placement). num_discs=1 here, so the J_d
// collision fix is inactive -- suspects are the three DRO defaults, toggled one at a time.
// If calibrated-radius-OFF restores the benefit, the small calibrated rho is the cause.
#include "experiment_harness.hpp"
#include "reference_path.hpp"
#include <cstdio>
#include <cmath>
#include <string>

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

static double coll_rate(ExperimentConfig cfg, int N) {
    int c = 0;
    for (int i = 0; i < N; ++i) c += run_experiment_rollout(cfg, 3000000u + unsigned(i)).collision ? 1 : 0;
    return double(c) / N;
}

int main() {
    const int N = 200;
    ReferencePath path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);
    ExperimentConfig base = mk(path, place_oncoming(path, 0.0));

    std::printf("### Diagnose WDRO benefit at offset 0 (road ON, S=40, N=%d) ###\n", N);
    ExperimentConfig b = base; b.dro.enabled = false;
    double base_c = coll_rate(b, N);
    std::printf("base (no DRO): %.3f\n\n", base_c);
    std::printf("%-42s %8s %9s\n", "WDRO config", "coll", "benefit");
    std::fflush(stdout);

    struct Cfg { const char* name; bool calib; DRORiskMeasure rm; bool ot; };
    Cfg cfgs[] = {
        {"NEW default (calib+Bonferroni+primalOT)", true,  DRORiskMeasure::SURROGATE_VAR_BONFERRONI, true},
        {"calib radius OFF (old heuristic rho)",    false, DRORiskMeasure::SURROGATE_VAR_BONFERRONI, true},
        {"risk = SURROGATE_VAR (not Bonferroni)",   true,  DRORiskMeasure::SURROGATE_VAR,            true},
        {"primal OT OFF (heuristic recovery)",      true,  DRORiskMeasure::SURROGATE_VAR_BONFERRONI, false},
        {"ALL OLD DRO (calib off+VAR+heuristic)",   false, DRORiskMeasure::SURROGATE_VAR,            false},
    };
    for (const auto& k : cfgs) {
        ExperimentConfig w = base;
        w.dro.enabled = true; w.dro.injection_mode = InjectionMode::QSTAR_SAMPLE;
        w.dro.solver.radius_calibration.use_calibrated_radius = k.calib; w.dro.solver.radius_calibration.risk_measure = k.rm; w.dro.solver.radius_calibration.use_primal_ot = k.ot;
        double wc = coll_rate(w, N);
        std::printf("%-42s %8.3f %+8.1fpp\n", k.name, wc, 100.0*(base_c - wc));
        std::fflush(stdout);
    }
    std::printf("\n(old paper benefit at offset 0: +47.6pp)\n");
    return 0;
}
