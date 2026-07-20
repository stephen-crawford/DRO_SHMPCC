// paper_recompute: regenerate the ACC paper's SAFETY artifacts on the CORRECTED code
// (calibrated radius + Bonferroni VaR + primal OT + J_d collision + velocity bounds +
// Dirichlet belief). Two artifacts:
//   (1) fig:offset  -- obstacle lateral-offset sweep, base vs WDRO.
//   (2) tab:road_on -- 5-arm comparison at offset 0 (base / eps-greedy / uniform /
//                      WDRO raw-LP / WDRO entropic) with control effort (R7.5).
// Placement replicates create_environment(ONCOMING) with a CONTROLLED offset (no jitter):
// obstacle at s=0.60 of the S-curve (len 25, amp 3), offset laterally, moving oncoming.
// Harness collision geometry: ego_radius 0.5 + obstacle_radius 0.35 = 0.85 m combined.
#include "experiment_harness.hpp"
#include "reference_path.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace scenario_mpc;

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
    c.horizon = DEFAULT_HORIZON;
    c.num_scenarios = DEFAULT_BASE_SCENARIOS;          // 40
    c.switch_prob = 0.2;
    c.rollout_steps = DEFAULT_ROLLOUT_STEPS;           // 200
    c.obs_modes = {"constant_velocity","turn_left","turn_right","decelerating"};
    c.rare_mode = "lane_change_left";
    c.rare_switch_prob = 0.1;
    c.num_discs = 1;
    c.vehicle_length = 1.5;
    c.safe_horizon_enabled = true;
    c.safe_horizon_min = 3;
    c.path_completion_termination = true;
    c.path_completion_fraction = 0.95;
    c.weight_type = WeightType::FREQUENCY;
    c.num_obstacles = 1;
    c.enable_contouring_constraints = true;            // road ON
    c.road_width = 4.0;                                 // lane 4.0 m, ego +/- 2.0 m
    c.custom_ref_path = path;
    c.custom_initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
    c.initial_obstacle_states = {obs};
    return c;
}

struct Res { double coll, lo, hi, clear, contour, velerr, effort, ms; };

static Res run(const ExperimentConfig& base, bool dro, WeightType wt, bool entropic, int N) {
    int c = 0; double cl=0, ct=0, ve=0, ef=0, ms=0;
    for (int i = 0; i < N; ++i) {
        ExperimentConfig cfg = base;
        cfg.weight_type = wt;
        cfg.enable_dro = dro;
        cfg.injection_mode = dro ? InjectionMode::QSTAR_SAMPLE : InjectionMode::NONE;
        cfg.use_entropic_allocator = entropic;
        cfg.entropic_tau = 0.05;
        RolloutRecord r = run_experiment_rollout(cfg, 3000000u + unsigned(i));
        c += r.collision ? 1 : 0; cl += r.min_clearance; ct += r.mean_contouring_error();
        ve += r.mean_velocity_error(); ef += r.control_effort; ms += r.avg_solve_ms;
    }
    double p = double(c)/N, se = std::sqrt(std::max(0.0, p*(1-p)/N));
    return {p, std::max(0.0,p-1.96*se), std::min(1.0,p+1.96*se), cl/N, ct/N, ve/N, ef/N, ms/N};
}

int main() {
    const int N = 250;
    ReferencePath path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);

    std::printf("### CORRECTED-CODE RECOMPUTE (main): calib radius + Bonferroni VaR + primal OT ###\n\n");

    // ---- (1) Offset sweep -------------------------------------------------
    std::printf("== fig:offset -- lateral-offset sweep (road ON, S=40, N=%d/arm) ==\n", N);
    std::printf("%-8s %10s %10s %9s %9s %9s\n","offset","base_coll","WDRO_coll","benefit","base_clr","WDRO_clr");
    const double offs[] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};
    for (double d : offs) {
        ExperimentConfig base = mk(path, place_oncoming(path, d));
        Res b = run(base, false, WeightType::FREQUENCY, false, N);
        Res w = run(base, true,  WeightType::FREQUENCY, false, N);
        std::printf("%-8.1f %10.3f %10.3f %+8.1fpp %9.3f %9.3f\n",
                    d, b.coll, w.coll, 100.0*(b.coll-w.coll), b.clear, w.clear);
        std::fflush(stdout);
    }

    // ---- (2) Road-on 5-arm table at offset 0 ------------------------------
    std::printf("\n== tab:road_on -- 5-arm at offset 0 (road ON, S=40, N=%d/arm) ==\n", N);
    std::printf("%-22s %7s %14s %8s %8s %8s %9s %6s\n",
                "Method","Coll.","[95%% CI]","Clear.","Contour","Vel.err","Effort","ms");
    ExperimentConfig base0 = mk(path, place_oncoming(path, 0.0));
    struct Arm { const char* name; bool dro; WeightType wt; bool ent; };
    Arm arms[] = {
        {"base (no DRO)",        false, WeightType::FREQUENCY,      false},
        {"eps-greedy belief",    false, WeightType::EPSILON_GREEDY, false},
        {"uniform belief",       false, WeightType::UNIFORM,        false},
        {"WDRO-sampling (rawLP)",true,  WeightType::FREQUENCY,      false},
        {"WDRO entropic t=0.05", true,  WeightType::FREQUENCY,      true },
    };
    for (const auto& a : arms) {
        Res r = run(base0, a.dro, a.wt, a.ent, N);
        std::printf("%-22s %6.3f [%.3f,%.3f] %8.3f %8.3f %8.3f %9.3f %6.2f\n",
                    a.name, r.coll, r.lo, r.hi, r.clear, r.contour, r.velerr, r.effort, r.ms);
        std::fflush(stdout);
    }
    std::printf("\nEffort = mean_k(a^2+omega^2) per rollout. Compare to OLD paper: base 0.656 -> WDRO 0.140.\n");
    return 0;
}
