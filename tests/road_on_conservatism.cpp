// road_on_conservatism: reproduce the road-constrained 5-arm comparison (tab:road_on)
// and additionally report CONTROL EFFORT (CDC Reviewer 7.5) alongside the existing
// conservatism columns, all from the SAME runs so the table is internally consistent.
//
// Arms: base (no DRO) | eps-greedy belief | uniform belief | WDRO raw-LP | WDRO entropic.
// Config mirrors test_cdc_experiments::make_base_config + road ON (lane 4.0 m), ONCOMING
// obstacle, S=40, N=250 rollouts/arm on identical seeds. Because the controller defaults
// were corrected since the original table (Bonferroni VaR, calibrated radius, primal OT),
// we also print collision so reproduction fidelity can be judged before the table is edited.
#include "experiment_harness.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace scenario_mpc;

struct Arm { std::string name; bool dro; WeightType wt; bool entropic; };

struct Agg {
    int n = 0, coll = 0;
    double clear = 0, contour = 0, velerr = 0, effort = 0, ms = 0;
    void add(const RolloutRecord& r) {
        ++n; coll += r.collision ? 1 : 0;
        clear += r.min_clearance; contour += r.mean_contouring_error();
        velerr += r.mean_velocity_error(); effort += r.control_effort; ms += r.avg_solve_ms;
    }
};

static ExperimentConfig base_cfg() {
    ExperimentConfig c;
    c.horizon = DEFAULT_HORIZON;
    c.num_scenarios = DEFAULT_BASE_SCENARIOS;      // 40
    c.switch_prob = 0.2;
    c.rollout_steps = DEFAULT_ROLLOUT_STEPS;       // 200
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
    // Road ON (the corrected benchmark): lane 4.0 m, ego +/- 2.0 m.
    c.enable_contouring_constraints = true;
    c.road_width = 4.0;
    return c;
}

int main() {
    const int N = 250;
    const std::vector<Arm> arms = {
        {"base (no DRO)",        false, WeightType::FREQUENCY,      false},
        {"eps-greedy belief",    false, WeightType::EPSILON_GREEDY, false},
        {"uniform belief",       false, WeightType::UNIFORM,        false},
        {"WDRO-sampling (rawLP)",true,  WeightType::FREQUENCY,      false},
        {"WDRO entropic t=0.05", true,  WeightType::FREQUENCY,      true },
    };

    std::printf("=== Road-constrained 5-arm comparison (S=40, N=%d/arm, road ON, ONCOMING) ===\n", N);
    std::printf("%-24s %7s %14s %7s %8s %8s %9s %6s\n",
                "Method","Coll.","[95%% CI]","Clear.","Contour","Vel.err","Effort","ms");
    std::printf("--------------------------------------------------------------------------------------------\n");

    for (const auto& a : arms) {
        Agg g;
        for (int i = 0; i < N; ++i) {
            unsigned seed = 2000000u + static_cast<unsigned>(i);   // identical across arms
            std::mt19937 env_rng(seed);
            EnvironmentSetup env = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig c = base_cfg();
            c.weight_type = a.wt;
            c.enable_dro = a.dro;
            c.injection_mode = a.dro ? InjectionMode::QSTAR_SAMPLE : InjectionMode::NONE;
            c.use_entropic_allocator = a.entropic;
            c.entropic_tau = 0.05;
            c.initial_obstacle_states = {env.initial_obs};
            c.obs_modes = env.obs_modes;

            g.add(run_experiment_rollout(c, seed));
        }
        const double p = double(g.coll) / g.n;
        const double se = std::sqrt(std::max(0.0, p*(1-p)/g.n));
        std::printf("%-24s %6.3f [%.3f,%.3f] %7.3f %8.3f %8.3f %9.3f %6.2f\n",
                    a.name.c_str(), p, std::max(0.0,p-1.96*se), std::min(1.0,p+1.96*se),
                    g.clear/g.n, g.contour/g.n, g.velerr/g.n, g.effort/g.n, g.ms/g.n);
    }
    std::printf("--------------------------------------------------------------------------------------------\n");
    std::printf("Effort = sum_k (a_k^2 + omega_k^2) per rollout, mean over rollouts (R7.5).\n");
    return 0;
}
