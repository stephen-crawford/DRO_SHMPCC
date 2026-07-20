// paper_recompute_coverage: regenerate fig:coverage (the coverage-null mechanism figure)
// on the CORRECTED code. For V in {3,4} multi-obstacle base-planner rollouts, split into
// collided vs safe and compare their JOINT-missed-mode rate (no scenario covers every
// obstacle's true mode at once). A NEGATIVE gap (collided <= safe) => coverage is causally
// inert. Multi-obstacle auto-placement along the S-curve (harness default), road ON, S=40.
#include "experiment_harness.hpp"
#include "reference_path.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace scenario_mpc;

int main() {
    const int N = 250;
    ReferencePath path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);

    std::printf("### CORRECTED-CODE RECOMPUTE: fig:coverage (joint-missed-mode gap) ###\n");
    std::printf("%-4s %10s %10s %10s %8s %8s\n","V","coll%","jm|collided","jm|safe","gap","n_coll");
    for (int V : {3, 4}) {
        int coll = 0; double jm_c = 0, jm_s = 0; int nc = 0, ns = 0;
        for (int i = 0; i < N; ++i) {
            ExperimentConfig c;
            c.horizon = DEFAULT_HORIZON;
            c.num_scenarios = DEFAULT_BASE_SCENARIOS;
            c.switch_prob = 0.2;
            c.rollout_steps = DEFAULT_ROLLOUT_STEPS;
            c.obs_modes = {"constant_velocity","turn_left","turn_right","decelerating"};
            c.rare_mode = "lane_change_left";
            c.rare_switch_prob = 0.1;
            c.num_discs = 1; c.vehicle_length = 1.5;
            c.safe_horizon_enabled = true; c.safe_horizon_min = 3;
            c.path_completion_termination = true; c.path_completion_fraction = 0.95;
            c.weight_type = WeightType::FREQUENCY;
            c.enable_contouring_constraints = true; c.road_width = 4.0;
            c.custom_ref_path = path; c.custom_initial_ego = EgoState(0.0,0.0,0.0,1.5);
            c.num_obstacles = V;             // auto-placed along the S-curve
            c.enable_dro = false;            // base planner
            RolloutRecord r = run_experiment_rollout(c, 4000000u + unsigned(V*100000 + i));
            double jm = r.joint_mode_checks > 0 ? double(r.joint_missed_mode_steps)/r.joint_mode_checks : 0.0;
            if (r.collision) { coll++; jm_c += jm; nc++; } else { jm_s += jm; ns++; }
        }
        double mc = nc? jm_c/nc : 0.0, ms = ns? jm_s/ns : 0.0;
        std::printf("%-4d %9.1f%% %10.3f %10.3f %+7.3f %8d\n",
                    V, 100.0*coll/N, mc, ms, mc-ms, nc);
        std::fflush(stdout);
    }
    std::printf("\nNegative gap (collided <= safe joint-miss) => joint coverage is causally inert.\n");
    return 0;
}
