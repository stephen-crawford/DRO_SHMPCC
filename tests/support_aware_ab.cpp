// support_aware_ab: closed-loop A/B of the support-aware Wasserstein floor.
//
// Base version      = raw-LP WDRO reweighting            (support_aware_alpha = 0).
// Support-enforced  = q <- alpha*p_hat + (1-alpha)*q*    (support_aware_alpha > 0).
//
// Same scenario, same seeds, DRO on in every arm. We sweep the mixing weight alpha
// at two scenario budgets: a TIGHT budget (S=5) where mode coverage actually binds,
// and the DEFAULT budget (S=20). The coverage stressor is the 5% rare mode
// (lane_change_left): if q* collapses to one mode, tight-budget scenario sets can
// miss the rare mode and collide; the floor keeps every mode sampled.
//
// Reports, per arm (mean over N seeds): collision rate (+/- 95% CI), marginal and
// rare-mode miss rates, path progress, min clearance, planning conservatism
// (contouring / velocity error), and solve time. No result is assumed -- if the
// floor changes nothing in closed loop, the table will say so.
#include "experiment_harness.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

using namespace scenario_mpc;

struct Agg {
    int n = 0, collisions = 0;
    long missed = 0, mode_checks = 0, rare_active = 0, rare_missed = 0;
    double progress = 0, min_clear = 0, contour = 0, vel_err = 0, solve_ms = 0, injected = 0;
    void add(const RolloutRecord& r) {
        ++n;
        collisions += r.collision ? 1 : 0;
        missed += r.missed_mode_steps;  mode_checks += r.total_mode_checks;
        rare_active += r.rare_mode_active; rare_missed += r.rare_mode_missed;
        progress += r.total_progress; min_clear += r.min_clearance;
        contour += r.mean_contouring_error(); vel_err += r.mean_velocity_error();
        solve_ms += r.avg_solve_ms; injected += r.total_dro_injected;
    }
};

int main() {
    const int N = 200;
    const std::vector<int> budgets = {5, 20};
    const std::vector<double> alphas = {0.0, 0.2, 0.5};

    std::printf("=== Support-aware Wasserstein floor: closed-loop A/B (N=%d seeds/arm) ===\n", N);
    std::printf("scenario: default S-curve rollout, DRO on, rare mode lane_change_left @5%%\n\n");
    std::printf("%3s %6s | %7s %14s | %8s %8s | %7s %8s | %7s %7s | %6s\n",
                "S", "alpha", "coll%", "[95% CI]", "miss%", "rareMiss%",
                "prog", "minClr", "contour", "velErr", "ms");
    std::printf("----------------------------------------------------------------------------------------------------\n");

    for (int S : budgets) {
        for (double alpha : alphas) {
            Agg a;
            for (int s = 0; s < N; ++s) {
                ExperimentConfig cfg;               // defaults: DRO plumbing on
                cfg.enable_dro = true;
                cfg.num_scenarios = S;
                cfg.rollout_steps = 60;
                cfg.support_aware_alpha = alpha;
                // Keep every other knob at the justified main defaults.
                RolloutRecord r = run_experiment_rollout(cfg, 1000u + s);
                a.add(r);
            }
            const double p = static_cast<double>(a.collisions) / a.n;
            const double se = std::sqrt(std::max(0.0, p * (1 - p) / a.n));
            const double lo = std::max(0.0, p - 1.96 * se), hi = std::min(1.0, p + 1.96 * se);
            const double miss = a.mode_checks ? 100.0 * a.missed / a.mode_checks : 0.0;
            const double rmiss = a.rare_active ? 100.0 * a.rare_missed / a.rare_active : 0.0;
            std::printf("%3d %6.2f | %6.1f%% [%4.1f, %4.1f]%% | %7.2f %8.2f | %7.3f %8.3f | %7.4f %7.4f | %5.2f\n",
                        S, alpha, 100.0 * p, 100.0 * lo, 100.0 * hi, miss, rmiss,
                        a.progress / a.n, a.min_clear / a.n,
                        a.contour / a.n, a.vel_err / a.n, a.solve_ms / a.n);
        }
        std::printf("----------------------------------------------------------------------------------------------------\n");
    }
    std::printf("\nRead: alpha=0 is the base raw-LP WDRO. Compare each S-block's alpha rows.\n");
    std::printf("A real coverage benefit shows as lower rareMiss%% and/or coll%% at alpha>0,\n");
    std::printf("bought (if at all) with higher contour/velErr conservatism.\n");
    return 0;
}
