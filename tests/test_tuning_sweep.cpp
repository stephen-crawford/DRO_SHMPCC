/**
 * @file test_tuning_sweep.cpp
 * @brief Parameter sweep to find strategy combinations that improve BOTH
 *        collision avoidance AND mode coverage with statistical significance.
 *
 * Tests variations of DRO(inj), DRO(inj)+SH, and OT+DRO(inj) with tuned parameters.
 *
 * Usage: ./test_tuning_sweep [max_rollouts] [--4obs]
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <sstream>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "dro.hpp"

using namespace dro_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "paper_figures/";

// ============================================================================
// Tuning configuration
// ============================================================================

struct TuningConfig {
    std::string name;
    bool enable_dro;
    InjectionMode injection_mode;
    bool safe_horizon;
    double rho;                     // DRO rho (eps_wass)
    int forced_safe_horizon;        // -1 = auto
    int safe_horizon_min;           // min constrained steps
    int num_scenarios;              // base scenarios
};

static const int    BASE_S = DEFAULT_BASE_SCENARIOS;  // 40

static std::vector<TuningConfig> build_configs() {
    std::vector<TuningConfig> cfgs;

    // === Reference strategies (same as main test) ===
    cfgs.push_back({"Base",
        false, InjectionMode::NONE, false,
        0.1, -1, 3, BASE_S});
    cfgs.push_back({"DRO(inj)",
        true, InjectionMode::TOP_RISK_INJECT, false,
        0.1, -1, 3, BASE_S});
    cfgs.push_back({"DRO(inj)+SH",
        true, InjectionMode::TOP_RISK_INJECT, true,
        0.1, -1, 3, BASE_S});

    // === Sweep 1: DRO rho for DRO(inj) alone ===
    for (double rho : {0.01, 0.05, 0.2, 0.3, 0.5}) {
        std::ostringstream ss;
        ss << "DRO(inj) rho=" << std::fixed << std::setprecision(2) << rho;
        cfgs.push_back({ss.str(),
            true, InjectionMode::TOP_RISK_INJECT, false,
            rho, -1, 3, BASE_S});
    }

    // === Sweep 2: DRO(inj)+SH with higher safe_horizon_min ===
    // Current min=3 is very aggressive (only 3/N steps constrained)
    // Higher values keep more constraints, reducing collision rate
    for (int sh_min : {6, 9, 12, 15}) {
        std::ostringstream ss;
        ss << "DRO(inj)+SH min=" << sh_min;
        cfgs.push_back({ss.str(),
            true, InjectionMode::TOP_RISK_INJECT, true,
            0.1, -1, sh_min, BASE_S});
    }

    // === Sweep 3: DRO(inj)+SH with forced safe horizon ===
    for (int fsh : {8, 11, 14}) {
        std::ostringstream ss;
        ss << "DRO(inj)+SH forced=" << fsh;
        cfgs.push_back({ss.str(),
            true, InjectionMode::TOP_RISK_INJECT, true,
            0.1, fsh, 3, BASE_S});
    }

    // === Sweep 4: More scenarios ===
    for (int S : {60, 80}) {
        std::ostringstream ss;
        ss << "DRO(inj) S=" << S;
        cfgs.push_back({ss.str(),
            true, InjectionMode::TOP_RISK_INJECT, false,
            0.1, -1, 3, S});
    }

    // === Sweep 5: Best combo attempts ===
    // DRO(inj)+SH with tuned rho and higher SH min
    cfgs.push_back({"DRO(inj)+SH rho=0.2 min=12",
        true, InjectionMode::TOP_RISK_INJECT, true,
        0.2, -1, 12, BASE_S});
    cfgs.push_back({"DRO(inj)+SH rho=0.05 min=15",
        true, InjectionMode::TOP_RISK_INJECT, true,
        0.05, -1, 15, BASE_S});
    cfgs.push_back({"DRO(inj) rho=0.2 S=60",
        true, InjectionMode::TOP_RISK_INJECT, false,
        0.2, -1, 3, 60});

    return cfgs;
}

// ============================================================================
// Metrics (simplified from test_paper_strategies)
// ============================================================================

struct Metrics {
    std::string name;
    int collisions = 0;
    int total = 0;
    int total_missed = 0;
    int total_checks = 0;
    int total_rare_active = 0;
    int total_rare_missed = 0;
    double sum_solve_ms = 0;
    int completed = 0;

    double collision_rate() const { return total > 0 ? (double)collisions / total : 0; }
    double mode_coverage() const { return total_checks > 0 ? 1.0 - (double)total_missed / total_checks : 1.0; }
    double rare_coverage() const { return total_rare_active > 0 ? 1.0 - (double)total_rare_missed / total_rare_active : 1.0; }
    double mean_solve() const { return total > 0 ? sum_solve_ms / total : 0; }
    double completion() const { return total > 0 ? (double)completed / total : 0; }

    std::pair<double,double> collision_ci() const { return wilson_ci(collisions, total, 1.96); }
    std::pair<double,double> mode_ci() const {
        int covered = total_checks - total_missed;
        return wilson_ci(covered, total_checks, 1.96);
    }
    std::pair<double,double> rare_ci() const {
        int covered = total_rare_active - total_rare_missed;
        return wilson_ci(covered, total_rare_active, 1.96);
    }

    void add(const RolloutRecord& rec) {
        collisions += rec.collision ? 1 : 0;
        total++;
        total_missed += rec.missed_mode_steps;
        total_checks += rec.total_mode_checks;
        total_rare_active += rec.rare_mode_active;
        total_rare_missed += rec.rare_mode_missed;
        sum_solve_ms += rec.avg_solve_ms;
        completed += rec.completed_path ? 1 : 0;
    }
};

static bool cis_overlap(std::pair<double,double> a, std::pair<double,double> b) {
    return a.second >= b.first && b.second >= a.first;
}

static std::string pct(double v) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(1) << (v * 100) << "%";
    return s.str();
}

static std::string ci_str(std::pair<double,double> ci) {
    std::ostringstream s;
    s << "[" << std::fixed << std::setprecision(1) << (ci.first * 100) << ","
      << (ci.second * 100) << "]";
    return s.str();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    fs::create_directories(OUTPUT_DIR);

    int max_rollouts = 2000;
    int num_obstacles = 4;
    int obstacles_per_class = 1;
    std::string suffix = "_4obs";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--1obs") {
            num_obstacles = 1;
            obstacles_per_class = 1;
            suffix = "";
        } else if (arg == "--4obs") {
            num_obstacles = 4;
            obstacles_per_class = 1;
            suffix = "_4obs";
        } else if (std::atoi(arg.c_str()) > 0 && i == 1) {
            max_rollouts = std::atoi(arg.c_str());
        }
    }

    auto all_configs = build_configs();
    int n_configs = static_cast<int>(all_configs.size());

    std::cout << "================================================================\n";
    std::cout << "  Parameter Tuning Sweep\n";
    std::cout << "  " << n_configs << " configurations, " << max_rollouts
              << " rollouts each, " << num_obstacles << " obstacle(s)\n";
    std::cout << "================================================================\n\n";

    // Multi-seed
    std::vector<unsigned> master_seeds = {42u, 1042u, 2042u, 3042u, 4042u};

    std::vector<Metrics> metrics(n_configs);
    for (int i = 0; i < n_configs; i++) {
        metrics[i].name = all_configs[i].name;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // Run each config
    for (int ci = 0; ci < n_configs; ci++) {
        const auto& tc = all_configs[ci];
        auto t0 = std::chrono::high_resolution_clock::now();

        ExperimentConfig cfg;
        cfg.mpc.horizon = DEFAULT_HORIZON;
        cfg.mpc.sampling.num_scenarios = tc.num_scenarios;
        cfg.obstacles.switch_prob = 0.1;
        cfg.rollout.rollout_steps = DEFAULT_ROLLOUT_STEPS;
        cfg.obstacles.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
        cfg.obstacles.rare_mode = "lane_change_left";
        cfg.obstacles.rare_switch_prob = 0.05;
        cfg.mpc.ego.num_discs = 1;
        cfg.mpc.ego.length = 1.5;
        cfg.environment.path_completion_termination = true;
        cfg.environment.path_completion_fraction = PATH_COMPLETE_FRAC;
        cfg.rollout.method_name = tc.name;

        cfg.dro.enabled = (tc.enable_dro);
        cfg.dro.solver.base_radius = tc.rho;

        cfg.mpc.safe_horizon_enabled = tc.safe_horizon;
        cfg.mpc.constraints.forced_safe_horizon = tc.forced_safe_horizon;
        cfg.mpc.constraints.safe_horizon_min = tc.safe_horizon_min;

        cfg.obstacles.num_obstacles = num_obstacles;
        cfg.obstacles.obstacles_per_class = obstacles_per_class;
        if (num_obstacles == 4) {
            cfg.obstacles.obs_arc_fractions = OBS_ARC_FRACS_4;
        }

        for (int r = 0; r < max_rollouts; r++) {
            unsigned ms = master_seeds[r % master_seeds.size()];
            SeedBundle seeds = derive_seeds(ms, r / static_cast<int>(master_seeds.size()));
            auto rec = run_experiment_rollout(cfg, seeds.env);
            metrics[ci].add(rec);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        auto [clo, chi] = metrics[ci].collision_ci();
        auto [mlo, mhi] = metrics[ci].mode_ci();
        std::cout << std::setw(3) << (ci + 1) << "/" << n_configs << " "
                  << std::setw(30) << std::left << tc.name
                  << std::right
                  << " coll=" << std::setw(5) << pct(metrics[ci].collision_rate())
                  << " " << std::setw(12) << ci_str({clo, chi})
                  << " mcov=" << std::setw(5) << pct(metrics[ci].mode_coverage())
                  << " " << std::setw(12) << ci_str({mlo, mhi})
                  << " rare=" << std::setw(5) << pct(metrics[ci].rare_coverage())
                  << "  " << std::fixed << std::setprecision(1) << secs << "s\n"
                  << std::flush;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(total_end - total_start).count();

    // === Write CSV ===
    std::string csv_path = OUTPUT_DIR + "tuning_sweep" + suffix + ".csv";
    {
        std::ofstream ofs(csv_path);
        ofs << "config,n,collision_rate,coll_ci_lo,coll_ci_hi,"
            << "mode_coverage,mode_ci_lo,mode_ci_hi,"
            << "rare_coverage,rare_ci_lo,rare_ci_hi,"
            << "mean_solve_ms,completion_rate\n";
        for (int i = 0; i < n_configs; i++) {
            auto& m = metrics[i];
            auto [clo, chi] = m.collision_ci();
            auto [mlo, mhi] = m.mode_ci();
            auto [rlo, rhi] = m.rare_ci();
            ofs << m.name << "," << m.total << ","
                << std::fixed << std::setprecision(4)
                << m.collision_rate() << "," << clo << "," << chi << ","
                << m.mode_coverage() << "," << mlo << "," << mhi << ","
                << m.rare_coverage() << "," << rlo << "," << rhi << ","
                << std::setprecision(2) << m.mean_solve() << ","
                << std::setprecision(4) << m.completion() << "\n";
        }
        ofs.close();
    }

    // === Analysis: find configs that beat Base on BOTH metrics ===
    std::cout << "\n================================================================\n";
    std::cout << "  Analysis: configs that beat Base on BOTH collision AND mode cov\n";
    std::cout << "  (non-overlapping CIs = statistically significant)\n";
    std::cout << "================================================================\n\n";

    auto base_coll_ci = metrics[0].collision_ci();
    auto base_mode_ci = metrics[0].mode_ci();
    double base_cr = metrics[0].collision_rate();
    double base_mc = metrics[0].mode_coverage();

    std::cout << "  Base: coll=" << pct(base_cr) << " " << ci_str(base_coll_ci)
              << "  mcov=" << pct(base_mc) << " " << ci_str(base_mode_ci) << "\n\n";

    // Also track which beat DRO(inj) (index 1)
    auto dro_coll_ci = metrics[1].collision_ci();
    auto dro_mode_ci = metrics[1].mode_ci();

    int found = 0;
    for (int i = 1; i < n_configs; i++) {
        auto& m = metrics[i];
        auto cci = m.collision_ci();
        auto mci = m.mode_ci();

        bool coll_better = cci.second < base_coll_ci.first;  // upper < base lower
        bool mode_better = mci.first > base_mode_ci.second;  // lower > base upper
        bool coll_sig = !cis_overlap(cci, base_coll_ci);
        bool mode_sig = !cis_overlap(mci, base_mode_ci);

        if (coll_better && mode_better) {
            found++;
            bool beats_dro_coll = cci.second < dro_coll_ci.first;
            bool beats_dro_mode = mci.first > dro_mode_ci.second;

            std::cout << "  ** " << m.name << " **\n"
                      << "     coll=" << pct(m.collision_rate()) << " " << ci_str(cci)
                      << (coll_sig ? " SIG" : " (ns)")
                      << "  mcov=" << pct(m.mode_coverage()) << " " << ci_str(mci)
                      << (mode_sig ? " SIG" : " (ns)")
                      << "  rare=" << pct(m.rare_coverage()) << "\n"
                      << "     vs DRO(inj): coll "
                      << (beats_dro_coll ? "BETTER" : "worse/same")
                      << "  mode " << (beats_dro_mode ? "BETTER" : "worse/same") << "\n\n";
        }
    }

    if (found == 0) {
        std::cout << "  No configuration beat Base on BOTH metrics simultaneously.\n";
    }

    // === Full ranking by collision rate ===
    std::cout << "\n  Full ranking by collision rate:\n";
    std::vector<int> order(n_configs);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return metrics[a].collision_rate() < metrics[b].collision_rate();
    });
    for (int i : order) {
        auto& m = metrics[i];
        auto cci = m.collision_ci();
        auto mci = m.mode_ci();
        bool coll_sig = !cis_overlap(cci, base_coll_ci);
        bool mode_sig = !cis_overlap(mci, base_mode_ci);
        std::cout << "    " << std::setw(30) << std::left << m.name
                  << std::right
                  << " coll=" << std::setw(5) << pct(m.collision_rate())
                  << " " << std::setw(12) << ci_str(cci)
                  << (coll_sig ? "*" : " ")
                  << " mcov=" << std::setw(5) << pct(m.mode_coverage())
                  << " " << std::setw(12) << ci_str(mci)
                  << (mode_sig ? "*" : " ")
                  << " rare=" << std::setw(5) << pct(m.rare_coverage())
                  << "\n";
    }
    std::cout << "    (* = CI separated from Base)\n";

    std::cout << "\n  Total time: " << std::fixed << std::setprecision(1)
              << total_s << "s (" << total_s / 60.0 << " min)\n";
    std::cout << "  Output: " << csv_path << "\n";
    std::cout << "================================================================\n";

    return 0;
}
