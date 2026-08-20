/**
 * @file test_strategy_sweep.cpp
 * @brief Comprehensive sweep of 6 strategies across conditions.
 *
 * Sweeps: switch_prob x rare_prob x obstacle_config.
 * Each cell runs batched rollouts with Wilson CI early stopping.
 *
 * Usage:
 *   ./test_strategy_sweep                    # all 60 conditions
 *   ./test_strategy_sweep --max-rollouts 2000
 *   ./test_strategy_sweep --switch-only      # fix rare=0.05, obs=1obs
 *   ./test_strategy_sweep --rare-only        # fix switch=0.1, obs=1obs
 *   ./test_strategy_sweep --obs-only         # fix switch=0.1, rare=0.05
 *
 * Outputs:
 *   paper_figures/sweep_summary.csv
 *   paper_figures/sweep_config.txt
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
#include <cassert>
#include <filesystem>
#include <sstream>
#include <cstring>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"

using namespace dro_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "paper_figures/";

// ============================================================================
// Strategies (same as test_paper_strategies.cpp)
// ============================================================================

struct StrategyDef {
    std::string name;
    bool enable_dro;
    InjectionMode injection_mode;
    bool safe_horizon;
};

static const std::vector<StrategyDef> STRATEGIES = {
    // Non-SH baselines
    {"Base",            false, InjectionMode::NONE,         false},
    {"DRO(inj)",        true,  InjectionMode::TOP_RISK_INJECT,          false},
    // SH variants
    {"SH",              false, InjectionMode::NONE,         true},
    {"DRO(inj)+SH",     true,  InjectionMode::TOP_RISK_INJECT,          true},
    {"DRO(q*)+SH",      true,  InjectionMode::QSTAR_SAMPLE, true},
};

// ============================================================================
// Condition specification
// ============================================================================

struct ConditionSpec {
    double switch_prob;
    double rare_prob;
    int num_obstacles;
    int obstacles_per_class;  // obs_per_class == num_obstacles => all independent
    std::string obs_tag;      // "1obs", "4obs_ind", "4obs_shared"

    std::string label() const {
        std::ostringstream s;
        s << "sw" << std::fixed << std::setprecision(2) << switch_prob
          << "_rp" << std::setprecision(2) << rare_prob
          << "_" << obs_tag;
        return s.str();
    }
};

static const std::vector<double> SWITCH_PROBS = {0.02, 0.05, 0.1, 0.2, 0.4};
static const std::vector<double> RARE_PROBS   = {0.01, 0.05, 0.1, 0.2};

struct ObsConfig {
    int num_obs;
    int per_class;
    std::string tag;
};
static const std::vector<ObsConfig> OBS_CONFIGS = {
    {1, 1, "1obs"},
    {4, 1, "4obs_ind"},     // 4 obstacles, each its own class
    {4, 4, "4obs_shared"},  // 4 obstacles, all same class
};

static std::vector<ConditionSpec> build_full_grid() {
    std::vector<ConditionSpec> grid;
    for (double sp : SWITCH_PROBS) {
        for (double rp : RARE_PROBS) {
            for (const auto& oc : OBS_CONFIGS) {
                grid.push_back({sp, rp, oc.num_obs, oc.per_class, oc.tag});
            }
        }
    }
    return grid;
}

// ============================================================================
// Defaults
// ============================================================================

static constexpr int    BATCH_SIZE     = 200;
static constexpr int    MIN_ROLLOUTS   = 400;
static constexpr int    MAX_ROLLOUTS   = 5000;
static constexpr double CI_Z           = 1.96;
static constexpr int    HORIZON        = DEFAULT_HORIZON;
static constexpr int    NUM_SCENARIOS  = DEFAULT_BASE_SCENARIOS;
static constexpr int    NUM_DISCS      = 1;
static constexpr double VEHICLE_LENGTH = 1.5;
static constexpr double DRO_RHO_BASE  = 0.1;

static const std::vector<std::string> OBS_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Metrics (same struct as test_paper_strategies)
// ============================================================================

struct StrategyMetrics {
    std::string name;
    std::vector<bool> collisions;
    std::vector<int> missed_mode_steps;
    std::vector<int> total_mode_checks;
    std::vector<int> rare_mode_active;
    std::vector<int> rare_mode_missed;
    std::vector<double> solve_times_avg;
    std::vector<double> solve_times_p50;
    std::vector<double> solve_times_p95;
    std::vector<double> clearance_5pct;
    std::vector<bool> completed_path;
    std::vector<int> dro_injected;
    std::vector<double> avg_safe_horizon;

    int n() const { return static_cast<int>(collisions.size()); }
    int collision_count() const {
        int c = 0; for (bool b : collisions) if (b) c++; return c;
    }
    double collision_rate() const {
        return n() > 0 ? static_cast<double>(collision_count()) / n() : 0.0;
    }
    std::pair<double,double> collision_ci() const {
        return wilson_ci(collision_count(), n(), CI_Z);
    }
    int total_missed() const {
        return std::accumulate(missed_mode_steps.begin(), missed_mode_steps.end(), 0);
    }
    int total_checks() const {
        return std::accumulate(total_mode_checks.begin(), total_mode_checks.end(), 0);
    }
    double mode_coverage() const {
        int tc = total_checks();
        return tc > 0 ? 1.0 - static_cast<double>(total_missed()) / tc : 1.0;
    }
    std::pair<double,double> mode_coverage_ci() const {
        int tc = total_checks(), cov = tc - total_missed();
        return wilson_ci(cov, tc, CI_Z);
    }
    int total_rare_active() const {
        return std::accumulate(rare_mode_active.begin(), rare_mode_active.end(), 0);
    }
    int total_rare_missed() const {
        return std::accumulate(rare_mode_missed.begin(), rare_mode_missed.end(), 0);
    }
    double rare_mode_coverage() const {
        int a = total_rare_active();
        return a > 0 ? 1.0 - static_cast<double>(total_rare_missed()) / a : 1.0;
    }
    std::pair<double,double> rare_mode_ci() const {
        int a = total_rare_active(), cov = a - total_rare_missed();
        return wilson_ci(cov, a, CI_Z);
    }
    double mean_solve_ms() const {
        if (solve_times_avg.empty()) return 0;
        return std::accumulate(solve_times_avg.begin(), solve_times_avg.end(), 0.0)
               / solve_times_avg.size();
    }
    double median_solve_ms() const {
        if (solve_times_p50.empty()) return 0;
        auto v = solve_times_p50; std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
    double p95_solve_ms() const {
        if (solve_times_p95.empty()) return 0;
        auto v = solve_times_p95; std::sort(v.begin(), v.end());
        return v[static_cast<int>(0.95 * (v.size() - 1))];
    }
    double mean_clearance_5pct() const {
        if (clearance_5pct.empty()) return 0;
        return std::accumulate(clearance_5pct.begin(), clearance_5pct.end(), 0.0)
               / clearance_5pct.size();
    }
    double completion_rate() const {
        int c = 0; for (bool b : completed_path) if (b) c++;
        return n() > 0 ? static_cast<double>(c) / n() : 0.0;
    }
    double mean_dro_injected() const {
        if (dro_injected.empty()) return 0;
        return std::accumulate(dro_injected.begin(), dro_injected.end(), 0.0)
               / dro_injected.size();
    }
    double mean_safe_horizon() const {
        if (avg_safe_horizon.empty()) return 0;
        return std::accumulate(avg_safe_horizon.begin(), avg_safe_horizon.end(), 0.0)
               / avg_safe_horizon.size();
    }

    void add(const RolloutRecord& rec) {
        collisions.push_back(rec.collision);
        missed_mode_steps.push_back(rec.missed_mode_steps);
        total_mode_checks.push_back(rec.total_mode_checks);
        rare_mode_active.push_back(rec.rare_mode_active);
        rare_mode_missed.push_back(rec.rare_mode_missed);
        solve_times_avg.push_back(rec.avg_solve_ms);
        solve_times_p50.push_back(rec.p50_solve_ms);
        solve_times_p95.push_back(rec.p95_solve_ms);
        clearance_5pct.push_back(rec.clearance_5pct);
        completed_path.push_back(rec.completed_path);
        dro_injected.push_back(rec.total_dro_injected);
        avg_safe_horizon.push_back(rec.avg_safe_horizon);
    }
};

// ============================================================================
// Build config
// ============================================================================

static ExperimentConfig make_config(const StrategyDef& strat,
                                     const ConditionSpec& cond) {
    ExperimentConfig cfg;
    cfg.mpc.horizon = HORIZON;
    cfg.mpc.sampling.num_scenarios = NUM_SCENARIOS;
    cfg.rollout.rollout_steps = DEFAULT_ROLLOUT_STEPS;
    cfg.mpc.ego.num_discs = NUM_DISCS;
    cfg.mpc.ego.length = VEHICLE_LENGTH;
    cfg.environment.path_completion_termination = true;
    cfg.environment.path_completion_fraction = PATH_COMPLETE_FRAC;
    cfg.rollout.method_name = strat.name;
    cfg.obstacles.obs_modes = OBS_MODES;
    cfg.obstacles.rare_mode = RARE_MODE;
    cfg.obstacles.switch_prob = cond.switch_prob;
    cfg.obstacles.rare_switch_prob = cond.rare_prob;

    cfg.obstacles.num_obstacles = cond.num_obstacles;
    cfg.obstacles.obstacles_per_class = cond.obstacles_per_class;
    if (cond.num_obstacles > 1) {
        cfg.obstacles.obs_arc_fractions = OBS_ARC_FRACS_4;
    }

    cfg.dro.enabled = (strat.enable_dro);
    cfg.dro.injection_mode = strat.injection_mode;
    cfg.dro.solver.base_radius = DRO_RHO_BASE;
    cfg.mpc.safe_horizon_enabled = strat.safe_horizon;

    return cfg;
}

// ============================================================================
// CI overlap
// ============================================================================

static bool cis_overlap(std::pair<double,double> a, std::pair<double,double> b) {
    return a.second >= b.first && b.second >= a.first;
}

// ============================================================================
// Run one condition
// ============================================================================

struct ConditionResult {
    ConditionSpec cond;
    std::vector<StrategyMetrics> metrics;
    int total_rollouts;
    bool early_stopped;
    double elapsed_s;
};

static ConditionResult run_condition(
    const ConditionSpec& cond,
    int max_rollouts,
    int batch_size
) {
    ConditionResult result;
    result.cond = cond;
    result.metrics.resize(STRATEGIES.size());
    for (size_t i = 0; i < STRATEGIES.size(); i++)
        result.metrics[i].name = STRATEGIES[i].name;
    result.total_rollouts = 0;
    result.early_stopped = false;

    // Per-condition seed: mix condition index into master seed
    auto hash_seed = [](unsigned a, unsigned b) -> unsigned {
        unsigned h = 2166136261u;
        h ^= a; h *= 16777619u;
        h ^= b; h *= 16777619u;
        return h;
    };
    // Derive a unique master seed from condition parameters
    unsigned cond_seed = hash_seed(42u,
        static_cast<unsigned>(cond.switch_prob * 1000) * 1000 +
        static_cast<unsigned>(cond.rare_prob * 1000) * 10 +
        static_cast<unsigned>(cond.num_obstacles * cond.obstacles_per_class));

    std::vector<ExperimentConfig> configs(STRATEGIES.size());
    for (size_t i = 0; i < STRATEGIES.size(); i++)
        configs[i] = make_config(STRATEGIES[i], cond);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int batch = 0; result.total_rollouts < max_rollouts; batch++) {
        int end = std::min(result.total_rollouts + batch_size, max_rollouts);
        int count = end - result.total_rollouts;

        for (size_t si = 0; si < STRATEGIES.size(); si++) {
            for (int r = result.total_rollouts; r < end; r++) {
                SeedBundle seeds = derive_seeds(cond_seed, r);
                auto rec = run_experiment_rollout(configs[si], seeds.env);
                result.metrics[si].add(rec);
            }
        }
        result.total_rollouts = end;

        // Early stop: all non-baseline collision CIs separated
        if (result.total_rollouts >= MIN_ROLLOUTS) {
            auto base_ci = result.metrics[0].collision_ci();
            int separated = 0;
            for (size_t i = 1; i < STRATEGIES.size(); i++) {
                if (!cis_overlap(base_ci, result.metrics[i].collision_ci()))
                    separated++;
            }
            if (separated == static_cast<int>(STRATEGIES.size()) - 1) {
                result.early_stopped = true;
                break;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

// ============================================================================
// CSV output
// ============================================================================

static void write_summary_header(std::ofstream& ofs) {
    ofs << "condition,switch_prob,rare_prob,num_obstacles,obstacles_per_class,obs_config,"
        << "strategy,n,"
        << "collision_rate,collision_ci_lo,collision_ci_hi,"
        << "mode_coverage,mode_cov_ci_lo,mode_cov_ci_hi,"
        << "rare_mode_coverage,rare_cov_ci_lo,rare_cov_ci_hi,"
        << "mean_solve_ms,median_solve_ms,p95_solve_ms,"
        << "mean_clearance_5pct,completion_rate,"
        << "mean_dro_injected,mean_safe_horizon,"
        << "early_stopped\n";
}

static void write_condition_rows(std::ofstream& ofs, const ConditionResult& cr) {
    for (const auto& m : cr.metrics) {
        auto [clo, chi] = m.collision_ci();
        auto [mlo, mhi] = m.mode_coverage_ci();
        auto [rlo, rhi] = m.rare_mode_ci();

        ofs << cr.cond.label() << ","
            << std::fixed << std::setprecision(2)
            << cr.cond.switch_prob << "," << cr.cond.rare_prob << ","
            << cr.cond.num_obstacles << "," << cr.cond.obstacles_per_class << ","
            << cr.cond.obs_tag << ","
            << m.name << "," << m.n() << ","
            << std::setprecision(4)
            << m.collision_rate() << "," << clo << "," << chi << ","
            << m.mode_coverage() << "," << mlo << "," << mhi << ","
            << m.rare_mode_coverage() << "," << rlo << "," << rhi << ","
            << std::setprecision(2)
            << m.mean_solve_ms() << "," << m.median_solve_ms() << ","
            << m.p95_solve_ms() << ","
            << std::setprecision(4) << m.mean_clearance_5pct() << ","
            << m.completion_rate() << ","
            << std::setprecision(1) << m.mean_dro_injected() << ","
            << m.mean_safe_horizon() << ","
            << (cr.early_stopped ? 1 : 0) << "\n";
    }
    ofs.flush();
}

// ============================================================================
// Print one condition result
// ============================================================================

static std::string fmt_ci(double lo, double hi) {
    std::ostringstream s;
    s << "[" << std::fixed << std::setprecision(1)
      << (lo*100) << "," << (hi*100) << "]";
    return s.str();
}

static void print_condition(const ConditionResult& cr) {
    std::cout << "  " << std::setw(8)  << "Strategy"
              << std::setw(8)  << "Coll%"
              << std::setw(14) << "CI"
              << std::setw(8)  << "MCov%"
              << std::setw(8)  << "Rare%"
              << std::setw(8)  << "Solve"
              << "\n";
    for (const auto& m : cr.metrics) {
        auto [clo, chi] = m.collision_ci();
        std::cout << "  " << std::setw(8) << std::left << m.name
                  << std::fixed << std::setprecision(1)
                  << std::setw(8)  << std::right << (m.collision_rate()*100)
                  << std::setw(14) << fmt_ci(clo, chi)
                  << std::setw(8)  << (m.mode_coverage()*100)
                  << std::setw(8)  << (m.rare_mode_coverage()*100)
                  << std::setprecision(2) << std::setw(8) << m.mean_solve_ms()
                  << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    fs::create_directories(OUTPUT_DIR);

    int max_rollouts = MAX_ROLLOUTS;
    int batch_size = BATCH_SIZE;

    // Parse simple flags
    enum SweepMode { ALL, SWITCH_ONLY, RARE_ONLY, OBS_ONLY };
    SweepMode mode = ALL;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--max-rollouts") == 0 && i+1 < argc)
            max_rollouts = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--batch-size") == 0 && i+1 < argc)
            batch_size = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--switch-only") == 0)
            mode = SWITCH_ONLY;
        else if (std::strcmp(argv[i], "--rare-only") == 0)
            mode = RARE_ONLY;
        else if (std::strcmp(argv[i], "--obs-only") == 0)
            mode = OBS_ONLY;
    }

    // Build condition grid, possibly filtered
    auto full_grid = build_full_grid();
    std::vector<ConditionSpec> grid;

    switch (mode) {
        case ALL:
            grid = full_grid;
            break;
        case SWITCH_ONLY:
            for (auto& c : full_grid)
                if (std::abs(c.rare_prob - 0.05) < 0.001 && c.obs_tag == "1obs")
                    grid.push_back(c);
            break;
        case RARE_ONLY:
            for (auto& c : full_grid)
                if (std::abs(c.switch_prob - 0.1) < 0.001 && c.obs_tag == "1obs")
                    grid.push_back(c);
            break;
        case OBS_ONLY:
            for (auto& c : full_grid)
                if (std::abs(c.switch_prob - 0.1) < 0.001 && std::abs(c.rare_prob - 0.05) < 0.001)
                    grid.push_back(c);
            break;
    }

    std::cout << "================================================================\n";
    std::cout << "  Strategy Sweep: " << grid.size() << " conditions x "
              << STRATEGIES.size() << " strategies\n";
    std::cout << "  Max " << max_rollouts << " rollouts/strategy, batch=" << batch_size << "\n";
    std::cout << "================================================================\n\n" << std::flush;

    // Open summary CSV (append if exists and has data, otherwise create with header)
    std::string summary_path = OUTPUT_DIR + "sweep_summary.csv";
    bool file_exists = fs::exists(summary_path) && fs::file_size(summary_path) > 10;
    std::ofstream summary_ofs(summary_path, file_exists ? std::ios::app : std::ios::trunc);
    if (!file_exists)
        write_summary_header(summary_ofs);

    auto total_start = std::chrono::high_resolution_clock::now();
    int conditions_done = 0;

    for (size_t ci = 0; ci < grid.size(); ci++) {
        const auto& cond = grid[ci];
        std::cout << "[" << (ci+1) << "/" << grid.size() << "] "
                  << cond.label() << " ..." << std::flush;

        auto cr = run_condition(cond, max_rollouts, batch_size);
        write_condition_rows(summary_ofs, cr);
        conditions_done++;

        std::cout << " done in " << std::fixed << std::setprecision(1)
                  << cr.elapsed_s << "s, " << cr.total_rollouts << " rollouts"
                  << (cr.early_stopped ? " (early)" : " (cap)") << "\n";
        print_condition(cr);
        std::cout << std::flush;
    }

    summary_ofs.close();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(total_end - total_start).count();

    // Write config
    {
        std::ofstream cfg(OUTPUT_DIR + "sweep_config.txt");
        cfg << "# Strategy Sweep Configuration\n";
        cfg << "conditions = " << grid.size() << "\n";
        cfg << "strategies = " << STRATEGIES.size() << "\n";
        cfg << "max_rollouts = " << max_rollouts << "\n";
        cfg << "batch_size = " << batch_size << "\n";
        cfg << "switch_probs =";
        for (double v : SWITCH_PROBS) cfg << " " << v;
        cfg << "\nrare_probs =";
        for (double v : RARE_PROBS) cfg << " " << v;
        cfg << "\nobs_configs = 1obs 4obs_ind 4obs_shared\n";
        cfg << "total_time_s = " << total_s << "\n";
        cfg.close();
    }

    std::cout << "\n================================================================\n";
    std::cout << "  Sweep complete: " << conditions_done << " conditions in "
              << std::fixed << std::setprecision(0) << total_s << "s ("
              << std::setprecision(1) << total_s/60.0 << " min)\n";
    std::cout << "  Output: " << summary_path << "\n";
    std::cout << "================================================================\n";

    return 0;
}
