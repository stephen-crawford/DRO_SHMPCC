/**
 * @file test_paper_strategies.cpp
 * @brief Comprehensive comparison of OT/DRO/SH strategies for the paper.
 *
 * Strategies tested (8 total, with and without safe horizon):
 *   1. Base               — No OT, no DRO, no SH (baseline)
 *   2. OT                 — OT predictor only
 *   3. DRO(inj)           — DRO worst-case injection only
 *   4. SH                 — Safe Horizon only
 *   5. OT+SH              — OT + Safe Horizon
 *   6. DRO(inj)+SH        — DRO injection + SH
 *   7. OT+DRO(inj)+SH     — OT + DRO injection + SH
 *   8. DRO(q*)+SH         — DRO q* resampling + SH
 *
 * Runs in batches with Wilson 95% CIs. Stops early when all non-baseline
 * collision CIs separate from baseline, or at a rollout cap.
 *
 * Outputs:
 *   paper_figures/strategy_comparison_results.csv   — per-rollout data
 *   paper_figures/strategy_comparison_summary.csv   — per-strategy CIs
 *   paper_figures/strategy_comparison_config.txt    — exact configuration
 *
 * Usage: ./test_paper_strategies [max_rollouts] [batch_size]
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
#include <cassert>
#include <filesystem>
#include <sstream>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "paper_figures/";

// ============================================================================
// Strategy definition
// ============================================================================

struct StrategyDef {
    std::string name;
    bool enable_dro;
    InjectionMode injection_mode;
    bool safe_horizon;
};

static const std::vector<StrategyDef> STRATEGIES = {
    // Non-SH baselines (expect higher collision rates)
    {"Base",            false, InjectionMode::NONE,         false},
    {"DRO(inj)",        true,  InjectionMode::DRO,          false},
    // SH variants
    {"SH",              false, InjectionMode::NONE,         true},
    {"DRO(inj)+SH",     true,  InjectionMode::DRO,          true},
    {"DRO(q*)+SH",      true,  InjectionMode::QSTAR_SAMPLE, true},
};

// ============================================================================
// Test parameters
// ============================================================================

static constexpr int    BATCH_SIZE        = 200;
static constexpr int    MIN_ROLLOUTS      = 400;
static constexpr int    MAX_ROLLOUTS      = 5000;
static constexpr double CI_Z              = 1.96;    // 95% Wilson CI
static constexpr int    ROLLOUT_STEPS     = DEFAULT_ROLLOUT_STEPS;
static constexpr int    HORIZON           = DEFAULT_HORIZON;
static constexpr int    NUM_SCENARIOS     = DEFAULT_BASE_SCENARIOS;
static constexpr double DT                = DEFAULT_DT;
static constexpr double SWITCH_PROB       = 0.1;
static constexpr double RARE_PROB         = 0.05;
static constexpr int    NUM_DISCS         = 1;
static constexpr double VEHICLE_LENGTH    = 1.5;
static constexpr double DRO_RHO_BASE     = 0.1;

static const std::vector<std::string> OBS_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Accumulated metrics per strategy
// ============================================================================

struct StrategyMetrics {
    std::string name;
    std::vector<bool> collisions;
    std::vector<double> min_clearances;
    std::vector<double> solve_times_avg;
    std::vector<double> solve_times_p50;
    std::vector<double> solve_times_p95;
    std::vector<double> solve_times_max;
    std::vector<int> missed_mode_steps;
    std::vector<int> total_mode_checks;
    std::vector<int> rare_mode_active;
    std::vector<int> rare_mode_missed;
    std::vector<double> total_progress;
    std::vector<bool> completed_path;
    std::vector<int> dro_injected;
    std::vector<double> avg_safe_horizon;
    std::vector<double> clearance_5pct;
    std::vector<int> total_steps;

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

    // Aggregate mode coverage as proportion: (total_checks - missed) / total_checks
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
        int tc = total_checks(), covered = tc - total_missed();
        return wilson_ci(covered, tc, CI_Z);
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
        int a = total_rare_active(), covered = a - total_rare_missed();
        return wilson_ci(covered, a, CI_Z);
    }

    double mean_solve_ms() const {
        if (solve_times_avg.empty()) return 0;
        return std::accumulate(solve_times_avg.begin(), solve_times_avg.end(), 0.0)
               / solve_times_avg.size();
    }
    double median_solve_ms() const {
        if (solve_times_p50.empty()) return 0;
        auto v = solve_times_p50;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }
    double p95_solve_ms() const {
        if (solve_times_p95.empty()) return 0;
        auto v = solve_times_p95;
        std::sort(v.begin(), v.end());
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
        min_clearances.push_back(rec.min_clearance);
        solve_times_avg.push_back(rec.avg_solve_ms);
        solve_times_p50.push_back(rec.p50_solve_ms);
        solve_times_p95.push_back(rec.p95_solve_ms);
        solve_times_max.push_back(rec.max_solve_ms);
        missed_mode_steps.push_back(rec.missed_mode_steps);
        total_mode_checks.push_back(rec.total_mode_checks);
        rare_mode_active.push_back(rec.rare_mode_active);
        rare_mode_missed.push_back(rec.rare_mode_missed);
        total_progress.push_back(rec.total_progress);
        completed_path.push_back(rec.completed_path);
        dro_injected.push_back(rec.total_dro_injected);
        avg_safe_horizon.push_back(rec.avg_safe_horizon);
        clearance_5pct.push_back(rec.clearance_5pct);
        total_steps.push_back(rec.total_steps);
    }
};

// ============================================================================
// Build ExperimentConfig for a strategy
// ============================================================================

static ExperimentConfig make_strategy_config(const StrategyDef& strat) {
    ExperimentConfig cfg;
    cfg.horizon = HORIZON;
    cfg.num_scenarios = NUM_SCENARIOS;
    cfg.switch_prob = SWITCH_PROB;
    cfg.rollout_steps = ROLLOUT_STEPS;
    cfg.obs_modes = OBS_MODES;
    cfg.rare_mode = RARE_MODE;
    cfg.rare_switch_prob = RARE_PROB;
    cfg.num_discs = NUM_DISCS;
    cfg.vehicle_length = VEHICLE_LENGTH;
    cfg.path_completion_termination = true;
    cfg.path_completion_fraction = PATH_COMPLETE_FRAC;
    cfg.method_name = strat.name;
    cfg.ablation = AblationVariant::NO_INJECTION;

    cfg.weight_type = WeightType::FREQUENCY;

    cfg.enable_dro = strat.enable_dro;
    cfg.injection_mode = strat.injection_mode;
    cfg.eps_wass = DRO_RHO_BASE;

    cfg.safe_horizon_enabled = strat.safe_horizon;

    return cfg;
}

// ============================================================================
// CI overlap check: do two Wilson CIs overlap?
// ============================================================================

static bool cis_overlap(std::pair<double,double> a, std::pair<double,double> b) {
    return a.second >= b.first && b.second >= a.first;
}

// ============================================================================
// Write configuration file
// ============================================================================

static void write_config_file(const std::string& filepath, int total_rollouts,
                               int batch_size, int max_rollouts) {
    std::ofstream ofs(filepath);
    ofs << "# Strategy Comparison Experiment Configuration\n";
    ofs << "# Generated: " << __DATE__ << " " << __TIME__ << "\n\n";

    ofs << "[experiment]\n";
    ofs << "batch_size = " << batch_size << "\n";
    ofs << "min_rollouts = " << MIN_ROLLOUTS << "\n";
    ofs << "max_rollouts = " << max_rollouts << "\n";
    ofs << "actual_rollouts = " << total_rollouts << "\n";
    ofs << "ci_z = " << CI_Z << " (95% Wilson CI)\n\n";

    ofs << "[mpc]\n";
    ofs << "horizon = " << HORIZON << "\n";
    ofs << "dt = " << DT << "\n";
    ofs << "num_scenarios = " << NUM_SCENARIOS << "\n";
    ofs << "rollout_steps = " << ROLLOUT_STEPS << "\n";
    ofs << "num_discs = " << NUM_DISCS << "\n";
    ofs << "vehicle_length = " << VEHICLE_LENGTH << "\n";
    ofs << "ego_radius = 0.5\n";
    ofs << "obstacle_radius = 0.35\n";
    ofs << "safety_margin = 0.2\n";
    ofs << "safe_horizon_min = 3\n";
    ofs << "safe_horizon_mode = PRACTICAL\n";
    ofs << "ensure_mode_coverage = true\n";
    ofs << "use_sqp_solver = true\n\n";

    ofs << "[environment]\n";
    ofs << "path_type = s_curve\n";
    ofs << "s_curve_length = " << S_CURVE_LENGTH << "\n";
    ofs << "s_curve_amplitude = " << S_CURVE_AMPLITUDE << "\n";
    ofs << "path_completion_fraction = " << PATH_COMPLETE_FRAC << "\n";
    ofs << "obs_arc_fraction = " << OBS_PATH_FRACTION << "\n";
    ofs << "switch_prob = " << SWITCH_PROB << "\n";
    ofs << "rare_mode_prob = " << RARE_PROB << "\n";
    ofs << "rare_mode = " << RARE_MODE << "\n";
    ofs << "modes = constant_velocity, turn_left, turn_right, decelerating, "
        << RARE_MODE << "\n\n";

    ofs << "[dro]\n";
    ofs << "rho_base = " << DRO_RHO_BASE << "\n";
    ofs << "rho_min = 0.01\n";
    ofs << "rho_max = 0.5\n";
    ofs << "adaptive_rho = true\n";
    ofs << "alpha_one_sided = 0.95\n";
    ofs << "ground_cost = W2_BURES\n";
    ofs << "risk_mode = FULL\n\n";

    ofs << "[strategies]\n";
    for (size_t i = 0; i < STRATEGIES.size(); i++) {
        const auto& s = STRATEGIES[i];
        ofs << i << ". " << s.name << "\n";
        ofs << "   enable_dro = " << (s.enable_dro ? "true" : "false") << "\n";
        ofs << "   injection_mode = ";
        switch (s.injection_mode) {
            case InjectionMode::NONE: ofs << "NONE"; break;
            case InjectionMode::DRO: ofs << "DRO (worst-case injection)"; break;
            case InjectionMode::QSTAR_SAMPLE: ofs << "QSTAR_SAMPLE (resample from q*)"; break;
            case InjectionMode::ADVERSARIAL: ofs << "ADVERSARIAL"; break;
            case InjectionMode::RANDOM: ofs << "RANDOM"; break;
            case InjectionMode::ALL_MODES: ofs << "ALL_MODES"; break;
        }
        ofs << "\n";
        ofs << "   safe_horizon = " << (s.safe_horizon ? "true" : "false") << "\n\n";
    }
    ofs.close();
}

// ============================================================================
// Write per-rollout CSV
// ============================================================================

static void write_detail_csv(
    const std::string& filepath,
    const std::vector<StrategyMetrics>& metrics
) {
    std::ofstream ofs(filepath);
    ofs << "strategy,rollout_idx,collision,min_clearance,clearance_5pct,"
        << "avg_solve_ms,p50_solve_ms,p95_solve_ms,max_solve_ms,"
        << "missed_mode_steps,total_mode_checks,mode_coverage,"
        << "rare_mode_active,rare_mode_missed,rare_mode_coverage,"
        << "total_progress,completed_path,dro_injected,avg_safe_horizon,total_steps\n";

    for (const auto& m : metrics) {
        for (int i = 0; i < m.n(); i++) {
            double mc = m.total_mode_checks[i] > 0
                ? 1.0 - static_cast<double>(m.missed_mode_steps[i]) / m.total_mode_checks[i] : 1.0;
            double rmc = m.rare_mode_active[i] > 0
                ? 1.0 - static_cast<double>(m.rare_mode_missed[i]) / m.rare_mode_active[i] : 1.0;

            ofs << m.name << "," << i << ","
                << (m.collisions[i] ? 1 : 0) << ","
                << std::fixed << std::setprecision(4)
                << m.min_clearances[i] << ","
                << m.clearance_5pct[i] << ","
                << m.solve_times_avg[i] << ","
                << m.solve_times_p50[i] << ","
                << m.solve_times_p95[i] << ","
                << m.solve_times_max[i] << ","
                << m.missed_mode_steps[i] << ","
                << m.total_mode_checks[i] << ","
                << std::setprecision(4) << mc << ","
                << m.rare_mode_active[i] << ","
                << m.rare_mode_missed[i] << ","
                << std::setprecision(4) << rmc << ","
                << m.total_progress[i] << ","
                << (m.completed_path[i] ? 1 : 0) << ","
                << m.dro_injected[i] << ","
                << std::setprecision(2) << m.avg_safe_horizon[i] << ","
                << m.total_steps[i] << "\n";
        }
    }
    ofs.close();
}

// ============================================================================
// Write summary CSV with Wilson CIs for all proportion metrics
// ============================================================================

static void write_summary_csv(
    const std::string& filepath,
    const std::vector<StrategyMetrics>& metrics
) {
    std::ofstream ofs(filepath);
    ofs << "strategy,n,"
        << "collision_rate,collision_ci_lo,collision_ci_hi,"
        << "mode_coverage,mode_cov_ci_lo,mode_cov_ci_hi,"
        << "rare_mode_coverage,rare_cov_ci_lo,rare_cov_ci_hi,"
        << "mean_solve_ms,median_solve_ms,p95_solve_ms,"
        << "mean_clearance_5pct,completion_rate,"
        << "mean_dro_injected,mean_safe_horizon\n";

    for (const auto& m : metrics) {
        auto [clo, chi] = m.collision_ci();
        auto [mlo, mhi] = m.mode_coverage_ci();
        auto [rlo, rhi] = m.rare_mode_ci();

        ofs << m.name << "," << m.n() << ","
            << std::fixed << std::setprecision(4)
            << m.collision_rate() << "," << clo << "," << chi << ","
            << m.mode_coverage() << "," << mlo << "," << mhi << ","
            << m.rare_mode_coverage() << "," << rlo << "," << rhi << ","
            << std::setprecision(2)
            << m.mean_solve_ms() << ","
            << m.median_solve_ms() << ","
            << m.p95_solve_ms() << ","
            << std::setprecision(4) << m.mean_clearance_5pct() << ","
            << m.completion_rate() << ","
            << std::setprecision(1) << m.mean_dro_injected() << ","
            << m.mean_safe_horizon() << "\n";
    }
    ofs.close();
}

// ============================================================================
// Print status: CI table for each metric
// ============================================================================

static std::string fmt_ci(double lo, double hi) {
    std::ostringstream s;
    s << "[" << std::fixed << std::setprecision(1)
      << (lo * 100.0) << "," << (hi * 100.0) << "]";
    return s.str();
}

static void print_status(
    int batch_num,
    int total_rollouts,
    const std::vector<StrategyMetrics>& metrics,
    double batch_time_s
) {
    std::cout << "\n========== Batch " << batch_num
              << " (" << total_rollouts << " rollouts/strategy, "
              << std::fixed << std::setprecision(1) << batch_time_s << "s)"
              << " ==========\n";

    // Collision rate with CIs
    std::cout << "\n  Collision rate (95% Wilson CI):\n";
    for (const auto& m : metrics) {
        auto [lo, hi] = m.collision_ci();
        std::cout << "    " << std::setw(20) << std::left << m.name
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << std::right << (m.collision_rate() * 100.0) << "%  "
                  << std::setw(14) << fmt_ci(lo, hi) << "\n";
    }

    // Mode coverage with CIs
    std::cout << "\n  Mode coverage (95% Wilson CI):\n";
    for (const auto& m : metrics) {
        auto [lo, hi] = m.mode_coverage_ci();
        std::cout << "    " << std::setw(20) << std::left << m.name
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << std::right << (m.mode_coverage() * 100.0) << "%  "
                  << std::setw(14) << fmt_ci(lo, hi) << "\n";
    }

    // Rare mode coverage with CIs
    std::cout << "\n  Rare mode coverage (95% Wilson CI):\n";
    for (const auto& m : metrics) {
        auto [lo, hi] = m.rare_mode_ci();
        std::cout << "    " << std::setw(20) << std::left << m.name
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << std::right << (m.rare_mode_coverage() * 100.0) << "%  "
                  << std::setw(14) << fmt_ci(lo, hi) << "\n";
    }

    // Solve time
    std::cout << "\n  Solve time (ms): mean / p50 / p95\n";
    for (const auto& m : metrics) {
        std::cout << "    " << std::setw(20) << std::left << m.name
                  << std::fixed << std::setprecision(2)
                  << m.mean_solve_ms() << " / "
                  << m.median_solve_ms() << " / "
                  << m.p95_solve_ms() << "\n";
    }

    // CI separation vs baseline (index 0)
    const auto& base = metrics[0];
    auto base_coll_ci = base.collision_ci();
    auto base_mc_ci = base.mode_coverage_ci();
    int separated = 0;
    std::cout << "\n  CI separation vs " << base.name << ":\n";
    for (size_t i = 1; i < metrics.size(); i++) {
        auto ci = metrics[i].collision_ci();
        auto mci = metrics[i].mode_coverage_ci();
        bool coll_sep = !cis_overlap(base_coll_ci, ci);
        bool mc_sep = !cis_overlap(base_mc_ci, mci);
        if (coll_sep) separated++;
        std::cout << "    " << std::setw(20) << std::left << metrics[i].name
                  << "coll:" << (coll_sep ? " SEPARATED" : " overlaps ")
                  << "  mode_cov:" << (mc_sep ? " SEPARATED" : " overlaps ")
                  << "\n";
    }
    std::cout << "  (" << separated << "/" << (metrics.size() - 1)
              << " collision CIs separated from baseline)\n" << std::flush;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    fs::create_directories(OUTPUT_DIR);

    int max_rollouts = MAX_ROLLOUTS;
    int batch_size = BATCH_SIZE;
    int num_obstacles = 1;
    int obstacles_per_class = 1;
    int num_seeds = 1;
    std::string output_suffix = "";

    // Parse args: [max_rollouts] [batch_size] or --4obs / --seeds N
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--4obs") {
            num_obstacles = 4;
            obstacles_per_class = 1;
            output_suffix = "_4obs";
        } else if (arg == "--4obs-shared") {
            num_obstacles = 4;
            obstacles_per_class = 4;
            output_suffix = "_4obs_shared";
        } else if (arg == "--2obs") {
            num_obstacles = 2;
            obstacles_per_class = 1;
            output_suffix = "_2obs";
        } else if (arg == "--seeds" && i + 1 < argc) {
            num_seeds = std::atoi(argv[++i]);
            if (num_seeds < 1) num_seeds = 1;
        } else if (i == 1 && std::atoi(arg.c_str()) > 0) {
            max_rollouts = std::atoi(arg.c_str());
        } else if (i == 2 && std::atoi(arg.c_str()) > 0) {
            batch_size = std::atoi(arg.c_str());
        }
    }
    if (max_rollouts <= 0) max_rollouts = MAX_ROLLOUTS;
    if (batch_size <= 0) batch_size = BATCH_SIZE;

    auto total_start = std::chrono::high_resolution_clock::now();

    std::string obs_desc = std::to_string(num_obstacles) + " obstacle(s)";
    if (num_obstacles > 1 && obstacles_per_class > 1)
        obs_desc += " (shared class)";
    else if (num_obstacles > 1)
        obs_desc += " (independent)";

    std::cout << "================================================================\n";
    std::cout << "  OT/DRO/SH Strategy Comparison Experiment\n";
    std::cout << "  " << STRATEGIES.size() << " strategies, max "
              << max_rollouts << " rollouts/strategy, " << obs_desc << "\n";
    if (num_seeds > 1)
        std::cout << "  " << num_seeds << " master seeds (varying switching behavior)\n";
    std::cout << "  Batch size: " << batch_size
              << ", early stop when all collision CIs separate from baseline\n";
    std::cout << "================================================================\n\n"
              << std::flush;

    // Initialize
    std::vector<StrategyMetrics> metrics(STRATEGIES.size());
    for (size_t i = 0; i < STRATEGIES.size(); i++) {
        metrics[i].name = STRATEGIES[i].name;
    }

    std::vector<ExperimentConfig> configs(STRATEGIES.size());
    for (size_t i = 0; i < STRATEGIES.size(); i++) {
        configs[i] = make_strategy_config(STRATEGIES[i]);
        configs[i].num_obstacles = num_obstacles;
        configs[i].obstacles_per_class = obstacles_per_class;
        if (num_obstacles == 4) {
            configs[i].obs_arc_fractions = OBS_ARC_FRACS_4;
        } else if (num_obstacles == 2) {
            configs[i].obs_arc_fractions = {0.30, 0.55};
        }
    }

    // Master seeds: use different seeds to vary obstacle switching behavior
    std::vector<unsigned> master_seeds;
    for (int s = 0; s < num_seeds; s++) {
        master_seeds.push_back(42u + s * 1000u);
    }

    int total_rollouts = 0;
    bool early_stop = false;

    if (num_seeds > 1) {
        std::cout << "  Using " << num_seeds << " master seeds to vary switching behavior\n"
                  << "  Seeds:";
        for (auto s : master_seeds) std::cout << " " << s;
        std::cout << "\n\n" << std::flush;
    }

    // Sequential batching loop
    for (int batch = 0; total_rollouts < max_rollouts; batch++) {
        auto batch_start = std::chrono::high_resolution_clock::now();

        int batch_end = std::min(total_rollouts + batch_size, max_rollouts);
        int batch_count = batch_end - total_rollouts;

        std::cout << "Running batch " << batch << ": rollouts "
                  << total_rollouts << ".." << (batch_end - 1)
                  << " (" << batch_count << " per strategy)...\n" << std::flush;

        for (size_t si = 0; si < STRATEGIES.size(); si++) {
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int r = total_rollouts; r < batch_end; r++) {
                // Cycle through master seeds to vary switching behavior
                unsigned ms = master_seeds[r % master_seeds.size()];
                SeedBundle seeds = derive_seeds(ms, r / static_cast<int>(master_seeds.size()));
                auto rec = run_experiment_rollout(configs[si], seeds.env);
                metrics[si].add(rec);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "  " << std::setw(20) << std::left << STRATEGIES[si].name
                      << batch_count << " in "
                      << std::fixed << std::setprecision(1) << s << "s"
                      << "  coll=" << std::setprecision(1)
                      << (metrics[si].collision_rate() * 100.0) << "%\n" << std::flush;
        }

        total_rollouts = batch_end;

        auto batch_end_t = std::chrono::high_resolution_clock::now();
        double batch_s = std::chrono::duration<double>(batch_end_t - batch_start).count();
        print_status(batch, total_rollouts, metrics, batch_s);

        // Write intermediate results
        write_detail_csv(OUTPUT_DIR + "strategy_comparison_results" + output_suffix + ".csv", metrics);
        write_summary_csv(OUTPUT_DIR + "strategy_comparison_summary" + output_suffix + ".csv", metrics);
        write_config_file(OUTPUT_DIR + "strategy_comparison_config" + output_suffix + ".txt",
                         total_rollouts, batch_size, max_rollouts);

        // Early stop: all non-baseline collision CIs separated from baseline
        if (total_rollouts >= MIN_ROLLOUTS) {
            auto base_ci = metrics[0].collision_ci();
            int separated = 0;
            for (size_t i = 1; i < metrics.size(); i++) {
                if (!cis_overlap(base_ci, metrics[i].collision_ci()))
                    separated++;
            }
            if (separated == static_cast<int>(metrics.size()) - 1) {
                early_stop = true;
                std::cout << "\n*** All collision CIs separated from baseline at "
                          << total_rollouts << " rollouts. ***\n" << std::flush;
                break;
            }
        }
    }

    // Final write
    write_detail_csv(OUTPUT_DIR + "strategy_comparison_results" + output_suffix + ".csv", metrics);
    write_summary_csv(OUTPUT_DIR + "strategy_comparison_summary" + output_suffix + ".csv", metrics);
    write_config_file(OUTPUT_DIR + "strategy_comparison_config" + output_suffix + ".txt",
                     total_rollouts, batch_size, max_rollouts);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(total_end - total_start).count();

    // Final summary
    std::cout << "\n================================================================\n";
    std::cout << "  Done in " << std::fixed << std::setprecision(1)
              << total_s << "s (" << total_s / 60.0 << " min), "
              << total_rollouts << " rollouts/strategy"
              << (early_stop ? " (early stop)" : " (cap)") << "\n\n";

    std::cout << "  " << std::setw(20) << std::left << "Strategy"
              << std::setw(8)  << "Coll%"
              << std::setw(15) << "CollCI"
              << std::setw(8)  << "MCov%"
              << std::setw(15) << "MCovCI"
              << std::setw(8)  << "Rare%"
              << std::setw(15) << "RareCI"
              << std::setw(9)  << "Solve ms"
              << std::setw(7)  << "DROInj"
              << "\n";
    std::cout << "  " << std::string(105, '-') << "\n";

    for (const auto& m : metrics) {
        auto [clo, chi] = m.collision_ci();
        auto [mlo, mhi] = m.mode_coverage_ci();
        auto [rlo, rhi] = m.rare_mode_ci();

        std::cout << "  " << std::setw(20) << std::left << m.name
                  << std::fixed
                  << std::setprecision(1) << std::setw(8) << (m.collision_rate() * 100.0)
                  << std::setw(15) << fmt_ci(clo, chi)
                  << std::setprecision(1) << std::setw(8) << (m.mode_coverage() * 100.0)
                  << std::setw(15) << fmt_ci(mlo, mhi)
                  << std::setprecision(1) << std::setw(8) << (m.rare_mode_coverage() * 100.0)
                  << std::setw(15) << fmt_ci(rlo, rhi)
                  << std::setprecision(2) << std::setw(9) << m.mean_solve_ms()
                  << std::setprecision(1) << std::setw(7) << m.mean_dro_injected()
                  << "\n";
    }
    std::cout << "  " << std::string(105, '-') << "\n";

    std::cout << "\n  Output files:\n";
    std::cout << "    " << OUTPUT_DIR << "strategy_comparison_results.csv\n";
    std::cout << "    " << OUTPUT_DIR << "strategy_comparison_summary.csv\n";
    std::cout << "    " << OUTPUT_DIR << "strategy_comparison_config.txt\n";
    std::cout << "================================================================\n";

    return 0;
}
