/**
 * @file test_path_obstacle_sweep.cpp
 * @brief Sweep over reference path shapes, obstacle counts, and obstacle behaviors.
 *
 * Compares CDC paper strategies (no OT-based methods):
 *   Base, SH, DRO(inj), DRO(inj)+SH, DRO(q*)+SH
 *
 * Outputs: paper_figures/path_obstacle_sweep.csv
 *          paper_figures/path_obstacle_raw.csv
 *
 * Usage:
 *   ./test_path_obstacle_sweep           # run all three focused sweeps
 *   ./test_path_obstacle_sweep paths     # sweep path shapes only
 *   ./test_path_obstacle_sweep obstacles # sweep obstacle counts only
 *   ./test_path_obstacle_sweep behaviors # sweep behavior profiles only
 *   ./test_path_obstacle_sweep full      # full cross-product (slow)
 */

#include "experiment_harness.hpp"
#include "reference_path.hpp"
#include "config.hpp"
#include "types.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

using namespace dro_mpc;

// ============================================================================
// Configuration
// ============================================================================

static constexpr int ROLLOUTS_PER_CONDITION = 300;
static constexpr int ROLLOUT_STEPS = 200;
static constexpr int NUM_SCENARIOS = 40;
static constexpr int HORIZON = DEFAULT_HORIZON;
static constexpr double DRO_RHO = 0.1;

static const std::string OUT_DIR = "paper_figures/";

// ============================================================================
// CDC Strategy Definitions (no OT)
// ============================================================================

struct StrategyDef {
    std::string name;
    bool enable_dro;
    InjectionMode injection_mode;
    bool safe_horizon;
};

static const std::vector<StrategyDef> STRATEGIES = {
    {"Base",         false, InjectionMode::NONE,         false},
    {"SH",           false, InjectionMode::NONE,         true},
    {"DRO(inj)",     true,  InjectionMode::TOP_RISK_INJECT,          false},
    {"DRO(inj)+SH",  true,  InjectionMode::TOP_RISK_INJECT,          true},
    {"DRO(q*)+SH",   true,  InjectionMode::QSTAR_SAMPLE, true},
};

// ============================================================================
// Path Shape Definitions
// ============================================================================

struct PathDef {
    std::string name;
    ReferencePath path;
    EgoState initial_ego;
};

std::vector<PathDef> create_path_defs() {
    std::vector<PathDef> defs;

    // 1. Straight line (25m)
    defs.push_back({
        "straight",
        ReferencePath::create_straight({0, 0}, {25, 0}, 200),
        EgoState(0.0, 0.0, 0.0, 1.5)
    });

    // 2. Standard S-curve (25m, amplitude 3m)
    defs.push_back({
        "s_curve",
        ReferencePath::create_s_curve(25.0, 3.0, 200),
        EgoState(0.0, 0.0, 0.0, 1.5)
    });

    // 3. Tight chicane (shorter, higher amplitude)
    defs.push_back({
        "chicane",
        ReferencePath::create_s_curve(15.0, 5.0, 200),
        EgoState(0.0, 0.0, 0.0, 1.5)
    });

    // 4. Wide arc (R=10m, 150 degree turn)
    {
        double R = 10.0;
        double start_angle = -M_PI / 2;
        double end_angle = start_angle + 5.0 * M_PI / 6.0;
        defs.push_back({
            "wide_arc",
            ReferencePath::create_circle({0, R}, R, start_angle, end_angle, 200),
            EgoState(0.0, 0.0, 0.0, 1.5)
        });
    }

    // 5. Hairpin U-turn (R=4m, 180 degree turn)
    {
        double R = 4.0;
        double start_angle = -M_PI / 2;
        double end_angle = M_PI / 2;
        defs.push_back({
            "hairpin",
            ReferencePath::create_circle({0, R}, R, start_angle, end_angle, 200),
            EgoState(0.0, 0.0, 0.0, 1.2)
        });
    }

    return defs;
}

// ============================================================================
// Obstacle Behavior Definitions
// ============================================================================

struct BehaviorDef {
    std::string name;
    std::vector<std::string> modes;
    std::string rare_mode;
    double rare_prob;
    double switch_prob;
};

std::vector<BehaviorDef> create_behavior_defs() {
    return {
        {"conservative",
            {"constant_velocity", "decelerating"},
            "", 0.0, 0.05},
        {"balanced",
            {"constant_velocity", "turn_left", "turn_right", "decelerating"},
            "", 0.0, 0.1},
        {"aggressive",
            {"turn_left", "turn_right", "lane_change_left", "lane_change_right"},
            "", 0.0, 0.2},
        {"erratic",
            {"constant_velocity", "turn_left", "turn_right", "decelerating"},
            "lane_change_left", 0.1, 0.3},
    };
}

// ============================================================================
// Obstacle Count Configurations
// ============================================================================

struct ObsCountDef {
    int num_obstacles;
    int obstacles_per_class;
    std::vector<double> arc_fractions;
};

std::vector<ObsCountDef> create_obs_count_defs() {
    return {
        {1, 1, {0.35}},
        {2, 1, {0.30, 0.55}},
        {4, 2, {0.20, 0.35, 0.50, 0.65}},
    };
}

// ============================================================================
// Build ExperimentConfig from StrategyDef + sweep parameters
// ============================================================================

ExperimentConfig make_sweep_config(
    const StrategyDef& strat,
    const BehaviorDef& behav,
    const PathDef& path_def,
    const ObsCountDef& obs_def
) {
    ExperimentConfig cfg;
    cfg.mpc.horizon = HORIZON;
    cfg.mpc.sampling.num_scenarios = NUM_SCENARIOS;
    cfg.rollout.rollout_steps = ROLLOUT_STEPS;
    cfg.mpc.ego.num_discs = 1;
    cfg.mpc.ego.length = 1.5;
    cfg.environment.path_completion_termination = true;
    cfg.environment.path_completion_fraction = PATH_COMPLETE_FRAC;
    cfg.dro.enabled = (strat.enable_dro);
    cfg.dro.injection_mode = strat.injection_mode;
    cfg.dro.solver.base_radius = DRO_RHO;
    cfg.mpc.safe_horizon_enabled = strat.safe_horizon;
    cfg.mpc.constraints.safe_horizon_min = 3;
    cfg.rollout.method_name = strat.name;

    // Behavior settings
    cfg.obstacles.obs_modes = behav.modes;
    cfg.obstacles.rare_mode = behav.rare_mode;
    cfg.obstacles.rare_switch_prob = behav.rare_prob;
    cfg.obstacles.switch_prob = behav.switch_prob;

    // Path and obstacle settings
    cfg.environment.custom_ref_path = path_def.path;
    cfg.environment.custom_initial_ego = path_def.initial_ego;
    cfg.obstacles.num_obstacles = obs_def.num_obstacles;
    cfg.obstacles.obstacles_per_class = obs_def.obstacles_per_class;
    cfg.obstacles.obs_arc_fractions = obs_def.arc_fractions;

    return cfg;
}

// ============================================================================
// Summary Record
// ============================================================================

struct SweepSummary {
    std::string path_shape;
    int num_obstacles;
    std::string behavior;
    std::string variant;
    int num_rollouts;
    double collision_rate, collision_ci_lo, collision_ci_hi;
    double mode_coverage, mode_cov_ci_lo, mode_cov_ci_hi;
    double mean_clearance_5pct, mean_progress, mean_solve_ms, completion_rate;
    double rare_coverage, rare_cov_ci_lo, rare_cov_ci_hi;
};

// ============================================================================
// Run a Single Sweep Condition
// ============================================================================

SweepSummary run_condition(
    const StrategyDef& strat,
    const PathDef& path_def,
    const ObsCountDef& obs_def,
    const BehaviorDef& behav_def,
    int num_rollouts,
    std::ofstream& raw_csv
) {
    int collisions = 0, completions = 0;
    int missed_mode_total = 0, mode_check_total = 0;
    int rare_active_total = 0, rare_missed_total = 0;
    std::vector<double> clearances_5pct, progresses, solve_times;

    for (int i = 0; i < num_rollouts; ++i) {
        unsigned seed = static_cast<unsigned>(42 + i * 7919);

        ExperimentConfig cfg = make_sweep_config(strat, behav_def, path_def, obs_def);
        RolloutRecord rec = run_experiment_rollout(cfg, seed);

        if (rec.collision) collisions++;
        if (rec.completed_path) completions++;
        missed_mode_total += rec.missed_mode_steps;
        mode_check_total += rec.total_mode_checks;
        rare_active_total += rec.rare_mode_active;
        rare_missed_total += rec.rare_mode_missed;
        clearances_5pct.push_back(rec.clearance_5pct);
        progresses.push_back(rec.total_progress);
        solve_times.push_back(rec.avg_solve_ms);

        raw_csv << path_def.name << ","
                << obs_def.num_obstacles << ","
                << behav_def.name << ","
                << strat.name << ","
                << seed << ","
                << (rec.collision ? 1 : 0) << ","
                << std::fixed << std::setprecision(4)
                << rec.min_clearance << ","
                << rec.total_progress << ","
                << (rec.completed_path ? 1 : 0) << ","
                << rec.missed_mode_steps << ","
                << rec.total_mode_checks << ","
                << rec.avg_solve_ms << ","
                << rec.rare_mode_active << ","
                << rec.rare_mode_missed << ","
                << rec.clearance_5pct << "\n";
    }

    SweepSummary s;
    s.path_shape = path_def.name;
    s.num_obstacles = obs_def.num_obstacles;
    s.behavior = behav_def.name;
    s.variant = strat.name;
    s.num_rollouts = num_rollouts;

    s.collision_rate = static_cast<double>(collisions) / num_rollouts;
    auto [coll_lo, coll_hi] = wilson_ci(collisions, num_rollouts);
    s.collision_ci_lo = coll_lo;
    s.collision_ci_hi = coll_hi;

    int mode_covered = mode_check_total - missed_mode_total;
    s.mode_coverage = mode_check_total > 0
        ? static_cast<double>(mode_covered) / mode_check_total : 1.0;
    auto [mc_lo, mc_hi] = wilson_ci(mode_covered, mode_check_total);
    s.mode_cov_ci_lo = mc_lo;
    s.mode_cov_ci_hi = mc_hi;

    s.mean_clearance_5pct = clearances_5pct.empty() ? 0.0
        : std::accumulate(clearances_5pct.begin(), clearances_5pct.end(), 0.0) / clearances_5pct.size();
    s.mean_progress = progresses.empty() ? 0.0
        : std::accumulate(progresses.begin(), progresses.end(), 0.0) / progresses.size();
    s.mean_solve_ms = solve_times.empty() ? 0.0
        : std::accumulate(solve_times.begin(), solve_times.end(), 0.0) / solve_times.size();
    s.completion_rate = static_cast<double>(completions) / num_rollouts;

    int rare_covered = rare_active_total - rare_missed_total;
    s.rare_coverage = rare_active_total > 0
        ? static_cast<double>(rare_covered) / rare_active_total : 1.0;
    if (rare_active_total > 0) {
        auto [rc_lo, rc_hi] = wilson_ci(rare_covered, rare_active_total);
        s.rare_cov_ci_lo = rc_lo;
        s.rare_cov_ci_hi = rc_hi;
    } else {
        s.rare_cov_ci_lo = 1.0;
        s.rare_cov_ci_hi = 1.0;
    }

    return s;
}

// ============================================================================
// CSV Writers
// ============================================================================

void write_summary_header(std::ofstream& f) {
    f << "path_shape,num_obstacles,behavior,variant,num_rollouts,"
      << "collision_rate,collision_ci_lo,collision_ci_hi,"
      << "mode_coverage,mode_cov_ci_lo,mode_cov_ci_hi,"
      << "mean_clearance_5pct,mean_progress,mean_solve_ms,completion_rate,"
      << "rare_coverage,rare_cov_ci_lo,rare_cov_ci_hi\n";
}

void write_summary_row(std::ofstream& f, const SweepSummary& s) {
    f << s.path_shape << ","
      << s.num_obstacles << ","
      << s.behavior << ","
      << s.variant << ","
      << s.num_rollouts << ","
      << std::fixed << std::setprecision(4)
      << s.collision_rate << "," << s.collision_ci_lo << "," << s.collision_ci_hi << ","
      << s.mode_coverage << "," << s.mode_cov_ci_lo << "," << s.mode_cov_ci_hi << ","
      << s.mean_clearance_5pct << "," << s.mean_progress << ","
      << s.mean_solve_ms << "," << s.completion_rate << ","
      << s.rare_coverage << "," << s.rare_cov_ci_lo << "," << s.rare_cov_ci_hi << "\n";
}

void write_raw_header(std::ofstream& f) {
    f << "path_shape,num_obstacles,behavior,variant,seed,"
      << "collision,min_clearance,total_progress,completed_path,"
      << "missed_mode_steps,total_mode_checks,avg_solve_ms,"
      << "rare_mode_active,rare_mode_missed,clearance_5pct\n";
}

// ============================================================================
// Sweep Runners
// ============================================================================

void run_path_sweep(
    const std::vector<PathDef>& paths,
    std::ofstream& summary_csv, std::ofstream& raw_csv
) {
    ObsCountDef obs_def = {2, 1, {0.30, 0.55}};
    BehaviorDef behav_def = {"balanced",
        {"constant_velocity", "turn_left", "turn_right", "decelerating"},
        "", 0.0, 0.1};

    int total = static_cast<int>(paths.size() * STRATEGIES.size());
    int done = 0;

    for (const auto& path_def : paths) {
        for (const auto& strat : STRATEGIES) {
            done++;
            std::cout << "  [" << done << "/" << total << "] "
                      << path_def.name << " / " << strat.name
                      << " (" << ROLLOUTS_PER_CONDITION << " rollouts)..." << std::flush;

            auto s = run_condition(strat, path_def, obs_def, behav_def,
                                   ROLLOUTS_PER_CONDITION, raw_csv);
            write_summary_row(summary_csv, s);
            summary_csv.flush(); raw_csv.flush();

            std::cout << " coll=" << std::fixed << std::setprecision(1)
                      << s.collision_rate * 100 << "% ["
                      << s.collision_ci_lo * 100 << "," << s.collision_ci_hi * 100
                      << "] mode_cov=" << s.mode_coverage * 100 << "%\n";
        }
    }
}

void run_obstacle_sweep(
    const std::vector<ObsCountDef>& obs_defs,
    std::ofstream& summary_csv, std::ofstream& raw_csv
) {
    PathDef path_def = {"s_curve",
        ReferencePath::create_s_curve(25.0, 3.0, 200),
        EgoState(0.0, 0.0, 0.0, 1.5)};
    BehaviorDef behav_def = {"balanced",
        {"constant_velocity", "turn_left", "turn_right", "decelerating"},
        "", 0.0, 0.1};

    int total = static_cast<int>(obs_defs.size() * STRATEGIES.size());
    int done = 0;

    for (const auto& obs_def : obs_defs) {
        for (const auto& strat : STRATEGIES) {
            done++;
            std::cout << "  [" << done << "/" << total << "] "
                      << obs_def.num_obstacles << " obs / " << strat.name
                      << " (" << ROLLOUTS_PER_CONDITION << " rollouts)..." << std::flush;

            auto s = run_condition(strat, path_def, obs_def, behav_def,
                                   ROLLOUTS_PER_CONDITION, raw_csv);
            write_summary_row(summary_csv, s);
            summary_csv.flush(); raw_csv.flush();

            std::cout << " coll=" << std::fixed << std::setprecision(1)
                      << s.collision_rate * 100 << "% mode_cov="
                      << s.mode_coverage * 100 << "%\n";
        }
    }
}

void run_behavior_sweep(
    const std::vector<BehaviorDef>& behaviors,
    std::ofstream& summary_csv, std::ofstream& raw_csv
) {
    PathDef path_def = {"s_curve",
        ReferencePath::create_s_curve(25.0, 3.0, 200),
        EgoState(0.0, 0.0, 0.0, 1.5)};
    ObsCountDef obs_def = {2, 1, {0.30, 0.55}};

    int total = static_cast<int>(behaviors.size() * STRATEGIES.size());
    int done = 0;

    for (const auto& behav_def : behaviors) {
        for (const auto& strat : STRATEGIES) {
            done++;
            std::cout << "  [" << done << "/" << total << "] "
                      << behav_def.name << " / " << strat.name
                      << " (" << ROLLOUTS_PER_CONDITION << " rollouts)..." << std::flush;

            auto s = run_condition(strat, path_def, obs_def, behav_def,
                                   ROLLOUTS_PER_CONDITION, raw_csv);
            write_summary_row(summary_csv, s);
            summary_csv.flush(); raw_csv.flush();

            std::cout << " coll=" << std::fixed << std::setprecision(1)
                      << s.collision_rate * 100 << "% mode_cov="
                      << s.mode_coverage * 100 << "%\n";
        }
    }
}

void run_full_sweep(
    const std::vector<PathDef>& paths,
    const std::vector<ObsCountDef>& obs_defs,
    const std::vector<BehaviorDef>& behaviors,
    std::ofstream& summary_csv, std::ofstream& raw_csv
) {
    int total = static_cast<int>(paths.size() * obs_defs.size()
                                 * behaviors.size() * STRATEGIES.size());
    int done = 0;

    for (const auto& path_def : paths) {
        for (const auto& obs_def : obs_defs) {
            for (const auto& behav_def : behaviors) {
                for (const auto& strat : STRATEGIES) {
                    done++;
                    std::cout << "  [" << done << "/" << total << "] "
                              << path_def.name << " / " << obs_def.num_obstacles
                              << "obs / " << behav_def.name << " / "
                              << strat.name << "..." << std::flush;

                    auto s = run_condition(strat, path_def, obs_def, behav_def,
                                           ROLLOUTS_PER_CONDITION, raw_csv);
                    write_summary_row(summary_csv, s);
                    summary_csv.flush(); raw_csv.flush();

                    std::cout << " coll=" << std::fixed << std::setprecision(1)
                              << s.collision_rate * 100 << "%\n";
                }
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string filter = "all";
    if (argc > 1) filter = argv[1];

    std::cout << "=== Path / Obstacle / Behavior Sweep (CDC Strategies) ===\n"
              << "Filter: " << filter << "\n"
              << "Rollouts per condition: " << ROLLOUTS_PER_CONDITION << "\n"
              << "Strategies:";
    for (const auto& s : STRATEGIES) std::cout << " " << s.name;
    std::cout << "\n\n";

    auto paths = create_path_defs();
    auto obs_defs = create_obs_count_defs();
    auto behaviors = create_behavior_defs();

    std::cout << "Path shapes:\n";
    for (const auto& p : paths) {
        std::cout << "  " << p.name << ": length=" << std::fixed
                  << std::setprecision(1) << p.path.total_length() << "m\n";
    }
    std::cout << "\n";

    std::ofstream summary_csv(OUT_DIR + "path_obstacle_sweep.csv");
    std::ofstream raw_csv(OUT_DIR + "path_obstacle_raw.csv");
    write_summary_header(summary_csv);
    write_raw_header(raw_csv);

    if (filter == "paths" || filter == "all") {
        std::cout << "--- Path Shape Sweep (2 obs, balanced) ---\n";
        run_path_sweep(paths, summary_csv, raw_csv);
        std::cout << "\n";
    }

    if (filter == "obstacles" || filter == "all") {
        std::cout << "--- Obstacle Count Sweep (s_curve, balanced) ---\n";
        run_obstacle_sweep(obs_defs, summary_csv, raw_csv);
        std::cout << "\n";
    }

    if (filter == "behaviors" || filter == "all") {
        std::cout << "--- Behavior Profile Sweep (s_curve, 2 obs) ---\n";
        run_behavior_sweep(behaviors, summary_csv, raw_csv);
        std::cout << "\n";
    }

    if (filter == "full") {
        std::cout << "--- Full Cross-Product Sweep ---\n";
        int total_conditions = static_cast<int>(paths.size() * obs_defs.size()
                                                * behaviors.size() * STRATEGIES.size());
        std::cout << "Total conditions: " << total_conditions
                  << " (" << total_conditions * ROLLOUTS_PER_CONDITION << " rollouts)\n";
        run_full_sweep(paths, obs_defs, behaviors, summary_csv, raw_csv);
        std::cout << "\n";
    }

    summary_csv.close();
    raw_csv.close();

    std::cout << "=== Done ===\n"
              << "Summary: " << OUT_DIR << "path_obstacle_sweep.csv\n"
              << "Raw:     " << OUT_DIR << "path_obstacle_raw.csv\n"
              << "\nGenerate figures with:\n"
              << "  python3 scripts/generate_path_obstacle_figures.py\n";

    return 0;
}
