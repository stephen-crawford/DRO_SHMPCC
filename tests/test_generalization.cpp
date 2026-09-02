/**
 * @file test_generalization.cpp
 * @brief Generalization experiments: path geometry, multi-obstacle, switching dynamics.
 *
 * Extends CDC paper experiments beyond S-curve / single-obstacle / fixed switching:
 *   G1: Path geometry comparison (straight, circle, tight S-curve, gentle S-curve)
 *   G2: Multi-obstacle scaling (1, 2, 3 obstacles)
 *   G3: Switching dynamics sweep (switch_prob from 0.05 to 0.5)
 *   G4: Mode diversity (3, 4, 5, 6 available modes)
 *   G5: Path × obstacle interaction (2D grid)
 *
 * Output: generalization_figures/
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
#include <set>
#include <filesystem>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"
#include "reference_path.hpp"
#include "dynamics.hpp"

using namespace dro_mpc;
namespace fs = std::filesystem;

// ============================================================================
// Constants
// ============================================================================

static const std::string OUTPUT_DIR = "generalization_figures/";

static const int    HORIZON = DEFAULT_HORIZON;
static const double    DT = DEFAULT_DT;
static const int    NUM_SCENARIOS = DEFAULT_BASE_SCENARIOS;
static const int    ROLLOUT_STEPS = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;
static constexpr int    N_ROLLOUTS      = 500;

static const std::vector<std::string> BASE_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Metrics (same as CDC tests)
// ============================================================================

struct Metrics {
    std::string method;
    std::vector<bool> collisions;
    std::vector<int> missed_mode_steps;
    std::vector<int> total_mode_checks;
    std::vector<int> rare_mode_active;
    std::vector<int> rare_mode_missed;
    std::vector<double> min_clearances;
    std::vector<double> total_progress;
    std::vector<double> solve_times_ms;

    int n() const { return static_cast<int>(collisions.size()); }
    int coll_count() const { int c = 0; for (bool b : collisions) if (b) c++; return c; }
    double coll_rate() const { return n() > 0 ? double(coll_count()) / n() : 0; }
    std::pair<double,double> coll_ci() const { return wilson_ci(coll_count(), n()); }

    int tot_missed() const { return std::accumulate(missed_mode_steps.begin(), missed_mode_steps.end(), 0); }
    int tot_checks() const { return std::accumulate(total_mode_checks.begin(), total_mode_checks.end(), 0); }
    double mm_rate() const { int tc = tot_checks(); return tc > 0 ? double(tot_missed()) / tc : 0; }
    std::pair<double,double> mm_ci() const { return wilson_ci(tot_missed(), tot_checks()); }

    int tot_rare_act() const { return std::accumulate(rare_mode_active.begin(), rare_mode_active.end(), 0); }
    int tot_rare_miss() const { return std::accumulate(rare_mode_missed.begin(), rare_mode_missed.end(), 0); }
    double rare_rate() const { int a = tot_rare_act(); return a > 0 ? double(tot_rare_miss()) / a : 0; }
    std::pair<double,double> rare_ci() const { return wilson_ci(tot_rare_miss(), tot_rare_act()); }

    double mean_clearance() const {
        if (min_clearances.empty()) return 0;
        return std::accumulate(min_clearances.begin(), min_clearances.end(), 0.0) / min_clearances.size();
    }
    double p5_clearance() const {
        if (min_clearances.empty()) return 0;
        auto v = min_clearances;
        std::sort(v.begin(), v.end());
        return v[std::max(0, (int)(0.05 * (v.size() - 1)))];
    }
    double mean_progress() const {
        if (total_progress.empty()) return 0;
        return std::accumulate(total_progress.begin(), total_progress.end(), 0.0) / total_progress.size();
    }
    double mean_solve_ms() const {
        if (solve_times_ms.empty()) return 0;
        return std::accumulate(solve_times_ms.begin(), solve_times_ms.end(), 0.0) / solve_times_ms.size();
    }

    void add(const RolloutRecord& rec) {
        collisions.push_back(rec.collision);
        missed_mode_steps.push_back(rec.missed_mode_steps);
        total_mode_checks.push_back(rec.total_mode_checks);
        rare_mode_active.push_back(rec.rare_mode_active);
        rare_mode_missed.push_back(rec.rare_mode_missed);
        min_clearances.push_back(rec.min_clearance);
        total_progress.push_back(rec.total_progress);
        solve_times_ms.push_back(rec.avg_solve_ms);
    }
};

// ============================================================================
// Helpers
// ============================================================================

static double elapsed_sec(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct MethodDef {
    std::string name;
    bool enable_dro;
    InjectionMode injection_mode;
    double softmax_tau;
    double eps_greedy_epsilon;
    int injection_count;
};

static const std::vector<MethodDef> CORE_METHODS = {
    {"Base",              false, InjectionMode::NONE,               0.0, 0.0, 1},
    {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,      0.0, 0.0, 1},
    {"WDRO-inject-K1",    true,  InjectionMode::TOP_RISK_INJECT,               0.0, 0.0, 1},
    {"WDRO-inject-K2",    true,  InjectionMode::TOP_RISK_INJECT,               0.0, 0.0, 2},
    {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,   0.0, 0.0, 1},
    {"DiverseRisk-K1",    true,  InjectionMode::DIVERSE_RISK_INJECT, 0.0, 0.0, 1},
    {"Softmax-tau5",      true,  InjectionMode::SOFTMAX_RISK,      5.0, 0.0, 1},
};

static ExperimentConfig make_base_config(double switch_prob = 0.2, double rare_prob = 0.1,
                                          int num_obs = 1) {
    ExperimentConfig cfg;
    cfg.mpc.horizon = HORIZON;
    cfg.mpc.sampling.num_scenarios = NUM_SCENARIOS;
    cfg.obstacles.switch_prob = switch_prob;
    cfg.rollout.rollout_steps = ROLLOUT_STEPS;
    cfg.obstacles.obs_modes = BASE_MODES;
    cfg.obstacles.rare_mode = RARE_MODE;
    cfg.obstacles.rare_switch_prob = rare_prob;
    cfg.mpc.ego.num_discs = NUM_DISCS;
    cfg.mpc.ego.length = VEHICLE_LENGTH;
    cfg.mpc.safe_horizon_enabled = true;
    cfg.mpc.constraints.safe_horizon_min = SAFE_HORIZON_MIN;
    cfg.environment.path_completion_termination = true;
    cfg.environment.path_completion_fraction = 0.95;
    cfg.dro.enabled = false;
    cfg.obstacles.num_obstacles = num_obs;
    cfg.obstacles.obstacles_per_class = 1;
    return cfg;
}

static ExperimentConfig apply_method(ExperimentConfig cfg, const MethodDef& m) {
    cfg.dro.enabled = (m.enable_dro);
    cfg.rollout.method_name = m.name;
    return cfg;
}

static void write_csv_row(std::ofstream& ofs, const Metrics& met, int n_rollouts,
                           const std::string& extra_prefix = "") {
    auto [clo, chi] = met.coll_ci();
    auto [mlo, mhi] = met.mm_ci();
    auto [rlo, rhi] = met.rare_ci();
    ofs << extra_prefix
        << met.method << ","
        << std::fixed << std::setprecision(6)
        << met.coll_rate() << "," << clo << "," << chi << ","
        << met.mm_rate() << "," << mlo << "," << mhi << ","
        << met.rare_rate() << "," << rlo << "," << rhi << ","
        << std::setprecision(4) << met.mean_clearance() << ","
        << met.p5_clearance() << "," << met.mean_progress() << ","
        << std::setprecision(3) << met.mean_solve_ms() << ","
        << n_rollouts << "\n";
    ofs.flush();
}

// ============================================================================
// Path geometry definitions
// ============================================================================

struct PathSetup {
    std::string name;
    ReferencePath path;
    EgoState initial_ego;
};

static PathSetup make_straight_path() {
    auto path = ReferencePath::create_straight(
        Eigen::Vector2d(0, 0), Eigen::Vector2d(25, 0), 200);
    return {"Straight", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_gentle_scurve() {
    // Low amplitude = gentle curves
    auto path = ReferencePath::create_s_curve(25.0, 1.0, 200);
    return {"Gentle-S", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_standard_scurve() {
    // Default paper S-curve
    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);
    return {"S-curve", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_tight_scurve() {
    // High amplitude, shorter length = tight curves
    auto path = ReferencePath::create_s_curve(20.0, 5.0, 200);
    return {"Tight-S", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_circle_path() {
    // Quarter-circle arc, radius 10m
    double radius = 10.0;
    auto path = ReferencePath::create_circle(
        Eigen::Vector2d(0, -radius), radius, M_PI / 2, 0.0, 200);
    // Ego starts at top of circle, heading right (tangent)
    Eigen::Vector2d start_pos = path.get_position_at(0.0);
    double start_heading = path.get_heading_at(0.0);
    return {"Circle", path, EgoState(start_pos.x(), start_pos.y(), start_heading, 1.5)};
}

/// Place an obstacle along any reference path at given arc-length fraction.
static ObstacleState obstacle_on_path(const ReferencePath& path, double arc_fraction,
                                       std::mt19937& rng, bool oncoming = true) {
    double obs_s = arc_fraction * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d tangent(std::cos(pp.heading), std::sin(pp.heading));

    std::uniform_real_distribution<double> lat(-0.3, 0.3);
    std::uniform_real_distribution<double> spd(-0.2, 0.1);
    Eigen::Vector2d pos = pp.position + lat(rng) * normal;
    double v = 0.8 + spd(rng);
    double dir = oncoming ? -1.0 : 1.0;
    return ObstacleState(pos.x(), pos.y(), dir * v * tangent.x(), dir * v * tangent.y());
}

// ============================================================================
// G1: Path Geometry Comparison
// ============================================================================

static void run_g1() {
    std::cout << "\n================================================================\n";
    std::cout << "  G1: Path Geometry Comparison\n";
    std::cout << "  Paths: Straight, Gentle-S, S-curve, Tight-S, Circle\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    std::vector<PathSetup> paths = {
        make_straight_path(),
        make_gentle_scurve(),
        make_standard_scurve(),
        make_tight_scurve(),
        make_circle_path(),
    };

    const int N = N_ROLLOUTS;

    std::string filepath = OUTPUT_DIR + "g1_path_geometry.csv";
    std::ofstream ofs(filepath);
    ofs << "path,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& ps : paths) {
        for (const auto& method : CORE_METHODS) {
            std::cout << "  " << ps.name << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(ps.name) / 1000 +
                    static_cast<int>(method.injection_mode) * 100000 + i);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);
                cfg.environment.custom_ref_path = ps.path;
                cfg.environment.custom_initial_ego = ps.initial_ego;
                cfg.obstacles.initial_obstacle_states = {
                    obstacle_on_path(ps.path, 0.55, env_rng, true)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << ps.name << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " prog=" << std::setprecision(2) << met.mean_progress()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// G2: Multi-Obstacle Scaling
// ============================================================================

static void run_g2() {
    std::cout << "\n================================================================\n";
    std::cout << "  G2: Multi-Obstacle Scaling (1, 2, 3 obstacles)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> OBS_COUNTS = {1, 2, 3};
    // Fractions along S-curve for obstacle placement
    const std::vector<std::vector<double>> OBS_FRACS = {
        {0.50},
        {0.40, 0.60},
        {0.30, 0.45, 0.65},
    };

    const int N = N_ROLLOUTS;

    std::string filepath = OUTPUT_DIR + "g2_multi_obstacle.csv";
    std::ofstream ofs(filepath);
    ofs << "num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    for (size_t oi = 0; oi < OBS_COUNTS.size(); ++oi) {
        int n_obs = OBS_COUNTS[oi];
        const auto& fracs = OBS_FRACS[oi];

        for (const auto& method : CORE_METHODS) {
            std::cout << "  " << n_obs << " obs / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(n_obs * 1000000 +
                    static_cast<int>(method.injection_mode) * 100000 + i + 20000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, 0.1, n_obs);
                cfg = apply_method(cfg, method);

                std::vector<ObstacleState> obs_states;
                for (double frac : fracs) {
                    obs_states.push_back(obstacle_on_path(ref_path, frac, env_rng, true));
                }
                cfg.obstacles.initial_obstacle_states = obs_states;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << n_obs << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// G3: Switching Dynamics Sweep
// ============================================================================

static void run_g3() {
    std::cout << "\n================================================================\n";
    std::cout << "  G3: Switching Dynamics Sweep (switch_prob)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> SWITCH_PROBS = {0.02, 0.05, 0.10, 0.20, 0.35, 0.50};
    const int N = N_ROLLOUTS;

    // Subset of methods for readability
    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,               0.0, 0.0, 1},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,      0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::TOP_RISK_INJECT,               0.0, 0.0, 1},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,   0.0, 0.0, 1},
        {"Softmax-tau5",      true,  InjectionMode::SOFTMAX_RISK,      5.0, 0.0, 1},
    };

    std::string filepath = OUTPUT_DIR + "g3_switch_dynamics.csv";
    std::ofstream ofs(filepath);
    ofs << "switch_prob,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    for (double sp : SWITCH_PROBS) {
        for (const auto& method : METHODS) {
            std::cout << "  sp=" << std::setprecision(2) << sp << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(sp * 1000) * 1000000 +
                    static_cast<int>(method.injection_mode) * 100000 + i + 30000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(sp, 0.1);
                cfg = apply_method(cfg, method);
                cfg.obstacles.initial_obstacle_states = {
                    obstacle_on_path(ref_path, 0.55, env_rng, true)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(2) << sp << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// G4: Mode Diversity (vary number of available modes)
// ============================================================================

static void run_g4() {
    std::cout << "\n================================================================\n";
    std::cout << "  G4: Mode Diversity (3, 4, 5, 6 modes)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    struct ModeSet {
        std::string label;
        std::vector<std::string> modes;
        std::string rare;
    };

    const std::vector<ModeSet> MODE_SETS = {
        {"3-modes", {"constant_velocity", "turn_left", "turn_right"}, "decelerating"},
        {"4-modes", {"constant_velocity", "turn_left", "turn_right", "decelerating"}, "lane_change_left"},
        {"5-modes", {"constant_velocity", "turn_left", "turn_right", "decelerating", "lane_change_left"}, "lane_change_right"},
        {"6-modes", {"constant_velocity", "turn_left", "turn_right", "decelerating", "lane_change_left", "lane_change_right"}, ""},
    };

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,               0.0, 0.0, 1},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,      0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::TOP_RISK_INJECT,               0.0, 0.0, 1},
        {"WDRO-inject-K2",    true,  InjectionMode::TOP_RISK_INJECT,               0.0, 0.0, 2},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,   0.0, 0.0, 1},
        {"DiverseRisk-K1",    true,  InjectionMode::DIVERSE_RISK_INJECT, 0.0, 0.0, 1},
    };

    const int N = N_ROLLOUTS;

    std::string filepath = OUTPUT_DIR + "g4_mode_diversity.csv";
    std::ofstream ofs(filepath);
    ofs << "mode_set,num_modes,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    for (const auto& ms : MODE_SETS) {
        int total_modes = static_cast<int>(ms.modes.size()) + (ms.rare.empty() ? 0 : 1);
        for (const auto& method : METHODS) {
            std::cout << "  " << ms.label << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    total_modes * 1000000 +
                    static_cast<int>(method.injection_mode) * 100000 + i + 40000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, ms.rare.empty() ? 0.0 : 0.1);
                cfg = apply_method(cfg, method);
                cfg.obstacles.obs_modes = ms.modes;
                cfg.obstacles.rare_mode = ms.rare;
                cfg.obstacles.initial_obstacle_states = {
                    obstacle_on_path(ref_path, 0.55, env_rng, true)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << ms.label << "," << total_modes << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// G5: Path × Obstacle Interaction (2D grid)
// ============================================================================

static void run_g5() {
    std::cout << "\n================================================================\n";
    std::cout << "  G5: Path x Obstacle Interaction\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    std::vector<PathSetup> paths = {
        make_straight_path(),
        make_standard_scurve(),
        make_tight_scurve(),
        make_circle_path(),
    };

    const std::vector<int> OBS_COUNTS = {1, 2, 3};
    const std::vector<std::vector<double>> OBS_FRACS = {
        {0.50},
        {0.40, 0.60},
        {0.30, 0.45, 0.65},
    };

    // Compare only Base vs WDRO-injection to keep the grid manageable
    const std::vector<MethodDef> METHODS = {
        {"Base",            false, InjectionMode::NONE,           0.0, 0.0, 1},
        {"WDRO-sampling",   true,  InjectionMode::QSTAR_SAMPLE,  0.0, 0.0, 1},
        {"WDRO-inject-K1",  true,  InjectionMode::TOP_RISK_INJECT,           0.0, 0.0, 1},
    };

    const int N = N_ROLLOUTS;

    std::string filepath = OUTPUT_DIR + "g5_path_obstacle_grid.csv";
    std::ofstream ofs(filepath);
    ofs << "path,num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& ps : paths) {
        for (size_t oi = 0; oi < OBS_COUNTS.size(); ++oi) {
            int n_obs = OBS_COUNTS[oi];
            const auto& fracs = OBS_FRACS[oi];

            for (const auto& method : METHODS) {
                std::cout << "  " << ps.name << "/" << n_obs << "obs / " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();

                Metrics met;
                met.method = method.name;

                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        std::hash<std::string>{}(ps.name) / 1000 +
                        n_obs * 1000000 +
                        static_cast<int>(method.injection_mode) * 100000 + i + 50000000);
                    std::mt19937 env_rng(seed);

                    ExperimentConfig cfg = make_base_config(0.2, 0.1, n_obs);
                    cfg = apply_method(cfg, method);
                    cfg.environment.custom_ref_path = ps.path;
                    cfg.environment.custom_initial_ego = ps.initial_ego;

                    std::vector<ObstacleState> obs_states;
                    for (double frac : fracs) {
                        obs_states.push_back(obstacle_on_path(ps.path, frac, env_rng, true));
                    }
                    cfg.obstacles.initial_obstacle_states = obs_states;

                    RolloutRecord rec = run_experiment_rollout(cfg, seed);
                    met.add(rec);
                }

                ofs << ps.name << "," << n_obs << ",";
                write_csv_row(ofs, met, N);

                std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                          << " mm=" << (met.mm_rate() * 100) << "%"
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    fs::create_directories(OUTPUT_DIR);

    std::set<std::string> filters;
    for (int i = 1; i < argc; ++i) filters.insert(argv[i]);
    bool run_all = filters.empty();

    std::cout << "========================================\n";
    std::cout << "Generalization Experiments\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("g1")) run_g1();
    if (run_all || filters.count("g2")) run_g2();
    if (run_all || filters.count("g3")) run_g3();
    if (run_all || filters.count("g4")) run_g4();
    if (run_all || filters.count("g5")) run_g5();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total << "s"
              << " (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";

    return 0;
}
