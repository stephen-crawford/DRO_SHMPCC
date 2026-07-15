/**
 * @file test_exhaustive_suites.cpp
 * @brief Exhaustive runs of F1, F9, G1, G3, G6 with per-suite output
 *        directories and sensitivity sweeps over nuisance parameters.
 *
 * Each suite writes to figures/<suite>/.  For robustness, every suite
 * includes an extra sweep dimension so that we can confirm the primary
 * finding holds across configurations.
 *
 * Usage:
 *   ./test_exhaustive_suites f1 f9 g1 g3 g6   (run all five)
 *   ./test_exhaustive_suites f1                  (just F1)
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
#include <functional>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"
#include "reference_path.hpp"
#include "dynamics.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

// ============================================================================
// Constants
// ============================================================================

static constexpr int    HORIZON         = DEFAULT_HORIZON;
static constexpr double DT              = DEFAULT_DT;
static constexpr int    NUM_SCENARIOS   = DEFAULT_BASE_SCENARIOS;
static constexpr int    ROLLOUT_STEPS   = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;

static const std::vector<std::string> BASE_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Metrics
// ============================================================================

struct Metrics {
    std::string method;
    std::vector<bool> collisions;
    std::vector<int> missed_mode_steps, total_mode_checks;
    std::vector<int> rare_mode_active, rare_mode_missed;
    std::vector<double> min_clearances, total_progress, solve_times_ms;

    int n() const { return static_cast<int>(collisions.size()); }
    int cc() const { int c=0; for (bool b : collisions) if (b) c++; return c; }
    double cr() const { return n()>0 ? double(cc())/n() : 0; }
    std::pair<double,double> cci() const { return wilson_ci(cc(), n()); }
    int tm() const { return std::accumulate(missed_mode_steps.begin(), missed_mode_steps.end(), 0); }
    int tc() const { return std::accumulate(total_mode_checks.begin(), total_mode_checks.end(), 0); }
    double mr() const { int t=tc(); return t>0 ? double(tm())/t : 0; }
    std::pair<double,double> mci() const { return wilson_ci(tm(), tc()); }
    int tra() const { return std::accumulate(rare_mode_active.begin(), rare_mode_active.end(), 0); }
    int trm() const { return std::accumulate(rare_mode_missed.begin(), rare_mode_missed.end(), 0); }
    double rr() const { int a=tra(); return a>0 ? double(trm())/a : 0; }
    std::pair<double,double> rci() const { return wilson_ci(trm(), tra()); }
    double mc() const { return min_clearances.empty() ? 0 : std::accumulate(min_clearances.begin(), min_clearances.end(), 0.0)/min_clearances.size(); }
    double p5() const {
        if (min_clearances.empty()) return 0;
        auto v=min_clearances; std::sort(v.begin(),v.end());
        return v[std::max(0,(int)(0.05*(v.size()-1)))];
    }
    double mp() const { return total_progress.empty() ? 0 : std::accumulate(total_progress.begin(), total_progress.end(), 0.0)/total_progress.size(); }
    double ms() const { return solve_times_ms.empty() ? 0 : std::accumulate(solve_times_ms.begin(), solve_times_ms.end(), 0.0)/solve_times_ms.size(); }

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

static double elapsed_sec(std::chrono::steady_clock::time_point s) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
}

struct MethodDef {
    std::string name;
    bool enable_dro;
    InjectionMode injection_mode;
    double softmax_tau;
    double eps_greedy_epsilon;
    int injection_count;
};

static ExperimentConfig make_base_config(double switch_prob = 0.2,
                                          double rare_prob = 0.1,
                                          int num_obs = 1) {
    ExperimentConfig cfg;
    cfg.horizon = HORIZON;
    cfg.num_scenarios = NUM_SCENARIOS;
    cfg.switch_prob = switch_prob;
    cfg.rollout_steps = ROLLOUT_STEPS;
    cfg.obs_modes = BASE_MODES;
    cfg.rare_mode = RARE_MODE;
    cfg.rare_switch_prob = rare_prob;
    cfg.num_discs = NUM_DISCS;
    cfg.vehicle_length = VEHICLE_LENGTH;
    cfg.safe_horizon_enabled = true;
    cfg.safe_horizon_min = SAFE_HORIZON_MIN;
    cfg.path_completion_termination = true;
    cfg.path_completion_fraction = 0.95;
    cfg.weight_type = WeightType::FREQUENCY;
    cfg.enable_dro = false;
    cfg.injection_mode = InjectionMode::NONE;
    cfg.ablation = AblationVariant::NO_INJECTION;
    cfg.num_obstacles = num_obs;
    cfg.obstacles_per_class = 1;
    return cfg;
}

static ExperimentConfig apply_method(ExperimentConfig cfg, const MethodDef& m) {
    cfg.enable_dro = m.enable_dro;
    cfg.injection_mode = m.injection_mode;
    cfg.softmax_tau = m.softmax_tau;
    cfg.eps_greedy_epsilon = m.eps_greedy_epsilon;
    cfg.dro_injection_count = m.injection_count;
    cfg.method_name = m.name;
    return cfg;
}

static void write_csv_row(std::ofstream& ofs, const Metrics& m, int n,
                           const std::string& prefix = "") {
    auto [clo, chi] = m.cci();
    auto [mlo, mhi] = m.mci();
    auto [rlo, rhi] = m.rci();
    ofs << prefix << m.method << ","
        << std::fixed << std::setprecision(6)
        << m.cr() << "," << clo << "," << chi << ","
        << m.mr() << "," << mlo << "," << mhi << ","
        << m.rr() << "," << rlo << "," << rhi << ","
        << std::setprecision(4) << m.mc() << "," << m.p5() << ","
        << m.mp() << ","
        << std::setprecision(3) << m.ms() << "," << n << "\n";
    ofs.flush();
}

// ============================================================================
// Path helpers
// ============================================================================

struct PathSetup {
    std::string name;
    ReferencePath path;
    EgoState initial_ego;
};

static PathSetup make_straight_path() {
    auto p = ReferencePath::create_straight(
        Eigen::Vector2d(0, 0), Eigen::Vector2d(25, 0), 200);
    return {"Straight", p, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_gentle_scurve() {
    auto p = ReferencePath::create_s_curve(25.0, 1.0, 200);
    return {"Gentle-S", p, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_standard_scurve() {
    auto p = ReferencePath::create_s_curve(25.0, 3.0, 200);
    return {"S-curve", p, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_tight_scurve() {
    auto p = ReferencePath::create_s_curve(20.0, 5.0, 200);
    return {"Tight-S", p, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_circle_path() {
    double radius = 10.0;
    auto p = ReferencePath::create_circle(
        Eigen::Vector2d(0, -radius), radius, M_PI / 2, 0.0, 200);
    Eigen::Vector2d sp = p.get_position_at(0.0);
    double sh = p.get_heading_at(0.0);
    return {"Circle", p, EgoState(sp.x(), sp.y(), sh, 1.5)};
}

// ============================================================================
// Obstacle helpers
// ============================================================================

static ObstacleState obs_oncoming(const ReferencePath& path, double frac,
                                   std::mt19937& rng, double speed = 0.8) {
    double s = frac * path.total_length();
    PathPoint pp = path.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
    std::uniform_real_distribution<double> lat(-0.3, 0.3), spd(-0.1, 0.1);
    Eigen::Vector2d pos = pp.position + lat(rng) * n;
    double v = speed + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -v * t.x(), -v * t.y());
}

static ObstacleState obs_high_speed(const ReferencePath& path, double frac,
                                     std::mt19937& rng, double speed = 1.8) {
    return obs_oncoming(path, frac, rng, speed);
}

static ObstacleState obs_crossing(const ReferencePath& path, double frac,
                                   std::mt19937& rng, double speed = 1.0) {
    double s = frac * path.total_length();
    PathPoint pp = path.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
    std::uniform_real_distribution<double> off(1.5, 3.0), spd(-0.1, 0.1);
    double side = (rng() % 2 == 0) ? 1.0 : -1.0;
    Eigen::Vector2d pos = pp.position + side * off(rng) * n;
    double v = speed + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -side * v * n.x(), -side * v * n.y());
}

static ObstacleState obs_diagonal(const ReferencePath& path, double frac,
                                   std::mt19937& rng, double speed = 1.0) {
    double s = frac * path.total_length();
    PathPoint pp = path.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
    std::uniform_real_distribution<double> lat(1.0, 2.0), spd(-0.1, 0.1);
    double side = (rng() % 2 == 0) ? 1.0 : -1.0;
    Eigen::Vector2d pos = pp.position + side * lat(rng) * n + 2.0 * t;
    double v = speed + spd(rng);
    Eigen::Vector2d d = (-t + (-side) * n).normalized();
    return ObstacleState(pos.x(), pos.y(), v * d.x(), v * d.y());
}

// ============================================================================
// Method sets
// ============================================================================

static const std::vector<MethodDef> ALL_METHODS = {
    {"Base",              false, InjectionMode::NONE,                  0.0, 0.0, 1},
    {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,         0.0, 0.0, 1},
    {"WDRO-inject-K1",    true,  InjectionMode::DRO,                  0.0, 0.0, 1},
    {"WDRO-inject-K2",    true,  InjectionMode::DRO,                  0.0, 0.0, 2},
    {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,      0.0, 0.0, 1},
    {"TopRisk-K2",        true,  InjectionMode::TOP_RISK_INJECT,      0.0, 0.0, 2},
    {"DiverseRisk-K1",    true,  InjectionMode::DIVERSE_RISK_INJECT,  0.0, 0.0, 1},
    {"Softmax-tau5",      true,  InjectionMode::SOFTMAX_RISK,         5.0, 0.0, 1},
};

static const std::vector<MethodDef> CORE6_METHODS = {
    {"Base",              false, InjectionMode::NONE,                  0.0, 0.0, 1},
    {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,         0.0, 0.0, 1},
    {"WDRO-inject-K1",    true,  InjectionMode::DRO,                  0.0, 0.0, 1},
    {"WDRO-inject-K2",    true,  InjectionMode::DRO,                  0.0, 0.0, 2},
    {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,      0.0, 0.0, 1},
    {"TopRisk-K2",        true,  InjectionMode::TOP_RISK_INJECT,      0.0, 0.0, 2},
};

// ============================================================================
// F1: High-N Variant Differentiation
//     Primary: 8 methods × 2000 rollouts on oncoming.
//     Sensitivity: obstacle speed {0.8, 1.2} to confirm ordering holds.
// ============================================================================

static void run_f1() {
    const std::string DIR = "figures/f1/";
    fs::create_directories(DIR);
    std::cout << "\n================================================================\n";
    std::cout << "  F1: High-N Variant Differentiation (N=2000)\n";
    std::cout << "  Sweep obstacle speed: {0.8, 1.2}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 2000;
    const std::vector<double> SPEEDS = {0.8, 1.2};
    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::string filepath = DIR + "f1_high_n_oncoming.csv";
    std::ofstream ofs(filepath);
    ofs << "obstacle_speed,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double speed : SPEEDS) {
        for (const auto& method : ALL_METHODS) {
            std::cout << "  v=" << std::fixed << std::setprecision(1) << speed
                      << " " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(speed * 100) * 10000000 +
                    static_cast<int>(method.injection_mode) * 100000 +
                    method.injection_count * 10000 + i + 200000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {obs_oncoming(ref_path, 0.55, rng, speed)};

                met.add(run_experiment_rollout(cfg, seed));
            }

            std::ostringstream pfx;
            pfx << std::fixed << std::setprecision(1) << speed << ",";
            write_csv_row(ofs, met, N, pfx.str());

            auto [lo, hi] = met.cci();
            std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                      << "% [" << (lo*100) << "," << (hi*100) << "]"
                      << " clr=" << std::setprecision(2) << met.mc()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// F9: Multi-Obstacle Scaling at DRO Sweet Spot
//     Primary: 5 methods × {1,2,3} obstacles at v=1.0.
//     Sensitivity: path type {S-curve, Tight-S} to confirm scaling holds.
// ============================================================================

static void run_f9() {
    const std::string DIR = "figures/f9/";
    fs::create_directories(DIR);
    std::cout << "\n================================================================\n";
    std::cout << "  F9: Multi-Obstacle Scaling (N=1000)\n";
    std::cout << "  Sweep path: {S-curve, Tight-S} × obs: {1,2,3}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    const double SPEED = 1.0;
    const std::vector<int> OBS_COUNTS = {1, 2, 3};

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,              0.0, 0.0, 1},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,     0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,              0.0, 0.0, 1},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,  0.0, 0.0, 1},
        {"TopRisk-K2",        true,  InjectionMode::TOP_RISK_INJECT,  0.0, 0.0, 2},
    };

    std::vector<PathSetup> paths = {make_standard_scurve(), make_tight_scurve()};

    std::string filepath = DIR + "f9_multi_obs_sweet_spot.csv";
    std::ofstream ofs(filepath);
    ofs << "path,num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& ps : paths) {
        for (int nobs : OBS_COUNTS) {
            for (const auto& method : METHODS) {
                std::cout << "  " << ps.name << " obs=" << nobs
                          << " " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();
                Metrics met; met.method = method.name;

                std::vector<double> fracs = {0.55, 0.35, 0.75};
                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        std::hash<std::string>{}(ps.name) / 1000 +
                        nobs * 10000000 +
                        static_cast<int>(method.injection_mode) * 100000 + i + 210000000);
                    std::mt19937 rng(seed);

                    ExperimentConfig cfg = make_base_config(0.2, 0.1, nobs);
                    cfg = apply_method(cfg, method);
                    cfg.custom_ref_path = ps.path;
                    cfg.custom_initial_ego = ps.initial_ego;

                    std::vector<ObstacleState> obs;
                    for (int k = 0; k < nobs; ++k)
                        obs.push_back(obs_oncoming(ps.path, fracs[k], rng, SPEED));
                    cfg.initial_obstacle_states = obs;

                    met.add(run_experiment_rollout(cfg, seed));
                }

                std::ostringstream pfx;
                pfx << ps.name << "," << nobs << ",";
                write_csv_row(ofs, met, N, pfx.str());

                std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                          << "% clr=" << std::setprecision(2) << met.mc()
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// G1: Path Geometry Comparison
//     Primary: 5 paths × 8 methods.
//     Sensitivity: obstacle speed {0.8, 1.2} to confirm path ranking holds.
// ============================================================================

static void run_g1() {
    const std::string DIR = "figures/g1/";
    fs::create_directories(DIR);
    std::cout << "\n================================================================\n";
    std::cout << "  G1: Path Geometry Comparison (N=500)\n";
    std::cout << "  Sweep obstacle speed: {0.8, 1.2}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 500;
    const std::vector<double> SPEEDS = {0.8, 1.2};
    std::vector<PathSetup> paths = {
        make_straight_path(), make_gentle_scurve(),
        make_standard_scurve(), make_tight_scurve(), make_circle_path(),
    };

    std::string filepath = DIR + "g1_path_geometry.csv";
    std::ofstream ofs(filepath);
    ofs << "obstacle_speed,path,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double speed : SPEEDS) {
        for (const auto& ps : paths) {
            for (const auto& method : ALL_METHODS) {
                std::cout << "  v=" << std::fixed << std::setprecision(1) << speed
                          << " " << ps.name << " / " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();
                Metrics met; met.method = method.name;

                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        static_cast<int>(speed * 100) * 10000000 +
                        std::hash<std::string>{}(ps.name) / 1000 +
                        static_cast<int>(method.injection_mode) * 100000 + i + 220000000);
                    std::mt19937 rng(seed);

                    ExperimentConfig cfg = make_base_config();
                    cfg = apply_method(cfg, method);
                    cfg.custom_ref_path = ps.path;
                    cfg.custom_initial_ego = ps.initial_ego;
                    cfg.initial_obstacle_states = {obs_oncoming(ps.path, 0.55, rng, speed)};

                    met.add(run_experiment_rollout(cfg, seed));
                }

                std::ostringstream pfx;
                pfx << std::fixed << std::setprecision(1) << speed << "," << ps.name << ",";
                write_csv_row(ofs, met, N, pfx.str());

                std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                          << "% mm=" << (met.mr()*100) << "%"
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// G3: Switching Dynamics Sweep
//     Primary: 6 switch probs × 6 methods.
//     Sensitivity: path {S-curve, Tight-S} to confirm switching trends hold.
// ============================================================================

static void run_g3() {
    const std::string DIR = "figures/g3/";
    fs::create_directories(DIR);
    std::cout << "\n================================================================\n";
    std::cout << "  G3: Switching Dynamics Sweep (N=500)\n";
    std::cout << "  Sweep path: {S-curve, Tight-S}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 500;
    const std::vector<double> SWITCH_PROBS = {0.02, 0.05, 0.10, 0.20, 0.35, 0.50};
    std::vector<PathSetup> paths = {make_standard_scurve(), make_tight_scurve()};

    std::string filepath = DIR + "g3_switch_dynamics.csv";
    std::ofstream ofs(filepath);
    ofs << "path,switch_prob,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& ps : paths) {
        for (double sp : SWITCH_PROBS) {
            for (const auto& method : CORE6_METHODS) {
                std::cout << "  " << ps.name << " sp=" << std::setprecision(2) << sp
                          << " / " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();
                Metrics met; met.method = method.name;

                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        std::hash<std::string>{}(ps.name) / 1000 +
                        static_cast<int>(sp * 1000) * 1000000 +
                        static_cast<int>(method.injection_mode) * 100000 + i + 230000000);
                    std::mt19937 rng(seed);

                    ExperimentConfig cfg = make_base_config(sp, 0.1);
                    cfg = apply_method(cfg, method);
                    cfg.custom_ref_path = ps.path;
                    cfg.custom_initial_ego = ps.initial_ego;
                    cfg.initial_obstacle_states = {obs_oncoming(ps.path, 0.55, rng)};

                    met.add(run_experiment_rollout(cfg, seed));
                }

                std::ostringstream pfx;
                pfx << ps.name << "," << std::fixed << std::setprecision(2) << sp << ",";
                write_csv_row(ofs, met, N, pfx.str());

                std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                          << "% mm=" << (met.mr()*100) << "%"
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// G6: Challenging Environment Variants
//     Primary: 8 challenge envs × 8 methods.
//     Sensitivity: switch_prob {0.1, 0.3} to confirm challenge ranking holds.
// ============================================================================

struct ChallengeEnv {
    std::string name;
    std::function<std::vector<ObstacleState>(const ReferencePath&, std::mt19937&)> make;
    int n_obs;
};

static void run_g6() {
    const std::string DIR = "figures/g6/";
    fs::create_directories(DIR);
    std::cout << "\n================================================================\n";
    std::cout << "  G6: Challenging Environment Variants (N=500)\n";
    std::cout << "  Sweep switch_prob: {0.1, 0.3}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::vector<ChallengeEnv> envs = {
        {"HighSpeed-1.5",
         [](const ReferencePath& p, std::mt19937& r) -> std::vector<ObstacleState> {
             return {obs_high_speed(p, 0.55, r, 1.5)};
         }, 1},
        {"HighSpeed-2.0",
         [](const ReferencePath& p, std::mt19937& r) -> std::vector<ObstacleState> {
             return {obs_high_speed(p, 0.55, r, 2.0)};
         }, 1},
        {"Crossing",
         [](const ReferencePath& p, std::mt19937& r) -> std::vector<ObstacleState> {
             return {obs_crossing(p, 0.50, r, 1.0)};
         }, 1},
        {"Diagonal",
         [](const ReferencePath& p, std::mt19937& r) -> std::vector<ObstacleState> {
             return {obs_diagonal(p, 0.50, r, 1.0)};
         }, 1},
        {"Mixed-2obs",
         [](const ReferencePath& p, std::mt19937& r) -> std::vector<ObstacleState> {
             return {obs_oncoming(p, 0.45, r),
                     obs_crossing(p, 0.55, r, 0.8)};
         }, 2},
        {"Mixed-3obs",
         [](const ReferencePath& p, std::mt19937& r) -> std::vector<ObstacleState> {
             return {obs_high_speed(p, 0.40, r, 1.5),
                     obs_oncoming(p, 0.55, r),
                     obs_crossing(p, 0.60, r, 0.8)};
         }, 3},
        {"Dense-4obs",
         [](const ReferencePath& p, std::mt19937& r) -> std::vector<ObstacleState> {
             return {obs_oncoming(p, 0.35, r),
                     obs_oncoming(p, 0.45, r),
                     obs_high_speed(p, 0.55, r, 1.2),
                     obs_crossing(p, 0.50, r, 0.8)};
         }, 4},
        {"HighSpeed-Dense",
         [](const ReferencePath& p, std::mt19937& r) -> std::vector<ObstacleState> {
             return {obs_high_speed(p, 0.35, r, 1.5),
                     obs_high_speed(p, 0.50, r, 1.8),
                     obs_high_speed(p, 0.65, r, 1.5)};
         }, 3},
    };

    const int N = 500;
    const std::vector<double> SWITCH_PROBS = {0.1, 0.3};

    std::string filepath = DIR + "g6_challenging_envs.csv";
    std::ofstream ofs(filepath);
    ofs << "switch_prob,challenge,num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double sp : SWITCH_PROBS) {
        for (const auto& env : envs) {
            for (const auto& method : ALL_METHODS) {
                std::cout << "  sp=" << std::setprecision(1) << sp
                          << " " << env.name << " / " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();
                Metrics met; met.method = method.name;

                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        static_cast<int>(sp * 1000) * 10000000 +
                        std::hash<std::string>{}(env.name) / 1000 +
                        static_cast<int>(method.injection_mode) * 100000 + i + 260000000);
                    std::mt19937 rng(seed);

                    ExperimentConfig cfg = make_base_config(sp, 0.1, env.n_obs);
                    cfg = apply_method(cfg, method);
                    cfg.initial_obstacle_states = env.make(ref_path, rng);

                    met.add(run_experiment_rollout(cfg, seed));
                }

                std::ostringstream pfx;
                pfx << std::fixed << std::setprecision(1) << sp << ","
                    << env.name << "," << env.n_obs << ",";
                write_csv_row(ofs, met, N, pfx.str());

                std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                          << "% clr=" << std::setprecision(2) << met.mc()
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::set<std::string> filters;
    for (int i = 1; i < argc; ++i) filters.insert(argv[i]);
    bool run_all = filters.empty();

    std::cout << "========================================\n";
    std::cout << "Exhaustive Suite Runner\n";
    std::cout << "  F1:  High-N method differentiation\n";
    std::cout << "  F9:  Multi-obstacle scaling\n";
    std::cout << "  G1:  Path geometry comparison\n";
    std::cout << "  G3:  Switching dynamics sweep\n";
    std::cout << "  G6:  Challenging environment variants\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("f1")) run_f1();
    if (run_all || filters.count("f9")) run_f9();
    if (run_all || filters.count("g1")) run_g1();
    if (run_all || filters.count("g3")) run_g3();
    if (run_all || filters.count("g6")) run_g6();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0)
              << total << "s (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";
    return 0;
}
