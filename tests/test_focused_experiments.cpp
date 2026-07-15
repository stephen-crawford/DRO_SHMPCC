/**
 * @file test_focused_experiments.cpp
 * @brief Focused experiments targeting statistical significance gaps.
 *
 * Based on findings context analysis, these experiments target:
 *   F1: High-N variant differentiation (N=2000) on Oncoming — separate Mode-DRO(inj)
 *       from TopRisk-K1, DiverseRisk-K1, WDRO-sampling
 *   F2: Speed × Path interaction — medium speeds (0.8, 1.0, 1.3) on paths
 *       (Straight, S-curve, Tight-S) to find DRO sweet spots
 *   F3: K=1 vs K=2 injection with multi-obstacle scaling (1, 2, 3 obs)
 *   F4: Clearance-focused: raw per-rollout clearance data on best DRO scenarios
 *       for statistical testing (N=2000, Oncoming + Crossing)
 *   F5: Traj-DRO(comb) vs Mode-DRO(inj) at speed 1.3 (N=2000) — resolve
 *       whether the apparent Traj-DRO advantage is real
 *
 * Output: focused_figures/
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

using namespace scenario_mpc;
namespace fs = std::filesystem;

// ============================================================================
// Constants
// ============================================================================

static const std::string OUTPUT_DIR = "focused_figures/";

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

static ExperimentConfig make_base_config(double switch_prob = 0.2, double rare_prob = 0.1,
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
// Path/obstacle helpers
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

static PathSetup make_standard_scurve() {
    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);
    return {"S-curve", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_tight_scurve() {
    auto path = ReferencePath::create_s_curve(20.0, 5.0, 200);
    return {"Tight-S", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

/// Place an oncoming obstacle at given arc fraction and speed.
static ObstacleState obstacle_on_path(const ReferencePath& path, double arc_fraction,
                                       std::mt19937& rng, double speed = 0.8) {
    double obs_s = arc_fraction * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d tangent(std::cos(pp.heading), std::sin(pp.heading));

    std::uniform_real_distribution<double> lat(-0.3, 0.3);
    std::uniform_real_distribution<double> spd(-0.1, 0.1);
    Eigen::Vector2d pos = pp.position + lat(rng) * normal;
    double v = speed + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -v * tangent.x(), -v * tangent.y());
}

/// Place a crossing obstacle (perpendicular approach).
static ObstacleState obstacle_crossing(const ReferencePath& path, double arc_fraction,
                                        std::mt19937& rng, double speed = 1.0) {
    double obs_s = arc_fraction * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));

    std::uniform_real_distribution<double> offset(3.0, 5.0);
    double d = offset(rng);
    Eigen::Vector2d pos = pp.position + d * normal;
    return ObstacleState(pos.x(), pos.y(), -speed * normal.x(), -speed * normal.y());
}


// ============================================================================
// F1: High-N Variant Differentiation (N=2000, Oncoming)
// Goal: Separate Mode-DRO(inj) from TopRisk-K1, WDRO-sampling, DiverseRisk
// ============================================================================

static void run_f1() {
    std::cout << "\n================================================================\n";
    std::cout << "  F1: High-N Variant Differentiation (N=2000, Oncoming)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 2000;

    // All injection methods we want to differentiate
    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,                0.0, 0.0, 1},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,       0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,                0.0, 0.0, 1},
        {"WDRO-inject-K2",    true,  InjectionMode::DRO,                0.0, 0.0, 2},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 1},
        {"TopRisk-K2",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 2},
        {"DiverseRisk-K1",    true,  InjectionMode::DIVERSE_RISK_INJECT,0.0, 0.0, 1},
        {"Softmax-tau5",      true,  InjectionMode::SOFTMAX_RISK,       5.0, 0.0, 1},
    };

    std::string filepath = OUTPUT_DIR + "f1_high_n_oncoming.csv";
    std::ofstream ofs(filepath);
    ofs << "method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& method : METHODS) {
        std::cout << "  " << method.name << ": ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        Metrics met;
        met.method = method.name;

        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(
                static_cast<int>(method.injection_mode) * 100000 +
                method.injection_count * 10000 + i + 80000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig cfg = make_base_config();
            cfg = apply_method(cfg, method);
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;

            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        write_csv_row(ofs, met, N);

        std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                  << " [" << std::setprecision(1) << (met.coll_ci().first * 100) << "%, "
                  << std::setprecision(1) << (met.coll_ci().second * 100) << "%]"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F2: Speed × Path Interaction
// Goal: Find DRO sweet spots at medium speeds + path complexity
// ============================================================================

static void run_f2() {
    std::cout << "\n================================================================\n";
    std::cout << "  F2: Speed x Path Interaction\n";
    std::cout << "  Speeds: {0.8, 1.0, 1.3} x Paths: {Straight, S-curve, Tight-S}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    const std::vector<double> SPEEDS = {0.8, 1.0, 1.3};
    std::vector<PathSetup> paths = {
        make_straight_path(),
        make_standard_scurve(),
        make_tight_scurve(),
    };

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,                0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,                0.0, 0.0, 1},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 1},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,       0.0, 0.0, 1},
    };

    std::string filepath = OUTPUT_DIR + "f2_speed_path_interaction.csv";
    std::ofstream ofs(filepath);
    ofs << "obstacle_speed,path,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double speed : SPEEDS) {
        for (const auto& ps : paths) {
            for (const auto& method : METHODS) {
                std::cout << "  v=" << std::setprecision(1) << speed
                          << " " << ps.name << " / " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();

                Metrics met;
                met.method = method.name;

                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        static_cast<int>(speed * 100) * 10000000 +
                        std::hash<std::string>{}(ps.name) / 1000 +
                        static_cast<int>(method.injection_mode) * 100000 + i + 81000000);
                    std::mt19937 env_rng(seed);

                    ExperimentConfig cfg = make_base_config();
                    cfg = apply_method(cfg, method);
                    cfg.custom_ref_path = ps.path;
                    cfg.custom_initial_ego = ps.initial_ego;
                    cfg.initial_obstacle_states = {
                        obstacle_on_path(ps.path, 0.55, env_rng, speed)
                    };

                    RolloutRecord rec = run_experiment_rollout(cfg, seed);
                    met.add(rec);
                }

                ofs << std::fixed << std::setprecision(1) << speed << ","
                    << ps.name << ",";
                write_csv_row(ofs, met, N);

                std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                          << " clr=" << std::setprecision(2) << met.mean_clearance()
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F3: K=1 vs K=2 Injection with Multi-Obstacle Scaling
// Goal: Determine if K=2 helps with more obstacles
// ============================================================================

static void run_f3() {
    std::cout << "\n================================================================\n";
    std::cout << "  F3: K=1 vs K=2 with Multi-Obstacle Scaling\n";
    std::cout << "  Obstacles: {1, 2, 3}, S-curve, N=1000\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    const std::vector<int> OBS_COUNTS = {1, 2, 3};
    const std::vector<std::vector<double>> OBS_FRACS = {
        {0.50},
        {0.40, 0.60},
        {0.30, 0.45, 0.65},
    };

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,                0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,                0.0, 0.0, 1},
        {"WDRO-inject-K2",    true,  InjectionMode::DRO,                0.0, 0.0, 2},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 1},
        {"TopRisk-K2",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 2},
    };

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::string filepath = OUTPUT_DIR + "f3_k1_vs_k2_multi_obs.csv";
    std::ofstream ofs(filepath);
    ofs << "num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (size_t oi = 0; oi < OBS_COUNTS.size(); ++oi) {
        int n_obs = OBS_COUNTS[oi];
        const auto& fracs = OBS_FRACS[oi];

        for (const auto& method : METHODS) {
            std::cout << "  " << n_obs << " obs / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(n_obs * 1000000 +
                    static_cast<int>(method.injection_mode) * 100000 +
                    method.injection_count * 50000 + i + 82000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, 0.1, n_obs);
                cfg = apply_method(cfg, method);

                std::vector<ObstacleState> obs_states;
                for (double frac : fracs) {
                    obs_states.push_back(obstacle_on_path(ref_path, frac, env_rng));
                }
                cfg.initial_obstacle_states = obs_states;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << n_obs << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F4: Clearance-Focused Raw Data (N=2000, Oncoming + Crossing)
// Goal: Raw per-rollout clearance for statistical testing
// ============================================================================

static void run_f4() {
    std::cout << "\n================================================================\n";
    std::cout << "  F4: Clearance-Focused Raw Data (N=2000)\n";
    std::cout << "  Envs: Oncoming + Crossing on S-curve\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 2000;

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,                0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,                0.0, 0.0, 1},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 1},
    };

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    struct EnvDef {
        std::string name;
        bool use_crossing;  // false = oncoming, true = crossing
    };
    std::vector<EnvDef> envs = {
        {"Oncoming", false},
        {"Crossing", true},
    };

    // Raw per-rollout data file
    std::string raw_filepath = OUTPUT_DIR + "f4_clearance_raw.csv";
    std::ofstream raw_ofs(raw_filepath);
    raw_ofs << "environment,method,seed,collision,min_clearance,progress\n";

    // Summary stats file
    std::string sum_filepath = OUTPUT_DIR + "f4_clearance_summary.csv";
    std::ofstream sum_ofs(sum_filepath);
    sum_ofs << "environment,method,collision_rate,coll_ci_lo,coll_ci_hi,"
            << "mean_clearance,std_clearance,clearance_ci_lo,clearance_ci_hi,"
            << "p5,p10,p25,p50,p75,p90,p95,"
            << "mean_clearance_coll,mean_clearance_nocoll,n_rollouts\n";

    for (const auto& env : envs) {
        for (const auto& method : METHODS) {
            std::cout << "  " << env.name << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;
            std::vector<double> coll_clr, nocoll_clr;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(env.name) / 1000 +
                    static_cast<int>(method.injection_mode) * 100000 + i + 83000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);
                cfg.custom_ref_path = ref_path;
                cfg.custom_initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);

                if (env.use_crossing) {
                    cfg.initial_obstacle_states = {
                        obstacle_crossing(ref_path, 0.55, env_rng, 1.0)
                    };
                } else {
                    cfg.initial_obstacle_states = {
                        obstacle_on_path(ref_path, 0.55, env_rng, 0.8)
                    };
                }

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);

                // Raw data
                raw_ofs << env.name << "," << method.name << "," << seed << ","
                        << (rec.collision ? 1 : 0) << ","
                        << std::fixed << std::setprecision(6) << rec.min_clearance << ","
                        << std::setprecision(4) << rec.total_progress << "\n";

                if (rec.collision) coll_clr.push_back(rec.min_clearance);
                else nocoll_clr.push_back(rec.min_clearance);
            }
            raw_ofs.flush();

            // Compute clearance statistics
            auto sorted_clr = met.min_clearances;
            std::sort(sorted_clr.begin(), sorted_clr.end());
            auto pct = [&](double p) {
                return sorted_clr[std::max(0, (int)(p * (sorted_clr.size() - 1)))];
            };
            double mean_clr = met.mean_clearance();
            double var_clr = 0;
            for (double c : met.min_clearances) var_clr += (c - mean_clr) * (c - mean_clr);
            double std_clr = std::sqrt(var_clr / met.min_clearances.size());
            double ci_margin = 1.96 * std_clr / std::sqrt(met.min_clearances.size());

            double mean_coll_clr = coll_clr.empty() ? 0 :
                std::accumulate(coll_clr.begin(), coll_clr.end(), 0.0) / coll_clr.size();
            double mean_nocoll_clr = nocoll_clr.empty() ? 0 :
                std::accumulate(nocoll_clr.begin(), nocoll_clr.end(), 0.0) / nocoll_clr.size();

            auto [clo, chi] = met.coll_ci();
            sum_ofs << env.name << "," << method.name << ","
                    << std::fixed << std::setprecision(6)
                    << met.coll_rate() << "," << clo << "," << chi << ","
                    << std::setprecision(4) << mean_clr << "," << std_clr << ","
                    << (mean_clr - ci_margin) << "," << (mean_clr + ci_margin) << ","
                    << pct(0.05) << "," << pct(0.10) << "," << pct(0.25) << ","
                    << pct(0.50) << "," << pct(0.75) << "," << pct(0.90) << "," << pct(0.95) << ","
                    << mean_coll_clr << "," << mean_nocoll_clr << "," << N << "\n";
            sum_ofs.flush();

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(3) << mean_clr
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    raw_ofs.close();
    sum_ofs.close();
    std::cout << "  Written: " << raw_filepath << " and " << sum_filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F5: Traj-DRO(comb) vs Mode-DRO(inj) at Speed 1.3 (N=2000)
// Goal: Resolve whether Traj-DRO(comb) can beat Mode-DRO at medium speeds
//
// Uses step callback approach for trajectory-level DRO (same as
// test_traj_dro_generalization.cpp).
// ============================================================================

// --- Trajectory DRO helper functions ---

static double compute_trajectory_risk(
    const ObstacleTrajectory& traj,
    const std::vector<EgoState>& ego_ref,
    double safety_radius, int num_discs, double vehicle_length,
    double z_alpha = 1.6449
) {
    double max_risk = 0.0;
    int H = std::min(static_cast<int>(traj.steps.size()),
                     static_cast<int>(ego_ref.size()));
    for (int k = 1; k < H; ++k) {
        const auto& step = traj.steps[k];
        const auto& ego = ego_ref[k];
        for (int d = 0; d < num_discs; ++d) {
            Eigen::Vector2d disc_pos;
            if (num_discs > 1) {
                double theta = ego.theta;
                Eigen::Vector2d dir(std::cos(theta), std::sin(theta));
                double step_offset = vehicle_length / (num_discs - 1);
                double offset = -vehicle_length / 2.0 + d * step_offset;
                disc_pos = ego.position() + offset * dir;
            } else {
                disc_pos = ego.position();
            }
            Eigen::Vector2d diff = step.mean - disc_pos;
            double dist = diff.norm();
            if (dist < 1e-12) { max_risk = std::max(max_risk, safety_radius); continue; }
            Eigen::Vector2d n_dir = diff / dist;
            double var_dir = static_cast<double>(n_dir.transpose() * step.covariance * n_dir);
            double sigma_dir = std::sqrt(std::max(1e-12, var_dir));
            double risk = std::max(0.0, safety_radius + z_alpha * sigma_dir - dist);
            max_risk = std::max(max_risk, risk);
        }
    }
    return max_risk;
}

static std::vector<std::vector<double>> compute_trajectory_transport_costs(
    const std::vector<ObstacleTrajectory>& trajectories
) {
    int S = static_cast<int>(trajectories.size());
    std::vector<std::vector<double>> D(S, std::vector<double>(S, 0.0));
    for (int i = 0; i < S; ++i) {
        for (int j = i + 1; j < S; ++j) {
            int H = std::min(static_cast<int>(trajectories[i].steps.size()),
                             static_cast<int>(trajectories[j].steps.size()));
            double total = 0.0;
            for (int k = 0; k < H; ++k) {
                Eigen::Vector2d diff = trajectories[i].steps[k].mean - trajectories[j].steps[k].mean;
                total += diff.squaredNorm();
            }
            double cost = (H > 0) ? total / H : 0.0;
            D[i][j] = cost; D[j][i] = cost;
        }
    }
    return D;
}

struct TrajectoryDROResult {
    std::vector<double> weights;
    double optimal_lambda = 0.0;
    double worst_case_risk = 0.0;
    double rho_used = 0.0;
    int worst_trajectory_idx = -1;
};

static TrajectoryDROResult solve_trajectory_dro(
    const std::vector<double>& risks,
    const std::vector<std::vector<double>>& D,
    double rho
) {
    int S = static_cast<int>(risks.size());
    TrajectoryDROResult result;
    result.rho_used = rho;
    if (S == 0) return result;
    double w_nom = 1.0 / S;
    // Find worst-case via bisection on lambda
    double lambda_lo = 0.0, lambda_hi = 100.0;
    for (int iter = 0; iter < 50; ++iter) {
        double lambda = (lambda_lo + lambda_hi) / 2.0;
        double obj = lambda * rho;
        for (int i = 0; i < S; ++i) {
            double max_gain = 0.0;
            for (int j = 0; j < S; ++j) {
                double gain = risks[j] - lambda * D[i][j];
                max_gain = std::max(max_gain, gain);
            }
            obj += w_nom * max_gain;
        }
        if (obj > lambda * rho + 1e-8) lambda_lo = lambda;
        else lambda_hi = lambda;
    }
    double lambda_star = (lambda_lo + lambda_hi) / 2.0;
    result.optimal_lambda = lambda_star;
    // Compute weights
    result.weights.resize(S, 0.0);
    for (int i = 0; i < S; ++i) {
        int best_j = 0;
        double best_gain = risks[0] - lambda_star * D[i][0];
        for (int j = 1; j < S; ++j) {
            double gain = risks[j] - lambda_star * D[i][j];
            if (gain > best_gain) { best_gain = gain; best_j = j; }
        }
        result.weights[best_j] += w_nom;
    }
    // Find worst trajectory
    int worst_idx = 0;
    double worst_risk = risks[0];
    for (int j = 1; j < S; ++j) {
        if (risks[j] > worst_risk) { worst_risk = risks[j]; worst_idx = j; }
    }
    result.worst_trajectory_idx = worst_idx;
    result.worst_case_risk = worst_risk;
    return result;
}

// --- F5 Traj-DRO method enum + callback ---

enum class F5Method {
    BASE, MODE_DRO_INJECT_K1, MODE_DRO_INJECT_K2, MODE_DRO_SAMPLE,
    TOPRISK_K1, TOPRISK_K2,
    TRAJ_DRO_INJECT, TRAJ_DRO_COMBINED
};

static std::string f5_method_name(F5Method m) {
    switch (m) {
        case F5Method::BASE:              return "Base";
        case F5Method::MODE_DRO_INJECT_K1:return "WDRO-inject-K1";
        case F5Method::MODE_DRO_INJECT_K2:return "WDRO-inject-K2";
        case F5Method::MODE_DRO_SAMPLE:   return "WDRO-sampling";
        case F5Method::TOPRISK_K1:        return "TopRisk-K1";
        case F5Method::TOPRISK_K2:        return "TopRisk-K2";
        case F5Method::TRAJ_DRO_INJECT:   return "Traj-DRO(inj)";
        case F5Method::TRAJ_DRO_COMBINED: return "Traj-DRO(comb)";
    }
    return "?";
}

static std::function<void(int, int, ObstacleSim&, AdaptiveScenarioMPC&, std::mt19937&)>
make_traj_dro_callback(F5Method method, double rho, int num_scenarios) {
    return [method, rho, num_scenarios](
        int step, int obs_id, ObstacleSim& obs_sim,
        AdaptiveScenarioMPC& controller, std::mt19937& rng
    ) {
        if (obs_id != 0) return;
        const auto& modes = obs_sim.mode_models;
        int horizon = controller.config().horizon;
        double ego_r = controller.config().ego_radius;
        double obs_r = controller.config().obstacle_radius;
        double margin = controller.config().safety_margin;
        double safety_radius = ego_r + obs_r + margin;
        int n_discs = controller.config().num_discs;
        double veh_len = controller.config().vehicle_length;

        int S_particles = num_scenarios;
        std::vector<ObstacleTrajectory> traj_particles;
        traj_particles.reserve(S_particles);

        std::map<std::string, double> freq_weights;
        int total_obs_count = 0;
        for (const auto& [mode_id, _] : modes) freq_weights[mode_id] = 0.0;
        const auto& prev_scenarios = controller.scenarios();
        if (!prev_scenarios.empty()) {
            for (const auto& sc : prev_scenarios) {
                auto it = sc.trajectories.find(0);
                if (it != sc.trajectories.end()) {
                    freq_weights[it->second.mode_id] += 1.0;
                    total_obs_count++;
                }
            }
            if (total_obs_count > 0) { for (auto& [_, w] : freq_weights) w /= total_obs_count; }
        }
        if (total_obs_count == 0) { for (auto& [_, w] : freq_weights) w = 1.0 / freq_weights.size(); }

        std::normal_distribution<double> normal_dist(0.0, 1.0);
        for (int s = 0; s < S_particles; ++s) {
            std::string sampled_mode = sample_mode_from_weights(freq_weights, rng);
            if (modes.find(sampled_mode) == modes.end()) continue;
            const ModeModel& mode = modes.at(sampled_mode);
            std::vector<PredictionStep> steps;
            steps.reserve(horizon + 1);
            Eigen::Vector4d x = obs_sim.state.to_array();
            Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
            steps.emplace_back(0, x.head<2>(), cov.block<2,2>(0,0));
            for (int k = 0; k < horizon; ++k) {
                Eigen::VectorXd noise(mode.noise_dim());
                for (int d = 0; d < mode.noise_dim(); ++d) noise(d) = normal_dist(rng);
                x = mode.A * x + mode.b + mode.G * noise;
                cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();
                steps.emplace_back(k + 1, x.head<2>(), cov.block<2,2>(0,0));
            }
            ObstacleTrajectory traj(0, sampled_mode, steps, freq_weights[sampled_mode]);
            traj_particles.push_back(std::move(traj));
        }
        if (static_cast<int>(traj_particles.size()) < 3) return;

        Eigen::Vector2d obs_pos = obs_sim.state.position();
        double heading = std::atan2(obs_pos.y(), obs_pos.x());
        std::vector<EgoState> ego_ref;
        ego_ref.reserve(horizon + 1);
        for (int k = 0; k <= horizon; ++k) {
            EgoState es;
            es.x = 1.5 * k * 0.1 * std::cos(heading);
            es.y = 1.5 * k * 0.1 * std::sin(heading);
            es.theta = heading;
            es.v = 1.5;
            ego_ref.push_back(es);
        }

        int S = static_cast<int>(traj_particles.size());
        std::vector<double> risks(S);
        for (int s = 0; s < S; ++s) {
            risks[s] = compute_trajectory_risk(traj_particles[s], ego_ref, safety_radius, n_discs, veh_len);
        }
        auto D = compute_trajectory_transport_costs(traj_particles);
        auto dro_result = solve_trajectory_dro(risks, D, rho);
        if (dro_result.worst_trajectory_idx < 0) return;

        if (method == F5Method::TRAJ_DRO_INJECT || method == F5Method::TRAJ_DRO_COMBINED) {
            int worst_idx = dro_result.worst_trajectory_idx;
            const auto& worst_traj = traj_particles[worst_idx];
            Scenario inj_scenario;
            inj_scenario.scenario_id = num_scenarios + 100 + step;
            inj_scenario.is_injected = true;
            inj_scenario.probability = 1.0;
            inj_scenario.trajectories[0] = worst_traj;
            inj_scenario.trajectories[0].probability = 1.0;
            controller.inject_scenario(inj_scenario);
        }
        if (method == F5Method::TRAJ_DRO_COMBINED) {
            std::map<std::string, double> mode_weights;
            for (const auto& [mode_id, _] : modes) mode_weights[mode_id] = 0.0;
            for (int s = 0; s < S; ++s) {
                mode_weights[traj_particles[s].mode_id] += dro_result.weights[s];
            }
            double sum_w = 0.0;
            for (auto& [_, w] : mode_weights) sum_w += w;
            if (sum_w > 1e-12) { for (auto& [_, w] : mode_weights) w /= sum_w; }
            controller.reset_scenarios();
            controller.set_custom_mode_weights(0, mode_weights);
        }
    };
}

static ExperimentConfig apply_f5_method(ExperimentConfig cfg, F5Method method, double rho = 0.1) {
    cfg.method_name = f5_method_name(method);
    switch (method) {
        case F5Method::BASE:
            break;
        case F5Method::MODE_DRO_SAMPLE:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::QSTAR_SAMPLE;
            break;
        case F5Method::MODE_DRO_INJECT_K1:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::DRO;
            cfg.dro_injection_count = 1;
            break;
        case F5Method::MODE_DRO_INJECT_K2:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::DRO;
            cfg.dro_injection_count = 2;
            break;
        case F5Method::TOPRISK_K1:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::TOP_RISK_INJECT;
            cfg.dro_injection_count = 1;
            break;
        case F5Method::TOPRISK_K2:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::TOP_RISK_INJECT;
            cfg.dro_injection_count = 2;
            break;
        case F5Method::TRAJ_DRO_INJECT:
        case F5Method::TRAJ_DRO_COMBINED:
            cfg.step_callback = make_traj_dro_callback(method, rho, NUM_SCENARIOS);
            break;
    }
    return cfg;
}

static void run_f5() {
    std::cout << "\n================================================================\n";
    std::cout << "  F5: Mode-DRO Speed Comparison (N=2000)\n";
    std::cout << "  Tight S-curve, 2 oncoming obstacles, no mode switching\n";
    std::cout << "  Speeds: {0.5, 1.0, 1.5, 2.0}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 2000;
    const std::vector<double> SPEEDS = {0.5, 1.0, 1.5, 2.0};

    const std::vector<F5Method> METHODS = {
        F5Method::BASE,
        F5Method::MODE_DRO_SAMPLE,
        F5Method::MODE_DRO_INJECT_K1,
        F5Method::MODE_DRO_INJECT_K2,
        F5Method::TOPRISK_K1,
        F5Method::TOPRISK_K2,
    };

    auto ref_path = ReferencePath::create_s_curve(20.0, 5.0, 200);  // Tight S-curve

    std::string filepath = OUTPUT_DIR + "f5_traj_vs_mode_dro.csv";
    std::ofstream ofs(filepath);
    ofs << "obstacle_speed,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double speed : SPEEDS) {
        for (F5Method method : METHODS) {
            std::string mname = f5_method_name(method);
            std::cout << "  v=" << std::setprecision(1) << speed
                      << " " << mname << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = mname;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(speed * 100) * 10000000 +
                    static_cast<int>(method) * 100000 + i + 84000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.0, 0.0, 2);  // no switching, 2 obstacles
                cfg = apply_f5_method(cfg, method);
                cfg.custom_ref_path = ref_path;
                cfg.custom_initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
                cfg.path_following_obstacles = true;
                cfg.initial_obstacle_states = {
                    obstacle_on_path(ref_path, 0.45, env_rng, speed),
                    obstacle_on_path(ref_path, 0.65, env_rng, speed),
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(1) << speed << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " [" << std::setprecision(1) << (met.coll_ci().first * 100) << "%, "
                      << std::setprecision(1) << (met.coll_ci().second * 100) << "%]"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F6: Traj-DRO Transition Speed (N=1500)
// Goal: Find the speed where Traj-DRO(comb) overtakes Mode-DRO(inj)
// ============================================================================

static void run_f6() {
    std::cout << "\n================================================================\n";
    std::cout << "  F6: Traj-DRO Transition Speed Sweep (N=1500)\n";
    std::cout << "  S-curve, speeds: {0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1500;
    const std::vector<double> SPEEDS = {0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5};

    const std::vector<F5Method> METHODS = {
        F5Method::BASE,
        F5Method::MODE_DRO_INJECT_K1,
        F5Method::TRAJ_DRO_COMBINED,
    };

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::string filepath = OUTPUT_DIR + "f6_transition_speed.csv";
    std::ofstream ofs(filepath);
    ofs << "obstacle_speed,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double speed : SPEEDS) {
        for (F5Method method : METHODS) {
            std::string mname = f5_method_name(method);
            std::cout << "  v=" << std::setprecision(1) << speed
                      << " " << mname << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = mname;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(speed * 100) * 10000000 +
                    static_cast<int>(method) * 100000 + i + 85000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_f5_method(cfg, method);
                cfg.custom_ref_path = ref_path;
                cfg.custom_initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
                cfg.initial_obstacle_states = {
                    obstacle_on_path(ref_path, 0.55, env_rng, speed)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(1) << speed << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " [" << std::setprecision(1) << (met.coll_ci().first * 100) << "%, "
                      << std::setprecision(1) << (met.coll_ci().second * 100) << "%]"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F7: Traj-DRO on Tight-S at Medium Speeds (N=1500)
// Goal: Test if Traj-DRO advantage at v=1.1-1.4 holds on hardest path
// ============================================================================

static void run_f7() {
    std::cout << "\n================================================================\n";
    std::cout << "  F7: Traj-DRO on Tight-S at Medium Speeds (N=1500)\n";
    std::cout << "  Tight-S, speeds: {0.8, 1.0, 1.2, 1.3}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1500;
    const std::vector<double> SPEEDS = {0.8, 1.0, 1.2, 1.3};

    const std::vector<F5Method> METHODS = {
        F5Method::BASE,
        F5Method::MODE_DRO_INJECT_K1,
        F5Method::MODE_DRO_SAMPLE,
        F5Method::TRAJ_DRO_COMBINED,
    };

    auto tight_s = make_tight_scurve();

    std::string filepath = OUTPUT_DIR + "f7_traj_dro_tight_s.csv";
    std::ofstream ofs(filepath);
    ofs << "obstacle_speed,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double speed : SPEEDS) {
        for (F5Method method : METHODS) {
            std::string mname = f5_method_name(method);
            std::cout << "  v=" << std::setprecision(1) << speed
                      << " " << mname << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = mname;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(speed * 100) * 10000000 +
                    static_cast<int>(method) * 100000 + i + 86000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_f5_method(cfg, method);
                cfg.custom_ref_path = tight_s.path;
                cfg.custom_initial_ego = tight_s.initial_ego;
                cfg.initial_obstacle_states = {
                    obstacle_on_path(tight_s.path, 0.55, env_rng, speed)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(1) << speed << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " [" << std::setprecision(1) << (met.coll_ci().first * 100) << "%, "
                      << std::setprecision(1) << (met.coll_ci().second * 100) << "%]"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F8: Best DRO Showcase — environments where DRO excels most (N=2000)
// Goal: Comprehensive data for the strongest results in the paper
// ============================================================================

static void run_f8() {
    std::cout << "\n================================================================\n";
    std::cout << "  F8: Best DRO Showcase (N=2000)\n";
    std::cout << "  Best scenarios: Oncoming, Tight-S@1.0, Tight-S@1.3, Crossing\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 2000;

    struct ScenarioDef {
        std::string name;
        bool use_env_setup;  // true = use create_environment (Oncoming)
        EnvironmentType env_type;
        PathSetup path;
        double speed;
        bool crossing;  // if true, use crossing obstacle instead of oncoming
    };

    auto s_curve = make_standard_scurve();
    auto tight_s = make_tight_scurve();

    std::vector<ScenarioDef> scenarios = {
        {"Oncoming-default", true, EnvironmentType::ONCOMING, s_curve, 0.8, false},
        {"Tight-S@v1.0", false, EnvironmentType::ONCOMING, tight_s, 1.0, false},
        {"Tight-S@v1.3", false, EnvironmentType::ONCOMING, tight_s, 1.3, false},
        {"S-curve-crossing", false, EnvironmentType::ONCOMING, s_curve, 1.0, true},
    };

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,                0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,                0.0, 0.0, 1},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 1},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,       0.0, 0.0, 1},
    };

    std::string filepath = OUTPUT_DIR + "f8_best_dro_showcase.csv";
    std::ofstream ofs(filepath);
    ofs << "scenario,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    // Also raw clearance data
    std::string raw_filepath = OUTPUT_DIR + "f8_showcase_clearance_raw.csv";
    std::ofstream raw_ofs(raw_filepath);
    raw_ofs << "scenario,method,seed,collision,min_clearance,progress\n";

    for (const auto& sc : scenarios) {
        for (const auto& method : METHODS) {
            std::cout << "  " << sc.name << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(sc.name) / 1000 +
                    static_cast<int>(method.injection_mode) * 100000 + i + 87000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);

                if (sc.use_env_setup) {
                    EnvironmentSetup env_setup = create_environment(sc.env_type, env_rng);
                    cfg.initial_obstacle_states = {env_setup.initial_obs};
                    cfg.obs_modes = env_setup.obs_modes;
                } else {
                    cfg.custom_ref_path = sc.path.path;
                    cfg.custom_initial_ego = sc.path.initial_ego;
                    if (sc.crossing) {
                        cfg.initial_obstacle_states = {
                            obstacle_crossing(sc.path.path, 0.55, env_rng, sc.speed)
                        };
                    } else {
                        cfg.initial_obstacle_states = {
                            obstacle_on_path(sc.path.path, 0.55, env_rng, sc.speed)
                        };
                    }
                }

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);

                raw_ofs << sc.name << "," << method.name << "," << seed << ","
                        << (rec.collision ? 1 : 0) << ","
                        << std::fixed << std::setprecision(6) << rec.min_clearance << ","
                        << std::setprecision(4) << rec.total_progress << "\n";
            }
            raw_ofs.flush();

            ofs << sc.name << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " [" << std::setprecision(1) << (met.coll_ci().first * 100) << "%, "
                      << std::setprecision(1) << (met.coll_ci().second * 100) << "%]"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    raw_ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F9: Multi-obstacle scaling at DRO sweet spot (v=1.0, Tight-S, N=1500)
// Goal: Does DRO's 74 pp benefit degrade with more obstacles?
// ============================================================================

static void run_f9() {
    std::cout << "\n================================================================\n";
    std::cout << "  F9: Multi-Obstacle Scaling at DRO Sweet Spot (N=1500)\n";
    std::cout << "  Tight-S, v=1.0, obstacles: {1, 2, 3}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1500;
    const double SPEED = 1.0;
    const std::vector<int> OBS_COUNTS = {1, 2, 3};

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,                0.0, 0.0, 1},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,       0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,                0.0, 0.0, 1},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 1},
        {"TopRisk-K2",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 2},
    };

    auto tight_s = make_tight_scurve();

    std::string filepath = OUTPUT_DIR + "f9_multi_obs_sweet_spot.csv";
    std::ofstream ofs(filepath);
    ofs << "num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (int nobs : OBS_COUNTS) {
        for (const auto& method : METHODS) {
            std::cout << "  obs=" << nobs << " " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    nobs * 10000000 +
                    static_cast<int>(method.injection_mode) * 100000 + i + 90000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, 0.1, nobs);
                cfg = apply_method(cfg, method);
                cfg.custom_ref_path = tight_s.path;
                cfg.custom_initial_ego = tight_s.initial_ego;

                // Place obstacles at different arc fractions
                std::vector<ObstacleState> obstacles;
                std::vector<double> fractions = {0.55, 0.35, 0.75};
                for (int k = 0; k < nobs; ++k) {
                    obstacles.push_back(
                        obstacle_on_path(tight_s.path, fractions[k], env_rng, SPEED));
                }
                cfg.initial_obstacle_states = obstacles;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << nobs << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " [" << std::setprecision(1) << (met.coll_ci().first * 100) << "%, "
                      << std::setprecision(1) << (met.coll_ci().second * 100) << "%]"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F10: Fine-grained Tight-S speed sweep with DRO crossover (N=1500)
// Goal: Map where DRO collision rate crosses Base, find exact sweet spot
// ============================================================================

static void run_f10() {
    std::cout << "\n================================================================\n";
    std::cout << "  F10: Tight-S Speed Sweep — DRO Crossover (N=1500)\n";
    std::cout << "  Speeds: {0.6, 0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 2.0}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1500;
    const std::vector<double> SPEEDS = {0.6, 0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 2.0};

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,                0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,                0.0, 0.0, 1},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,       0.0, 0.0, 1},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 1},
    };

    auto tight_s = make_tight_scurve();

    std::string filepath = OUTPUT_DIR + "f10_tight_s_speed_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "obstacle_speed,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double speed : SPEEDS) {
        for (const auto& method : METHODS) {
            std::cout << "  v=" << std::setprecision(1) << speed
                      << " " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(speed * 100) * 10000000 +
                    static_cast<int>(method.injection_mode) * 100000 + i + 91000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);
                cfg.custom_ref_path = tight_s.path;
                cfg.custom_initial_ego = tight_s.initial_ego;
                cfg.initial_obstacle_states = {
                    obstacle_on_path(tight_s.path, 0.55, env_rng, speed)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(1) << speed << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " [" << std::setprecision(1) << (met.coll_ci().first * 100) << "%, "
                      << std::setprecision(1) << (met.coll_ci().second * 100) << "%]"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F11: S-curve vs Tight-S at High Speeds — Path Geometry Effect on DRO
// Goal: Quantify why DRO hurts on S-curve at v>1.5 but not on Tight-S
// ============================================================================

static void run_f11() {
    std::cout << "\n================================================================\n";
    std::cout << "  F11: S-curve vs Tight-S at High Speeds (N=1500)\n";
    std::cout << "  Paths: {S-curve, Tight-S, Straight}, Speeds: {1.3, 1.5, 1.7, 2.0}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1500;
    const std::vector<double> SPEEDS = {1.3, 1.5, 1.7, 2.0};

    struct PathEntry {
        std::string name;
        PathSetup setup;
    };
    std::vector<PathEntry> paths = {
        {"S-curve", make_standard_scurve()},
        {"Tight-S", make_tight_scurve()},
        {"Straight", make_straight_path()},
    };

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,                0.0, 0.0, 1},
        {"WDRO-inject-K1",    true,  InjectionMode::DRO,                0.0, 0.0, 1},
        {"TopRisk-K1",        true,  InjectionMode::TOP_RISK_INJECT,    0.0, 0.0, 1},
    };

    std::string filepath = OUTPUT_DIR + "f11_path_geometry_high_speed.csv";
    std::ofstream ofs(filepath);
    ofs << "path,obstacle_speed,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& pe : paths) {
        for (double speed : SPEEDS) {
            for (const auto& method : METHODS) {
                std::cout << "  " << pe.name << " v=" << std::setprecision(1) << speed
                          << " " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();

                Metrics met;
                met.method = method.name;

                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 500 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        std::hash<std::string>{}(pe.name) / 1000 +
                        static_cast<int>(speed * 100) * 10000 +
                        static_cast<int>(method.injection_mode) * 100000 + i + 92000000);
                    std::mt19937 env_rng(seed);

                    ExperimentConfig cfg = make_base_config();
                    cfg = apply_method(cfg, method);
                    cfg.custom_ref_path = pe.setup.path;
                    cfg.custom_initial_ego = pe.setup.initial_ego;
                    cfg.initial_obstacle_states = {
                        obstacle_on_path(pe.setup.path, 0.55, env_rng, speed)
                    };

                    RolloutRecord rec = run_experiment_rollout(cfg, seed);
                    met.add(rec);
                }

                ofs << pe.name << "," << std::fixed << std::setprecision(1) << speed << ",";
                write_csv_row(ofs, met, N);

                std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                          << " [" << std::setprecision(1) << (met.coll_ci().first * 100) << "%, "
                          << std::setprecision(1) << (met.coll_ci().second * 100) << "%]"
                          << " clr=" << std::setprecision(2) << met.mean_clearance()
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}


// ============================================================================
// F5 Diagnostic: Test obstacle setup fixes at 2.0 m/s
// ============================================================================

static void run_f5_diag() {
    std::cout << "\n================================================================\n";
    std::cout << "  F5-diag: Obstacle setup fixes at 2.0 m/s (N=500)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 500;
    const double SPEED = 2.0;

    const std::vector<F5Method> METHODS = {
        F5Method::BASE, F5Method::MODE_DRO_SAMPLE,
        F5Method::MODE_DRO_INJECT_K1, F5Method::MODE_DRO_INJECT_K2,
        F5Method::TOPRISK_K1, F5Method::TOPRISK_K2,
    };

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    struct DiagSetup {
        std::string name;
        double switch_prob;
        double rare_prob;
        double arc_frac;
        bool tight_path;       // use tight S-curve (20m, amplitude 5)
        int num_obstacles;     // >1 = staggered oncoming obstacles
    };

    std::vector<DiagSetup> setups = {
        {"no_switch",           0.0, 0.0, 0.55, false, 1},
        {"closer_0.35",         0.2, 0.1, 0.35, false, 1},
        {"no_switch+closer",    0.0, 0.0, 0.35, false, 1},
        {"tight_path",          0.2, 0.1, 0.55, true,  1},
        {"tight_no_switch",     0.0, 0.0, 0.55, true,  1},
        {"two_obstacles",       0.2, 0.1, 0.55, false, 2},
        {"tight+two+no_sw",    0.0, 0.0, 0.55, true,  2},
    };

    auto tight_path = ReferencePath::create_s_curve(20.0, 5.0, 200);

    std::string filepath = OUTPUT_DIR + "f5_diag_obstacle_setup.csv";
    std::ofstream ofs(filepath);
    ofs << "setup,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& setup : setups) {
        const auto& path = setup.tight_path ? tight_path : ref_path;
        for (F5Method method : METHODS) {
            std::string mname = f5_method_name(method);
            std::cout << "  " << setup.name << " / " << mname << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = mname;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(setup.name) / 1000 +
                    static_cast<int>(method) * 100000 + i + 99000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(setup.switch_prob, setup.rare_prob,
                                                         setup.num_obstacles);
                cfg = apply_f5_method(cfg, method);
                cfg.custom_ref_path = path;
                cfg.custom_initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
                cfg.path_following_obstacles = true;

                std::vector<ObstacleState> obs_states;
                if (setup.num_obstacles == 1) {
                    obs_states.push_back(obstacle_on_path(path, setup.arc_frac, env_rng, SPEED));
                } else {
                    // Stagger obstacles along the path
                    obs_states.push_back(obstacle_on_path(path, 0.45, env_rng, SPEED));
                    obs_states.push_back(obstacle_on_path(path, 0.65, env_rng, SPEED));
                }
                cfg.initial_obstacle_states = obs_states;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << setup.name << ",";
            write_csv_row(ofs, met, N);

            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
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
    std::cout << "Focused Experiments for Statistical Significance\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("f1")) run_f1();
    if (run_all || filters.count("f2")) run_f2();
    if (run_all || filters.count("f3")) run_f3();
    if (run_all || filters.count("f4")) run_f4();
    if (run_all || filters.count("f5")) run_f5();
    if (run_all || filters.count("f6")) run_f6();
    if (run_all || filters.count("f7")) run_f7();
    if (run_all || filters.count("f8")) run_f8();
    if (run_all || filters.count("f9")) run_f9();
    if (run_all || filters.count("f10")) run_f10();
    if (run_all || filters.count("f11")) run_f11();
    if (filters.count("f5d")) run_f5_diag();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total << "s"
              << " (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";

    return 0;
}
