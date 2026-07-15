/**
 * @file test_collision_clearance.cpp
 * @brief Collision clearance and safety behavior tests across switching patterns,
 *        obstacle counts, environment types, and random seeds.
 *
 * Test suites:
 *   C1: Clearance vs switching probability (random seeds, per-method clearance distributions)
 *   C2: Clearance vs obstacle count (1-4 obstacles, independent + shared classes)
 *   C3: Clearance across environment types (Straight, Narrow, Intersection, Oncoming)
 *   C4: Multi-seed robustness (5 master seeds x all environments x all methods)
 *   C5: Clearance under rare-mode stress (rare_prob sweep with high switch_prob)
 *   C6: Combined stress test (environment x obstacle count x switching, reduced rollouts)
 *
 * Each suite records per-rollout clearance data for detailed statistical analysis.
 *
 * Usage:
 *   ./test_collision_clearance             # all suites
 *   ./test_collision_clearance c1          # single suite
 *   ./test_collision_clearance c1 c3 c4    # selected suites
 *
 * Output: paper_figures/c1_clearance_vs_switching.csv, etc.
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
#include <functional>
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

static const std::string OUTPUT_DIR = "paper_figures/";

static constexpr int    HORIZON         = DEFAULT_HORIZON;
static constexpr double DT              = DEFAULT_DT;
static constexpr int    NUM_SCENARIOS   = DEFAULT_BASE_SCENARIOS;
static constexpr int    ROLLOUT_STEPS   = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;
static constexpr int    N_ROLLOUTS      = 300;
static constexpr int    N_ROLLOUTS_HEAVY = 200;  // for combinatorial suites

static const std::vector<std::string> BASE_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

static const std::vector<unsigned> MASTER_SEEDS = {42, 1042, 2042, 3042, 4042};

// ============================================================================
// Method definitions
// ============================================================================

struct MethodDef {
    std::string name;
    bool enable_dro;
    InjectionMode injection_mode;
    int injection_count;
    bool safe_horizon;
};

static const std::vector<MethodDef> METHODS = {
    {"Base",            false, InjectionMode::NONE,             1, false},
    {"Base+SH",         false, InjectionMode::NONE,             1, true},
    {"DRO(inj)",        true,  InjectionMode::DRO,              1, false},
    {"DRO(inj)+SH",     true,  InjectionMode::DRO,              1, true},
    {"DRO(q*)+SH",      true,  InjectionMode::QSTAR_SAMPLE,     1, true},
    {"TopRisk+SH",      true,  InjectionMode::TOP_RISK_INJECT,  1, true},
};

static const std::vector<MethodDef> CORE_METHODS = {
    {"Base",            false, InjectionMode::NONE,             1, false},
    {"DRO(inj)",        true,  InjectionMode::DRO,              1, false},
    {"DRO(inj)+SH",     true,  InjectionMode::DRO,              1, true},
};

// ============================================================================
// Helpers
// ============================================================================

static double elapsed_sec(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static ExperimentConfig make_base_config(double switch_prob = 0.1, double rare_prob = 0.05,
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
    cfg.safe_horizon_enabled = false;
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
    cfg.dro_injection_count = m.injection_count;
    cfg.safe_horizon_enabled = m.safe_horizon;
    cfg.method_name = m.name;
    return cfg;
}

static double percentile_val(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    int idx = std::max(0, std::min((int)(p * (v.size() - 1)), (int)v.size() - 1));
    return v[idx];
}

static double mean_val(const std::vector<double>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

static double stddev_val(const std::vector<double>& v) {
    if (v.size() < 2) return 0;
    double m = mean_val(v);
    double sum = 0;
    for (double x : v) sum += (x - m) * (x - m);
    return std::sqrt(sum / (v.size() - 1));
}

// Aggregated clearance + collision stats per condition/method
struct ClearanceStats {
    std::string method;
    std::vector<bool> collisions;
    std::vector<double> min_clearances;
    std::vector<double> clearance_5pcts;
    std::vector<double> total_progress;
    std::vector<double> solve_times_ms;
    std::vector<int> collision_steps;

    int n() const { return (int)collisions.size(); }
    int coll_count() const { int c = 0; for (bool b : collisions) if (b) c++; return c; }
    double coll_rate() const { return n() > 0 ? (double)coll_count() / n() : 0; }
    std::pair<double,double> coll_ci() const { return wilson_ci(coll_count(), n()); }

    double mean_clearance() const { return mean_val(min_clearances); }
    double p5_clearance() const { return percentile_val(min_clearances, 0.05); }
    double p25_clearance() const { return percentile_val(min_clearances, 0.25); }
    double median_clearance() const { return percentile_val(min_clearances, 0.50); }
    double stddev_clearance() const { return stddev_val(min_clearances); }
    double mean_progress() const { return mean_val(total_progress); }
    double mean_solve_ms() const { return mean_val(solve_times_ms); }

    // Mean collision step among collisions (how early do collisions happen?)
    double mean_collision_step() const {
        int sum = 0, cnt = 0;
        for (int s : collision_steps) {
            if (s >= 0) { sum += s; cnt++; }
        }
        return cnt > 0 ? (double)sum / cnt : -1;
    }

    void add(const RolloutRecord& rec) {
        collisions.push_back(rec.collision);
        min_clearances.push_back(rec.min_clearance);
        clearance_5pcts.push_back(rec.clearance_5pct);
        total_progress.push_back(rec.total_progress);
        solve_times_ms.push_back(rec.avg_solve_ms);
        collision_steps.push_back(rec.collision_step);
    }
};

// Write a CSV row with clearance-focused columns
static void write_clearance_row(std::ofstream& ofs, const ClearanceStats& st, int n_rollouts,
                                 const std::string& prefix = "") {
    auto [clo, chi] = st.coll_ci();
    ofs << prefix
        << st.method << ","
        << n_rollouts << ","
        << std::fixed << std::setprecision(6)
        << st.coll_rate() << "," << clo << "," << chi << ","
        << std::setprecision(4)
        << st.mean_clearance() << ","
        << st.median_clearance() << ","
        << st.p5_clearance() << ","
        << st.p25_clearance() << ","
        << st.stddev_clearance() << ","
        << std::setprecision(2)
        << st.mean_collision_step() << ","
        << std::setprecision(4)
        << st.mean_progress() << ","
        << std::setprecision(3)
        << st.mean_solve_ms() << "\n";
    ofs.flush();
}

static const std::string CLEARANCE_HEADER =
    "method,n_rollouts,collision_rate,coll_ci_lo,coll_ci_hi,"
    "mean_min_clearance,median_min_clearance,p5_min_clearance,p25_min_clearance,"
    "stddev_min_clearance,mean_collision_step,mean_progress,mean_solve_ms\n";

// ============================================================================
// C1: Clearance vs Switching Probability
// ============================================================================

static void run_c1() {
    std::cout << "\n================================================================\n";
    std::cout << "  C1: Clearance vs Switching Probability\n";
    std::cout << "  switch_prob: {0.02, 0.05, 0.10, 0.20, 0.35, 0.50}\n";
    std::cout << "  Methods: " << METHODS.size() << ", Rollouts: " << N_ROLLOUTS << "\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> SWITCH_PROBS = {0.02, 0.05, 0.10, 0.20, 0.35, 0.50};

    std::string filepath = OUTPUT_DIR + "c1_clearance_vs_switching.csv";
    std::ofstream ofs(filepath);
    ofs << "switch_prob," << CLEARANCE_HEADER;

    auto ref_path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);

    for (double sp : SWITCH_PROBS) {
        for (const auto& method : METHODS) {
            std::cout << "  sp=" << std::fixed << std::setprecision(2) << sp
                      << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            ClearanceStats st;
            st.method = method.name;

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(sp * 1000) * 1000000 +
                    static_cast<int>(method.injection_mode) * 100000 +
                    (method.safe_horizon ? 50000 : 0) + i + 10000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(sp);
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {
                    obstacle_on_s_curve(ref_path, OBS_PATH_FRACTION, env_rng)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                st.add(rec);
            }

            ofs << std::fixed << std::setprecision(2) << sp << ",";
            write_clearance_row(ofs, st, N_ROLLOUTS);

            auto [clo, chi] = st.coll_ci();
            std::cout << N_ROLLOUTS
                      << " coll=" << std::setprecision(1) << (st.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(3) << st.mean_clearance()
                      << " p5=" << st.p5_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// C2: Clearance vs Obstacle Count
// ============================================================================

static void run_c2() {
    std::cout << "\n================================================================\n";
    std::cout << "  C2: Clearance vs Obstacle Count\n";
    std::cout << "  Configs: 1obs, 2obs-ind, 3obs-ind, 4obs-ind, 4obs-shared\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    struct ObsConfig {
        std::string tag;
        int num_obs;
        int per_class;
        std::vector<double> arc_fracs;
    };

    const std::vector<ObsConfig> configs = {
        {"1obs",        1, 1, {0.50}},
        {"2obs-ind",    2, 1, {0.40, 0.60}},
        {"3obs-ind",    3, 1, {0.30, 0.45, 0.65}},
        {"4obs-ind",    4, 1, {0.25, 0.38, 0.52, 0.68}},
        {"4obs-shared", 4, 4, {0.25, 0.38, 0.52, 0.68}},
    };

    std::string filepath = OUTPUT_DIR + "c2_clearance_vs_obstacles.csv";
    std::ofstream ofs(filepath);
    ofs << "obs_config,num_obstacles,class_sharing," << CLEARANCE_HEADER;

    auto ref_path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);

    for (const auto& oc : configs) {
        for (const auto& method : METHODS) {
            std::cout << "  " << oc.tag << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            ClearanceStats st;
            st.method = method.name;

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    oc.num_obs * 1000000 + oc.per_class * 500000 +
                    static_cast<int>(method.injection_mode) * 100000 +
                    (method.safe_horizon ? 50000 : 0) + i + 20000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.15, 0.05, oc.num_obs);
                cfg = apply_method(cfg, method);
                cfg.obstacles_per_class = oc.per_class;

                std::vector<ObstacleState> obs_states;
                for (double frac : oc.arc_fracs) {
                    obs_states.push_back(obstacle_on_s_curve(ref_path, frac, env_rng));
                }
                cfg.initial_obstacle_states = obs_states;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                st.add(rec);
            }

            std::string sharing = (oc.per_class == oc.num_obs) ? "shared" : "independent";
            ofs << oc.tag << "," << oc.num_obs << "," << sharing << ",";
            write_clearance_row(ofs, st, N_ROLLOUTS);

            std::cout << N_ROLLOUTS
                      << " coll=" << std::setprecision(1) << (st.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(3) << st.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// C3: Clearance Across Environment Types
// ============================================================================

static void run_c3() {
    std::cout << "\n================================================================\n";
    std::cout << "  C3: Clearance Across Environment Types\n";
    std::cout << "  Environments: Straight, Narrow, Intersection, Oncoming\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<EnvironmentType> ENVS = {
        EnvironmentType::STRAIGHT,
        EnvironmentType::NARROW_CORRIDOR,
        EnvironmentType::INTERSECTION,
        EnvironmentType::ONCOMING,
    };

    std::string filepath = OUTPUT_DIR + "c3_clearance_vs_environment.csv";
    std::ofstream ofs(filepath);
    ofs << "environment," << CLEARANCE_HEADER;

    for (EnvironmentType env : ENVS) {
        std::string env_name = environment_name(env);

        for (const auto& method : METHODS) {
            std::cout << "  " << env_name << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            ClearanceStats st;
            st.method = method.name;

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(env) * 1000000 +
                    static_cast<int>(method.injection_mode) * 100000 +
                    (method.safe_horizon ? 50000 : 0) + i + 30000000);
                std::mt19937 env_rng(seed);

                EnvironmentSetup env_setup = create_environment(env, env_rng);

                ExperimentConfig cfg = make_base_config(0.15, 0.05);
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obs_modes = env_setup.obs_modes;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                st.add(rec);
            }

            ofs << env_name << ",";
            write_clearance_row(ofs, st, N_ROLLOUTS);

            std::cout << N_ROLLOUTS
                      << " coll=" << std::setprecision(1) << (st.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(3) << st.mean_clearance()
                      << " p5=" << st.p5_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// C4: Multi-Seed Robustness
// ============================================================================

static void run_c4() {
    std::cout << "\n================================================================\n";
    std::cout << "  C4: Multi-Seed Robustness\n";
    std::cout << "  5 master seeds x 4 environments x " << CORE_METHODS.size() << " methods\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<EnvironmentType> ENVS = {
        EnvironmentType::STRAIGHT,
        EnvironmentType::NARROW_CORRIDOR,
        EnvironmentType::INTERSECTION,
        EnvironmentType::ONCOMING,
    };

    std::string filepath = OUTPUT_DIR + "c4_multi_seed_robustness.csv";
    std::ofstream ofs(filepath);
    ofs << "master_seed,environment," << CLEARANCE_HEADER;

    for (unsigned master_seed : MASTER_SEEDS) {
        for (EnvironmentType env : ENVS) {
            std::string env_name = environment_name(env);

            for (const auto& method : CORE_METHODS) {
                std::cout << "  seed=" << master_seed << " " << env_name
                          << " / " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();

                ClearanceStats st;
                st.method = method.name;

                for (int i = 0; i < N_ROLLOUTS_HEAVY; ++i) {
                    if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                    SeedBundle seeds = derive_seeds(master_seed, i);
                    unsigned seed = seeds.env + static_cast<unsigned>(env) * 7919;
                    std::mt19937 env_rng(seed);

                    EnvironmentSetup env_setup = create_environment(env, env_rng);

                    ExperimentConfig cfg = make_base_config(0.15, 0.05);
                    cfg = apply_method(cfg, method);
                    cfg.initial_obstacle_states = {env_setup.initial_obs};
                    cfg.obs_modes = env_setup.obs_modes;

                    RolloutRecord rec = run_experiment_rollout(cfg, seed);
                    st.add(rec);
                }

                ofs << master_seed << "," << env_name << ",";
                write_clearance_row(ofs, st, N_ROLLOUTS_HEAVY);

                std::cout << N_ROLLOUTS_HEAVY
                          << " coll=" << std::setprecision(1) << (st.coll_rate() * 100) << "%"
                          << " clr=" << std::setprecision(3) << st.mean_clearance()
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// C5: Clearance Under Rare-Mode Stress
// ============================================================================

static void run_c5() {
    std::cout << "\n================================================================\n";
    std::cout << "  C5: Clearance Under Rare-Mode Stress\n";
    std::cout << "  rare_prob x switch_prob interaction\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> RARE_PROBS   = {0.01, 0.05, 0.10, 0.20};
    const std::vector<double> SWITCH_PROBS = {0.05, 0.15, 0.30};

    std::string filepath = OUTPUT_DIR + "c5_clearance_rare_stress.csv";
    std::ofstream ofs(filepath);
    ofs << "rare_prob,switch_prob," << CLEARANCE_HEADER;

    auto ref_path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);

    for (double rp : RARE_PROBS) {
        for (double sp : SWITCH_PROBS) {
            for (const auto& method : CORE_METHODS) {
                std::cout << "  rp=" << std::fixed << std::setprecision(2) << rp
                          << " sp=" << sp << " / " << method.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();

                ClearanceStats st;
                st.method = method.name;

                for (int i = 0; i < N_ROLLOUTS; ++i) {
                    if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        static_cast<int>(rp * 1000) * 10000000 +
                        static_cast<int>(sp * 1000) * 1000000 +
                        static_cast<int>(method.injection_mode) * 100000 +
                        (method.safe_horizon ? 50000 : 0) + i + 50000000);
                    std::mt19937 env_rng(seed);

                    ExperimentConfig cfg = make_base_config(sp, rp);
                    cfg = apply_method(cfg, method);
                    cfg.initial_obstacle_states = {
                        obstacle_on_s_curve(ref_path, OBS_PATH_FRACTION, env_rng)
                    };

                    RolloutRecord rec = run_experiment_rollout(cfg, seed);
                    st.add(rec);
                }

                ofs << std::fixed << std::setprecision(2) << rp << ","
                    << sp << ",";
                write_clearance_row(ofs, st, N_ROLLOUTS);

                std::cout << N_ROLLOUTS
                          << " coll=" << std::setprecision(1) << (st.coll_rate() * 100) << "%"
                          << " clr=" << std::setprecision(3) << st.mean_clearance()
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// C6: Combined Stress Test (environment x obstacles x switching)
// ============================================================================

static void run_c6() {
    std::cout << "\n================================================================\n";
    std::cout << "  C6: Combined Stress Test\n";
    std::cout << "  environment x obstacle_count x switch_prob\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<EnvironmentType> ENVS = {
        EnvironmentType::STRAIGHT,
        EnvironmentType::INTERSECTION,
        EnvironmentType::ONCOMING,
    };
    const std::vector<int> OBS_COUNTS = {1, 2, 3};
    const std::vector<double> SWITCH_PROBS = {0.05, 0.20, 0.40};

    // Additional obstacle arc fractions for multi-obstacle in each env
    const std::vector<std::vector<double>> EXTRA_FRACS = {
        {},             // 1 obs: use env default only
        {0.55},         // 2 obs: env default + one extra
        {0.45, 0.65},   // 3 obs: env default + two extra
    };

    std::string filepath = OUTPUT_DIR + "c6_combined_stress.csv";
    std::ofstream ofs(filepath);
    ofs << "environment,num_obstacles,switch_prob," << CLEARANCE_HEADER;

    auto ref_path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);

    int total_cells = ENVS.size() * OBS_COUNTS.size() * SWITCH_PROBS.size() * CORE_METHODS.size();
    int cell = 0;

    for (EnvironmentType env : ENVS) {
        std::string env_name = environment_name(env);

        for (size_t oi = 0; oi < OBS_COUNTS.size(); ++oi) {
            int n_obs = OBS_COUNTS[oi];

            for (double sp : SWITCH_PROBS) {
                for (const auto& method : CORE_METHODS) {
                    cell++;
                    std::cout << "  [" << cell << "/" << total_cells << "] "
                              << env_name << "/" << n_obs << "obs/sp=" << std::fixed
                              << std::setprecision(2) << sp
                              << " / " << method.name << ": ";
                    std::cout.flush();
                    auto t1 = std::chrono::steady_clock::now();

                    ClearanceStats st;
                    st.method = method.name;

                    for (int i = 0; i < N_ROLLOUTS_HEAVY; ++i) {
                        if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                        unsigned seed = static_cast<unsigned>(
                            static_cast<int>(env) * 100000000 +
                            n_obs * 10000000 +
                            static_cast<int>(sp * 1000) * 1000000 +
                            static_cast<int>(method.injection_mode) * 100000 +
                            (method.safe_horizon ? 50000 : 0) + i + 60000000);
                        std::mt19937 env_rng(seed);

                        // Primary obstacle from environment setup
                        EnvironmentSetup env_setup = create_environment(env, env_rng);

                        ExperimentConfig cfg = make_base_config(sp, 0.05, n_obs);
                        cfg = apply_method(cfg, method);
                        cfg.obs_modes = env_setup.obs_modes;

                        // Build obstacle list: env obstacle + extras on S-curve
                        std::vector<ObstacleState> obs_states = {env_setup.initial_obs};
                        for (double frac : EXTRA_FRACS[oi]) {
                            obs_states.push_back(obstacle_on_s_curve(ref_path, frac, env_rng));
                        }
                        cfg.initial_obstacle_states = obs_states;

                        RolloutRecord rec = run_experiment_rollout(cfg, seed);
                        st.add(rec);
                    }

                    ofs << env_name << "," << n_obs << ","
                        << std::fixed << std::setprecision(2) << sp << ",";
                    write_clearance_row(ofs, st, N_ROLLOUTS_HEAVY);

                    std::cout << N_ROLLOUTS_HEAVY
                              << " coll=" << std::setprecision(1) << (st.coll_rate() * 100) << "%"
                              << " clr=" << std::setprecision(3) << st.mean_clearance()
                              << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
                }
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
    fs::create_directories(OUTPUT_DIR);

    std::set<std::string> filters;
    for (int i = 1; i < argc; ++i) filters.insert(argv[i]);
    bool run_all = filters.empty();

    std::cout << "========================================\n";
    std::cout << "Collision Clearance Tests\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("c1")) run_c1();
    if (run_all || filters.count("c2")) run_c2();
    if (run_all || filters.count("c3")) run_c3();
    if (run_all || filters.count("c4")) run_c4();
    if (run_all || filters.count("c5")) run_c5();
    if (run_all || filters.count("c6")) run_c6();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total << "s"
              << " (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";

    return 0;
}
