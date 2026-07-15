/**
 * @file test_cdc_experiments.cpp
 * @brief CDC paper experiments with feasible-by-construction Q* recovery.
 *
 * Experiments:
 *   E1: Main method comparison (Base / WDRO-sampling / WDRO-injection) across environments
 *   E2: Injection count sweep (K)
 *   E3: Risk lift validation
 *   E4: Runtime measurement
 *   E5: Feasibility tightness (cost/rho)
 *   B:  Baseline comparison (Uniform / Softmax / eps-greedy / top-risk-inject)
 *   A1: Ground cost ablation (W2 Bures / 0-1 / Euclidean mean)
 *   A2: Radius rho sweep
 *
 * Output: figures/cdc/
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

using namespace scenario_mpc;
namespace fs = std::filesystem;

// ============================================================================
// Constants
// ============================================================================

static const std::string OUTPUT_DIR = "figures/cdc/";

static constexpr int    HORIZON         = DEFAULT_HORIZON;
static constexpr double DT              = DEFAULT_DT;
static constexpr int    NUM_SCENARIOS   = DEFAULT_BASE_SCENARIOS;
static constexpr int    ROLLOUT_STEPS   = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;

static const std::vector<std::string> OBS_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Metrics (same as test_dro_benefits)
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
    double clearance_percentile(double p) const {
        if (min_clearances.empty()) return 0;
        auto v = min_clearances;
        std::sort(v.begin(), v.end());
        int idx = std::max(0, std::min(static_cast<int>(v.size()) - 1,
                  static_cast<int>(p * (v.size() - 1))));
        return v[idx];
    }
    double p5_clearance() const { return clearance_percentile(0.05); }
    double p10_clearance() const { return clearance_percentile(0.10); }
    double p25_clearance() const { return clearance_percentile(0.25); }
    double p50_clearance() const { return clearance_percentile(0.50); }
    double p75_clearance() const { return clearance_percentile(0.75); }
    double p90_clearance() const { return clearance_percentile(0.90); }
    double p95_clearance() const { return clearance_percentile(0.95); }
    double std_clearance() const {
        if (min_clearances.size() < 2) return 0;
        double m = mean_clearance();
        double sum = 0;
        for (double c : min_clearances) sum += (c - m) * (c - m);
        return std::sqrt(sum / (min_clearances.size() - 1));
    }
    // Mean clearance for collision vs non-collision runs separately
    std::pair<double, double> clearance_by_collision() const {
        double sum_coll = 0, sum_nocoll = 0;
        int n_coll = 0, n_nocoll = 0;
        for (size_t i = 0; i < min_clearances.size() && i < collisions.size(); ++i) {
            if (collisions[i]) { sum_coll += min_clearances[i]; n_coll++; }
            else               { sum_nocoll += min_clearances[i]; n_nocoll++; }
        }
        return {n_coll > 0 ? sum_coll / n_coll : 0.0,
                n_nocoll > 0 ? sum_nocoll / n_nocoll : 0.0};
    }
    // Bootstrap CI on mean clearance
    std::pair<double, double> clearance_mean_ci(int n_boot = 5000, unsigned boot_seed = 42) const {
        if (min_clearances.size() < 2) return {0, 0};
        std::mt19937 rng(boot_seed);
        std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(min_clearances.size()) - 1);
        std::vector<double> boot_means(n_boot);
        for (int b = 0; b < n_boot; ++b) {
            double s = 0;
            for (size_t j = 0; j < min_clearances.size(); ++j) {
                s += min_clearances[idx_dist(rng)];
            }
            boot_means[b] = s / min_clearances.size();
        }
        std::sort(boot_means.begin(), boot_means.end());
        return {boot_means[static_cast<int>(0.025 * n_boot)],
                boot_means[static_cast<int>(0.975 * n_boot)]};
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
};

static ExperimentConfig make_base_config(double switch_prob = 0.2, double rare_prob = 0.1,
                                          int num_obs = 1) {
    ExperimentConfig cfg;
    cfg.horizon = HORIZON;
    cfg.num_scenarios = NUM_SCENARIOS;
    cfg.switch_prob = switch_prob;
    cfg.rollout_steps = ROLLOUT_STEPS;
    cfg.obs_modes = OBS_MODES;
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
// E1: Main Method Comparison across environments
// ============================================================================

static void run_e1() {
    std::cout << "\n================================================================\n";
    std::cout << "  E1: Main Method Comparison\n";
    std::cout << "  Environments: Straight, Narrow, Intersection, Oncoming\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<EnvironmentType> ENVS = {
        EnvironmentType::STRAIGHT, EnvironmentType::NARROW_CORRIDOR,
        EnvironmentType::INTERSECTION, EnvironmentType::ONCOMING
    };
    const std::vector<std::string> ENV_NAMES = {"Straight", "Narrow", "Intersection", "Oncoming"};
    const std::vector<MethodDef> METHODS = {
        {"Base",            false, InjectionMode::NONE,         0.0, 0.0},
        {"WDRO-sampling",   true,  InjectionMode::QSTAR_SAMPLE, 0.0, 0.0},
        {"WDRO-injection",  true,  InjectionMode::DRO,          0.0, 0.0},
    };
    const int N = 1000;

    std::string filepath = OUTPUT_DIR + "cdc_e1_method_comparison.csv";
    std::ofstream ofs(filepath);
    ofs << "environment,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (size_t e = 0; e < ENVS.size(); ++e) {
        for (const auto& method : METHODS) {
            std::cout << "  " << ENV_NAMES[e] << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(e * 1000000 + static_cast<int>(method.injection_mode) * 100000 + i);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(ENVS[e], env_rng);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obs_modes = env_setup.obs_modes;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << ENV_NAMES[e] << ",";
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
// E2: Injection Count Sweep (rerun with feasible recovery)
// ============================================================================

static void run_e2() {
    std::cout << "\n================================================================\n";
    std::cout << "  E2: Injection Count Sweep (K)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> K_VALUES = {0, 1, 2, 3, 5, -1};
    const int N = 1000;

    std::string filepath = OUTPUT_DIR + "cdc_e2_injection_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "K,K_label,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (int K : K_VALUES) {
        std::string k_label = (K < 0) ? "all" : std::to_string(K);
        std::cout << "  K=" << k_label << ": ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        Metrics met;
        met.method = "DRO_K" + k_label;

        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(K * 100000 + i + 7000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig cfg = make_base_config();
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;

            if (K == 0) {
                cfg.enable_dro = false;
                cfg.injection_mode = InjectionMode::NONE;
            } else {
                cfg.enable_dro = true;
                cfg.injection_mode = InjectionMode::DRO;
                cfg.dro_injection_count = K;
            }
            cfg.method_name = met.method;

            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        ofs << K << "," << k_label << ",";
        write_csv_row(ofs, met, N);

        std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                  << " mm=" << (met.mm_rate() * 100) << "%"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// E4: Runtime measurement
// ============================================================================

static void run_e4() {
    std::cout << "\n================================================================\n";
    std::cout << "  E4: Runtime Measurement\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,              0.0, 0.0},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,     0.0, 0.0},
        {"WDRO-injection",    true,  InjectionMode::DRO,              0.0, 0.0},
        {"Uniform-coverage",  true,  InjectionMode::UNIFORM_COVERAGE, 0.0, 0.0},
        {"Softmax-risk",      true,  InjectionMode::SOFTMAX_RISK,     5.0, 0.0},
        {"Eps-greedy",        true,  InjectionMode::EPSILON_GREEDY_INJ, 0.0, 0.3},
        {"Top-risk-inject",   true,  InjectionMode::TOP_RISK_INJECT,  0.0, 0.0},
    };
    const int N = 500;

    std::string filepath = OUTPUT_DIR + "cdc_e4_runtime.csv";
    std::ofstream ofs(filepath);
    ofs << "method,mean_solve_ms,p50_solve_ms,p95_solve_ms,max_solve_ms,n_rollouts\n";

    for (const auto& method : METHODS) {
        std::cout << "  " << method.name << ": ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        std::vector<double> all_avg, all_p50, all_p95, all_max;

        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(static_cast<int>(method.injection_mode) * 100000 + i + 9000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig cfg = make_base_config();
            cfg = apply_method(cfg, method);
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;

            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            all_avg.push_back(rec.avg_solve_ms);
            all_p50.push_back(rec.p50_solve_ms);
            all_p95.push_back(rec.p95_solve_ms);
            all_max.push_back(rec.max_solve_ms);
        }

        auto mean = [](const std::vector<double>& v) {
            return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        ofs << method.name << ","
            << std::fixed << std::setprecision(3)
            << mean(all_avg) << "," << mean(all_p50) << ","
            << mean(all_p95) << "," << mean(all_max) << ","
            << N << "\n";
        ofs.flush();

        std::cout << N << " avg=" << std::setprecision(2) << mean(all_avg) << "ms"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// B: Baseline Comparison (all baselines vs WDRO methods)
// ============================================================================

static void run_baselines() {
    std::cout << "\n================================================================\n";
    std::cout << "  B: Baseline Comparison (Oncoming, switch_prob=0.2)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,              0.0, 0.0},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,     0.0, 0.0},
        {"WDRO-injection",    true,  InjectionMode::DRO,              0.0, 0.0},
        {"Uniform-coverage",  true,  InjectionMode::UNIFORM_COVERAGE, 0.0, 0.0},
        {"Softmax-risk",      true,  InjectionMode::SOFTMAX_RISK,     5.0, 0.0},
        {"Eps-greedy",        true,  InjectionMode::EPSILON_GREEDY_INJ, 0.0, 0.3},
        {"Top-risk-inject",   true,  InjectionMode::TOP_RISK_INJECT,  0.0, 0.0},
    };
    const int N = 1000;

    std::string filepath = OUTPUT_DIR + "cdc_baselines.csv";
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
            if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(static_cast<int>(method.injection_mode) * 100000 + i + 8000000);
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
                  << " mm=" << (met.mm_rate() * 100) << "%"
                  << " clr=" << std::setprecision(2) << met.mean_clearance()
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// A1: Ground Cost Ablation
// ============================================================================

static void run_a1() {
    std::cout << "\n================================================================\n";
    std::cout << "  A1: Ground Cost Ablation (Oncoming, WDRO-injection)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    struct GCDef { std::string name; DROGroundCostType type; };
    const std::vector<GCDef> COSTS = {
        {"W2-Bures",       DROGroundCostType::W2_BURES},
        {"Zero-One",       DROGroundCostType::ZERO_ONE},
        {"Euclidean-Mean", DROGroundCostType::EUCLIDEAN_MEAN},
    };
    const int N = 1000;

    std::string filepath = OUTPUT_DIR + "cdc_a1_ground_cost.csv";
    std::ofstream ofs(filepath);
    ofs << "ground_cost,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& gc : COSTS) {
        std::cout << "  " << gc.name << ": ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        Metrics met;
        met.method = gc.name;

        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(static_cast<int>(gc.type) * 100000 + i + 6000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig cfg = make_base_config();
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::DRO;
            cfg.ground_cost = gc.type;
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;
            cfg.method_name = gc.name;

            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        ofs << gc.name << ",";
        write_csv_row(ofs, met, N);

        std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                  << " mm=" << (met.mm_rate() * 100) << "%"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// A2: Radius rho sweep
// ============================================================================

static void run_a2() {
    std::cout << "\n================================================================\n";
    std::cout << "  A2: Radius rho Sweep (Oncoming, WDRO-injection)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> RHO_VALUES = {0.01, 0.05, 0.1, 0.2, 0.3, 0.5};
    const int N = 1000;

    std::string filepath = OUTPUT_DIR + "cdc_a2_rho_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "rho,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double rho : RHO_VALUES) {
        std::cout << "  rho=" << std::setprecision(2) << rho << ": ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        std::string label = "rho=" + std::to_string(rho).substr(0, 4);
        Metrics met;
        met.method = label;

        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(static_cast<int>(rho * 1000) * 100000 + i + 5000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig cfg = make_base_config();
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::DRO;
            cfg.eps_wass = rho;
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;
            cfg.method_name = label;

            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        ofs << std::fixed << std::setprecision(2) << rho << ",";
        write_csv_row(ofs, met, N);

        std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                  << " mm=" << (met.mm_rate() * 100) << "%"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Generic D-experiment runner parameterized by environment
// ============================================================================

static void run_d1_env(EnvironmentType env_type, const std::string& env_label,
                        const std::string& suffix) {
    std::cout << "\n================================================================\n";
    std::cout << "  D1" << suffix << ": WDRO-K vs TopRisk-K vs DiverseRisk-K (" << env_label << ")\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> K_VALUES = {1, 2, 3, 4};
    const int N = 5000;

    std::string filepath = OUTPUT_DIR + "cdc_d1_selection" + suffix + ".csv";
    std::ofstream ofs(filepath);
    ofs << "K,selector,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (int K : K_VALUES) {
        struct SelectorDef { std::string name; InjectionMode mode; };
        std::vector<SelectorDef> selectors = {
            {"WDRO",        InjectionMode::DRO},
            {"TopRisk",     InjectionMode::TOP_RISK_INJECT},
            {"DiverseRisk", InjectionMode::DIVERSE_RISK_INJECT},
        };

        for (const auto& sel : selectors) {
            std::string label = sel.name + "_K" + std::to_string(K);
            std::cout << "  " << label << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = label;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(K * 1000000 + static_cast<int>(sel.mode) * 100000 + i + 10000000);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(env_type, env_rng);

                ExperimentConfig cfg = make_base_config();
                cfg.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obs_modes = env_setup.obs_modes;
                cfg.enable_dro = true;
                cfg.injection_mode = sel.mode;
                cfg.dro_injection_count = K;
                cfg.method_name = label;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << K << "," << sel.name << ",";
            write_csv_row(ofs, met, N);

            auto [clo, chi] = met.coll_ci();
            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " [" << std::setprecision(1) << (clo * 100) << "," << (chi * 100) << "]"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

static void run_d2_env(EnvironmentType env_type, const std::string& env_label,
                        const std::string& suffix) {
    std::cout << "\n================================================================\n";
    std::cout << "  D2" << suffix << ": WDRO-sampling vs Softmax (" << env_label << ")\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> TAU_VALUES = {1.0, 3.0, 5.0, 10.0, 20.0};
    const int N = 1000;

    std::string filepath = OUTPUT_DIR + "cdc_d2_sampling" + suffix + ".csv";
    std::ofstream ofs(filepath);
    ofs << "variant,tau,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    // WDRO-sampling
    {
        std::cout << "  WDRO-sampling: ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();
        Metrics met; met.method = "WDRO-sampling";
        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(i + 11000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(env_type, env_rng);
            ExperimentConfig cfg = make_base_config();
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::QSTAR_SAMPLE;
            cfg.method_name = "WDRO-sampling";
            met.add(run_experiment_rollout(cfg, seed));
        }
        ofs << "WDRO,0,"; write_csv_row(ofs, met, N);
        std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate()*100) << "%"
                  << " mm=" << (met.mm_rate()*100) << "% (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    // Softmax sweep
    for (double tau : TAU_VALUES) {
        std::string label = "Softmax-tau" + std::to_string(static_cast<int>(tau));
        std::cout << "  " << label << ": ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();
        Metrics met; met.method = label;
        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(static_cast<int>(tau * 100) * 100000 + i + 11000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(env_type, env_rng);
            ExperimentConfig cfg = make_base_config();
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::SOFTMAX_RISK;
            cfg.softmax_tau = tau;
            cfg.method_name = label;
            met.add(run_experiment_rollout(cfg, seed));
        }
        ofs << "Softmax," << std::setprecision(1) << tau << ","; write_csv_row(ofs, met, N);
        std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate()*100) << "%"
                  << " mm=" << (met.mm_rate()*100) << "% (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    // Base
    {
        std::cout << "  Base: ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();
        Metrics met; met.method = "Base";
        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(i + 11000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(env_type, env_rng);
            ExperimentConfig cfg = make_base_config();
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;
            cfg.enable_dro = false;
            cfg.injection_mode = InjectionMode::NONE;
            cfg.method_name = "Base";
            met.add(run_experiment_rollout(cfg, seed));
        }
        ofs << "Base,0,"; write_csv_row(ofs, met, N);
        std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate()*100) << "%"
                  << " mm=" << (met.mm_rate()*100) << "% (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

static void run_d3_env(EnvironmentType env_type, const std::string& env_label,
                        const std::string& suffix) {
    std::cout << "\n================================================================\n";
    std::cout << "  D3" << suffix << ": Low-Observation Regime (" << env_label << ")\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> HIST_VALUES = {5, 10, 20, 50, 100, 150};
    const int N = 1000;

    struct MethodD3 { std::string name; bool dro; InjectionMode mode; };
    const std::vector<MethodD3> METHODS = {
        {"Base",            false, InjectionMode::NONE},
        {"WDRO-injection",  true,  InjectionMode::DRO},
        {"TopRisk-inject",  true,  InjectionMode::TOP_RISK_INJECT},
        {"WDRO-sampling",   true,  InjectionMode::QSTAR_SAMPLE},
    };

    std::string filepath = OUTPUT_DIR + "cdc_d3_observation" + suffix + ".csv";
    std::ofstream ofs(filepath);
    ofs << "max_history,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (int hist_len : HIST_VALUES) {
        for (const auto& method : METHODS) {
            std::string label = method.name + "_H" + std::to_string(hist_len);
            std::cout << "  " << label << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = method.name;
            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(hist_len * 100000 + static_cast<int>(method.mode) * 10000 + i + 12000000);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(env_type, env_rng);
                ExperimentConfig cfg = make_base_config();
                cfg.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obs_modes = env_setup.obs_modes;
                cfg.enable_dro = method.dro;
                cfg.injection_mode = method.mode;
                cfg.max_history_length = hist_len;
                cfg.method_name = method.name;
                met.add(run_experiment_rollout(cfg, seed));
            }
            ofs << hist_len << ","; write_csv_row(ofs, met, N);
            auto [clo, chi] = met.coll_ci();
            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate()*100) << "%"
                      << " [" << std::setprecision(1) << (clo*100) << "," << (chi*100) << "]"
                      << " mm=" << (met.mm_rate()*100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// D4: Ground-cost ablation for WDRO-K2 injection and WDRO-sampling coverage
// ============================================================================

static void run_d4() {
    std::cout << "\n================================================================\n";
    std::cout << "  D4: Ground-Cost Ablation (WDRO-K2 + WDRO-sampling)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    struct GCDef { std::string name; DROGroundCostType type; };
    const std::vector<GCDef> COSTS = {
        {"W2-Bures",       DROGroundCostType::W2_BURES},
        {"Zero-One",       DROGroundCostType::ZERO_ONE},
        {"Euclidean-Mean", DROGroundCostType::EUCLIDEAN_MEAN},
    };

    struct MethodD4 { std::string name; InjectionMode mode; int K; };
    const std::vector<MethodD4> METHODS = {
        {"WDRO-K2",       InjectionMode::DRO,          2},
        {"WDRO-K3",       InjectionMode::DRO,          3},
        {"WDRO-sampling", InjectionMode::QSTAR_SAMPLE, 1},
    };

    const int N = 1000;

    std::string filepath = OUTPUT_DIR + "cdc_d4_ground_cost_ablation.csv";
    std::ofstream ofs(filepath);
    ofs << "ground_cost,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& gc : COSTS) {
        for (const auto& method : METHODS) {
            std::string label = method.name + "/" + gc.name;
            std::cout << "  " << label << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(gc.type) * 1000000 + static_cast<int>(method.mode) * 100000 + i + 13000000);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

                ExperimentConfig cfg = make_base_config();
                cfg.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obs_modes = env_setup.obs_modes;
                cfg.enable_dro = true;
                cfg.injection_mode = method.mode;
                cfg.dro_injection_count = method.K;
                cfg.ground_cost = gc.type;
                cfg.method_name = method.name;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << gc.name << ",";
            write_csv_row(ofs, met, N);

            auto [clo, chi] = met.coll_ci();
            std::cout << N << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// C1: Clearance Analysis — full distribution comparison across methods
//
// Even when collision rate CIs overlap, clearance distributions may differ
// significantly. This experiment outputs:
//   - Full percentile ladder (p5, p10, p25, p50, p75, p90, p95)
//   - Mean + std + bootstrap 95% CI on mean clearance
//   - Conditional clearance: collision runs vs non-collision runs
//   - Per-method clearance CDFs (raw values for external plotting)
//
// Runs E1 methods across all 4 environments, 1000 rollouts each.
// ============================================================================

static void run_c1() {
    std::cout << "\n================================================================\n";
    std::cout << "  C1: Clearance Distribution Analysis\n";
    std::cout << "  All environments × {Base, WDRO-sampling, WDRO-injection}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<EnvironmentType> ENVS = {
        EnvironmentType::STRAIGHT, EnvironmentType::NARROW_CORRIDOR,
        EnvironmentType::INTERSECTION, EnvironmentType::ONCOMING
    };
    const std::vector<std::string> ENV_NAMES = {"Straight", "Narrow", "Intersection", "Oncoming"};
    const std::vector<MethodDef> METHODS = {
        {"Base",            false, InjectionMode::NONE,         0.0, 0.0},
        {"WDRO-sampling",   true,  InjectionMode::QSTAR_SAMPLE, 0.0, 0.0},
        {"WDRO-injection",  true,  InjectionMode::DRO,          0.0, 0.0},
    };
    const int N = 1000;

    // Summary CSV: percentile ladder + conditional stats
    std::string summary_path = OUTPUT_DIR + "cdc_c1_clearance_analysis.csv";
    std::ofstream ofs(summary_path);
    ofs << "environment,method,"
        << "collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mean_clearance,std_clearance,clearance_ci_lo,clearance_ci_hi,"
        << "p5,p10,p25,p50,p75,p90,p95,"
        << "mean_clearance_coll,mean_clearance_nocoll,"
        << "n_rollouts\n";

    // Raw clearance CSV: one row per rollout for CDF / histogram plotting
    std::string raw_path = OUTPUT_DIR + "cdc_c1_clearance_raw.csv";
    std::ofstream raw_ofs(raw_path);
    raw_ofs << "environment,method,seed,collision,min_clearance\n";

    for (size_t e = 0; e < ENVS.size(); ++e) {
        for (const auto& method : METHODS) {
            std::cout << "  " << ENV_NAMES[e] << " / " << method.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    e * 1000000 + static_cast<int>(method.injection_mode) * 100000 + i);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(ENVS[e], env_rng);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obs_modes = env_setup.obs_modes;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);

                // Write raw clearance data
                raw_ofs << ENV_NAMES[e] << "," << method.name << ","
                        << seed << "," << (rec.collision ? 1 : 0) << ","
                        << std::fixed << std::setprecision(6) << rec.min_clearance << "\n";
            }
            raw_ofs.flush();

            auto [clo, chi] = met.coll_ci();
            auto [clr_ci_lo, clr_ci_hi] = met.clearance_mean_ci();
            auto [clr_coll, clr_nocoll] = met.clearance_by_collision();

            ofs << ENV_NAMES[e] << "," << method.name << ","
                << std::fixed << std::setprecision(6)
                << met.coll_rate() << "," << clo << "," << chi << ","
                << std::setprecision(4)
                << met.mean_clearance() << "," << met.std_clearance() << ","
                << clr_ci_lo << "," << clr_ci_hi << ","
                << met.p5_clearance() << ","
                << met.p10_clearance() << ","
                << met.p25_clearance() << ","
                << met.p50_clearance() << ","
                << met.p75_clearance() << ","
                << met.p90_clearance() << ","
                << met.p95_clearance() << ","
                << clr_coll << "," << clr_nocoll << ","
                << N << "\n";
            ofs.flush();

            std::cout << N
                      << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(3) << met.mean_clearance()
                      << " [" << clr_ci_lo << "," << clr_ci_hi << "]"
                      << " p5=" << std::setprecision(3) << met.p5_clearance()
                      << " p50=" << met.p50_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    raw_ofs.close();
    std::cout << "  Written: " << summary_path << "\n";
    std::cout << "  Written: " << raw_path << "\n";
    std::cout << "  (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// C2: Clearance Analysis — Baseline comparison (all baselines)
//
// Same clearance statistics for all baselines on Oncoming env.
// Reveals whether non-WDRO baselines (softmax, eps-greedy, etc.) have
// different clearance distributions even if collision CIs are similar.
// ============================================================================

static void run_c2() {
    std::cout << "\n================================================================\n";
    std::cout << "  C2: Clearance Analysis — All Baselines (Oncoming)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<MethodDef> METHODS = {
        {"Base",              false, InjectionMode::NONE,              0.0, 0.0},
        {"WDRO-sampling",     true,  InjectionMode::QSTAR_SAMPLE,     0.0, 0.0},
        {"WDRO-injection",    true,  InjectionMode::DRO,              0.0, 0.0},
        {"Uniform-coverage",  true,  InjectionMode::UNIFORM_COVERAGE, 0.0, 0.0},
        {"Softmax-risk",      true,  InjectionMode::SOFTMAX_RISK,     5.0, 0.0},
        {"Eps-greedy",        true,  InjectionMode::EPSILON_GREEDY_INJ, 0.0, 0.3},
        {"Top-risk-inject",   true,  InjectionMode::TOP_RISK_INJECT,  0.0, 0.0},
    };
    const int N = 1000;

    std::string filepath = OUTPUT_DIR + "cdc_c2_clearance_baselines.csv";
    std::ofstream ofs(filepath);
    ofs << "method,"
        << "collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mean_clearance,std_clearance,clearance_ci_lo,clearance_ci_hi,"
        << "p5,p10,p25,p50,p75,p90,p95,"
        << "mean_clearance_coll,mean_clearance_nocoll,"
        << "n_rollouts\n";

    for (const auto& method : METHODS) {
        std::cout << "  " << method.name << ": ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        Metrics met;
        met.method = method.name;

        for (int i = 0; i < N; ++i) {
            if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(
                static_cast<int>(method.injection_mode) * 100000 + i + 8000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig cfg = make_base_config();
            cfg = apply_method(cfg, method);
            cfg.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obs_modes = env_setup.obs_modes;

            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        auto [clo, chi] = met.coll_ci();
        auto [clr_ci_lo, clr_ci_hi] = met.clearance_mean_ci();
        auto [clr_coll, clr_nocoll] = met.clearance_by_collision();

        ofs << method.name << ","
            << std::fixed << std::setprecision(6)
            << met.coll_rate() << "," << clo << "," << chi << ","
            << std::setprecision(4)
            << met.mean_clearance() << "," << met.std_clearance() << ","
            << clr_ci_lo << "," << clr_ci_hi << ","
            << met.p5_clearance() << ","
            << met.p10_clearance() << ","
            << met.p25_clearance() << ","
            << met.p50_clearance() << ","
            << met.p75_clearance() << ","
            << met.p90_clearance() << ","
            << met.p95_clearance() << ","
            << clr_coll << "," << clr_nocoll << ","
            << N << "\n";
        ofs.flush();

        std::cout << N
                  << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                  << " clr=" << std::setprecision(3) << met.mean_clearance()
                  << " [" << clr_ci_lo << "," << clr_ci_hi << "]"
                  << " p5=" << std::setprecision(3) << met.p5_clearance()
                  << " p50=" << met.p50_clearance()
                  << " coll|nocoll=" << std::setprecision(3) << clr_coll << "|" << clr_nocoll
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Wrappers: original Oncoming variants and new Intersection variants
// ============================================================================

static void run_d1_oncoming()      { run_d1_env(EnvironmentType::ONCOMING,      "Oncoming",      "_oncoming"); }
static void run_d2_oncoming()      { run_d2_env(EnvironmentType::ONCOMING,      "Oncoming",      "_oncoming"); }
static void run_d3_oncoming()      { run_d3_env(EnvironmentType::ONCOMING,      "Oncoming",      "_oncoming"); }
static void run_d1_intersection()  { run_d1_env(EnvironmentType::INTERSECTION,  "Intersection",  "_intersection"); }
static void run_d2_intersection()  { run_d2_env(EnvironmentType::INTERSECTION,  "Intersection",  "_intersection"); }
static void run_d3_intersection()  { run_d3_env(EnvironmentType::INTERSECTION,  "Intersection",  "_intersection"); }

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    fs::create_directories(OUTPUT_DIR);

    std::set<std::string> filters;
    for (int i = 1; i < argc; ++i) filters.insert(argv[i]);
    bool run_all = filters.empty();

    std::cout << "========================================\n";
    std::cout << "CDC Paper Experiments\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("e1")) run_e1();
    if (run_all || filters.count("e2")) run_e2();
    if (run_all || filters.count("e4")) run_e4();
    if (run_all || filters.count("b"))  run_baselines();
    if (run_all || filters.count("a1")) run_a1();
    if (run_all || filters.count("a2")) run_a2();
    // D experiments: Oncoming
    if (run_all || filters.count("d1")) run_d1_oncoming();
    if (run_all || filters.count("d2")) run_d2_oncoming();
    if (run_all || filters.count("d3")) run_d3_oncoming();
    // D experiments: Intersection
    if (run_all || filters.count("d1i")) run_d1_intersection();
    if (run_all || filters.count("d2i")) run_d2_intersection();
    if (run_all || filters.count("d3i")) run_d3_intersection();
    // Ground cost ablation
    if (run_all || filters.count("d4")) run_d4();
    // Clearance analysis
    if (run_all || filters.count("c1")) run_c1();
    if (run_all || filters.count("c2")) run_c2();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total << "s"
              << " (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";

    return 0;
}
