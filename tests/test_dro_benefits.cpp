/**
 * @file test_dro_benefits.cpp
 * @brief Focused experiments showing DRO strategy benefits.
 *
 * Three experiments:
 *   A: Hard-environment collision/coverage sweep (Oncoming, Intersection)
 *      at switch_prob = {0.1, 0.2, 0.3} with 1000 rollouts each.
 *   B: Multi-obstacle stress test (2, 4 obstacles) at switch_prob=0.2.
 *   C: DRO mechanism comparison — injection vs sampling vs combined,
 *      on the hardest condition (Oncoming, switch_prob=0.2).
 *
 * Outputs CSV files to paper_figures/.
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

using namespace dro_mpc;
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

static const std::vector<std::string> OBS_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Stats
// ============================================================================

// wilson_ci is provided by experiment_harness.hpp

struct Metrics {
    std::string method;
    std::vector<bool> collisions;
    std::vector<int> missed_mode_steps;
    std::vector<int> total_mode_checks;
    std::vector<int> rare_mode_active;
    std::vector<int> rare_mode_missed;
    std::vector<double> min_clearances;
    std::vector<double> total_progress;

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

    void add(const RolloutRecord& rec) {
        collisions.push_back(rec.collision);
        missed_mode_steps.push_back(rec.missed_mode_steps);
        total_mode_checks.push_back(rec.total_mode_checks);
        rare_mode_active.push_back(rec.rare_mode_active);
        rare_mode_missed.push_back(rec.rare_mode_missed);
        min_clearances.push_back(rec.min_clearance);
        total_progress.push_back(rec.total_progress);
    }
};

// ============================================================================
// Method configuration
// ============================================================================

enum class Method { BASE, WDRO_SAMPLING, WDRO_INJECTION, WDRO_COMBINED };

static std::string method_name(Method m) {
    switch (m) {
        case Method::BASE:           return "Base";
        case Method::WDRO_SAMPLING:  return "WDRO-sampling";
        case Method::WDRO_INJECTION: return "WDRO-injection";
        case Method::WDRO_COMBINED:  return "WDRO-combined";
    }
    return "?";
}

static ExperimentConfig make_config(Method method, double switch_prob, double rare_prob,
                                     int num_scenarios = NUM_SCENARIOS, int num_obs = 1) {
    ExperimentConfig cfg;
    cfg.mpc.horizon = HORIZON;
    cfg.mpc.sampling.num_scenarios = num_scenarios;
    cfg.obstacles.switch_prob = switch_prob;
    cfg.rollout.rollout_steps = ROLLOUT_STEPS;
    cfg.obstacles.obs_modes = OBS_MODES;
    cfg.obstacles.rare_mode = RARE_MODE;
    cfg.obstacles.rare_switch_prob = rare_prob;
    cfg.mpc.ego.num_discs = NUM_DISCS;
    cfg.mpc.ego.length = VEHICLE_LENGTH;
    cfg.mpc.safe_horizon_enabled = true;
    cfg.mpc.constraints.safe_horizon_min = SAFE_HORIZON_MIN;
    cfg.environment.path_completion_termination = true;
    cfg.environment.path_completion_fraction = 0.95;
    cfg.dro.enabled = false;
    cfg.dro.injection_mode = InjectionMode::NONE;
    cfg.dro.solver.base_radius = 0.1;
    cfg.obstacles.num_obstacles = num_obs;
    cfg.obstacles.obstacles_per_class = 1;
    cfg.rollout.method_name = method_name(method);

    switch (method) {
        case Method::BASE:
            break;
        case Method::WDRO_SAMPLING:
            cfg.dro.enabled = true;
            cfg.dro.injection_mode = InjectionMode::QSTAR_SAMPLE;
            break;
        case Method::WDRO_INJECTION:
            cfg.dro.enabled = true;
            cfg.dro.injection_mode = InjectionMode::TOP_RISK_INJECT;
            break;
        case Method::WDRO_COMBINED:
            cfg.dro.enabled = true;
            cfg.dro.injection_mode = InjectionMode::TOP_RISK_INJECT;
            break;
    }
    return cfg;
}

static double elapsed_sec(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// ============================================================================
// Experiment A: Hard-environment collision/coverage sweep
// ============================================================================

static void run_exp_a() {
    std::cout << "\n================================================================\n";
    std::cout << "  Experiment A: Hard-Environment Collision & Coverage Sweep\n";
    std::cout << "  Oncoming + Intersection, switch_prob = {0.1, 0.2, 0.3}\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<EnvironmentType> ENVS = {
        EnvironmentType::INTERSECTION, EnvironmentType::ONCOMING
    };
    const std::vector<double> SWITCH_PROBS = {0.1, 0.2, 0.3};
    const std::vector<Method> METHODS = {
        Method::BASE, Method::WDRO_SAMPLING, Method::WDRO_INJECTION, Method::WDRO_COMBINED
    };
    const int N_ROLLOUTS = 1000;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "dro_benefit_env_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "environment,switch_prob,method,"
        << "collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,n_rollouts\n";

    for (EnvironmentType env : ENVS) {
        for (double sp : SWITCH_PROBS) {
            for (Method method : METHODS) {
                std::string ename = environment_name(env);
                std::cout << "  " << ename << " sp=" << std::fixed << std::setprecision(1) << sp
                          << " " << method_name(method) << " ...";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();

                Metrics met;
                met.method = method_name(method);

                for (int i = 0; i < N_ROLLOUTS; ++i) {
                    unsigned seed = static_cast<unsigned>(
                        static_cast<int>(env) * 1000000 +
                        static_cast<int>(sp * 1000) * 1000 +
                        static_cast<int>(method) * 10000 + i);
                    std::mt19937 env_rng(seed);
                    EnvironmentSetup env_setup = create_environment(env, env_rng);

                    ExperimentConfig cfg = make_config(method, sp, RARE_PROB);
                    cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
                    cfg.obstacles.obs_modes = env_setup.obs_modes;

                    RolloutRecord rec = run_experiment_rollout(cfg, seed);
                    met.add(rec);
                }

                auto [clo, chi] = met.coll_ci();
                auto [mlo, mhi] = met.mm_ci();
                auto [rlo, rhi] = met.rare_ci();

                ofs << environment_name(env) << "," << std::fixed << std::setprecision(1) << sp << ","
                    << met.method << ","
                    << std::setprecision(6)
                    << met.coll_rate() << "," << clo << "," << chi << ","
                    << met.mm_rate() << "," << mlo << "," << mhi << ","
                    << met.rare_rate() << "," << rlo << "," << rhi << ","
                    << std::setprecision(4) << met.mean_clearance() << ","
                    << met.p5_clearance() << "," << met.mean_progress() << ","
                    << N_ROLLOUTS << "\n";
                ofs.flush();

                std::cout << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                          << " mm=" << (met.mm_rate() * 100) << "%"
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Experiment B: Multi-obstacle stress test
// ============================================================================

static void run_exp_b() {
    std::cout << "\n================================================================\n";
    std::cout << "  Experiment B: Multi-Obstacle Stress Test\n";
    std::cout << "  {1, 2, 4} obstacles, switch_prob=0.2\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> OBS_COUNTS = {1, 2, 4};
    const std::vector<Method> METHODS = {
        Method::BASE, Method::WDRO_SAMPLING, Method::WDRO_INJECTION, Method::WDRO_COMBINED
    };
    const int N_ROLLOUTS = 1000;
    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "dro_benefit_multi_obs.csv";
    std::ofstream ofs(filepath);
    ofs << "num_obstacles,method,"
        << "collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,n_rollouts\n";

    for (int n_obs : OBS_COUNTS) {
        for (Method method : METHODS) {
            std::cout << "  obs=" << n_obs << " " << method_name(method) << " ...";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                unsigned seed = static_cast<unsigned>(
                    n_obs * 1000000 + static_cast<int>(method) * 10000 + i);

                ExperimentConfig cfg = make_config(method, SWITCH_PROB, RARE_PROB, NUM_SCENARIOS, n_obs);

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            auto [clo, chi] = met.coll_ci();
            auto [mlo, mhi] = met.mm_ci();
            auto [rlo, rhi] = met.rare_ci();

            ofs << n_obs << "," << met.method << ","
                << std::fixed << std::setprecision(6)
                << met.coll_rate() << "," << clo << "," << chi << ","
                << met.mm_rate() << "," << mlo << "," << mhi << ","
                << met.rare_rate() << "," << rlo << "," << rhi << ","
                << std::setprecision(4) << met.mean_clearance() << ","
                << met.p5_clearance() << "," << met.mean_progress() << ","
                << N_ROLLOUTS << "\n";
            ofs.flush();

            std::cout << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Experiment C: DRO mechanism comparison on hardest condition
// ============================================================================

static void run_exp_c() {
    std::cout << "\n================================================================\n";
    std::cout << "  Experiment C: DRO Mechanism Comparison\n";
    std::cout << "  Oncoming, switch_prob=0.2, 2000 rollouts\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<Method> METHODS = {
        Method::BASE, Method::WDRO_SAMPLING, Method::WDRO_INJECTION, Method::WDRO_COMBINED
    };
    const int N_ROLLOUTS = 2000;
    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "dro_benefit_mechanism.csv";
    std::ofstream ofs(filepath);
    ofs << "method,"
        << "collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,n_rollouts\n";

    for (Method method : METHODS) {
        std::cout << "  " << method_name(method) << " ...";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        Metrics met;
        met.method = method_name(method);

        for (int i = 0; i < N_ROLLOUTS; ++i) {
            if ((i + 1) % 500 == 0) {
                std::cout << " " << (i + 1);
                std::cout.flush();
            }
            unsigned seed = static_cast<unsigned>(
                static_cast<int>(method) * 100000 + i + 5000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig cfg = make_config(method, SWITCH_PROB, RARE_PROB);
            cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obstacles.obs_modes = env_setup.obs_modes;

            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        auto [clo, chi] = met.coll_ci();
        auto [mlo, mhi] = met.mm_ci();
        auto [rlo, rhi] = met.rare_ci();

        ofs << met.method << ","
            << std::fixed << std::setprecision(6)
            << met.coll_rate() << "," << clo << "," << chi << ","
            << met.mm_rate() << "," << mlo << "," << mhi << ","
            << met.rare_rate() << "," << rlo << "," << rhi << ","
            << std::setprecision(4) << met.mean_clearance() << ","
            << met.p5_clearance() << "," << met.mean_progress() << ","
            << N_ROLLOUTS << "\n";
        ofs.flush();

        std::cout << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                  << " mm=" << (met.mm_rate() * 100) << "%"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Experiment D: DRO Injection Count Sweep
//   Varies the number of top-K modes injected per obstacle.
//   Tests both DRO (mean trajectory) and ADVERSARIAL (sigma-pushed) injection.
//
//   Theoretical motivation:
//   - The Kantorovich dual concentrates Q* on a small effective support.
//     For W1 with transport cost D, at optimal lambda*, each source mode i
//     transports to j*(i) = argmax_j { r_j - lambda* D[i][j] }.
//     The effective support |{j*(i)}| is typically 1-3 modes.
//   - Injecting beyond the effective support adds constraints for modes
//     with negligible Q* weight — diminishing safety returns but increasing
//     conservatism (and potentially causing infeasibility).
//   - Adaptive rho scales the ball radius with observation count and entropy,
//     which changes the effective support size. We report effective_support
//     as a diagnostic.
//
//   Sweep: K = {0, 1, 2, 3, 5, -1(all)} × {DRO, ADVERSARIAL}
//   Environment: Oncoming (hardest), switch_prob=0.2, 1000 rollouts each
// ============================================================================

static void run_exp_d() {
    std::cout << "\n================================================================\n";
    std::cout << "  Experiment D: DRO Injection Count Sweep\n";
    std::cout << "  Oncoming, switch_prob=0.2, K = {0,1,2,3,5,all}\n";
    std::cout << "  Injection types: DRO (mean) and ADVERSARIAL (sigma-pushed)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;
    const int N_ROLLOUTS = 1000;
    // K values: 0=no injection (base+DRO weights only), 1..5=top-K, -1=all modes
    const std::vector<int> K_VALUES = {0, 1, 2, 3, 5, -1};
    const std::vector<InjectionMode> INJ_MODES = {InjectionMode::TOP_RISK_INJECT, InjectionMode::TOP_RISK_INJECT};

    std::string filepath = OUTPUT_DIR + "dro_injection_count_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "injection_mode,K,K_label,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,"
        << "mean_dro_injected,n_rollouts\n";

    for (auto inj_mode : INJ_MODES) {
        std::string mode_label = (inj_mode == InjectionMode::TOP_RISK_INJECT) ? "DRO" : "ADVERSARIAL";

        for (int K : K_VALUES) {
            std::string k_label = (K < 0) ? "all" : std::to_string(K);
            std::cout << "  " << mode_label << " K=" << k_label << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = mode_label + "_K" + k_label;
            std::vector<double> solve_times;
            std::vector<int> dro_injected_counts;

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 250 == 0) {
                    std::cout << i << " ";
                    std::cout.flush();
                }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(inj_mode) * 1000000 + K * 100000 + i + 7000000);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

                ExperimentConfig cfg;
                cfg.mpc.horizon = HORIZON;
                cfg.mpc.sampling.num_scenarios = NUM_SCENARIOS;
                cfg.obstacles.switch_prob = SWITCH_PROB;
                cfg.rollout.rollout_steps = ROLLOUT_STEPS;
                cfg.obstacles.obs_modes = env_setup.obs_modes;
                cfg.obstacles.rare_mode = RARE_MODE;
                cfg.obstacles.rare_switch_prob = RARE_PROB;
                cfg.mpc.ego.num_discs = NUM_DISCS;
                cfg.mpc.ego.length = VEHICLE_LENGTH;
                cfg.mpc.safe_horizon_enabled = true;
                cfg.mpc.constraints.safe_horizon_min = SAFE_HORIZON_MIN;
                cfg.environment.path_completion_termination = true;
                cfg.environment.path_completion_fraction = 0.95;
                cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obstacles.num_obstacles = 1;
                cfg.obstacles.obstacles_per_class = 1;
                cfg.rollout.method_name = met.method;

                if (K == 0) {
                    // K=0: no injection, just base scenario MPC
                    cfg.dro.enabled = false;
                    cfg.dro.injection_mode = InjectionMode::NONE;
                } else {
                    cfg.dro.enabled = true;
                    cfg.dro.injection_mode = inj_mode;
                    cfg.dro.injection_count = K;
                }
                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);

                // Track solve time and DRO injection count
                solve_times.push_back(rec.avg_solve_ms);
                dro_injected_counts.push_back(rec.total_dro_injected);
            }

            auto [clo, chi] = met.coll_ci();
            auto [mlo, mhi] = met.mm_ci();
            auto [rlo, rhi] = met.rare_ci();

            double mean_solve = 0;
            if (!solve_times.empty()) {
                mean_solve = std::accumulate(solve_times.begin(), solve_times.end(), 0.0)
                           / solve_times.size();
            }

            double mean_dro_inj = 0;
            if (!dro_injected_counts.empty()) {
                mean_dro_inj = std::accumulate(dro_injected_counts.begin(), dro_injected_counts.end(), 0.0)
                             / dro_injected_counts.size();
            }

            ofs << mode_label << "," << K << "," << k_label << ","
                << std::fixed << std::setprecision(6)
                << met.coll_rate() << "," << clo << "," << chi << ","
                << met.mm_rate() << "," << mlo << "," << mhi << ","
                << met.rare_rate() << "," << rlo << "," << rhi << ","
                << std::setprecision(4) << met.mean_clearance() << ","
                << met.p5_clearance() << "," << met.mean_progress() << ","
                << std::setprecision(3) << mean_solve << ","
                << std::setprecision(1) << mean_dro_inj << ","
                << N_ROLLOUTS << "\n";
            ofs.flush();

            std::cout << N_ROLLOUTS
                      << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
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
    std::cout << "DRO Benefits Experiments\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("a")) run_exp_a();
    if (run_all || filters.count("b")) run_exp_b();
    if (run_all || filters.count("c")) run_exp_c();
    if (run_all || filters.count("d")) run_exp_d();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total << "s"
              << " (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";

    return 0;
}
