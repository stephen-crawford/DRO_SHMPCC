/**
 * @file test_paper_figures.cpp
 * @brief Generate CSV data for paper figures 40-47.
 *
 * Figures:
 *   40: Risk Lift Validation          (fig40_risk_lift.csv)
 *   41: Budget Scaling                 (fig41_budget_scaling.csv)
 *   42: Rare-Mode Sweep                (fig42_rare_mode_sweep.csv)
 *   43: Geometry Ablation              (fig43_geometry_ablation.csv)
 *   44: Rho Sweep / Pareto             (fig44_rho_sweep_pareto.csv)
 *   45: Recovery Feasibility           (fig45_recovery_feasibility.csv)
 *   46: Environment Robustness         (fig46_env_robustness.csv)
 *   47: Runtime                        (fig47_runtime.csv)
 *
 * Usage:
 *   ./test_paper_figures              # run all
 *   ./test_paper_figures fig41        # run only figure 41
 *   ./test_paper_figures fig40 fig45  # run figures 40 and 45
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
#include <cassert>
#include <filesystem>
#include <sstream>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "dynamics.hpp"
#include "collision_constraints.hpp"

using namespace dro_mpc;
namespace fs = std::filesystem;

// ============================================================================
// Constants
// ============================================================================

static const std::string OUTPUT_DIR = "paper_figures/";

static const int    HORIZON = DEFAULT_HORIZON;   // 15
static const double    DT = DEFAULT_DT;        // 0.1
static const int    NUM_SCENARIOS = DEFAULT_BASE_SCENARIOS; // 40
static const int    ROLLOUT_STEPS = DEFAULT_ROLLOUT_STEPS;  // 200
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;
static constexpr double CI_Z            = 1.96;

static const std::vector<std::string> OBS_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Statistical helpers
// ============================================================================

static std::pair<double, double> wilson_ci_local(int successes, int n, double z = 1.96) {
    if (n == 0) return {0.0, 1.0};
    double p_hat = static_cast<double>(successes) / n;
    double denom = 1.0 + z * z / n;
    double center = (p_hat + z * z / (2.0 * n)) / denom;
    double half_width = z * std::sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom;
    return {std::max(0.0, center - half_width), std::min(1.0, center + half_width)};
}

// ============================================================================
// Metrics accumulator (reusable across experiments)
// ============================================================================

struct Metrics {
    std::string method;
    std::vector<bool> collisions;
    std::vector<int> missed_mode_steps;
    std::vector<int> total_mode_checks;
    std::vector<int> rare_mode_active;
    std::vector<int> rare_mode_missed;
    std::vector<double> solve_times_avg;
    std::vector<double> solve_times_p50;
    std::vector<double> solve_times_p95;
    std::vector<double> solve_times_max;
    std::vector<double> total_progress;

    int n() const { return static_cast<int>(collisions.size()); }

    int collision_count() const {
        int c = 0; for (bool b : collisions) if (b) c++; return c;
    }
    double collision_rate() const {
        return n() > 0 ? static_cast<double>(collision_count()) / n() : 0.0;
    }
    std::pair<double,double> collision_ci() const {
        return wilson_ci_local(collision_count(), n(), CI_Z);
    }

    int total_missed() const {
        return std::accumulate(missed_mode_steps.begin(), missed_mode_steps.end(), 0);
    }
    int total_checks() const {
        return std::accumulate(total_mode_checks.begin(), total_mode_checks.end(), 0);
    }
    double missed_mode_rate() const {
        int tc = total_checks();
        return tc > 0 ? static_cast<double>(total_missed()) / tc : 0.0;
    }
    std::pair<double,double> missed_mode_ci() const {
        int tc = total_checks();
        return wilson_ci_local(total_missed(), tc, CI_Z);
    }

    int total_rare_act() const {
        return std::accumulate(rare_mode_active.begin(), rare_mode_active.end(), 0);
    }
    int total_rare_miss() const {
        return std::accumulate(rare_mode_missed.begin(), rare_mode_missed.end(), 0);
    }
    double rare_miss_rate() const {
        int a = total_rare_act();
        return a > 0 ? static_cast<double>(total_rare_miss()) / a : 0.0;
    }
    std::pair<double,double> rare_miss_ci() const {
        int a = total_rare_act();
        return wilson_ci_local(total_rare_miss(), a, CI_Z);
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
    double p95_solve() const {
        if (solve_times_p95.empty()) return 0;
        auto v = solve_times_p95;
        std::sort(v.begin(), v.end());
        return v[static_cast<size_t>(0.95 * (v.size() - 1))];
    }
    double max_solve() const {
        if (solve_times_max.empty()) return 0;
        return *std::max_element(solve_times_max.begin(), solve_times_max.end());
    }
    double mean_progress() const {
        if (total_progress.empty()) return 0;
        return std::accumulate(total_progress.begin(), total_progress.end(), 0.0)
               / total_progress.size();
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
        solve_times_max.push_back(rec.max_solve_ms);
        total_progress.push_back(rec.total_progress);
    }
};

// ============================================================================
// Method configs — three primary methods
// ============================================================================

enum class MethodType { BASE, WDRO_SAMPLING, WDRO_INJECTION, WDRO_COMBINED };

static std::string method_name(MethodType m) {
    switch (m) {
        case MethodType::BASE:           return "Base";
        case MethodType::WDRO_SAMPLING:  return "WDRO-sampling";
        case MethodType::WDRO_INJECTION: return "WDRO-injection";
        case MethodType::WDRO_COMBINED:  return "WDRO-combined";
    }
    return "?";
}

static ExperimentConfig make_base_config(double switch_prob, double rare_prob,
                                          int num_scenarios = NUM_SCENARIOS) {
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
    cfg.dro.solver.base_radius = 0.1;
    return cfg;
}

static ExperimentConfig make_method_config(MethodType method, double switch_prob,
                                            double rare_prob, int num_scenarios = NUM_SCENARIOS) {
    ExperimentConfig cfg = make_base_config(switch_prob, rare_prob, num_scenarios);
    cfg.rollout.method_name = method_name(method);

    switch (method) {
        case MethodType::BASE:
            // Already configured
            break;
        case MethodType::WDRO_SAMPLING:
            cfg.dro.enabled = true;
            break;
        case MethodType::WDRO_INJECTION:
            cfg.dro.enabled = true;
            break;
        case MethodType::WDRO_COMBINED:
            cfg.dro.enabled = true;
            break;
    }
    return cfg;
}

// ============================================================================
// Timing helper
// ============================================================================

static double elapsed_sec(std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start).count();
}

// ============================================================================
// Fig 40: Risk Lift Validation
// ============================================================================

static void run_fig40_risk_lift() {
    std::cout << "\n=== Fig 40: Risk Lift Validation ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const int NUM_SOLVES = 500;
    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;
    const double RHO_BASE = 0.1;

    std::string filepath = OUTPUT_DIR + "fig40_risk_lift.csv";
    std::ofstream ofs(filepath);
    ofs << "solve_id,risk_lift,injected_mode_risk,nominal_expected_risk,"
        << "qstar_expected_risk,rho_used,implied_cost,feasible\n";

    auto mode_models = create_obstacle_mode_models(DT);

    // Build mode model set for obstacle (base + rare)
    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : OBS_MODES) {
        if (mode_models.count(m)) obs_mode_models[m] = mode_models[m];
    }
    if (mode_models.count(RARE_MODE)) obs_mode_models[RARE_MODE] = mode_models[RARE_MODE];

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pos_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> vel_dist(-0.5, 0.5);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    // Collect all mode names
    std::vector<std::string> all_modes;
    for (const auto& m : OBS_MODES) all_modes.push_back(m);
    all_modes.push_back(RARE_MODE);

    // DRO probe for diagnostics
    DROConfig dro_cfg;
    dro_cfg.base_radius = RHO_BASE;
    dro_cfg.min_radius = 0.01;
    dro_cfg.max_radius = 0.5;
    dro_cfg.radius_calibration.use_calibrated_radius = true;
    dro_cfg.ground_cost_type = DROGroundCostType::W2_BURES;
    WassersteinDRO dro_probe(dro_cfg);

    for (int i = 0; i < NUM_SOLVES; ++i) {
        if ((i + 1) % 100 == 0 || i == 0) {
            std::cout << "  Solve " << (i + 1) << "/" << NUM_SOLVES << "\n";
        }

        // Random obstacle state
        ObstacleState obs_state(3.0 + pos_dist(rng), pos_dist(rng) * 0.5,
                                 vel_dist(rng), vel_dist(rng));

        // Generate a plausible ego reference trajectory (straight line)
        std::vector<EgoState> ego_ref;
        ego_ref.reserve(HORIZON + 1);
        for (int k = 0; k <= HORIZON; ++k) {
            ego_ref.emplace_back(k * DT * 1.5, 0.0, 0.0, 1.5);
        }

        // Build nominal weights from a synthetic observation history
        // Simulate some history: mainly dominant mode, sometimes rare
        ModeHistory hist(0, obs_mode_models);
        int n_obs = 20 + static_cast<int>(u01(rng) * 30);
        for (int t = 0; t < n_obs; ++t) {
            double r = u01(rng);
            std::string mode;
            if (r < RARE_PROB) {
                mode = RARE_MODE;
            } else {
                // Pick uniformly from base modes
                std::uniform_int_distribution<int> base_idx(0, static_cast<int>(OBS_MODES.size()) - 1);
                mode = OBS_MODES[base_idx(rng)];
            }
            hist.record_observation(t, mode);
        }
        auto nominal_weights = compute_mode_weights(hist);

        // Set observation count for adaptive rho
        dro_probe.set_observation_count(n_obs);

        // Compute DRO result
        DROResult dro_result = dro_probe.compute_worst_case_weights(
            nominal_weights, obs_state, obs_mode_models, ego_ref,
            HORIZON, 0.5, 0.35, 0.2, -1, NUM_DISCS, VEHICLE_LENGTH);

        // Compute nominal expected risk: E_phat[r]
        double nominal_exp_risk = 0.0;
        for (const auto& [mode, w] : nominal_weights) {
            if (dro_result.risk_per_mode.count(mode)) {
                nominal_exp_risk += w * dro_result.risk_per_mode.at(mode);
            }
        }

        // Compute q* expected risk: E_q*[r]
        double qstar_exp_risk = 0.0;
        for (const auto& [mode, w] : dro_result.worst_case_weights) {
            if (dro_result.risk_per_mode.count(mode)) {
                qstar_exp_risk += w * dro_result.risk_per_mode.at(mode);
            }
        }

        double risk_lift = qstar_exp_risk - nominal_exp_risk;

        // Injected mode risk: mode with highest q* weight
        std::string injected_mode;
        double max_q = -1.0;
        for (const auto& [mode, w] : dro_result.worst_case_weights) {
            if (w > max_q) { max_q = w; injected_mode = mode; }
        }
        double injected_risk = dro_result.risk_per_mode.count(injected_mode)
                                   ? dro_result.risk_per_mode.at(injected_mode) : 0.0;

        ofs << i << "," << std::fixed << std::setprecision(6)
            << risk_lift << "," << injected_risk << ","
            << nominal_exp_risk << "," << qstar_exp_risk << ","
            << dro_result.rho_used << "," << dro_result.implied_transport_cost << ","
            << (dro_result.recovery_feasible ? 1 : 0) << "\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " (" << NUM_SOLVES << " solves, "
              << std::fixed << std::setprecision(1) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Fig 41: Budget Scaling
// ============================================================================

static void run_fig41_budget_scaling() {
    std::cout << "\n=== Fig 41: Budget Scaling ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> S_VALUES = {5, 10, 15, 20, 30, 40};
    const std::vector<MethodType> METHODS = {
        MethodType::BASE, MethodType::WDRO_SAMPLING, MethodType::WDRO_INJECTION, MethodType::WDRO_COMBINED
    };
    const int N_ROLLOUTS = 400;
    const double SWITCH_PROB = 0.1;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "fig41_budget_scaling.csv";
    std::ofstream ofs(filepath);
    ofs << "S,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_solve_ms,n_rollouts\n";

    for (int S : S_VALUES) {
        for (MethodType method : METHODS) {
            std::cout << "  S=" << S << " method=" << method_name(method) << " ...";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            ExperimentConfig cfg = make_method_config(method, SWITCH_PROB, RARE_PROB, S);
            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                unsigned seed = static_cast<unsigned>(S * 100000 + static_cast<int>(method) * 10000 + i);
                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            auto [clo, chi] = met.collision_ci();
            auto [mlo, mhi] = met.missed_mode_ci();
            auto [rlo, rhi] = met.rare_miss_ci();

            ofs << S << "," << met.method << ","
                << std::fixed << std::setprecision(6)
                << met.collision_rate() << "," << clo << "," << chi << ","
                << met.missed_mode_rate() << "," << mlo << "," << mhi << ","
                << met.rare_miss_rate() << "," << rlo << "," << rhi << ","
                << std::setprecision(2) << met.mean_solve_ms() << ","
                << N_ROLLOUTS << "\n";

            std::cout << " coll=" << std::fixed << std::setprecision(3)
                      << met.collision_rate() << " ("
                      << std::setprecision(1) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " ("
              << std::setprecision(1) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Fig 42: Rare-Mode Sweep
// ============================================================================

static void run_fig42_rare_mode_sweep() {
    std::cout << "\n=== Fig 42: Rare-Mode Sweep ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> RARE_PROBS = {0.01, 0.05, 0.10, 0.20};
    const std::vector<MethodType> METHODS = {
        MethodType::BASE, MethodType::WDRO_SAMPLING, MethodType::WDRO_INJECTION, MethodType::WDRO_COMBINED
    };
    const int N_ROLLOUTS = 400;
    const double SWITCH_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "fig42_rare_mode_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "rare_prob,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "n_rollouts\n";

    for (double rp : RARE_PROBS) {
        for (MethodType method : METHODS) {
            std::cout << "  rare_prob=" << std::fixed << std::setprecision(2) << rp
                      << " method=" << method_name(method) << " ...";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            ExperimentConfig cfg = make_method_config(method, SWITCH_PROB, rp);
            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(rp * 10000) * 100 + static_cast<int>(method) * 10000 + i);
                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            auto [clo, chi] = met.collision_ci();
            auto [rlo, rhi] = met.rare_miss_ci();
            auto [mlo, mhi] = met.missed_mode_ci();

            ofs << std::fixed << std::setprecision(2) << rp << ","
                << met.method << ","
                << std::setprecision(6)
                << met.collision_rate() << "," << clo << "," << chi << ","
                << met.rare_miss_rate() << "," << rlo << "," << rhi << ","
                << met.missed_mode_rate() << "," << mlo << "," << mhi << ","
                << N_ROLLOUTS << "\n";

            std::cout << " coll=" << std::setprecision(3)
                      << met.collision_rate() << " ("
                      << std::setprecision(1) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " ("
              << std::setprecision(1) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Fig 43: Geometry Ablation
// ============================================================================

static void run_fig43_geometry_ablation() {
    std::cout << "\n=== Fig 43: Geometry Ablation ===\n";
    auto t0 = std::chrono::steady_clock::now();

    struct GeomVariant {
        std::string name;
        bool enable_dro;
        DROGroundCostType ground_cost;
    };

    const std::vector<GeomVariant> VARIANTS = {
        {"Base (no DRO)", false, DROGroundCostType::W2_BURES},
        {"W2_BURES",      true,  DROGroundCostType::W2_BURES},
        {"ZERO_ONE",      true,  DROGroundCostType::ZERO_ONE},
        {"EUCLIDEAN_MEAN",true,  DROGroundCostType::EUCLIDEAN_MEAN},
    };

    const int N_ROLLOUTS = 400;
    const double SWITCH_PROB = 0.1;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "fig43_geometry_ablation.csv";
    std::ofstream ofs(filepath);
    ofs << "ground_cost,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "mean_solve_ms,n_rollouts\n";

    for (const auto& var : VARIANTS) {
        std::cout << "  ground_cost=" << var.name << " ...";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        ExperimentConfig cfg = make_base_config(SWITCH_PROB, RARE_PROB);
        cfg.rollout.method_name = var.name;
        cfg.dro.enabled = (var.enable_dro);
        cfg.dro.solver.ground_cost_type = var.ground_cost;

        Metrics met;
        met.method = var.name;

        for (int i = 0; i < N_ROLLOUTS; ++i) {
            unsigned seed = static_cast<unsigned>(
                static_cast<int>(var.ground_cost) * 100000 + i + (var.enable_dro ? 0 : 999000));
            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        auto [clo, chi] = met.collision_ci();
        auto [mlo, mhi] = met.missed_mode_ci();

        ofs << var.name << ","
            << std::fixed << std::setprecision(6)
            << met.collision_rate() << "," << clo << "," << chi << ","
            << met.missed_mode_rate() << "," << mlo << "," << mhi << ","
            << std::setprecision(2) << met.mean_solve_ms() << ","
            << N_ROLLOUTS << "\n";

        std::cout << " coll=" << std::setprecision(3) << met.collision_rate()
                  << " (" << std::setprecision(1) << elapsed_sec(t1) << "s)\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " ("
              << std::setprecision(1) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Fig 44: Rho Sweep / Pareto
// ============================================================================

static void run_fig44_rho_sweep() {
    std::cout << "\n=== Fig 44: Rho Sweep / Pareto ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> RHO_VALUES = {0.01, 0.05, 0.1, 0.2, 0.3, 0.5};
    const int N_ROLLOUTS = 400;
    const double SWITCH_PROB = 0.1;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "fig44_rho_sweep_pareto.csv";
    std::ofstream ofs(filepath);
    ofs << "rho,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "n_rollouts\n";

    // Base reference (no DRO)
    {
        std::cout << "  Base (no DRO) ...";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        ExperimentConfig cfg = make_base_config(SWITCH_PROB, RARE_PROB);
        cfg.rollout.method_name = "Base";
        Metrics met;
        met.method = "Base";

        for (int i = 0; i < N_ROLLOUTS; ++i) {
            RolloutRecord rec = run_experiment_rollout(cfg, static_cast<unsigned>(800000 + i));
            met.add(rec);
        }

        auto [clo, chi] = met.collision_ci();
        auto [mlo, mhi] = met.missed_mode_ci();

        ofs << "0.0,Base,"
            << std::fixed << std::setprecision(6)
            << met.collision_rate() << "," << clo << "," << chi << ","
            << met.missed_mode_rate() << "," << mlo << "," << mhi << ","
            << N_ROLLOUTS << "\n";

        std::cout << " coll=" << std::setprecision(3) << met.collision_rate()
                  << " (" << std::setprecision(1) << elapsed_sec(t1) << "s)\n";
    }

    // WDRO-sampling at each rho
    for (double rho : RHO_VALUES) {
        std::cout << "  rho=" << std::fixed << std::setprecision(2) << rho << " ...";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        ExperimentConfig cfg = make_base_config(SWITCH_PROB, RARE_PROB);
        cfg.rollout.method_name = "WDRO-sampling";
        cfg.dro.enabled = true;
        cfg.dro.solver.base_radius = rho;

        Metrics met;
        met.method = "WDRO-sampling";

        for (int i = 0; i < N_ROLLOUTS; ++i) {
            unsigned seed = static_cast<unsigned>(static_cast<int>(rho * 100000) + i);
            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        auto [clo, chi] = met.collision_ci();
        auto [mlo, mhi] = met.missed_mode_ci();

        ofs << std::fixed << std::setprecision(2) << rho << ","
            << "WDRO-sampling,"
            << std::setprecision(6)
            << met.collision_rate() << "," << clo << "," << chi << ","
            << met.missed_mode_rate() << "," << mlo << "," << mhi << ","
            << N_ROLLOUTS << "\n";

        std::cout << " coll=" << std::setprecision(3) << met.collision_rate()
                  << " (" << std::setprecision(1) << elapsed_sec(t1) << "s)\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " ("
              << std::setprecision(1) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Fig 45: Recovery Feasibility
// ============================================================================

static void run_fig45_feasibility() {
    std::cout << "\n=== Fig 45: Recovery Feasibility ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const int NUM_SOLVES = 500;
    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;
    const double RHO_BASE = 0.1;

    std::string filepath = OUTPUT_DIR + "fig45_recovery_feasibility.csv";
    std::ofstream ofs(filepath);
    ofs << "solve_id,transport_ratio,feasible,rho_used\n";

    auto mode_models = create_obstacle_mode_models(DT);

    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : OBS_MODES) {
        if (mode_models.count(m)) obs_mode_models[m] = mode_models[m];
    }
    if (mode_models.count(RARE_MODE)) obs_mode_models[RARE_MODE] = mode_models[RARE_MODE];

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> pos_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> vel_dist(-0.5, 0.5);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    DROConfig dro_cfg;
    dro_cfg.base_radius = RHO_BASE;
    dro_cfg.min_radius = 0.01;
    dro_cfg.max_radius = 0.5;
    dro_cfg.radius_calibration.use_calibrated_radius = true;
    dro_cfg.ground_cost_type = DROGroundCostType::W2_BURES;
    WassersteinDRO dro_probe(dro_cfg);

    int feasible_count = 0;

    for (int i = 0; i < NUM_SOLVES; ++i) {
        if ((i + 1) % 100 == 0 || i == 0) {
            std::cout << "  Solve " << (i + 1) << "/" << NUM_SOLVES << "\n";
        }

        ObstacleState obs_state(3.0 + pos_dist(rng), pos_dist(rng) * 0.5,
                                 vel_dist(rng), vel_dist(rng));

        std::vector<EgoState> ego_ref;
        ego_ref.reserve(HORIZON + 1);
        for (int k = 0; k <= HORIZON; ++k) {
            ego_ref.emplace_back(k * DT * 1.5, 0.0, 0.0, 1.5);
        }

        // Build varied observation history
        ModeHistory hist(0, obs_mode_models);
        int n_obs = 10 + static_cast<int>(u01(rng) * 40);
        for (int t = 0; t < n_obs; ++t) {
            double r = u01(rng);
            std::string mode;
            if (r < RARE_PROB) {
                mode = RARE_MODE;
            } else {
                std::uniform_int_distribution<int> base_idx(0, static_cast<int>(OBS_MODES.size()) - 1);
                mode = OBS_MODES[base_idx(rng)];
            }
            hist.record_observation(t, mode);
        }
        auto nominal_weights = compute_mode_weights(hist);
        dro_probe.set_observation_count(n_obs);

        DROResult dro_result = dro_probe.compute_worst_case_weights(
            nominal_weights, obs_state, obs_mode_models, ego_ref,
            HORIZON, 0.5, 0.35, 0.2, -1, NUM_DISCS, VEHICLE_LENGTH);

        double transport_ratio = (dro_result.rho_used > 1e-12)
            ? dro_result.implied_transport_cost / dro_result.rho_used
            : 0.0;

        if (dro_result.recovery_feasible) feasible_count++;

        ofs << i << "," << std::fixed << std::setprecision(6)
            << transport_ratio << ","
            << (dro_result.recovery_feasible ? 1 : 0) << ","
            << dro_result.rho_used << "\n";
    }

    ofs.close();
    double feas_rate = static_cast<double>(feasible_count) / NUM_SOLVES;
    std::cout << "  Feasibility rate: " << std::fixed << std::setprecision(1)
              << (feas_rate * 100.0) << "% (" << feasible_count << "/" << NUM_SOLVES << ")\n";
    std::cout << "  Written: " << filepath << " (" << NUM_SOLVES << " solves, "
              << std::setprecision(1) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Fig 46: Environment Robustness
// ============================================================================

static void run_fig46_env_robustness() {
    std::cout << "\n=== Fig 46: Environment Robustness ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<EnvironmentType> ENVS = {
        EnvironmentType::OVERTAKE_SLOW_LEAD,
        EnvironmentType::NARROW_CORRIDOR,
        EnvironmentType::INTERSECTION,
        EnvironmentType::ONCOMING
    };
    const std::vector<MethodType> METHODS = {
        MethodType::BASE, MethodType::WDRO_SAMPLING, MethodType::WDRO_INJECTION, MethodType::WDRO_COMBINED
    };
    const int N_ROLLOUTS = 300;
    const double SWITCH_PROB = 0.1;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "fig46_env_robustness.csv";
    std::ofstream ofs(filepath);
    ofs << "environment,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "mean_progress,n_rollouts\n";

    for (EnvironmentType env : ENVS) {
        for (MethodType method : METHODS) {
            std::string env_name = environment_name(env);
            std::cout << "  env=" << env_name << " method=" << method_name(method) << " ...";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(env) * 100000 + static_cast<int>(method) * 10000 + i);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(env, env_rng);

                // Build config via ExperimentConfig for more control
                ExperimentConfig cfg = make_method_config(method, SWITCH_PROB, RARE_PROB);
                cfg.rollout.method_name = method_name(method);
                cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obstacles.obs_modes = env_setup.obs_modes;
                cfg.obstacles.rare_mode = RARE_MODE;
                cfg.obstacles.rare_switch_prob = RARE_PROB;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            auto [clo, chi] = met.collision_ci();
            auto [mlo, mhi] = met.missed_mode_ci();

            ofs << env_name << "," << met.method << ","
                << std::fixed << std::setprecision(6)
                << met.collision_rate() << "," << clo << "," << chi << ","
                << met.missed_mode_rate() << "," << mlo << "," << mhi << ","
                << std::setprecision(4) << met.mean_progress() << ","
                << N_ROLLOUTS << "\n";

            std::cout << " coll=" << std::setprecision(3) << met.collision_rate()
                      << " (" << std::setprecision(1) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " ("
              << std::setprecision(1) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Fig 47: Runtime
// ============================================================================

static void run_fig47_runtime() {
    std::cout << "\n=== Fig 47: Runtime ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<MethodType> METHODS = {
        MethodType::BASE, MethodType::WDRO_SAMPLING, MethodType::WDRO_INJECTION, MethodType::WDRO_COMBINED
    };
    const int N_ROLLOUTS = 400;
    const double SWITCH_PROB = 0.1;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "fig47_runtime.csv";
    std::ofstream ofs(filepath);
    ofs << "method,mean_solve_ms,p50_solve_ms,p95_solve_ms,max_solve_ms,"
        << "mean_constraint_time_pct\n";

    for (MethodType method : METHODS) {
        std::cout << "  method=" << method_name(method) << " ...";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        ExperimentConfig cfg = make_method_config(method, SWITCH_PROB, RARE_PROB);

        Metrics met;
        met.method = method_name(method);

        // Accumulate raw solve times and constraint construction times across rollouts
        std::vector<double> all_solve_times;
        double total_constraint_time = 0.0;
        double total_solve_time = 0.0;

        for (int i = 0; i < N_ROLLOUTS; ++i) {
            unsigned seed = static_cast<unsigned>(700000 + static_cast<int>(method) * 10000 + i);
            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        // Constraint time percentage is approximated from the difference
        // between total solve time and QP time. Since we don't track it
        // separately in the record, we report 0 as placeholder.
        double constraint_pct = 0.0;

        ofs << met.method << ","
            << std::fixed << std::setprecision(3)
            << met.mean_solve_ms() << ","
            << met.median_solve_ms() << ","
            << met.p95_solve() << ","
            << met.max_solve() << ","
            << std::setprecision(1) << constraint_pct << "\n";

        std::cout << " mean=" << std::setprecision(2) << met.mean_solve_ms() << "ms"
                  << " p95=" << met.p95_solve() << "ms"
                  << " (" << std::setprecision(1) << elapsed_sec(t1) << "s)\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " ("
              << std::setprecision(1) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Ensure output directory exists
    fs::create_directories(OUTPUT_DIR);

    // Parse filter arguments
    std::set<std::string> filters;
    for (int i = 1; i < argc; ++i) {
        filters.insert(argv[i]);
    }

    bool run_all = filters.empty();

    auto should_run = [&](const std::string& name) {
        return run_all || filters.count(name);
    };

    std::cout << "========================================\n";
    std::cout << "Paper Figures 40-47 Data Generation\n";
    std::cout << "========================================\n";
    if (!run_all) {
        std::cout << "Filters:";
        for (const auto& f : filters) std::cout << " " << f;
        std::cout << "\n";
    }

    auto overall_start = std::chrono::steady_clock::now();

    if (should_run("fig40")) run_fig40_risk_lift();
    if (should_run("fig41")) run_fig41_budget_scaling();
    if (should_run("fig42")) run_fig42_rare_mode_sweep();
    if (should_run("fig43")) run_fig43_geometry_ablation();
    if (should_run("fig44")) run_fig44_rho_sweep();
    if (should_run("fig45")) run_fig45_feasibility();
    if (should_run("fig46")) run_fig46_env_robustness();
    if (should_run("fig47")) run_fig47_runtime();

    double total_time = elapsed_sec(overall_start);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total time: " << std::fixed << std::setprecision(1)
              << total_time << "s";
    if (total_time > 60.0) {
        std::cout << " (" << std::setprecision(1) << total_time / 60.0 << " min)";
    }
    std::cout << "\n========================================\n";

    return 0;
}
