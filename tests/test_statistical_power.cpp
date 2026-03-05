/**
 * @file test_statistical_power.cpp
 * @brief High-power statistical tests for OT mode-learning differences.
 *
 * Implements four experiments:
 *   H1. High-power paired collision test   (McNemar, bootstrap CI, Cohen's h)
 *   H2. Missed-mode significance test      (z-test, permutation test, Cohen's h)
 *   H3. Epsilon-bound exceedance test      (one-sided binomial tests)
 *   H4. Increased-power ablation table     (1000 rollouts, tighter Wilson CIs)
 *
 * Outputs CSV files to paper_figures/ for post-processing by
 * generate_results_figures.py (fig10, fig11, fig12).
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
#include <sstream>
#include <cassert>
#include <filesystem>

#include "mpc_controller.hpp"
#include "contouring_mpc.hpp"
#include "optimal_transport_predictor.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

// ============================================================================
// Configuration
// ============================================================================

static const std::string OUTPUT_DIR = "paper_figures/";

// Rollout counts for statistical power (z-tests, bootstrap, McNemar).
static constexpr int H1_ROLLOUTS      = 2000;
static constexpr int H2_ROLLOUTS      = 1500;
static constexpr int H4_ROLLOUTS      = 1500;
static constexpr int ROLLOUT_STEPS    = 150;   // 15 s at dt=0.1
static constexpr double DT            = 0.1;
static constexpr int HORIZON          = 15;
static constexpr int BASE_SCENARIOS   = 40;
static constexpr int BOOTSTRAP_RESAMPLES = 10000;
static constexpr int PERMUTATION_ITERS   = 5000;

// ============================================================================
// Variant enum (mirrored from test_results_section.cpp)
// ============================================================================

enum class Variant { BASE, DRO_ONLY, OT_ONLY, OT_PLUS_DRO };

static const std::vector<Variant> ALL_VARIANTS = {
    Variant::BASE, Variant::DRO_ONLY, Variant::OT_ONLY, Variant::OT_PLUS_DRO
};

static std::string variant_name(Variant v) {
    switch (v) {
        case Variant::BASE:          return "Base";
        case Variant::DRO_ONLY:      return "DRO";
        case Variant::OT_ONLY:       return "OT";
        case Variant::OT_PLUS_DRO:   return "OT+DRO";
    }
    return "?";
}

// ============================================================================
// RolloutResult
// ============================================================================

struct RolloutResult {
    bool collision = false;
    double min_clearance = 1e9;
    double total_progress = 0.0;
    double avg_solve_time = 0.0;
    double max_solve_time = 0.0;
    int missed_mode_steps = 0;
    int total_steps = 0;
    int active_constraints = 0;
    std::vector<double> solve_times;
};

// ============================================================================
// Obstacle simulator (ground truth)
// ============================================================================

struct ObstacleSim {
    ObstacleState state;
    std::string current_mode;
    std::vector<std::string> available_modes;
    std::map<std::string, ModeModel> mode_models;

    void step(double dt, std::mt19937& rng) {
        if (mode_models.find(current_mode) == mode_models.end()) return;
        const auto& model = mode_models.at(current_mode);
        Eigen::VectorXd noise = Eigen::VectorXd::Zero(model.noise_dim());
        std::normal_distribution<double> nd(0, 1);
        for (int i = 0; i < model.noise_dim(); ++i) noise(i) = nd(rng) * 0.02;
        state = model.propagate(state, &noise);
        double spd = std::sqrt(state.vx*state.vx + state.vy*state.vy);
        if (spd > 2.0) {
            state.vx *= 2.0 / spd;
            state.vy *= 2.0 / spd;
        }
    }

    void maybe_switch(double switch_prob, std::mt19937& rng) {
        std::uniform_real_distribution<double> u(0, 1);
        if (u(rng) < switch_prob && !available_modes.empty()) {
            std::uniform_int_distribution<int> idx(0, available_modes.size() - 1);
            current_mode = available_modes[idx(rng)];
        }
    }
};

// ============================================================================
// run_single_rollout (same as test_results_section.cpp)
// ============================================================================

RolloutResult run_single_rollout(
    Variant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    unsigned seed,
    const std::vector<std::string>& obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"}
) {
    std::mt19937 rng(seed);
    RolloutResult result;

    auto mode_models = create_obstacle_mode_models(DT);

    ScenarioMPCConfig config;
    config.horizon = HORIZON;
    config.dt = DT;
    config.num_scenarios = num_scenarios;
    config.ego_radius = 0.5;
    config.obstacle_radius = 0.35;
    config.safety_margin = 0.2;
    config.use_sqp_solver = true;
    config.ensure_mode_coverage = true;

    bool use_ot = (variant == Variant::OT_ONLY || variant == Variant::OT_PLUS_DRO);
    bool use_dro = (variant == Variant::DRO_ONLY || variant == Variant::OT_PLUS_DRO);

    config.weight_type = use_ot ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
    config.enable_dro = use_dro;

    AdaptiveScenarioMPC controller(config);

    OptimalTransportPredictor ot_predictor(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

    int obs_id = 0;
    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : obs_modes) {
        if (mode_models.find(m) != mode_models.end())
            obs_mode_models[m] = mode_models[m];
    }

    controller.initialize_obstacle(obs_id, obs_mode_models);

    ObstacleSim obs_sim;
    std::uniform_real_distribution<double> y_dist(-0.5, 0.5);
    std::uniform_real_distribution<double> vx_dist(-0.3, 0.1);
    obs_sim.state = ObstacleState(3.0 + y_dist(rng), 0.3 + y_dist(rng) * 0.5,
                                   vx_dist(rng), y_dist(rng) * 0.3);
    obs_sim.current_mode = obs_modes.empty() ? "constant_velocity" : obs_modes[0];
    obs_sim.available_modes = obs_modes;
    obs_sim.mode_models = obs_mode_models;

    EgoState ego(0.0, 0.0, 0.0, 1.5);
    Eigen::Vector2d goal(20.0, 0.0);
    EgoDynamics dynamics(DT);

    double collision_radius = config.ego_radius + config.obstacle_radius;

    for (int i = 0; i < 5; ++i) {
        controller.update_mode_observation(obs_id, obs_sim.current_mode, i);
        if (use_ot) {
            ot_predictor.observe(obs_id, obs_sim.state.position(), obs_sim.current_mode);
            ot_predictor.advance_timestep();
        }
    }

    for (int step = 0; step < rollout_steps; ++step) {
        obs_sim.maybe_switch(switch_prob, rng);

        controller.update_mode_observation(obs_id, obs_sim.current_mode, step + 5);

        if (use_ot) {
            ot_predictor.observe(obs_id, obs_sim.state.position(), obs_sim.current_mode);
            ot_predictor.advance_timestep();
        }

        std::map<int, ObstacleState> obstacles;
        obstacles[obs_id] = obs_sim.state;

        auto mpc_result = controller.solve(ego, obstacles, goal, 1.5);

        result.solve_times.push_back(mpc_result.solve_time);
        result.active_constraints += static_cast<int>(mpc_result.active_scenarios.size());

        double dist = (ego.position() - obs_sim.state.position()).norm();
        result.min_clearance = std::min(result.min_clearance, dist);
        if (dist < collision_radius) {
            result.collision = true;
        }

        bool mode_found = false;
        for (const auto& sc : controller.scenarios()) {
            for (const auto& [oid, traj] : sc.trajectories) {
                if (oid == obs_id && traj.mode_id == obs_sim.current_mode) {
                    mode_found = true;
                    break;
                }
            }
            if (mode_found) break;
        }
        if (!mode_found) result.missed_mode_steps++;

        if (mpc_result.success && mpc_result.first_input().has_value()) {
            ego = dynamics.propagate(ego, mpc_result.first_input().value());
        }

        obs_sim.step(DT, rng);

        result.total_steps++;
    }

    if (!result.solve_times.empty()) {
        double sum = std::accumulate(result.solve_times.begin(), result.solve_times.end(), 0.0);
        result.avg_solve_time = sum / result.solve_times.size();
        result.max_solve_time = *std::max_element(result.solve_times.begin(), result.solve_times.end());
    }
    result.total_progress = ego.x;
    result.active_constraints /= std::max(1, result.total_steps);

    return result;
}

// ============================================================================
// Wilson confidence interval for binomial proportion
// ============================================================================

std::pair<double, double> wilson_ci(int successes, int n, double z = 1.96) {
    if (n == 0) return {0.0, 1.0};
    double p_hat = static_cast<double>(successes) / n;
    double denom = 1.0 + z * z / n;
    double center = (p_hat + z * z / (2.0 * n)) / denom;
    double half_width = z * std::sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom;
    return {std::max(0.0, center - half_width), std::min(1.0, center + half_width)};
}

// ============================================================================
// Statistical test functions
// ============================================================================

/// McNemar's chi-squared test statistic (with continuity correction)
double mcnemar_chi2(int n_10, int n_01) {
    int denom = n_10 + n_01;
    if (denom == 0) return 0.0;
    double num = std::abs(static_cast<double>(n_10 - n_01)) - 1.0;  // continuity correction
    if (num < 0) num = 0.0;
    return (num * num) / denom;
}

/// McNemar's chi-squared WITHOUT continuity correction (for comparison)
double mcnemar_chi2_no_correction(int n_10, int n_01) {
    int denom = n_10 + n_01;
    if (denom == 0) return 0.0;
    double diff = static_cast<double>(n_10 - n_01);
    return (diff * diff) / denom;
}

/// Two-proportion z-test statistic
double two_proportion_z(int x1, int n1, int x2, int n2) {
    double p1 = static_cast<double>(x1) / n1;
    double p2 = static_cast<double>(x2) / n2;
    double p_pool = static_cast<double>(x1 + x2) / (n1 + n2);
    double se = std::sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2));
    if (se < 1e-15) return 0.0;
    return (p1 - p2) / se;
}

/// Cohen's h effect size for proportions
double cohens_h(double p1, double p2) {
    return 2.0 * std::asin(std::sqrt(p1)) - 2.0 * std::asin(std::sqrt(p2));
}

/// Approximate p-value from z-statistic (two-tailed, standard normal)
double z_to_pvalue(double z) {
    // Using complementary error function: P(|Z|>z) = erfc(|z|/sqrt(2))
    return std::erfc(std::abs(z) / std::sqrt(2.0));
}

/// Approximate p-value from chi-squared with 1 df
double chi2_to_pvalue_1df(double chi2) {
    // P(X > chi2) for X ~ chi2(1) = erfc(sqrt(chi2/2))
    if (chi2 <= 0) return 1.0;
    return std::erfc(std::sqrt(chi2 / 2.0));
}

/// One-sided binomial test: P(X >= k) under Bin(n, p0) using normal approximation
double binomial_pvalue_greater(int k, int n, double p0) {
    double mu = n * p0;
    double sigma = std::sqrt(n * p0 * (1.0 - p0));
    if (sigma < 1e-15) return (k > mu) ? 0.0 : 1.0;
    double z = (k - 0.5 - mu) / sigma;  // continuity correction
    return 0.5 * std::erfc(z / std::sqrt(2.0));
}

/// One-sided binomial test: P(X <= k) under Bin(n, p0) using normal approximation
double binomial_pvalue_less_or_equal(int k, int n, double p0) {
    double mu = n * p0;
    double sigma = std::sqrt(n * p0 * (1.0 - p0));
    if (sigma < 1e-15) return (k <= mu) ? 1.0 : 0.0;
    double z = (k + 0.5 - mu) / sigma;  // continuity correction
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

// ============================================================================
// Experiment H1: High-power paired collision test
// ============================================================================

void run_experiment_h1() {
    std::cout << "\n========================================\n"
              << "  Experiment H1: High-Power Paired Collision Test\n"
              << "  (n=" << H1_ROLLOUTS << ", switch_prob=0.5)\n"
              << "========================================\n" << std::endl;

    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.5;

    // Per-rollout paired results
    std::ofstream f_paired(OUTPUT_DIR + "exp_h1_paired_high_power.csv");
    f_paired << "seed,base_collision,dro_collision,ot_collision,ot_dro_collision,"
             << "base_missed,dro_missed,ot_missed,ot_dro_missed,"
             << "base_clearance,dro_clearance,ot_clearance,ot_dro_clearance\n";

    // Collision counts per variant
    std::map<Variant, int> coll_counts;
    std::map<Variant, int> total_missed;
    std::map<Variant, int> total_steps;
    for (auto v : ALL_VARIANTS) {
        coll_counts[v] = 0;
        total_missed[v] = 0;
        total_steps[v] = 0;
    }

    // Paired collision vectors for bootstrap
    std::vector<int> base_coll_vec(H1_ROLLOUTS);
    std::vector<int> ot_dro_coll_vec(H1_ROLLOUTS);
    std::vector<int> ot_coll_vec(H1_ROLLOUTS);
    std::vector<int> dro_coll_vec(H1_ROLLOUTS);

    // McNemar contingency: Base vs OT+DROremes
    int mc_base_vs_otdro_n10 = 0, mc_base_vs_otdro_n01 = 0;
    // McNemar contingency: Base vs OT
    int mc_base_vs_ot_n10 = 0, mc_base_vs_ot_n01 = 0;
    // McNemar contingency: Base vs DROremes
    int mc_base_vs_dro_n10 = 0, mc_base_vs_dro_n01 = 0;

    for (int r = 0; r < H1_ROLLOUTS; ++r) {
        if (r % 100 == 0) {
            std::cout << "  Rollout " << r << "/" << H1_ROLLOUTS << " ..." << std::endl;
        }

        unsigned seed = static_cast<unsigned>(r * 1000 + 50);  // unique seeds

        auto res_base = run_single_rollout(Variant::BASE, switch_prob,
                                            BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_dro = run_single_rollout(Variant::DRO_ONLY, switch_prob,
                                           BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_ot = run_single_rollout(Variant::OT_ONLY, switch_prob,
                                          BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_ot_dro = run_single_rollout(Variant::OT_PLUS_DRO, switch_prob,
                                              BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);

        f_paired << seed << ","
                 << (res_base.collision ? 1 : 0) << ","
                 << (res_dro.collision ? 1 : 0) << ","
                 << (res_ot.collision ? 1 : 0) << ","
                 << (res_ot_dro.collision ? 1 : 0) << ","
                 << res_base.missed_mode_steps << ","
                 << res_dro.missed_mode_steps << ","
                 << res_ot.missed_mode_steps << ","
                 << res_ot_dro.missed_mode_steps << ","
                 << std::setprecision(4) << res_base.min_clearance << ","
                 << res_dro.min_clearance << ","
                 << res_ot.min_clearance << ","
                 << res_ot_dro.min_clearance << "\n";

        // Accumulate
        if (res_base.collision)   coll_counts[Variant::BASE]++;
        if (res_dro.collision)    coll_counts[Variant::DRO_ONLY]++;
        if (res_ot.collision)     coll_counts[Variant::OT_ONLY]++;
        if (res_ot_dro.collision) coll_counts[Variant::OT_PLUS_DRO]++;

        total_missed[Variant::BASE] += res_base.missed_mode_steps;
        total_missed[Variant::DRO_ONLY] += res_dro.missed_mode_steps;
        total_missed[Variant::OT_ONLY] += res_ot.missed_mode_steps;
        total_missed[Variant::OT_PLUS_DRO] += res_ot_dro.missed_mode_steps;

        total_steps[Variant::BASE] += res_base.total_steps;
        total_steps[Variant::DRO_ONLY] += res_dro.total_steps;
        total_steps[Variant::OT_ONLY] += res_ot.total_steps;
        total_steps[Variant::OT_PLUS_DRO] += res_ot_dro.total_steps;

        base_coll_vec[r] = res_base.collision ? 1 : 0;
        dro_coll_vec[r] = res_dro.collision ? 1 : 0;
        ot_coll_vec[r] = res_ot.collision ? 1 : 0;
        ot_dro_coll_vec[r] = res_ot_dro.collision ? 1 : 0;

        // McNemar: Base vs OT+DROremes
        if (res_base.collision && !res_ot_dro.collision) mc_base_vs_otdro_n10++;
        if (!res_base.collision && res_ot_dro.collision) mc_base_vs_otdro_n01++;
        // McNemar: Base vs OT
        if (res_base.collision && !res_ot.collision) mc_base_vs_ot_n10++;
        if (!res_base.collision && res_ot.collision) mc_base_vs_ot_n01++;
        // McNemar: Base vs DROremes
        if (res_base.collision && !res_dro.collision) mc_base_vs_dro_n10++;
        if (!res_base.collision && res_dro.collision) mc_base_vs_dro_n01++;
    }

    f_paired.close();

    // --- Statistical tests ---
    std::cout << "\n  --- H1 Statistical Results ---\n";

    // Collision rates
    for (auto v : ALL_VARIANTS) {
        double rate = static_cast<double>(coll_counts[v]) / H1_ROLLOUTS;
        auto [lo, hi] = wilson_ci(coll_counts[v], H1_ROLLOUTS);
        std::cout << "    " << variant_name(v) << ": " << coll_counts[v] << "/" << H1_ROLLOUTS
                  << " = " << std::setprecision(4) << rate
                  << " [" << lo << ", " << hi << "]\n";
    }

    // McNemar's tests
    auto print_mcnemar = [](const std::string& label, int n10, int n01) {
        double chi2_cc = mcnemar_chi2(n10, n01);
        double chi2_no = mcnemar_chi2_no_correction(n10, n01);
        double p_cc = chi2_to_pvalue_1df(chi2_cc);
        double p_no = chi2_to_pvalue_1df(chi2_no);
        std::cout << "    McNemar " << label << ": n10=" << n10 << " n01=" << n01
                  << " chi2(cc)=" << std::setprecision(2) << chi2_cc
                  << " p=" << std::setprecision(4) << p_cc
                  << " | chi2(no-cc)=" << std::setprecision(2) << chi2_no
                  << " p=" << std::setprecision(4) << p_no
                  << (p_cc < 0.05 ? " ***" : (p_no < 0.05 ? " **" : "")) << "\n";
    };

    print_mcnemar("Base vs OT+DRO", mc_base_vs_otdro_n10, mc_base_vs_otdro_n01);
    print_mcnemar("Base vs OT", mc_base_vs_ot_n10, mc_base_vs_ot_n01);
    print_mcnemar("Base vs DRO", mc_base_vs_dro_n10, mc_base_vs_dro_n01);

    // Two-proportion z-tests
    auto print_ztest = [](const std::string& label, int x1, int n1, int x2, int n2) {
        double z = two_proportion_z(x1, n1, x2, n2);
        double p = z_to_pvalue(z);
        double p1 = static_cast<double>(x1) / n1;
        double p2 = static_cast<double>(x2) / n2;
        double h = cohens_h(p1, p2);
        std::cout << "    z-test " << label << ": z=" << std::setprecision(3) << z
                  << " p=" << std::setprecision(4) << p
                  << " Cohen's h=" << std::setprecision(3) << h
                  << (p < 0.05 ? " ***" : "") << "\n";
    };

    print_ztest("Base vs OT+DRO", coll_counts[Variant::BASE], H1_ROLLOUTS,
                coll_counts[Variant::OT_PLUS_DRO], H1_ROLLOUTS);
    print_ztest("Base vs OT", coll_counts[Variant::BASE], H1_ROLLOUTS,
                coll_counts[Variant::OT_ONLY], H1_ROLLOUTS);
    print_ztest("Base vs DRO", coll_counts[Variant::BASE], H1_ROLLOUTS,
                coll_counts[Variant::DRO_ONLY], H1_ROLLOUTS);

    // Bootstrap CI for collision rate difference (Base - OT+DRO)
    std::cout << "\n    Bootstrap CI (collision rate diff: Base - OT+Ext):\n";
    std::mt19937 boot_rng(12345);
    std::uniform_int_distribution<int> boot_idx(0, H1_ROLLOUTS - 1);
    std::vector<double> boot_diffs(BOOTSTRAP_RESAMPLES);

    for (int b = 0; b < BOOTSTRAP_RESAMPLES; ++b) {
        int base_sum = 0, otdro_sum = 0;
        for (int i = 0; i < H1_ROLLOUTS; ++i) {
            int idx = boot_idx(boot_rng);
            base_sum += base_coll_vec[idx];
            otdro_sum += ot_dro_coll_vec[idx];
        }
        boot_diffs[b] = static_cast<double>(base_sum - otdro_sum) / H1_ROLLOUTS;
    }

    std::sort(boot_diffs.begin(), boot_diffs.end());
    double boot_lo = boot_diffs[static_cast<int>(0.025 * BOOTSTRAP_RESAMPLES)];
    double boot_hi = boot_diffs[static_cast<int>(0.975 * BOOTSTRAP_RESAMPLES)];
    double boot_mean = std::accumulate(boot_diffs.begin(), boot_diffs.end(), 0.0) / BOOTSTRAP_RESAMPLES;

    std::cout << "      Mean diff = " << std::setprecision(4) << boot_mean
              << "  95% CI = [" << boot_lo << ", " << boot_hi << "]\n";

    // Write bootstrap CI CSV
    std::ofstream f_boot(OUTPUT_DIR + "exp_h1_bootstrap_ci.csv");
    f_boot << "comparison,mean_diff,ci_lo,ci_hi,n_rollouts,n_resamples\n";

    // Base vs OT+DROremes
    f_boot << "Base_vs_OT+DRO," << std::setprecision(6) << boot_mean << ","
           << boot_lo << "," << boot_hi << "," << H1_ROLLOUTS << "," << BOOTSTRAP_RESAMPLES << "\n";

    // Base vs OT (separate bootstrap)
    {
        std::vector<double> bd(BOOTSTRAP_RESAMPLES);
        std::mt19937 rng2(23456);
        std::uniform_int_distribution<int> idx2(0, H1_ROLLOUTS - 1);
        for (int b = 0; b < BOOTSTRAP_RESAMPLES; ++b) {
            int bs = 0, os = 0;
            for (int i = 0; i < H1_ROLLOUTS; ++i) {
                int j = idx2(rng2);
                bs += base_coll_vec[j];
                os += ot_coll_vec[j];
            }
            bd[b] = static_cast<double>(bs - os) / H1_ROLLOUTS;
        }
        std::sort(bd.begin(), bd.end());
        double lo = bd[static_cast<int>(0.025 * BOOTSTRAP_RESAMPLES)];
        double hi = bd[static_cast<int>(0.975 * BOOTSTRAP_RESAMPLES)];
        double mn = std::accumulate(bd.begin(), bd.end(), 0.0) / BOOTSTRAP_RESAMPLES;
        f_boot << "Base_vs_OT," << std::setprecision(6) << mn << ","
               << lo << "," << hi << "," << H1_ROLLOUTS << "," << BOOTSTRAP_RESAMPLES << "\n";
        std::cout << "    Bootstrap Base vs OT: mean=" << std::setprecision(4) << mn
                  << " CI=[" << lo << ", " << hi << "]\n";
    }

    // Base vs DROremes (separate bootstrap)
    {
        std::vector<double> bd(BOOTSTRAP_RESAMPLES);
        std::mt19937 rng3(34567);
        std::uniform_int_distribution<int> idx3(0, H1_ROLLOUTS - 1);
        for (int b = 0; b < BOOTSTRAP_RESAMPLES; ++b) {
            int bs = 0, es = 0;
            for (int i = 0; i < H1_ROLLOUTS; ++i) {
                int j = idx3(rng3);
                bs += base_coll_vec[j];
                es += dro_coll_vec[j];
            }
            bd[b] = static_cast<double>(bs - es) / H1_ROLLOUTS;
        }
        std::sort(bd.begin(), bd.end());
        double lo = bd[static_cast<int>(0.025 * BOOTSTRAP_RESAMPLES)];
        double hi = bd[static_cast<int>(0.975 * BOOTSTRAP_RESAMPLES)];
        double mn = std::accumulate(bd.begin(), bd.end(), 0.0) / BOOTSTRAP_RESAMPLES;
        f_boot << "Base_vs_DRO," << std::setprecision(6) << mn << ","
               << lo << "," << hi << "," << H1_ROLLOUTS << "," << BOOTSTRAP_RESAMPLES << "\n";
        std::cout << "    Bootstrap Base vs DRO: mean=" << std::setprecision(4) << mn
                  << " CI=[" << lo << ", " << hi << "]\n";
    }

    f_boot.close();

    // Write summary with all test statistics
    std::ofstream f_summary(OUTPUT_DIR + "exp_h1_summary.csv");
    f_summary << "test,comparison,statistic,value,p_value,significant_005\n";

    // McNemar tests
    {
        double chi2 = mcnemar_chi2(mc_base_vs_otdro_n10, mc_base_vs_otdro_n01);
        double p = chi2_to_pvalue_1df(chi2);
        f_summary << "McNemar,Base_vs_OT+DRO,chi2," << std::setprecision(4) << chi2
                  << "," << p << "," << (p < 0.05 ? "yes" : "no") << "\n";
    }
    {
        double chi2 = mcnemar_chi2(mc_base_vs_ot_n10, mc_base_vs_ot_n01);
        double p = chi2_to_pvalue_1df(chi2);
        f_summary << "McNemar,Base_vs_OT,chi2," << std::setprecision(4) << chi2
                  << "," << p << "," << (p < 0.05 ? "yes" : "no") << "\n";
    }
    {
        double chi2 = mcnemar_chi2(mc_base_vs_dro_n10, mc_base_vs_dro_n01);
        double p = chi2_to_pvalue_1df(chi2);
        f_summary << "McNemar,Base_vs_DRO,chi2," << std::setprecision(4) << chi2
                  << "," << p << "," << (p < 0.05 ? "yes" : "no") << "\n";
    }

    // z-tests
    {
        double z = two_proportion_z(coll_counts[Variant::BASE], H1_ROLLOUTS,
                                     coll_counts[Variant::OT_PLUS_DRO], H1_ROLLOUTS);
        double p = z_to_pvalue(z);
        double h = cohens_h(static_cast<double>(coll_counts[Variant::BASE]) / H1_ROLLOUTS,
                            static_cast<double>(coll_counts[Variant::OT_PLUS_DRO]) / H1_ROLLOUTS);
        f_summary << "z-test,Base_vs_OT+DRO,z," << std::setprecision(4) << z
                  << "," << p << "," << (p < 0.05 ? "yes" : "no") << "\n";
        f_summary << "cohens_h,Base_vs_OT+DRO,h," << std::setprecision(4) << h
                  << ",NA,NA\n";
    }
    {
        double z = two_proportion_z(coll_counts[Variant::BASE], H1_ROLLOUTS,
                                     coll_counts[Variant::OT_ONLY], H1_ROLLOUTS);
        double p = z_to_pvalue(z);
        double h = cohens_h(static_cast<double>(coll_counts[Variant::BASE]) / H1_ROLLOUTS,
                            static_cast<double>(coll_counts[Variant::OT_ONLY]) / H1_ROLLOUTS);
        f_summary << "z-test,Base_vs_OT,z," << std::setprecision(4) << z
                  << "," << p << "," << (p < 0.05 ? "yes" : "no") << "\n";
        f_summary << "cohens_h,Base_vs_OT,h," << std::setprecision(4) << h
                  << ",NA,NA\n";
    }
    {
        double z = two_proportion_z(coll_counts[Variant::BASE], H1_ROLLOUTS,
                                     coll_counts[Variant::DRO_ONLY], H1_ROLLOUTS);
        double p = z_to_pvalue(z);
        double h = cohens_h(static_cast<double>(coll_counts[Variant::BASE]) / H1_ROLLOUTS,
                            static_cast<double>(coll_counts[Variant::DRO_ONLY]) / H1_ROLLOUTS);
        f_summary << "z-test,Base_vs_DRO,z," << std::setprecision(4) << z
                  << "," << p << "," << (p < 0.05 ? "yes" : "no") << "\n";
        f_summary << "cohens_h,Base_vs_DRO,h," << std::setprecision(4) << h
                  << ",NA,NA\n";
    }

    f_summary.close();
    std::cout << "\n  Experiment H1 complete. CSVs written to " << OUTPUT_DIR << std::endl;
}

// ============================================================================
// Experiment H2: Missed-mode significance test
// ============================================================================

void run_experiment_h2() {
    std::cout << "\n========================================\n"
              << "  Experiment H2: Missed-Mode Significance Test\n"
              << "  (n=" << H2_ROLLOUTS << " per variant per switch_prob)\n"
              << "========================================\n" << std::endl;

    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    std::vector<double> switch_probs = {0.2, 0.3, 0.5};

    std::ofstream f_miss(OUTPUT_DIR + "exp_h2_missed_mode_significance.csv");
    f_miss << "switch_prob,variant,missed_mode_rate,num_rollouts,"
           << "z_vs_base,p_vs_base,cohens_h_vs_base,"
           << "permutation_p_vs_base\n";

    for (double sp : switch_probs) {
        std::cout << "  switch_prob=" << sp << std::endl;

        // Collect per-rollout missed-mode rates for each variant
        std::map<Variant, std::vector<double>> missed_rates;
        std::map<Variant, int> total_missed_steps;
        std::map<Variant, int> total_all_steps;

        for (auto v : ALL_VARIANTS) {
            total_missed_steps[v] = 0;
            total_all_steps[v] = 0;
        }

        for (auto v : ALL_VARIANTS) {
            std::cout << "    " << variant_name(v) << " ... " << std::flush;

            for (int r = 0; r < H2_ROLLOUTS; ++r) {
                unsigned seed = static_cast<unsigned>(r * 2000 + static_cast<int>(sp * 1000));
                auto res = run_single_rollout(v, sp, BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);

                double rate = res.total_steps > 0 ?
                    static_cast<double>(res.missed_mode_steps) / res.total_steps : 0.0;
                missed_rates[v].push_back(rate);
                total_missed_steps[v] += res.missed_mode_steps;
                total_all_steps[v] += res.total_steps;
            }

            double overall_rate = total_all_steps[v] > 0 ?
                static_cast<double>(total_missed_steps[v]) / total_all_steps[v] : 0.0;
            std::cout << "missed_rate=" << std::setprecision(4) << overall_rate << std::endl;
        }

        // Compute statistics: each variant vs Base
        // For the z-test, we treat each rollout as a Bernoulli trial where
        // "success" = having any missed-mode steps
        double base_rate = total_all_steps[Variant::BASE] > 0 ?
            static_cast<double>(total_missed_steps[Variant::BASE]) / total_all_steps[Variant::BASE] : 0.0;

        // Count rollouts with any missed mode steps
        auto count_nonzero = [](const std::vector<double>& v) {
            int c = 0;
            for (auto x : v) if (x > 0) c++;
            return c;
        };

        int base_nonzero = count_nonzero(missed_rates[Variant::BASE]);

        for (auto v : ALL_VARIANTS) {
            double rate = total_all_steps[v] > 0 ?
                static_cast<double>(total_missed_steps[v]) / total_all_steps[v] : 0.0;

            double z = 0, p_z = 1.0, h = 0, perm_p = 1.0;

            if (v != Variant::BASE) {
                // Two-proportion z-test on rollout-level missed-mode occurrence
                int v_nonzero = count_nonzero(missed_rates[v]);
                z = two_proportion_z(base_nonzero, H2_ROLLOUTS, v_nonzero, H2_ROLLOUTS);
                p_z = z_to_pvalue(z);
                double p_base = static_cast<double>(base_nonzero) / H2_ROLLOUTS;
                double p_v = static_cast<double>(v_nonzero) / H2_ROLLOUTS;
                h = cohens_h(p_base, p_v);

                // Permutation test for mean missed-mode rate difference
                double observed_diff = 0;
                for (int i = 0; i < H2_ROLLOUTS; ++i) {
                    observed_diff += missed_rates[Variant::BASE][i] - missed_rates[v][i];
                }
                observed_diff /= H2_ROLLOUTS;

                // Pool and permute
                std::vector<double> pooled;
                pooled.reserve(2 * H2_ROLLOUTS);
                pooled.insert(pooled.end(), missed_rates[Variant::BASE].begin(),
                              missed_rates[Variant::BASE].end());
                pooled.insert(pooled.end(), missed_rates[v].begin(), missed_rates[v].end());

                std::mt19937 perm_rng(static_cast<unsigned>(sp * 10000 + static_cast<int>(v)));
                int extreme_count = 0;
                for (int perm = 0; perm < PERMUTATION_ITERS; ++perm) {
                    std::shuffle(pooled.begin(), pooled.end(), perm_rng);
                    double perm_diff = 0;
                    for (int i = 0; i < H2_ROLLOUTS; ++i) {
                        perm_diff += pooled[i] - pooled[H2_ROLLOUTS + i];
                    }
                    perm_diff /= H2_ROLLOUTS;
                    if (std::abs(perm_diff) >= std::abs(observed_diff)) {
                        extreme_count++;
                    }
                }
                perm_p = static_cast<double>(extreme_count + 1) / (PERMUTATION_ITERS + 1);

                std::cout << "    " << variant_name(v) << " vs Base: z=" << std::setprecision(3) << z
                          << " p_z=" << std::setprecision(4) << p_z
                          << " h=" << std::setprecision(3) << h
                          << " perm_p=" << std::setprecision(4) << perm_p
                          << (p_z < 0.05 ? " ***" : "") << "\n";
            }

            f_miss << sp << "," << variant_name(v) << ","
                   << std::setprecision(6) << rate << "," << H2_ROLLOUTS << ","
                   << std::setprecision(4) << z << "," << p_z << "," << h << ","
                   << perm_p << "\n";
        }
    }

    f_miss.close();
    std::cout << "\n  Experiment H2 complete. CSV written to " << OUTPUT_DIR << std::endl;
}

// ============================================================================
// Experiment H3: Epsilon-bound exceedance test
// ============================================================================

void run_experiment_h3() {
    std::cout << "\n========================================\n"
              << "  Experiment H3: Epsilon-Bound Exceedance Test\n"
              << "  (reusing H1 data at switch_prob=0.5)\n"
              << "========================================\n" << std::endl;

    // Re-run at switch_prob=0.5 with H1_ROLLOUTS to get collision counts
    // (In a production system we'd read from H1 CSV, but re-running is cleaner here)
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.5;
    double epsilon = 0.05;

    // Read H1 paired data if available, otherwise re-count
    // For simplicity, do a quick count run
    std::map<Variant, int> coll_counts;
    for (auto v : ALL_VARIANTS) coll_counts[v] = 0;

    std::cout << "  Running " << H1_ROLLOUTS << " rollouts per variant...\n";
    for (int r = 0; r < H1_ROLLOUTS; ++r) {
        if (r % 500 == 0) std::cout << "    " << r << "/" << H1_ROLLOUTS << std::endl;
        unsigned seed = static_cast<unsigned>(r * 1000 + 50);
        for (auto v : ALL_VARIANTS) {
            auto res = run_single_rollout(v, switch_prob, BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
            if (res.collision) coll_counts[v]++;
        }
    }

    std::ofstream f_eps(OUTPUT_DIR + "exp_h3_epsilon_exceedance.csv");
    f_eps << "variant,n_rollouts,collisions,collision_rate,epsilon,"
          << "binomial_p_greater,binomial_p_leq,"
          << "exceeds_epsilon,within_epsilon\n";

    std::cout << "\n  --- H3 Binomial Test Results (epsilon=" << epsilon << ") ---\n";

    for (auto v : ALL_VARIANTS) {
        int k = coll_counts[v];
        double rate = static_cast<double>(k) / H1_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(k, H1_ROLLOUTS);

        // One-sided: P(X >= k | p = epsilon) — evidence that true rate > epsilon
        double p_greater = binomial_pvalue_greater(k, H1_ROLLOUTS, epsilon);
        // One-sided: P(X <= k | p = epsilon) — evidence that true rate <= epsilon
        double p_leq = binomial_pvalue_less_or_equal(k, H1_ROLLOUTS, epsilon);

        bool exceeds = (p_greater < 0.05);
        bool within = (p_leq > 0.95 || ci_hi <= epsilon);  // alternative check

        f_eps << variant_name(v) << "," << H1_ROLLOUTS << "," << k << ","
              << std::setprecision(4) << rate << "," << epsilon << ","
              << std::setprecision(6) << p_greater << "," << p_leq << ","
              << (exceeds ? "yes" : "no") << "," << (within ? "yes" : "no") << "\n";

        std::cout << "    " << variant_name(v) << ": " << k << "/" << H1_ROLLOUTS
                  << " = " << std::setprecision(4) << rate
                  << "  CI=[" << ci_lo << "," << ci_hi << "]"
                  << "  P(rate>eps)=" << std::setprecision(4) << p_greater
                  << (exceeds ? " EXCEEDS" : "")
                  << "  P(rate<=eps)=" << p_leq
                  << (within ? " WITHIN" : "")
                  << "\n";
    }

    f_eps.close();
    std::cout << "\n  Experiment H3 complete. CSV written to " << OUTPUT_DIR << std::endl;
}

// ============================================================================
// Experiment H4: Increased-power ablation table
// ============================================================================

void run_experiment_h4() {
    std::cout << "\n========================================\n"
              << "  Experiment H4: Ablation Table (n=" << H4_ROLLOUTS << ")\n"
              << "  switch_prob=0.2\n"
              << "========================================\n" << std::endl;

    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.2;

    std::ofstream f_abl(OUTPUT_DIR + "exp_h4_ablation_table_1000.csv");
    f_abl << "variant,uses_ot,uses_dro,collision_rate,ci_lo,ci_hi,"
          << "missed_mode_rate,avg_progress,avg_clearance,avg_solve_ms\n";

    for (auto v : ALL_VARIANTS) {
        std::cout << "  " << variant_name(v) << " ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

        for (int r = 0; r < H4_ROLLOUTS; ++r) {
            if (r % 200 == 0 && r > 0) std::cout << r << " " << std::flush;
            unsigned seed = static_cast<unsigned>(r * 1000 + static_cast<int>(switch_prob * 100));
            auto res = run_single_rollout(v, switch_prob, BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);

            if (res.collision) collisions++;
            total_missed += res.missed_mode_steps;
            total_steps_all += res.total_steps;
            sum_progress += res.total_progress;
            sum_clearance += res.min_clearance;
            sum_solve += res.avg_solve_time;
        }

        double coll_rate = static_cast<double>(collisions) / H4_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, H4_ROLLOUTS);
        double missed_rate = total_steps_all > 0 ?
            static_cast<double>(total_missed) / total_steps_all : 0.0;
        double avg_progress = sum_progress / H4_ROLLOUTS;
        double avg_clearance = sum_clearance / H4_ROLLOUTS;
        double avg_solve = sum_solve / H4_ROLLOUTS * 1000;

        bool uses_ot = (v == Variant::OT_ONLY || v == Variant::OT_PLUS_DRO);
        bool uses_dro = (v == Variant::DRO_ONLY || v == Variant::OT_PLUS_DRO);

        f_abl << variant_name(v) << ","
               << (uses_ot ? "yes" : "no") << ","
               << (uses_dro ? "yes" : "no") << ","
               << std::setprecision(4) << coll_rate << "," << ci_lo << "," << ci_hi << ","
               << missed_rate << "," << avg_progress << "," << avg_clearance << ","
               << std::setprecision(2) << avg_solve << "\n";

        std::cout << "coll=" << std::setprecision(4) << coll_rate
                  << " [" << ci_lo << "," << ci_hi << "]"
                  << " missed=" << std::setprecision(4) << missed_rate << std::endl;
    }

    f_abl.close();
    std::cout << "\n  Experiment H4 complete. CSV written to " << OUTPUT_DIR << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n"
              << "  High-Power Statistical Tests: OT Mode-Learning Differences\n"
              << "================================================================\n" << std::endl;

    fs::create_directories(OUTPUT_DIR);

    auto start = std::chrono::high_resolution_clock::now();

    run_experiment_h1();
    run_experiment_h2();
    run_experiment_h3();
    run_experiment_h4();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n================================================================\n"
              << "  All statistical tests complete in " << std::fixed << std::setprecision(1)
              << elapsed << " seconds.\n"
              << "  CSV files written to " << OUTPUT_DIR << "\n"
              << "  Run: python3 ../generate_results_figures.py\n"
              << "================================================================\n" << std::endl;

    return 0;
}
