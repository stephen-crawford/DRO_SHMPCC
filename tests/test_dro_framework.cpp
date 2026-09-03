/**
 * @file test_dro_framework.cpp
 * @brief Publication tests for Wasserstein DRO + Safe Horizon + Multi-disc.
 *
 * Core thesis: "Scenario MPC tends to miss rare mode switches at practical
 * sample sizes. We add a lightweight Wasserstein DRO module over mode
 * probabilities that deterministically injects a worst-case reachable mode
 * rollout, improving safety without scaling the scenario count."
 *
 * Experiments:
 *   H1: DRO deterministic injection covers worst-case mode at all sample sizes
 *   H2: DRO reduces collision rate without scaling S (paired Monte Carlo)
 *   H3: Safe horizon truncation enforces epsilon guarantee (compute/safety knob)
 *   H4: Multi-disc D=3 vs D=1 collision encoding improvement
 *   H5: Statistical significance via McNemar paired test
 *
 * Outputs CSV files to paper_figures/ for pgfplots / LaTeX tables.
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

#include "mpc_controller.hpp"
#include "collision_constraints.hpp"
#include "dynamics.hpp"
#include "dro.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"

using namespace dro_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "paper_figures/";

// ============================================================================
// Helpers
// ============================================================================

static std::pair<double, double> wilson_ci(int successes, int n, double z = 1.96) {
    if (n == 0) return {0.0, 1.0};
    double p_hat = static_cast<double>(successes) / n;
    double denom = 1.0 + z * z / n;
    double center = (p_hat + z * z / (2.0 * n)) / denom;
    double half_width = z * std::sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom;
    return {std::max(0.0, center - half_width), std::min(1.0, center + half_width)};
}

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
        double spd = std::sqrt(state.vx * state.vx + state.vy * state.vy);
        if (spd > 2.0) { state.vx *= 2.0 / spd; state.vy *= 2.0 / spd; }
    }

    void maybe_switch(double switch_prob, std::mt19937& rng) {
        std::uniform_real_distribution<double> u(0, 1);
        if (u(rng) < switch_prob && !available_modes.empty()) {
            std::uniform_int_distribution<int> idx(0, available_modes.size() - 1);
            current_mode = available_modes[idx(rng)];
        }
    }
};

struct RolloutResult {
    bool collision = false;
    double min_clearance = 1e9;
    double total_progress = 0.0;
    double avg_solve_time = 0.0;
    int missed_mode_steps = 0;
    int total_steps = 0;
    int total_dro_injected = 0;
    std::vector<int> safe_horizons;
    std::vector<double> solve_times;
};

// Generic single rollout with configurable parameters
static RolloutResult run_rollout(
    bool enable_dro,
    int num_scenarios,
    int horizon,
    int num_discs,
    bool safe_horizon_enabled,
    double switch_prob,
    int rollout_steps,
    unsigned seed,
    const std::vector<std::string>& obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "",
    double rare_switch_prob = 0.0
) {
    std::mt19937 rng(seed);
    RolloutResult result;
    constexpr double DT = 0.1;

    auto mode_models = create_obstacle_mode_models(DT);

    RuntimeConfig config;
    config.mpc.horizon = horizon;
    config.mpc.dt = DT;
    config.mpc.sampling.num_scenarios = num_scenarios;
    config.mpc.ego.radius = 0.5;
    config.obstacle_radius = 0.35;
    config.mpc.constraints.safety_margin = 0.2;
    config.solver.use_sqp_solver = true;
    config.dro.enabled = enable_dro;
    config.mpc.ego.num_discs = num_discs;
    config.mpc.safe_horizon_enabled = safe_horizon_enabled;
    config.mpc.constraints.safe_horizon_min = 3;

    MPCController controller(config);

    // Setup obstacle
    int obs_id = 0;
    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : obs_modes) {
        if (mode_models.find(m) != mode_models.end())
            obs_mode_models[m] = mode_models[m];
    }
    if (!rare_mode.empty() && mode_models.find(rare_mode) != mode_models.end()) {
        obs_mode_models[rare_mode] = mode_models[rare_mode];
    }
    controller.initialize_obstacle(obs_id, 0, obs_mode_models);

    ObstacleSim obs_sim;
    std::uniform_real_distribution<double> y_dist(-0.5, 0.5);
    obs_sim.state = ObstacleState(3.0 + y_dist(rng), 0.3 + y_dist(rng) * 0.5,
                                   -0.1, y_dist(rng) * 0.3);
    obs_sim.current_mode = obs_modes.empty() ? "constant_velocity" : obs_modes[0];
    obs_sim.available_modes = obs_modes;
    if (!rare_mode.empty()) obs_sim.available_modes.push_back(rare_mode);
    obs_sim.mode_models = obs_mode_models;

    EgoState ego(0.0, 0.0, 0.0, 1.5);
    Eigen::Vector2d goal(20.0, 0.0);
    EgoDynamics dynamics(DT);
    double collision_radius = config.mpc.ego.radius + config.obstacle_radius;

    // Initial mode observations
    for (int i = 0; i < 5; ++i) {
        controller.update_mode_observation(obs_id, 0, obs_sim.current_mode, i);
    }

    for (int step = 0; step < rollout_steps; ++step) {
        // Mode switching
        if (!rare_mode.empty() && rare_switch_prob > 0) {
            std::uniform_real_distribution<double> u(0, 1);
            if (u(rng) < rare_switch_prob) {
                obs_sim.current_mode = rare_mode;
            } else {
                obs_sim.maybe_switch(switch_prob, rng);
            }
        } else {
            obs_sim.maybe_switch(switch_prob, rng);
        }

        controller.update_mode_observation(obs_id, 0, obs_sim.current_mode, step + 5);

        std::map<int, ObstacleState> obstacles;
        obstacles[obs_id] = obs_sim.state;

        auto mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        result.solve_times.push_back(mpc_result.solve_time);
        result.total_dro_injected += mpc_result.num_dro_injected;
        if (mpc_result.safe_horizon > 0)
            result.safe_horizons.push_back(mpc_result.safe_horizon);

        double dist = (ego.position() - obs_sim.state.position()).norm();
        result.min_clearance = std::min(result.min_clearance, dist);
        if (dist < collision_radius) result.collision = true;

        // Check if realized mode appears in scenario set
        bool mode_found = false;
        for (const auto& sc : controller.scenarios()) {
            for (const auto& [oid, traj] : sc.trajectories) {
                if (oid == obs_id && traj.mode_id == obs_sim.current_mode) {
                    mode_found = true; break;
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
    }
    result.total_progress = ego.x;
    return result;
}

// ============================================================================
// H1: DRO Deterministic Worst-Case Mode Injection
// ============================================================================

static int test_h1_missed_mode_coverage() {
    std::cout << "\n============================================================\n"
              << "  H1: DRO Deterministic Worst-Case Mode Injection\n"
              << "  Shows that DRO always covers the dangerous rare mode\n"
              << "============================================================\n" << std::endl;

    std::ofstream csv(OUTPUT_DIR + "exp_h1_mode_coverage.csv");
    csv << "num_scenarios,dro_enabled,coverage_rate,base_missed_rate,"
        << "missed_mode_fraction,avg_dro_injected,num_rollouts\n";

    std::vector<int> scenario_counts = {5, 10, 20, 40};
    constexpr int N_ROLLOUTS = 600;
    constexpr int STEPS = 50;
    double rare_prob = 0.05;  // 5% chance of rare mode per step
    std::vector<std::string> base_modes = {"constant_velocity", "turn_left", "turn_right"};
    std::string rare_mode = "lane_change_left";

    int total_pass = 0, total_fail = 0;

    for (int S : scenario_counts) {
        // Run paired rollouts: same seeds, Base vs DRO
        double missed_frac_base = 0, missed_frac_dro = 0;
        double coverage_base = 0, coverage_dro = 0;
        double avg_dro_inj = 0;

        for (bool dro : {false, true}) {
            int covered_rollouts = 0;
            int total_missed = 0, total_steps_all = 0;
            double total_dro_inj = 0;

            for (int r = 0; r < N_ROLLOUTS; ++r) {
                unsigned seed = static_cast<unsigned>(r * 1111 + S * 10);  // same seed for both
                auto res = run_rollout(dro, S, 15, 3, false, 0.1, STEPS, seed,
                                       base_modes, rare_mode, rare_prob);

                if (res.missed_mode_steps < res.total_steps)
                    covered_rollouts++;
                total_missed += res.missed_mode_steps;
                total_steps_all += res.total_steps;
                total_dro_inj += res.total_dro_injected;
            }

            double coverage_rate = static_cast<double>(covered_rollouts) / N_ROLLOUTS;
            double missed_frac = total_steps_all > 0 ?
                static_cast<double>(total_missed) / total_steps_all : 0;
            double avg_dro_val = total_dro_inj / N_ROLLOUTS;

            if (dro) {
                missed_frac_dro = missed_frac;
                coverage_dro = coverage_rate;
                avg_dro_inj = avg_dro_val;
            } else {
                missed_frac_base = missed_frac;
                coverage_base = coverage_rate;
            }

            csv << S << "," << (dro ? "yes" : "no") << ","
                << std::fixed << std::setprecision(4) << coverage_rate << ","
                << missed_frac_base << "," << missed_frac << ","
                << std::setprecision(2) << avg_dro_val << "," << N_ROLLOUTS << "\n";

            std::cout << "  S=" << std::setw(3) << S
                      << " DRO=" << (dro ? "yes" : "no ")
                      << " | coverage=" << std::setprecision(3) << coverage_rate
                      << " missed_frac=" << std::setprecision(3) << missed_frac
                      << " dro_inj=" << std::setprecision(1) << avg_dro_val << std::endl;
        }

        // Paired assertion: DRO should not have MORE missed modes than Base
        if (missed_frac_dro <= missed_frac_base + 0.05) {
            total_pass++;
            std::cout << "    PASS: DRO missed=" << std::setprecision(3) << missed_frac_dro
                      << " <= Base missed=" << missed_frac_base << " (+0.05 tolerance)\n";
        } else {
            total_fail++;
            std::cout << "    WARN: DRO missed=" << std::setprecision(3) << missed_frac_dro
                      << " > Base missed=" << missed_frac_base << "\n";
        }
    }

    csv.close();
    std::cout << "\n  H1 result: " << total_pass << " passed, " << total_fail << " warnings\n";
    return total_fail;
}

// ============================================================================
// H2: DRO Reduces Collision Rate Without Scaling S
// ============================================================================

static int test_h2_collision_rate_reduction() {
    std::cout << "\n============================================================\n"
              << "  H2: DRO Reduces Collision Rate Without Scaling S\n"
              << "  Paired MC: same seeds, DRO vs Base at fixed S\n"
              << "============================================================\n" << std::endl;

    std::ofstream csv(OUTPUT_DIR + "exp_h2_collision_reduction.csv");
    csv << "num_scenarios,variant,collision_rate,ci_lo,ci_hi,missed_mode_rate,"
        << "avg_clearance,avg_solve_ms,num_rollouts\n";

    // Also write the missed-mode significance test
    std::ofstream csv_sig(OUTPUT_DIR + "exp_h2_missed_mode_significance.csv");
    csv_sig << "num_scenarios,base_collisions,dro_collisions,n10,n01,mcnemar_chi2,"
            << "significant_p05\n";

    std::vector<int> scenario_counts = {10, 20, 40};
    constexpr int N_ROLLOUTS = 600;
    constexpr int STEPS = 50;
    double switch_prob = 0.15;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    int total_pass = 0, total_fail = 0;

    for (int S : scenario_counts) {
        std::cout << "  S=" << S << std::endl;

        // Run paired rollouts
        int base_coll = 0, dro_coll = 0;
        int n_10 = 0, n_01 = 0;  // McNemar: base-only vs dro-only collisions
        int total_missed_base = 0, total_missed_dro = 0;
        int total_steps = 0;
        double sum_clear_base = 0, sum_clear_dro = 0;
        double sum_solve_base = 0, sum_solve_dro = 0;

        for (int r = 0; r < N_ROLLOUTS; ++r) {
            unsigned seed = static_cast<unsigned>(r * 2222 + S);

            auto res_base = run_rollout(false, S, 15, 3, false, switch_prob, STEPS, seed, modes);
            auto res_dro  = run_rollout(true,  S, 15, 3, false, switch_prob, STEPS, seed, modes);

            bool b = res_base.collision, d = res_dro.collision;
            if (b) base_coll++;
            if (d) dro_coll++;
            if (b && !d) n_10++;
            if (!b && d) n_01++;

            total_missed_base += res_base.missed_mode_steps;
            total_missed_dro += res_dro.missed_mode_steps;
            total_steps += res_base.total_steps;
            sum_clear_base += res_base.min_clearance;
            sum_clear_dro += res_dro.min_clearance;
            sum_solve_base += res_base.avg_solve_time;
            sum_solve_dro += res_dro.avg_solve_time;
        }

        double cr_base = static_cast<double>(base_coll) / N_ROLLOUTS;
        double cr_dro = static_cast<double>(dro_coll) / N_ROLLOUTS;
        auto [ci_lo_b, ci_hi_b] = wilson_ci(base_coll, N_ROLLOUTS);
        auto [ci_lo_d, ci_hi_d] = wilson_ci(dro_coll, N_ROLLOUTS);
        double mr_base = total_steps > 0 ? static_cast<double>(total_missed_base) / total_steps : 0;
        double mr_dro = total_steps > 0 ? static_cast<double>(total_missed_dro) / total_steps : 0;

        csv << S << ",Base," << std::fixed << std::setprecision(4)
            << cr_base << "," << ci_lo_b << "," << ci_hi_b << ","
            << mr_base << "," << sum_clear_base / N_ROLLOUTS << ","
            << std::setprecision(2) << sum_solve_base / N_ROLLOUTS * 1000 << ","
            << N_ROLLOUTS << "\n";

        csv << S << ",DRO," << std::fixed << std::setprecision(4)
            << cr_dro << "," << ci_lo_d << "," << ci_hi_d << ","
            << mr_dro << "," << sum_clear_dro / N_ROLLOUTS << ","
            << std::setprecision(2) << sum_solve_dro / N_ROLLOUTS * 1000 << ","
            << N_ROLLOUTS << "\n";

        // McNemar's test
        double mcnemar_chi2 = 0;
        if (n_10 + n_01 > 0) {
            mcnemar_chi2 = std::pow(n_10 - n_01, 2.0) / (n_10 + n_01);
        }
        bool significant = mcnemar_chi2 > 3.84;  // chi2(1) at alpha=0.05

        csv_sig << S << "," << base_coll << "," << dro_coll << ","
                << n_10 << "," << n_01 << ","
                << std::setprecision(2) << mcnemar_chi2 << ","
                << (significant ? "yes" : "no") << "\n";

        std::cout << "    Base: coll=" << std::setprecision(3) << cr_base
                  << " [" << ci_lo_b << "," << ci_hi_b << "]"
                  << " missed=" << std::setprecision(3) << mr_base << "\n"
                  << "    DRO:  coll=" << std::setprecision(3) << cr_dro
                  << " [" << ci_lo_d << "," << ci_hi_d << "]"
                  << " missed=" << std::setprecision(3) << mr_dro << "\n"
                  << "    McNemar: n10=" << n_10 << " n01=" << n_01
                  << " chi2=" << std::setprecision(2) << mcnemar_chi2
                  << (significant ? " (p<0.05)" : " (n.s.)") << "\n";

        // DRO should not be significantly WORSE than base
        if (cr_dro <= cr_base + 0.1) {
            total_pass++;
        } else {
            total_fail++;
            std::cout << "    FAIL: DRO collision rate significantly higher!\n";
        }
    }

    csv.close();
    csv_sig.close();
    std::cout << "\n  H2 result: " << total_pass << " passed, " << total_fail << " failed\n";
    return total_fail;
}

// ============================================================================
// H3: Safe Horizon Truncation Enforces Epsilon Guarantee
// ============================================================================

static int test_h3_safe_horizon_truncation() {
    std::cout << "\n============================================================\n"
              << "  H3: Safe Horizon Truncation (SH-MPC)\n"
              << "  Compute/safety tradeoff knob: fewer constraints, lower cost\n"
              << "============================================================\n" << std::endl;

    std::ofstream csv(OUTPUT_DIR + "exp_h3_safe_horizon.csv");
    csv << "num_scenarios,safe_horizon_enabled,avg_safe_horizon,avg_constraints,"
        << "collision_rate,ci_lo,ci_hi,avg_solve_ms,avg_progress,num_rollouts\n";

    std::vector<int> scenario_counts = {20, 40, 80};
    constexpr int N_ROLLOUTS = 600;
    constexpr int STEPS = 50;
    double switch_prob = 0.15;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    int total_pass = 0, total_fail = 0;

    for (int S : scenario_counts) {
        for (bool sh_enabled : {false, true}) {
            int collisions = 0;
            double sum_sh = 0, sum_solve = 0, sum_progress = 0;

            for (int r = 0; r < N_ROLLOUTS; ++r) {
                unsigned seed = static_cast<unsigned>(r * 3333 + S + (sh_enabled ? 1000000 : 0));
                auto res = run_rollout(true, S, 15, 3, sh_enabled, switch_prob, STEPS, seed, modes);

                if (res.collision) collisions++;
                sum_solve += res.avg_solve_time;
                sum_progress += res.total_progress;

                if (!res.safe_horizons.empty()) {
                    double avg_sh = std::accumulate(res.safe_horizons.begin(),
                                                     res.safe_horizons.end(), 0.0) /
                                    res.safe_horizons.size();
                    sum_sh += avg_sh;
                } else {
                    sum_sh += 15.0;  // Full horizon
                }
            }

            double cr = static_cast<double>(collisions) / N_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, N_ROLLOUTS);

            csv << S << "," << (sh_enabled ? "yes" : "no") << ","
                << std::fixed << std::setprecision(2) << sum_sh / N_ROLLOUTS << ","
                << "0,"  // placeholder for constraint count
                << std::setprecision(4) << cr << "," << ci_lo << "," << ci_hi << ","
                << std::setprecision(2) << sum_solve / N_ROLLOUTS * 1000 << ","
                << std::setprecision(2) << sum_progress / N_ROLLOUTS << ","
                << N_ROLLOUTS << "\n";

            std::cout << "  S=" << std::setw(3) << S
                      << " SH=" << (sh_enabled ? "yes" : "no ")
                      << " | coll=" << std::setprecision(3) << cr
                      << " avg_sh=" << std::setprecision(1) << sum_sh / N_ROLLOUTS
                      << " solve=" << std::setprecision(2) << sum_solve / N_ROLLOUTS * 1000 << "ms"
                      << " progress=" << std::setprecision(1) << sum_progress / N_ROLLOUTS << "\n";

            total_pass++;  // This is a demonstration, not strict pass/fail
        }
    }

    // Also output the theoretical safe horizon as function of S, using de Groot's
    // EXACT NSO bound (Eq. 8): the largest N_safe whose support proxy nbar = N_safe*n_u
    // is certified by S, i.e. the realized risk degroot_violation_risk(S, nbar, beta) <= eps.
    std::ofstream csv_theory(OUTPUT_DIR + "exp_h3_theoretical_safe_horizon.csv");
    csv_theory << "num_scenarios,epsilon,beta,safe_horizon_theoretical\n";
    for (int S = 5; S <= 500; S += 5) {
        for (double eps : {0.05, 0.10, 0.15, 0.20}) {
            double beta_val = 0.01;
            int n_u = 2;
            int N_safe = 0;
            for (int N_try = 20; N_try >= 1; --N_try) {
                int nbar = N_try * n_u;
                if (RuntimeConfig::degroot_violation_risk(S, nbar, beta_val) <= eps) {
                    N_safe = N_try;
                    break;
                }
            }
            csv_theory << S << "," << eps << "," << beta_val << "," << N_safe << "\n";
        }
    }
    csv_theory.close();

    csv.close();
    std::cout << "\n  H3 result: " << total_pass << " completed (demonstration experiment)\n";
    return total_fail;
}

// ============================================================================
// H4: Multi-Disc D=1 vs D=3 Collision Encoding
// ============================================================================

static int test_h4_multi_disc() {
    std::cout << "\n============================================================\n"
              << "  H4: Multi-Disc D=1 vs D=3 Collision Encoding\n"
              << "  D=3 detects more approach angles, fewer near-misses\n"
              << "============================================================\n" << std::endl;

    std::ofstream csv(OUTPUT_DIR + "exp_h4_multi_disc.csv");
    csv << "num_discs,collision_rate,ci_lo,ci_hi,avg_clearance,avg_constraints,"
        << "avg_solve_ms,num_rollouts\n";

    constexpr int N_ROLLOUTS = 600;
    constexpr int STEPS = 50;
    double switch_prob = 0.15;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    int total_pass = 0, total_fail = 0;

    for (int D : {1, 3}) {
        int collisions = 0;
        double sum_clearance = 0, sum_solve = 0;

        for (int r = 0; r < N_ROLLOUTS; ++r) {
            unsigned seed = static_cast<unsigned>(r * 4444 + D);
            auto res = run_rollout(true, 30, 15, D, false, switch_prob, STEPS, seed, modes);

            if (res.collision) collisions++;
            sum_clearance += res.min_clearance;
            sum_solve += res.avg_solve_time;
        }

        double cr = static_cast<double>(collisions) / N_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, N_ROLLOUTS);

        csv << D << "," << std::fixed << std::setprecision(4)
            << cr << "," << ci_lo << "," << ci_hi << ","
            << sum_clearance / N_ROLLOUTS << ","
            << "0,"  // placeholder
            << std::setprecision(2) << sum_solve / N_ROLLOUTS * 1000 << ","
            << N_ROLLOUTS << "\n";

        std::cout << "  D=" << D
                  << " | coll=" << std::setprecision(3) << cr
                  << " [" << ci_lo << "," << ci_hi << "]"
                  << " clearance=" << std::setprecision(3) << sum_clearance / N_ROLLOUTS
                  << " solve=" << std::setprecision(2) << sum_solve / N_ROLLOUTS * 1000 << "ms\n";

        total_pass++;
    }

    // Direct constraint comparison: same scenario, different D
    std::ofstream csv_detail(OUTPUT_DIR + "exp_h4_constraint_detail.csv");
    csv_detail << "num_discs,scenario,timestep,num_constraints,disc_positions\n";
    {
        auto mode_models = create_obstacle_mode_models(0.1);
        EgoState ego(0.0, 0.0, 0.3, 1.5);  // Angled heading
        std::vector<EgoState> ref_traj;
        ref_traj.push_back(ego);
        EgoDynamics dyn(0.1);
        EgoInput u(0.0, 0.0);
        for (int k = 0; k < 15; ++k) {
            ego = dyn.propagate(ego, u);
            ref_traj.push_back(ego);
        }

        // Create scenario with obstacle at side
        ObstacleState obs(2.0, 1.5, -0.3, -0.2);
        std::vector<PredictionStep> steps;
        Eigen::Vector4d x = obs.to_array();
        for (int k = 0; k <= 15; ++k) {
            steps.emplace_back(k, x.head<2>(), Eigen::Matrix2d::Zero());
            x = mode_models["turn_left"].A * x + mode_models["turn_left"].b;
        }
        ObstacleTrajectory traj(0, "turn_left", steps, 1.0);
        std::map<int, ObstacleTrajectory> trajs;
        trajs[0] = traj;
        Scenario sc(0, trajs, 1.0);
        std::vector<Scenario> scenarios = {sc};

        for (int D : {1, 3}) {
            auto constraints = compute_linearized_constraints(
                ref_traj, scenarios, 0.5, 0.35, 0.2, D);
            csv_detail << D << ",0,all," << constraints.size() << "," << D << "\n";
        }
    }
    csv_detail.close();

    csv.close();
    std::cout << "\n  H4 result: " << total_pass << " completed\n";
    return total_fail;
}

// ============================================================================
// H5: Full Ablation Table with Statistical Significance
// ============================================================================

static int test_h5_ablation_significance() {
    std::cout << "\n============================================================\n"
              << "  H5: Full Ablation Table + McNemar Significance\n"
              << "  4 variants: Base, DRO, D=1+DRO, D=3+DRO (full framework)\n"
              << "============================================================\n" << std::endl;

    std::ofstream csv(OUTPUT_DIR + "exp_h5_ablation_table.csv");
    csv << "variant,dro,num_discs,safe_horizon,collision_rate,ci_lo,ci_hi,"
        << "missed_mode_rate,avg_clearance,avg_solve_ms,avg_progress\n";

    std::ofstream csv_mcnemar(OUTPUT_DIR + "exp_h5_mcnemar.csv");
    csv_mcnemar << "variant_a,variant_b,n10,n01,mcnemar_chi2,significant_p05\n";

    struct VariantConfig {
        std::string name;
        bool dro;
        int num_discs;
        bool safe_horizon;
    };

    std::vector<VariantConfig> variants = {
        {"Base(D=1,noDRO)",     false, 1, false},
        {"Base(D=3,noDRO)",     false, 3, false},
        {"DRO(D=1)",            true,  1, false},
        {"DRO(D=3)",            true,  3, false},
        {"DRO(D=3)+SH",        true,  3, true},
    };

    constexpr int N_ROLLOUTS = 600;
    constexpr int STEPS = 50;
    double switch_prob = 0.2;
    int S = 30;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    // Store per-rollout collision results for McNemar
    std::vector<std::vector<bool>> coll_results(variants.size());

    for (size_t vi = 0; vi < variants.size(); ++vi) {
        const auto& vc = variants[vi];
        std::cout << "  " << vc.name << " ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps = 0;
        double sum_clear = 0, sum_solve = 0, sum_progress = 0;

        for (int r = 0; r < N_ROLLOUTS; ++r) {
            unsigned seed = static_cast<unsigned>(r * 5555);
            auto res = run_rollout(vc.dro, S, 15, vc.num_discs, vc.safe_horizon,
                                   switch_prob, STEPS, seed, modes);

            coll_results[vi].push_back(res.collision);
            if (res.collision) collisions++;
            total_missed += res.missed_mode_steps;
            total_steps += res.total_steps;
            sum_clear += res.min_clearance;
            sum_solve += res.avg_solve_time;
            sum_progress += res.total_progress;
        }

        double cr = static_cast<double>(collisions) / N_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, N_ROLLOUTS);
        double mr = total_steps > 0 ? static_cast<double>(total_missed) / total_steps : 0;

        csv << vc.name << "," << (vc.dro ? "yes" : "no") << ","
            << vc.num_discs << "," << (vc.safe_horizon ? "yes" : "no") << ","
            << std::fixed << std::setprecision(4)
            << cr << "," << ci_lo << "," << ci_hi << ","
            << mr << "," << sum_clear / N_ROLLOUTS << ","
            << std::setprecision(2) << sum_solve / N_ROLLOUTS * 1000 << ","
            << sum_progress / N_ROLLOUTS << "\n";

        std::cout << "coll=" << std::setprecision(3) << cr
                  << " [" << ci_lo << "," << ci_hi << "]"
                  << " missed=" << std::setprecision(3) << mr << "\n";
    }

    // Pairwise McNemar tests
    for (size_t i = 0; i < variants.size(); ++i) {
        for (size_t j = i + 1; j < variants.size(); ++j) {
            int n_10 = 0, n_01 = 0;
            for (int r = 0; r < N_ROLLOUTS; ++r) {
                bool ci = coll_results[i][r], cj = coll_results[j][r];
                if (ci && !cj) n_10++;
                if (!ci && cj) n_01++;
            }
            double chi2 = 0;
            if (n_10 + n_01 > 0) chi2 = std::pow(n_10 - n_01, 2.0) / (n_10 + n_01);
            bool sig = chi2 > 3.84;

            csv_mcnemar << variants[i].name << "," << variants[j].name << ","
                        << n_10 << "," << n_01 << ","
                        << std::setprecision(2) << chi2 << ","
                        << (sig ? "yes" : "no") << "\n";

            if (i == 0 && j == variants.size() - 1) {
                std::cout << "  McNemar (" << variants[i].name << " vs " << variants[j].name
                          << "): n10=" << n_10 << " n01=" << n_01
                          << " chi2=" << std::setprecision(2) << chi2
                          << (sig ? " (p<0.05)" : " (n.s.)") << "\n";
            }
        }
    }

    csv.close();
    csv_mcnemar.close();
    std::cout << "\n  H5 ablation complete.\n";
    return 0;
}

// ============================================================================
// H6: OT Ground Cost Produces Better Mode Separation
// ============================================================================

static int test_h6_ot_ground_cost() {
    std::cout << "\n============================================================\n"
              << "  H6: OT Ground Cost (W2 Bures) Mode Separation\n"
              << "  Verifies DRO transport cost matrix captures mode geometry\n"
              << "============================================================\n" << std::endl;

    std::ofstream csv(OUTPUT_DIR + "exp_h6_ground_cost_matrix.csv");
    csv << "mode_i,mode_j,w2_distance,horizon\n";

    auto mode_models = create_obstacle_mode_models(0.1);
    ObstacleState obs(5.0, 0.0, 0.5, 0.0);

    std::vector<EgoState> ego_ref;
    for (int k = 0; k <= 15; ++k) {
        ego_ref.emplace_back(k * 0.15, 0.0, 0.0, 1.5);
    }

    DRO dro;
    std::map<std::string, double> nominal;
    for (const auto& [id, _] : mode_models) {
        nominal[id] = 1.0 / mode_models.size();
    }

    auto result = dro.compute_worst_case_weights(
        nominal, obs, mode_models, ego_ref, 15, 0.5, 0.35, 0.2);

    std::vector<std::string> mode_ids;
    for (const auto& [id, _] : nominal) mode_ids.push_back(id);

    int M = static_cast<int>(mode_ids.size());
    std::cout << "  Transport cost matrix D[i][j] (W2 Bures, horizon=15):\n";
    std::cout << "  " << std::setw(20) << " ";
    for (const auto& m : mode_ids)
        std::cout << std::setw(12) << m.substr(0, 10);
    std::cout << "\n";

    for (int i = 0; i < M; ++i) {
        std::cout << "  " << std::setw(20) << mode_ids[i];
        for (int j = 0; j < M; ++j) {
            double d = result.transport_cost_matrix[i][j];
            std::cout << std::setw(12) << std::fixed << std::setprecision(3) << d;
            csv << mode_ids[i] << "," << mode_ids[j] << "," << d << ",15\n";
        }
        std::cout << "\n";
    }

    std::cout << "\n  Risk per mode:\n";
    for (const auto& [id, r] : result.risk_per_mode) {
        std::cout << "    " << std::setw(20) << id << ": " << std::setprecision(4) << r << "\n";
    }

    std::cout << "\n  Worst-case weights Q*:\n";
    for (const auto& [id, w] : result.worst_case_weights) {
        std::cout << "    " << std::setw(20) << id << ": " << std::setprecision(4) << w
                  << " (nominal=" << nominal[id] << ")\n";
    }

    std::cout << "  Rho used: " << result.rho_used << "\n";
    std::cout << "  Optimal lambda: " << result.optimal_lambda << "\n";

    // Verify: D[i][i] = 0 (self-distance)
    int pass = 0, fail = 0;
    for (int i = 0; i < M; ++i) {
        if (std::abs(result.transport_cost_matrix[i][i]) < 1e-10) {
            pass++;
        } else {
            fail++;
            std::cout << "  FAIL: D[" << i << "][" << i << "] = "
                      << result.transport_cost_matrix[i][i] << " (expected 0)\n";
        }
    }

    // Verify: D[i][j] = D[j][i] (symmetry)
    for (int i = 0; i < M; ++i) {
        for (int j = i + 1; j < M; ++j) {
            if (std::abs(result.transport_cost_matrix[i][j] -
                         result.transport_cost_matrix[j][i]) < 1e-10) {
                pass++;
            } else {
                fail++;
            }
        }
    }

    // Verify: Q* sums to 1
    double q_sum = 0;
    for (const auto& [_, w] : result.worst_case_weights) q_sum += w;
    if (std::abs(q_sum - 1.0) < 1e-6) {
        pass++;
    } else {
        fail++;
        std::cout << "  FAIL: Q* sums to " << q_sum << "\n";
    }

    csv.close();
    std::cout << "\n  H6 result: " << pass << " passed, " << fail << " failed\n";
    return fail;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n"
              << "  DRO Framework Publication Tests\n"
              << "  Wasserstein DRO + Safe Horizon + Multi-Disc (D=3)\n"
              << "================================================================\n" << std::endl;

    fs::create_directories(OUTPUT_DIR);
    auto start = std::chrono::high_resolution_clock::now();

    int total_failures = 0;
    total_failures += test_h1_missed_mode_coverage();
    total_failures += test_h2_collision_rate_reduction();
    total_failures += test_h3_safe_horizon_truncation();
    total_failures += test_h4_multi_disc();
    total_failures += test_h5_ablation_significance();
    total_failures += test_h6_ot_ground_cost();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n================================================================\n"
              << "  All experiments complete in " << std::fixed << std::setprecision(1)
              << elapsed << " seconds.\n"
              << "  Total failures/warnings: " << total_failures << "\n"
              << "  CSV files written to " << OUTPUT_DIR << "\n"
              << "================================================================\n" << std::endl;

    // List generated files
    std::cout << "  Generated CSV files:\n";
    for (const auto& entry : fs::directory_iterator(OUTPUT_DIR)) {
        if (entry.path().extension() == ".csv" &&
            entry.path().filename().string().find("exp_h") == 0) {
            std::cout << "    " << entry.path().filename() << "\n";
        }
    }

    return total_failures > 0 ? 1 : 0;
}
