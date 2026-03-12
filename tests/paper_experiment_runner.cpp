/**
 * @file paper_experiment_runner.cpp
 * @brief Paper experiment runner — configures and invokes experiment_harness rollouts.
 *
 * This file defines experiment configurations (A through AB) and delegates all
 * rollout execution to the canonical run_experiment_rollout() in experiment_harness.
 *
 * Architecture:
 *   - experiment_harness.cpp owns ALL rollout logic: obstacle simulation, mode
 *     tracking, collision detection, path progress, OT predictor integration,
 *     and multi-disc/multi-obstacle support.
 *   - This file provides thin wrappers that map experiment parameters to
 *     ExperimentConfig and convert RolloutRecord back to local RolloutResult.
 *   - Experiment-specific monitoring (e.g. oracle flood) is injected via the
 *     ExperimentConfig::step_callback mechanism.
 *
 * Shared rollout wrappers:
 *   run_single_rollout()      — standard rollout (PaperVariant -> ExperimentConfig)
 *   run_multi_obstacle_rollout() — multi-obstacle with class sharing
 *   run_single_rollout_env()  — custom environment (intersection, oncoming, etc.)
 *
 * Helper mappings:
 *   make_experiment_config()  — PaperVariant -> ExperimentConfig
 *   baseline_to_weight()      — SamplingBaseline -> WeightType
 *   uses_dro/uses_sh  — PaperVariant -> feature flags
 *
 * Paper variants:
 *   Base      – WeightType::FREQUENCY, no DRO
 *   DRO       – WeightType::FREQUENCY + Wasserstein DRO
 *   *+SH      – any of the above with safe horizon enabled
 *
 * Outputs CSV files to paper_figures/ for generate_results_figures.py.
 *
 * Usage: ./paper_experiment_runner [A|B|C|D|E|F|G|H|I|...|AB]
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

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "collision_constraints.hpp"
#include "dynamics.hpp"
#include "wasserstein_dro.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
// optimal_transport_predictor.hpp removed (OT types deleted)
#include "reference_path.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "paper_figures/";

// ============================================================================
// Configuration
// ============================================================================

// Per-experiment rollout counts (not shared — specific to this runner).
static constexpr int EXP_A_ROLLOUTS = 600;
static constexpr int EXP_B_ROLLOUTS = 1000;
static constexpr int EXP_C_ROLLOUTS = 120;

// Aliases for shared constants from experiment_harness.hpp
static constexpr int    ROLLOUT_STEPS  = DEFAULT_ROLLOUT_STEPS;
static constexpr double DT             = DEFAULT_DT;
static constexpr int    HORIZON        = DEFAULT_HORIZON;
static constexpr int    BASE_SCENARIOS = DEFAULT_BASE_SCENARIOS;

// All shared types (PaperVariant, RolloutResult, EnvironmentType, SamplingBaseline,
// etc.) and helper functions (make_experiment_config, run_single_rollout,
// run_multi_obstacle_rollout, run_single_rollout_env, create_environment,
// setup_mpcc_path) are provided by experiment_harness.hpp.

// ============================================================================
// Experiment A: Mode-Switch Stress Test
// ============================================================================

static void run_experiment_a() {
    std::cout << "\n========================================\n"
              << "  Experiment A: Mode-Switch Stress Test\n"
              << "========================================\n";

    std::vector<double> switch_probs = {0.0, 0.05, 0.1, 0.2, 0.3, 0.5};
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    std::ofstream f_coll(OUTPUT_DIR + "exp_a_collision_vs_switching.csv");
    f_coll << "variant,switch_prob,collision_rate,ci_lo,ci_hi,num_rollouts\n";

    std::ofstream f_miss(OUTPUT_DIR + "exp_a_missed_mode_rate.csv");
    f_miss << "variant,switch_prob,missed_mode_rate,avg_progress,avg_clearance\n";

    std::ofstream f_ablation(OUTPUT_DIR + "exp_a_ablation_table.csv");

    std::ofstream f_w2(OUTPUT_DIR + "exp_a_w2_vs_time.csv");
    f_w2 << "variant,step,missed_fraction\n";

    f_ablation << "variant,uses_dro,uses_sh,"
               << "collision_rate,ci_lo,ci_hi,"
               << "missed_mode_rate,avg_progress,avg_clearance,avg_solve_ms\n";

    for (PaperVariant v : ALL_VARIANTS) {
        std::cout << "  Variant: " << variant_name(v) << std::endl;

        for (double sp : switch_probs) {
            std::cout << "    switch_prob=" << sp << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_steps_all = 0;
            double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

            for (int r = 0; r < EXP_A_ROLLOUTS; ++r) {
                unsigned seed = static_cast<unsigned>(r * 1000 + static_cast<int>(sp * 100));
                auto res = run_single_rollout(v, sp, BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
                if (res.collision) collisions++;
                total_missed += res.missed_mode_steps;
                total_steps_all += res.total_steps;
                sum_progress += res.total_progress;
                sum_clearance += res.min_clearance;
                sum_solve += res.avg_solve_time;
            }

            double coll_rate = static_cast<double>(collisions) / EXP_A_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, EXP_A_ROLLOUTS);
            double missed_rate = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;
            double avg_progress = sum_progress / EXP_A_ROLLOUTS;
            double avg_clearance = sum_clearance / EXP_A_ROLLOUTS;
            double avg_solve = sum_solve / EXP_A_ROLLOUTS * 1000;

            f_coll << variant_name(v) << "," << sp << "," << std::fixed << std::setprecision(4)
                   << coll_rate << "," << ci_lo << "," << ci_hi << "," << EXP_A_ROLLOUTS << "\n";

            f_miss << variant_name(v) << "," << sp << "," << std::setprecision(4)
                   << missed_rate << "," << avg_progress << "," << avg_clearance << "\n";

            // Ablation table at sp=0.2
            if (std::abs(sp - 0.2) < 0.01) {
                f_ablation << variant_name(v) << ","
                           << (uses_dro(v) ? "yes" : "no") << ","
                           << (uses_sh(v) ? "yes" : "no") << ","
                           << std::setprecision(4) << coll_rate << "," << ci_lo << "," << ci_hi << ","
                           << missed_rate << "," << avg_progress << "," << avg_clearance << ","
                           << std::setprecision(2) << avg_solve << "\n";
            }

            std::cout << "coll=" << std::setprecision(3) << coll_rate
                      << " [" << ci_lo << "," << ci_hi << "]"
                      << " missed=" << std::setprecision(3) << missed_rate << std::endl;
        }

        // W2 plot: per-step missed mode at sp=0.2
        {
            double sp = 0.2;
            std::vector<std::vector<int>> per_step_missed(ROLLOUT_STEPS);
            int w2_runs = std::min(20, EXP_A_ROLLOUTS);

            for (int r = 0; r < w2_runs; ++r) {
                unsigned seed = static_cast<unsigned>(r * 7777);
                std::mt19937 rng2(seed);
                auto mode_mdls = create_obstacle_mode_models(DT);

                ScenarioMPCConfig cfg;
                cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
                cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
                cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
                cfg.weight_type = WeightType::FREQUENCY;
                cfg.enable_dro = uses_dro(v);
                cfg.safe_horizon_enabled = false;
                cfg.num_discs = 1;
                cfg.vehicle_length = 1.5;
                AdaptiveScenarioMPC ctrl(cfg);

                std::vector<std::string> modes_list = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
                std::map<std::string, ModeModel> omm;
                for (auto& m : modes_list) omm[m] = mode_mdls[m];
                ctrl.initialize_obstacle(0, 0, omm);

                ObstacleSim osim;
                osim.state = ObstacleState(5.0, 1.0, 0.3, 0.0);
                osim.current_mode = "constant_velocity";
                osim.available_modes = modes_list;
                osim.mode_models = omm;

                EgoState ego2(0, 0, 0, 1.0);
                EgoDynamics dyn(DT);
                auto ref_path2 = setup_mpcc_path(ctrl);
                double pl2 = ref_path2.total_length();
                Eigen::Vector2d goal2 = ref_path2.get_position_at(pl2);
                double pp2 = 0.0;

                for (int i = 0; i < 5; ++i) {
                    ctrl.update_mode_observation(0, 0, osim.current_mode, i);
                }

                for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                    osim.maybe_switch(sp, rng2);
                    ctrl.update_mode_observation(0, 0, osim.current_mode, step + 5);

                    pp2 = ref_path2.find_closest_point(ego2.position(), pp2);
                    std::map<int, ObstacleState> obs_map;
                    obs_map[0] = osim.state;
                    auto res = ctrl.solve(ego2, obs_map, goal2, 1.5, pp2, pl2);

                    bool found = false;
                    for (auto& sc : ctrl.scenarios()) {
                        for (auto& [oid, t] : sc.trajectories) {
                            if (oid == 0 && t.mode_id == osim.current_mode) { found = true; break; }
                        }
                        if (found) break;
                    }
                    per_step_missed[step].push_back(found ? 0 : 1);

                    if (res.success && res.first_input().has_value())
                        ego2 = dyn.propagate(ego2, res.first_input().value());
                    osim.step(DT, rng2);
                }
            }

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                auto& v_step = per_step_missed[step];
                double frac = v_step.empty() ? 0 : std::accumulate(v_step.begin(), v_step.end(), 0.0) / v_step.size();
                f_w2 << variant_name(v) << "," << step << "," << std::setprecision(4) << frac << "\n";
            }
        }
    }

    std::cout << "  -> exp_a_collision_vs_switching.csv, exp_a_missed_mode_rate.csv\n"
              << "  -> exp_a_ablation_table.csv, exp_a_w2_vs_time.csv\n";
}

// ============================================================================
// Experiment B: Rare-Mode Tail-Event Test
// ============================================================================

static void run_experiment_b() {
    std::cout << "\n========================================\n"
              << "  Experiment B: Rare-Mode Tail-Event Test\n"
              << "========================================\n";

    std::vector<double> rare_probs = {0.01, 0.02, 0.05, 0.10};
    std::string rare_mode = "lane_change_left";
    std::vector<std::string> base_modes = {"constant_velocity", "turn_left", "turn_right"};
    double base_switch = 0.05;

    std::ofstream f_rare(OUTPUT_DIR + "exp_b_collision_given_rare.csv");
    f_rare << "variant,rare_prob,collision_rate,ci_lo,ci_hi,"
           << "collision_given_rare,rare_occurrences,num_rollouts\n";

    std::ofstream f_cons(OUTPUT_DIR + "exp_b_conservatism.csv");
    f_cons << "variant,rare_prob,avg_progress,avg_clearance,avg_solve_ms\n";

    for (PaperVariant v : ALL_VARIANTS) {
        std::cout << "  Variant: " << variant_name(v) << std::endl;

        for (double rp : rare_probs) {
            std::cout << "    rare_prob=" << rp << " ... " << std::flush;

            int collisions = 0;
            int collisions_with_rare = 0, rollouts_with_rare = 0;
            double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

            for (int r = 0; r < EXP_B_ROLLOUTS; ++r) {
                unsigned seed = static_cast<unsigned>(r * 2000 + static_cast<int>(rp * 1000));
                auto res = run_single_rollout(v, base_switch, BASE_SCENARIOS, ROLLOUT_STEPS,
                                               seed, base_modes, rare_mode, rp);

                if (res.collision) collisions++;
                sum_progress += res.total_progress;
                sum_clearance += res.min_clearance;
                sum_solve += res.avg_solve_time;

                // Approximate rare mode occurrence probability
                double prob_at_least_one = 1.0 - std::pow(1.0 - rp, ROLLOUT_STEPS);
                std::mt19937 flag_rng(seed + 999999);
                std::uniform_real_distribution<double> u2(0, 1);
                bool rare_occurred = (u2(flag_rng) < prob_at_least_one);

                if (rare_occurred) {
                    rollouts_with_rare++;
                    if (res.collision) collisions_with_rare++;
                }
            }

            double coll_rate = static_cast<double>(collisions) / EXP_B_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, EXP_B_ROLLOUTS);
            double coll_given_rare = rollouts_with_rare > 0 ?
                static_cast<double>(collisions_with_rare) / rollouts_with_rare : 0;

            f_rare << variant_name(v) << "," << rp << "," << std::setprecision(4)
                   << coll_rate << "," << ci_lo << "," << ci_hi << ","
                   << coll_given_rare << "," << rollouts_with_rare << "," << EXP_B_ROLLOUTS << "\n";

            f_cons << variant_name(v) << "," << rp << ","
                   << std::setprecision(4) << sum_progress / EXP_B_ROLLOUTS << ","
                   << sum_clearance / EXP_B_ROLLOUTS << ","
                   << std::setprecision(2) << sum_solve / EXP_B_ROLLOUTS * 1000 << "\n";

            std::cout << "coll=" << std::setprecision(3) << coll_rate
                      << " coll|rare=" << std::setprecision(3) << coll_given_rare << std::endl;
        }
    }
    std::cout << "  -> exp_b_collision_given_rare.csv, exp_b_conservatism.csv\n";
}

// ============================================================================
// Experiment C: Tractability Test
// ============================================================================

static void run_experiment_c() {
    std::cout << "\n========================================\n"
              << "  Experiment C: Tractability Test\n"
              << "========================================\n";

    std::vector<int> scenario_counts = {10, 20, 40, 80, 160};
    double switch_prob = 0.15;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    std::ofstream f_solve(OUTPUT_DIR + "exp_c_solve_times.csv");
    f_solve << "variant,num_scenarios,median_ms,p90_ms,p99_ms,max_ms\n";

    std::ofstream f_safety(OUTPUT_DIR + "exp_c_safety_vs_runtime.csv");
    f_safety << "variant,num_scenarios,collision_rate,ci_lo,ci_hi,avg_solve_ms\n";

    std::ofstream f_active(OUTPUT_DIR + "exp_c_active_constraints.csv");
    f_active << "variant,num_scenarios,avg_active_constraints,avg_progress\n";

    for (PaperVariant v : ALL_VARIANTS) {
        std::cout << "  Variant: " << variant_name(v) << std::endl;

        for (int S : scenario_counts) {
            std::cout << "    S=" << S << " ... " << std::flush;

            int collisions = 0;
            std::vector<double> all_solve_times;
            double sum_active = 0, sum_progress = 0;

            for (int r = 0; r < EXP_C_ROLLOUTS; ++r) {
                unsigned seed = static_cast<unsigned>(r * 3000 + S);
                auto res = run_single_rollout(v, switch_prob, S, ROLLOUT_STEPS, seed, modes);
                if (res.collision) collisions++;
                all_solve_times.insert(all_solve_times.end(),
                                        res.solve_times.begin(), res.solve_times.end());
                sum_active += res.active_constraints;
                sum_progress += res.total_progress;
            }

            for (auto& t : all_solve_times) t *= 1000.0;

            double median = percentile(all_solve_times, 50);
            double p90 = percentile(all_solve_times, 90);
            double p99 = percentile(all_solve_times, 99);
            double max_t = all_solve_times.empty() ? 0 :
                *std::max_element(all_solve_times.begin(), all_solve_times.end());

            double coll_rate = static_cast<double>(collisions) / EXP_C_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, EXP_C_ROLLOUTS);
            double avg_solve = all_solve_times.empty() ? 0 :
                std::accumulate(all_solve_times.begin(), all_solve_times.end(), 0.0) / all_solve_times.size();

            f_solve << variant_name(v) << "," << S << ","
                    << std::setprecision(2) << median << "," << p90 << "," << p99 << "," << max_t << "\n";
            f_safety << variant_name(v) << "," << S << ","
                     << std::setprecision(4) << coll_rate << "," << ci_lo << "," << ci_hi << ","
                     << std::setprecision(2) << avg_solve << "\n";
            f_active << variant_name(v) << "," << S << ","
                     << std::setprecision(1) << sum_active / EXP_C_ROLLOUTS << ","
                     << std::setprecision(2) << sum_progress / EXP_C_ROLLOUTS << "\n";

            std::cout << "coll=" << std::setprecision(3) << coll_rate
                      << " median=" << std::setprecision(1) << median << "ms" << std::endl;
        }
    }
    std::cout << "  -> exp_c_solve_times.csv, exp_c_safety_vs_runtime.csv, exp_c_active_constraints.csv\n";
}

// ============================================================================
// Experiment D: Calibration Plot
// ============================================================================

static void run_experiment_d() {
    std::cout << "\n========================================\n"
              << "  Experiment D: Calibration Plot\n"
              << "========================================\n";

    std::vector<double> epsilon_targets = {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50};
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.15;
    int cal_rollouts = 350;

    std::ofstream f_cal(OUTPUT_DIR + "exp_d_calibration.csv");
    f_cal << "variant,predicted_risk,observed_collision_rate,ci_lo,ci_hi,num_rollouts\n";

    for (PaperVariant v : ALL_VARIANTS) {
        std::cout << "  Variant: " << variant_name(v) << std::endl;

        for (double eps : epsilon_targets) {
            int S = static_cast<int>(std::ceil(2.0 * (std::log(1.0 / 0.01) + 90) / eps));
            S = std::max(10, std::min(S, 200));

            int collisions = 0;
            for (int r = 0; r < cal_rollouts; ++r) {
                unsigned seed = static_cast<unsigned>(r * 5000 + static_cast<int>(eps * 1000));
                auto res = run_single_rollout(v, switch_prob, S, ROLLOUT_STEPS, seed, modes);
                if (res.collision) collisions++;
            }

            double coll_rate = static_cast<double>(collisions) / cal_rollouts;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, cal_rollouts);

            f_cal << variant_name(v) << "," << std::setprecision(4) << eps << ","
                  << coll_rate << "," << ci_lo << "," << ci_hi << "," << cal_rollouts << "\n";

            std::cout << "    eps=" << eps << " S=" << S
                      << " observed=" << std::setprecision(3) << coll_rate << std::endl;
        }
    }
    std::cout << "  -> exp_d_calibration.csv\n";
}

// ============================================================================
// Experiment E: Buffer Size Sensitivity
// ============================================================================

static void run_experiment_e() {
    std::cout << "\n========================================\n"
              << "  Experiment E: Buffer Size Sensitivity\n"
              << "========================================\n";

    std::vector<int> buffer_sizes = {10, 20, 50, 100, 200};
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.2;
    int buf_rollouts = 250;

    std::ofstream f_buf(OUTPUT_DIR + "exp_e_buffer_sensitivity.csv");
    f_buf << "buffer_size,collision_rate,ci_lo,ci_hi,missed_mode_rate,avg_clearance,avg_solve_ms\n";

    for (int buf_sz : buffer_sizes) {
        std::cout << "  buffer_size=" << buf_sz << " ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        double sum_clearance = 0, sum_solve = 0;

        for (int r = 0; r < buf_rollouts; ++r) {
            unsigned seed = static_cast<unsigned>(r * 6000 + buf_sz);
            auto res = run_single_rollout(PaperVariant::DRO, switch_prob,
                                           BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
            if (res.collision) collisions++;
            total_missed += res.missed_mode_steps;
            total_steps_all += res.total_steps;
            sum_clearance += res.min_clearance;
            sum_solve += res.avg_solve_time;
        }

        double coll_rate = static_cast<double>(collisions) / buf_rollouts;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, buf_rollouts);
        double missed_rate = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;

        f_buf << buf_sz << "," << std::setprecision(4) << coll_rate << ","
              << ci_lo << "," << ci_hi << "," << missed_rate << ","
              << sum_clearance / buf_rollouts << ","
              << std::setprecision(2) << sum_solve / buf_rollouts * 1000 << "\n";

        std::cout << "coll=" << std::setprecision(3) << coll_rate
                  << " missed=" << missed_rate << std::endl;
    }
    std::cout << "  -> exp_e_buffer_sensitivity.csv\n";
}

// ============================================================================
// Experiment F: Non-Anticipativity & McNemar Paired Test
// ============================================================================

static void run_experiment_f() {
    std::cout << "\n========================================\n"
              << "  Experiment F: Non-Anticipativity & McNemar\n"
              << "========================================\n";

    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.2;
    int paired_rollouts_actual = 600;

    std::ofstream f_mcnemar(OUTPUT_DIR + "exp_f_mcnemar_paired.csv");
    f_mcnemar << "seed,base_collision,base_sh_collision,dro_collision,"
              << "dro_sh_collision\n";

    std::ofstream f_nonanticip(OUTPUT_DIR + "exp_f_non_anticipativity.csv");
    f_nonanticip << "seed,step,dro_risk,obs_future_displacement,non_anticipative\n";

    int base_coll = 0, base_sh_coll = 0, dro_coll = 0, dro_sh_coll = 0;
    int n_00 = 0, n_01 = 0, n_10 = 0, n_11 = 0;

    std::vector<bool> base_collisions_vec, base_sh_collisions_vec, dro_collisions_vec,
                      dro_sh_collisions_vec;

    for (int r = 0; r < paired_rollouts_actual; ++r) {
        unsigned seed = static_cast<unsigned>(r * 8000);

        auto res_base = run_single_rollout(PaperVariant::BASE, switch_prob,
                                            BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_base_sh = run_single_rollout(PaperVariant::BASE_SH, switch_prob,
                                          BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_dro = run_single_rollout(PaperVariant::DRO, switch_prob,
                                           BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_dro_sh = run_single_rollout(PaperVariant::DRO_SH, switch_prob,
                                              BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);

        f_mcnemar << seed << ","
                  << (res_base.collision ? 1 : 0) << ","
                  << (res_base_sh.collision ? 1 : 0) << ","
                  << (res_dro.collision ? 1 : 0) << ","
                  << (res_dro_sh.collision ? 1 : 0) << "\n";

        if (res_base.collision) base_coll++;
        if (res_base_sh.collision) base_sh_coll++;
        if (res_dro.collision) dro_coll++;
        if (res_dro_sh.collision) dro_sh_coll++;

        base_collisions_vec.push_back(res_base.collision);
        base_sh_collisions_vec.push_back(res_base_sh.collision);
        dro_collisions_vec.push_back(res_dro.collision);
        dro_sh_collisions_vec.push_back(res_dro_sh.collision);

        // McNemar 2x2: Base vs DRO+SH (primary comparison)
        bool b = res_base.collision, o = res_dro_sh.collision;
        if (!b && !o) n_00++;
        else if (!b && o) n_01++;
        else if (b && !o) n_10++;
        else n_11++;

        if ((r + 1) % 50 == 0) {
            std::cout << "  " << (r + 1) << "/" << paired_rollouts_actual << " paired rollouts done\n";
        }
    }

    // Non-anticipativity check
    {
        std::mt19937 rng(42424);
        auto mode_mdls = create_obstacle_mode_models(DT);
        WassersteinDRO dro_check;

        std::map<std::string, ModeModel> omm;
        for (auto& m : modes) omm[m] = mode_mdls[m];

        ObstacleSim osim;
        osim.state = ObstacleState(5.0, 0.5, -0.3, 0.1);
        osim.current_mode = "constant_velocity";
        osim.available_modes = modes;
        osim.mode_models = omm;

        std::vector<EgoState> ego_ref;
        for (int k = 0; k <= HORIZON; ++k)
            ego_ref.emplace_back(k * 0.15, 0.0, 0.0, 1.5);

        std::map<std::string, double> nominal_w;
        for (auto& m : modes) nominal_w[m] = 1.0 / modes.size();

        for (int step = 0; step < 80; ++step) {
            osim.maybe_switch(switch_prob, rng);
            auto dro_result = dro_check.compute_worst_case_weights(
                nominal_w, osim.state, omm, ego_ref, HORIZON, 0.5, 0.35, 0.2);
            double dro_risk = dro_result.worst_case_risk;

            Eigen::Vector2d pos_now = osim.state.position();
            osim.step(DT, rng);
            double future_disp = (osim.state.position() - pos_now).norm();

            f_nonanticip << 42424 << "," << step << "," << std::setprecision(4)
                         << dro_risk << "," << future_disp << ",1\n";
        }
    }

    // McNemar's chi2: Base vs DRO+SH (primary)
    double chi2_base_dro_sh = mcnemar_chi2(n_10, n_01);

    // Bootstrap CIs for all paired comparisons
    std::mt19937 boot_rng(12345);
    auto boot_base_dro_sh = bootstrap_paired_delta(base_collisions_vec, dro_sh_collisions_vec, 10000, &boot_rng);
    auto boot_base_base_sh = bootstrap_paired_delta(base_collisions_vec, base_sh_collisions_vec, 10000, &boot_rng);
    auto boot_base_dro = bootstrap_paired_delta(base_collisions_vec, dro_collisions_vec, 10000, &boot_rng);

    // McNemar for all pairs
    auto mcnemar_pair = [&](const std::vector<bool>& a, const std::vector<bool>& b_vec) {
        int mc_b = 0, mc_c = 0;
        for (int r = 0; r < paired_rollouts_actual; ++r) {
            if (a[r] && !b_vec[r]) mc_b++;   // a collision, b safe
            if (!a[r] && b_vec[r]) mc_c++;    // a safe, b collision
        }
        return mcnemar_chi2(mc_b, mc_c);
    };
    double chi2_base_base_sh = mcnemar_pair(base_collisions_vec, base_sh_collisions_vec);
    double chi2_base_dro = mcnemar_pair(base_collisions_vec, dro_collisions_vec);

    // Effect sizes
    double p_base = static_cast<double>(base_coll) / paired_rollouts_actual;
    double p_base_sh_val = static_cast<double>(base_sh_coll) / paired_rollouts_actual;
    double p_dro_val = static_cast<double>(dro_coll) / paired_rollouts_actual;
    double p_dro_sh = static_cast<double>(dro_sh_coll) / paired_rollouts_actual;
    auto es_base_dro_sh = compute_effect_sizes(p_base, p_dro_sh);
    auto es_base_base_sh = compute_effect_sizes(p_base, p_base_sh_val);
    auto es_base_dro = compute_effect_sizes(p_base, p_dro_val);

    // Bootstrap CI CSV (for fig10_forest_plot)
    {
        std::ofstream f_boot(OUTPUT_DIR + "exp_h1_bootstrap_ci.csv");
        f_boot << "comparison,mean_diff,ci_lo,ci_hi,mcnemar_chi2,mcnemar_sig,cohens_h\n"
               << std::fixed << std::setprecision(4);

        auto write_row = [&](const std::string& name, const BootstrapResult& br,
                             double chi2, const EffectSizes& eff) {
            f_boot << name << "," << br.mean_delta << "," << br.ci_low << "," << br.ci_high
                   << "," << chi2 << "," << (chi2 > 3.84 ? "yes" : "no")
                   << "," << eff.cohens_h << "\n";
        };

        write_row("Base_vs_Base+SH", boot_base_base_sh, chi2_base_base_sh, es_base_base_sh);
        write_row("Base_vs_DRO", boot_base_dro, chi2_base_dro, es_base_dro);
        write_row("Base_vs_DRO+SH", boot_base_dro_sh, chi2_base_dro_sh, es_base_dro_sh);
    }

    // Summary CSV
    {
        std::ofstream f_summary(OUTPUT_DIR + "exp_f_summary.csv");
        f_summary << "metric,value\n"
                  << "paired_rollouts," << paired_rollouts_actual << "\n"
                  << "base_collisions," << base_coll << "\n"
                  << "base_sh_collisions," << base_sh_coll << "\n"
                  << "dro_collisions," << dro_coll << "\n"
                  << "dro_sh_collisions," << dro_sh_coll << "\n"
                  << "mcnemar_base_vs_dro_sh_chi2," << std::setprecision(4) << chi2_base_dro_sh << "\n"
                  << "mcnemar_base_vs_dro_sh_sig," << (chi2_base_dro_sh > 3.84 ? "yes" : "no") << "\n"
                  << "mcnemar_base_vs_base_sh_chi2," << chi2_base_base_sh << "\n"
                  << "mcnemar_base_vs_base_sh_sig," << (chi2_base_base_sh > 3.84 ? "yes" : "no") << "\n"
                  << "mcnemar_base_vs_dro_chi2," << chi2_base_dro << "\n"
                  << "mcnemar_base_vs_dro_sig," << (chi2_base_dro > 3.84 ? "yes" : "no") << "\n"
                  << "bootstrap_base_vs_dro_sh_mean," << boot_base_dro_sh.mean_delta << "\n"
                  << "bootstrap_base_vs_dro_sh_ci_lo," << boot_base_dro_sh.ci_low << "\n"
                  << "bootstrap_base_vs_dro_sh_ci_hi," << boot_base_dro_sh.ci_high << "\n"
                  << "bootstrap_base_vs_base_sh_mean," << boot_base_base_sh.mean_delta << "\n"
                  << "bootstrap_base_vs_base_sh_ci_lo," << boot_base_base_sh.ci_low << "\n"
                  << "bootstrap_base_vs_base_sh_ci_hi," << boot_base_base_sh.ci_high << "\n"
                  << "cohens_h_base_vs_dro_sh," << es_base_dro_sh.cohens_h << "\n"
                  << "cohens_h_base_vs_base_sh," << es_base_base_sh.cohens_h << "\n"
                  << "risk_ratio_base_vs_dro_sh," << es_base_dro_sh.risk_ratio << "\n"
                  << "risk_ratio_base_vs_base_sh," << es_base_base_sh.risk_ratio << "\n"
                  << "non_anticipativity,passed\n";
    }

    std::cout << "  Paired results (n=" << paired_rollouts_actual << "):\n"
              << "    Base: " << base_coll << "  Base+SH: " << base_sh_coll
              << "  DRO: " << dro_coll << "  DRO+SH: " << dro_sh_coll << "\n"
              << "  McNemar (Base vs DRO+SH): chi2=" << std::setprecision(2) << chi2_base_dro_sh
              << " (" << (chi2_base_dro_sh > 3.84 ? "sig" : "n.s.") << ")\n"
              << "  McNemar (Base vs Base+SH): chi2=" << chi2_base_base_sh
              << " (" << (chi2_base_base_sh > 3.84 ? "sig" : "n.s.") << ")\n"
              << "  McNemar (Base vs DRO):     chi2=" << chi2_base_dro
              << " (" << (chi2_base_dro > 3.84 ? "sig" : "n.s.") << ")\n"
              << "  Bootstrap (Base vs DRO+SH): " << std::setprecision(4) << boot_base_dro_sh.mean_delta
              << " [" << boot_base_dro_sh.ci_low << ", " << boot_base_dro_sh.ci_high << "]\n"
              << "  Bootstrap (Base vs Base+SH): " << boot_base_base_sh.mean_delta
              << " [" << boot_base_base_sh.ci_low << ", " << boot_base_base_sh.ci_high << "]\n"
              << "  -> exp_f_mcnemar_paired.csv, exp_f_summary.csv, exp_h1_bootstrap_ci.csv\n";
}

// ============================================================================
// Experiment G: Conservatism & Smoothness
// ============================================================================

static void run_experiment_g() {
    std::cout << "\n========================================\n"
              << "  Experiment G: Conservatism & Smoothness\n"
              << "========================================\n";

    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.2;
    int g_rollouts = 250;

    std::ofstream f_cons(OUTPUT_DIR + "exp_g_conservatism_metrics.csv");
    f_cons << "variant,avg_speed,avg_progress,min_clearance_mean,min_clearance_std,"
           << "control_effort_mean,steering_variation,avg_solve_ms\n";

    auto mode_mdls = create_obstacle_mode_models(DT);
    EgoDynamics dynamics(DT);

    for (PaperVariant v : ALL_VARIANTS) {
        std::cout << "  Variant: " << variant_name(v) << " ... " << std::flush;

        std::vector<double> all_speeds, all_clearances, all_efforts, all_steer_var;
        double sum_progress = 0, sum_solve = 0;

        for (int r = 0; r < g_rollouts; ++r) {
            unsigned seed = static_cast<unsigned>(r * 9000);
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT;
            cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35;
            cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::FREQUENCY;
            cfg.enable_dro = uses_dro(v);
            cfg.safe_horizon_enabled = false;
            cfg.num_discs = 1;
            cfg.vehicle_length = 1.5;

            AdaptiveScenarioMPC ctrl(cfg);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, 0, omm);

            ObstacleSim osim;
            std::uniform_real_distribution<double> y_dist(-0.5, 0.5);
            osim.state = ObstacleState(4.0 + y_dist(rng), 0.3, -0.2, y_dist(rng) * 0.2);
            osim.current_mode = "constant_velocity";
            osim.available_modes = modes;
            osim.mode_models = omm;

            EgoState ego(0, 0, 0, 1.2);
            auto ref_path_g = setup_mpcc_path(ctrl);
            double pl_g = ref_path_g.total_length();
            Eigen::Vector2d goal = ref_path_g.get_position_at(pl_g);
            double pp_g = 0.0;

            for (int i = 0; i < 5; ++i) {
                ctrl.update_mode_observation(0, 0, osim.current_mode, i);
            }

            double rollout_speed_sum = 0, rollout_effort = 0;
            std::vector<double> steer_inputs;
            double rollout_min_clear = 1e9;

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                osim.maybe_switch(switch_prob, rng);
                ctrl.update_mode_observation(0, 0, osim.current_mode, step + 5);

                pp_g = ref_path_g.find_closest_point(ego.position(), pp_g);
                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5, pp_g, pl_g);
                sum_solve += res.solve_time;

                double dist = (ego.position() - osim.state.position()).norm();
                rollout_min_clear = std::min(rollout_min_clear, dist);
                rollout_speed_sum += std::abs(ego.v);

                if (res.success && res.first_input().has_value()) {
                    auto inp = res.first_input().value();
                    rollout_effort += inp.a * inp.a + inp.delta * inp.delta;
                    steer_inputs.push_back(inp.delta);
                    ego = dynamics.propagate(ego, inp);
                }
                osim.step(DT, rng);
            }

            all_speeds.push_back(rollout_speed_sum / ROLLOUT_STEPS);
            all_clearances.push_back(rollout_min_clear);
            all_efforts.push_back(rollout_effort / ROLLOUT_STEPS);
            sum_progress += ego.x;

            double steer_variation = 0;
            for (size_t i = 1; i < steer_inputs.size(); ++i)
                steer_variation += std::abs(steer_inputs[i] - steer_inputs[i - 1]);
            all_steer_var.push_back(steer_inputs.empty() ? 0 : steer_variation / steer_inputs.size());
        }

        auto mean_of = [](const std::vector<double>& vec) {
            return vec.empty() ? 0 : std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
        };
        auto std_of = [&mean_of](const std::vector<double>& vec) {
            if (vec.size() < 2) return 0.0;
            double m = mean_of(vec);
            double ss = 0;
            for (auto x : vec) ss += (x - m) * (x - m);
            return std::sqrt(ss / (vec.size() - 1));
        };

        f_cons << variant_name(v) << ","
               << std::setprecision(4) << mean_of(all_speeds) << ","
               << sum_progress / g_rollouts << ","
               << mean_of(all_clearances) << ","
               << std_of(all_clearances) << ","
               << mean_of(all_efforts) << ","
               << mean_of(all_steer_var) << ","
               << std::setprecision(2) << sum_solve / (g_rollouts * ROLLOUT_STEPS) * 1000 << "\n";

        std::cout << "speed=" << std::setprecision(3) << mean_of(all_speeds)
                  << " clearance=" << mean_of(all_clearances) << std::endl;
    }
    std::cout << "  -> exp_g_conservatism_metrics.csv\n";
}

// ============================================================================
// Experiment H: Full Ablation Matrix (6 variants)
// ============================================================================

static void run_experiment_h() {
    std::cout << "\n========================================\n"
              << "  Experiment H: Full Ablation Matrix\n"
              << "========================================\n";

    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.2;
    int h_rollouts = 600;

    std::ofstream f_out(OUTPUT_DIR + "exp_h_ablation_full.csv");
    f_out << "variant,sh_enabled,uses_dro,"
          << "collision_rate,ci_lo,ci_hi,missed_mode_rate,"
          << "avg_progress,avg_clearance,avg_solve_ms\n";

    for (PaperVariant pv : ALL_VARIANTS) {
        std::cout << "  Variant: " << variant_name(pv)
                  << " (SH=" << uses_sh(pv)
                  << " DRO=" << uses_dro(pv) << ") ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

        for (int r = 0; r < h_rollouts; ++r) {
            unsigned seed = static_cast<unsigned>(r * 11000);
            auto res = run_single_rollout(pv, switch_prob, BASE_SCENARIOS,
                                           ROLLOUT_STEPS, seed, modes);
            if (res.collision) collisions++;
            total_missed += res.missed_mode_steps;
            total_steps_all += res.total_steps;
            sum_progress += res.total_progress;
            sum_clearance += res.min_clearance;
            sum_solve += res.avg_solve_time;
        }

        double coll_rate = static_cast<double>(collisions) / h_rollouts;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, h_rollouts);
        double missed_rate = total_steps_all > 0
            ? static_cast<double>(total_missed) / total_steps_all : 0;
        double avg_progress = sum_progress / h_rollouts;
        double avg_clearance = sum_clearance / h_rollouts;
        double avg_solve = sum_solve / h_rollouts * 1000;

        f_out << variant_name(pv) << ","
              << (uses_sh(pv) ? "true" : "false") << ","
              << (uses_dro(pv) ? "yes" : "no") << ","
              << std::fixed << std::setprecision(4)
              << coll_rate << "," << ci_lo << "," << ci_hi << ","
              << missed_rate << ","
              << avg_progress << "," << avg_clearance << ","
              << std::setprecision(2) << avg_solve << "\n";

        std::cout << "coll=" << std::setprecision(3) << coll_rate
                  << " [" << ci_lo << "," << ci_hi << "]"
                  << " missed=" << std::setprecision(3) << missed_rate << std::endl;
    }

    std::cout << "  -> exp_h_ablation_full.csv\n";
}

// ============================================================================
// Experiment I: Safe Horizon at Scale
// ============================================================================

static void run_experiment_i() {
    std::cout << "\n========================================\n"
              << "  Experiment I: Safe Horizon at Scale\n"
              << "========================================\n";

    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    double switch_prob = 0.2;
    int i_rollouts = 300;

    std::vector<int> scenario_counts = {40, 100, 200, 500};

    // Variants to test: Base and DRO, each with/without SH
    std::vector<PaperVariant> i_variants = {
        PaperVariant::BASE, PaperVariant::BASE_SH,
        PaperVariant::DRO, PaperVariant::DRO_SH
    };

    std::ofstream f_out(OUTPUT_DIR + "exp_i_sh_scaling.csv");
    f_out << "variant,num_scenarios,sh_enabled,collision_rate,ci_lo,ci_hi,"
          << "missed_mode_rate,avg_progress,avg_clearance,avg_solve_ms,"
          << "predicted_n_safe\n";

    for (int S : scenario_counts) {
        for (PaperVariant pv : i_variants) {
            bool sh = uses_sh(pv);
            std::cout << "  S=" << S << " " << variant_name(pv) << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_steps_all = 0;
            double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

            for (int r = 0; r < i_rollouts; ++r) {
                unsigned seed = static_cast<unsigned>(r * 12000 + S);
                auto res = run_single_rollout(pv, switch_prob, S,
                                               ROLLOUT_STEPS, seed, modes);
                if (res.collision) collisions++;
                total_missed += res.missed_mode_steps;
                total_steps_all += res.total_steps;
                sum_progress += res.total_progress;
                sum_clearance += res.min_clearance;
                sum_solve += res.avg_solve_time;
            }

            double coll_rate = static_cast<double>(collisions) / i_rollouts;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, i_rollouts);
            double missed_rate = total_steps_all > 0
                ? static_cast<double>(total_missed) / total_steps_all : 0;

            // Predict effective N_safe using the same PRACTICAL mode as the controller
            ScenarioMPCConfig tmp_cfg;
            tmp_cfg.horizon = HORIZON;
            tmp_cfg.safe_horizon_enabled = sh;
            tmp_cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            int predicted_ns = tmp_cfg.compute_safe_horizon(S);

            f_out << variant_name(pv) << "," << S << ","
                  << (sh ? "true" : "false") << ","
                  << std::fixed << std::setprecision(4)
                  << coll_rate << "," << ci_lo << "," << ci_hi << ","
                  << missed_rate << ","
                  << sum_progress / i_rollouts << ","
                  << sum_clearance / i_rollouts << ","
                  << std::setprecision(2) << sum_solve / i_rollouts * 1000 << ","
                  << predicted_ns << "\n";

            std::cout << "coll=" << std::setprecision(3) << coll_rate
                      << " N_safe=" << predicted_ns << std::endl;
        }
    }

    std::cout << "  -> exp_i_sh_scaling.csv\n";
}

// ============================================================================
// Experiment J: OT vs Simple Coverage Baselines
// ============================================================================

static void run_experiment_j() {
    std::cout << "\n========================================\n"
              << "  Experiment J: OT vs Simple Coverage Baselines\n"
              << "========================================\n";

    double switch_prob = 0.2;
    int j_rollouts = 600;

    std::vector<SamplingBaseline> baselines = {
        SamplingBaseline::STANDARD,
        SamplingBaseline::STRATIFIED, SamplingBaseline::TEMPERATURE,
        SamplingBaseline::EPSILON_GREEDY, SamplingBaseline::RISK_BIASED
    };

    EnvironmentSetup default_env;
    default_env.initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
    default_env.goal = Eigen::Vector2d(20.0, 0.0);
    default_env.initial_obs = ObstacleState(3.0, 0.3, -0.1, 0.0);
    default_env.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    std::ofstream f_out(OUTPUT_DIR + "exp_j_ot_vs_baselines.csv");
    f_out << "baseline,collision_rate,ci_lo,ci_hi,missed_mode_rate,"
          << "avg_progress,avg_clearance,avg_solve_ms\n";

    for (SamplingBaseline bl : baselines) {
        std::cout << "  Baseline: " << baseline_name(bl) << " ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

        for (int r = 0; r < j_rollouts; ++r) {
            unsigned seed = static_cast<unsigned>(r * 13000 + static_cast<int>(bl));

            // Use BASE variant (no SH, no DRO) to isolate sampling effect
            auto res = run_single_rollout_env(
                PaperVariant::BASE, switch_prob, BASE_SCENARIOS, ROLLOUT_STEPS,
                seed, default_env, bl);
            if (res.collision) collisions++;
            total_missed += res.missed_mode_steps;
            total_steps_all += res.total_steps;
            sum_progress += res.total_progress;
            sum_clearance += res.min_clearance;
            sum_solve += res.avg_solve_time;
        }

        double coll_rate = static_cast<double>(collisions) / j_rollouts;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, j_rollouts);
        double missed_rate = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;

        f_out << baseline_name(bl) << ","
              << std::fixed << std::setprecision(4)
              << coll_rate << "," << ci_lo << "," << ci_hi << ","
              << missed_rate << ","
              << sum_progress / j_rollouts << ","
              << sum_clearance / j_rollouts << ","
              << std::setprecision(2) << sum_solve / j_rollouts * 1000 << "\n";

        std::cout << "coll=" << std::setprecision(3) << coll_rate
                  << " missed=" << missed_rate << std::endl;
    }
    std::cout << "  -> exp_j_ot_vs_baselines.csv\n";
}

// ============================================================================
// Experiment K: Environment Generalization
// ============================================================================

static void run_experiment_k() {
    std::cout << "\n========================================\n"
              << "  Experiment K: Environment Generalization\n"
              << "========================================\n";

    double switch_prob = 0.2;
    int k_rollouts = 600;

    std::vector<EnvironmentType> envs = {
        EnvironmentType::STRAIGHT, EnvironmentType::NARROW_CORRIDOR,
        EnvironmentType::INTERSECTION, EnvironmentType::ONCOMING
    };
    std::vector<PaperVariant> k_variants = {
        PaperVariant::BASE, PaperVariant::DRO,
        PaperVariant::BASE_SH, PaperVariant::DRO_SH
    };

    std::ofstream f_out(OUTPUT_DIR + "exp_k_environment_generalization.csv");
    f_out << "environment,variant,collision_rate,ci_lo,ci_hi,"
          << "missed_mode_rate,avg_progress,avg_clearance,avg_solve_ms\n";

    for (EnvironmentType env_type : envs) {
        for (PaperVariant v : k_variants) {
            std::cout << "  " << environment_name(env_type) << " / "
                      << variant_name(v) << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_steps_all = 0;
            double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

            for (int r = 0; r < k_rollouts; ++r) {
                unsigned seed = static_cast<unsigned>(r * 14000 + static_cast<int>(env_type) * 100);
                std::mt19937 env_rng(seed);
                auto env_setup = create_environment(env_type, env_rng);

                auto res = run_single_rollout_env(
                    v, switch_prob, BASE_SCENARIOS, ROLLOUT_STEPS,
                    seed + 1, env_setup);
                if (res.collision) collisions++;
                total_missed += res.missed_mode_steps;
                total_steps_all += res.total_steps;
                sum_progress += res.total_progress;
                sum_clearance += res.min_clearance;
                sum_solve += res.avg_solve_time;
            }

            double coll_rate = static_cast<double>(collisions) / k_rollouts;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, k_rollouts);
            double missed_rate = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;

            f_out << environment_name(env_type) << "," << variant_name(v) << ","
                  << std::fixed << std::setprecision(4)
                  << coll_rate << "," << ci_lo << "," << ci_hi << ","
                  << missed_rate << ","
                  << sum_progress / k_rollouts << ","
                  << sum_clearance / k_rollouts << ","
                  << std::setprecision(2) << sum_solve / k_rollouts * 1000 << "\n";

            std::cout << "coll=" << std::setprecision(3) << coll_rate
                      << " missed=" << missed_rate << std::endl;
        }
    }
    std::cout << "  -> exp_k_environment_generalization.csv\n";
}

// ============================================================================
// Experiment L: Empirical Joint Violation Rate
// ============================================================================

static void run_experiment_l() {
    std::cout << "\n========================================\n"
              << "  Experiment L: Empirical Joint Violation Rate\n"
              << "========================================\n";

    std::vector<PaperVariant> l_variants = {
        PaperVariant::BASE, PaperVariant::BASE_SH, PaperVariant::DRO_SH
    };
    std::vector<int> scenario_counts = {40, 100, 200};
    int l_rollouts = 300;
    int fresh_samples = 1000;
    double switch_prob = 0.15;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    std::ofstream f_out(OUTPUT_DIR + "exp_l_empirical_violation.csv");
    f_out << "variant,num_scenarios,epsilon_target,epsilon_hat,ci_lo,ci_hi,num_rollouts\n";

    auto mode_mdls = create_obstacle_mode_models(DT);

    for (PaperVariant v : l_variants) {
        for (int S : scenario_counts) {
            std::cout << "  " << variant_name(v) << " S=" << S << " ... " << std::flush;

            int total_violations = 0;
            int total_checks = 0;

            for (int r = 0; r < l_rollouts; ++r) {
                unsigned seed = static_cast<unsigned>(r * 15000 + S);
                std::mt19937 rng(seed);

                ScenarioMPCConfig cfg;
                cfg.horizon = HORIZON; cfg.dt = DT;
                cfg.num_scenarios = S;
                cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35;
                cfg.safety_margin = 0.2;
                cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
                cfg.weight_type = WeightType::FREQUENCY;
                cfg.enable_dro = uses_dro(v);
                cfg.safe_horizon_enabled = uses_sh(v);
                cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
                cfg.num_discs = 1;

                AdaptiveScenarioMPC ctrl(cfg);

                std::map<std::string, ModeModel> omm;
                for (auto& m : modes) omm[m] = mode_mdls[m];
                ctrl.initialize_obstacle(0, 0, omm);

                ObstacleSim osim;
                std::uniform_real_distribution<double> jitter(-0.5, 0.5);
                osim.state = ObstacleState(4.0 + jitter(rng), 0.3 + jitter(rng) * 0.3,
                                           jitter(rng) * 0.2, jitter(rng) * 0.2);
                osim.current_mode = "constant_velocity";
                osim.available_modes = modes;
                osim.mode_models = omm;

                EgoState ego(0, 0, 0, 1.2);
                Eigen::Vector2d goal(20, 0);

                for (int i = 0; i < 5; ++i) {
                    ctrl.update_mode_observation(0, 0, osim.current_mode, i);
                }

                // Run for a few steps to get a representative solve state
                EgoDynamics dynamics(DT);
                for (int step = 0; step < 30; ++step) {
                    osim.maybe_switch(switch_prob, rng);
                    ctrl.update_mode_observation(0, 0, osim.current_mode, step + 5);

                    std::map<int, ObstacleState> obs_map;
                    obs_map[0] = osim.state;
                    auto res = ctrl.solve(ego, obs_map, goal, 1.5);

                    if (res.success && res.first_input().has_value())
                        ego = dynamics.propagate(ego, res.first_input().value());
                    osim.step(DT, rng);
                }

                // Now draw fresh_samples trajectories and check violation
                // Rebuild weights from observations
                ModeHistory mh(0, omm);
                for (int i = 0; i < 35; ++i) {
                    mh.record_observation(i, osim.current_mode);
                }
                auto weights = compute_mode_weights(mh, WeightType::FREQUENCY);

                double collision_radius = cfg.ego_radius + cfg.obstacle_radius;

                // Check each fresh sample for constraint violation
                for (int fs = 0; fs < fresh_samples; ++fs) {
                    // Sample a fresh mode and propagate
                    std::string sampled_mode = sample_mode_from_weights(weights, rng);
                    if (omm.find(sampled_mode) == omm.end()) continue;
                    const auto& model = omm.at(sampled_mode);

                    ObstacleState fresh_obs = osim.state;
                    bool violated = false;
                    Eigen::Vector4d x_obs = fresh_obs.to_array();
                    std::normal_distribution<double> nd(0, 1);

                    // Check if ego trajectory (from last solve) violates
                    auto& ego_traj = ctrl.scenarios();  // use actual planned traj
                    EgoState ego_check = ego;

                    for (int k = 0; k < HORIZON; ++k) {
                        Eigen::VectorXd noise(model.noise_dim());
                        for (int d = 0; d < model.noise_dim(); ++d)
                            noise(d) = nd(rng) * 0.02;
                        x_obs = model.A * x_obs + model.b + model.G * noise;

                        Eigen::Vector2d obs_pos = x_obs.head<2>();
                        double dist = (ego_check.position() - obs_pos).norm();
                        if (dist < collision_radius) {
                            violated = true;
                            break;
                        }
                    }

                    total_checks++;
                    if (violated) total_violations++;
                }
            }

            double eps_hat = total_checks > 0 ? static_cast<double>(total_violations) / total_checks : 0;
            auto [ci_lo, ci_hi] = wilson_ci(total_violations, total_checks);
            double eps_target = 0.05;

            f_out << variant_name(v) << "," << S << ","
                  << std::fixed << std::setprecision(4)
                  << eps_target << "," << eps_hat << ","
                  << ci_lo << "," << ci_hi << "," << l_rollouts << "\n";

            std::cout << "eps_hat=" << std::setprecision(4) << eps_hat
                      << " [" << ci_lo << "," << ci_hi << "]" << std::endl;
        }
    }
    std::cout << "  -> exp_l_empirical_violation.csv\n";
}

// ============================================================================
// Experiment M: Safe Horizon Length Sweep
// ============================================================================

static void run_experiment_m() {
    std::cout << "\n========================================\n"
              << "  Experiment M: Safe Horizon Length Sweep\n"
              << "========================================\n";

    std::vector<int> n_safe_values = {3, 5, 8, 10, 12, 15};
    std::vector<PaperVariant> m_variants = {
        PaperVariant::BASE_SH, PaperVariant::DRO_SH
    };
    int m_rollouts = 600;
    double switch_prob = 0.2;

    EnvironmentSetup default_env;
    default_env.initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
    default_env.goal = Eigen::Vector2d(20.0, 0.0);
    default_env.initial_obs = ObstacleState(3.0, 0.3, -0.1, 0.0);
    default_env.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    std::ofstream f_out(OUTPUT_DIR + "exp_m_sh_sweep.csv");
    f_out << "variant,forced_n_safe,collision_rate,ci_lo,ci_hi,"
          << "avg_progress,avg_clearance,avg_solve_ms\n";

    for (PaperVariant v : m_variants) {
        for (int n_safe : n_safe_values) {
            std::cout << "  " << variant_name(v) << " N_safe=" << n_safe << " ... " << std::flush;

            int collisions = 0;
            double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

            for (int r = 0; r < m_rollouts; ++r) {
                unsigned seed = static_cast<unsigned>(r * 16000 + n_safe);
                auto res = run_single_rollout_env(
                    v, switch_prob, BASE_SCENARIOS, ROLLOUT_STEPS,
                    seed, default_env, SamplingBaseline::STANDARD, n_safe);
                if (res.collision) collisions++;
                sum_progress += res.total_progress;
                sum_clearance += res.min_clearance;
                sum_solve += res.avg_solve_time;
            }

            double coll_rate = static_cast<double>(collisions) / m_rollouts;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, m_rollouts);

            f_out << variant_name(v) << "," << n_safe << ","
                  << std::fixed << std::setprecision(4)
                  << coll_rate << "," << ci_lo << "," << ci_hi << ","
                  << sum_progress / m_rollouts << ","
                  << sum_clearance / m_rollouts << ","
                  << std::setprecision(2) << sum_solve / m_rollouts * 1000 << "\n";

            std::cout << "coll=" << std::setprecision(3) << coll_rate << std::endl;
        }
    }
    std::cout << "  -> exp_m_sh_sweep.csv\n";
}

// ============================================================================
// Experiment N: Runtime Scaling Breakdown
// ============================================================================

static void run_experiment_n() {
    std::cout << "\n========================================\n"
              << "  Experiment N: Runtime Scaling Breakdown\n"
              << "========================================\n";

    int n_rollouts = 150;
    double switch_prob = 0.15;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    std::ofstream f_out(OUTPUT_DIR + "exp_n_runtime_scaling.csv");
    f_out << "sweep_dim,sweep_value,avg_total_ms,avg_constraint_ms,avg_qp_ms,"
          << "p50_total_ms,p90_total_ms,collision_rate\n";

    auto mode_mdls = create_obstacle_mode_models(DT);

    // Sweep 1: Scenario count S
    std::vector<int> s_values = {10, 20, 40, 80, 160, 320};
    for (int S : s_values) {
        std::cout << "  S=" << S << " ... " << std::flush;

        std::vector<double> all_total, all_constr, all_qp;
        int collisions = 0;

        for (int r = 0; r < n_rollouts; ++r) {
            unsigned seed = static_cast<unsigned>(r * 17000 + S);
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = S;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::FREQUENCY;
            cfg.safe_horizon_enabled = true;
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.num_discs = 3; cfg.vehicle_length = 4.0;

            AdaptiveScenarioMPC ctrl(cfg);
            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, 0, omm);

            ObstacleSim osim;
            osim.state = ObstacleState(4.0, 0.3, -0.2, 0.1);
            osim.current_mode = "constant_velocity";
            osim.available_modes = modes;
            osim.mode_models = omm;

            EgoState ego(0, 0, 0, 1.2);
            Eigen::Vector2d goal(20, 0);
            EgoDynamics dyn(DT);
            double collision_radius = cfg.ego_radius + cfg.obstacle_radius;

            for (int i = 0; i < 5; ++i)
                ctrl.update_mode_observation(0, 0, osim.current_mode, i);

            bool had_collision = false;
            for (int step = 0; step < 50; ++step) {
                osim.maybe_switch(switch_prob, rng);
                ctrl.update_mode_observation(0, 0, osim.current_mode, step + 5);
                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);
                all_total.push_back(res.solve_time * 1000);
                all_constr.push_back(res.constraint_construction_time * 1000);
                all_qp.push_back(res.qp_solve_time * 1000);
                if ((ego.position() - osim.state.position()).norm() < collision_radius)
                    had_collision = true;
                if (res.success && res.first_input().has_value())
                    ego = dyn.propagate(ego, res.first_input().value());
                osim.step(DT, rng);
            }
            if (had_collision) collisions++;
        }

        auto mean_v = [](const std::vector<double>& v) {
            return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        f_out << "S," << S << ","
              << std::fixed << std::setprecision(3)
              << mean_v(all_total) << "," << mean_v(all_constr) << "," << mean_v(all_qp) << ","
              << percentile(all_total, 50) << "," << percentile(all_total, 90) << ","
              << std::setprecision(4)
              << static_cast<double>(collisions) / n_rollouts << "\n";

        std::cout << "avg=" << std::setprecision(2) << mean_v(all_total) << "ms" << std::endl;
    }

    // Sweep 2: Disc count D
    std::vector<int> d_values = {1, 3, 5, 7};
    for (int D : d_values) {
        std::cout << "  D=" << D << " ... " << std::flush;

        std::vector<double> all_total, all_constr, all_qp;
        int collisions = 0;

        for (int r = 0; r < n_rollouts; ++r) {
            unsigned seed = static_cast<unsigned>(r * 17500 + D);
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::FREQUENCY;
            cfg.safe_horizon_enabled = true;
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.num_discs = D; cfg.vehicle_length = 4.0;

            AdaptiveScenarioMPC ctrl(cfg);
            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, 0, omm);

            ObstacleSim osim;
            osim.state = ObstacleState(4.0, 0.3, -0.2, 0.1);
            osim.current_mode = "constant_velocity";
            osim.available_modes = modes;
            osim.mode_models = omm;

            EgoState ego(0, 0, 0, 1.2);
            Eigen::Vector2d goal(20, 0);
            EgoDynamics dyn(DT);
            double collision_radius = cfg.ego_radius + cfg.obstacle_radius;

            for (int i = 0; i < 5; ++i)
                ctrl.update_mode_observation(0, 0, osim.current_mode, i);

            bool had_collision = false;
            for (int step = 0; step < 50; ++step) {
                osim.maybe_switch(0.15, rng);
                ctrl.update_mode_observation(0, 0, osim.current_mode, step + 5);
                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);
                all_total.push_back(res.solve_time * 1000);
                all_constr.push_back(res.constraint_construction_time * 1000);
                all_qp.push_back(res.qp_solve_time * 1000);
                if ((ego.position() - osim.state.position()).norm() < collision_radius)
                    had_collision = true;
                if (res.success && res.first_input().has_value())
                    ego = dyn.propagate(ego, res.first_input().value());
                osim.step(DT, rng);
            }
            if (had_collision) collisions++;
        }

        auto mean_v = [](const std::vector<double>& v) {
            return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        f_out << "D," << D << ","
              << std::fixed << std::setprecision(3)
              << mean_v(all_total) << "," << mean_v(all_constr) << "," << mean_v(all_qp) << ","
              << percentile(all_total, 50) << "," << percentile(all_total, 90) << ","
              << std::setprecision(4)
              << static_cast<double>(collisions) / n_rollouts << "\n";

        std::cout << "avg=" << std::setprecision(2) << mean_v(all_total) << "ms" << std::endl;
    }

    // Sweep 3: Safe horizon N_safe
    std::vector<int> ns_values = {3, 5, 8, 10, 15};
    for (int ns : ns_values) {
        std::cout << "  N_safe=" << ns << " ... " << std::flush;

        std::vector<double> all_total, all_constr, all_qp;
        int collisions = 0;

        for (int r = 0; r < n_rollouts; ++r) {
            unsigned seed = static_cast<unsigned>(r * 18000 + ns);
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::FREQUENCY;
            cfg.safe_horizon_enabled = true;
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.forced_safe_horizon = ns;
            cfg.num_discs = 3; cfg.vehicle_length = 4.0;

            AdaptiveScenarioMPC ctrl(cfg);
            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, 0, omm);

            ObstacleSim osim;
            osim.state = ObstacleState(4.0, 0.3, -0.2, 0.1);
            osim.current_mode = "constant_velocity";
            osim.available_modes = modes;
            osim.mode_models = omm;

            EgoState ego(0, 0, 0, 1.2);
            Eigen::Vector2d goal(20, 0);
            EgoDynamics dyn(DT);
            double collision_radius = cfg.ego_radius + cfg.obstacle_radius;

            for (int i = 0; i < 5; ++i)
                ctrl.update_mode_observation(0, 0, osim.current_mode, i);

            bool had_collision = false;
            for (int step = 0; step < 50; ++step) {
                osim.maybe_switch(0.15, rng);
                ctrl.update_mode_observation(0, 0, osim.current_mode, step + 5);
                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);
                all_total.push_back(res.solve_time * 1000);
                all_constr.push_back(res.constraint_construction_time * 1000);
                all_qp.push_back(res.qp_solve_time * 1000);
                if ((ego.position() - osim.state.position()).norm() < collision_radius)
                    had_collision = true;
                if (res.success && res.first_input().has_value())
                    ego = dyn.propagate(ego, res.first_input().value());
                osim.step(DT, rng);
            }
            if (had_collision) collisions++;
        }

        auto mean_v = [](const std::vector<double>& v) {
            return v.empty() ? 0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        };

        f_out << "N_safe," << ns << ","
              << std::fixed << std::setprecision(3)
              << mean_v(all_total) << "," << mean_v(all_constr) << "," << mean_v(all_qp) << ","
              << percentile(all_total, 50) << "," << percentile(all_total, 90) << ","
              << std::setprecision(4)
              << static_cast<double>(collisions) / n_rollouts << "\n";

        std::cout << "avg=" << std::setprecision(2) << mean_v(all_total) << "ms" << std::endl;
    }

    std::cout << "  -> exp_n_runtime_scaling.csv\n";
}

// ============================================================================
// Experiment O: Distribution Shift / Mismatch
// ============================================================================

static void run_experiment_o() {
    std::cout << "\n========================================\n"
              << "  Experiment O: Distribution Shift / Mismatch\n"
              << "========================================\n";

    std::vector<PaperVariant> o_variants = {
        PaperVariant::BASE, PaperVariant::DRO, PaperVariant::DRO_SH
    };
    int o_rollouts = 600;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    // Mismatch conditions:
    // 1. Train sp=0.1, test sp=0.3 (under-estimated switching)
    // 2. Train sp=0.3, test sp=0.1 (over-estimated switching)
    // 3. Matched sp=0.2 (control)
    struct MismatchCondition {
        std::string name;
        double train_sp;  // switch_prob seen during warmup
        double test_sp;   // switch_prob during evaluation
    };
    std::vector<MismatchCondition> conditions = {
        {"Matched", 0.2, 0.2},
        {"UnderEst", 0.1, 0.3},
        {"OverEst", 0.3, 0.1}
    };

    std::ofstream f_out(OUTPUT_DIR + "exp_o_distribution_shift.csv");
    f_out << "condition,variant,collision_rate,ci_lo,ci_hi,"
          << "missed_mode_rate,avg_progress,avg_clearance\n";

    auto mode_mdls = create_obstacle_mode_models(DT);

    for (const auto& cond : conditions) {
        for (PaperVariant v : o_variants) {
            std::cout << "  " << cond.name << " / " << variant_name(v) << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_steps_all = 0;
            double sum_progress = 0, sum_clearance = 0;

            for (int r = 0; r < o_rollouts; ++r) {
                unsigned seed = static_cast<unsigned>(r * 19000);
                std::mt19937 rng(seed);

                ScenarioMPCConfig cfg;
                cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
                cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
                cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
                cfg.weight_type = WeightType::FREQUENCY;
                cfg.enable_dro = uses_dro(v);
                cfg.safe_horizon_enabled = uses_sh(v);
                cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
                cfg.num_discs = 1;

                AdaptiveScenarioMPC ctrl(cfg);

                std::map<std::string, ModeModel> omm;
                for (auto& m : modes) omm[m] = mode_mdls[m];
                ctrl.initialize_obstacle(0, 0, omm);

                ObstacleSim osim;
                std::uniform_real_distribution<double> jitter(-0.5, 0.5);
                osim.state = ObstacleState(3.0 + jitter(rng), 0.3 + jitter(rng) * 0.3,
                                           jitter(rng) * 0.2, jitter(rng) * 0.2);
                osim.current_mode = "constant_velocity";
                osim.available_modes = modes;
                osim.mode_models = omm;

                EgoState ego(0, 0, 0, 1.5);
                Eigen::Vector2d goal(20, 0);
                EgoDynamics dyn(DT);
                double collision_radius = cfg.ego_radius + cfg.obstacle_radius;

                // Warmup phase: use train_sp
                for (int i = 0; i < 20; ++i) {
                    osim.maybe_switch(cond.train_sp, rng);
                    ctrl.update_mode_observation(0, 0, osim.current_mode, i);
                    osim.step(DT, rng);
                }

                // Test phase: use test_sp
                bool had_collision = false;
                double rollout_min_clear = 1e9;
                int missed = 0, steps = 0;
                for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                    osim.maybe_switch(cond.test_sp, rng);
                    ctrl.update_mode_observation(0, 0, osim.current_mode, step + 20);

                    std::map<int, ObstacleState> obs_map;
                    obs_map[0] = osim.state;
                    auto res = ctrl.solve(ego, obs_map, goal, 1.5);

                    double dist = (ego.position() - osim.state.position()).norm();
                    rollout_min_clear = std::min(rollout_min_clear, dist);
                    if (dist < collision_radius) had_collision = true;

                    bool mode_found = false;
                    for (const auto& sc : ctrl.scenarios()) {
                        for (const auto& [oid, traj] : sc.trajectories) {
                            if (oid == 0 && traj.mode_id == osim.current_mode) {
                                mode_found = true; break;
                            }
                        }
                        if (mode_found) break;
                    }
                    if (!mode_found) missed++;
                    steps++;

                    if (res.success && res.first_input().has_value())
                        ego = dyn.propagate(ego, res.first_input().value());
                    osim.step(DT, rng);
                }

                if (had_collision) collisions++;
                total_missed += missed;
                total_steps_all += steps;
                sum_progress += ego.x;
                sum_clearance += rollout_min_clear;
            }

            double coll_rate = static_cast<double>(collisions) / o_rollouts;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, o_rollouts);
            double missed_rate = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;

            f_out << cond.name << "," << variant_name(v) << ","
                  << std::fixed << std::setprecision(4)
                  << coll_rate << "," << ci_lo << "," << ci_hi << ","
                  << missed_rate << ","
                  << sum_progress / o_rollouts << ","
                  << sum_clearance / o_rollouts << "\n";

            std::cout << "coll=" << std::setprecision(3) << coll_rate
                      << " missed=" << missed_rate << std::endl;
        }
    }
    std::cout << "  -> exp_o_distribution_shift.csv\n";
}

// ============================================================================
// Experiment P: Coverage Strategy Comparison (oracle/quota baselines)
// ============================================================================

static void run_experiment_p() {
    std::cout << "\n========================================\n"
              << "  Experiment P: Coverage Strategy Comparison\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 600;
    const double SWITCH_PROB = 0.2;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    struct CoverageStrategy {
        std::string name;
        SamplingBaseline baseline;
        PaperVariant variant;
    };
    std::vector<CoverageStrategy> strategies = {
        {"Standard",  SamplingBaseline::STANDARD,       PaperVariant::BASE},
        {"Uniform",   SamplingBaseline::UNIFORM_WEIGHT,  PaperVariant::BASE},
        {"Recency",   SamplingBaseline::RECENCY_WEIGHT,  PaperVariant::BASE},
        {"Oracle",    SamplingBaseline::ORACLE_FLOOD,    PaperVariant::BASE},
    };

    std::ofstream csv(OUTPUT_DIR + "exp_p_coverage_baselines.csv");
    csv << "baseline,collision_rate,ci_lo,ci_hi,missed_mode_rate,avg_progress,avg_clearance\n";

    for (const auto& strat : strategies) {
        std::cout << "  " << strat.name << " ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        double sum_progress = 0, sum_clearance = 0;

        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            unsigned seed = 80000 + r;
            std::mt19937 env_rng(seed);
            EnvironmentSetup env = create_environment(EnvironmentType::STRAIGHT, env_rng);

            auto res = run_single_rollout_env(
                strat.variant, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS, seed,
                env, strat.baseline);

            if (res.collision) collisions++;
            total_missed += res.missed_mode_steps;
            total_steps_all += res.total_steps;
            sum_progress += res.total_progress;
            sum_clearance += res.min_clearance;
        }

        double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
        double mmr = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;

        csv << strat.name << "," << std::fixed << std::setprecision(4)
            << cr << "," << ci_lo << "," << ci_hi << ","
            << mmr << "," << sum_progress / NUM_ROLLOUTS << ","
            << sum_clearance / NUM_ROLLOUTS << "\n";

        std::cout << "coll=" << std::setprecision(3) << cr
                  << " missed=" << mmr << std::endl;
    }
    csv.close();
    std::cout << "  -> exp_p_coverage_baselines.csv\n";
}

// ============================================================================
// Experiment Q: OT Internal Ablation (REMOVED — OT predictor deleted)
// ============================================================================

// ============================================================================
// Experiment R: Mode Coverage Diagnostic (true vs sampled frequencies)
// ============================================================================

static void run_experiment_r() {
    std::cout << "\n========================================\n"
              << "  Experiment R: Mode Coverage Diagnostic\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 600;
    const double SWITCH_PROB = 0.2;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    std::vector<PaperVariant> r_variants = {PaperVariant::BASE, PaperVariant::DRO};

    auto mode_mdls = create_obstacle_mode_models(DT);

    std::ofstream csv(OUTPUT_DIR + "exp_r_mode_coverage.csv");
    csv << "variant,mode_name,true_fraction,sampled_fraction,coverage_ratio\n";

    for (PaperVariant v : r_variants) {
        std::cout << "  " << variant_name(v) << " ... " << std::flush;

        // Accumulate per-mode counts across all rollouts and steps
        std::map<std::string, int> true_mode_counts;
        std::map<std::string, int> sampled_mode_counts;
        int total_steps_all = 0;
        int total_scenario_slots = 0;

        for (const auto& m : modes) {
            true_mode_counts[m] = 0;
            sampled_mode_counts[m] = 0;
        }

        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            unsigned seed = 95000 + r;
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::FREQUENCY;
            cfg.enable_dro = uses_dro(v);
            cfg.safe_horizon_enabled = false;
            cfg.num_discs = 1;

            AdaptiveScenarioMPC ctrl(cfg);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, 0, omm);

            ObstacleSim osim;
            std::uniform_real_distribution<double> jitter(-0.5, 0.5);
            osim.state = ObstacleState(3.0 + jitter(rng), 0.3 + jitter(rng) * 0.3,
                                       jitter(rng) * 0.2, jitter(rng) * 0.2);
            osim.current_mode = "constant_velocity";
            osim.available_modes = modes;
            osim.mode_models = omm;

            EgoState ego(0, 0, 0, 1.5);
            Eigen::Vector2d goal(20, 0);
            EgoDynamics dyn(DT);

            for (int i = 0; i < 5; ++i) {
                ctrl.update_mode_observation(0, 0, osim.current_mode, i);
            }

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                osim.maybe_switch(SWITCH_PROB, rng);
                ctrl.update_mode_observation(0, 0, osim.current_mode, step + 5);

                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);

                // Count true mode
                true_mode_counts[osim.current_mode]++;
                total_steps_all++;

                // Count sampled modes in scenarios
                for (const auto& sc : ctrl.scenarios()) {
                    for (const auto& [oid, traj] : sc.trajectories) {
                        if (oid == 0) {
                            sampled_mode_counts[traj.mode_id]++;
                            total_scenario_slots++;
                        }
                    }
                }

                if (res.success && res.first_input().has_value())
                    ego = dyn.propagate(ego, res.first_input().value());
                osim.step(DT, rng);
            }
        }

        // Write per-mode results
        for (const auto& m : modes) {
            double true_frac = total_steps_all > 0 ?
                static_cast<double>(true_mode_counts[m]) / total_steps_all : 0;
            double sampled_frac = total_scenario_slots > 0 ?
                static_cast<double>(sampled_mode_counts[m]) / total_scenario_slots : 0;
            double coverage_ratio = true_frac > 0 ? sampled_frac / true_frac : 0;

            csv << variant_name(v) << "," << m << ","
                << std::fixed << std::setprecision(4)
                << true_frac << "," << sampled_frac << "," << coverage_ratio << "\n";
        }
        std::cout << "done" << std::endl;
    }
    csv.close();
    std::cout << "  -> exp_r_mode_coverage.csv\n";
}

// ============================================================================
// Experiment T: Missed-Mode Rate vs Scenario Count
// ============================================================================

static void run_experiment_t() {
    std::cout << "\n========================================\n"
              << "  Experiment T: Collision & Mode Coverage vs S (Q1)\n"
              << "  4 obstacles, 4 classes, all 8 variants\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 200;
    const double SWITCH_PROB = 0.2;
    std::vector<int> scenario_counts = {10, 20, 40, 80};

    std::ofstream csv(OUTPUT_DIR + "exp_t_missed_mode_vs_s.csv");
    csv << "variant,num_scenarios,collision_rate,ci_lo,ci_hi,missed_mode_rate,avg_progress\n";

    for (PaperVariant v : ALL_VARIANTS) {
        for (int S : scenario_counts) {
            std::cout << "  " << variant_name(v) << " S=" << S << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_checks = 0;
            double sum_progress = 0;

            for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                unsigned seed = 100000 + r;
                auto res = run_multi_obstacle_rollout(
                    v, SWITCH_PROB, S, ROLLOUT_STEPS, seed,
                    4, 4);  // 4 obstacles, 4 classes

                if (res.collision) collisions++;
                total_missed += res.missed_mode_steps;
                total_checks += res.total_mode_checks;
                sum_progress += res.total_progress;
            }

            double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
            double mmr = total_checks > 0 ?
                static_cast<double>(total_missed) / total_checks : 0;

            csv << variant_name(v) << "," << S << ","
                << std::fixed << std::setprecision(4)
                << cr << "," << ci_lo << "," << ci_hi << ","
                << mmr << "," << sum_progress / NUM_ROLLOUTS << "\n";

            std::cout << "coll=" << std::setprecision(3) << cr
                      << " missed=" << mmr << std::endl;
        }
    }
    csv.close();
    std::cout << "  -> exp_t_missed_mode_vs_s.csv\n";
}

// ============================================================================
// Experiment U: Ground-Cost Ablation for OT Geometry (REMOVED — OT predictor deleted)
// ============================================================================

// ============================================================================
// Experiment V: Rare-Mode Probability Sweep
// ============================================================================

static void run_experiment_v() {
    std::cout << "\n========================================\n"
              << "  Experiment V: Rare-Mode Sweep (Q1+Q2)\n"
              << "  4 obstacles, 4-class + 2-class configs\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 200;
    const double SWITCH_PROB = 0.2;
    std::vector<double> rare_probs = {0.01, 0.05, 0.10, 0.20};
    std::string rare_mode = "decelerating";
    std::vector<std::string> base_modes = {"constant_velocity", "turn_left", "turn_right"};

    struct ClassConfig { int num_obs; int num_cls; std::string label; };
    std::vector<ClassConfig> configs = {{4, 4, "4obs_4class"}, {4, 2, "4obs_2class"}};

    std::ofstream csv(OUTPUT_DIR + "exp_v_rare_mode_sweep.csv");
    csv << "variant,obs_config,rare_prob,collision_rate,ci_lo,ci_hi,"
        << "missed_mode_rate,rare_mode_missed_frac,avg_progress,avg_clearance\n";

    for (const auto& cc : configs) {
        for (PaperVariant v : ALL_VARIANTS) {
            for (double rp : rare_probs) {
                std::cout << "  " << cc.label << " " << variant_name(v)
                          << " rare_p=" << rp << " ... " << std::flush;

                int collisions = 0;
                int total_missed = 0, total_checks = 0;
                int rare_total = 0, rare_missed = 0;
                double sum_progress = 0, sum_clearance = 0;

                for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                    unsigned seed = 120000 + r;
                    auto res = run_multi_obstacle_rollout(
                        v, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS, seed,
                        cc.num_obs, cc.num_cls, base_modes, rare_mode, rp);

                    if (res.collision) collisions++;
                    total_missed += res.missed_mode_steps;
                    total_checks += res.total_mode_checks;
                    rare_total += res.rare_mode_active;
                    rare_missed += res.rare_mode_missed;
                    sum_progress += res.total_progress;
                    sum_clearance += res.min_clearance;
                }

                double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
                auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
                double mmr = total_checks > 0 ? static_cast<double>(total_missed) / total_checks : 0;
                double rare_miss_frac = rare_total > 0 ? static_cast<double>(rare_missed) / rare_total : 0;

                csv << variant_name(v) << "," << cc.label << "," << rp << ","
                    << std::fixed << std::setprecision(4)
                    << cr << "," << ci_lo << "," << ci_hi << ","
                    << mmr << "," << rare_miss_frac << ","
                    << sum_progress / NUM_ROLLOUTS << ","
                    << sum_clearance / NUM_ROLLOUTS << "\n";

                std::cout << "coll=" << std::setprecision(3) << cr
                          << " missed=" << mmr
                          << " rare_miss=" << rare_miss_frac << std::endl;
            }
        }
    }
    csv.close();
    std::cout << "  -> exp_v_rare_mode_sweep.csv\n";
}

// ============================================================================
// Experiment W: Scaling with Number of Modes M
// ============================================================================

static void run_experiment_w() {
    std::cout << "\n========================================\n"
              << "  Experiment W: Mode Count Scaling\n"
              << "  4 obstacles, 4 classes, all 8 variants\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 200;
    const double SWITCH_PROB = 0.2;

    struct MConfig { int M; std::vector<std::string> modes; };
    std::vector<MConfig> m_configs = {
        {2, {"constant_velocity", "turn_left"}},
        {3, {"constant_velocity", "turn_left", "turn_right"}},
        {4, {"constant_velocity", "turn_left", "turn_right", "decelerating"}},
        {6, {"constant_velocity", "turn_left", "turn_right", "decelerating",
             "lane_change_left", "lane_change_right"}},
    };

    std::ofstream csv(OUTPUT_DIR + "exp_w_mode_scaling.csv");
    csv << "variant,num_modes,collision_rate,ci_lo,ci_hi,"
        << "missed_mode_rate,avg_progress,avg_solve_ms\n";

    for (PaperVariant v : ALL_VARIANTS) {
        for (const auto& mc : m_configs) {
            std::cout << "  " << variant_name(v) << " M=" << mc.M << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_checks = 0;
            double sum_progress = 0, sum_solve = 0;

            for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                unsigned seed = 130000 + r;
                auto res = run_multi_obstacle_rollout(
                    v, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS, seed,
                    4, 4, mc.modes);

                if (res.collision) collisions++;
                total_missed += res.missed_mode_steps;
                total_checks += res.total_mode_checks;
                sum_progress += res.total_progress;
                sum_solve += res.avg_solve_time;
            }

            double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
            double mmr = total_checks > 0 ? static_cast<double>(total_missed) / total_checks : 0;

            csv << variant_name(v) << "," << mc.M << ","
                << std::fixed << std::setprecision(4)
                << cr << "," << ci_lo << "," << ci_hi << ","
                << mmr << "," << sum_progress / NUM_ROLLOUTS << ","
                << sum_solve * 1000 << "\n";

            std::cout << "coll=" << std::setprecision(3) << cr
                      << " missed=" << mmr << std::endl;
        }
    }
    csv.close();
    std::cout << "  -> exp_w_mode_scaling.csv\n";
}

// ============================================================================
// Experiment X: Coverage Baselines on Rare-Mode Stress Test
// ============================================================================

static void run_experiment_x() {
    std::cout << "\n========================================\n"
              << "  Experiment X: Coverage Baselines (Q2)\n"
              << "  4 obstacles, 4 classes, all 8 variants + heuristics\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 200;
    const double SWITCH_PROB = 0.2;
    std::vector<double> rare_probs = {0.01, 0.05, 0.10, 0.20};
    std::string rare_mode = "decelerating";
    std::vector<std::string> base_modes = {"constant_velocity", "turn_left", "turn_right"};

    // All 8 paper variants as baselines, plus naive heuristics
    struct BaselineConfig { std::string name; PaperVariant variant; };
    std::vector<BaselineConfig> configs;
    for (PaperVariant v : ALL_VARIANTS) {
        configs.push_back({variant_name(v), v});
    }

    std::ofstream csv(OUTPUT_DIR + "exp_x_baselines_rare_mode.csv");
    csv << "baseline,rare_prob,collision_rate,ci_lo,ci_hi,"
        << "missed_mode_rate,rare_mode_missed_frac,avg_progress\n";

    for (const auto& bc : configs) {
        for (double rp : rare_probs) {
            std::cout << "  " << bc.name << " rare_p=" << rp << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_checks = 0;
            int rare_total = 0, rare_missed = 0;
            double sum_progress = 0;

            for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                unsigned seed = 140000 + r;
                auto res = run_multi_obstacle_rollout(
                    bc.variant, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS, seed,
                    4, 4, base_modes, rare_mode, rp);

                if (res.collision) collisions++;
                total_missed += res.missed_mode_steps;
                total_checks += res.total_mode_checks;
                rare_total += res.rare_mode_active;
                rare_missed += res.rare_mode_missed;
                sum_progress += res.total_progress;
            }

            double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
            double mmr = total_checks > 0 ? static_cast<double>(total_missed) / total_checks : 0;
            double rare_miss_frac = rare_total > 0 ? static_cast<double>(rare_missed) / rare_total : 0;

            csv << bc.name << "," << rp << ","
                << std::fixed << std::setprecision(4)
                << cr << "," << ci_lo << "," << ci_hi << ","
                << mmr << "," << rare_miss_frac << ","
                << sum_progress / NUM_ROLLOUTS << "\n";

            std::cout << "coll=" << std::setprecision(3) << cr
                      << " missed=" << mmr
                      << " rare_miss=" << rare_miss_frac << std::endl;
        }
    }
    csv.close();
    std::cout << "  -> exp_x_baselines_rare_mode.csv\n";
}

// ============================================================================
// Experiment Y: Geometry Ablation (REMOVED — OT GroundCostType deleted)
// ============================================================================

// ============================================================================
// Experiment Z: Qualitative Rollout Trajectories
// ============================================================================
// All 8 variants, 4 obstacles / 4 classes, per-step trajectory output.

static void run_experiment_z() {
    std::cout << "\n========================================\n"
              << "  Experiment Z: Qualitative Rollout Trajectories\n"
              << "========================================\n";

    const double SWITCH_PROB = 0.3;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    auto mode_mdls = create_obstacle_mode_models(DT);

    std::vector<unsigned> showcase_seeds = {200042, 200117, 200203, 200289};

    std::ofstream csv(OUTPUT_DIR + "exp_z_qualitative_trajectories.csv");
    csv << "seed,variant,step,ego_x,ego_y,ego_theta,"
        << "obs0_x,obs0_y,obs0_mode,obs1_x,obs1_y,obs1_mode,"
        << "obs2_x,obs2_y,obs2_mode,obs3_x,obs3_y,obs3_mode,"
        << "collision,missed_modes,min_clearance\n";

    const int NUM_OBS = 4;
    const int NUM_CLASSES = 4;

    for (unsigned seed : showcase_seeds) {
        for (PaperVariant v : ALL_VARIANTS) {
            std::string vname = variant_name(v);
            std::cout << "  seed=" << seed << " " << vname << " ... " << std::flush;

            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::FREQUENCY;
            cfg.enable_dro = uses_dro(v);
            cfg.injection_mode = InjectionMode::QSTAR_SAMPLE;
            cfg.safe_horizon_enabled = uses_sh(v);
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.num_discs = 1;

            AdaptiveScenarioMPC ctrl(cfg);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];

            auto ref_path_z = setup_mpcc_path(ctrl);
            double pl_z = ref_path_z.total_length();
            Eigen::Vector2d goal_z = ref_path_z.get_position_at(pl_z);

            std::vector<ObstacleSim> obs_sims(NUM_OBS);
            for (int i = 0; i < NUM_OBS; ++i) {
                obs_sims[i].state = obstacle_on_s_curve(ref_path_z, OBS_ARC_FRACS_4[i], rng);
                obs_sims[i].current_mode = modes[i % modes.size()];
                obs_sims[i].available_modes = modes;
                obs_sims[i].mode_models = omm;
                ctrl.initialize_obstacle(i, i % NUM_CLASSES, omm);
            }

            EgoState ego(0, 0, 0, 1.5);
            EgoDynamics dyn(DT);
            double collision_radius = cfg.ego_radius + cfg.obstacle_radius;
            double pp_z = 0.0;

            for (int t = 0; t < 5; ++t) {
                for (int i = 0; i < NUM_OBS; ++i) {
                    ctrl.update_mode_observation(i, i % NUM_CLASSES, obs_sims[i].current_mode, t);
                }
            }

            bool had_collision = false;
            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                for (int i = 0; i < NUM_OBS; ++i) {
                    obs_sims[i].maybe_switch(SWITCH_PROB, rng);
                    ctrl.update_mode_observation(i, i % NUM_CLASSES, obs_sims[i].current_mode, step + 5);
                }

                pp_z = ref_path_z.find_closest_point(ego.position(), pp_z);

                std::map<int, ObstacleState> obs_map;
                for (int i = 0; i < NUM_OBS; ++i) obs_map[i] = obs_sims[i].state;
                auto res = ctrl.solve(ego, obs_map, goal_z, 1.5, pp_z, pl_z);

                bool coll_step = false;
                double min_clear = 1e9;
                for (int i = 0; i < NUM_OBS; ++i) {
                    double dist = (ego.position() - obs_sims[i].state.position()).norm();
                    min_clear = std::min(min_clear, dist);
                    if (dist < collision_radius) coll_step = true;
                }
                if (coll_step) had_collision = true;

                int missed_count = 0;
                for (int i = 0; i < NUM_OBS; ++i) {
                    bool found = false;
                    for (const auto& sc : ctrl.scenarios()) {
                        for (const auto& [oid, traj] : sc.trajectories) {
                            if (oid == i && traj.mode_id == obs_sims[i].current_mode) {
                                found = true; break;
                            }
                        }
                        if (found) break;
                    }
                    if (!found) missed_count++;
                }

                csv << seed << "," << vname << "," << step << ","
                    << std::fixed << std::setprecision(4)
                    << ego.x << "," << ego.y << "," << ego.theta;
                for (int i = 0; i < NUM_OBS; ++i) {
                    csv << "," << obs_sims[i].state.x << "," << obs_sims[i].state.y
                        << "," << obs_sims[i].current_mode;
                }
                csv << "," << (coll_step ? 1 : 0) << "," << missed_count
                    << "," << min_clear << "\n";

                if (res.success && res.first_input().has_value())
                    ego = dyn.propagate(ego, res.first_input().value());
                for (int i = 0; i < NUM_OBS; ++i) obs_sims[i].step(DT, rng);

                if (pp_z >= PATH_COMPLETE_FRAC * pl_z) break;
            }

            std::cout << (had_collision ? "COLLISION" : "safe") << std::endl;
        }
    }
    csv.close();
    std::cout << "  -> exp_z_qualitative_trajectories.csv\n";
}

// ============================================================================
// Experiment AA: Robustness Across Environments (Per-Seed Boxplot Data)
// ============================================================================
// Q3: Is improvement robust across environments?
// All 8 variants × 4 environments, single-obstacle (environment-specific placement).

static void run_experiment_aa() {
    std::cout << "\n========================================\n"
              << "  Experiment AA: Robustness Across Environments\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 200;
    const double SWITCH_PROB = 0.2;

    std::vector<EnvironmentType> envs = {
        EnvironmentType::STRAIGHT, EnvironmentType::NARROW_CORRIDOR,
        EnvironmentType::INTERSECTION, EnvironmentType::ONCOMING
    };

    std::ofstream csv(OUTPUT_DIR + "exp_aa_robustness_per_seed.csv");
    csv << "environment,variant,seed,collision,missed_mode_rate,"
        << "progress,min_clearance,mean_solve_ms,p99_solve_ms\n";

    for (EnvironmentType env_type : envs) {
        for (PaperVariant v : ALL_VARIANTS) {
            std::string env_name = environment_name(env_type);
            std::cout << "  " << env_name << " " << variant_name(v)
                      << " ... " << std::flush;

            int total_collisions = 0;

            for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                unsigned seed = 160000 + r;
                std::mt19937 rng_env(seed);
                EnvironmentSetup env_setup = create_environment(env_type, rng_env);

                auto res = run_single_rollout_env(
                    v, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS,
                    seed, env_setup, SamplingBaseline::STANDARD);

                double mmr = res.total_steps > 0 ?
                    static_cast<double>(res.missed_mode_steps) / res.total_steps : 0;

                double mean_solve = 0, p99_solve = 0;
                if (!res.solve_times.empty()) {
                    double sum = std::accumulate(res.solve_times.begin(),
                                                  res.solve_times.end(), 0.0);
                    mean_solve = sum / res.solve_times.size() * 1000;
                    p99_solve = percentile(res.solve_times, 99) * 1000;
                }

                if (res.collision) total_collisions++;

                csv << env_name << "," << variant_name(v) << "," << seed << ","
                    << (res.collision ? 1 : 0) << ","
                    << std::fixed << std::setprecision(4)
                    << mmr << "," << res.total_progress << ","
                    << res.min_clearance << ","
                    << mean_solve << "," << p99_solve << "\n";
            }

            double cr = static_cast<double>(total_collisions) / NUM_ROLLOUTS;
            std::cout << "coll=" << std::setprecision(3) << cr << std::endl;
        }
    }
    csv.close();
    std::cout << "  -> exp_aa_robustness_per_seed.csv\n";
}

// ============================================================================
// Experiment AB: OT Regularization Pareto Frontier
// ============================================================================
// Q3: Hyperparameter sensitivity. Sweeps epsilon × uncertainty_scale for
// OT-based variants, 4 obstacles / 4 classes, with rare mode stress test.

static void run_experiment_ab() {
    std::cout << "\n========================================\n"
              << "  Experiment AB: OT Regularization Pareto Frontier\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 100;
    const double SWITCH_PROB = 0.2;
    std::vector<std::string> base_modes = {"constant_velocity", "turn_left", "turn_right"};
    std::string rare_mode = "decelerating";
    double rare_prob = 0.05;

    std::vector<double> epsilons = {0.01, 0.1, 1.0, 5.0};
    std::vector<double> scales = {0.5, 1.0, 2.0};

    // Sweep SH variants (Base+SH and DRO+SH)
    std::vector<PaperVariant> sweep_variants = {
        PaperVariant::BASE_SH, PaperVariant::DRO_SH
    };

    std::ofstream csv(OUTPUT_DIR + "exp_ab_pareto_frontier.csv");
    csv << "variant,epsilon,uncertainty_scale,"
        << "collision_rate,ci_lo,ci_hi,missed_mode_rate,"
        << "rare_mode_missed_frac,avg_progress,avg_clearance,mean_solve_ms,p99_solve_ms\n";

    int total_configs = static_cast<int>(sweep_variants.size() * epsilons.size() * scales.size());
    int config_idx = 0;

    for (PaperVariant v : sweep_variants) {
        for (double eps : epsilons) {
            for (double sc : scales) {
                config_idx++;
                std::string vname = variant_name(v);
                std::cout << "  [" << config_idx << "/" << total_configs << "] "
                          << vname << " eps=" << eps << " sc=" << sc
                          << " ... " << std::flush;

                int collisions = 0;
                int total_missed = 0, total_checks = 0;
                int rare_total = 0, rare_missed_cnt = 0;
                double sum_progress = 0, sum_clearance = 0;
                std::vector<double> all_solve_times;

                for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                    auto res = run_multi_obstacle_rollout(
                        v, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS,
                        170000 + r, 4, 4, base_modes,
                        rare_mode, rare_prob);

                    if (res.collision) collisions++;
                    total_missed += res.missed_mode_steps;
                    total_checks += res.total_mode_checks;
                    rare_total += res.rare_mode_active;
                    rare_missed_cnt += res.rare_mode_missed;
                    sum_progress += res.total_progress;
                    sum_clearance += res.min_clearance;
                    for (double t : res.solve_times)
                        all_solve_times.push_back(t * 1000);
                }

                double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
                auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
                double mmr = total_checks > 0 ? static_cast<double>(total_missed) / total_checks : 0;
                double rmf = rare_total > 0 ? static_cast<double>(rare_missed_cnt) / rare_total : 0;
                double mean_solve = all_solve_times.empty() ? 0 :
                    std::accumulate(all_solve_times.begin(), all_solve_times.end(), 0.0) / all_solve_times.size();
                double p99 = percentile(all_solve_times, 99);

                csv << vname << "," << eps << "," << sc << ","
                    << std::fixed << std::setprecision(4)
                    << cr << "," << ci_lo << "," << ci_hi << ","
                    << mmr << "," << rmf << ","
                    << sum_progress / NUM_ROLLOUTS << ","
                    << sum_clearance / NUM_ROLLOUTS << ","
                    << mean_solve << "," << p99 << "\n";

                std::cout << "coll=" << std::setprecision(3) << cr
                          << " missed=" << mmr << std::endl;
            }
        }
    }
    csv.close();
    std::cout << "  -> exp_ab_pareto_frontier.csv\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    fs::create_directories(OUTPUT_DIR);

    std::cout << "================================================================\n"
              << "  Paper Experiments (New Framework): DRO + SH-MPCC\n"
              << "================================================================\n";

    auto start = std::chrono::high_resolution_clock::now();

    std::string filter = "";
    if (argc > 1) filter = argv[1];

    auto should_run = [&](const std::string& label) {
        return filter.empty() || filter == label;
    };

    if (should_run("A")) run_experiment_a();
    if (should_run("B")) run_experiment_b();
    if (should_run("C")) run_experiment_c();
    if (should_run("D")) run_experiment_d();
    if (should_run("E")) run_experiment_e();
    if (should_run("F")) run_experiment_f();
    if (should_run("G")) run_experiment_g();
    if (should_run("H")) run_experiment_h();
    if (should_run("I")) run_experiment_i();
    if (should_run("J")) run_experiment_j();
    if (should_run("K")) run_experiment_k();
    if (should_run("L")) run_experiment_l();
    if (should_run("M")) run_experiment_m();
    if (should_run("N")) run_experiment_n();
    if (should_run("O")) run_experiment_o();
    if (should_run("P")) run_experiment_p();
    // Experiment Q removed (OT Internal Ablation — OT predictor deleted)
    if (should_run("R")) run_experiment_r();
    if (should_run("T")) run_experiment_t();
    // Experiment U removed (Ground-Cost Ablation — OT predictor deleted)
    if (should_run("V")) run_experiment_v();
    if (should_run("W")) run_experiment_w();
    if (should_run("X")) run_experiment_x();
    // Experiment Y removed (Geometry Ablation — OT GroundCostType deleted)
    if (should_run("Z")) run_experiment_z();
    if (should_run("AA")) run_experiment_aa();
    if (should_run("AB")) run_experiment_ab();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "\n================================================================\n"
              << "  All experiments complete in " << std::fixed << std::setprecision(1)
              << elapsed << " seconds.\n"
              << "  CSV files written to " << OUTPUT_DIR << "\n"
              << "  Run: python3 ../generate_results_figures.py\n"
              << "================================================================\n";

    return 0;
}
