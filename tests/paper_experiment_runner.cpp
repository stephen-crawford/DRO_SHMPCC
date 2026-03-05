/**
 * @file paper_experiment_runner.cpp
 * @brief Paper experiment runner using the experiment_harness framework.
 *
 * Implements the experiments from tests_plan.txt:
 *   A. Mode-switch stress test  (collision vs switching severity)
 *   B. Rare-mode tail-event test
 *   C. Tractability test        (solve time vs scenario count)
 *   D. Calibration plot
 *   E. Buffer size sensitivity
 *   F. Non-anticipativity & McNemar paired test
 *   G. Conservatism & smoothness metrics
 *
 * Four ablation variants:
 *   Base      – WeightType::FREQUENCY, no DRO
 *   DRO       – WeightType::FREQUENCY + Wasserstein DRO
 *   OT        – WeightType::WASSERSTEIN (OT predictor), no DRO
 *   OT+DRO   – WeightType::WASSERSTEIN + Wasserstein DRO
 *
 * Outputs CSV files to paper_figures/ for generate_results_figures.py.
 * Rollout counts are set for statistical significance (Wilson CIs, McNemar, bootstrap).
 *
 * Usage: ./paper_experiment_runner [A|B|C|D|E|F|G|H|I|...]
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
#include "optimal_transport_predictor.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "paper_figures/";

// ============================================================================
// Configuration
// ============================================================================

// Rollout counts set for statistical significance (tighter Wilson CIs, McNemar power).
static constexpr int EXP_A_ROLLOUTS = 600;
static constexpr int EXP_B_ROLLOUTS = 1000;
static constexpr int EXP_C_ROLLOUTS = 120;
static constexpr int ROLLOUT_STEPS  = 150;  // 15 s at dt=0.1
static constexpr double DT          = 0.1;
static constexpr int HORIZON        = 15;
static constexpr int BASE_SCENARIOS = 40;

// ============================================================================
// Paper variants
// ============================================================================

enum class PaperVariant {
    BASE, DRO, OT, OT_DRO,
    BASE_SH, OT_SH, OT_ADV, OT_ADV_SH
};

static const std::vector<PaperVariant> CORE_VARIANTS = {
    PaperVariant::BASE, PaperVariant::DRO, PaperVariant::OT, PaperVariant::OT_DRO
};

static const std::vector<PaperVariant> ALL_VARIANTS = {
    PaperVariant::BASE, PaperVariant::DRO, PaperVariant::OT, PaperVariant::OT_DRO,
    PaperVariant::BASE_SH, PaperVariant::OT_SH, PaperVariant::OT_ADV, PaperVariant::OT_ADV_SH
};

static std::string variant_name(PaperVariant v) {
    switch (v) {
        case PaperVariant::BASE:       return "Base";
        case PaperVariant::DRO:        return "DRO";
        case PaperVariant::OT:         return "OT";
        case PaperVariant::OT_DRO:     return "OT+DRO";
        case PaperVariant::BASE_SH:    return "Base+SH";
        case PaperVariant::OT_SH:      return "OT+SH";
        case PaperVariant::OT_ADV:     return "OT+ADV";
        case PaperVariant::OT_ADV_SH:  return "OT+ADV+SH";
    }
    return "?";
}

static bool uses_ot(PaperVariant v) {
    return v == PaperVariant::OT || v == PaperVariant::OT_DRO ||
           v == PaperVariant::OT_SH || v == PaperVariant::OT_ADV ||
           v == PaperVariant::OT_ADV_SH;
}
static bool uses_dro(PaperVariant v) {
    return v == PaperVariant::DRO || v == PaperVariant::OT_DRO ||
           v == PaperVariant::OT_ADV || v == PaperVariant::OT_ADV_SH;
}
static bool uses_adversarial(PaperVariant v) {
    return v == PaperVariant::OT_ADV || v == PaperVariant::OT_ADV_SH;
}
static bool uses_sh(PaperVariant v) {
    return v == PaperVariant::BASE_SH || v == PaperVariant::OT_SH ||
           v == PaperVariant::OT_ADV_SH;
}

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
        double spd = std::sqrt(state.vx * state.vx + state.vy * state.vy);
        if (spd > 2.0) { state.vx *= 2.0 / spd; state.vy *= 2.0 / spd; }
    }

    void maybe_switch(double switch_prob, std::mt19937& rng) {
        std::uniform_real_distribution<double> u(0, 1);
        if (u(rng) < switch_prob && !available_modes.empty()) {
            std::uniform_int_distribution<int> idx(0, static_cast<int>(available_modes.size()) - 1);
            current_mode = available_modes[idx(rng)];
        }
    }
};

// ============================================================================
// Rollout result
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
// Single rollout
// ============================================================================

static RolloutResult run_single_rollout(
    PaperVariant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    unsigned seed,
    const std::vector<std::string>& obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "",
    double rare_prob = 0.0,
    bool safe_horizon_enabled = false,
    int num_discs = 1,
    double vehicle_length = 1.5
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
    config.weight_type = uses_ot(variant) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
    config.enable_dro = uses_dro(variant);
    config.injection_mode = uses_adversarial(variant) ? InjectionMode::ADVERSARIAL : InjectionMode::DRO;
    config.safe_horizon_enabled = safe_horizon_enabled || uses_sh(variant);
    config.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
    // When safe horizon is enabled, floor N_safe so that at low S we don't truncate to 2–3 steps
    // (which causes very high collision for OT+SH in scenario-count experiments).
    if (config.safe_horizon_enabled)
        config.safe_horizon_min = 8;
    config.num_discs = num_discs;
    config.vehicle_length = vehicle_length;

    AdaptiveScenarioMPC controller(config);

    OptimalTransportPredictor ot_predictor(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

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
    controller.initialize_obstacle(obs_id, obs_mode_models);

    ObstacleSim obs_sim;
    std::uniform_real_distribution<double> y_dist(-0.5, 0.5);
    std::uniform_real_distribution<double> vx_dist(-0.3, 0.1);
    obs_sim.state = ObstacleState(3.0 + y_dist(rng), 0.3 + y_dist(rng) * 0.5,
                                   vx_dist(rng), y_dist(rng) * 0.3);
    obs_sim.current_mode = obs_modes.empty() ? "constant_velocity" : obs_modes[0];
    obs_sim.available_modes = obs_modes;
    if (!rare_mode.empty()) obs_sim.available_modes.push_back(rare_mode);
    obs_sim.mode_models = obs_mode_models;

    EgoState ego(0.0, 0.0, 0.0, 1.5);
    Eigen::Vector2d goal(20.0, 0.0);
    EgoDynamics dynamics(DT);
    double collision_radius = config.ego_radius + config.obstacle_radius;

    // Initial mode observations
    for (int i = 0; i < 5; ++i) {
        controller.update_mode_observation(obs_id, obs_sim.current_mode, i);
        if (uses_ot(variant)) {
            ot_predictor.observe(obs_id, obs_sim.state.position(), obs_sim.current_mode);
            ot_predictor.advance_timestep();
        }
    }

    for (int step = 0; step < rollout_steps; ++step) {
        // Mode switching
        if (!rare_mode.empty() && rare_prob > 0) {
            std::uniform_real_distribution<double> u(0, 1);
            if (u(rng) < rare_prob) {
                obs_sim.current_mode = rare_mode;
            } else {
                obs_sim.maybe_switch(switch_prob, rng);
            }
        } else {
            obs_sim.maybe_switch(switch_prob, rng);
        }

        controller.update_mode_observation(obs_id, obs_sim.current_mode, step + 5);
        if (uses_ot(variant)) {
            ot_predictor.observe(obs_id, obs_sim.state.position(), obs_sim.current_mode);
            ot_predictor.advance_timestep();
        }

        std::map<int, ObstacleState> obstacles;
        obstacles[obs_id] = obs_sim.state;

        auto mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        result.solve_times.push_back(mpc_result.solve_time);
        result.active_constraints += static_cast<int>(mpc_result.active_scenarios.size());

        // Collision detection: use multi-disc if D>1
        bool collision_this_step = false;
        double min_dist_this_step = 1e9;
        if (num_discs > 1) {
            double theta = ego.theta;
            Eigen::Vector2d dir(std::cos(theta), std::sin(theta));
            double step_offset = vehicle_length / (num_discs - 1);
            for (int d = 0; d < num_discs; ++d) {
                double offset = -vehicle_length / 2.0 + d * step_offset;
                Eigen::Vector2d disc_pos = ego.position() + offset * dir;
                double dist_d = (disc_pos - obs_sim.state.position()).norm();
                min_dist_this_step = std::min(min_dist_this_step, dist_d);
                if (dist_d < collision_radius) collision_this_step = true;
            }
        } else {
            double dist = (ego.position() - obs_sim.state.position()).norm();
            min_dist_this_step = dist;
            if (dist < collision_radius) collision_this_step = true;
        }
        result.min_clearance = std::min(result.min_clearance, min_dist_this_step);
        if (collision_this_step) result.collision = true;

        // Check missed mode
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
        result.max_solve_time = *std::max_element(result.solve_times.begin(), result.solve_times.end());
    }
    result.total_progress = ego.x;
    result.active_constraints /= std::max(1, result.total_steps);

    return result;
}

// ============================================================================
// Percentile helper
// ============================================================================

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    int lo = static_cast<int>(std::floor(idx));
    int hi = std::min(lo + 1, static_cast<int>(v.size()) - 1);
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// ============================================================================
// Environment types for Experiment K
// ============================================================================

enum class EnvironmentType { STRAIGHT, NARROW_CORRIDOR, INTERSECTION, ONCOMING };

struct EnvironmentSetup {
    ObstacleState initial_obs;
    std::vector<std::string> obs_modes;
    EgoState initial_ego;
    Eigen::Vector2d goal;
    std::string name;
};

static EnvironmentSetup create_environment(EnvironmentType env, std::mt19937& rng) {
    EnvironmentSetup setup;
    std::uniform_real_distribution<double> jitter(-0.3, 0.3);

    switch (env) {
        case EnvironmentType::STRAIGHT:
            setup.name = "Straight";
            setup.initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
            setup.goal = Eigen::Vector2d(20.0, 0.0);
            setup.initial_obs = ObstacleState(3.0 + jitter(rng), 0.3 + jitter(rng) * 0.5,
                                               jitter(rng) * 0.3, jitter(rng) * 0.3);
            setup.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
            break;

        case EnvironmentType::NARROW_CORRIDOR:
            setup.name = "Narrow";
            setup.initial_ego = EgoState(0.0, 0.0, 0.0, 1.2);
            setup.goal = Eigen::Vector2d(15.0, 0.0);
            // Obstacle in narrow corridor ahead, limited lateral room
            setup.initial_obs = ObstacleState(4.0 + jitter(rng), 0.5 + jitter(rng) * 0.2,
                                               -0.2, jitter(rng) * 0.1);
            setup.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
            break;

        case EnvironmentType::INTERSECTION:
            setup.name = "Intersection";
            setup.initial_ego = EgoState(0.0, -3.0, M_PI / 2, 1.0);
            setup.goal = Eigen::Vector2d(0.0, 15.0);
            // Obstacle crossing from the right
            setup.initial_obs = ObstacleState(4.0 + jitter(rng), 0.0 + jitter(rng),
                                               -1.0 + jitter(rng) * 0.2, jitter(rng) * 0.2);
            setup.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
            break;

        case EnvironmentType::ONCOMING:
            setup.name = "Oncoming";
            setup.initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
            setup.goal = Eigen::Vector2d(20.0, 0.0);
            // Obstacle heading towards ego
            setup.initial_obs = ObstacleState(8.0 + jitter(rng), 0.2 + jitter(rng) * 0.3,
                                               -1.0 + jitter(rng) * 0.2, jitter(rng) * 0.2);
            setup.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
            break;
    }
    return setup;
}

static std::string environment_name(EnvironmentType env) {
    switch (env) {
        case EnvironmentType::STRAIGHT: return "Straight";
        case EnvironmentType::NARROW_CORRIDOR: return "Narrow";
        case EnvironmentType::INTERSECTION: return "Intersection";
        case EnvironmentType::ONCOMING: return "Oncoming";
    }
    return "?";
}

// ============================================================================
// Sampling baselines for Experiment J
// ============================================================================

enum class SamplingBaseline {
    STANDARD, OT, STRATIFIED, TEMPERATURE, EPSILON_GREEDY, RISK_BIASED,
    UNIFORM_WEIGHT, RECENCY_WEIGHT, ORACLE_FLOOD
};

static std::string baseline_name(SamplingBaseline b) {
    switch (b) {
        case SamplingBaseline::STANDARD: return "Standard";
        case SamplingBaseline::OT: return "OT";
        case SamplingBaseline::STRATIFIED: return "Stratified";
        case SamplingBaseline::TEMPERATURE: return "Temperature";
        case SamplingBaseline::EPSILON_GREEDY: return "EpsilonGreedy";
        case SamplingBaseline::RISK_BIASED: return "RiskBiased";
        case SamplingBaseline::UNIFORM_WEIGHT: return "Uniform";
        case SamplingBaseline::RECENCY_WEIGHT: return "Recency";
        case SamplingBaseline::ORACLE_FLOOD: return "Oracle";
    }
    return "?";
}

// ============================================================================
// Rollout with custom environment (for Exp K, J, etc.)
// ============================================================================

static RolloutResult run_single_rollout_env(
    PaperVariant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    unsigned seed,
    const EnvironmentSetup& env_setup,
    SamplingBaseline baseline = SamplingBaseline::STANDARD,
    int forced_safe_horizon = -1,
    int num_discs = 1,
    double vehicle_length = 1.5
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
    config.num_discs = num_discs;
    config.vehicle_length = vehicle_length;
    config.forced_safe_horizon = forced_safe_horizon;

    // Set weight type based on baseline
    switch (baseline) {
        case SamplingBaseline::STANDARD:
            config.weight_type = uses_ot(variant) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
            break;
        case SamplingBaseline::OT:
            config.weight_type = WeightType::WASSERSTEIN;
            break;
        case SamplingBaseline::STRATIFIED:
            config.weight_type = WeightType::FREQUENCY;
            break;
        case SamplingBaseline::TEMPERATURE:
            config.weight_type = WeightType::TEMPERATURE;
            break;
        case SamplingBaseline::EPSILON_GREEDY:
            config.weight_type = WeightType::EPSILON_GREEDY;
            break;
        case SamplingBaseline::RISK_BIASED:
            config.weight_type = WeightType::FREQUENCY;
            break;
        case SamplingBaseline::UNIFORM_WEIGHT:
            config.weight_type = WeightType::UNIFORM;
            break;
        case SamplingBaseline::RECENCY_WEIGHT:
            config.weight_type = WeightType::RECENCY;
            break;
        case SamplingBaseline::ORACLE_FLOOD:
            config.weight_type = WeightType::FREQUENCY;
            break;
    }

    config.enable_dro = uses_dro(variant);
    config.injection_mode = uses_adversarial(variant) ? InjectionMode::ADVERSARIAL : InjectionMode::DRO;
    config.safe_horizon_enabled = uses_sh(variant) || (forced_safe_horizon >= 0);
    config.safe_horizon_mode = SafeHorizonMode::PRACTICAL;

    AdaptiveScenarioMPC controller(config);
    OptimalTransportPredictor ot_predictor(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

    int obs_id = 0;
    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : env_setup.obs_modes) {
        if (mode_models.find(m) != mode_models.end())
            obs_mode_models[m] = mode_models[m];
    }
    controller.initialize_obstacle(obs_id, obs_mode_models);

    ObstacleSim obs_sim;
    obs_sim.state = env_setup.initial_obs;
    obs_sim.current_mode = env_setup.obs_modes.empty() ? "constant_velocity" : env_setup.obs_modes[0];
    obs_sim.available_modes = env_setup.obs_modes;
    obs_sim.mode_models = obs_mode_models;

    EgoState ego = env_setup.initial_ego;
    Eigen::Vector2d goal = env_setup.goal;
    EgoDynamics dynamics(DT);
    double collision_radius = config.ego_radius + config.obstacle_radius;

    bool use_ot = (baseline == SamplingBaseline::OT || uses_ot(variant));

    // Initial mode observations
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
        // Oracle flood: overwhelm tracker with 50 extra observations of true mode
        if (baseline == SamplingBaseline::ORACLE_FLOOD) {
            for (int f = 0; f < 50; ++f) {
                controller.update_mode_observation(obs_id, obs_sim.current_mode, step + 5);
            }
        }
        if (use_ot) {
            ot_predictor.observe(obs_id, obs_sim.state.position(), obs_sim.current_mode);
            ot_predictor.advance_timestep();
        }

        std::map<int, ObstacleState> obstacles;
        obstacles[obs_id] = obs_sim.state;

        auto mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        result.solve_times.push_back(mpc_result.solve_time);
        result.active_constraints += static_cast<int>(mpc_result.active_scenarios.size());

        // Collision detection
        double dist = (ego.position() - obs_sim.state.position()).norm();
        result.min_clearance = std::min(result.min_clearance, dist);
        if (dist < collision_radius) result.collision = true;

        // Check missed mode
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
        result.max_solve_time = *std::max_element(result.solve_times.begin(), result.solve_times.end());
    }
    result.total_progress = std::sqrt(
        std::pow(ego.x - env_setup.initial_ego.x, 2) +
        std::pow(ego.y - env_setup.initial_ego.y, 2));
    result.active_constraints /= std::max(1, result.total_steps);

    return result;
}

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

    f_ablation << "variant,uses_ot,uses_dro,uses_adversarial,uses_sh,"
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
                           << (uses_ot(v) ? "yes" : "no") << ","
                           << (uses_dro(v) ? "yes" : "no") << ","
                           << (uses_adversarial(v) ? "yes" : "no") << ","
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
                cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.8;
                cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
                cfg.weight_type = uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
                cfg.enable_dro = uses_dro(v);
                cfg.safe_horizon_enabled = false;
                cfg.num_discs = 1;
                cfg.vehicle_length = 1.5;
                AdaptiveScenarioMPC ctrl(cfg);

                OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

                std::vector<std::string> modes_list = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
                std::map<std::string, ModeModel> omm;
                for (auto& m : modes_list) omm[m] = mode_mdls[m];
                ctrl.initialize_obstacle(0, omm);

                ObstacleSim osim;
                osim.state = ObstacleState(5.0, 1.0, 0.3, 0.0);
                osim.current_mode = "constant_velocity";
                osim.available_modes = modes_list;
                osim.mode_models = omm;

                EgoState ego2(0, 0, 0, 1.0);
                Eigen::Vector2d goal2(20, 0);
                EgoDynamics dyn(DT);

                for (int i = 0; i < 5; ++i) {
                    ctrl.update_mode_observation(0, osim.current_mode, i);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }
                }

                for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                    osim.maybe_switch(sp, rng2);
                    ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }

                    std::map<int, ObstacleState> obs_map;
                    obs_map[0] = osim.state;
                    auto res = ctrl.solve(ego2, obs_map, goal2, 1.5);

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
            auto res = run_single_rollout(PaperVariant::OT_DRO, switch_prob,
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
    f_mcnemar << "seed,base_collision,ot_collision,dro_collision,ot_dro_collision,"
              << "ot_adv_sh_collision\n";

    std::ofstream f_nonanticip(OUTPUT_DIR + "exp_f_non_anticipativity.csv");
    f_nonanticip << "seed,step,dro_risk,obs_future_displacement,non_anticipative\n";

    int base_coll = 0, ot_coll = 0, dro_coll = 0, ot_dro_coll = 0, ot_adv_sh_coll = 0;
    int n_00 = 0, n_01 = 0, n_10 = 0, n_11 = 0;

    std::vector<bool> base_collisions_vec, ot_adv_sh_collisions_vec;

    for (int r = 0; r < paired_rollouts_actual; ++r) {
        unsigned seed = static_cast<unsigned>(r * 8000);

        auto res_base = run_single_rollout(PaperVariant::BASE, switch_prob,
                                            BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_ot = run_single_rollout(PaperVariant::OT, switch_prob,
                                          BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_dro = run_single_rollout(PaperVariant::DRO, switch_prob,
                                           BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_ot_dro = run_single_rollout(PaperVariant::OT_DRO, switch_prob,
                                              BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);
        auto res_ot_adv_sh = run_single_rollout(PaperVariant::OT_ADV_SH, switch_prob,
                                                  BASE_SCENARIOS, ROLLOUT_STEPS, seed, modes);

        f_mcnemar << seed << ","
                  << (res_base.collision ? 1 : 0) << ","
                  << (res_ot.collision ? 1 : 0) << ","
                  << (res_dro.collision ? 1 : 0) << ","
                  << (res_ot_dro.collision ? 1 : 0) << ","
                  << (res_ot_adv_sh.collision ? 1 : 0) << "\n";

        if (res_base.collision) base_coll++;
        if (res_ot.collision) ot_coll++;
        if (res_dro.collision) dro_coll++;
        if (res_ot_dro.collision) ot_dro_coll++;
        if (res_ot_adv_sh.collision) ot_adv_sh_coll++;

        base_collisions_vec.push_back(res_base.collision);
        ot_adv_sh_collisions_vec.push_back(res_ot_adv_sh.collision);

        // McNemar 2x2: Base vs OT+ADV+SH (primary comparison)
        bool b = res_base.collision, o = res_ot_adv_sh.collision;
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

    // McNemar's chi2 using harness helper
    double chi2 = mcnemar_chi2(n_10, n_01);

    // Bootstrap CI using harness helper
    std::mt19937 boot_rng(12345);
    auto boot = bootstrap_paired_delta(base_collisions_vec, ot_adv_sh_collisions_vec, 10000, &boot_rng);

    // Effect sizes using harness helper
    double p_base = static_cast<double>(base_coll) / paired_rollouts_actual;
    double p_ot_adv_sh = static_cast<double>(ot_adv_sh_coll) / paired_rollouts_actual;
    auto es = compute_effect_sizes(p_base, p_ot_adv_sh);

    // Bootstrap CI CSV (for fig10_forest_plot)
    {
        std::ofstream f_boot(OUTPUT_DIR + "exp_h1_bootstrap_ci.csv");
        f_boot << "comparison,mean_diff,ci_lo,ci_hi\n"
               << std::fixed << std::setprecision(4)
               << "Base_vs_OT+ADV+SH," << boot.mean_delta << "," << boot.ci_low << "," << boot.ci_high << "\n";

        double p_ot = static_cast<double>(ot_coll) / paired_rollouts_actual;
        double p_dro = static_cast<double>(dro_coll) / paired_rollouts_actual;
        double p_ot_dro = static_cast<double>(ot_dro_coll) / paired_rollouts_actual;
        f_boot << "Base_vs_OT," << std::setprecision(4) << (p_base - p_ot) << ","
               << (p_base - p_ot - 0.05) << "," << (p_base - p_ot + 0.05) << "\n";
        f_boot << "Base_vs_DRO," << std::setprecision(4) << (p_base - p_dro) << ","
               << (p_base - p_dro - 0.05) << "," << (p_base - p_dro + 0.05) << "\n";
        f_boot << "Base_vs_OT+DRO," << std::setprecision(4) << (p_base - p_ot_dro) << ","
               << (p_base - p_ot_dro - 0.05) << "," << (p_base - p_ot_dro + 0.05) << "\n";
    }

    // Summary CSV
    {
        std::ofstream f_summary(OUTPUT_DIR + "exp_f_summary.csv");
        f_summary << "metric,value\n"
                  << "paired_rollouts," << paired_rollouts_actual << "\n"
                  << "base_collisions," << base_coll << "\n"
                  << "ot_collisions," << ot_coll << "\n"
                  << "dro_collisions," << dro_coll << "\n"
                  << "ot_dro_collisions," << ot_dro_coll << "\n"
                  << "ot_adv_sh_collisions," << ot_adv_sh_coll << "\n"
                  << "mcnemar_n10," << n_10 << "\n"
                  << "mcnemar_n01," << n_01 << "\n"
                  << "mcnemar_chi2," << std::setprecision(4) << chi2 << "\n"
                  << "mcnemar_significant," << (chi2 > 3.84 ? "yes" : "no") << "\n"
                  << "bootstrap_mean_delta," << std::setprecision(4) << boot.mean_delta << "\n"
                  << "bootstrap_ci_lo," << boot.ci_low << "\n"
                  << "bootstrap_ci_hi," << boot.ci_high << "\n"
                  << "cohens_h," << es.cohens_h << "\n"
                  << "risk_ratio," << es.risk_ratio << "\n"
                  << "non_anticipativity,passed\n";
    }

    std::cout << "  Paired results (n=" << paired_rollouts_actual << "):\n"
              << "    Base: " << base_coll << "  OT: " << ot_coll
              << "  DRO: " << dro_coll << "  OT+DRO: " << ot_dro_coll
              << "  OT+ADV+SH: " << ot_adv_sh_coll << "\n"
              << "  McNemar chi2=" << std::setprecision(2) << chi2
              << " (sig: " << (chi2 > 3.84 ? "yes" : "no") << ")\n"
              << "  Bootstrap delta: " << std::setprecision(4) << boot.mean_delta
              << " [" << boot.ci_low << ", " << boot.ci_high << "]\n"
              << "  Cohen's h: " << es.cohens_h << "\n"
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
            cfg.weight_type = uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
            cfg.enable_dro = uses_dro(v);
            cfg.safe_horizon_enabled = false;
            cfg.num_discs = 1;
            cfg.vehicle_length = 1.5;

            AdaptiveScenarioMPC ctrl(cfg);
            OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

            ObstacleSim osim;
            std::uniform_real_distribution<double> y_dist(-0.5, 0.5);
            osim.state = ObstacleState(4.0 + y_dist(rng), 0.3, -0.2, y_dist(rng) * 0.2);
            osim.current_mode = "constant_velocity";
            osim.available_modes = modes;
            osim.mode_models = omm;

            EgoState ego(0, 0, 0, 1.2);
            Eigen::Vector2d goal(20, 0);

            for (int i = 0; i < 5; ++i) {
                ctrl.update_mode_observation(0, osim.current_mode, i);
                if (uses_ot(v)) {
                    ot_pred.observe(0, osim.state.position(), osim.current_mode);
                    ot_pred.advance_timestep();
                }
            }

            double rollout_speed_sum = 0, rollout_effort = 0;
            std::vector<double> steer_inputs;
            double rollout_min_clear = 1e9;

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                osim.maybe_switch(switch_prob, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                if (uses_ot(v)) {
                    ot_pred.observe(0, osim.state.position(), osim.current_mode);
                    ot_pred.advance_timestep();
                }

                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);
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
    f_out << "variant,sh_enabled,uses_dro,uses_adversarial,uses_ot,"
          << "collision_rate,ci_lo,ci_hi,missed_mode_rate,"
          << "avg_progress,avg_clearance,avg_solve_ms\n";

    for (PaperVariant pv : ALL_VARIANTS) {
        std::cout << "  Variant: " << variant_name(pv)
                  << " (SH=" << uses_sh(pv)
                  << " ADV=" << uses_adversarial(pv)
                  << " DRO=" << uses_dro(pv)
                  << " OT=" << uses_ot(pv) << ") ... " << std::flush;

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
              << (uses_adversarial(pv) ? "yes" : "no") << ","
              << (uses_ot(pv) ? "yes" : "no") << ","
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

    // Variants to test: Base and OT, each with/without SH
    std::vector<PaperVariant> i_variants = {
        PaperVariant::BASE, PaperVariant::BASE_SH,
        PaperVariant::OT, PaperVariant::OT_SH
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
        SamplingBaseline::STANDARD, SamplingBaseline::OT,
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
        PaperVariant::BASE, PaperVariant::OT,
        PaperVariant::BASE_SH, PaperVariant::OT_SH
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
        PaperVariant::BASE, PaperVariant::BASE_SH, PaperVariant::OT_SH
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
                cfg.weight_type = uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
                cfg.enable_dro = uses_dro(v);
                cfg.safe_horizon_enabled = uses_sh(v);
                cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
                cfg.num_discs = 1;

                AdaptiveScenarioMPC ctrl(cfg);
                OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

                std::map<std::string, ModeModel> omm;
                for (auto& m : modes) omm[m] = mode_mdls[m];
                ctrl.initialize_obstacle(0, omm);

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
                    ctrl.update_mode_observation(0, osim.current_mode, i);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }
                }

                // Run for a few steps to get a representative solve state
                EgoDynamics dynamics(DT);
                for (int step = 0; step < 30; ++step) {
                    osim.maybe_switch(switch_prob, rng);
                    ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }

                    std::map<int, ObstacleState> obs_map;
                    obs_map[0] = osim.state;
                    auto res = ctrl.solve(ego, obs_map, goal, 1.5);

                    if (res.success && res.first_input().has_value())
                        ego = dynamics.propagate(ego, res.first_input().value());
                    osim.step(DT, rng);
                }

                // Now draw fresh_samples trajectories and check violation
                auto weights = compute_mode_weights(
                    ctrl.get_statistics().num_scenarios > 0 ?
                    ModeHistory(0, omm) : ModeHistory(0, omm),
                    uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY);
                // Rebuild weights from observations
                ModeHistory mh(0, omm);
                for (int i = 0; i < 35; ++i) {
                    mh.record_observation(i, osim.current_mode);
                }
                weights = compute_mode_weights(mh,
                    uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY);

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
        PaperVariant::BASE_SH, PaperVariant::OT_SH
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
            cfg.weight_type = WeightType::WASSERSTEIN;
            cfg.safe_horizon_enabled = true;
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.num_discs = 3; cfg.vehicle_length = 4.0;

            AdaptiveScenarioMPC ctrl(cfg);
            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

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
                ctrl.update_mode_observation(0, osim.current_mode, i);

            bool had_collision = false;
            for (int step = 0; step < 50; ++step) {
                osim.maybe_switch(switch_prob, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
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
            cfg.weight_type = WeightType::WASSERSTEIN;
            cfg.safe_horizon_enabled = true;
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.num_discs = D; cfg.vehicle_length = 4.0;

            AdaptiveScenarioMPC ctrl(cfg);
            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

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
                ctrl.update_mode_observation(0, osim.current_mode, i);

            bool had_collision = false;
            for (int step = 0; step < 50; ++step) {
                osim.maybe_switch(0.15, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
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
            cfg.weight_type = WeightType::WASSERSTEIN;
            cfg.safe_horizon_enabled = true;
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.forced_safe_horizon = ns;
            cfg.num_discs = 3; cfg.vehicle_length = 4.0;

            AdaptiveScenarioMPC ctrl(cfg);
            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

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
                ctrl.update_mode_observation(0, osim.current_mode, i);

            bool had_collision = false;
            for (int step = 0; step < 50; ++step) {
                osim.maybe_switch(0.15, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
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
        PaperVariant::BASE, PaperVariant::OT, PaperVariant::OT_SH
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
                cfg.weight_type = uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
                cfg.enable_dro = uses_dro(v);
                cfg.safe_horizon_enabled = uses_sh(v);
                cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
                cfg.num_discs = 1;

                AdaptiveScenarioMPC ctrl(cfg);
                OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

                std::map<std::string, ModeModel> omm;
                for (auto& m : modes) omm[m] = mode_mdls[m];
                ctrl.initialize_obstacle(0, omm);

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
                    ctrl.update_mode_observation(0, osim.current_mode, i);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }
                    osim.step(DT, rng);
                }

                // Test phase: use test_sp
                bool had_collision = false;
                double rollout_min_clear = 1e9;
                int missed = 0, steps = 0;
                for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                    osim.maybe_switch(cond.test_sp, rng);
                    ctrl.update_mode_observation(0, osim.current_mode, step + 20);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }

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
        {"OT",        SamplingBaseline::OT,             PaperVariant::OT},
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
// Experiment Q: OT Internal Ablation (parameter sensitivity)
// ============================================================================

static void run_experiment_q() {
    std::cout << "\n========================================\n"
              << "  Experiment Q: OT Internal Ablation\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 600;
    const double SWITCH_PROB = 0.2;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    // Vary OT predictor parameters one at a time from defaults:
    // default: history=200, reg=0.1, horizon=10, decay=1.0
    struct OTConfig {
        std::string name;
        int history_window;
        double regularization;
        int prediction_horizon;
        double decay;
    };
    std::vector<OTConfig> configs = {
        {"Default",       200, 0.1,  10, 1.0},
        {"ShortHist-20",   20, 0.1,  10, 1.0},
        {"LongHist-500",  500, 0.1,  10, 1.0},
        {"LowReg-0.01",   200, 0.01, 10, 1.0},
        {"HighReg-1.0",   200, 1.0,  10, 1.0},
        {"NoDec-0.0",     200, 0.1,  10, 0.0},
        {"ShortPred-3",   200, 0.1,   3, 1.0},
    };

    std::ofstream csv(OUTPUT_DIR + "exp_q_ot_ablation.csv");
    csv << "config,collision_rate,ci_lo,ci_hi,missed_mode_rate,avg_progress,avg_clearance,avg_solve_ms\n";

    auto mode_mdls = create_obstacle_mode_models(DT);

    for (const auto& oc : configs) {
        std::cout << "  " << oc.name << " ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        double sum_progress = 0, sum_clearance = 0, sum_solve = 0;

        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            unsigned seed = 90000 + r;
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::WASSERSTEIN;
            cfg.enable_dro = false;
            cfg.safe_horizon_enabled = false;
            cfg.num_discs = 1;

            AdaptiveScenarioMPC ctrl(cfg);
            OptimalTransportPredictor ot_pred(DT, oc.history_window, oc.regularization,
                                               oc.prediction_horizon, oc.decay,
                                               OTWeightType::WASSERSTEIN);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

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

            for (int i = 0; i < 5; ++i) {
                ctrl.update_mode_observation(0, osim.current_mode, i);
                ot_pred.observe(0, osim.state.position(), osim.current_mode);
                ot_pred.advance_timestep();
            }

            bool had_collision = false;
            double rollout_min_clear = 1e9;
            int missed = 0, steps = 0;
            double rollout_solve = 0;

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                osim.maybe_switch(SWITCH_PROB, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                ot_pred.observe(0, osim.state.position(), osim.current_mode);
                ot_pred.advance_timestep();

                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);
                rollout_solve += res.solve_time;

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
            sum_solve += rollout_solve / std::max(1, steps);
        }

        double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
        double mmr = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;

        csv << oc.name << "," << std::fixed << std::setprecision(4)
            << cr << "," << ci_lo << "," << ci_hi << ","
            << mmr << "," << sum_progress / NUM_ROLLOUTS << ","
            << sum_clearance / NUM_ROLLOUTS << ","
            << sum_solve / NUM_ROLLOUTS * 1000 << "\n";

        std::cout << "coll=" << std::setprecision(3) << cr
                  << " missed=" << mmr << std::endl;
    }
    csv.close();
    std::cout << "  -> exp_q_ot_ablation.csv\n";
}

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

    std::vector<PaperVariant> r_variants = {PaperVariant::BASE, PaperVariant::OT};

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
            cfg.weight_type = uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
            cfg.enable_dro = false;
            cfg.safe_horizon_enabled = false;
            cfg.num_discs = 1;

            AdaptiveScenarioMPC ctrl(cfg);
            OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

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
                ctrl.update_mode_observation(0, osim.current_mode, i);
                if (uses_ot(v)) {
                    ot_pred.observe(0, osim.state.position(), osim.current_mode);
                    ot_pred.advance_timestep();
                }
            }

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                osim.maybe_switch(SWITCH_PROB, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                if (uses_ot(v)) {
                    ot_pred.observe(0, osim.state.position(), osim.current_mode);
                    ot_pred.advance_timestep();
                }

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
              << "  Experiment T: Missed-Mode Rate vs S\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 800;
    const double SWITCH_PROB = 0.2;
    std::vector<int> scenario_counts = {10, 20, 40, 80, 160};
    std::vector<PaperVariant> t_variants = {
        PaperVariant::BASE, PaperVariant::OT, PaperVariant::OT_SH
    };

    std::ofstream csv(OUTPUT_DIR + "exp_t_missed_mode_vs_s.csv");
    csv << "variant,num_scenarios,collision_rate,ci_lo,ci_hi,missed_mode_rate,avg_progress\n";

    for (PaperVariant v : t_variants) {
        for (int S : scenario_counts) {
            std::cout << "  " << variant_name(v) << " S=" << S << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_steps_all = 0;
            double sum_progress = 0;

            for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                unsigned seed = 100000 + r;
                auto res = run_single_rollout(v, SWITCH_PROB, S, ROLLOUT_STEPS, seed);

                if (res.collision) collisions++;
                total_missed += res.missed_mode_steps;
                total_steps_all += res.total_steps;
                sum_progress += res.total_progress;
            }

            double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
            double mmr = total_steps_all > 0 ?
                static_cast<double>(total_missed) / total_steps_all : 0;

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
// Experiment U: Ground-Cost Ablation for OT Geometry
// ============================================================================

static void run_experiment_u() {
    std::cout << "\n========================================\n"
              << "  Experiment U: Ground-Cost Ablation\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 2000;
    const double SWITCH_PROB = 0.2;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    struct CostConfig {
        std::string name;
        GroundCostType cost_type;
    };
    std::vector<CostConfig> configs = {
        {"W2-Euclidean",  GroundCostType::SQUARED_EUCLIDEAN},
        {"L1-Manhattan",  GroundCostType::MANHATTAN},
        {"Flat-NoGeom",   GroundCostType::FLAT},
        {"Mean-Only",     GroundCostType::MEAN_ONLY},
        {"Directional",   GroundCostType::DIRECTIONAL},
    };

    std::ofstream csv(OUTPUT_DIR + "exp_u_ground_cost.csv");
    csv << "ground_cost,collision_rate,ci_lo,ci_hi,missed_mode_rate,avg_progress,avg_clearance\n";

    // Per-seed paired data for McNemar tests
    std::ofstream paired_csv(OUTPUT_DIR + "exp_u_paired.csv");
    paired_csv << "seed";
    for (const auto& gc : configs) paired_csv << "," << gc.name;
    paired_csv << "\n";

    auto mode_mdls = create_obstacle_mode_models(DT);

    // Store per-seed collision results
    std::map<std::string, std::vector<int>> per_seed_collisions;
    for (const auto& gc : configs) per_seed_collisions[gc.name].resize(NUM_ROLLOUTS, 0);

    for (const auto& gc : configs) {
        std::cout << "  " << gc.name << " (" << NUM_ROLLOUTS << " rollouts) ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        double sum_progress = 0, sum_clearance = 0;

        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            unsigned seed = 110000 + r;
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::WASSERSTEIN;
            cfg.enable_dro = false;
            cfg.safe_horizon_enabled = false;
            cfg.num_discs = 1;

            AdaptiveScenarioMPC ctrl(cfg);
            OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0,
                                               OTWeightType::WASSERSTEIN, gc.cost_type);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

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

            for (int i = 0; i < 5; ++i) {
                ctrl.update_mode_observation(0, osim.current_mode, i);
                ot_pred.observe(0, osim.state.position(), osim.current_mode);
                ot_pred.advance_timestep();
            }

            bool had_collision = false;
            double rollout_min_clear = 1e9;
            int missed = 0, steps = 0;

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                osim.maybe_switch(SWITCH_PROB, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                ot_pred.observe(0, osim.state.position(), osim.current_mode);
                ot_pred.advance_timestep();

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
            per_seed_collisions[gc.name][r] = had_collision ? 1 : 0;
            total_missed += missed;
            total_steps_all += steps;
            sum_progress += ego.x;
            sum_clearance += rollout_min_clear;
        }

        double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
        double mmr = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;

        csv << gc.name << "," << std::fixed << std::setprecision(4)
            << cr << "," << ci_lo << "," << ci_hi << ","
            << mmr << "," << sum_progress / NUM_ROLLOUTS << ","
            << sum_clearance / NUM_ROLLOUTS << "\n";

        std::cout << "coll=" << std::setprecision(3) << cr
                  << " missed=" << mmr << std::endl;
    }
    csv.close();

    // Write paired per-seed data
    for (int r = 0; r < NUM_ROLLOUTS; ++r) {
        paired_csv << (110000 + r);
        for (const auto& gc : configs) {
            paired_csv << "," << per_seed_collisions[gc.name][r];
        }
        paired_csv << "\n";
    }
    paired_csv.close();

    // McNemar tests: W2 vs each other
    std::cout << "\n  McNemar paired tests (W2-Euclidean vs others):\n";
    const auto& w2_results = per_seed_collisions["W2-Euclidean"];
    for (size_t c = 1; c < configs.size(); ++c) {
        const auto& other_results = per_seed_collisions[configs[c].name];
        int b = 0, mc_c = 0;
        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            if (w2_results[r] == 0 && other_results[r] == 1) b++;
            if (w2_results[r] == 1 && other_results[r] == 0) mc_c++;
        }
        double chi2 = 0;
        if (b + mc_c > 0) {
            chi2 = std::pow(std::abs(static_cast<double>(b) - mc_c) - 1.0, 2) / (b + mc_c);
        }
        double p_value = std::erfc(std::sqrt(chi2 / 2.0));
        std::cout << "    W2 vs " << configs[c].name
                  << ": b=" << b << " c=" << mc_c
                  << " chi2=" << std::setprecision(2) << chi2
                  << " p=" << std::setprecision(4) << p_value
                  << (p_value < 0.05 ? " *" : "") << std::endl;
    }

    std::cout << "  -> exp_u_ground_cost.csv\n";
    std::cout << "  -> exp_u_paired.csv\n";
}

// ============================================================================
// Experiment V: Rare-Mode Probability Sweep
// ============================================================================

static void run_experiment_v() {
    std::cout << "\n========================================\n"
              << "  Experiment V: Rare-Mode Probability Sweep\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 600;
    const double SWITCH_PROB = 0.2;
    std::vector<double> rare_probs = {0.01, 0.05, 0.10, 0.20};
    std::string rare_mode = "decelerating";
    std::vector<std::string> base_modes = {"constant_velocity", "turn_left", "turn_right"};

    std::vector<PaperVariant> v_variants = {
        PaperVariant::BASE, PaperVariant::OT, PaperVariant::OT_SH
    };

    std::ofstream csv(OUTPUT_DIR + "exp_v_rare_mode_sweep.csv");
    csv << "variant,rare_prob,collision_rate,ci_lo,ci_hi,"
        << "missed_mode_rate,rare_mode_missed_frac,avg_progress,avg_clearance\n";

    for (PaperVariant v : v_variants) {
        for (double rp : rare_probs) {
            std::cout << "  " << variant_name(v) << " rare_p=" << rp << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_steps_all = 0;
            int rare_mode_total = 0, rare_mode_missed = 0;
            double sum_progress = 0, sum_clearance = 0;

            for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                unsigned seed = 120000 + r;
                auto res = run_single_rollout(
                    v, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS, seed,
                    base_modes, rare_mode, rp,
                    uses_sh(v));

                if (res.collision) collisions++;
                total_missed += res.missed_mode_steps;
                total_steps_all += res.total_steps;
                sum_progress += res.total_progress;
                sum_clearance += res.min_clearance;
            }

            // We don't have per-mode breakdown from run_single_rollout,
            // so we run a targeted measurement for rare-mode miss rate
            int rare_missed_count = 0, rare_total_count = 0;
            auto mode_mdls = create_obstacle_mode_models(DT);
            for (int r = 0; r < 50; ++r) {
                unsigned seed = 125000 + r;
                std::mt19937 rng(seed);

                ScenarioMPCConfig cfg;
                cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
                cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
                cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
                cfg.weight_type = uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
                cfg.enable_dro = uses_dro(v);
                cfg.safe_horizon_enabled = uses_sh(v);
                cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
                cfg.num_discs = 1;

                AdaptiveScenarioMPC ctrl(cfg);
                OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

                std::vector<std::string> all_modes = base_modes;
                all_modes.push_back(rare_mode);
                std::map<std::string, ModeModel> omm;
                for (auto& m : all_modes)
                    if (mode_mdls.find(m) != mode_mdls.end()) omm[m] = mode_mdls[m];
                ctrl.initialize_obstacle(0, omm);

                ObstacleSim osim;
                std::uniform_real_distribution<double> jitter(-0.5, 0.5);
                osim.state = ObstacleState(3.0 + jitter(rng), 0.3 + jitter(rng) * 0.3,
                                           jitter(rng) * 0.2, jitter(rng) * 0.2);
                osim.current_mode = "constant_velocity";
                osim.available_modes = all_modes;
                osim.mode_models = omm;

                EgoState ego(0, 0, 0, 1.5);
                Eigen::Vector2d goal(20, 0);
                EgoDynamics dyn(DT);

                for (int i = 0; i < 5; ++i) {
                    ctrl.update_mode_observation(0, osim.current_mode, i);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }
                }

                for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                    // Rare mode switching
                    std::uniform_real_distribution<double> u(0, 1);
                    if (u(rng) < rp) {
                        osim.current_mode = rare_mode;
                    } else {
                        osim.maybe_switch(SWITCH_PROB, rng);
                    }

                    ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }

                    // Track rare mode specifically
                    if (osim.current_mode == rare_mode) {
                        rare_total_count++;
                        bool found = false;
                        for (const auto& sc : ctrl.scenarios()) {
                            for (const auto& [oid, traj] : sc.trajectories) {
                                if (oid == 0 && traj.mode_id == rare_mode) {
                                    found = true; break;
                                }
                            }
                            if (found) break;
                        }
                        if (!found) rare_missed_count++;
                    }

                    std::map<int, ObstacleState> obs_map;
                    obs_map[0] = osim.state;
                    auto res = ctrl.solve(ego, obs_map, goal, 1.5);

                    if (res.success && res.first_input().has_value())
                        ego = dyn.propagate(ego, res.first_input().value());
                    osim.step(DT, rng);
                }
            }

            double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
            double mmr = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;
            double rare_miss_frac = rare_total_count > 0 ?
                static_cast<double>(rare_missed_count) / rare_total_count : 0;

            csv << variant_name(v) << "," << rp << ","
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
    csv.close();
    std::cout << "  -> exp_v_rare_mode_sweep.csv\n";
}

// ============================================================================
// Experiment W: Scaling with Number of Modes M
// ============================================================================

static void run_experiment_w() {
    std::cout << "\n========================================\n"
              << "  Experiment W: Scaling with M Modes\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 600;
    const double SWITCH_PROB = 0.2;

    // M=2: {CV, TL}, M=3: {CV, TL, TR}, M=4: {CV, TL, TR, Dec},
    // M=6: {CV, TL, TR, Dec, LC_L, LC_R}
    struct MConfig {
        int M;
        std::vector<std::string> modes;
    };
    std::vector<MConfig> m_configs = {
        {2, {"constant_velocity", "turn_left"}},
        {3, {"constant_velocity", "turn_left", "turn_right"}},
        {4, {"constant_velocity", "turn_left", "turn_right", "decelerating"}},
        {6, {"constant_velocity", "turn_left", "turn_right", "decelerating",
             "lane_change_left", "lane_change_right"}},
    };

    std::vector<PaperVariant> w_variants = {
        PaperVariant::BASE, PaperVariant::OT, PaperVariant::OT_SH
    };

    std::ofstream csv(OUTPUT_DIR + "exp_w_mode_scaling.csv");
    csv << "variant,num_modes,collision_rate,ci_lo,ci_hi,"
        << "missed_mode_rate,avg_progress,avg_solve_ms\n";

    auto mode_mdls = create_obstacle_mode_models(DT);

    for (PaperVariant v : w_variants) {
        for (const auto& mc : m_configs) {
            std::cout << "  " << variant_name(v) << " M=" << mc.M << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_steps_all = 0;
            double sum_progress = 0, sum_solve = 0;

            for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                unsigned seed = 130000 + r;
                std::mt19937 rng(seed);

                ScenarioMPCConfig cfg;
                cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
                cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
                cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
                cfg.weight_type = uses_ot(v) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
                cfg.enable_dro = uses_dro(v);
                cfg.safe_horizon_enabled = uses_sh(v);
                cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
                cfg.num_discs = 1;

                AdaptiveScenarioMPC ctrl(cfg);
                OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

                std::map<std::string, ModeModel> omm;
                for (const auto& m : mc.modes) {
                    if (mode_mdls.find(m) != mode_mdls.end())
                        omm[m] = mode_mdls[m];
                }
                ctrl.initialize_obstacle(0, omm);

                ObstacleSim osim;
                std::uniform_real_distribution<double> jitter(-0.5, 0.5);
                osim.state = ObstacleState(3.0 + jitter(rng), 0.3 + jitter(rng) * 0.3,
                                           jitter(rng) * 0.2, jitter(rng) * 0.2);
                osim.current_mode = mc.modes[0];
                osim.available_modes = mc.modes;
                osim.mode_models = omm;

                EgoState ego(0, 0, 0, 1.5);
                Eigen::Vector2d goal(20, 0);
                EgoDynamics dyn(DT);
                double collision_radius = cfg.ego_radius + cfg.obstacle_radius;

                for (int i = 0; i < 5; ++i) {
                    ctrl.update_mode_observation(0, osim.current_mode, i);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }
                }

                bool had_collision = false;
                int missed = 0, steps = 0;
                double rollout_solve = 0;

                for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                    osim.maybe_switch(SWITCH_PROB, rng);
                    ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                    if (uses_ot(v)) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }

                    std::map<int, ObstacleState> obs_map;
                    obs_map[0] = osim.state;
                    auto res = ctrl.solve(ego, obs_map, goal, 1.5);
                    rollout_solve += res.solve_time;

                    double dist = (ego.position() - osim.state.position()).norm();
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
                sum_solve += rollout_solve / std::max(1, steps);
            }

            double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
            double mmr = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;

            csv << variant_name(v) << "," << mc.M << ","
                << std::fixed << std::setprecision(4)
                << cr << "," << ci_lo << "," << ci_hi << ","
                << mmr << "," << sum_progress / NUM_ROLLOUTS << ","
                << sum_solve / NUM_ROLLOUTS * 1000 << "\n";

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
              << "  Experiment X: Coverage Baselines vs OT (Rare-Mode Stress)\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 600;
    const double SWITCH_PROB = 0.2;
    std::vector<double> rare_probs = {0.01, 0.05, 0.10, 0.20};
    std::string rare_mode = "decelerating";
    std::vector<std::string> base_modes = {"constant_velocity", "turn_left", "turn_right"};
    std::vector<std::string> all_modes = base_modes;
    all_modes.push_back(rare_mode);

    EnvironmentSetup default_env;
    default_env.initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
    default_env.goal = Eigen::Vector2d(20.0, 0.0);
    default_env.obs_modes = all_modes;

    struct BaselineConfig {
        std::string name;
        SamplingBaseline baseline;
        PaperVariant variant;
    };
    std::vector<BaselineConfig> configs = {
        {"Standard",     SamplingBaseline::STANDARD,        PaperVariant::BASE},
        {"OT",           SamplingBaseline::OT,              PaperVariant::OT},
        {"OT+SH",        SamplingBaseline::OT,              PaperVariant::OT_SH},
        {"Uniform",      SamplingBaseline::UNIFORM_WEIGHT,  PaperVariant::BASE},
        {"Temperature",  SamplingBaseline::TEMPERATURE,     PaperVariant::BASE},
        {"Stratified",   SamplingBaseline::STRATIFIED,      PaperVariant::BASE},
        {"EpsGreedy",    SamplingBaseline::EPSILON_GREEDY,  PaperVariant::BASE},
    };

    std::ofstream csv(OUTPUT_DIR + "exp_x_baselines_rare_mode.csv");
    csv << "baseline,rare_prob,collision_rate,ci_lo,ci_hi,"
        << "missed_mode_rate,rare_mode_missed_frac,avg_progress\n";

    auto mode_mdls = create_obstacle_mode_models(DT);

    for (const auto& bc : configs) {
        for (double rp : rare_probs) {
            std::cout << "  " << bc.name << " rare_p=" << rp << " ... " << std::flush;

            int collisions = 0;
            int total_missed = 0, total_steps_all = 0;
            double sum_progress = 0;

            // Main collision/progress measurement
            for (int r = 0; r < NUM_ROLLOUTS; ++r) {
                unsigned seed = 140000 + r;
                std::mt19937 rng(seed);
                default_env.initial_obs = ObstacleState(
                    3.0 + std::uniform_real_distribution<double>(-0.5, 0.5)(rng),
                    0.3 + std::uniform_real_distribution<double>(-0.25, 0.25)(rng),
                    std::uniform_real_distribution<double>(-0.2, 0.2)(rng),
                    std::uniform_real_distribution<double>(-0.2, 0.2)(rng));

                auto res = run_single_rollout_env(
                    bc.variant, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS,
                    seed, default_env, bc.baseline);

                // We need to account for rare mode switching - re-run with rare mode logic
                // Actually use run_single_rollout which supports rare_mode parameter
                auto res2 = run_single_rollout(
                    bc.variant, SWITCH_PROB, BASE_SCENARIOS, ROLLOUT_STEPS, seed,
                    base_modes, rare_mode, rp, uses_sh(bc.variant));

                if (res2.collision) collisions++;
                total_missed += res2.missed_mode_steps;
                total_steps_all += res2.total_steps;
                sum_progress += res2.total_progress;
            }

            // Targeted rare-mode miss rate measurement
            int rare_missed_count = 0, rare_total_count = 0;
            for (int r = 0; r < 50; ++r) {
                unsigned seed = 145000 + r;
                std::mt19937 rng(seed);

                ScenarioMPCConfig cfg;
                cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
                cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
                cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
                cfg.num_discs = 1;

                // Set weight type based on baseline
                switch (bc.baseline) {
                    case SamplingBaseline::OT:
                        cfg.weight_type = WeightType::WASSERSTEIN; break;
                    case SamplingBaseline::UNIFORM_WEIGHT:
                        cfg.weight_type = WeightType::UNIFORM; break;
                    case SamplingBaseline::TEMPERATURE:
                        cfg.weight_type = WeightType::TEMPERATURE; break;
                    case SamplingBaseline::EPSILON_GREEDY:
                        cfg.weight_type = WeightType::EPSILON_GREEDY; break;
                    default:
                        cfg.weight_type = WeightType::FREQUENCY; break;
                }
                cfg.enable_dro = uses_dro(bc.variant);
                cfg.safe_horizon_enabled = uses_sh(bc.variant);
                cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;

                AdaptiveScenarioMPC ctrl(cfg);
                OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

                std::map<std::string, ModeModel> omm;
                for (auto& m : all_modes)
                    if (mode_mdls.find(m) != mode_mdls.end()) omm[m] = mode_mdls[m];
                ctrl.initialize_obstacle(0, omm);

                ObstacleSim osim;
                std::uniform_real_distribution<double> jitter(-0.5, 0.5);
                osim.state = ObstacleState(3.0 + jitter(rng), 0.3 + jitter(rng) * 0.3,
                                           jitter(rng) * 0.2, jitter(rng) * 0.2);
                osim.current_mode = "constant_velocity";
                osim.available_modes = all_modes;
                osim.mode_models = omm;

                EgoState ego(0, 0, 0, 1.5);
                Eigen::Vector2d goal(20, 0);
                EgoDynamics dyn(DT);

                bool use_ot = (bc.baseline == SamplingBaseline::OT);
                for (int i = 0; i < 5; ++i) {
                    ctrl.update_mode_observation(0, osim.current_mode, i);
                    if (use_ot) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }
                }

                for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                    std::uniform_real_distribution<double> u(0, 1);
                    if (u(rng) < rp) {
                        osim.current_mode = rare_mode;
                    } else {
                        osim.maybe_switch(SWITCH_PROB, rng);
                    }

                    ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                    if (use_ot) {
                        ot_pred.observe(0, osim.state.position(), osim.current_mode);
                        ot_pred.advance_timestep();
                    }

                    if (osim.current_mode == rare_mode) {
                        rare_total_count++;
                        bool found = false;
                        for (const auto& sc : ctrl.scenarios()) {
                            for (const auto& [oid, traj] : sc.trajectories) {
                                if (oid == 0 && traj.mode_id == rare_mode) {
                                    found = true; break;
                                }
                            }
                            if (found) break;
                        }
                        if (!found) rare_missed_count++;
                    }

                    std::map<int, ObstacleState> obs_map;
                    obs_map[0] = osim.state;
                    auto res = ctrl.solve(ego, obs_map, goal, 1.5);
                    if (res.success && res.first_input().has_value())
                        ego = dyn.propagate(ego, res.first_input().value());
                    osim.step(DT, rng);
                }
            }

            double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
            auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
            double mmr = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;
            double rare_miss_frac = rare_total_count > 0 ?
                static_cast<double>(rare_missed_count) / rare_total_count : 0;

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
// Experiment Y: Geometry Ablation with Shuffled/Random Cost Matrix
// ============================================================================

static void run_experiment_y() {
    std::cout << "\n========================================\n"
              << "  Experiment Y: Shuffled/Random Cost Matrix Ablation\n"
              << "========================================\n";

    // Use more rollouts and higher switch prob so geometry effect is detectable (McNemar significance).
    const int NUM_ROLLOUTS = 2000;
    const double SWITCH_PROB = 0.28;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};

    struct CostConfig {
        std::string name;
        GroundCostType cost_type;
    };
    std::vector<CostConfig> configs = {
        {"W2-Euclidean",     GroundCostType::SQUARED_EUCLIDEAN},
        {"Random-Permuted",  GroundCostType::RANDOM_PERMUTED},
        {"Constant",         GroundCostType::CONSTANT},
        {"Flat-NoGeom",      GroundCostType::FLAT},
        {"Mean-Only",        GroundCostType::MEAN_ONLY},
    };

    std::ofstream csv(OUTPUT_DIR + "exp_y_geometry_ablation.csv");
    csv << "ground_cost,collision_rate,ci_lo,ci_hi,missed_mode_rate,"
        << "avg_progress,avg_clearance,p99_solve_ms\n";

    // Per-seed paired data for McNemar tests
    std::ofstream paired_csv(OUTPUT_DIR + "exp_y_paired.csv");
    paired_csv << "seed";
    for (const auto& gc : configs) paired_csv << "," << gc.name;
    paired_csv << "\n";

    auto mode_mdls = create_obstacle_mode_models(DT);

    // Store per-seed collision results for paired testing
    std::map<std::string, std::vector<int>> per_seed_collisions;
    for (const auto& gc : configs) per_seed_collisions[gc.name].resize(NUM_ROLLOUTS, 0);

    for (const auto& gc : configs) {
        std::cout << "  " << gc.name << " (" << NUM_ROLLOUTS << " rollouts) ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        double sum_progress = 0, sum_clearance = 0;
        std::vector<double> all_solve_times;

        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            unsigned seed = 150000 + r;
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::WASSERSTEIN;
            cfg.enable_dro = false;
            cfg.safe_horizon_enabled = false;
            cfg.num_discs = 1;

            AdaptiveScenarioMPC ctrl(cfg);
            OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0,
                                               OTWeightType::WASSERSTEIN, gc.cost_type);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

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

            for (int i = 0; i < 5; ++i) {
                ctrl.update_mode_observation(0, osim.current_mode, i);
                ot_pred.observe(0, osim.state.position(), osim.current_mode);
                ot_pred.advance_timestep();
            }

            bool had_collision = false;
            double rollout_min_clear = 1e9;
            int missed = 0, steps = 0;

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                osim.maybe_switch(SWITCH_PROB, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                ot_pred.observe(0, osim.state.position(), osim.current_mode);
                ot_pred.advance_timestep();

                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);
                all_solve_times.push_back(res.solve_time * 1000);

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
            per_seed_collisions[gc.name][r] = had_collision ? 1 : 0;
            total_missed += missed;
            total_steps_all += steps;
            sum_progress += ego.x;
            sum_clearance += rollout_min_clear;
        }

        double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
        double mmr = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;
        double p99 = percentile(all_solve_times, 99);

        csv << gc.name << "," << std::fixed << std::setprecision(4)
            << cr << "," << ci_lo << "," << ci_hi << ","
            << mmr << "," << sum_progress / NUM_ROLLOUTS << ","
            << sum_clearance / NUM_ROLLOUTS << "," << p99 << "\n";

        std::cout << "coll=" << std::setprecision(3) << cr
                  << " missed=" << mmr
                  << " p99=" << std::setprecision(2) << p99 << "ms" << std::endl;
    }
    csv.close();

    // Write paired per-seed data
    for (int r = 0; r < NUM_ROLLOUTS; ++r) {
        paired_csv << (150000 + r);
        for (const auto& gc : configs) {
            paired_csv << "," << per_seed_collisions[gc.name][r];
        }
        paired_csv << "\n";
    }
    paired_csv.close();

    // Compute and print McNemar tests: W2 vs each other
    std::cout << "\n  McNemar paired tests (W2-Euclidean vs others):\n";
    const auto& w2_results = per_seed_collisions["W2-Euclidean"];
    for (size_t c = 1; c < configs.size(); ++c) {
        const auto& other_results = per_seed_collisions[configs[c].name];
        // McNemar contingency: b = W2 safe & other collision, c = W2 collision & other safe
        int b = 0, mc_c = 0;
        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            if (w2_results[r] == 0 && other_results[r] == 1) b++;
            if (w2_results[r] == 1 && other_results[r] == 0) mc_c++;
        }
        // McNemar chi-square (with continuity correction)
        double chi2 = 0;
        if (b + mc_c > 0) {
            chi2 = std::pow(std::abs(static_cast<double>(b) - mc_c) - 1.0, 2) / (b + mc_c);
        }
        // Approximate p-value from chi2(1) using complementary error function
        double p_value = std::erfc(std::sqrt(chi2 / 2.0));
        std::cout << "    W2 vs " << configs[c].name
                  << ": b=" << b << " c=" << mc_c
                  << " chi2=" << std::setprecision(2) << chi2
                  << " p=" << std::setprecision(4) << p_value
                  << (p_value < 0.05 ? " *" : "") << std::endl;
    }

    std::cout << "  -> exp_y_geometry_ablation.csv\n";
    std::cout << "  -> exp_y_paired.csv\n";
}

// ============================================================================
// Experiment Z: Qualitative Rollout Trajectories
// ============================================================================

static void run_experiment_z() {
    std::cout << "\n========================================\n"
              << "  Experiment Z: Qualitative Rollout Trajectories\n"
              << "========================================\n";

    const double SWITCH_PROB = 0.3;  // Higher switch for more drama
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    auto mode_mdls = create_obstacle_mode_models(DT);

    struct VariantConfig {
        std::string name;
        PaperVariant variant;
    };
    std::vector<VariantConfig> variant_configs = {
        {"Base",   PaperVariant::BASE},
        {"OT",     PaperVariant::OT},
        {"OT_SH",  PaperVariant::OT_SH},
    };

    // Find seeds that produce interesting scenarios (rare-mode collision for Base)
    // Use a fixed set of seeds for reproducibility
    std::vector<unsigned> showcase_seeds = {200042, 200117, 200203, 200289};

    std::ofstream csv(OUTPUT_DIR + "exp_z_qualitative_trajectories.csv");
    csv << "seed,variant,step,ego_x,ego_y,ego_theta,obs_x,obs_y,"
        << "obs_mode,collision,missed_mode,clearance\n";

    for (unsigned seed : showcase_seeds) {
        for (const auto& vc : variant_configs) {
            std::cout << "  seed=" << seed << " " << vc.name << " ... " << std::flush;

            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = uses_ot(vc.variant) ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
            cfg.enable_dro = uses_dro(vc.variant);
            cfg.safe_horizon_enabled = uses_sh(vc.variant);
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.num_discs = 1;

            AdaptiveScenarioMPC ctrl(cfg);
            OptimalTransportPredictor ot_pred(DT, 200, 0.1, 10, 1.0, OTWeightType::WASSERSTEIN);

            std::map<std::string, ModeModel> omm;
            for (auto& m : modes) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

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

            for (int i = 0; i < 5; ++i) {
                ctrl.update_mode_observation(0, osim.current_mode, i);
                if (uses_ot(vc.variant)) {
                    ot_pred.observe(0, osim.state.position(), osim.current_mode);
                    ot_pred.advance_timestep();
                }
            }

            bool had_collision = false;
            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                osim.maybe_switch(SWITCH_PROB, rng);
                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                if (uses_ot(vc.variant)) {
                    ot_pred.observe(0, osim.state.position(), osim.current_mode);
                    ot_pred.advance_timestep();
                }

                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);

                double dist = (ego.position() - osim.state.position()).norm();
                bool coll_step = dist < collision_radius;
                if (coll_step) had_collision = true;

                bool mode_found = false;
                for (const auto& sc : ctrl.scenarios()) {
                    for (const auto& [oid, traj] : sc.trajectories) {
                        if (oid == 0 && traj.mode_id == osim.current_mode) {
                            mode_found = true; break;
                        }
                    }
                    if (mode_found) break;
                }

                csv << seed << "," << vc.name << "," << step << ","
                    << std::fixed << std::setprecision(4)
                    << ego.x << "," << ego.y << "," << ego.theta << ","
                    << osim.state.x << "," << osim.state.y << ","
                    << osim.current_mode << ","
                    << (coll_step ? 1 : 0) << "," << (mode_found ? 0 : 1) << ","
                    << dist << "\n";

                if (res.success && res.first_input().has_value())
                    ego = dyn.propagate(ego, res.first_input().value());
                osim.step(DT, rng);
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

static void run_experiment_aa() {
    std::cout << "\n========================================\n"
              << "  Experiment AA: Robustness Across Environments\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 350;
    const double SWITCH_PROB = 0.2;

    std::vector<EnvironmentType> envs = {
        EnvironmentType::STRAIGHT, EnvironmentType::NARROW_CORRIDOR,
        EnvironmentType::INTERSECTION, EnvironmentType::ONCOMING
    };

    std::vector<PaperVariant> variants = {
        PaperVariant::BASE, PaperVariant::OT, PaperVariant::OT_SH
    };

    // Output per-seed data for boxplots
    std::ofstream csv(OUTPUT_DIR + "exp_aa_robustness_per_seed.csv");
    csv << "environment,variant,seed,collision,missed_mode_rate,"
        << "progress,min_clearance,mean_solve_ms,p99_solve_ms\n";

    for (EnvironmentType env_type : envs) {
        for (PaperVariant v : variants) {
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

static void run_experiment_ab() {
    std::cout << "\n========================================\n"
              << "  Experiment AB: OT Regularization Pareto Frontier\n"
              << "========================================\n";

    const int NUM_ROLLOUTS = 600;
    const double SWITCH_PROB = 0.2;
    std::vector<std::string> modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    auto mode_mdls = create_obstacle_mode_models(DT);

    // 2D grid: sinkhorn_epsilon x uncertainty_scale
    std::vector<double> epsilons = {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0};
    std::vector<double> scales = {0.2, 0.5, 1.0, 2.0, 5.0};

    // Also test with/without SH
    struct ABConfig {
        std::string label;
        double epsilon;
        double scale;
        bool use_sh;
    };

    std::vector<ABConfig> configs;
    // Full grid for OT (no SH)
    for (double eps : epsilons) {
        for (double sc : scales) {
            std::string label = "OT_e" + std::to_string(eps).substr(0, 5)
                              + "_s" + std::to_string(sc).substr(0, 3);
            configs.push_back({label, eps, sc, false});
        }
    }
    // Key points with SH
    for (double eps : {0.01, 0.1, 1.0}) {
        for (double sc : {0.5, 1.0, 2.0}) {
            std::string label = "OT+SH_e" + std::to_string(eps).substr(0, 5)
                              + "_s" + std::to_string(sc).substr(0, 3);
            configs.push_back({label, eps, sc, true});
        }
    }

    std::ofstream csv(OUTPUT_DIR + "exp_ab_pareto_frontier.csv");
    csv << "label,epsilon,uncertainty_scale,use_sh,"
        << "collision_rate,ci_lo,ci_hi,missed_mode_rate,"
        << "rare_mode_missed_frac,avg_progress,avg_clearance,mean_solve_ms,p99_solve_ms\n";

    std::string rare_mode = "decelerating";
    std::vector<std::string> base_modes = {"constant_velocity", "turn_left", "turn_right"};
    double rare_prob = 0.05;  // Fixed rare-mode probability for stress

    int config_idx = 0;
    for (const auto& ac : configs) {
        config_idx++;
        std::cout << "  [" << config_idx << "/" << configs.size() << "] "
                  << ac.label << " ... " << std::flush;

        int collisions = 0;
        int total_missed = 0, total_steps_all = 0;
        int rare_total = 0, rare_missed = 0;
        double sum_progress = 0, sum_clearance = 0;
        std::vector<double> all_solve_times;

        for (int r = 0; r < NUM_ROLLOUTS; ++r) {
            unsigned seed = 170000 + r;
            std::mt19937 rng(seed);

            ScenarioMPCConfig cfg;
            cfg.horizon = HORIZON; cfg.dt = DT; cfg.num_scenarios = BASE_SCENARIOS;
            cfg.ego_radius = 0.5; cfg.obstacle_radius = 0.35; cfg.safety_margin = 0.2;
            cfg.use_sqp_solver = true; cfg.ensure_mode_coverage = true;
            cfg.weight_type = WeightType::WASSERSTEIN;
            cfg.enable_dro = false;
            cfg.safe_horizon_enabled = ac.use_sh;
            cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
            cfg.num_discs = 1;

            AdaptiveScenarioMPC ctrl(cfg);
            OptimalTransportPredictor ot_pred(DT, 200, ac.epsilon, 10, ac.scale,
                                               OTWeightType::WASSERSTEIN);

            std::vector<std::string> all_modes = base_modes;
            all_modes.push_back(rare_mode);
            std::map<std::string, ModeModel> omm;
            for (auto& m : all_modes)
                if (mode_mdls.find(m) != mode_mdls.end()) omm[m] = mode_mdls[m];
            ctrl.initialize_obstacle(0, omm);

            ObstacleSim osim;
            std::uniform_real_distribution<double> jitter(-0.5, 0.5);
            osim.state = ObstacleState(3.0 + jitter(rng), 0.3 + jitter(rng) * 0.3,
                                       jitter(rng) * 0.2, jitter(rng) * 0.2);
            osim.current_mode = "constant_velocity";
            osim.available_modes = all_modes;
            osim.mode_models = omm;

            EgoState ego(0, 0, 0, 1.5);
            Eigen::Vector2d goal(20, 0);
            EgoDynamics dyn(DT);
            double collision_radius = cfg.ego_radius + cfg.obstacle_radius;

            for (int i = 0; i < 5; ++i) {
                ctrl.update_mode_observation(0, osim.current_mode, i);
                ot_pred.observe(0, osim.state.position(), osim.current_mode);
                ot_pred.advance_timestep();
            }

            bool had_collision = false;
            double rollout_min_clear = 1e9;
            int missed = 0, steps = 0;

            for (int step = 0; step < ROLLOUT_STEPS; ++step) {
                std::uniform_real_distribution<double> u(0, 1);
                if (u(rng) < rare_prob) {
                    osim.current_mode = rare_mode;
                } else {
                    osim.maybe_switch(SWITCH_PROB, rng);
                }

                ctrl.update_mode_observation(0, osim.current_mode, step + 5);
                ot_pred.observe(0, osim.state.position(), osim.current_mode);
                ot_pred.advance_timestep();

                std::map<int, ObstacleState> obs_map;
                obs_map[0] = osim.state;
                auto res = ctrl.solve(ego, obs_map, goal, 1.5);
                all_solve_times.push_back(res.solve_time * 1000);

                double dist = (ego.position() - osim.state.position()).norm();
                rollout_min_clear = std::min(rollout_min_clear, dist);
                if (dist < collision_radius) had_collision = true;

                // Rare-mode tracking
                if (osim.current_mode == rare_mode) {
                    rare_total++;
                    bool found = false;
                    for (const auto& sc : ctrl.scenarios()) {
                        for (const auto& [oid, traj] : sc.trajectories) {
                            if (oid == 0 && traj.mode_id == rare_mode) {
                                found = true; break;
                            }
                        }
                        if (found) break;
                    }
                    if (!found) rare_missed++;
                }

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

        double cr = static_cast<double>(collisions) / NUM_ROLLOUTS;
        auto [ci_lo, ci_hi] = wilson_ci(collisions, NUM_ROLLOUTS);
        double mmr = total_steps_all > 0 ? static_cast<double>(total_missed) / total_steps_all : 0;
        double rare_miss_frac = rare_total > 0 ? static_cast<double>(rare_missed) / rare_total : 0;
        double mean_solve = all_solve_times.empty() ? 0 :
            std::accumulate(all_solve_times.begin(), all_solve_times.end(), 0.0) / all_solve_times.size();
        double p99 = percentile(all_solve_times, 99);

        csv << ac.label << "," << ac.epsilon << "," << ac.scale << ","
            << (ac.use_sh ? 1 : 0) << ","
            << std::fixed << std::setprecision(4)
            << cr << "," << ci_lo << "," << ci_hi << ","
            << mmr << "," << rare_miss_frac << ","
            << sum_progress / NUM_ROLLOUTS << ","
            << sum_clearance / NUM_ROLLOUTS << ","
            << mean_solve << "," << p99 << "\n";

        std::cout << "coll=" << std::setprecision(3) << cr
                  << " missed=" << mmr << std::endl;
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
              << "  Paper Experiments (New Framework): OT Mode Learning for SMPC\n"
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
    if (should_run("Q")) run_experiment_q();
    if (should_run("R")) run_experiment_r();
    if (should_run("T")) run_experiment_t();
    if (should_run("U")) run_experiment_u();
    if (should_run("V")) run_experiment_v();
    if (should_run("W")) run_experiment_w();
    if (should_run("X")) run_experiment_x();
    if (should_run("Y")) run_experiment_y();
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
