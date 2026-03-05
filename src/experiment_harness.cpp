/**
 * @file experiment_harness.cpp
 * @brief Implementation of experiment harness: rollout runner, stats, CSV.
 */

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "collision_constraints.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "adaptive_dro_shift.hpp"
#include "runtime_assurance.hpp"
#include "conformal_safety.hpp"
#include "hazard_switch_sampling.hpp"
#include "risk_directed_bandit.hpp"
#include "certificate_first.hpp"
#include "scenario_compiler.hpp"
#include "gan_scenario_loader.hpp"
#include "scenario_pruning.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <functional>

namespace scenario_mpc {

// ============================================================================
// CSV Writer
// ============================================================================

CSVWriter::CSVWriter(const std::string& filepath) : ofs_(filepath) {}

CSVWriter::~CSVWriter() {
    if (ofs_.is_open()) ofs_.close();
}

void CSVWriter::write_header() {
    ofs_ << "seed,method,scenario,S,eps_wass,sigma,shift_rho,shift_boost,ground_cost,"
         << "collision,collision_step,min_clearance,min_clearance_step,"
         << "total_progress,control_effort,constraint_active_count,"
         << "missed_mode_steps,total_steps,"
         << "avg_solve_ms,p50_solve_ms,p95_solve_ms,max_solve_ms,"
         << "total_dro_injected,avg_safe_horizon,clearance_5pct\n";
}

void CSVWriter::write_record(const RolloutRecord& rec) {
    ofs_ << rec.seed << "," << rec.method << "," << rec.scenario << "," << rec.S << ","
         << std::fixed << std::setprecision(4)
         << rec.eps_wass << "," << rec.sigma << ","
         << rec.shift_rho << "," << rec.shift_boost << "," << rec.ground_cost << ","
         << (rec.collision ? 1 : 0) << "," << rec.collision_step << ","
         << std::setprecision(4) << rec.min_clearance << "," << rec.min_clearance_step << ","
         << std::setprecision(4) << rec.total_progress << "," << rec.control_effort << ","
         << rec.constraint_active_count << ","
         << rec.missed_mode_steps << "," << rec.total_steps << ","
         << std::setprecision(4) << rec.avg_solve_ms << "," << rec.p50_solve_ms << ","
         << rec.p95_solve_ms << "," << rec.max_solve_ms << ","
         << rec.total_dro_injected << ","
         << std::setprecision(4) << rec.avg_safe_horizon << "," << rec.clearance_5pct << "\n";
}

void CSVWriter::flush() {
    ofs_.flush();
}

// ============================================================================
// Statistical Helpers
// ============================================================================

std::pair<double, double> wilson_ci(int successes, int n, double z) {
    if (n == 0) return {0.0, 1.0};
    double p_hat = static_cast<double>(successes) / n;
    double denom = 1.0 + z * z / n;
    double center = (p_hat + z * z / (2.0 * n)) / denom;
    double half_width = z * std::sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom;
    return {std::max(0.0, center - half_width), std::min(1.0, center + half_width)};
}

BootstrapResult bootstrap_paired_delta(
    const std::vector<bool>& base_collisions,
    const std::vector<bool>& dro_collisions,
    int n_bootstrap,
    std::mt19937* rng
) {
    int n = static_cast<int>(base_collisions.size());
    assert(n == static_cast<int>(dro_collisions.size()));
    assert(n > 0);

    std::mt19937 local_rng(42);
    if (!rng) rng = &local_rng;
    std::uniform_int_distribution<int> idx_dist(0, n - 1);

    std::vector<double> deltas(n_bootstrap);
    for (int b = 0; b < n_bootstrap; ++b) {
        int base_sum = 0, dro_sum = 0;
        for (int i = 0; i < n; ++i) {
            int j = idx_dist(*rng);
            if (base_collisions[j]) base_sum++;
            if (dro_collisions[j]) dro_sum++;
        }
        double p_base = static_cast<double>(base_sum) / n;
        double p_dro = static_cast<double>(dro_sum) / n;
        deltas[b] = p_base - p_dro;
    }

    std::sort(deltas.begin(), deltas.end());
    double mean_delta = std::accumulate(deltas.begin(), deltas.end(), 0.0) / n_bootstrap;
    double ci_low = deltas[static_cast<int>(0.025 * n_bootstrap)];
    double ci_high = deltas[static_cast<int>(0.975 * n_bootstrap)];

    return {mean_delta, ci_low, ci_high};
}

double mcnemar_chi2(int b, int c) {
    if (b + c == 0) return 0.0;
    double num = std::abs(static_cast<double>(b) - c) - 1.0;  // continuity correction
    num = std::max(0.0, num);
    return (num * num) / (b + c);
}

EffectSizes compute_effect_sizes(double p_base, double p_dro) {
    EffectSizes es;
    es.abs_delta = p_base - p_dro;
    es.rel_delta = (p_base > 1e-12) ? (p_base - p_dro) / p_base : 0.0;
    es.risk_ratio = (p_base > 1e-12) ? p_dro / p_base : 0.0;
    es.cohens_h = 2.0 * std::asin(std::sqrt(p_base)) - 2.0 * std::asin(std::sqrt(p_dro));
    return es;
}

// ============================================================================
// Seed Derivation
// ============================================================================

SeedBundle derive_seeds(unsigned master_seed, int idx) {
    // Hash-based derivation for reproducibility
    auto hash = [](unsigned a, unsigned b) -> unsigned {
        // FNV-1a style mixing
        unsigned h = 2166136261u;
        h ^= a; h *= 16777619u;
        h ^= b; h *= 16777619u;
        return h;
    };

    SeedBundle sb;
    sb.master = master_seed;
    sb.env = hash(master_seed, static_cast<unsigned>(idx * 3 + 0));
    sb.predictor = hash(master_seed, static_cast<unsigned>(idx * 3 + 1));
    sb.scenario = hash(master_seed, static_cast<unsigned>(idx * 3 + 2));
    return sb;
}

// ============================================================================
// Obstacle Simulator (internal)
// ============================================================================

namespace {

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

/// Apply distribution shift to mode observations
void apply_distribution_shift(
    const DistributionShiftConfig& shift,
    ObstacleSim& obs_sim,
    std::mt19937& rng
) {
    if (shift.rho <= 0.0 && shift.dangerous_boost <= 0.0) return;

    std::uniform_real_distribution<double> u(0, 1);

    // With probability rho, switch to a uniform-random mode
    if (shift.rho > 0.0 && u(rng) < shift.rho) {
        if (!obs_sim.available_modes.empty()) {
            std::uniform_int_distribution<int> idx(0, static_cast<int>(obs_sim.available_modes.size()) - 1);
            obs_sim.current_mode = obs_sim.available_modes[idx(rng)];
        }
    }

    // With probability dangerous_boost, force to the boosted mode
    if (shift.dangerous_boost > 0.0 && u(rng) < shift.dangerous_boost) {
        int boost_idx = shift.boosted_mode;
        if (boost_idx < 0) {
            // Default: last mode (typically the rare/dangerous one)
            boost_idx = static_cast<int>(obs_sim.available_modes.size()) - 1;
        }
        if (boost_idx >= 0 && boost_idx < static_cast<int>(obs_sim.available_modes.size())) {
            obs_sim.current_mode = obs_sim.available_modes[boost_idx];
        }
    }
}

/// Configure MPC controller for an ablation variant
void configure_ablation(
    ScenarioMPCConfig& config,
    DROConfig& dro_cfg,
    AblationVariant variant,
    double eps_wass,
    double sigma_scale
) {
    switch (variant) {
        case AblationVariant::NO_INJECTION:
            config.enable_dro = false;
            break;
        case AblationVariant::DRO_FULL:
            config.enable_dro = true;
            dro_cfg.risk_mode = DRORiskMode::FULL;
            dro_cfg.epsilon_base = eps_wass;
            dro_cfg.risk_sigma_scale = sigma_scale;
            break;
        case AblationVariant::DRO_NO_COV:
            config.enable_dro = true;
            dro_cfg.risk_mode = DRORiskMode::NO_COV;
            dro_cfg.epsilon_base = eps_wass;
            dro_cfg.risk_sigma_scale = 0.0;
            break;
        case AblationVariant::DRO_DISTANCE_ONLY:
            config.enable_dro = true;
            dro_cfg.risk_mode = DRORiskMode::DISTANCE_ONLY;
            dro_cfg.epsilon_base = eps_wass;
            dro_cfg.risk_sigma_scale = 0.0;
            break;
        case AblationVariant::RANDOM_INJECTION:
            // We handle random injection in the rollout loop
            config.enable_dro = false;
            break;
        case AblationVariant::ALWAYS_INJECT:
            // We handle always-inject in the rollout loop
            config.enable_dro = false;
            break;
    }
}

}  // anonymous namespace

// ============================================================================
// Rollout Runner
// ============================================================================

RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config,
    unsigned seed
) {
    std::mt19937 rng(seed);
    RolloutRecord rec;
    rec.seed = seed;
    rec.method = ablation_variant_name(config.ablation);
    rec.S = config.num_scenarios;
    rec.eps_wass = config.eps_wass;
    rec.sigma = config.sigma_scale;
    rec.shift_rho = config.shift.rho;
    rec.shift_boost = config.shift.dangerous_boost;
    rec.ground_cost = ground_cost_name(config.ground_cost);

    constexpr double DT = 0.1;
    auto mode_models = create_obstacle_mode_models(DT);

    // Configure MPC
    ScenarioMPCConfig mpc_cfg;
    mpc_cfg.horizon = config.horizon;
    mpc_cfg.dt = DT;
    mpc_cfg.num_scenarios = config.num_scenarios;
    mpc_cfg.ego_radius = 0.5;
    mpc_cfg.obstacle_radius = 0.35;
    mpc_cfg.safety_margin = 0.2;
    mpc_cfg.use_sqp_solver = true;
    mpc_cfg.ensure_mode_coverage = true;
    mpc_cfg.num_discs = config.num_discs;
    mpc_cfg.safe_horizon_enabled = config.safe_horizon_enabled;
    mpc_cfg.safe_horizon_min = 3;
    mpc_cfg.weight_type = WeightType::FREQUENCY;

    // Configure DRO
    DROConfig dro_cfg;
    dro_cfg.ground_cost_type = config.ground_cost;
    configure_ablation(mpc_cfg, dro_cfg, config.ablation,
                       config.eps_wass, config.sigma_scale);

    if (mpc_cfg.enable_dro) {
        mpc_cfg.dro_epsilon_base = dro_cfg.epsilon_base;
        mpc_cfg.dro_risk_sigma_scale = dro_cfg.risk_sigma_scale;
        mpc_cfg.dro_adaptive_epsilon = dro_cfg.adaptive_epsilon;
    }

    AdaptiveScenarioMPC controller(mpc_cfg);

    // Setup obstacle
    int obs_id = 0;
    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : config.obs_modes) {
        if (mode_models.find(m) != mode_models.end())
            obs_mode_models[m] = mode_models[m];
    }
    if (!config.rare_mode.empty() && mode_models.find(config.rare_mode) != mode_models.end()) {
        obs_mode_models[config.rare_mode] = mode_models[config.rare_mode];
    }
    controller.initialize_obstacle(obs_id, obs_mode_models);

    ObstacleSim obs_sim;
    std::uniform_real_distribution<double> y_dist(-0.5, 0.5);
    obs_sim.state = ObstacleState(3.0 + y_dist(rng), 0.3 + y_dist(rng) * 0.5,
                                   -0.1, y_dist(rng) * 0.3);
    obs_sim.current_mode = config.obs_modes.empty() ? "constant_velocity" : config.obs_modes[0];
    obs_sim.available_modes = config.obs_modes;
    if (!config.rare_mode.empty()) obs_sim.available_modes.push_back(config.rare_mode);
    obs_sim.mode_models = obs_mode_models;

    EgoState ego(0.0, 0.0, 0.0, 1.5);
    Eigen::Vector2d goal(20.0, 0.0);
    EgoDynamics dynamics(DT);
    double collision_radius = mpc_cfg.ego_radius + mpc_cfg.obstacle_radius;

    // Initial mode observations
    for (int i = 0; i < 5; ++i) {
        controller.update_mode_observation(obs_id, obs_sim.current_mode, i);
    }

    std::vector<double> solve_times;
    std::vector<double> clearances;
    std::vector<int> safe_horizons;
    double control_effort = 0.0;
    int constraint_active_total = 0;

    for (int step = 0; step < config.rollout_steps; ++step) {
        // Mode switching
        if (!config.rare_mode.empty() && config.rare_switch_prob > 0) {
            std::uniform_real_distribution<double> u(0, 1);
            if (u(rng) < config.rare_switch_prob) {
                obs_sim.current_mode = config.rare_mode;
            } else {
                obs_sim.maybe_switch(config.switch_prob, rng);
            }
        } else {
            obs_sim.maybe_switch(config.switch_prob, rng);
        }

        // Apply distribution shift
        apply_distribution_shift(config.shift, obs_sim, rng);

        controller.update_mode_observation(obs_id, obs_sim.current_mode, step + 5);

        std::map<int, ObstacleState> obstacles;
        obstacles[obs_id] = obs_sim.state;

        auto mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        solve_times.push_back(mpc_result.solve_time * 1000.0);  // ms
        rec.total_dro_injected += mpc_result.num_dro_injected;
        if (mpc_result.safe_horizon > 0)
            safe_horizons.push_back(mpc_result.safe_horizon);

        // Collision check
        double dist = (ego.position() - obs_sim.state.position()).norm();
        clearances.push_back(dist);
        if (dist < rec.min_clearance) {
            rec.min_clearance = dist;
            rec.min_clearance_step = step;
        }
        if (dist < collision_radius && !rec.collision) {
            rec.collision = true;
            rec.collision_step = step;
        }

        // Constraint active count
        constraint_active_total += static_cast<int>(mpc_result.active_scenarios.size());

        // Missed mode check
        bool mode_found = false;
        for (const auto& sc : controller.scenarios()) {
            for (const auto& [oid, traj] : sc.trajectories) {
                if (oid == obs_id && traj.mode_id == obs_sim.current_mode) {
                    mode_found = true; break;
                }
            }
            if (mode_found) break;
        }
        if (!mode_found) rec.missed_mode_steps++;

        // Apply control
        if (mpc_result.success && mpc_result.first_input().has_value()) {
            auto input = mpc_result.first_input().value();
            control_effort += input.a * input.a + input.delta * input.delta;
            ego = dynamics.propagate(ego, input);
        }
        obs_sim.step(DT, rng);
        rec.total_steps++;
    }

    // Compute summary metrics
    rec.total_progress = ego.x;
    rec.control_effort = control_effort;
    rec.constraint_active_count = constraint_active_total;

    // Solve time stats
    if (!solve_times.empty()) {
        rec.avg_solve_ms = std::accumulate(solve_times.begin(), solve_times.end(), 0.0) / solve_times.size();
        std::vector<double> sorted_times = solve_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        int n = static_cast<int>(sorted_times.size());
        rec.p50_solve_ms = sorted_times[n / 2];
        rec.p95_solve_ms = sorted_times[std::min(n - 1, static_cast<int>(0.95 * n))];
        rec.max_solve_ms = sorted_times.back();
    }

    // Safe horizon average
    if (!safe_horizons.empty()) {
        rec.avg_safe_horizon = std::accumulate(safe_horizons.begin(), safe_horizons.end(), 0.0) / safe_horizons.size();
    }

    // Clearance 5th percentile
    if (!clearances.empty()) {
        std::vector<double> sorted_clear = clearances;
        std::sort(sorted_clear.begin(), sorted_clear.end());
        int idx_5pct = std::max(0, static_cast<int>(0.05 * sorted_clear.size()) - 1);
        rec.clearance_5pct = sorted_clear[idx_5pct];
    }

    return rec;
}

/// Build a scenario with a single deterministic (mean) trajectory for one mode (for scenario compiler).
Scenario build_scenario_for_mode(
    int scenario_id,
    int obstacle_id,
    const ObstacleState& obs_state,
    const std::string& mode_id,
    const std::map<std::string, ModeModel>& mode_models,
    int horizon
) {
    auto it = mode_models.find(mode_id);
    if (it == mode_models.end()) return Scenario(scenario_id, {}, 0.0);
    const ModeModel& mode = it->second;
    std::vector<PredictionStep> steps;
    steps.reserve(static_cast<size_t>(horizon + 1));
    Eigen::Vector4d x = obs_state.to_array();
    steps.emplace_back(0, x.head<2>(), Eigen::Matrix2d::Zero());
    for (int k = 0; k < horizon; ++k) {
        x = mode.A * x + mode.b;
        steps.emplace_back(k + 1, x.head<2>(), Eigen::Matrix2d::Zero());
    }
    ObstacleTrajectory traj(obstacle_id, mode_id, steps, 1.0);
    std::map<int, ObstacleTrajectory> trajs;
    trajs[obstacle_id] = traj;
    return Scenario(scenario_id, trajs, 1.0);
}

// ============================================================================
// Future SL MPC extended rollout (SHMPC baseline vs extensions)
// ============================================================================

RolloutRecord run_experiment_rollout_future_sl(
    const ExperimentConfig& config,
    unsigned seed,
    const std::string& method
) {
    std::mt19937 rng(seed);
    RolloutRecord rec;
    rec.seed = seed;
    rec.method = method;
    rec.scenario = config.scenario_tag;
    rec.S = config.num_scenarios;
    rec.eps_wass = config.eps_wass;
    rec.sigma = config.sigma_scale;
    rec.shift_rho = config.shift.rho;
    rec.shift_boost = config.shift.dangerous_boost;
    rec.ground_cost = ground_cost_name(config.ground_cost);

    constexpr double DT = 0.1;
    auto mode_models = create_obstacle_mode_models(DT);

    ScenarioMPCConfig mpc_cfg;
    mpc_cfg.horizon = config.horizon;
    mpc_cfg.dt = DT;
    mpc_cfg.num_scenarios = config.num_scenarios;
    mpc_cfg.ego_radius = 0.5;
    mpc_cfg.obstacle_radius = 0.35;
    mpc_cfg.safety_margin = 0.2;
    mpc_cfg.use_sqp_solver = true;
    mpc_cfg.ensure_mode_coverage = true;
    mpc_cfg.num_discs = config.num_discs;
    mpc_cfg.safe_horizon_enabled = true;  // SHMPC for all variants
    mpc_cfg.safe_horizon_min = 3;
    mpc_cfg.weight_type = WeightType::FREQUENCY;

    bool use_dro = (method == "SHMPC_DRO" || method == "SHMPC_AdaptiveDRO");
    mpc_cfg.enable_dro = use_dro;
    if (use_dro) {
        mpc_cfg.dro_epsilon_base = config.eps_wass;
        mpc_cfg.dro_risk_sigma_scale = config.sigma_scale;
    }

    AdaptiveScenarioMPC controller(mpc_cfg);

    int obs_id = 0;
    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : config.obs_modes) {
        if (mode_models.find(m) != mode_models.end())
            obs_mode_models[m] = mode_models[m];
    }
    if (!config.rare_mode.empty() && mode_models.find(config.rare_mode) != mode_models.end()) {
        obs_mode_models[config.rare_mode] = mode_models[config.rare_mode];
    }
    controller.initialize_obstacle(obs_id, obs_mode_models);

    ObstacleSim obs_sim;
    std::uniform_real_distribution<double> y_dist(-0.5, 0.5);
    obs_sim.state = ObstacleState(3.0 + y_dist(rng), 0.3 + y_dist(rng) * 0.5,
                                  -0.1, y_dist(rng) * 0.3);
    obs_sim.current_mode = config.obs_modes.empty() ? "constant_velocity" : config.obs_modes[0];
    obs_sim.available_modes = config.obs_modes;
    if (!config.rare_mode.empty()) obs_sim.available_modes.push_back(config.rare_mode);
    obs_sim.mode_models = obs_mode_models;

    EgoState ego(0.0, 0.0, 0.0, 1.5);
    Eigen::Vector2d goal(20.0, 0.0);
    EgoDynamics dynamics(DT);
    double collision_radius = mpc_cfg.ego_radius + mpc_cfg.obstacle_radius;

    AdaptiveDROShift adaptive_shift(0.01, 1.0, 0.5, 20);
    const double rta_threshold = config.rta_threshold_override > 0 ? config.rta_threshold_override : 1.5;
    auto rta_monitor = [&](const EgoState& st, const std::optional<double>&) {
        std::map<int, ObstacleState> ob;
        ob[obs_id] = obs_sim.state;
        return simple_ttc_monitor(st, ob, rta_threshold);
    };
    RuntimeAssurance rta(rta_monitor, EgoInput(-2.0, 0.0));

    // Extension state: hazard, bandit, certificate, compiler
    HazardModel hazard_model;
    std::map<std::string, double> hazard_scales;
    for (const auto& m : obs_sim.available_modes)
        hazard_scales[m] = (m == config.rare_mode) ? 2.0 : 1.0;
    hazard_model.set_mode_coefficients(hazard_scales);
    const double bandit_beta = config.bandit_beta_override > 0 ? config.bandit_beta_override : 1.0;
    RiskDirectedBandit bandit(bandit_beta);
    bandit.set_modes(std::vector<std::string>(obs_sim.available_modes.begin(), obs_sim.available_modes.end()));
    int last_switch_step = -10;
    std::string prev_mode = obs_sim.current_mode;
    bool prev_collision = false;
    const double cert_radius = config.certificate_radius_override > 0 ? config.certificate_radius_override : 0.15;
    const int compiler_initial_S = 5;
    const int compiler_max_iter = 5;

    // Double-dual: persist mode weights from previous step's active (binding) scenarios
    std::map<int, std::map<std::string, double>> dual_mode_weights;

    for (int i = 0; i < 5; ++i) {
        controller.update_mode_observation(obs_id, obs_sim.current_mode, i);
    }

    std::vector<double> solve_times;
    std::vector<double> clearances;
    std::vector<int> safe_horizons;
    double control_effort = 0.0;
    int constraint_active_total = 0;

    // GAN cache: load CSV once per rollout for GAN variants (avoids file I/O every step)
    std::optional<GANScenarioCache> gan_cache;
    if ((method == "SHMPC_GAN" || method == "SHMPC_GAN_Quotient" || method == "SHMPC_GAN_Reduced") &&
        config.use_gan_cache && !config.gan_scenario_csv_path.empty()) {
        gan_cache.emplace();
        if (!gan_cache->load(config.gan_scenario_csv_path, config.horizon))
            gan_cache.reset();
    }
    std::optional<GANScenarioCache> stiefel_gan_cache;
    if (method == "SHMPC_GAN_Stiefel" && !config.stiefel_gan_scenario_csv_path.empty()) {
        stiefel_gan_cache.emplace();
        if (!stiefel_gan_cache->load(config.stiefel_gan_scenario_csv_path, config.horizon))
            stiefel_gan_cache.reset();
    }

    for (int step = 0; step < config.rollout_steps; ++step) {
        if (!config.rare_mode.empty() && config.rare_switch_prob > 0) {
            std::uniform_real_distribution<double> u(0, 1);
            if (u(rng) < config.rare_switch_prob) {
                obs_sim.current_mode = config.rare_mode;
            } else {
                obs_sim.maybe_switch(config.switch_prob, rng);
            }
        } else {
            obs_sim.maybe_switch(config.switch_prob, rng);
        }
        apply_distribution_shift(config.shift, obs_sim, rng);
        controller.update_mode_observation(obs_id, obs_sim.current_mode, step + 5);

        if (obs_sim.current_mode != prev_mode) last_switch_step = step;
        int time_since_switch = step - last_switch_step;

        if (method == "SHMPC_AdaptiveDRO") {
            double dist = (ego.position() - obs_sim.state.position()).norm();
            double residual = (dist < 5.0) ? (5.0 - dist) * 0.2 : 0.0;
            adaptive_shift.push_residual(residual);
            controller.set_dro_epsilon_override(adaptive_shift.get_rho());
        }

        if (method == "SHMPC_Conformal") {
            std::map<std::string, double> boundary_scores;
            for (const auto& m : obs_sim.available_modes)
                boundary_scores[m] = (m == config.rare_mode) ? 2.0 : 0.5;
            auto alloc = allocate_scenarios_boundary_aware(boundary_scores, config.num_scenarios, 1.0);
            std::map<std::string, double> weights;
            int sum = 0;
            for (const auto& [_, Sm] : alloc) sum += Sm;
            for (const auto& [mid, Sm] : alloc) weights[mid] = (sum > 0) ? (static_cast<double>(Sm) / sum) : (1.0 / alloc.size());
            controller.set_custom_mode_weights(obs_id, weights);
        } else if (method == "SHMPC_Hazard") {
            HazardFeatures hf;
            hf.time_since_last_switch = static_cast<double>(std::max(1, time_since_switch));
            std::map<std::string, double> hazard_per_mode;
            for (const auto& m : obs_sim.available_modes) hazard_per_mode[m] = hazard_model.hazard(m, hf);
            std::map<std::string, double> nominal;
            for (const auto& m : obs_sim.available_modes) nominal[m] = 1.0 / obs_sim.available_modes.size();
            auto reweighted = reweight_by_hazard(nominal, hazard_per_mode, 1.0);
            double max_h = 0;
            for (const auto& [_, v] : hazard_per_mode) if (v > max_h) max_h = v;
            auto alloc = allocate_with_hazard_trigger(reweighted, config.num_scenarios, max_h, 0.5, 1);
            std::map<std::string, double> weights;
            int sum = 0;
            for (const auto& [_, Sm] : alloc) sum += Sm;
            for (const auto& [mid, Sm] : alloc) weights[mid] = (sum > 0) ? (static_cast<double>(Sm) / sum) : (1.0 / alloc.size());
            controller.set_custom_mode_weights(obs_id, weights);
        } else if (method == "SHMPC_Bandit") {
            bandit.update(step, prev_mode, prev_collision);
            auto alloc = bandit.allocate(config.num_scenarios);
            std::map<std::string, double> weights;
            int sum = 0;
            for (const auto& [_, Sm] : alloc) sum += Sm;
            for (const auto& [mid, Sm] : alloc) weights[mid] = (sum > 0) ? (static_cast<double>(Sm) / sum) : (1.0 / alloc.size());
            controller.set_custom_mode_weights(obs_id, weights);
        } else if (method == "SHMPC_DoubleDual") {
            auto it = dual_mode_weights.find(obs_id);
            if (it != dual_mode_weights.end() && !it->second.empty())
                controller.set_custom_mode_weights(obs_id, it->second);
        } else if (method == "SHMPC_Certificate" || method == "CertificateFirst") {
            std::vector<double> radii(static_cast<size_t>(config.horizon + 1), cert_radius);
            controller.set_certificate_radii(radii);
        }

        std::map<int, ObstacleState> obstacles;
        obstacles[obs_id] = obs_sim.state;

        MPCResult mpc_result;
        if (method == "SHMPC_GAN" && !config.gan_scenario_csv_path.empty()) {
            std::vector<Scenario> gan_scenarios;
            int S_use = config.gan_num_scenarios_override > 0 ? config.gan_num_scenarios_override : config.num_scenarios;
            if (gan_cache) {
                gan_scenarios = gan_cache->materialize(obstacles, config.horizon, S_use);
            } else {
                gan_scenarios = load_scenarios_from_gan_csv(config.gan_scenario_csv_path, obstacles, config.horizon);
                if (static_cast<int>(gan_scenarios.size()) > S_use)
                    gan_scenarios.resize(S_use);
            }
            if (!gan_scenarios.empty()) {
                controller.set_scenarios(gan_scenarios);
                controller.set_use_current_scenarios_next(true);
            }
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        } else if (method == "SHMPC_GAN_Reduced" && !config.gan_scenario_csv_path.empty()) {
            std::vector<Scenario> gan_scenarios;
            const int S_reduced = config.gan_reduced_num_scenarios > 0 ? config.gan_reduced_num_scenarios : 12;
            if (gan_cache) {
                gan_scenarios = gan_cache->materialize(obstacles, config.horizon, S_reduced);
            } else {
                auto full = load_scenarios_from_gan_csv(config.gan_scenario_csv_path, obstacles, config.horizon);
                if (static_cast<int>(full.size()) > S_reduced)
                    gan_scenarios.assign(full.begin(), full.begin() + S_reduced);
                else
                    gan_scenarios = full;
            }
            if (!gan_scenarios.empty()) {
                controller.set_scenarios(gan_scenarios);
                controller.set_use_current_scenarios_next(true);
            }
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        } else if (method == "SHMPC_GAN_Quotient" && !config.gan_scenario_csv_path.empty()) {
            std::vector<Scenario> gan_scenarios;
            if (gan_cache) {
                gan_scenarios = gan_cache->materialize(obstacles, config.horizon, config.num_scenarios);
            } else {
                gan_scenarios = load_scenarios_from_gan_csv(config.gan_scenario_csv_path, obstacles, config.horizon);
            }
            const int K_quotient = config.quotient_num_override > 0 ? config.quotient_num_override : std::max(1, static_cast<int>(0.4 * config.num_scenarios));
            if (static_cast<int>(gan_scenarios.size()) > K_quotient) {
                std::vector<Scenario> reduced = reduce_scenarios_quotient_space(gan_scenarios, K_quotient, config.horizon);
                controller.set_scenarios(reduced);
                controller.set_use_current_scenarios_next(true);
            } else if (!gan_scenarios.empty()) {
                controller.set_scenarios(gan_scenarios);
                controller.set_use_current_scenarios_next(true);
            }
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        } else if (method == "SHMPC_Reservoir" && !config.reservoir_scenario_csv_path.empty()) {
            auto res_scenarios = load_scenarios_from_gan_csv(config.reservoir_scenario_csv_path, obstacles, config.horizon);
            if (!res_scenarios.empty()) {
                int S_use = std::min(config.num_scenarios, static_cast<int>(res_scenarios.size()));
                std::vector<Scenario> selected(res_scenarios.begin(), res_scenarios.begin() + S_use);
                controller.set_scenarios(selected);
                controller.set_use_current_scenarios_next(true);
            }
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        } else if (method == "SHMPC_SeekAvoid" && !config.seek_avoid_scenario_csv_path.empty()) {
            auto seek_scenarios = load_scenarios_from_gan_csv(config.seek_avoid_scenario_csv_path, obstacles, config.horizon);
            if (!seek_scenarios.empty()) {
                int S_use = std::min(config.num_scenarios, static_cast<int>(seek_scenarios.size()));
                std::vector<Scenario> selected(seek_scenarios.begin(), seek_scenarios.begin() + S_use);
                controller.set_scenarios(selected);
                controller.set_use_current_scenarios_next(true);
            }
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        } else if (method == "SHMPC_SeekAvoidML" && !config.seek_avoid_ml_scenario_csv_path.empty()) {
            auto ml_scenarios = load_scenarios_from_gan_csv(config.seek_avoid_ml_scenario_csv_path, obstacles, config.horizon);
            if (!ml_scenarios.empty()) {
                int S_use = std::min(config.num_scenarios, static_cast<int>(ml_scenarios.size()));
                std::vector<Scenario> selected(ml_scenarios.begin(), ml_scenarios.begin() + S_use);
                controller.set_scenarios(selected);
                controller.set_use_current_scenarios_next(true);
            }
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        } else if (method == "SHMPC_GAN_Stiefel" && !config.stiefel_gan_scenario_csv_path.empty()) {
            std::vector<Scenario> stiefel_scenarios;
            if (stiefel_gan_cache) {
                stiefel_scenarios = stiefel_gan_cache->materialize(obstacles, config.horizon, config.num_scenarios);
            } else {
                stiefel_scenarios = load_scenarios_from_gan_csv(config.stiefel_gan_scenario_csv_path, obstacles, config.horizon);
                if (static_cast<int>(stiefel_scenarios.size()) > config.num_scenarios)
                    stiefel_scenarios.resize(static_cast<size_t>(config.num_scenarios));
            }
            if (!stiefel_scenarios.empty()) {
                controller.set_scenarios(stiefel_scenarios);
                controller.set_use_current_scenarios_next(true);
            }
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        } else if (method == "SHMPC_QuotientSpace") {
            controller.sample_and_set_scenarios(obstacles, config.num_scenarios);
            const std::vector<Scenario>& full = controller.scenarios();
            int K = config.quotient_num_override > 0
                ? config.quotient_num_override
                : std::max(1, static_cast<int>(0.4 * config.num_scenarios));
            if (static_cast<int>(full.size()) > K) {
                std::vector<Scenario> reduced = reduce_scenarios_quotient_space(full, K, config.horizon);
                controller.set_scenarios(reduced);
                controller.set_use_current_scenarios_next(true);
            }
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        } else if (method == "SHMPC_Compiler" || method == "ScenarioCompiler") {
            controller.sample_and_set_scenarios(obstacles, compiler_initial_S);
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
            for (int comp_iter = 0; comp_iter < compiler_max_iter - 1 && mpc_result.success; ++comp_iter) {
                std::vector<Scenario> witness(controller.scenarios().begin(), controller.scenarios().end());
                Scenario worst;
                bool found = false;
                int next_id = static_cast<int>(witness.size());
                for (const auto& m : obs_sim.available_modes) {
                    Scenario sc = build_scenario_for_mode(next_id, obs_id, obs_sim.state, m, obs_mode_models, config.horizon);
                    if (scenario_violates_plan(sc, mpc_result.ego_trajectory, collision_radius, 0.0)) {
                        worst = sc;
                        found = true;
                        next_id++;
                        break;
                    }
                }
                if (!found) break;
                witness.push_back(worst);
                controller.set_scenarios(witness);
                controller.set_use_current_scenarios_next(true);
                mpc_result = controller.solve(ego, obstacles, goal, 1.5);
            }
        } else {
            mpc_result = controller.solve(ego, obstacles, goal, 1.5);
        }

        if (method == "SHMPC_AdaptiveDRO") controller.clear_dro_epsilon_override();
        if (method == "SHMPC_Conformal" || method == "SHMPC_Hazard" || method == "SHMPC_Bandit") controller.clear_custom_mode_weights();
        if (method == "SHMPC_DoubleDual" && mpc_result.success) {
            std::map<std::string, double> count_per_mode;
            for (const auto& m : obs_sim.available_modes) count_per_mode[m] = 0.0;
            for (const auto& sc : controller.scenarios()) {
                auto it = sc.trajectories.find(obs_id);
                if (it != sc.trajectories.end())
                    count_per_mode[it->second.mode_id] += 1.0;
            }
            const double floor = 0.1;
            double sum = 0.0;
            for (const auto& m : obs_sim.available_modes)
                sum += count_per_mode[m] + floor;
            std::map<std::string, double> weights;
            for (const auto& m : obs_sim.available_modes)
                weights[m] = (count_per_mode[m] + floor) / (sum > 0 ? sum : 1.0);
            dual_mode_weights[obs_id] = std::move(weights);
        }
        if (method == "SHMPC_Certificate" || method == "CertificateFirst") controller.clear_certificate_radii();

        solve_times.push_back(mpc_result.solve_time * 1000.0);
        rec.total_dro_injected += mpc_result.num_dro_injected;
        if (mpc_result.safe_horizon > 0)
            safe_horizons.push_back(mpc_result.safe_horizon);

        double dist = (ego.position() - obs_sim.state.position()).norm();
        clearances.push_back(dist);
        if (dist < rec.min_clearance) {
            rec.min_clearance = dist;
            rec.min_clearance_step = step;
        }
        if (dist < collision_radius && !rec.collision) {
            rec.collision = true;
            rec.collision_step = step;
        }
        constraint_active_total += static_cast<int>(mpc_result.active_scenarios.size());

        bool mode_found = false;
        for (const auto& sc : controller.scenarios()) {
            for (const auto& [oid, traj] : sc.trajectories) {
                if (oid == obs_id && traj.mode_id == obs_sim.current_mode) {
                    mode_found = true; break;
                }
            }
            if (mode_found) break;
        }
        if (!mode_found) rec.missed_mode_steps++;

        if (mpc_result.success && mpc_result.first_input().has_value()) {
            EgoInput u = mpc_result.first_input().value();
            if (method == "SHMPC_RTA") {
                u = rta.apply(ego, u, std::nullopt);
            }
            control_effort += u.a * u.a + u.delta * u.delta;
            ego = dynamics.propagate(ego, u);
        }
        prev_collision = (dist < collision_radius);
        prev_mode = obs_sim.current_mode;
        obs_sim.step(DT, rng);
        rec.total_steps++;
    }

    rec.total_progress = ego.x;
    rec.control_effort = control_effort;
    rec.constraint_active_count = constraint_active_total;

    if (!solve_times.empty()) {
        rec.avg_solve_ms = std::accumulate(solve_times.begin(), solve_times.end(), 0.0) / solve_times.size();
        std::vector<double> sorted_times = solve_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        int n = static_cast<int>(sorted_times.size());
        rec.p50_solve_ms = sorted_times[n / 2];
        rec.p95_solve_ms = sorted_times[std::min(n - 1, static_cast<int>(0.95 * n))];
        rec.max_solve_ms = sorted_times.back();
    }
    if (!safe_horizons.empty()) {
        rec.avg_safe_horizon = std::accumulate(safe_horizons.begin(), safe_horizons.end(), 0.0) / safe_horizons.size();
    }
    if (!clearances.empty()) {
        std::vector<double> sorted_clear = clearances;
        std::sort(sorted_clear.begin(), sorted_clear.end());
        int idx_5pct = std::max(0, static_cast<int>(0.05 * sorted_clear.size()) - 1);
        rec.clearance_5pct = sorted_clear[idx_5pct];
    }
    return rec;
}

}  // namespace scenario_mpc
