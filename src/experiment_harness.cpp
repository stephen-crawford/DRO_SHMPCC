/**
 * @file experiment_harness.cpp
 * @brief Canonical rollout runner, obstacle simulator, stats, CSV writer.
 *
 * ALL rollout logic lives here. The paper_experiment_runner configures
 * ExperimentConfig and calls run_experiment_rollout().
 */

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "collision_constraints.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "reference_path.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <random>
#include <vector>

namespace scenario_mpc {

// ============================================================================
// ObstacleSim
// ============================================================================

void ObstacleSim::step(double dt, std::mt19937& rng) {
    if (mode_models.find(current_mode) == mode_models.end()) return;
    const auto& model = mode_models.at(current_mode);
    Eigen::VectorXd noise = Eigen::VectorXd::Zero(model.noise_dim());
    std::normal_distribution<double> nd(0, 1);
    for (int i = 0; i < model.noise_dim(); ++i) noise(i) = nd(rng) * 0.02;
    state = model.propagate(state, &noise);
    double spd = std::sqrt(state.vx * state.vx + state.vy * state.vy);
    if (spd > 2.0) { state.vx *= 2.0 / spd; state.vy *= 2.0 / spd; }
}

void ObstacleSim::maybe_switch(double switch_prob, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0, 1);
    if (u(rng) < switch_prob && !available_modes.empty()) {
        std::uniform_int_distribution<int> idx(0, static_cast<int>(available_modes.size()) - 1);
        current_mode = available_modes[idx(rng)];
    }
}

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

std::pair<double, double> wilson_ci(int successes, int num_trials, double z) {
    if (num_trials == 0) return {0.0, 1.0};
    double p_hat = static_cast<double>(successes) / num_trials;
    double denom = 1.0 + z * z / num_trials;
    double center = (p_hat + z * z / (2.0 * num_trials)) / denom;
    double half_width = z * std::sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * num_trials)) / num_trials) / denom;
    return {std::max(0.0, center - half_width), std::min(1.0, center + half_width)};
}

BootstrapResult bootstrap_paired_delta(
    const std::vector<bool>& base_collisions,
    const std::vector<bool>& dro_collisions,
    int n_bootstrap,
    std::mt19937* rng
) {
    const int n = static_cast<int>(base_collisions.size());
    assert(n == static_cast<int>(dro_collisions.size()));
    assert(n > 0);
    assert(n_bootstrap > 0);

    std::mt19937 local_rng;
    if (!rng) {
        std::random_device rd;
        std::seed_seq seq{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
        local_rng.seed(seq);
        rng = &local_rng;
    }

    std::uniform_int_distribution<int> idx_dist(0, n - 1);

    std::vector<int> diffs(n);
    for (int j = 0; j < n; ++j) {
        diffs[j] = (base_collisions[j] ? 1 : 0) - (dro_collisions[j] ? 1 : 0);
    }

    std::vector<double> deltas(static_cast<size_t>(n_bootstrap));
    for (int b = 0; b < n_bootstrap; ++b) {
        int sum_diff = 0;
        for (int i = 0; i < n; ++i) {
            sum_diff += diffs[idx_dist(*rng)];
        }
        deltas[static_cast<size_t>(b)] = static_cast<double>(sum_diff) / n;
    }

    std::sort(deltas.begin(), deltas.end());

    const double mean_delta =
        std::accumulate(deltas.begin(), deltas.end(), 0.0) / n_bootstrap;

    auto quantile_index = [n_bootstrap](double q) -> int {
        q = std::min(1.0, std::max(0.0, q));
        return static_cast<int>(std::floor(q * (n_bootstrap - 1)));
    };

    return {mean_delta, deltas[quantile_index(0.025)], deltas[quantile_index(0.975)]};
}

double mcnemar_chi2(int b, int c) {
    if (b + c == 0) return 0.0;
    double num = std::abs(static_cast<double>(b) - c) - 1.0;
    num = std::max(0.0, num);
    return (num * num) / (b + c);
}

EffectSizes compute_effect_sizes(double p_base, double p_comparison) {
    EffectSizes es;
    es.abs_delta = p_base - p_comparison;
    es.rel_delta = (p_base > 1e-12) ? (p_base - p_comparison) / p_base : 0.0;
    es.risk_ratio = (p_base > 1e-12) ? p_comparison / p_base : 0.0;
    es.cohens_h = 2.0 * std::asin(std::sqrt(p_base)) - 2.0 * std::asin(std::sqrt(p_comparison));
    return es;
}

// ============================================================================
// Utilities
// ============================================================================

SeedBundle derive_seeds(unsigned master_seed, int idx) {
    auto hash = [](unsigned a, unsigned b) -> unsigned {
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

void apply_distribution_shift(
    const DistributionShiftConfig& shift,
    ObstacleSim& obs_sim,
    std::mt19937& rng
) {
    if (shift.rho <= 0.0 && shift.dangerous_boost <= 0.0) return;

    std::uniform_real_distribution<double> u(0, 1);

    if (shift.rho > 0.0 && u(rng) < shift.rho) {
        if (!obs_sim.available_modes.empty()) {
            std::uniform_int_distribution<int> idx(0, static_cast<int>(obs_sim.available_modes.size()) - 1);
            obs_sim.current_mode = obs_sim.available_modes[idx(rng)];
        }
    }

    if (shift.dangerous_boost > 0.0 && u(rng) < shift.dangerous_boost) {
        int boost_idx = shift.boosted_mode;
        if (boost_idx < 0) {
            boost_idx = static_cast<int>(obs_sim.available_modes.size()) - 1;
        }
        if (boost_idx >= 0 && boost_idx < static_cast<int>(obs_sim.available_modes.size())) {
            obs_sim.current_mode = obs_sim.available_modes[boost_idx];
        }
    }
}

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
            dro_cfg.rho_base = eps_wass;
            // sigma_scale no longer in DROConfig; directional risk uses alpha_one_sided
            (void)sigma_scale;
            break;
        case AblationVariant::DRO_NO_COV:
            config.enable_dro = true;
            dro_cfg.risk_mode = DRORiskMode::NO_COV;
            dro_cfg.rho_base = eps_wass;
            // risk_sigma_scale removed; NO_COV/DISTANCE_ONLY handled by risk_mode
            break;
        case AblationVariant::DRO_DISTANCE_ONLY:
            config.enable_dro = true;
            dro_cfg.risk_mode = DRORiskMode::DISTANCE_ONLY;
            dro_cfg.rho_base = eps_wass;
            // risk_sigma_scale removed; NO_COV/DISTANCE_ONLY handled by risk_mode
            break;
        case AblationVariant::RANDOM_INJECTION:
        case AblationVariant::ALWAYS_INJECT:
            config.enable_dro = false;
            break;
    }
}

ObstacleState obstacle_on_s_curve(
    const ReferencePath& path,
    double arc_fraction,
    std::mt19937& rng
) {
    double obs_s = arc_fraction * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d tangent(std::cos(pp.heading), std::sin(pp.heading));

    std::uniform_real_distribution<double> lat(-0.5, 0.5);
    std::uniform_real_distribution<double> spd(-0.2, 0.1);
    Eigen::Vector2d pos = pp.position + lat(rng) * normal;
    double v = 0.3 + spd(rng);
    return ObstacleState(pos.x(), pos.y(), v * tangent.x(), v * tangent.y());
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    int lo = static_cast<int>(std::floor(idx));
    int hi = std::min(lo + 1, static_cast<int>(v.size()) - 1);
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

// ============================================================================
// Canonical Rollout Runner
// ============================================================================

RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config,
    unsigned seed
) {
    std::mt19937 rng(seed);
    RolloutRecord rec;
    rec.seed = seed;
    rec.method = config.method_name.empty()
        ? ablation_variant_name(config.ablation) : config.method_name;
    rec.scenario = config.scenario_tag;
    rec.S = config.num_scenarios;
    rec.eps_wass = config.eps_wass;
    rec.sigma = config.sigma_scale;
    rec.shift_rho = config.shift.rho;
    rec.shift_boost = config.shift.dangerous_boost;
    rec.ground_cost = ground_cost_name(config.ground_cost);

    constexpr double DT = 0.1;
    auto mode_models = create_obstacle_mode_models(DT);

    // ---- Configure MPC ----
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
    mpc_cfg.vehicle_length = config.vehicle_length;
    mpc_cfg.safe_horizon_enabled = config.safe_horizon_enabled;
    mpc_cfg.safe_horizon_min = config.safe_horizon_min;
    mpc_cfg.safe_horizon_mode = SafeHorizonMode::PRACTICAL;
    mpc_cfg.forced_safe_horizon = config.forced_safe_horizon;
    mpc_cfg.weight_type = config.weight_type;
    mpc_cfg.enable_dro = config.enable_dro;
    mpc_cfg.injection_mode = config.injection_mode;
    mpc_cfg.dro_injection_count = config.dro_injection_count;
    mpc_cfg.softmax_tau = config.softmax_tau;
    mpc_cfg.eps_greedy_epsilon = config.eps_greedy_epsilon;
    mpc_cfg.max_history_length = config.max_history_length;

    // Legacy ablation path: if enable_dro not set directly, derive from ablation
    DROConfig dro_cfg;
    dro_cfg.ground_cost_type = config.ground_cost;
    if (!config.enable_dro && config.ablation != AblationVariant::NO_INJECTION) {
        configure_ablation(mpc_cfg, dro_cfg, config.ablation,
                           config.eps_wass, config.sigma_scale);
    }

    if (mpc_cfg.enable_dro) {
        mpc_cfg.dro_rho_base = dro_cfg.rho_base > 0 ? dro_cfg.rho_base : config.eps_wass;
        mpc_cfg.dro_adaptive_rho = dro_cfg.adaptive_rho;
        // Forward the rest of dro_cfg. Without these the local dro_cfg above is
        // discarded and the controller silently falls back to W2_BURES / FULL,
        // which made the ground-cost and risk-mode ablations no-ops.
        mpc_cfg.dro_ground_cost = dro_cfg.ground_cost_type;
        mpc_cfg.dro_risk_mode = dro_cfg.risk_mode;
        mpc_cfg.dro_risk_measure = config.risk_measure;
        mpc_cfg.dro_joint_risk_samples = config.joint_risk_samples;
    }

    AdaptiveScenarioMPC controller(mpc_cfg);

    // ---- Build mode model set ----
    std::map<std::string, ModeModel> obs_mode_models;
    for (const auto& m : config.obs_modes) {
        if (mode_models.find(m) != mode_models.end())
            obs_mode_models[m] = mode_models[m];
    }
    if (!config.rare_mode.empty() && mode_models.find(config.rare_mode) != mode_models.end()) {
        obs_mode_models[config.rare_mode] = mode_models[config.rare_mode];
    }

    // ---- Reference path (custom or default S-curve) ----
    ReferencePath ref_path = config.custom_ref_path.has_value()
        ? config.custom_ref_path.value()
        : ReferencePath::create_s_curve(25.0, 3.0, 200);
    controller.set_reference_path(ref_path);
    double path_length = ref_path.total_length();
    Eigen::Vector2d goal = ref_path.get_position_at(path_length);

    EgoState ego = config.custom_initial_ego.value_or(EgoState(0.0, 0.0, 0.0, 1.5));
    EgoDynamics dynamics(DT);
    double collision_radius = mpc_cfg.ego_radius + mpc_cfg.obstacle_radius;
    double path_progress = 0.0;

    // ---- Setup obstacles ----
    const int n_obs = std::max(1, config.num_obstacles);
    const int per_class = std::max(1, config.obstacles_per_class);
    std::vector<ObstacleSim> obs_sims(n_obs);

    std::vector<std::string> all_modes = config.obs_modes;
    if (!config.rare_mode.empty()) all_modes.push_back(config.rare_mode);

    for (int i = 0; i < n_obs; ++i) {
        int obs_class = i / per_class;
        controller.initialize_obstacle(i, obs_class, obs_mode_models);

        // Determine initial state
        if (i < static_cast<int>(config.initial_obstacle_states.size())) {
            obs_sims[i].state = config.initial_obstacle_states[i];
        } else {
            double frac = 0.35;
            if (i < static_cast<int>(config.obs_arc_fractions.size())) {
                frac = config.obs_arc_fractions[i];
            } else if (n_obs > 1) {
                frac = 0.20 + 0.15 * i;
                frac = std::min(frac, 0.85);
            }
            obs_sims[i].state = obstacle_on_s_curve(ref_path, frac, rng);
        }

        obs_sims[i].current_mode = config.obs_modes.empty()
            ? "constant_velocity" : config.obs_modes[i % config.obs_modes.size()];
        obs_sims[i].available_modes = all_modes;
        obs_sims[i].mode_models = obs_mode_models;
    }

    // ---- Initial mode observations (5 warmup steps) ----
    for (int t = 0; t < 5; ++t) {
        for (int oi = 0; oi < n_obs; ++oi) {
            int obs_class = oi / per_class;
            controller.update_mode_observation(oi, obs_class, obs_sims[oi].current_mode, t);
        }
    }

    // ---- Main rollout loop ----
    std::vector<double> clearances;
    std::vector<int> safe_horizons;
    double control_effort = 0.0;
    int constraint_active_total = 0;

    for (int step = 0; step < config.rollout_steps; ++step) {
        // Mode switching and observation for each obstacle
        for (int oi = 0; oi < n_obs; ++oi) {
            if (!config.rare_mode.empty() && config.rare_switch_prob > 0) {
                std::uniform_real_distribution<double> u(0, 1);
                if (u(rng) < config.rare_switch_prob) {
                    obs_sims[oi].current_mode = config.rare_mode;
                } else {
                    obs_sims[oi].maybe_switch(config.switch_prob, rng);
                }
            } else {
                obs_sims[oi].maybe_switch(config.switch_prob, rng);
            }

            apply_distribution_shift(config.shift, obs_sims[oi], rng);

            int obs_class = oi / per_class;
            controller.update_mode_observation(oi, obs_class, obs_sims[oi].current_mode, step + 5);

            // Per-step callback (for experiment-specific logic like oracle flood)
            if (config.step_callback) {
                config.step_callback(step, oi, obs_sims[oi], controller, rng);
            }
        }
        // Track path progress
        path_progress = ref_path.find_closest_point(ego.position(), path_progress);

        // Build obstacle map
        std::map<int, ObstacleState> obstacles;
        for (int oi = 0; oi < n_obs; ++oi) {
            obstacles[oi] = obs_sims[oi].state;
        }

        auto mpc_result = controller.solve(ego, obstacles, goal, 1.5, path_progress, path_length);
        rec.solve_times_raw.push_back(mpc_result.solve_time);
        rec.total_dro_injected += mpc_result.num_dro_injected;
        if (mpc_result.safe_horizon > 0)
            safe_horizons.push_back(mpc_result.safe_horizon);

        // Collision detection (multi-disc if D>1)
        for (int oi = 0; oi < n_obs; ++oi) {
            bool collision_this_obs = false;
            double min_dist_this_obs = 1e9;

            if (config.num_discs > 1) {
                double theta = ego.theta;
                Eigen::Vector2d dir(std::cos(theta), std::sin(theta));
                double step_offset = config.vehicle_length / (config.num_discs - 1);
                for (int d = 0; d < config.num_discs; ++d) {
                    double offset = -config.vehicle_length / 2.0 + d * step_offset;
                    Eigen::Vector2d disc_pos = ego.position() + offset * dir;
                    double dist_d = (disc_pos - obs_sims[oi].state.position()).norm();
                    min_dist_this_obs = std::min(min_dist_this_obs, dist_d);
                    if (dist_d < collision_radius) collision_this_obs = true;
                }
            } else {
                double dist = (ego.position() - obs_sims[oi].state.position()).norm();
                min_dist_this_obs = dist;
                if (dist < collision_radius) collision_this_obs = true;
            }

            clearances.push_back(min_dist_this_obs);
            if (min_dist_this_obs < rec.min_clearance) {
                rec.min_clearance = min_dist_this_obs;
                rec.min_clearance_step = step;
            }
            if (collision_this_obs && !rec.collision) {
                rec.collision = true;
                rec.collision_step = step;
            }
        }

        // Constraint active count
        constraint_active_total += static_cast<int>(mpc_result.active_scenarios.size());

        // Missed mode tracking per obstacle
        for (int oi = 0; oi < n_obs; ++oi) {
            bool mode_found = false;
            for (const auto& sc : controller.scenarios()) {
                for (const auto& [oid, traj] : sc.trajectories) {
                    if (oid == oi && traj.mode_id == obs_sims[oi].current_mode) {
                        mode_found = true; break;
                    }
                }
                if (mode_found) break;
            }
            if (!mode_found) rec.missed_mode_steps++;
            rec.total_mode_checks++;

            // Rare mode tracking
            if (!config.rare_mode.empty() && obs_sims[oi].current_mode == config.rare_mode) {
                rec.rare_mode_active++;
                if (!mode_found) rec.rare_mode_missed++;
            }
        }

        // Apply control
        if (mpc_result.success && mpc_result.first_input().has_value()) {
            auto input = mpc_result.first_input().value();
            control_effort += input.a * input.a + input.delta * input.delta;
            ego = dynamics.propagate(ego, input);
        }

        // Propagate all obstacles
        for (int oi = 0; oi < n_obs; ++oi) {
            obs_sims[oi].step(DT, rng);
        }
        rec.total_steps++;

        // Path completion termination
        if (config.path_completion_termination &&
            path_progress >= config.path_completion_fraction * path_length) {
            rec.completed_path = true;
            break;
        }
    }

    // ---- Compute summary metrics ----
    rec.total_progress = path_length > 0 ? path_progress / path_length : ego.x;
    rec.control_effort = control_effort;
    rec.constraint_active_count = constraint_active_total;
    rec.active_constraints = rec.total_steps > 0
        ? constraint_active_total / rec.total_steps : 0;

    // Solve time stats
    if (!rec.solve_times_raw.empty()) {
        // Convert to ms for summary stats
        std::vector<double> times_ms;
        times_ms.reserve(rec.solve_times_raw.size());
        for (double t : rec.solve_times_raw) times_ms.push_back(t * 1000.0);

        rec.avg_solve_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
        std::sort(times_ms.begin(), times_ms.end());
        int n = static_cast<int>(times_ms.size());
        rec.p50_solve_ms = times_ms[n / 2];
        rec.p95_solve_ms = times_ms[std::min(n - 1, static_cast<int>(0.95 * n))];
        rec.max_solve_ms = times_ms.back();
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

// ============================================================================
// Paper Variant Helpers
// ============================================================================

ReferencePath setup_mpcc_path(AdaptiveScenarioMPC& ctrl) {
    auto ref_path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);
    ctrl.set_reference_path(ref_path);
    return ref_path;
}

ExperimentConfig make_experiment_config(
    PaperVariant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    const std::vector<std::string>& obs_modes,
    const std::string& rare_mode,
    double rare_prob,
    bool safe_horizon_enabled,
    int num_discs,
    double vehicle_length
) {
    ExperimentConfig cfg;
    cfg.horizon = DEFAULT_HORIZON;
    cfg.num_scenarios = num_scenarios;
    cfg.switch_prob = switch_prob;
    cfg.rollout_steps = rollout_steps;
    cfg.obs_modes = obs_modes;
    cfg.rare_mode = rare_mode;
    cfg.rare_switch_prob = rare_prob;
    cfg.num_discs = num_discs;
    cfg.vehicle_length = vehicle_length;
    cfg.weight_type = WeightType::FREQUENCY;
    cfg.enable_dro = uses_dro(variant);
    cfg.injection_mode = InjectionMode::QSTAR_SAMPLE;
    cfg.safe_horizon_enabled = safe_horizon_enabled || uses_sh(variant);
    cfg.path_completion_termination = true;
    cfg.path_completion_fraction = PATH_COMPLETE_FRAC;
    cfg.method_name = variant_name(variant);
    cfg.ablation = AblationVariant::NO_INJECTION;
    return cfg;
}

RolloutResult run_single_rollout(
    PaperVariant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    unsigned seed,
    const std::vector<std::string>& obs_modes,
    const std::string& rare_mode,
    double rare_prob,
    bool safe_horizon_enabled,
    int num_discs,
    double vehicle_length
) {
    auto cfg = make_experiment_config(
        variant, switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob, safe_horizon_enabled,
        num_discs, vehicle_length);
    return RolloutResult::from_record(run_experiment_rollout(cfg, seed));
}

RolloutResult run_multi_obstacle_rollout(
    PaperVariant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    unsigned seed,
    int num_obstacles,
    int num_classes,
    const std::vector<std::string>& obs_modes,
    const std::string& rare_mode,
    double rare_prob,
    const std::vector<double>& arc_fracs
) {
    auto cfg = make_experiment_config(
        variant, switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob, false, 1, 1.5);
    cfg.num_obstacles = num_obstacles;
    cfg.obstacles_per_class = (num_classes > 0 && num_classes < num_obstacles)
        ? (num_obstacles / num_classes) : 1;
    cfg.obs_arc_fractions = arc_fracs;
    return RolloutResult::from_record(run_experiment_rollout(cfg, seed));
}

EnvironmentSetup create_environment(EnvironmentType env, std::mt19937& rng) {
    EnvironmentSetup setup;
    std::uniform_real_distribution<double> jitter(-0.3, 0.3);

    auto path = ReferencePath::create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, S_CURVE_POINTS);
    setup.obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"};
    setup.initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
    setup.goal = path.get_position_at(path.total_length());

    switch (env) {
        case EnvironmentType::STRAIGHT: {
            setup.name = "Straight";
            setup.initial_obs = obstacle_on_s_curve(path, 0.35, rng);
            break;
        }
        case EnvironmentType::NARROW_CORRIDOR: {
            setup.name = "Narrow";
            double s25 = 0.25 * path.total_length();
            PathPoint pp = path.get_point_at(s25);
            Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
            Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
            Eigen::Vector2d pos = pp.position + jitter(rng) * 0.15 * n;
            setup.initial_obs = ObstacleState(pos.x(), pos.y(),
                                               -0.2 * t.x(), -0.2 * t.y());
            break;
        }
        case EnvironmentType::INTERSECTION: {
            setup.name = "Intersection";
            double s40 = 0.40 * path.total_length();
            PathPoint pp = path.get_point_at(s40);
            Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
            Eigen::Vector2d pos = pp.position + (2.0 + jitter(rng)) * n;
            setup.initial_obs = ObstacleState(pos.x(), pos.y(),
                                               -1.0 * n.x() + jitter(rng) * 0.2,
                                               -1.0 * n.y() + jitter(rng) * 0.2);
            break;
        }
        case EnvironmentType::ONCOMING: {
            setup.name = "Oncoming";
            double s60 = 0.60 * path.total_length();
            PathPoint pp = path.get_point_at(s60);
            Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
            Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
            Eigen::Vector2d pos = pp.position + jitter(rng) * 0.3 * n;
            setup.initial_obs = ObstacleState(pos.x(), pos.y(),
                                               -1.0 * t.x() + jitter(rng) * 0.2,
                                               -1.0 * t.y() + jitter(rng) * 0.2);
            break;
        }
    }
    return setup;
}

RolloutResult run_single_rollout_env(
    PaperVariant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    unsigned seed,
    const EnvironmentSetup& env_setup,
    SamplingBaseline baseline,
    int forced_safe_horizon,
    int num_discs,
    double vehicle_length
) {
    auto cfg = make_experiment_config(
        variant, switch_prob, num_scenarios, rollout_steps,
        env_setup.obs_modes, "", 0.0, false, num_discs, vehicle_length);

    cfg.weight_type = baseline_to_weight(baseline);
    cfg.forced_safe_horizon = forced_safe_horizon;
    cfg.safe_horizon_enabled = uses_sh(variant) || (forced_safe_horizon >= 0);
    cfg.initial_obstacle_states = {env_setup.initial_obs};

    if (baseline == SamplingBaseline::ORACLE_FLOOD) {
        cfg.step_callback = [](int step, int obs_id, ObstacleSim& obs,
                               AdaptiveScenarioMPC& ctrl, std::mt19937&) {
            for (int f = 0; f < 50; ++f) {
                ctrl.update_mode_observation(obs_id, 0, obs.current_mode, step + 5);
            }
        };
    }

    return RolloutResult::from_record(run_experiment_rollout(cfg, seed));
}

}  // namespace scenario_mpc
