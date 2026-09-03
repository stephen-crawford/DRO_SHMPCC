/**
 * @file experiment_harness.cpp
 * @brief Canonical rollout runner, obstacle simulator, stats, CSV writer.
 *
 * ALL rollout logic lives here. The paper_experiment_runner configures
 * ExperimentConfig and calls run_experiment_rollout().
 */

#include "experiment_harness.hpp"
#include "experiment_config_yaml.hpp"
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
#include <limits>
#include <random>
#include <vector>

namespace dro_mpc {

int    DEFAULT_ROLLOUT_STEPS  = 200;
double DEFAULT_DT             = 0.1;
int    DEFAULT_HORIZON        = 20;
int    DEFAULT_BASE_SCENARIOS = 40;
double S_CURVE_LENGTH         = 25.0;
double S_CURVE_AMPLITUDE      = 3.0;
int    S_CURVE_POINTS         = 200;
double PATH_COMPLETE_FRAC     = 0.95;
double OBS_PATH_FRACTION      = 0.35;
std::vector<double> OBS_ARC_FRACS_4 = {0.20, 0.35, 0.50, 0.65};

namespace {
const ExperimentConfig& cached_default_experiment_config() {
    static const ExperimentConfig cfg = yaml_config::load_experiment_config("");
    return cfg;
}

struct BindYamlDefaults {
    BindYamlDefaults() {
        const ExperimentConfig& d = cached_default_experiment_config();
        DEFAULT_ROLLOUT_STEPS  = d.rollout.rollout_steps;
        DEFAULT_DT             = d.mpc.dt;
        DEFAULT_HORIZON        = d.mpc.horizon;
        DEFAULT_BASE_SCENARIOS = d.mpc.sampling.num_scenarios;
        S_CURVE_LENGTH         = d.environment.s_curve_length;
        S_CURVE_AMPLITUDE      = d.environment.s_curve_amplitude;
        S_CURVE_POINTS         = d.environment.s_curve_points;
        PATH_COMPLETE_FRAC     = d.environment.path_completion_fraction;
        OBS_PATH_FRACTION      = d.obstacles.default_arc_fraction;
        if (!d.obstacles.obs_arc_fractions.empty()) {
            OBS_ARC_FRACS_4 = d.obstacles.obs_arc_fractions;
        }
    }
} bind_yaml_defaults;

std::map<std::string, ModeModel> mode_models_for(
    const std::vector<std::string>& mode_ids,
    const std::map<std::string, ModeModel>& mode_catalog
) {
    std::map<std::string, ModeModel> selected;
    for (const auto& mode_id : mode_ids) {
        const auto it = mode_catalog.find(mode_id);
        if (it != mode_catalog.end()) selected.emplace(mode_id, it->second);
    }
    return selected;
}
}  // namespace

ExperimentConfig default_experiment_config() {
    return cached_default_experiment_config();
}

EnvironmentExperimentConfig default_environment_experiment_config() {
    return cached_default_experiment_config().environment;
}

// ============================================================================
// ObstacleSim
// ============================================================================

void ObstacleSim::step(double dt, std::mt19937& rng, double process_noise, double speed_cap) {
    if (mode_models.find(current_mode) == mode_models.end()) return;
    const auto& model = mode_models.at(current_mode);
    Eigen::VectorXd noise = Eigen::VectorXd::Zero(model.noise_dim());
    std::normal_distribution<double> nd(0, 1);
    for (int i = 0; i < model.noise_dim(); ++i) noise(i) = nd(rng) * process_noise;
    state = model.propagate(state, &noise);
    double spd = std::sqrt(state.vx * state.vx + state.vy * state.vy);
    if (spd > speed_cap) { state.vx *= speed_cap / spd; state.vy *= speed_cap / spd; }
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
    ofs_ << "seed,plant_seed,predictor_seed,controller_seed,method,scenario,S,eps_wass,sigma,shift_rho,shift_boost,ground_cost,"
         << "collision,collision_step,min_clearance,min_clearance_step,"
         << "total_progress,control_effort,constraint_active_count,"
         << "missed_mode_steps,total_steps,"
         << "avg_solve_ms,p50_solve_ms,p95_solve_ms,max_solve_ms,"
         << "total_dro_injected,avg_safe_horizon,clearance_5pct,"
         << "mean_contouring_err,mean_velocity_err,mean_lag_err,config_source\n";
}

void CSVWriter::write_record(const RolloutRecord& rec) {
    ofs_ << rec.seed << "," << rec.plant_seed << "," << rec.predictor_seed << ","
         << rec.controller_seed << "," << rec.method << "," << rec.scenario << "," << rec.S << ","
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
         << std::setprecision(4) << rec.avg_safe_horizon << "," << rec.clearance_5pct << ","
         << rec.mean_contouring_error() << "," << rec.mean_velocity_error() << ","
         << (rec.metric_steps > 0 ? std::sqrt(rec.sum_lag_sq / rec.metric_steps) : 0.0) << ","
         << rec.config_source << "\n";
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
    if (shift.psi <= 0.0 && shift.dangerous_boost <= 0.0) return;

    std::uniform_real_distribution<double> u(0, 1);

    if (shift.psi > 0.0 && u(rng) < shift.psi) {
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

std::vector<ObstacleSim> construct_obstacles(
    const ObstacleExperimentConfig& config,
    const ObstacleState& environment_default,
    const ReferencePath& reference_path,
    const std::map<std::string, ModeModel>& mode_catalog,
    std::mt19937& rng
) {
    const int obstacle_count = std::max(1, config.num_obstacles);
    std::vector<ObstacleSim> obstacles(obstacle_count);
    const auto shared_modes = select_obstacle_mode_ids(
        config.obs_modes, config.rare_mode, mode_catalog,
        config.randomize_available_modes, config.num_modes, rng);

    for (int i = 0; i < obstacle_count; ++i) {
        auto available_modes = config.randomize_modes_per_obstacle
            ? select_obstacle_mode_ids(
                  config.obs_modes, config.rare_mode, mode_catalog,
                  config.randomize_available_modes, config.num_modes, rng)
            : shared_modes;
        // A configured rare mode must be representable by this obstacle before
        // the rollout can force it. This also keeps randomized mode sets valid.
        if (!config.rare_mode.empty() &&
            mode_catalog.find(config.rare_mode) != mode_catalog.end() &&
            std::find(available_modes.begin(), available_modes.end(),
                      config.rare_mode) == available_modes.end()) {
            available_modes.push_back(config.rare_mode);
        }
        ObstacleSim& obstacle = obstacles[i];
        obstacle.available_modes = available_modes;
        obstacle.mode_models = mode_models_for(available_modes, mode_catalog);
        obstacle.current_mode = available_modes.empty()
            ? "constant_velocity"
            : available_modes[i % available_modes.size()];

        if (i < static_cast<int>(config.initial_obstacle_states.size())) {
            obstacle.state = config.initial_obstacle_states[i];
        } else if (obstacle_count == 1 && config.initial_obstacle_states.empty()) {
            obstacle.state = environment_default;
        } else {
            double fraction = config.default_arc_fraction;
            if (i < static_cast<int>(config.obs_arc_fractions.size())) {
                fraction = config.obs_arc_fractions[i];
            } else if (obstacle_count > 1) {
                fraction = std::min(0.20 + 0.15 * i, 0.85);
            }
            obstacle.state = obstacle_on_s_curve(reference_path, fraction, rng);
        }
    }
    return obstacles;
}

// ============================================================================
// Canonical Rollout Runner
// ============================================================================

RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config_in,
    unsigned seed
) {
    ExperimentConfig config = config_in;
    config.normalize();

    const SeedBundle seeds = derive_seeds(seed, 0);
    std::mt19937 plant_rng(seeds.env);
    RolloutRecord rec;
    rec.seed = seed;
    rec.plant_seed = seeds.env;
    rec.predictor_seed = seeds.predictor;
    rec.controller_seed = seeds.scenario;
    rec.config_source = config.config_source;
    rec.method = arm_name(config);
    rec.scenario = config.rollout.scenario_tag;
    rec.S = config.mpc.sampling.num_scenarios;
    rec.eps_wass = 0.0;
    rec.sigma = 0.0;
    rec.shift_rho = config.obstacles.shift.psi;
    rec.shift_boost = config.obstacles.shift.dangerous_boost;
    rec.ground_cost = ground_cost_name(config.dro.solver.ground_cost_type);

    auto mode_models = create_obstacle_mode_models(config.mpc.dt);

    RuntimeConfig mpc_cfg = config.to_scenario_mpc_config();
    mpc_cfg.random_seed = seeds.scenario;

    MPCController controller(mpc_cfg);

    const double metrics_v_ref = config.rollout.metrics_v_ref;
    const double dt = config.mpc.dt;
    EnvironmentSetup env_setup = create_environment(
        config.environment.type, plant_rng, config.environment);
    ReferencePath ref_path = config.environment.custom_ref_path.has_value()
        ? config.environment.custom_ref_path.value()
        : env_setup.path;
    controller.set_reference_path(ref_path);
    double path_length = ref_path.total_length();
    Eigen::Vector2d goal = ref_path.get_position_at(path_length);

    EgoState ego = config.environment.custom_initial_ego.value_or(env_setup.initial_ego);
    EgoDynamics dynamics(config.mpc.ego.dynamics, dt);
    double collision_radius = mpc_cfg.combined_radius();
    double path_progress = 0.0;

    auto obs_sims = construct_obstacles(
        config.obstacles, env_setup.initial_obs, ref_path, mode_models, plant_rng);
    const int n_obs = static_cast<int>(obs_sims.size());
    const int per_class = std::max(1, config.obstacles.obstacles_per_class);

    for (int i = 0; i < n_obs; ++i) {
        int obs_class = i / per_class;
        controller.initialize_obstacle(i, obs_class, obs_sims[i].mode_models);
    }

    for (int t = 0; t < 5; ++t) {
        for (int oi = 0; oi < n_obs; ++oi) {
            int obs_class = oi / per_class;
            controller.update_mode_observation(
                oi, obs_class, obs_sims[oi].current_mode, t);
        }
    }

    std::vector<double> clearances;
    std::vector<int> safe_horizons;
    double control_effort = 0.0;
    int constraint_active_total = 0;

    for (int step = 0; step < config.rollout.rollout_steps; ++step) {
        const bool switch_allowed =
            (config.obstacles.switch_regime ==
                 ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM) ||
            (config.mpc.horizon > 0 && step % config.mpc.horizon == 0);

        for (int oi = 0; oi < n_obs; ++oi) {
            if (!switch_allowed) {
            } else if (!config.obstacles.rare_mode.empty() &&
                       config.obstacles.rare_switch_prob > 0 &&
                       obs_sims[oi].mode_models.count(config.obstacles.rare_mode) != 0) {
                std::uniform_real_distribution<double> u(0, 1);
                if (u(plant_rng) < config.obstacles.rare_switch_prob) {
                    obs_sims[oi].current_mode = config.obstacles.rare_mode;
                } else {
                    obs_sims[oi].maybe_switch(config.obstacles.switch_prob, plant_rng);
                }
            } else {
                obs_sims[oi].maybe_switch(config.obstacles.switch_prob, plant_rng);
            }

            apply_distribution_shift(config.obstacles.shift, obs_sims[oi], plant_rng);

            int obs_class = oi / per_class;
            controller.update_mode_observation(
                oi, obs_class, obs_sims[oi].current_mode, step + 5);

            if (config.rollout.step_callback) {
                config.rollout.step_callback(
                    step, oi, obs_sims[oi], controller, plant_rng);
            }
        }

        path_progress = ref_path.find_closest_point(ego.position(), path_progress);

        std::map<int, ObstacleState> obstacles;
        for (int oi = 0; oi < n_obs; ++oi) {
            obstacles[oi] = obs_sims[oi].state;
        }

        auto mpc_result = controller.solve(
            ego, obstacles, goal, metrics_v_ref, path_progress, path_length);
        rec.eps_wass = mpc_result.ambiguity_radius_used;
        rec.solve_times_raw.push_back(mpc_result.solve_time);
        rec.total_dro_injected += mpc_result.num_dro_injected;
        if (mpc_result.safe_horizon > 0)
            safe_horizons.push_back(mpc_result.safe_horizon);

        constraint_active_total +=
            static_cast<int>(mpc_result.active_scenarios.size());

        {
            bool joint_found = false;
            for (const auto& sc : controller.scenarios()) {
                bool all_match = true;
                for (int oi = 0; oi < n_obs; ++oi) {
                    bool this_match = false;
                    for (const auto& [oid, traj] : sc.trajectories) {
                        if (oid == oi &&
                            traj.mode_id == obs_sims[oi].current_mode) {
                            this_match = true;
                            break;
                        }
                    }
                    if (!this_match) {
                        all_match = false;
                        break;
                    }
                }
                if (all_match) {
                    joint_found = true;
                    break;
                }
            }
            if (!joint_found) rec.joint_missed_mode_steps++;
            rec.joint_mode_checks++;
        }

        for (int oi = 0; oi < n_obs; ++oi) {
            bool mode_found = false;
            for (const auto& sc : controller.scenarios()) {
                for (const auto& [oid, traj] : sc.trajectories) {
                    if (oid == oi &&
                        traj.mode_id == obs_sims[oi].current_mode) {
                        mode_found = true;
                        break;
                    }
                }
                if (mode_found) break;
            }
            if (!mode_found) rec.missed_mode_steps++;
            rec.total_mode_checks++;

            if (!config.obstacles.rare_mode.empty() &&
                obs_sims[oi].current_mode == config.obstacles.rare_mode) {
                rec.rare_mode_active++;
                if (!mode_found) rec.rare_mode_missed++;
            }
        }

        {
            const ReferencePath& rp = ref_path;
            const double s_closest = rp.find_closest_point(ego.position());
            const PathPoint pp = rp.get_point_at(s_closest);
            const Eigen::Vector2d d = ego.position() - pp.position;
            const Eigen::Vector2d tangent(std::cos(pp.heading), std::sin(pp.heading));
            const Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
            const double e_c = d.dot(normal);
            const double e_l = d.dot(tangent);
            const double v_err = ego.v - metrics_v_ref;
            rec.sum_contouring_sq += e_c * e_c;
            rec.sum_lag_sq += e_l * e_l;
            rec.sum_velocity_err_sq += v_err * v_err;
            rec.metric_steps++;
        }

        if (mpc_result.success && mpc_result.first_input().has_value()) {
            auto input = mpc_result.first_input().value();
            control_effort += input.a * input.a + input.omega * input.omega;
            ego = dynamics.propagate(ego, input);
        }

        for (int oi = 0; oi < n_obs; ++oi) {
            obs_sims[oi].step(dt, plant_rng, config.obstacles.process_noise,
                              config.obstacles.speed_cap);
        }

        // Evaluate the outcome at t + 1, after both actors have advanced.
        // This makes collision and completion accounting agree with the state
        // passed to the next controller solve.
        path_progress = ref_path.find_closest_point(ego.position(), path_progress);
        for (int oi = 0; oi < n_obs; ++oi) {
            bool collision_this_obs = false;
            double min_dist_this_obs = std::numeric_limits<double>::infinity();
            if (config.mpc.ego.num_discs > 1) {
                const Eigen::Vector2d direction(
                    std::cos(ego.theta), std::sin(ego.theta));
                const double disc_spacing = config.mpc.ego.length /
                    static_cast<double>(config.mpc.ego.num_discs - 1);
                for (int disc = 0; disc < config.mpc.ego.num_discs; ++disc) {
                    const double offset = -0.5 * config.mpc.ego.length +
                        disc * disc_spacing;
                    const double distance = (ego.position() + offset * direction -
                        obs_sims[oi].state.position()).norm();
                    min_dist_this_obs = std::min(min_dist_this_obs, distance);
                    collision_this_obs = collision_this_obs || distance < collision_radius;
                }
            } else {
                min_dist_this_obs =
                    (ego.position() - obs_sims[oi].state.position()).norm();
                collision_this_obs = min_dist_this_obs < collision_radius;
            }
            clearances.push_back(min_dist_this_obs);
            if (min_dist_this_obs < rec.min_clearance) {
                rec.min_clearance = min_dist_this_obs;
                rec.min_clearance_step = step + 1;
            }
            if (collision_this_obs && !rec.collision) {
                rec.collision = true;
                rec.collision_step = step + 1;
            }
        }
        rec.total_steps++;

        if (config.environment.path_completion_termination &&
            path_progress >=
                config.environment.path_completion_fraction * path_length) {
            rec.completed_path = true;
            break;
        }
    }

    rec.total_progress = path_length > 0 ? path_progress / path_length : ego.x;
    rec.control_effort = control_effort;
    rec.constraint_active_count = constraint_active_total;
    rec.active_constraints = rec.total_steps > 0
        ? constraint_active_total / rec.total_steps : 0;

    if (!rec.solve_times_raw.empty()) {
        std::vector<double> times_ms;
        times_ms.reserve(rec.solve_times_raw.size());
        for (double t : rec.solve_times_raw) times_ms.push_back(t * 1000.0);

        rec.avg_solve_ms =
            std::accumulate(times_ms.begin(), times_ms.end(), 0.0) /
            times_ms.size();
        std::sort(times_ms.begin(), times_ms.end());
        int n = static_cast<int>(times_ms.size());
        rec.p50_solve_ms = times_ms[n / 2];
        rec.p95_solve_ms =
            times_ms[std::min(n - 1, static_cast<int>(0.95 * n))];
        rec.max_solve_ms = times_ms.back();
    }

    if (!safe_horizons.empty()) {
        rec.avg_safe_horizon =
            std::accumulate(safe_horizons.begin(), safe_horizons.end(), 0.0) /
            safe_horizons.size();
    }

    if (!clearances.empty()) {
        std::vector<double> sorted_clear = clearances;
        std::sort(sorted_clear.begin(), sorted_clear.end());
        int idx_5pct =
            std::max(0, static_cast<int>(0.05 * sorted_clear.size()) - 1);
        rec.clearance_5pct = sorted_clear[idx_5pct];
    }

    return rec;
}


// ============================================================================
// Convenience Rollout Helpers
// ============================================================================

EnvironmentSetup create_environment(
    EnvironmentType env,
    std::mt19937& rng,
    const EnvironmentExperimentConfig& path_cfg
) {
    EnvironmentSetup setup;
    std::uniform_real_distribution<double> jitter(-0.3, 0.3);

    const ReferencePath path = build_environment_reference_path(env, path_cfg);
    setup.path = path;
    setup.obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"};
    const PathPoint path_start = path.get_point_at(0.0);
    setup.initial_ego = path_cfg.custom_initial_ego.value_or(
        EgoState(path_start.position.x(), path_start.position.y(),
                 path_start.heading, path_cfg.ego_initial_v));
    setup.goal = path.get_position_at(path.total_length());
    setup.name = environment_name(env);

    auto place_overtake = [&] {
        setup.initial_obs = obstacle_on_s_curve(path, OBS_PATH_FRACTION, rng);
    };
    auto place_narrow = [&] {
        double s25 = 0.25 * path.total_length();
        PathPoint pp = path.get_point_at(s25);
        Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
        Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
        Eigen::Vector2d pos = pp.position + jitter(rng) * 0.15 * n;
        setup.initial_obs = ObstacleState(
            pos.x(), pos.y(), -0.2 * t.x(), -0.2 * t.y());
    };
    auto place_intersection = [&] {
        double s40 = 0.40 * path.total_length();
        PathPoint pp = path.get_point_at(s40);
        Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
        Eigen::Vector2d pos = pp.position + (2.0 + jitter(rng)) * n;
        setup.initial_obs = ObstacleState(
            pos.x(), pos.y(),
            -1.0 * n.x() + jitter(rng) * 0.2,
            -1.0 * n.y() + jitter(rng) * 0.2);
    };
    auto place_oncoming = [&] {
        double s60 = 0.60 * path.total_length();
        PathPoint pp = path.get_point_at(s60);
        Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
        Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
        Eigen::Vector2d pos = pp.position + jitter(rng) * 0.3 * n;
        setup.initial_obs = ObstacleState(
            pos.x(), pos.y(),
            -1.0 * t.x() + jitter(rng) * 0.2,
            -1.0 * t.y() + jitter(rng) * 0.2);
    };

    switch (env) {
        case EnvironmentType::S_CURVE:
        case EnvironmentType::TWO_LANE_HIGHWAY:
        case EnvironmentType::FOUR_LANE_HIGHWAY:
        case EnvironmentType::OVERTAKE_SLOW_LEAD:
            place_overtake();
            break;
        case EnvironmentType::NARROW_CORRIDOR:
            place_narrow();
            break;
        case EnvironmentType::T_INTERSECTION:
        case EnvironmentType::FOUR_WAY_INTERSECTION:
        case EnvironmentType::TWO_LANE_ROUNDABOUT:
        case EnvironmentType::FOUR_LANE_ROUNDABOUT:
        case EnvironmentType::INTERSECTION:
            place_intersection();
            break;
        case EnvironmentType::ENTER_RAMP:
        case EnvironmentType::EXIT_RAMP:
        case EnvironmentType::ONCOMING:
            place_oncoming();
            break;
    }
    return setup;
}

RolloutResult run_single_rollout_env(
    ExperimentConfig cfg,
    unsigned seed,
    const EnvironmentSetup& env_setup,
    SamplingBaseline,
    int forced_safe_horizon
) {
    // An explicit YAML/in-memory mode list takes precedence over an
    // environment's legacy suggestion.
    if (cfg.obstacles.obs_modes.empty()) {
        cfg.obstacles.obs_modes = env_setup.obs_modes;
    }
    if (cfg.obstacles.initial_obstacle_states.empty()) {
        cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
    }
    cfg.environment.custom_initial_ego = env_setup.initial_ego;
    if (env_setup.path.total_length() > 0.0) {
        cfg.environment.custom_ref_path = env_setup.path;
    }
    cfg.mpc.constraints.forced_safe_horizon = forced_safe_horizon;
    if (forced_safe_horizon >= 0) {
        cfg.mpc.safe_horizon_enabled = true;
    }

    return run_configured_rollout(std::move(cfg), seed);
}

}  // namespace dro_mpc
