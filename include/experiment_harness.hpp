/**
 * @file experiment_harness.hpp
 * @brief World / experiment / test layer over the controller runtime config.
 *
 * config.hpp owns the controller runtime (RuntimeConfig = mpc + dro + solver).
 * THIS header owns everything around a trial: the ground-truth obstacle process,
 * the environment/road setup, the test (experiment) composition, result logging,
 * result reporting/statistics, and the rollout helper functions.
 *
 * ExperimentConfig composes the controller sections (MPCConfig, DROControllerConfig,
 * SolverSettings) with world sections (obstacles, environment, sampling, rollout)
 * and maps them down into a RuntimeConfig via to_scenario_mpc_config().
 *
 * ── File layout ──────────────────────────────────────────────────────────────
 *   1. Obstacle configurations   — mode-switch / history enums, obstacle config,
 *                                   ground-truth ObstacleSim
 *   2. Environment configurations — road / path / initial-state setup
 *   3. Test configurations       — sampling baselines, rollout protocol, the
 *                                   assembled ExperimentConfig + arm builders
 *   4. Test logging              — CSV writer
 *   5. Results reporting         — RolloutRecord / RolloutResult + statistics
 *   6. Helper functions          — seeds, shift, path/placement, the rollout runner
 *
 * run_experiment_rollout() is THE canonical rollout. paper_experiment_runner and
 * all tests configure an ExperimentConfig and call it — they do NOT duplicate
 * obstacle simulation, collision detection, or mode tracking.
 */

#ifndef DRO_MPC_EXPERIMENT_HARNESS_HPP
#define DRO_MPC_EXPERIMENT_HARNESS_HPP

#include "types.hpp"
#include "config.hpp"
#include "reference_path.hpp"

#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace dro_mpc {

// Forward declarations (controller lives in mpc_controller.hpp; only referenced
// by pointer/reference in callback signatures here).
class AdaptiveScenarioMPC;

// ############################################################################
// # 1. OBSTACLE CONFIGURATIONS
// ############################################################################

// ---- Enums -----------------------------------------------------------------

/**
 * @brief When the ground-truth obstacle is allowed to change mode.
 *
 * MARKOV_JUMP_SYSTEM — switch may fire every rollout step (realistic; CDC'26).
 * HOLD_OVER_HORIZON  — mode frozen for `horizon` steps so it is constant across
 *                      any one prediction horizon (Theorem-1 regime).
 */
enum class ModeSwitchConfiguration {
    MARKOV_JUMP_SYSTEM,
    HOLD_OVER_HORIZON
};

enum class ObstacleHistoryConfiguration {
    INDEPENDENT,  ///< Several obstacles, one class each
    SHARED        ///< Several obstacles sharing class history
};

// ---- Enum names ------------------------------------------------------------

inline std::string switch_regime_name(ModeSwitchConfiguration r) {
    switch (r) {
        case ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM: return "MarkovJump";
        case ModeSwitchConfiguration::HOLD_OVER_HORIZON:  return "HoldOverHorizon";
        default: return "HoldOverHorizon";
    }
}

inline std::string obstacle_layout_name(ObstacleHistoryConfiguration L) {
    switch (L) {
        case ObstacleHistoryConfiguration::INDEPENDENT: return "independent_history";
        case ObstacleHistoryConfiguration::SHARED:      return "shared_history_classes";
        default: return "independent_history";
    }
}

// ---- Settings --------------------------------------------------------------

struct DistributionShiftConfig {
    double rho = 0.0;
    double dangerous_boost = 0.0;
    int boosted_mode = -1;
};

/**
 * @brief Ground-truth obstacle / mode-process knobs for an experiment.
 */
struct ObstacleExperimentConfig {
    ObstacleHistoryConfiguration history =
        ObstacleHistoryConfiguration::INDEPENDENT;
    ModeSwitchConfiguration switch_regime =
        ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM;
    double switch_prob = 0.1;
    int num_modes = 4;
    int num_obstacles = 1;
    int obstacles_per_class = 1;

    std::vector<std::string> obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"
    };
    std::string rare_mode = "lane_change_left";
    double rare_switch_prob = 0.05;

    std::vector<double> obs_arc_fractions;               ///< Empty => auto placement
    std::vector<ObstacleState> initial_obstacle_states;  ///< Overrides arc fracs
    DistributionShiftConfig shift;

    void apply_layout() {
        if (num_obstacles <= 1) {
            num_obstacles = 1;
            obstacles_per_class = 1;
            return;
        }
        switch (history) {
            case ObstacleHistoryConfiguration::INDEPENDENT:
                obstacles_per_class = 1;
                break;
            case ObstacleHistoryConfiguration::SHARED:
                obstacles_per_class = num_obstacles;
                break;
        }
    }
};

// ---- Ground-truth simulator ------------------------------------------------

/**
 * @brief Ground-truth obstacle simulator with mode switching.
 *
 * Used by all rollouts. Propagates the obstacle state under the current mode
 * dynamics with small process noise and optional mode switching.
 */
struct ObstacleSim {
    ObstacleState state;
    std::string current_mode;
    std::vector<std::string> available_modes;
    std::map<std::string, ModeModel> mode_models;

    /// Propagate one step under current mode dynamics with noise.
    void step(double dt, std::mt19937& rng);

    /// Switch to a random mode with the given probability.
    void maybe_switch(double switch_prob, std::mt19937& rng);
};

// ############################################################################
// # 2. ENVIRONMENT CONFIGURATIONS
// ############################################################################

// ---- Enums -----------------------------------------------------------------

// NOTE: every EnvironmentType runs on the SAME S-curve reference path
// (create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, ...)). The variants differ
// ONLY in obstacle placement/velocity, never in road geometry. OVERTAKE_SLOW_LEAD
// places a slow lead obstacle (v in [0.1,0.4] m/s) travelling with the ego at
// arc-fraction 0.35. Do not confuse with test_generalization.cpp's "Straight",
// which IS a straight road carrying an oncoming obstacle.
enum class EnvironmentType {
    OVERTAKE_SLOW_LEAD,
    NARROW_CORRIDOR,
    INTERSECTION,
    ONCOMING
};

// ---- Enum names ------------------------------------------------------------

inline std::string environment_name(EnvironmentType env) {
    switch (env) {
        case EnvironmentType::OVERTAKE_SLOW_LEAD: return "OvertakeSlowLead";
        case EnvironmentType::NARROW_CORRIDOR:    return "Narrow";
        case EnvironmentType::INTERSECTION:       return "Intersection";
        case EnvironmentType::ONCOMING:           return "Oncoming";
        default: return "unknown";
    }
}

// ---- World / path constants ------------------------------------------------

// ================ ACC / paper defaults ======================================
inline constexpr int    DEFAULT_ROLLOUT_STEPS  = 200;
inline constexpr double DEFAULT_DT             = 0.1;
inline constexpr int    DEFAULT_HORIZON        = 15;
inline constexpr int    DEFAULT_BASE_SCENARIOS = 40;
inline constexpr double S_CURVE_LENGTH         = 25.0;
inline constexpr double S_CURVE_AMPLITUDE      = 3.0;
inline constexpr int    S_CURVE_POINTS         = 200;
inline constexpr double PATH_COMPLETE_FRAC     = 0.95;
inline constexpr double OBS_PATH_FRACTION      = 0.35;

/// Default arc fractions for placing 4 obstacles along the S-curve.
inline const std::vector<double> OBS_ARC_FRACS_4 = {0.20, 0.35, 0.50, 0.65};

// ---- Settings --------------------------------------------------------------

struct EnvironmentSetup {
    ObstacleState initial_obs;
    std::vector<std::string> obs_modes;
    EgoState initial_ego;
    Eigen::Vector2d goal;
    std::string name;
};

/**
 * @brief Road / reference-path / initial-state knobs for an experiment.
 */
struct EnvironmentExperimentConfig {
    EnvironmentType type = EnvironmentType::OVERTAKE_SLOW_LEAD;
    std::optional<ReferencePath> custom_ref_path;
    std::optional<EgoState> custom_initial_ego;
    bool path_completion_termination = true;
    double path_completion_fraction = 0.95;
};

// ############################################################################
// # 3. TEST CONFIGURATIONS
// ############################################################################

// ---- Sampling baselines ----------------------------------------------------

enum class SamplingBaseline {
    STANDARD,
    STRATIFIED,
    TEMPERATURE,
    EPSILON_GREEDY,
    RISK_BIASED,
    UNIFORM_WEIGHT,
    RECENCY_WEIGHT,
    ORACLE_FLOOD
};

inline std::string baseline_name(SamplingBaseline b) {
    switch (b) {
        case SamplingBaseline::STANDARD:       return "Standard";
        case SamplingBaseline::STRATIFIED:     return "Stratified";
        case SamplingBaseline::TEMPERATURE:    return "Temperature";
        case SamplingBaseline::EPSILON_GREEDY: return "EpsilonGreedy";
        case SamplingBaseline::RISK_BIASED:    return "RiskBiased";
        case SamplingBaseline::UNIFORM_WEIGHT: return "Uniform";
        case SamplingBaseline::RECENCY_WEIGHT: return "Recency";
        case SamplingBaseline::ORACLE_FLOOD:   return "Oracle";
        default: return "unknown";
    }
}

inline WeightType baseline_to_weight(SamplingBaseline bl) {
    switch (bl) {
        case SamplingBaseline::STANDARD:       return WeightType::FREQUENCY;
        case SamplingBaseline::STRATIFIED:     return WeightType::FREQUENCY;
        case SamplingBaseline::TEMPERATURE:    return WeightType::TEMPERATURE;
        case SamplingBaseline::EPSILON_GREEDY: return WeightType::EPSILON_GREEDY;
        case SamplingBaseline::UNIFORM_WEIGHT: return WeightType::UNIFORM;
        case SamplingBaseline::RECENCY_WEIGHT: return WeightType::RECENCY;
        case SamplingBaseline::RISK_BIASED:
        case SamplingBaseline::ORACLE_FLOOD:
        default:                               return WeightType::FREQUENCY;
    }
}

/**
 * @brief Thin experiment overlay that can override mpc.sampling.weight_type.
 */
struct SamplingExperimentConfig {
    SamplingBaseline baseline = SamplingBaseline::STANDARD;

    WeightType weight_type() const { return baseline_to_weight(baseline); }
};

// ---- DRO arm label ---------------------------------------------------------

/// Coarse experiment-arm label: is DRO reweighting on the controller or not.
/// (The runtime switch is DROControllerConfig::enabled; this is naming sugar
/// for arm builders.)
enum class DROConfiguration {
    BASE,  ///< Safe-horizon (scenario) MPCC, no DRO reweighting
    DRO    ///< DRO-enabled
};

inline std::string dro_configuration_name(DROConfiguration c) {
    switch (c) {
        case DROConfiguration::BASE: return "base_sh_mpcc";
        case DROConfiguration::DRO:  return "dro_enabled";
        default: return "unknown";
    }
}

// ---- Rollout protocol ------------------------------------------------------

struct RolloutExperimentConfig {
    int rollout_steps = 60;
    std::string scenario_tag = "baseline";
    std::string method_name;  ///< Empty => auto from DRO/MPC labels

    /// Called after mode observation, before solve.
    /// Args: (step, obstacle_id, obstacle_sim, controller, rng)
    std::function<void(int, int, ObstacleSim&, AdaptiveScenarioMPC&, std::mt19937&)>
        step_callback;
};

struct SeedBundle {
    unsigned master;
    unsigned env;
    unsigned predictor;
    unsigned scenario;
};

// ---- Assembled experiment configuration ------------------------------------

/**
 * @brief Full experiment configuration — composition of section configs only.
 *
 * Build by filling the nested members (or use make_arm_config helpers):
 *
 *   ExperimentConfig cfg;
 *   cfg.dro.enabled = true;
 *   cfg.dro.injection_mode = InjectionMode::QSTAR_SAMPLE;
 *   cfg.dro.solver.base_radius = 0.1;
 *   cfg.mpc.type = MPCConfiguration::SH_MPCC;
 *   cfg.mpc.sampling.num_scenarios = 40;
 *   cfg.obstacles.switch_prob = 0.2;
 *   cfg.rollout.rollout_steps = 200;
 */
struct ExperimentConfig {
    MPCConfig mpc;
    DROControllerConfig dro;
    SolverSettings solver{};
    double obstacle_radius = 0.35;
    ObstacleExperimentConfig obstacles;
    EnvironmentExperimentConfig environment;
    SamplingExperimentConfig sampling;
    RolloutExperimentConfig rollout;

    /// Apply layout rules; sync belief / fixed rho; auto-name the method.
    /// Does not call mpc.sync_from_type() (would overwrite SH overrides set
    /// after type selection).
    void normalize() {
        obstacles.apply_layout();
        dro.apply_fixed_rho();
        mpc.sampling.sync_belief();
        if (sampling.baseline != SamplingBaseline::STANDARD) {
            mpc.sampling.weight_type = sampling.weight_type();
        }
        if (rollout.method_name.empty()) {
            rollout.method_name =
                dro_configuration_name(
                    dro.enabled ? DROConfiguration::DRO
                                : DROConfiguration::BASE) +
                "_" + mpc_configuration_name(mpc.type);
            if (dro.enabled) {
                rollout.method_name += "_" +
                    reweighted_distribution_use_name(dro.reweighting);
            }
        }
    }

    /// Map the nested experiment sections into the controller runtime config.
    RuntimeConfig to_scenario_mpc_config(double dt = DEFAULT_DT) const {
        RuntimeConfig cfg;
        cfg.mpc = mpc;
        cfg.dro = dro;
        cfg.solver = solver;
        cfg.obstacle_radius = obstacle_radius;
        cfg.mpc.dt = dt;
        if (sampling.baseline != SamplingBaseline::STANDARD) {
            cfg.mpc.sampling.weight_type = sampling.weight_type();
        }
        cfg.normalize();  // belief sync + fixed_rho only
        return cfg;
    }
};

// ---- Arm builders ----------------------------------------------------------

/**
 * @brief Compose a standard Base / Base+SH / DRO / DRO+SH arm.
 *
 * Prefer filling ExperimentConfig sections directly for new experiments.
 */
inline ExperimentConfig make_arm_config(
    DROConfiguration dro_kind,
    MPCConfiguration mpc_kind,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "",
    double rare_prob = 0.0,
    int num_discs = 1,
    double vehicle_length = 1.5
) {
    ExperimentConfig cfg;
    cfg.dro.enabled = (dro_kind == DROConfiguration::DRO);
    if (cfg.dro.enabled) {
        cfg.dro.reweighting = ReweightedDistributionUse::SAMPLING_ONLY;
        cfg.dro.injection_mode = InjectionMode::QSTAR_SAMPLE;
    } else {
        cfg.dro.injection_mode = InjectionMode::NONE;
    }

    cfg.mpc.type = mpc_kind;
    cfg.mpc.sync_from_type();
    cfg.mpc.horizon = DEFAULT_HORIZON;
    cfg.mpc.sampling.num_scenarios = num_scenarios;
    cfg.mpc.ego.num_discs = num_discs;
    cfg.mpc.ego.length = vehicle_length;
    cfg.mpc.sampling.weight_type = WeightType::FREQUENCY;

    cfg.obstacles.switch_prob = switch_prob;
    cfg.obstacles.obs_modes = obs_modes;
    cfg.obstacles.rare_mode = rare_mode;
    cfg.obstacles.rare_switch_prob = rare_prob;

    cfg.environment.path_completion_termination = true;
    cfg.environment.path_completion_fraction = PATH_COMPLETE_FRAC;

    cfg.rollout.rollout_steps = rollout_steps;
    cfg.normalize();
    return cfg;
}

inline ExperimentConfig make_base_config(
    double switch_prob, int num_scenarios, int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "", double rare_prob = 0.0,
    int num_discs = 1, double vehicle_length = 1.5
) {
    return make_arm_config(
        DROConfiguration::BASE, MPCConfiguration::MPCC,
        switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob, num_discs, vehicle_length);
}

inline ExperimentConfig make_base_sh_config(
    double switch_prob, int num_scenarios, int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "", double rare_prob = 0.0,
    int num_discs = 1, double vehicle_length = 1.5
) {
    return make_arm_config(
        DROConfiguration::BASE, MPCConfiguration::SH_MPCC,
        switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob, num_discs, vehicle_length);
}

inline ExperimentConfig make_dro_config(
    double switch_prob, int num_scenarios, int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "", double rare_prob = 0.0,
    int num_discs = 1, double vehicle_length = 1.5
) {
    return make_arm_config(
        DROConfiguration::DRO, MPCConfiguration::MPCC,
        switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob, num_discs, vehicle_length);
}

inline ExperimentConfig make_dro_sh_config(
    double switch_prob, int num_scenarios, int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "", double rare_prob = 0.0,
    int num_discs = 1, double vehicle_length = 1.5
) {
    return make_arm_config(
        DROConfiguration::DRO, MPCConfiguration::SH_MPCC,
        switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob, num_discs, vehicle_length);
}

inline std::string arm_name(const ExperimentConfig& cfg) {
    if (!cfg.rollout.method_name.empty()) return cfg.rollout.method_name;
    const bool sh = cfg.mpc.uses_safe_horizon();
    if (!cfg.dro.enabled) return sh ? "Base+SH" : "Base";
    return sh ? "DRO+SH" : "DRO";
}

// ############################################################################
// # 4. TEST LOGGING
// ############################################################################

struct RolloutRecord;  // defined in section 5

class CSVWriter {
public:
    explicit CSVWriter(const std::string& filepath);
    ~CSVWriter();

    void write_header();
    void write_record(const RolloutRecord& rec);
    void flush();

private:
    std::ofstream ofs_;
};

// ############################################################################
// # 5. RESULTS REPORTING
// ############################################################################

// ---- Per-rollout record ----------------------------------------------------

/**
 * @brief Per-rollout record with all metrics.
 */
struct RolloutRecord {
    unsigned seed = 0;
    std::string method;
    std::string scenario = "baseline";
    int S = 0;
    double eps_wass = 0.0;
    double sigma = 0.0;
    double shift_rho = 0.0;
    double shift_boost = 0.0;
    std::string ground_cost;

    // Safety metrics
    bool collision = false;
    int collision_step = -1;
    double min_clearance = 1e9;
    int min_clearance_step = -1;

    // Progress metrics
    double total_progress = 0.0;
    bool completed_path = false;
    double control_effort = 0.0;
    // Conservatism metrics (CDC'26 Reviewer 3, major comment 5)
    double sum_contouring_sq = 0.0;
    double sum_lag_sq = 0.0;
    double sum_velocity_err_sq = 0.0;
    int metric_steps = 0;
    double mean_contouring_error() const {
        return metric_steps > 0 ? std::sqrt(sum_contouring_sq / metric_steps) : 0.0;
    }
    double mean_velocity_error() const {
        return metric_steps > 0 ? std::sqrt(sum_velocity_err_sq / metric_steps) : 0.0;
    }
    int constraint_active_count = 0;
    int missed_mode_steps = 0;
    int total_mode_checks = 0;
    int joint_missed_mode_steps = 0;
    int joint_mode_checks = 0;
    int total_steps = 0;

    // Rare mode tracking
    int rare_mode_active = 0;
    int rare_mode_missed = 0;

    // Timing metrics
    double avg_solve_ms = 0.0;
    double p50_solve_ms = 0.0;
    double p95_solve_ms = 0.0;
    double max_solve_ms = 0.0;
    std::vector<double> solve_times_raw;

    // DRO-specific
    int total_dro_injected = 0;
    double avg_safe_horizon = 0.0;
    double clearance_5pct = 0.0;
    int active_constraints = 0;
};

/**
 * @brief Thin adapter over RolloutRecord for lightweight callers.
 */
struct RolloutResult {
    bool collision = false;
    bool completed_path = false;
    double min_clearance = 1e9;
    double total_progress = 0.0;
    double avg_solve_time = 0.0;
    double max_solve_time = 0.0;
    int missed_mode_steps = 0;
    int total_mode_checks = 0;
    int rare_mode_active = 0;
    int rare_mode_missed = 0;
    int total_steps = 0;
    int active_constraints = 0;
    std::vector<double> solve_times;

    static RolloutResult from_record(const RolloutRecord& rec) {
        RolloutResult r;
        r.collision = rec.collision;
        r.completed_path = rec.completed_path;
        r.min_clearance = rec.min_clearance;
        r.total_progress = rec.total_progress;
        r.avg_solve_time = rec.avg_solve_ms / 1000.0;
        r.max_solve_time = rec.max_solve_ms / 1000.0;
        r.missed_mode_steps = rec.missed_mode_steps;
        r.total_mode_checks = rec.total_mode_checks;
        r.rare_mode_active = rec.rare_mode_active;
        r.rare_mode_missed = rec.rare_mode_missed;
        r.total_steps = rec.total_steps;
        r.active_constraints = rec.active_constraints;
        r.solve_times = rec.solve_times_raw;
        return r;
    }
};

// ---- Statistics ------------------------------------------------------------

std::pair<double, double> wilson_ci(int successes, int n, double z = 1.96);

struct BootstrapResult {
    double mean_delta;
    double ci_low;
    double ci_high;
};

BootstrapResult bootstrap_paired_delta(
    const std::vector<bool>& base_collisions,
    const std::vector<bool>& dro_collisions,
    int n_bootstrap = 10000,
    std::mt19937* rng = nullptr
);

double mcnemar_chi2(int b, int c);

struct EffectSizes {
    double abs_delta;
    double rel_delta;
    double risk_ratio;
    double cohens_h;
};

EffectSizes compute_effect_sizes(double p_base, double p_comparison);

// ############################################################################
// # 6. HELPER FUNCTIONS
// ############################################################################

// ---- Seeds / shift / placement ---------------------------------------------

SeedBundle derive_seeds(unsigned master_seed, int idx);

/// Apply distribution shift to an obstacle simulator.
void apply_distribution_shift(
    const DistributionShiftConfig& shift,
    ObstacleSim& obs_sim,
    std::mt19937& rng
);

/// Place an obstacle on an S-curve path at the given arc-length fraction.
ObstacleState obstacle_on_s_curve(
    const ReferencePath& path,
    double arc_fraction,
    std::mt19937& rng
);

/// Compute percentile of a vector (p in [0,100]).
double percentile(std::vector<double> v, double p);

// ---- Rollout runner --------------------------------------------------------

/**
 * @brief Run a single MPC rollout with the given configuration and seed.
 *
 * This is THE canonical rollout function. All experiments should call this.
 */
RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config,
    unsigned seed
);

/// Create a reference path and set it on the controller.
ReferencePath setup_mpcc_path(AdaptiveScenarioMPC& ctrl);

/// Run a single rollout from a fully composed ExperimentConfig.
inline RolloutResult run_configured_rollout(
    ExperimentConfig cfg,
    unsigned seed
) {
    cfg.normalize();
    return RolloutResult::from_record(run_experiment_rollout(cfg, seed));
}

/// Run a multi-obstacle rollout from a composed ExperimentConfig.
inline RolloutResult run_multi_obstacle_rollout(
    ExperimentConfig cfg,
    unsigned seed,
    int num_obstacles = 4,
    int num_classes = 4,
    const std::vector<double>& arc_fracs = OBS_ARC_FRACS_4
) {
    cfg.obstacles.num_obstacles = num_obstacles;
    cfg.obstacles.obstacles_per_class =
        (num_classes > 0 && num_classes < num_obstacles)
            ? (num_obstacles / num_classes) : 1;
    cfg.obstacles.obs_arc_fractions = arc_fracs;
    cfg.obstacles.history = ObstacleHistoryConfiguration::INDEPENDENT;
    return run_configured_rollout(std::move(cfg), seed);
}

/// Create an environment setup for a given type.
EnvironmentSetup create_environment(EnvironmentType env, std::mt19937& rng);

/// Run a rollout with a custom environment setup layered onto a base config.
RolloutResult run_single_rollout_env(
    ExperimentConfig cfg,
    unsigned seed,
    const EnvironmentSetup& env_setup,
    SamplingBaseline baseline = SamplingBaseline::STANDARD,
    int forced_safe_horizon = -1
);

}  // namespace dro_mpc

#endif  // DRO_MPC_EXPERIMENT_HARNESS_HPP
