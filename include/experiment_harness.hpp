/**
 * @file experiment_harness.hpp
 * @brief owns everything around a trial: the ground-truth obstacle process,
 * the environment/road setup, the test (experiment) composition, result logging,
 * result reporting/statistics, and the rollout helper functions.
 *
 * ExperimentConfig composes the controller sections (MPCConfig, DROControllerConfig,
 * SolverSettings) with world sections (obstacles, environment, sampling, rollout)
 * and maps them down into a RuntimeConfig via to_scenario_mpc_config().
 *
 * Numeric base settings come from configs/default.yaml (see default_experiment_config()).
 * The DEFAULT_* names below are filled from that file at startup.
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

enum class ModeSwitchConfiguration {
    MARKOV_JUMP_SYSTEM,
    HOLD_OVER_HORIZON
};

enum class ObstacleHistoryConfiguration {
    INDEPENDENT,  // Several obstacles, one class each
    SHARED        // Several obstacles sharing class history
};

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

struct DistributionShiftConfig {
    double psi = 0.0;
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
        ModeSwitchConfiguration::HOLD_OVER_HORIZON;
    double switch_prob = 0.1;
    int num_modes = 4;
    int num_obstacles = 1;
    int obstacles_per_class = 1;

    std::vector<std::string> obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"
    };
    std::string rare_mode = "lane_change_left";
    double rare_switch_prob = 0.05;

    std::vector<double> obs_arc_fractions;               // Empty => auto placement
    std::vector<ObstacleState> initial_obstacle_states;  // Overrides arc fracs
    DistributionShiftConfig shift;

    double default_arc_fraction = 0.35;
    double process_noise = 0.02;
    double speed_cap = 2.0;

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

/**
 * @brief Ground-truth obstacle simulator with mode switching.
 */
struct ObstacleSim {
    ObstacleState state;
    std::string current_mode;
    std::vector<std::string> available_modes;
    std::map<std::string, ModeModel> mode_models;

    /// Propagate one step under current mode dynamics with noise.
    void step(double dt, std::mt19937& rng,
              double process_noise = 0.02, double speed_cap = 2.0);

    /// Switch to a random mode with the given probability.
    void maybe_switch(double switch_prob, std::mt19937& rng);
};

// ############################################################################
// # 2. ENVIRONMENT CONFIGURATIONS
// ############################################################################

enum class EnvironmentType {
    T_INTERSECTION,
    FOUR_WAY_INTERSECTION,
    S_CURVE,
    TWO_LANE_ROUNDABOUT,
    FOUR_LANE_ROUNDABOUT,
    TWO_LANE_HIGHWAY,
    FOUR_LANE_HIGHWAY,
    ENTER_RAMP,
    EXIT_RAMP,
    // Legacy placement recipes used by existing experiments
    OVERTAKE_SLOW_LEAD,
    NARROW_CORRIDOR,
    INTERSECTION,
    ONCOMING,
};

inline std::string environment_name(EnvironmentType env) {
    switch (env) {
        case EnvironmentType::T_INTERSECTION: return "T_Intersection";
        case EnvironmentType::FOUR_WAY_INTERSECTION: return "FourWayIntersection";
        case EnvironmentType::S_CURVE: return "SCurve";
        case EnvironmentType::TWO_LANE_ROUNDABOUT: return "TwoLaneRoundabout";
        case EnvironmentType::FOUR_LANE_ROUNDABOUT: return "FourLaneRoundabout";
        case EnvironmentType::TWO_LANE_HIGHWAY: return "TwoLaneHighway";
        case EnvironmentType::FOUR_LANE_HIGHWAY: return "FourLaneHighway";
        case EnvironmentType::ENTER_RAMP: return "EnterRamp";
        case EnvironmentType::EXIT_RAMP: return "ExitRamp";
        case EnvironmentType::OVERTAKE_SLOW_LEAD: return "OvertakeSlowLead";
        case EnvironmentType::NARROW_CORRIDOR: return "Narrow";
        case EnvironmentType::INTERSECTION: return "Intersection";
        case EnvironmentType::ONCOMING: return "Oncoming";
        default: return "SCurve";
    }
}

struct EnvironmentSetup {
    ObstacleState initial_obs;
    std::vector<std::string> obs_modes;
    EgoState initial_ego;
    Eigen::Vector2d goal;
    std::string name;
    ReferencePath path;
};

struct EnvironmentExperimentConfig {
    EnvironmentType type = EnvironmentType::S_CURVE;
    std::optional<ReferencePath> custom_ref_path;
    std::optional<EgoState> custom_initial_ego;
    bool path_completion_termination = true;
    double path_completion_fraction = 0.95;
    double s_curve_length = 25.0;
    double s_curve_amplitude = 3.0;
    int s_curve_points = 200;
    double ego_initial_v = 1.5;
};

// Loaded from configs/default.yaml at process start (see experiment_harness.cpp).
extern int    DEFAULT_ROLLOUT_STEPS;
extern double DEFAULT_DT;
extern int    DEFAULT_HORIZON;
extern int    DEFAULT_BASE_SCENARIOS;
extern double S_CURVE_LENGTH;
extern double S_CURVE_AMPLITUDE;
extern int    S_CURVE_POINTS;
extern double PATH_COMPLETE_FRAC;
extern double OBS_PATH_FRACTION;
extern std::vector<double> OBS_ARC_FRACS_4;

// ############################################################################
// # 3. TEST CONFIGURATIONS
// ############################################################################

/// Paper-arm label: DRO on vs off. Controller knob is dro.enabled.
enum class DROConfiguration { BASE, DRO };

inline std::string dro_configuration_name(DROConfiguration d) {
    return d == DROConfiguration::DRO ? "dro" : "base";
}

/// Historical experiment-table labels. Not a controller knob: when
/// dro.enabled is true the controller always resamples i.i.d. from q*.
enum class InjectionMode {
    NONE,
    QSTAR_SAMPLE,
    TOP_RISK_INJECT,
    DIVERSE_RISK_INJECT,
    SOFTMAX_RISK,
    EPSILON_GREEDY_INJ,
    UNIFORM_COVERAGE
};

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

struct SamplingExperimentConfig {
    SamplingBaseline baseline = SamplingBaseline::STANDARD;
};

struct RolloutExperimentConfig {
    int rollout_steps = 200;
    std::string scenario_tag = "baseline";
    std::string method_name;  ///< Empty => auto from DRO/MPC labels
    double metrics_v_ref = 1.5;

    /// Called after mode observation, before solve.
    std::function<void(int, int, ObstacleSim&, AdaptiveScenarioMPC&, std::mt19937&)>
        step_callback;
};

struct SeedBundle {
    unsigned master;
    unsigned env;
    unsigned predictor;
    unsigned scenario;
};

/**
 * @brief Full experiment configuration — composition of section configs only.
 *
 * Prefer default_experiment_config() (loads configs/default.yaml) over
 * ExperimentConfig{} so numeric knobs come from the YAML, not C++ fallbacks.
 *
 *   ExperimentConfig cfg = default_experiment_config();
 *   cfg.dro.enabled = true;
 *   cfg.mpc.type = MPCType::SH_MPCC;
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
        mpc.sampling.sync_belief();
        mpc.sampling.markov_jump_system =
            (obstacles.switch_regime == ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM);
        if (rollout.method_name.empty()) {
            rollout.method_name =
                dro_configuration_name(
                    dro.enabled ? DROConfiguration::DRO
                                : DROConfiguration::BASE) +
                "_" + mpc_type_name(mpc.type);
        }
    }

    /// Map the nested experiment sections into the controller runtime config.
    RuntimeConfig to_scenario_mpc_config() const {
        RuntimeConfig cfg;
        cfg.mpc = mpc;
        cfg.dro = dro;
        cfg.solver = solver;
        cfg.obstacle_radius = obstacle_radius;
        cfg.mpc.sampling.markov_jump_system =
            (obstacles.switch_regime == ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM);
        cfg.normalize();
        return cfg;
    }
};

/// Base settings loaded from configs/default.yaml (C++ fallbacks if missing).
ExperimentConfig default_experiment_config();

inline std::string arm_name(const ExperimentConfig& cfg) {
    if (!cfg.rollout.method_name.empty()) return cfg.rollout.method_name;
    const bool sh = cfg.mpc.uses_safe_horizon();
    if (!cfg.dro.enabled) return sh ? "Base+SH" : "Base";
    return sh ? "DRO+SH" : "DRO";
}

/**
 * @brief Compose a standard Base / Base+SH / DRO / DRO+SH arm from default.yaml.
 */
inline ExperimentConfig make_arm_config(
    DROConfiguration dro_kind,
    MPCType mpc_kind,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "",
    double rare_prob = 0.0
) {
    ExperimentConfig cfg = default_experiment_config();
    cfg.dro.enabled = (dro_kind == DROConfiguration::DRO);
    cfg.mpc.type = mpc_kind;
    cfg.mpc.sync_from_type();
    cfg.mpc.sampling.num_scenarios = num_scenarios;
    cfg.obstacles.switch_prob = switch_prob;
    cfg.obstacles.obs_modes = obs_modes;
    cfg.obstacles.rare_mode = rare_mode;
    cfg.obstacles.rare_switch_prob = rare_prob;
    cfg.rollout.rollout_steps = rollout_steps;
    cfg.rollout.method_name.clear();
    cfg.normalize();
    return cfg;
}

inline ExperimentConfig make_base_config(
    double switch_prob, int num_scenarios, int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "", double rare_prob = 0.0
) {
    return make_arm_config(
        DROConfiguration::BASE, MPCType::MPCC,
        switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob);
}

inline ExperimentConfig make_base_sh_config(
    double switch_prob, int num_scenarios, int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "", double rare_prob = 0.0
) {
    return make_arm_config(
        DROConfiguration::BASE, MPCType::SH_MPCC,
        switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob);
}

inline ExperimentConfig make_dro_config(
    double switch_prob, int num_scenarios, int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "", double rare_prob = 0.0
) {
    return make_arm_config(
        DROConfiguration::DRO, MPCType::MPCC,
        switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob);
}

inline ExperimentConfig make_dro_sh_config(
    double switch_prob, int num_scenarios, int rollout_steps,
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "", double rare_prob = 0.0
) {
    return make_arm_config(
        DROConfiguration::DRO, MPCType::SH_MPCC,
        switch_prob, num_scenarios, rollout_steps,
        obs_modes, rare_mode, rare_prob);
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

    bool collision = false;
    int collision_step = -1;
    double min_clearance = 1e9;
    int min_clearance_step = -1;

    double total_progress = 0.0;
    bool completed_path = false;
    double control_effort = 0.0;
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

    int rare_mode_active = 0;
    int rare_mode_missed = 0;

    double avg_solve_ms = 0.0;
    double p50_solve_ms = 0.0;
    double p95_solve_ms = 0.0;
    double max_solve_ms = 0.0;
    std::vector<double> solve_times_raw;

    int total_dro_injected = 0;
    double avg_safe_horizon = 0.0;
    double clearance_5pct = 0.0;
    int active_constraints = 0;
};

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
// HELPER FUNCTIONS
// ############################################################################

SeedBundle derive_seeds(unsigned master_seed, int idx);

void apply_distribution_shift(
    const DistributionShiftConfig& shift,
    ObstacleSim& obs_sim,
    std::mt19937& rng
);

ObstacleState obstacle_on_s_curve(
    const ReferencePath& path,
    double arc_fraction,
    std::mt19937& rng
);

double percentile(std::vector<double> v, double p);

RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config,
    unsigned seed
);

ReferencePath setup_mpcc_path(AdaptiveScenarioMPC& ctrl);

inline RolloutResult run_configured_rollout(
    ExperimentConfig cfg,
    unsigned seed
) {
    cfg.normalize();
    return RolloutResult::from_record(run_experiment_rollout(cfg, seed));
}

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

EnvironmentSetup create_environment(
    EnvironmentType env,
    std::mt19937& rng,
    const EnvironmentExperimentConfig& path_cfg = {}
);

RolloutResult run_single_rollout_env(
    ExperimentConfig cfg,
    unsigned seed,
    const EnvironmentSetup& env_setup,
    SamplingBaseline baseline = SamplingBaseline::STANDARD,
    int forced_safe_horizon = -1
);

}  // namespace dro_mpc

#endif  // DRO_MPC_EXPERIMENT_HARNESS_HPP
