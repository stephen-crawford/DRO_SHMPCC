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
 *   3. Experiment configurations — sampling baselines, rollout protocol, the
 *                                   assembled ExperimentConfig + arm builders
 *   4. Artifact output           — reproducibility bundle + trace/SVG/GIF/RViz
 *   5. Results reporting         — RolloutRecord / RolloutResult + statistics
 *   6. Helper functions          — seeds, shift, path/placement, the rollout runner
 *
 * run_experiment_rollout() is THE canonical rollout. experiment_runner and all
 * tests configure an ExperimentConfig and call it — they do NOT duplicate
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
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

namespace dro_mpc {

// Forward declarations (controller lives in mpc_controller.hpp; only referenced
// by pointer/reference in callback signatures here).
class MPCController;

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
    /// Requested regular-mode subset size. This is used only when
    /// `randomize_available_modes` is true; otherwise every valid entry in
    /// `obs_modes` is available to each obstacle.
    int num_modes = 4;
    int num_obstacles = 1;
    /// <= 0 derives classes from `history`; a positive YAML value is preserved.
    int obstacles_per_class = 0;
    /// Explicit balanced class assignment i % num_classes; zero retains legacy layout.
    int num_classes = 0;
    /// Use seeded arc placement even for a single obstacle (paired matrix fixtures).
    bool place_on_path = false;

    int class_id(int obstacle_id) const {
        return num_classes > 0 ? obstacle_id % num_classes
                               : obstacle_id / std::max(1, obstacles_per_class);
    }

    std::vector<std::string> obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"
    };
    /// Select a reproducible random subset of `obs_modes` for each rollout.
    /// The rollout seed drives the selection; false preserves the YAML order/list.
    bool randomize_available_modes = false;
    /// When true, each obstacle receives its own random subset; otherwise the
    /// selected set is shared by every obstacle in the rollout.
    bool randomize_modes_per_obstacle = false;
    /// Empty disables rare-mode forcing. The canonical default is supplied by YAML.
    std::string rare_mode;
    double rare_switch_prob = 0.05;

    std::vector<double> obs_arc_fractions;               // Empty => auto placement
    /// Explicit world-frame starts, one per obstacle. Each state overrides the
    /// corresponding arc placement; YAML accepts `x,y` or `x,y,vx,vy` entries.
    std::vector<ObstacleState> initial_obstacle_states;
    DistributionShiftConfig shift;

    double default_arc_fraction = 0.35;
    double process_noise = 0.02;
    /// Disable model process noise in controller/predictor trajectories for deterministic fixtures.
    bool prediction_noise = true;
    double speed_cap = 2.0;
    /// Test-plant policy; does not modify the controller's mode models.
    std::string behavior = "mode_switching";
    double behavior_initial_speed = 1.0;
    double behavior_path_offset = 8.0;

    void apply_layout() {
        if (num_classes < 0 || (num_classes > 0 && num_classes > num_obstacles))
            throw std::invalid_argument("num_classes must be between zero and num_obstacles");
        if (behavior != "mode_switching" && behavior != "random_orientation" &&
            behavior != "pursuit" && behavior != "path_intersection" && behavior != "path_following")
            throw std::invalid_argument("unknown obstacle_behavior: " + behavior);
        if (!std::isfinite(behavior_initial_speed) || behavior_initial_speed <= 0 ||
            !std::isfinite(behavior_path_offset) || behavior_path_offset <= 0)
            throw std::invalid_argument("obstacle behavior speed and offset must be finite and positive");
        num_modes = std::max(1, num_modes);
        if (num_obstacles <= 1) {
            num_obstacles = num_obstacles == 0 ? 0 : 1;
            if (obstacles_per_class <= 0) obstacles_per_class = 1;
            return;
        }
        if (obstacles_per_class > 0) return;
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

/// Human-readable record of how `ObstacleExperimentConfig::num_modes` is
/// interpreted for a rollout. The actual support can still differ per
/// obstacle when a configured rare mode is appended after random selection.
inline std::string mode_selection_policy_name(
    const ObstacleExperimentConfig& config
) {
    if (!config.randomize_available_modes) return "configured_list";
    return config.randomize_modes_per_obstacle
        ? "random_subset_per_obstacle"
        : "random_subset_shared";
}

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

    // MRS-style route input: an explicit sequence of world-frame x,y points
    // is converted to one C2 polynomial trajectory.  It wins over the
    // generated environment route but remains below an explicit C++ path.
    std::vector<Eigen::Vector2d> path_waypoints;
    bool path_closed_loop = false;
    double path_sample_spacing = 0.25;
    double path_control_point_spacing = 4.0;

    // A positive value caps v^2 * |curvature| during trajectory timing.
    // Zero inherits the configured nominal speed and ego angular-rate limit.
    double path_max_lateral_acceleration = 0.0;

    double s_curve_length = 25.0;
    double s_curve_amplitude = 3.0;
    int s_curve_points = 200;
    double ego_initial_v = 1.5;

    // Shared road geometry (metres). Lane count selects the driven carriageway.
    int lane_count = 2;
    double lane_width = 3.6;
    double shoulder_width = 1.0;
    double median_width = 1.5;
    double road_length = 80.0;

    // Intersection and turning layouts.
    double intersection_box_size = 20.0;
    double corner_radius = 8.0;

    // Ramp and roundabout layouts.
    double ramp_length = 50.0;
    double merge_length = 30.0;
    double roundabout_radius = 18.0;

    // Corridor / S-curve layouts.
    double corridor_width = 4.5;
    double curve_radius = 45.0;
    double transition_length = 15.0;
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
// # 3. EXPERIMENT CONFIGURATIONS
// ############################################################################

/// Paper-arm label: DRO on vs off. Controller knob is dro.enabled.
enum class DROConfiguration { BASE, DRO };

inline std::string dro_configuration_name(DROConfiguration d) {
    return d == DROConfiguration::DRO ? "dro" : "base";
}

struct RolloutExperimentConfig {
    int rollout_steps = 200;
    std::string scenario_tag = "baseline";
    std::string method_name;  ///< Empty => auto from DRO/MPC labels
    double metrics_v_ref = 1.5;

    /// Called after mode observation, before solve.
    std::function<void(int, int, ObstacleSim&, MPCController&, std::mt19937&)>
        step_callback;
};

/**
 * @brief Optional self-contained outputs produced by the canonical rollout.
 *
 * Set `output_directory` to enable artifact generation.  The harness creates
 * one deterministic subdirectory per run, containing a resolved configuration
 * snapshot, seed/backend manifest, a machine-readable trace, and optional
 * SVG, animated GIF, and RViz-replay data. Empty keeps programmatic and
 * unit-test rollouts side-effect free.
 */
struct ExperimentArtifactConfig {
    std::string output_directory;
    std::string run_name;
    bool write_reproducibility_manifest = true;
    bool write_trace_csv = true;
    /// Decision timings/certificates and complete sampled mode coverage for matrix analysis.
    bool write_analysis_csv = false;
    bool write_visualization_svg = true;
    /// Native, dependency-free animated playback of the realized rollout.
    bool write_visualization_gif = false;
    /// Retain every Nth trace frame in rollout.gif; one preserves every recorded
    /// execution state, including the initial and final states.
    int gif_frame_stride = 1;
    bool show_linearized_constraints = true;
    /// Show all solver-reported support forecasts instead of constraint glyphs or a preview.
    bool show_support_scenarios = false;
    bool show_sampled_scenarios = true;
    int scenario_preview_count = 8;
    /// Replay-rate multiplier for recorded trace timestamps: one means the GIF
    /// spans the simulation's elapsed execution time exactly (up to GIF's
    /// centisecond resolution); two replays twice as fast.
    double gif_playback_rate = 1.0;
    /// Write scene.csv and rollout.rviz for the optional ROS 2 RViz replayer.
    /// RViz replay requires write_trace_csv to remain enabled.
    bool write_rviz_replay_bundle = false;

    bool enabled() const noexcept { return !output_directory.empty(); }

    void validate() const {
        if (scenario_preview_count < 1) throw std::invalid_argument("scenario_preview_count must be positive");
        if (gif_frame_stride < 1) {
            throw std::invalid_argument("artifact_gif_frame_stride must be at least one");
        }
        if (!std::isfinite(gif_playback_rate) || gif_playback_rate <= 0.0) {
            throw std::invalid_argument("artifact_gif_playback_rate must be finite and positive");
        }
        if (write_rviz_replay_bundle && !write_trace_csv) {
            throw std::invalid_argument(
                "artifact_write_rviz_replay requires artifact_write_trace_csv");
        }
    }
};

struct SeedBundle {
    unsigned master = 0;
    unsigned env = 0;        ///< Plant/environment and ground-truth obstacle stream.
    unsigned predictor = 0;  ///< Reserved predictor-estimation stream.
    unsigned scenario = 0;   ///< Controller scenario-sampling stream.
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
 *   cfg.mpc.sampling.set_manual_sample_count(40);
 *   cfg.obstacles.switch_prob = 0.2;
 *   cfg.rollout.rollout_steps = 200;
 */
struct ExperimentConfig {
    std::string config_source = "in_memory";
    MPCConfig mpc;
    DROControllerConfig dro;
    SolverSettings solver{};
    double obstacle_radius = 0.35;
    ObstacleExperimentConfig obstacles;
    EnvironmentExperimentConfig environment;
    RolloutExperimentConfig rollout;
    ExperimentArtifactConfig artifacts;

    /// Apply layout rules; sync belief / fixed rho; derive a certified scenario
    /// count when configured; and auto-name the method.
    /// Does not call mpc.sync_from_type() (would overwrite SH overrides set
    /// after type selection).
    void normalize() {
        artifacts.validate();
        obstacles.apply_layout();
        mpc.sampling.sync_belief();
        mpc.sampling.markov_jump_system =
            (obstacles.switch_regime == ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM);
        dro.apply_fixed_rho();
        if (mpc.uses_safe_horizon() &&
            mpc.sampling.automatically_compute_sample_size) {
            RuntimeConfig certifier;
            certifier.mpc = mpc;
            mpc.sampling.num_scenarios = certifier.compute_required_scenarios();
        }
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
    cfg.mpc.sampling.set_manual_sample_count(num_scenarios);
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
    unsigned seed = 0;             ///< Master seed supplied to the rollout.
    unsigned plant_seed = 0;       ///< Derived plant/environment RNG seed.
    unsigned predictor_seed = 0;   ///< Derived predictor RNG seed (reserved).
    unsigned controller_seed = 0;  ///< Derived controller scenario RNG seed.
    std::string config_source = "in_memory";
    /// Solver implementation recorded with the seed bundle for replayability.
    std::string qp_backend;
    /// acados/HPIPM build identity, not a source path or machine-specific value.
    std::string qp_solver_identity;
    /// Canonical artifact directory for this rollout; empty when artifacts are
    /// disabled in ExperimentArtifactConfig.
    std::string artifact_directory;
    std::string method;
    std::string scenario = "baseline";
    int S = 0;
    /// Ambiguity radius used by the final controller solve (zero when DRO is off).
    double eps_wass = 0.0;
    double sigma = 0.0;
    double shift_rho = 0.0;
    double shift_boost = 0.0;
    std::string ground_cost;
    std::string risk_scoring_model;
    std::string configured_risk_measure;
    /// Actual algorithm used after any explicit compatibility resolution.
    std::string effective_risk_measure;
    /// Whether `num_modes` selected a random subset or the full configured
    /// list was used; see mode_selection_policy_name().
    std::string mode_selection_policy;
    /// YAML `num_modes`: meaningful only for a random-subset policy.
    int requested_random_mode_count = 0;
    /// Actual mode support sizes, in obstacle-ID order, after catalog
    /// filtering and any rare-mode augmentation.
    std::vector<int> effective_available_mode_counts;

    bool collision = false;
    int collision_step = -1;
    double min_clearance = 1e9;
    int min_clearance_step = -1;

    double total_progress = 0.0;
    bool completed_path = false;
    std::string termination_reason = "step_limit";
    int failed_decision_step = -1;
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
    double avg_dro_risk_ms = 0.0;
    double p50_solve_ms = 0.0;
    double p95_solve_ms = 0.0;
    double max_solve_ms = 0.0;
    std::vector<double> solve_times_raw;
    std::vector<double> dro_risk_times_raw;

    /// Safe-Horizon solves for which a certificate was evaluated (excludes
    /// ordinary MPC solves where no certificate was requested).
    int safe_horizon_decisions = 0;
    /// Decisions that met the sample-count, feasibility, and support-cap
    /// conditions for a full-horizon certificate.
    int certified_decisions = 0;
    /// certified_decisions / safe_horizon_decisions, or zero when SH was not
    /// requested for the rollout.
    double certificate_rate = 0.0;
    /// Mean full horizon among valid certificates only; zero if none exists.
    double avg_certified_horizon = 0.0;
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

/**
 * @brief Build ground-truth obstacle simulators from one obstacle configuration.
 *
 * Explicit `initial_obstacle_states` take priority, followed by configured arc
 * fractions, then deterministic multi-obstacle arc placement. The returned
 * simulators already contain their selected mode sets and matching dynamics.
 */
std::vector<ObstacleSim> construct_obstacles(
    const ObstacleExperimentConfig& config,
    const ObstacleState& environment_default,
    const ReferencePath& reference_path,
    const std::map<std::string, ModeModel>& mode_catalog,
    std::mt19937& rng
);

double percentile(std::vector<double> v, double p);

EnvironmentExperimentConfig default_environment_experiment_config();

/// Build the nominal ego route for an environment geometry without creating a
/// controller or obstacle simulator. Shared by the rollout and topology tests.
ReferencePath build_environment_reference_path(
    EnvironmentType environment,
    const EnvironmentExperimentConfig& config
);

/// Road centerlines forming the visible environment layout. The first entry is
/// the ego route; subsequent entries describe cross streets, mainlines, or
/// other connected road geometry used by topology visualizations.
std::vector<ReferencePath> build_environment_road_centerlines(
    EnvironmentType environment,
    const EnvironmentExperimentConfig& config
);

/// Terminal goal for open paths and a wrapped horizon-lookahead target for
/// periodic paths. This keeps closed-loop rollouts moving along the route
/// rather than repeatedly aiming for their starting point.
Eigen::Vector2d rollout_tracking_goal(
    const ReferencePath& path,
    double path_progress,
    double reference_velocity,
    int horizon,
    double dt
);

RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config,
    unsigned seed
);

ReferencePath setup_mpcc_path(MPCController& ctrl);

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
            ? ((num_obstacles + num_classes - 1) / num_classes) : 1;
    cfg.obstacles.obs_arc_fractions = arc_fracs;
    cfg.obstacles.history = ObstacleHistoryConfiguration::INDEPENDENT;
    return run_configured_rollout(std::move(cfg), seed);
}

EnvironmentSetup create_environment(
    EnvironmentType env,
    std::mt19937& rng,
    const EnvironmentExperimentConfig& path_cfg = default_environment_experiment_config()
);

RolloutResult run_single_rollout_env(
    ExperimentConfig cfg,
    unsigned seed,
    const EnvironmentSetup& env_setup
);

}  // namespace dro_mpc

#endif  // DRO_MPC_EXPERIMENT_HARNESS_HPP
