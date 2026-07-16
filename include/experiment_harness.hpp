/**
 * @file experiment_harness.hpp
 * @brief Canonical rollout runner, obstacle simulator, stats, and CSV writer.
 *
 * ALL rollout logic lives here. The paper_experiment_runner configures
 * ExperimentConfig and calls run_experiment_rollout() — it does NOT
 * duplicate obstacle simulation, collision detection, or mode tracking.
 *
 * Key features of run_experiment_rollout():
 *   - S-curve reference path (L=25m, A=3m) matching Python tests
 *   - Multi-obstacle with class-based mode observation sharing
 *   - Multi-disc collision detection (configurable num_discs)
 *   - OT predictor integration (when use_ot_predictor is set)
 *   - Path completion termination at configurable fraction (default 95%)
 *   - Distribution shift injection
 *   - Per-step callbacks for experiment-specific logic (e.g. oracle flood)
 *
 * Obstacle class sharing:
 *   Obstacles assigned to the same class (via obstacles_per_class) share
 *   mode observations. When a mode is observed on any obstacle, it is
 *   broadcast to all siblings in the same class via the controller's
 *   update_mode_observation(). Late-joining obstacles inherit existing
 *   class history.
 */

#ifndef SCENARIO_MPC_EXPERIMENT_HARNESS_HPP
#define SCENARIO_MPC_EXPERIMENT_HARNESS_HPP

#include "types.hpp"
#include <cmath>
#include "config.hpp"
#include "wasserstein_dro.hpp"
#include "reference_path.hpp"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <random>
#include <functional>

namespace scenario_mpc {

// Forward declarations
class AdaptiveScenarioMPC;

// ============================================================================
// Enums
// ============================================================================

enum class AblationVariant {
    NO_INJECTION,
    DRO_FULL,
    DRO_NO_COV,
    DRO_DISTANCE_ONLY,
    RANDOM_INJECTION,
    ALWAYS_INJECT
};

inline std::string ablation_variant_name(AblationVariant v) {
    switch (v) {
        case AblationVariant::NO_INJECTION:      return "no_injection";
        case AblationVariant::DRO_FULL:          return "dro_full";
        case AblationVariant::DRO_NO_COV:        return "dro_no_cov";
        case AblationVariant::DRO_DISTANCE_ONLY: return "dro_distance_only";
        case AblationVariant::RANDOM_INJECTION:   return "random_injection";
        case AblationVariant::ALWAYS_INJECT:      return "always_inject";
        default: return "unknown";
    }
}

inline std::string ground_cost_name(DROGroundCostType t) {
    switch (t) {
        case DROGroundCostType::W2_BURES:       return "w2_bures";
        case DROGroundCostType::ZERO_ONE:       return "zero_one";
        case DROGroundCostType::EUCLIDEAN_MEAN: return "euclidean_mean";
        default: return "unknown";
    }
}

// ============================================================================
// Paper Experiment Constants
// ============================================================================

inline constexpr int    DEFAULT_ROLLOUT_STEPS  = 200;
inline constexpr double DEFAULT_DT             = 0.1;
inline constexpr int    DEFAULT_HORIZON        = 15;
inline constexpr int    DEFAULT_BASE_SCENARIOS = 40;
inline constexpr double S_CURVE_LENGTH         = 25.0;
inline constexpr double S_CURVE_AMPLITUDE      = 3.0;
inline constexpr int    S_CURVE_POINTS         = 200;
inline constexpr double PATH_COMPLETE_FRAC     = 0.95;
inline constexpr double OBS_PATH_FRACTION      = 0.35;

// ============================================================================
// Paper Variants
// ============================================================================

enum class PaperVariant {
    BASE, BASE_SH, DRO, DRO_SH
};

inline const std::vector<PaperVariant> ALL_VARIANTS = {
    PaperVariant::BASE,
    PaperVariant::BASE_SH,
    PaperVariant::DRO,
    PaperVariant::DRO_SH
};

inline std::string variant_name(PaperVariant v) {
    switch (v) {
        case PaperVariant::BASE:       return "Base";
        case PaperVariant::BASE_SH:    return "Base+SH";
        case PaperVariant::DRO:        return "DRO";
        case PaperVariant::DRO_SH:     return "DRO+SH";
    }
    return "?";
}

inline bool uses_dro(PaperVariant v) {
    return v == PaperVariant::DRO || v == PaperVariant::DRO_SH;
}
inline bool uses_sh(PaperVariant v) {
    return v == PaperVariant::BASE_SH || v == PaperVariant::DRO_SH;
}

// ============================================================================
// Environment Types
// ============================================================================

// NOTE: every EnvironmentType below runs on the SAME S-curve reference path
// (create_s_curve(S_CURVE_LENGTH, S_CURVE_AMPLITUDE, ...)). The variants differ
// ONLY in obstacle placement/velocity, never in road geometry. The first value
// was called STRAIGHT, which read as a road shape and was not one: it places a
// SLOW obstacle (v in [0.1,0.4] m/s) travelling in the SAME direction as the ego
// (+tangent) at arc-fraction 0.35, which the ego (1.5 m/s) overtakes. Renamed to
// OVERTAKE_SLOW_LEAD. The old name collided with tests/test_generalization.cpp's
// "Straight", which IS a straight road (create_straight) carrying an ONCOMING
// obstacle -- an unrelated experiment. That collision made a 0.1% base collision
// rate look inconsistent with a 55.6% one (see FINDINGS_CONTEXT.md 'Gap 13');
// they were never the same scenario.
enum class EnvironmentType { OVERTAKE_SLOW_LEAD, NARROW_CORRIDOR, INTERSECTION, ONCOMING };

struct EnvironmentSetup {
    ObstacleState initial_obs;
    std::vector<std::string> obs_modes;
    EgoState initial_ego;
    Eigen::Vector2d goal;
    std::string name;
};

inline std::string environment_name(EnvironmentType env) {
    switch (env) {
        case EnvironmentType::OVERTAKE_SLOW_LEAD: return "OvertakeSlowLead";
        case EnvironmentType::NARROW_CORRIDOR: return "Narrow";
        case EnvironmentType::INTERSECTION: return "Intersection";
        case EnvironmentType::ONCOMING: return "Oncoming";
    }
    return "?";
}

/**
 * @brief When the ground-truth obstacle is allowed to change mode.
 *
 * PER_STEP           -- switch may fire at every rollout step (realistic; CDC'26).
 * HOLD_OVER_HORIZON  -- mode frozen for `horizon` steps, so the mode is constant
 *                       across any one prediction horizon. This is the regime in
 *                       which the mode-conditional violation probability v_m, and
 *                       hence Theorem 1, is well defined.
 */
enum class ModeSwitchRegime { PER_STEP, HOLD_OVER_HORIZON };

inline std::string switch_regime_name(ModeSwitchRegime r) {
    switch (r) {
        case ModeSwitchRegime::PER_STEP:          return "PerStep";
        case ModeSwitchRegime::HOLD_OVER_HORIZON: return "HoldOverHorizon";
    }
    return "?";
}

// ============================================================================
// Sampling Baselines
// ============================================================================

enum class SamplingBaseline {
    STANDARD, STRATIFIED, TEMPERATURE, EPSILON_GREEDY, RISK_BIASED,
    UNIFORM_WEIGHT, RECENCY_WEIGHT, ORACLE_FLOOD
};

inline std::string baseline_name(SamplingBaseline b) {
    switch (b) {
        case SamplingBaseline::STANDARD: return "Standard";
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

// ============================================================================
// Obstacle Simulator
// ============================================================================

/**
 * @brief Ground-truth obstacle simulator with mode switching.
 *
 * Used by all rollouts. Propagates obstacle state under the current mode
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

// ============================================================================
// Data Structures
// ============================================================================

struct SeedBundle {
    unsigned master;
    unsigned env;
    unsigned predictor;
    unsigned scenario;
};

struct DistributionShiftConfig {
    double rho = 0.0;
    double dangerous_boost = 0.0;
    int boosted_mode = -1;
};

/**
 * @brief Full experiment configuration.
 *
 * Covers all parameters needed by any rollout variant. The paper
 * experiment runner maps its PaperVariant enum into these fields.
 */
struct ExperimentConfig {
    // MPC parameters
    int num_scenarios = 20;
    double eps_wass = 0.1;
    double sigma_scale = 1.0;
    int horizon = 20;
    int num_discs = 3;
    double vehicle_length = 4.0;
    bool safe_horizon_enabled = true;
    int forced_safe_horizon = -1;
    int safe_horizon_min = 3;              ///< Minimum safe horizon (default matches harness)
    double switch_prob = 0.1;
    /// Mode-switching regime of the GROUND-TRUTH obstacle simulator.
    ///
    /// THEORY GATE. Theorem 1 (see paper_lcss_mode_coverage/paper_lcss_draft.tex,
    /// Sec. IV-E) is stated in terms of the mode-conditional violation probability
    ///     v_m(U) := P[ U violates the collision constraint | mode m ].
    /// That object is only well defined if the obstacle's mode is CONSTANT over the
    /// prediction horizon. It is not, by default: maybe_switch() fires every rollout
    /// step with probability switch_prob, so the true mode can change mid-horizon and
    /// "conditioned on mode m" becomes ambiguous -- the honest index would be the
    /// mode SEQUENCE, which inflates the mode set from M to M^{N_s} and changes what
    /// the likelihood ratio L in Lemma 1 means.
    ///
    /// HOLD_OVER_HORIZON freezes the mode for `horizon` steps at a time, making v_m
    /// well defined and Theorem 1 exactly testable. PER_STEP is the realistic regime
    /// and the one CDC'26 used; the theorem there needs the sequence-indexed form.
    ///
    /// Use HOLD_OVER_HORIZON to validate the theory, PER_STEP to measure reality.
    /// Reporting a Theorem-1 certificate on PER_STEP data without the sequence-indexed
    /// extension is a claim the math does not support.
    ModeSwitchRegime switch_regime = ModeSwitchRegime::PER_STEP;
    int rollout_steps = 60;
    int num_modes = 4;

    // Weight type and DRO (set directly, not derived from AblationVariant)
    WeightType weight_type = WeightType::FREQUENCY;
    bool enable_dro = false;
    InjectionMode injection_mode = InjectionMode::QSTAR_SAMPLE;

    // Legacy ablation variant (used by configure_ablation helper)
    AblationVariant ablation = AblationVariant::DRO_FULL;
    DROGroundCostType ground_cost = DROGroundCostType::W2_BURES;
    /// Risk functional r[m] reports. SURROGATE_VAR is the CDC'26 default.
    /// JOINT_* costs ~28 ms/call vs ~0 for the surrogates -- do not use it in the
    /// large sweeps without shrinking joint_risk_samples and accepting the error.
    /// Confidence-calibrated ambiguity radius (theory chain link 1). Off by
    /// default so CDC'26 stays reproducible; on for anything claiming a
    /// belief-coverage guarantee.
    bool use_calibrated_radius = false;
    double confidence_beta = 0.05;
    /// Entropic allocator (the pivot's core claim): guarantees min_m q_m > 0 so
    /// the sampling certificate L <= 1/q_min is finite. Off by default.
    /// Road-boundary (contouring) constraints. THE EGO WAS NEVER CONSTRAINED TO THE
    /// ROAD in any experiment run before 2026-07-16 (this defaulted false and nothing
    /// set it). Reviewer 2 comment 3 asked exactly this. On => every collision number
    /// changes meaning, so re-run everything after enabling.
    bool enable_contouring_constraints = false;
    double road_width = 7.0;   ///< Symmetric road width [m] about the reference path
    bool use_entropic_allocator = false;
    double entropic_tau = 0.05;
    DRORiskMeasure risk_measure = DRORiskMeasure::SURROGATE_VAR;
    int joint_risk_samples = 8000;  ///< MC samples when risk_measure is JOINT_*
    /// Sample mode SEQUENCES from the estimated Markov transition matrix
    /// instead of one i.i.d. mode held for the horizon. Off by default so
    /// CDC'26 numbers stay reproducible.
    bool use_markov_mode_sampling = false;
    /// Dirichlet / sticky prior hyperparameters for the mode belief.
    ModeBeliefConfig mode_belief;
    DistributionShiftConfig shift;

    int dro_injection_count = 1;  ///< Top-K modes to inject per obstacle (0=none, -1=all)
    double softmax_tau = 5.0;     ///< Temperature for SOFTMAX_RISK baseline
    double eps_greedy_epsilon = 0.3;  ///< Epsilon for EPSILON_GREEDY_INJ baseline
    int max_history_length = -1;      ///< Max observation history length (-1 = default, i.e. horizon*10)

    // Mode configuration
    std::vector<std::string> obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"
    };
    std::string rare_mode = "lane_change_left";
    double rare_switch_prob = 0.05;

    // Multi-obstacle configuration
    int num_obstacles = 1;
    int obstacles_per_class = 1;
    std::vector<double> obs_arc_fractions;  ///< Where to place obstacles (empty = auto)

    // Custom initial obstacle states (overrides arc-fraction placement)
    std::vector<ObstacleState> initial_obstacle_states;

    // Custom reference path (overrides default S-curve)
    std::optional<ReferencePath> custom_ref_path;
    // Custom initial ego state (overrides default (0,0,0,1.5))
    std::optional<EgoState> custom_initial_ego;

    // Path and termination
    bool path_completion_termination = true;
    double path_completion_fraction = 0.95;

    // Per-step callback: called after mode observation, before solve.
    // Args: (step, obstacle_id, obstacle_sim, controller, rng)
    std::function<void(int, int, ObstacleSim&, AdaptiveScenarioMPC&, std::mt19937&)> step_callback;

    // Scenario / edge-case tag
    std::string scenario_tag = "baseline";
    std::string method_name;  ///< Override method name in record (empty = auto)
};

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
    // --- Conservatism metrics (CDC'26 Reviewer 3, major comment 5) ---
    // WDRO-sampling changes the sampled constraint set and WDRO-injection adds
    // deterministic constraints; both may shrink the feasible set and raise the
    // optimal objective. Without these the harness cannot report whether the
    // collision-rate gain is bought with conservative planning.
    // NOTE: the raw components are reported rather than a single weighted MPCC
    // cost, because the controller's stage-cost weights (w_c, w_l, w_a, w_omega)
    // are not exposed on ScenarioMPCConfig. Synthesising a weighted sum here
    // would invent numbers; the components answer the reviewer directly.
    double sum_contouring_sq = 0.0;   ///< sum_k e_c(k)^2  (lateral deviation from path)
    double sum_lag_sq = 0.0;          ///< sum_k e_l(k)^2  (along-path deviation)
    double sum_velocity_err_sq = 0.0; ///< sum_k (v_k - v_ref)^2
    int metric_steps = 0;             ///< steps contributing to the sums (for means)
    double mean_contouring_error() const {
        return metric_steps > 0 ? std::sqrt(sum_contouring_sq / metric_steps) : 0.0;
    }
    double mean_velocity_error() const {
        return metric_steps > 0 ? std::sqrt(sum_velocity_err_sq / metric_steps) : 0.0;
    }
    int constraint_active_count = 0;
    int missed_mode_steps = 0;
    int total_mode_checks = 0;
    // JOINT missed-mode: a step is missed if NO single scenario has ALL obstacles'
    // true modes simultaneously. Tests whether COVERAGE of the joint mode config
    // (size M^V) is the collision mechanism at high V, vs the marginal per-obstacle
    // metric above. One check per step (not per obstacle).
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
    std::vector<double> solve_times_raw;  ///< Raw per-step solve times [s]

    // DRO-specific
    int total_dro_injected = 0;
    double avg_safe_horizon = 0.0;
    double clearance_5pct = 0.0;
    int active_constraints = 0;
};

// ============================================================================
// RolloutResult — thin adapter over RolloutRecord
// ============================================================================

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

/// Default arc fractions for placing 4 obstacles along the S-curve.
inline const std::vector<double> OBS_ARC_FRACS_4 = {0.20, 0.35, 0.50, 0.65};

// ============================================================================
// CSV Writer
// ============================================================================

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

// ============================================================================
// Statistical Helpers
// ============================================================================

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

// ============================================================================
// Utilities
// ============================================================================

SeedBundle derive_seeds(unsigned master_seed, int idx);

/// Apply distribution shift to an obstacle simulator.
void apply_distribution_shift(
    const DistributionShiftConfig& shift,
    ObstacleSim& obs_sim,
    std::mt19937& rng
);

/// Configure ScenarioMPCConfig + DROConfig from an AblationVariant.
void configure_ablation(
    ScenarioMPCConfig& config,
    DROConfig& dro_cfg,
    AblationVariant variant,
    double eps_wass,
    double sigma_scale
);

/// Place an obstacle on an S-curve path at the given arc-length fraction.
ObstacleState obstacle_on_s_curve(
    const ReferencePath& path,
    double arc_fraction,
    std::mt19937& rng
);

/// Compute percentile of a vector (p in [0,100]).
double percentile(std::vector<double> v, double p);

// ============================================================================
// Rollout Runner
// ============================================================================

/**
 * @brief Run a single MPC rollout with the given configuration and seed.
 *
 * This is THE canonical rollout function. All experiments should call this.
 * Features:
 * - S-curve reference path (matching Python tests)
 * - Multi-obstacle with class-based mode sharing
 * - Multi-disc collision detection
 * - Path completion termination
 * - OT predictor integration (when enabled)
 * - Distribution shift
 * - Per-step callbacks for experiment-specific monitoring
 */
RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config,
    unsigned seed
);

// ============================================================================
// Paper Variant Helpers (implementations in experiment_harness.cpp)
// ============================================================================

/// Create a reference path and set it on the controller.
ReferencePath setup_mpcc_path(AdaptiveScenarioMPC& ctrl);

/// Map PaperVariant to ExperimentConfig fields.
ExperimentConfig make_experiment_config(
    PaperVariant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    const std::vector<std::string>& obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "",
    double rare_prob = 0.0,
    bool safe_horizon_enabled = false,
    int num_discs = 1,
    double vehicle_length = 1.5
);

/// Run a single rollout for a PaperVariant.
RolloutResult run_single_rollout(
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
);

/// Run a multi-obstacle rollout for a PaperVariant.
RolloutResult run_multi_obstacle_rollout(
    PaperVariant variant,
    double switch_prob,
    int num_scenarios,
    int rollout_steps,
    unsigned seed,
    int num_obstacles = 4,
    int num_classes = 4,
    const std::vector<std::string>& obs_modes = {"constant_velocity", "turn_left", "turn_right", "decelerating"},
    const std::string& rare_mode = "",
    double rare_prob = 0.0,
    const std::vector<double>& arc_fracs = OBS_ARC_FRACS_4
);

/// Create an environment setup for a given type.
EnvironmentSetup create_environment(EnvironmentType env, std::mt19937& rng);

/// Run a rollout with a custom environment setup.
RolloutResult run_single_rollout_env(
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
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_EXPERIMENT_HARNESS_HPP
