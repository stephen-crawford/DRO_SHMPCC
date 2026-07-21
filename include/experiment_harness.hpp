/**
 * @file experiment_harness.hpp
 * @brief Canonical rollout runner, obstacle simulator, stats, and CSV writer.
 *
 * ALL rollout logic lives here. The paper_experiment_runner configures
 * ExperimentConfig and calls run_experiment_rollout() — it does NOT
 * duplicate obstacle simulation, collision detection, or mode tracking.
 *
 * Configuration is split into independent sections so complex experiments
 * can be assembled by composing:
 *
 *   DRO  +  MPC  +  Obstacle  +  Environment  +  Sampling  +  Rollout
 *
 * Key features of run_experiment_rollout():
 *   - S-curve reference path (L=25m, A=3m) matching Python tests
 *   - Multi-obstacle with class-based mode observation sharing
 *   - Multi-disc collision detection (configurable num_discs)
 *   - Path completion termination at configurable fraction (default 95%)
 *   - Distribution shift injection
 *   - Per-step callbacks for experiment-specific logic (e.g. oracle flood)
 */

#ifndef DRO_MPC_EXPERIMENT_HARNESS_HPP
#define DRO_MPC_EXPERIMENT_HARNESS_HPP

#include "types.hpp"
#include "config.hpp"
#include "wasserstein_dro.hpp"
#include "reference_path.hpp"

#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace scenario_mpc {

// Forward declarations
class AdaptiveScenarioMPC;

// ============================================================================
// DRO Configuration Enums
// ============================================================================

enum class DROConfiguration {
    BASE,  ///< DRO off (nominal scenario MPC)
    DRO,   ///< DRO-enabled
};

enum class ReweightedDistributionConfiguration {
    SAMPLING_ONLY,          ///< Resample scenarios from Q*
    INJECTION_ONLY,         ///< Keep nominal samples; inject worst-case constraint(s)
    SAMPLING_AND_INJECTION, ///< Both Q* resampling and injection
};

enum class RiskMeasureConfiguration {
    VAR_BONFERRONI,  ///< Closed-form union-bound surrogate VaR
    VAR,             ///< Per-step surrogate VaR (CDC'26-style)
    CVAR,            ///< Per-step surrogate CVaR
    SURROGATE_VAR,   ///< Explicit CDC'26 default alias of VAR
    JOINT_RISK,      ///< Joint-horizon Euclidean risk (MC)
};

enum class NominalModeBeliefConfiguration {
    DIRICHLET,  ///< Symmetric Dirichlet prior, no sticky bias
    STICKY,     ///< Dirichlet rows + self-persistence prior
};

enum class GroundCostConfiguration {
    W2,             ///< Gaussian W2 / Bures metric
    W1_METRIC,      ///< W1 ground cost (when available)
    ZERO_ONE,       ///< Discrete 0-1 mode mismatch cost
    EUCLIDEAN_MEAN, ///< Mean-trajectory Euclidean distance
};

// ============================================================================
// DRO Configuration Enum Names
// ============================================================================

inline std::string dro_configuration_name(DROConfiguration c) {
    switch (c) {
        case DROConfiguration::BASE: return "base_sh_mpcc";
        case DROConfiguration::DRO:  return "dro_enabled";
        default: return "unknown";
    }
}

inline std::string reweighted_distribution_configuration_name(
    ReweightedDistributionConfiguration r
) {
    switch (r) {
        case ReweightedDistributionConfiguration::SAMPLING_ONLY:
            return "sampling_only";
        case ReweightedDistributionConfiguration::INJECTION_ONLY:
            return "injection_only";
        case ReweightedDistributionConfiguration::SAMPLING_AND_INJECTION:
            return "sampling_and_injection";
        default: return "unknown";
    }
}

inline std::string risk_measure_configuration_name(RiskMeasureConfiguration r) {
    switch (r) {
        case RiskMeasureConfiguration::VAR_BONFERRONI: return "var_bonferroni";
        case RiskMeasureConfiguration::VAR:            return "var";
        case RiskMeasureConfiguration::CVAR:           return "cvar";
        case RiskMeasureConfiguration::SURROGATE_VAR:  return "surrogate_var";
        case RiskMeasureConfiguration::JOINT_RISK:     return "joint_risk";
        default: return "unknown";
    }
}

inline std::string nominal_mode_belief_configuration_name(
    NominalModeBeliefConfiguration n
) {
    switch (n) {
        case NominalModeBeliefConfiguration::DIRICHLET: return "dirichlet";
        case NominalModeBeliefConfiguration::STICKY:    return "sticky";
        default: return "unknown";
    }
}

inline std::string ground_cost_name(GroundCostConfiguration g) {
    switch (g) {
        case GroundCostConfiguration::W2:             return "w2";
        case GroundCostConfiguration::W1_METRIC:      return "w1";
        case GroundCostConfiguration::ZERO_ONE:       return "zero_one";
        case GroundCostConfiguration::EUCLIDEAN_MEAN: return "euclidean_mean";
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
// DRO Configuration Mappings + Settings
// ============================================================================

inline DRORiskMeasure to_dro_risk_measure(RiskMeasureConfiguration r) {
    switch (r) {
        case RiskMeasureConfiguration::VAR_BONFERRONI:
            return DRORiskMeasure::SURROGATE_VAR_BONFERRONI;
        case RiskMeasureConfiguration::VAR:
        case RiskMeasureConfiguration::SURROGATE_VAR:
            return DRORiskMeasure::SURROGATE_VAR;
        case RiskMeasureConfiguration::CVAR:
            return DRORiskMeasure::SURROGATE_CVAR;
        case RiskMeasureConfiguration::JOINT_RISK:
            return DRORiskMeasure::JOINT_VAR;
        default:
            return DRORiskMeasure::SURROGATE_VAR_BONFERRONI;
    }
}

inline DROGroundCostType to_dro_ground_cost(GroundCostConfiguration g) {
    switch (g) {
        case GroundCostConfiguration::W2:
        case GroundCostConfiguration::W1_METRIC:
            return DROGroundCostType::W2_BURES;
        case GroundCostConfiguration::ZERO_ONE:
            return DROGroundCostType::ZERO_ONE;
        case GroundCostConfiguration::EUCLIDEAN_MEAN:
            return DROGroundCostType::EUCLIDEAN_MEAN;
        default:
            return DROGroundCostType::W2_BURES;
    }
}

inline InjectionMode to_injection_mode(ReweightedDistributionConfiguration r) {
    switch (r) {
        case ReweightedDistributionConfiguration::SAMPLING_ONLY:
            return InjectionMode::QSTAR_SAMPLE;
        case ReweightedDistributionConfiguration::INJECTION_ONLY:
            return InjectionMode::DRO;
        case ReweightedDistributionConfiguration::SAMPLING_AND_INJECTION:
            return InjectionMode::DRO;
        default:
            return InjectionMode::QSTAR_SAMPLE;
    }
}

inline ModeBeliefConfig to_mode_belief_config(
    NominalModeBeliefConfiguration n,
    double self_persistence_prior = 0.8
) {
    ModeBeliefConfig cfg;
    cfg.prior = DirichletPrior::KRICHEVSKY_TROFIMOV;
    cfg.self_persistence_prior =
        (n == NominalModeBeliefConfiguration::STICKY) ? self_persistence_prior
                                                      : 0.0;
    return cfg;
}

/**
 * @brief DRO-axis knobs for an experiment.
 *
 * Compose with MPC / Obstacle / Environment / Rollout settings to build a
 * full ExperimentConfig.
 */
struct DROExperimentConfig {
    DROConfiguration dro = DROConfiguration::BASE;
    ReweightedDistributionConfiguration reweighting =
        ReweightedDistributionConfiguration::SAMPLING_ONLY;
    RiskMeasureConfiguration risk_measure =
        RiskMeasureConfiguration::VAR_BONFERRONI;
    NominalModeBeliefConfiguration mode_belief_kind =
        NominalModeBeliefConfiguration::DIRICHLET;
    GroundCostConfiguration ground_cost = GroundCostConfiguration::W2;

    double eps_wass = 0.1;
    double sigma_scale = 1.0;
    double fixed_rho = -1.0;  ///< >0 forces constant radius
    bool use_calibrated_radius = true;
    bool use_primal_ot = true;
    double confidence_beta = 0.05;
    bool use_entropic_allocator = false;
    double entropic_tau = 0.05;
    int joint_risk_samples = 8000;
    int dro_injection_count = 1;
    double softmax_tau = 5.0;
    double eps_greedy_epsilon = 0.3;
    ModeBeliefConfig mode_belief{};
    bool use_markov_mode_sampling = false;

    bool enabled() const { return dro == DROConfiguration::DRO; }
};

// ============================================================================
// MPC Configuration Enums
// ============================================================================

enum class MPCConfiguration {
    MPC,     ///< Point-to-point / goal-tracking MPC
    MPCC,    ///< Contouring control (path following)
    SH_MPCC, ///< Safe-horizon MPCC
};

// ============================================================================
// MPC Configuration Enum Names
// ============================================================================

inline std::string mpc_configuration_name(MPCConfiguration m) {
    switch (m) {
        case MPCConfiguration::MPC:     return "mpc";
        case MPCConfiguration::MPCC:    return "mpcc";
        case MPCConfiguration::SH_MPCC: return "sh_mpcc";
        default: return "unknown";
    }
}

// ============================================================================
// MPC Configuration Settings
// ============================================================================

/**
 * @brief Controller / QP-axis knobs for an experiment.
 */
struct MPCExperimentConfig {
    MPCConfiguration mpc = MPCConfiguration::SH_MPCC;
    int horizon = 20;
    int num_scenarios = 20;
    int num_discs = 3;
    double vehicle_length = 4.0;
    bool safe_horizon_enabled = true;
    int forced_safe_horizon = -1;
    int safe_horizon_min = 3;
    bool enable_contouring_constraints = false;
    double road_width = 7.0;
    WeightType weight_type = WeightType::FREQUENCY;
    int max_history_length = -1;

    bool uses_safe_horizon() const {
        return mpc == MPCConfiguration::SH_MPCC || safe_horizon_enabled;
    }

    bool uses_contouring() const {
        return mpc == MPCConfiguration::MPCC ||
               mpc == MPCConfiguration::SH_MPCC ||
               enable_contouring_constraints;
    }
};

// ============================================================================
// Obstacle Configuration Enums
// ============================================================================

/**
 * @brief When the ground-truth obstacle is allowed to change mode.
 *
 * PER_STEP          -- switch may fire every rollout step (realistic; CDC'26).
 * HOLD_OVER_HORIZON -- mode frozen for `horizon` steps so the mode is constant
 *                      across any one prediction horizon (Theorem-1 regime).
 */
enum class ModeSwitchRegime {
    PER_STEP,
    HOLD_OVER_HORIZON
};

enum class ObstacleLayoutConfiguration {
    SINGLE,            ///< One obstacle
    MULTI_INDEPENDENT, ///< Several obstacles, one class each
    MULTI_SHARED_CLASS ///< Several obstacles sharing class history
};

// ============================================================================
// Obstacle Configuration Enum Names
// ============================================================================

inline std::string switch_regime_name(ModeSwitchRegime r) {
    switch (r) {
        case ModeSwitchRegime::PER_STEP:          return "PerStep";
        case ModeSwitchRegime::HOLD_OVER_HORIZON: return "HoldOverHorizon";
        default: return "unknown";
    }
}

inline std::string obstacle_layout_name(ObstacleLayoutConfiguration L) {
    switch (L) {
        case ObstacleLayoutConfiguration::SINGLE:            return "single";
        case ObstacleLayoutConfiguration::MULTI_INDEPENDENT: return "multi_independent";
        case ObstacleLayoutConfiguration::MULTI_SHARED_CLASS: return "multi_shared_class";
        default: return "unknown";
    }
}

// ============================================================================
// Obstacle Configuration Settings
// ============================================================================

struct DistributionShiftConfig {
    double rho = 0.0;
    double dangerous_boost = 0.0;
    int boosted_mode = -1;
};

/**
 * @brief Ground-truth obstacle / mode-process knobs for an experiment.
 */
struct ObstacleExperimentConfig {
    ObstacleLayoutConfiguration layout = ObstacleLayoutConfiguration::SINGLE;
    ModeSwitchRegime switch_regime = ModeSwitchRegime::PER_STEP;
    double switch_prob = 0.1;
    int num_modes = 4;
    int num_obstacles = 1;
    int obstacles_per_class = 1;

    std::vector<std::string> obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"
    };
    std::string rare_mode = "lane_change_left";
    double rare_switch_prob = 0.05;

    std::vector<double> obs_arc_fractions;  ///< Empty => auto placement
    std::vector<ObstacleState> initial_obstacle_states;  ///< Overrides arc fracs
    DistributionShiftConfig shift;

    void apply_layout() {
        switch (layout) {
            case ObstacleLayoutConfiguration::SINGLE:
                num_obstacles = 1;
                obstacles_per_class = 1;
                break;
            case ObstacleLayoutConfiguration::MULTI_INDEPENDENT:
                if (num_obstacles < 2) num_obstacles = 4;
                obstacles_per_class = 1;
                break;
            case ObstacleLayoutConfiguration::MULTI_SHARED_CLASS:
                if (num_obstacles < 2) num_obstacles = 4;
                obstacles_per_class = num_obstacles;
                break;
        }
    }
};

// ============================================================================
// Environment Configuration Enums
// ============================================================================

// NOTE: every EnvironmentType below runs on the SAME S-curve reference path
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

// ============================================================================
// Environment Configuration Enum Names
// ============================================================================

inline std::string environment_name(EnvironmentType env) {
    switch (env) {
        case EnvironmentType::OVERTAKE_SLOW_LEAD: return "OvertakeSlowLead";
        case EnvironmentType::NARROW_CORRIDOR:    return "Narrow";
        case EnvironmentType::INTERSECTION:       return "Intersection";
        case EnvironmentType::ONCOMING:           return "Oncoming";
        default: return "unknown";
    }
}

// ============================================================================
// Environment Configuration Settings
// ============================================================================

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

// ============================================================================
// Sampling Configuration Enums
// ============================================================================

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

// ============================================================================
// Sampling Configuration Enum Names
// ============================================================================

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

// ============================================================================
// Sampling Configuration Settings
// ============================================================================

struct SamplingExperimentConfig {
    SamplingBaseline baseline = SamplingBaseline::STANDARD;

    WeightType weight_type() const { return baseline_to_weight(baseline); }
};

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
// Rollout Configuration Constants
// ============================================================================

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

// ============================================================================
// Rollout Configuration Settings
// ============================================================================

struct RolloutExperimentConfig {
    int rollout_steps = 60;
    std::string scenario_tag = "baseline";
    std::string method_name;  ///< Empty => auto from DRO/MPC labels

    /// Called after mode observation, before solve.
    /// Args: (step, obstacle_id, obstacle_sim, controller, rng)
    std::function<void(int, int, ObstacleSim&, AdaptiveScenarioMPC&, std::mt19937&)>
        step_callback;
};

// ============================================================================
// Legacy Composition Helpers (AblationVariant / PaperVariant)
// ============================================================================
//
// Kept for existing runners and ablations. Prefer composing the section
// configs above for new experiments.

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
        case AblationVariant::RANDOM_INJECTION:  return "random_injection";
        case AblationVariant::ALWAYS_INJECT:     return "always_inject";
        default: return "unknown";
    }
}

enum class PaperVariant {
    BASE,
    BASE_SH,
    DRO,
    DRO_SH
};

inline const std::vector<PaperVariant> ALL_VARIANTS = {
    PaperVariant::BASE,
    PaperVariant::BASE_SH,
    PaperVariant::DRO,
    PaperVariant::DRO_SH
};

inline std::string variant_name(PaperVariant v) {
    switch (v) {
        case PaperVariant::BASE:    return "Base";
        case PaperVariant::BASE_SH: return "Base+SH";
        case PaperVariant::DRO:     return "DRO";
        case PaperVariant::DRO_SH:  return "DRO+SH";
        default: return "unknown";
    }
}

inline bool uses_dro(PaperVariant v) {
    return v == PaperVariant::DRO || v == PaperVariant::DRO_SH;
}

inline bool uses_sh(PaperVariant v) {
    return v == PaperVariant::BASE_SH || v == PaperVariant::DRO_SH;
}

// ============================================================================
// Assembled Experiment Configuration
// ============================================================================

struct SeedBundle {
    unsigned master;
    unsigned env;
    unsigned predictor;
    unsigned scenario;
};

/**
 * @brief Full experiment configuration.
 *
 * Flat fields remain for existing call sites. Prefer building via
 * assemble_experiment_config() from the section structs above when
 * composing new complex configurations.
 *
 * Section ownership of fields:
 *   DRO        — enable_dro, injection_mode, risk/ground-cost/radius knobs
 *   MPC        — horizon, scenarios, discs, safe horizon, contouring
 *   Obstacle   — modes, switching, multi-obstacle layout, shift
 *   Environment— path, ego init, completion
 *   Sampling   — weight_type / baseline
 *   Rollout    — steps, tags, callbacks
 */
struct ExperimentConfig {
    // ---- MPC ----
    int num_scenarios = 20;
    int horizon = 20;
    int num_discs = 3;
    double vehicle_length = 4.0;
    bool safe_horizon_enabled = true;
    int forced_safe_horizon = -1;
    int safe_horizon_min = 3;
    bool enable_contouring_constraints = false;
    double road_width = 7.0;
    WeightType weight_type = WeightType::FREQUENCY;
    int max_history_length = -1;

    // ---- DRO ----
    double eps_wass = 0.1;
    double sigma_scale = 1.0;
    bool enable_dro = false;
    InjectionMode injection_mode = InjectionMode::QSTAR_SAMPLE;
    AblationVariant ablation = AblationVariant::NO_INJECTION;
    DROGroundCostType ground_cost = DROGroundCostType::W2_BURES;
    bool use_calibrated_radius = true;
    bool use_primal_ot = true;
    double confidence_beta = 0.05;
    bool use_entropic_allocator = false;
    double entropic_tau = 0.05;
    DRORiskMeasure risk_measure = DRORiskMeasure::SURROGATE_VAR_BONFERRONI;
    int joint_risk_samples = 8000;
    double fixed_rho = -1.0;
    bool use_markov_mode_sampling = false;
    ModeBeliefConfig mode_belief;
    int dro_injection_count = 1;
    double softmax_tau = 5.0;
    double eps_greedy_epsilon = 0.3;

    // ---- Obstacle ----
    double switch_prob = 0.1;
    ModeSwitchRegime switch_regime = ModeSwitchRegime::PER_STEP;
    int rollout_steps = 60;
    int num_modes = 4;
    std::vector<std::string> obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"
    };
    std::string rare_mode = "lane_change_left";
    double rare_switch_prob = 0.05;
    int num_obstacles = 1;
    int obstacles_per_class = 1;
    std::vector<double> obs_arc_fractions;
    std::vector<ObstacleState> initial_obstacle_states;
    DistributionShiftConfig shift;

    // ---- Environment ----
    std::optional<ReferencePath> custom_ref_path;
    std::optional<EgoState> custom_initial_ego;
    bool path_completion_termination = true;
    double path_completion_fraction = 0.95;

    // ---- Rollout / meta ----
    std::function<void(int, int, ObstacleSim&, AdaptiveScenarioMPC&, std::mt19937&)>
        step_callback;
    std::string scenario_tag = "baseline";
    std::string method_name;
};

/**
 * @brief Assemble a flat ExperimentConfig from independent section configs.
 *
 * Example:
 *   auto cfg = assemble_experiment_config(
 *       dro_cfg, mpc_cfg, obs_cfg, env_cfg, sampling_cfg, rollout_cfg);
 */
inline ExperimentConfig assemble_experiment_config(
    DROExperimentConfig dro,
    MPCExperimentConfig mpc,
    ObstacleExperimentConfig obstacles,
    EnvironmentExperimentConfig environment,
    SamplingExperimentConfig sampling = {},
    RolloutExperimentConfig rollout = {}
) {
    obstacles.apply_layout();
    if (dro.mode_belief_kind == NominalModeBeliefConfiguration::STICKY &&
        dro.mode_belief.self_persistence_prior <= 0.0) {
        dro.mode_belief = to_mode_belief_config(dro.mode_belief_kind);
    } else if (dro.mode_belief_kind == NominalModeBeliefConfiguration::DIRICHLET) {
        dro.mode_belief = to_mode_belief_config(dro.mode_belief_kind);
    }

    ExperimentConfig cfg;

    // MPC
    cfg.horizon = mpc.horizon;
    cfg.num_scenarios = mpc.num_scenarios;
    cfg.num_discs = mpc.num_discs;
    cfg.vehicle_length = mpc.vehicle_length;
    cfg.safe_horizon_enabled = mpc.uses_safe_horizon();
    cfg.forced_safe_horizon = mpc.forced_safe_horizon;
    cfg.safe_horizon_min = mpc.safe_horizon_min;
    cfg.enable_contouring_constraints = mpc.uses_contouring();
    cfg.road_width = mpc.road_width;
    cfg.weight_type = sampling.baseline == SamplingBaseline::STANDARD
                          ? mpc.weight_type
                          : sampling.weight_type();
    cfg.max_history_length = mpc.max_history_length;

    // DRO
    cfg.enable_dro = dro.enabled();
    cfg.injection_mode = dro.enabled() ? to_injection_mode(dro.reweighting)
                                       : InjectionMode::NONE;
    cfg.eps_wass = dro.eps_wass;
    cfg.sigma_scale = dro.sigma_scale;
    cfg.fixed_rho = dro.fixed_rho;
    cfg.use_calibrated_radius = dro.use_calibrated_radius;
    cfg.use_primal_ot = dro.use_primal_ot;
    cfg.confidence_beta = dro.confidence_beta;
    cfg.use_entropic_allocator = dro.use_entropic_allocator;
    cfg.entropic_tau = dro.entropic_tau;
    cfg.risk_measure = to_dro_risk_measure(dro.risk_measure);
    cfg.joint_risk_samples = dro.joint_risk_samples;
    cfg.dro_injection_count = dro.dro_injection_count;
    cfg.softmax_tau = dro.softmax_tau;
    cfg.eps_greedy_epsilon = dro.eps_greedy_epsilon;
    cfg.ground_cost = to_dro_ground_cost(dro.ground_cost);
    cfg.mode_belief = dro.mode_belief;
    cfg.use_markov_mode_sampling = dro.use_markov_mode_sampling;
    cfg.ablation = dro.enabled() ? AblationVariant::DRO_FULL
                                 : AblationVariant::NO_INJECTION;

    // Q* sampling + injection: keep injection_mode as DRO (inject) while
    // callers that need pure Q* resampling use SAMPLING_ONLY -> QSTAR_SAMPLE.
    if (dro.enabled() &&
        dro.reweighting ==
            ReweightedDistributionConfiguration::SAMPLING_AND_INJECTION) {
        cfg.injection_mode = InjectionMode::DRO;
    }

    // Obstacle
    cfg.switch_prob = obstacles.switch_prob;
    cfg.switch_regime = obstacles.switch_regime;
    cfg.num_modes = obstacles.num_modes;
    cfg.num_obstacles = obstacles.num_obstacles;
    cfg.obstacles_per_class = obstacles.obstacles_per_class;
    cfg.obs_modes = obstacles.obs_modes;
    cfg.rare_mode = obstacles.rare_mode;
    cfg.rare_switch_prob = obstacles.rare_switch_prob;
    cfg.obs_arc_fractions = obstacles.obs_arc_fractions;
    cfg.initial_obstacle_states = obstacles.initial_obstacle_states;
    cfg.shift = obstacles.shift;

    // Environment
    cfg.custom_ref_path = environment.custom_ref_path;
    cfg.custom_initial_ego = environment.custom_initial_ego;
    cfg.path_completion_termination = environment.path_completion_termination;
    cfg.path_completion_fraction = environment.path_completion_fraction;

    // Rollout
    cfg.rollout_steps = rollout.rollout_steps;
    cfg.scenario_tag = rollout.scenario_tag;
    cfg.step_callback = std::move(rollout.step_callback);
    if (!rollout.method_name.empty()) {
        cfg.method_name = rollout.method_name;
    } else {
        cfg.method_name =
            dro_configuration_name(dro.dro) + "_" +
            mpc_configuration_name(mpc.mpc) + "_" +
            reweighted_distribution_configuration_name(dro.reweighting);
    }

    return cfg;
}

// ============================================================================
// Data Structures — Rollout Records
// ============================================================================

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
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
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
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
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
    const std::vector<std::string>& obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"},
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

#endif  // DRO_MPC_EXPERIMENT_HARNESS_HPP
