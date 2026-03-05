/**
 * @file experiment_harness.hpp
 * @brief Shared types, stat helpers, and CSV writer for experiment runner.
 *
 * Provides the infrastructure for all experiments A-K in the results-
 * strengthening upgrade. Each experiment uses run_rollout() to collect
 * per-rollout RolloutRecords, then writes them via CSVWriter.
 */

#ifndef SCENARIO_MPC_EXPERIMENT_HARNESS_HPP
#define SCENARIO_MPC_EXPERIMENT_HARNESS_HPP

#include "types.hpp"
#include "config.hpp"
#include "wasserstein_dro.hpp"
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <random>
#include <functional>

namespace scenario_mpc {

// ============================================================================
// Enums
// ============================================================================

/**
 * @brief Ablation variant for Section E experiments.
 */
enum class AblationVariant {
    NO_INJECTION,       ///< Base scenario MPC (no DRO)
    DRO_FULL,           ///< Full DRO risk + covariance
    DRO_NO_COV,         ///< DRO risk without covariance inflation
    DRO_DISTANCE_ONLY,  ///< DRO with sigma_scale=0 (distance only)
    RANDOM_INJECTION,   ///< Inject one random mode per step
    ALWAYS_INJECT       ///< Inject all modes deterministically
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
// Data Structures
// ============================================================================

/**
 * @brief Deterministic seed bundle derived from a master seed.
 */
struct SeedBundle {
    unsigned master;
    unsigned env;
    unsigned predictor;
    unsigned scenario;
};

/**
 * @brief Distribution shift configuration for Section D.
 */
struct DistributionShiftConfig {
    double rho = 0.0;             ///< Blend: p_true = (1-rho)*p_hat + rho*uniform
    double dangerous_boost = 0.0; ///< Extra mass on highest-risk mode
    int boosted_mode = -1;        ///< Specific mode index to boost (-1 = auto)
};

/**
 * @brief Full experiment configuration wrapping all MPC + experiment params.
 */
struct ExperimentConfig {
    // MPC parameters
    int num_scenarios = 20;
    double eps_wass = 0.1;
    double sigma_scale = 1.0;
    int horizon = 20;
    int num_discs = 3;
    bool safe_horizon_enabled = true;
    double switch_prob = 0.1;
    int rollout_steps = 60;
    int num_modes = 4;

    // Experiment variant
    AblationVariant ablation = AblationVariant::DRO_FULL;
    DROGroundCostType ground_cost = DROGroundCostType::W2_BURES;
    DistributionShiftConfig shift;

    // Mode configuration
    std::vector<std::string> obs_modes = {
        "constant_velocity", "turn_left", "turn_right", "decelerating"
    };
    std::string rare_mode = "lane_change_left";
    double rare_switch_prob = 0.05;

    // Scenario / edge-case tag (e.g. "baseline", "high_switch", "low_S", "distribution_shift")
    std::string scenario_tag = "baseline";

    // Tuning overrides (<= 0 means use default)
    double certificate_radius_override = 0.0;  ///< default 0.15
    double rta_threshold_override = 0.0;       ///< default 1.5
    double bandit_beta_override = 0.0;          ///< default 1.0
    // Quotient-space reduction (SHMPC_QuotientSpace): reduce S scenarios to K for efficiency
    int quotient_num_override = 0;              ///< if > 0, number of quotient representatives K; else K = max(1, 40% of num_scenarios)

    // GAN adversarial scenarios (optional path to CSV from gan_adversarial.py)
    std::string gan_scenario_csv_path;
    bool use_gan_cache = true;           ///< if true, load CSV once per rollout and reuse (faster)
    int gan_num_scenarios_override = 0;   ///< if > 0, use this many GAN scenarios per solve (fewer = faster)
    int gan_reduced_num_scenarios = 12;    ///< SHMPC_GAN_Reduced uses this (12 = real-time: >=50% safety, <10ms)
    // Reservoir adversarial scenarios (optional path to CSV from reservoir_adversarial.py; same format)
    std::string reservoir_scenario_csv_path;
    // Seek-avoid (pursuit) scenarios from seek_avoid.py
    std::string seek_avoid_scenario_csv_path;
    // ML model trained on seek-avoid data (seek_avoid_ml.py)
    std::string seek_avoid_ml_scenario_csv_path;
    // Stiefel manifold GAN scenarios (stiefel_gan.py; same CSV format as GAN)
    std::string stiefel_gan_scenario_csv_path;
};

/**
 * @brief Per-rollout record with all metrics (CSV row).
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
    double control_effort = 0.0;
    int constraint_active_count = 0;
    int missed_mode_steps = 0;
    int total_steps = 0;

    // Timing metrics
    double avg_solve_ms = 0.0;
    double p50_solve_ms = 0.0;
    double p95_solve_ms = 0.0;
    double max_solve_ms = 0.0;

    // DRO-specific
    int total_dro_injected = 0;
    double avg_safe_horizon = 0.0;
    double clearance_5pct = 0.0;
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

/// Wilson score confidence interval for a proportion
std::pair<double, double> wilson_ci(int successes, int n, double z = 1.96);

/**
 * @brief Bootstrap paired delta for collision rate differences.
 * @param base_collisions  Boolean vector for base method
 * @param dro_collisions   Boolean vector for DRO method (paired)
 * @param n_bootstrap      Number of bootstrap resamples
 * @param rng              Random number generator
 * @return (mean_delta, ci_low, ci_high) where delta = base_rate - dro_rate
 */
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

/**
 * @brief McNemar's chi-squared test with continuity correction.
 * @param b  count of (base=collision, dro=no collision) pairs
 * @param c  count of (base=no collision, dro=collision) pairs
 * @return chi2 statistic (compare to chi2(1) critical values: 3.84 for p<0.05)
 */
double mcnemar_chi2(int b, int c);

/**
 * @brief Effect size summary.
 */
struct EffectSizes {
    double abs_delta;    ///< p_base - p_dro
    double rel_delta;    ///< (p_base - p_dro) / p_base
    double risk_ratio;   ///< p_dro / p_base
    double cohens_h;     ///< 2*arcsin(sqrt(p_base)) - 2*arcsin(sqrt(p_dro))
};

EffectSizes compute_effect_sizes(double p_base, double p_dro);

// ============================================================================
// Seed Derivation
// ============================================================================

SeedBundle derive_seeds(unsigned master_seed, int idx);

// ============================================================================
// Rollout Runner
// ============================================================================

/**
 * @brief Run a single MPC rollout with the given configuration and seed.
 */
RolloutRecord run_experiment_rollout(
    const ExperimentConfig& config,
    unsigned seed
);

/**
 * @brief Run a single rollout with a named method for Future SL MPC comparison.
 * method: "SHMPC" (safe horizon, no DRO), "SHMPC_DRO", "SHMPC_AdaptiveDRO", "SHMPC_RTA".
 */
RolloutRecord run_experiment_rollout_future_sl(
    const ExperimentConfig& config,
    unsigned seed,
    const std::string& method
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_EXPERIMENT_HARNESS_HPP
