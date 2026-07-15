/**
 * @file scenario_sampler.hpp
 * @brief Scenario sampling for adaptive scenario-based MPC.
 *
 * Implements Algorithm 1: SampleScenarios
 *
 * Generates scenarios by sampling mode sequences and noise realizations
 * for each obstacle, then combining into joint scenarios.
 */

#ifndef SCENARIO_MPC_SCENARIO_SAMPLER_HPP
#define SCENARIO_MPC_SCENARIO_SAMPLER_HPP

#include "types.hpp"
#include "mode_weights.hpp"
#include <random>

namespace scenario_mpc {

/**
 * @brief Sample scenarios following Algorithm 1.
 *
 * Algorithm 1: SampleScenarios
 * Input: obstacles, mode_histories, num_scenarios S, horizon N
 * Output: List of scenarios
 *
 * For each scenario s = 1, ..., S:
 *     For each obstacle o:
 *         1. Compute mode weights w_m from history
 *         2. Sample mode sequence m^(s) ~ Categorical(w)
 *         3. Sample noise sequence w_k ~ N(0, I)
 *         4. Propagate trajectory using sampled modes and noise
 *
 * @param obstacles Dict mapping obstacle_id to current ObstacleState
 * @param mode_histories Dict mapping obstacle_id to ModeHistory
 * @param horizon Prediction horizon N
 * @param num_scenarios Number of scenarios to sample S
 * @param weight_type Strategy for computing mode weights
 * @param recency_decay Decay factor for recency weighting
 * @param current_timestep Current timestep for recency computation
 * @param rng Random number generator
 * @return List of Scenario objects
 */
std::vector<Scenario> sample_scenarios(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    int horizon,
    int num_scenarios,
    WeightType weight_type = WeightType::FREQUENCY,
    double recency_decay = 0.9,
    int current_timestep = 0,
    std::mt19937* rng = nullptr
);

/**
 * @brief Sample scenarios with mode coverage guarantee.
 *
 * Two-phase algorithm:
 *   Phase 1 (Coverage): For each obstacle, collect all modes with nonzero weight.
 *     The first num_coverage scenarios deterministically cover each mode, ensuring
 *     at least one scenario per mode per obstacle.
 *   Phase 2 (Remaining): Fill remaining scenarios via normal weight-based sampling.
 *
 * This eliminates the probability of zero representation for rare modes.
 *
 * @param obstacles Dict mapping obstacle_id to current ObstacleState
 * @param mode_histories Dict mapping obstacle_id to ModeHistory
 * @param horizon Prediction horizon N
 * @param num_scenarios Number of scenarios to sample S
 * @param weight_type Strategy for computing mode weights
 * @param recency_decay Decay factor for recency weighting
 * @param current_timestep Current timestep for recency computation
 * @param rng Random number generator
 * @return List of Scenario objects with mode coverage guarantee
 */
std::vector<Scenario> sample_scenarios_with_mode_coverage(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    int horizon,
    int num_scenarios,
    WeightType weight_type = WeightType::FREQUENCY,
    double recency_decay = 0.9,
    int current_timestep = 0,
    std::mt19937* rng = nullptr
);

/**
 * @brief Sample scenarios with pre-computed per-obstacle mode weights.
 *
 * Same as sample_scenarios() but uses externally provided weights (e.g. from DRO)
 * instead of computing them from mode histories internally.
 *
 * @param obstacles Dict mapping obstacle_id to current ObstacleState
 * @param mode_histories Dict mapping obstacle_id to ModeHistory
 * @param per_obstacle_weights Pre-computed weights: obstacle_id -> {mode_id -> weight}
 * @param horizon Prediction horizon N
 * @param num_scenarios Number of scenarios to sample S
 * @param ensure_mode_coverage If true, guarantee at least one scenario per mode
 * @param rng Random number generator
 * @return List of Scenario objects
 */
std::vector<Scenario> sample_scenarios_with_weights(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    const std::map<int, std::map<std::string, double>>& per_obstacle_weights,
    int horizon,
    int num_scenarios,
    bool ensure_mode_coverage = false,
    std::mt19937* rng = nullptr
);

/**
 * @brief Sample scenarios with time-varying mode sequences.
 *
 * More sophisticated version that samples a different mode for each timestep,
 * allowing for mode switches during the prediction horizon.
 *
 * @param obstacles Dict mapping obstacle_id to current ObstacleState
 * @param mode_histories Dict mapping obstacle_id to ModeHistory
 * @param horizon Prediction horizon
 * @param num_scenarios Number of scenarios to sample
 * @param weight_type Strategy for computing mode weights
 * @param rng Random number generator
 * @return List of Scenario objects
 */
std::vector<Scenario> sample_scenarios_with_mode_sequences(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    int horizon,
    int num_scenarios,
    WeightType weight_type = WeightType::FREQUENCY,
    std::mt19937* rng = nullptr
);

/**
 * @brief Markov mode-sequence sampling with an explicit initial belief.
 *
 * The live entry point for use_markov_mode_sampling. Differs from
 * sample_scenarios_with_mode_sequences in two ways that matter:
 *
 *  1. It accepts a per-obstacle INITIAL BELIEF. Pass the DRO Q* here and the
 *     Markov propagation starts from the reweighted distribution; pass nullptr
 *     and the belief is computed from weight_type + belief_cfg. Without this the
 *     DRO path could not use Markov sampling at all, since Q* is a weight map and
 *     sample_scenarios_with_mode_sequences only takes a WeightType.
 *  2. The belief and transition matrix are built ONCE per obstacle rather than
 *     once per (scenario, obstacle) -- they do not depend on the scenario index,
 *     so recomputing them S times was pure waste.
 *
 * @param per_obstacle_belief Optional initial belief per obstacle (e.g. DRO Q*).
 *        nullptr => derive from weight_type. Obstacles absent from the map fall
 *        back to the derived belief.
 * @param belief_cfg Dirichlet / sticky prior hyperparameters.
 */
std::vector<Scenario> sample_scenarios_markov(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    const std::map<int, std::map<std::string, double>>* per_obstacle_belief,
    int horizon,
    int num_scenarios,
    WeightType weight_type,
    const ModeBeliefConfig& belief_cfg,
    std::mt19937* rng = nullptr
);

/**
 * @brief Sample scenarios with stratified allocation across modes.
 *
 * Allocates floor(S * w_m) scenarios per mode m, fills remainder proportionally.
 * Guarantees representation proportional to mode weights.
 *
 * @param obstacles Dict mapping obstacle_id to current ObstacleState
 * @param mode_histories Dict mapping obstacle_id to ModeHistory
 * @param horizon Prediction horizon N
 * @param num_scenarios Number of scenarios to sample S
 * @param weight_type Strategy for computing mode weights
 * @param recency_decay Decay factor for recency weighting
 * @param current_timestep Current timestep for recency computation
 * @param rng Random number generator
 * @return List of Scenario objects with stratified mode allocation
 */
std::vector<Scenario> sample_scenarios_stratified(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    int horizon,
    int num_scenarios,
    WeightType weight_type = WeightType::FREQUENCY,
    double recency_decay = 0.9,
    int current_timestep = 0,
    std::mt19937* rng = nullptr
);

/**
 * @brief Compute required number of scenarios using Theorem 1.
 *
 * Theorem 1: For epsilon-chance constraint satisfaction with confidence 1-beta,
 * the required number of scenarios is:
 *     S >= 2/epsilon * (ln(1/beta) + n_x)
 *
 * where n_x is the number of decision variables.
 *
 * @param epsilon Violation probability (1 - confidence_level)
 * @param beta Risk parameter
 * @param num_decision_vars Number of decision variables n_x
 * @return Minimum number of scenarios required
 */
int compute_required_scenarios(double epsilon, double beta, int num_decision_vars);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_SCENARIO_SAMPLER_HPP
