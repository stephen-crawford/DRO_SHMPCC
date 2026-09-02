/**
 * @file scenario_sampler.hpp
 * @brief Scenario sampling for adaptive scenario-based MPC.
 *
 * There are two sampling approaches:
 *
 *   (1) i.i.d. scenario sampling (de Groot 2023, arXiv:2307.01070). Each of the S
 *       scenarios is an INDEPENDENT draw: one mode per obstacle held over the horizon
 *       (drawn from a per-obstacle categorical) plus an i.i.d. Gaussian noise path.
 *       This is the sampling law the de Groot / Campi-Garatti scenario bound assumes;
 *       the WDRO reweighting only changes WHICH categorical (nominal p_hat vs worst-case
 *       q*) is drawn from — the draws stay i.i.d., so the bound still applies.
 *       Entry points: sample_scenarios (internal nominal weights) and
 *       sample_scenarios_with_weights (external weights, e.g. q* or a baseline).
 *
 *   (2) Markov-jump switching (Schuurmans & Patrinos DR-MJS, arXiv:2106.00561): the
 *       mode SWITCHES within the horizon along a mode transition matrix. mode_0 is the
 *       one-step predictive of the initial belief, then mode_k ~ T[mode_{k-1}, :].
 *       Entry point: sample_scenarios_markov.
 *
 */

#ifndef DRO_MPC_SCENARIO_SAMPLER_HPP
#define DRO_MPC_SCENARIO_SAMPLER_HPP

#include "types.hpp"
#include "mode_weights.hpp"
#include <random>

namespace dro_mpc {

/**
 * @brief Sample scenarios following Algorithm 1.
 *
 * Algorithm 1: SampleScenarios
 * Input: obstacles, mode_histories, num_scenarios S, horizon N
 * Output: List of scenarios
 *
 * Compute per-obstacle nominal weights w_m from history once (independent of s).
 * For each scenario s = 1, ..., S:
 *     For each obstacle o:
 *         1. Sample mode m^(s) ~ Categorical(w)  (held over the horizon)
 *         2. Sample noise sequence w_k ~ N(0, I)
 *         3. Propagate trajectory using the sampled mode and noise
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
    const ModeBeliefConfig& mode_belief = {},
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
 * @param rng Random number generator
 * @return List of Scenario objects
 */
std::vector<Scenario> sample_scenarios_with_weights(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    const std::map<int, std::map<std::string, double>>& per_obstacle_weights,
    int horizon,
    int num_scenarios,
    std::mt19937* rng = nullptr
);

/**
 * @brief Markov mode-sequence sampling with an explicit initial belief.
 *
 * The live entry point for markov_jump_system sampling. Differs from
 * sample_scenarios_with_mode_sequences in two ways that matter:
 *
 *  1. It accepts a per-obstacle INITIAL BELIEF. Pass the DRO Q* here and the
 *     Markov propagation starts from the reweighted distribution; pass nullptr
 *     and the belief is computed from weight_type + belief_cfg. 
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
    const ModeBeliefConfig& belief_cfg,
    std::mt19937* rng = nullptr
);

}  // namespace dro_mpc

#endif  // DRO_MPC_SCENARIO_SAMPLER_HPP
