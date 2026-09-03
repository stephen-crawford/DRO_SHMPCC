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
 *       Entry point: sample_scenarios, with an optional per-obstacle distribution.
 *
 *   (2) Markov-jump switching (Schuurmans & Patrinos DR-MJS, arXiv:2106.00561): the
 *       mode SWITCHES within the horizon along a mode transition matrix. mode_0 is the
 *       one-step predictive of the initial belief, then mode_k ~ T[mode_{k-1}, :].
 *       Pass an optional transition matrix to sample_scenarios.
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
 * @param per_obstacle_distribution Optional mode distributions. If null, nominal
 *        distributions are computed from each obstacle's history.
 * @param per_obstacle_transitions Optional Markov transition matrices keyed by
 *        obstacle ID. A supplied obstacle matrix enables switching for that
 *        obstacle; otherwise its sampled mode is held over the horizon.
 * @param rng Random number generator
 * @return List of Scenario objects
 */
std::vector<Scenario> sample_scenarios(
    const std::map<int, ObstacleState>& obstacles,
    const std::map<int, ModeHistory>& mode_histories,
    const std::map<int, std::map<std::string, double>>* per_obstacle_distribution,
    int horizon,
    int num_scenarios,
    const ModeBeliefConfig& mode_belief = {},
    const std::map<int, Eigen::MatrixXd>* per_obstacle_transitions = nullptr,
    std::mt19937* rng = nullptr
);

}  // namespace dro_mpc

#endif  // DRO_MPC_SCENARIO_SAMPLER_HPP
