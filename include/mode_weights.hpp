/**
 * @file mode_weights.hpp
 * @brief Mode weight computation for adaptive scenario-based MPC.
 *
 * Implements Section 4: Mode History and Weights
 *
 * Supports three weight computation strategies:
 * - Uniform: Equal weights for all modes (Eq. 4)
 * - Recency: Exponential decay weighting recent observations (Eq. 5)
 * - Frequency: Weights based on observation frequency (Eq. 6)
 */

#ifndef SCENARIO_MPC_MODE_WEIGHTS_HPP
#define SCENARIO_MPC_MODE_WEIGHTS_HPP

#include "types.hpp"
#include <random>

namespace scenario_mpc {

/**
 * @brief Compute mode weights based on observation history.
 *
 * @param mode_history Observed mode history for an obstacle
 * @param weight_type Weight computation strategy
 * @param recency_decay Decay factor lambda for recency weighting
 * @param current_timestep Current timestep for recency computation
 * @return Dictionary mapping mode_id to weight (normalized to sum to 1)
 */
std::map<std::string, double> compute_mode_weights(
    const ModeHistory& mode_history,
    WeightType weight_type = WeightType::FREQUENCY,
    double recency_decay = 0.9,
    int current_timestep = 0
);

/**
 * @brief Sample a mode sequence for the prediction horizon.
 *
 * Assumes modes are i.i.d. across timesteps (can be extended for Markov).
 *
 * @param mode_weights Weights for each mode
 * @param horizon Number of timesteps to sample
 * @param rng Random number generator
 * @return List of mode_ids of length horizon
 */
std::vector<std::string> sample_mode_sequence(
    const std::map<std::string, double>& mode_weights,
    int horizon,
    std::mt19937& rng
);

/**
 * @brief Sample a single mode from the weight distribution.
 *
 * @param mode_weights Weights for each mode
 * @param rng Random number generator
 * @return Sampled mode_id
 */
std::string sample_mode_from_weights(
    const std::map<std::string, double>& mode_weights,
    std::mt19937& rng
);

/**
 * @brief Estimate mode transition probabilities from history.
 *
 * P[i,j] = P(mode_j | mode_i) estimated from observations.
 *
 * @param mode_history Observation history
 * @param modes List of mode_ids
 * @return Transition matrix (num_modes x num_modes)
 */
Eigen::MatrixXd compute_mode_transition_matrix(
    const ModeHistory& mode_history,
    const std::vector<std::string>& modes
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_MODE_WEIGHTS_HPP
