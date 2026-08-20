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

namespace dro_mpc {

/// Belief over discrete modes: mode_id -> probability. Canonical type shared by the
/// Bayesian mode-belief estimator, the Markov transition/prediction functions, and the
/// scenario sampler.
using ModeDistribution = std::map<std::string, double>;

/**
 * @brief Compute the nominal mode belief from observation history.
 *
 * The belief is the Dirichlet posterior-predictive mean
 *   p_m = (n_m + a) / (N + M a),   a = belief.alpha(M),
 * so EVERY mode in the library keeps strictly positive mass (no zero-mask on
 * unobserved modes). The belief KIND (NominalBeliefKind: DIRICHLET vs STICKY)
 * shares this marginal; stickiness enters only through the Markov transition
 * prior (ModeBeliefConfig::kappa), applied by the mode-sequence samplers.
 *
 * @param mode_history Observed mode history for an obstacle (available_modes = full library).
 * @param belief Nominal-belief hyperparameters (Dirichlet prior + optional stickiness).
 * @param current_timestep Reserved (unused; retained for call-site stability).
 * @return Dictionary mapping mode_id to weight (normalized to sum to 1).
 */
std::map<std::string, double> compute_mode_weights(
    const ModeHistory& mode_history,
    const ModeBeliefConfig& belief = {},
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
 * @brief Sample a MARKOVIAN mode sequence over the horizon.
 *
 * mode_0 is drawn from `initial_belief`; each subsequent mode_k is drawn from the
 * transition row of the previous mode, T[mode_{k-1}, :]. Because the transition matrix
 * is strictly positive (Dirichlet prior), the chain can reach never-observed modes.
 *
 * @param initial_belief Belief used to seed mode_0 (e.g. the Q* override or the estimator).
 * @param transition M x M row-stochastic transition matrix indexed by `modes`.
 * @param modes Mode ordering matching `transition`.
 * @param horizon Sequence length.
 * @param rng Random number generator.
 * @return Sequence of mode_ids of length `horizon`.
 */
std::vector<std::string> sample_mode_sequence(
    const ModeDistribution& initial_belief,
    const Eigen::MatrixXd& transition,
    const std::vector<std::string>& modes,
    int horizon,
    std::mt19937& rng,
    bool predict_before_first_sample = false
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
 * @brief Estimate the mode transition matrix (Dirichlet + sticky self-persistence prior).
 *
 *   T(i,j) = (N[i][j] + a + kappa*[i==j]) / (rowsum_i + M a + kappa),
 *
 * where N[i][j] counts observed i->j transitions, `a = dirichlet_alpha` is the symmetric
 * per-row Dirichlet pseudocount, and `kappa = sticky_kappa` adds a self-transition bias so
 * that the prior diagonal E[T_ii] = (a+kappa)/(M a + kappa). With a>0 every entry is strictly
 * positive (no unreachable mode); a never-observed source row is the pure prior. Use
 * ModeBeliefConfig::alpha(M)/kappa(M) to derive (a, kappa) from a self-persistence prior theta.
 *
 * @param mode_history Observation history.
 * @param modes Mode ordering (defines the row/col indexing of T).
 * @param dirichlet_alpha Symmetric Dirichlet pseudocount a.
 * @param sticky_kappa Self-transition pseudocount kappa.
 * @return Row-stochastic transition matrix (M x M).
 */
Eigen::MatrixXd compute_mode_transition_matrix(
    const ModeHistory& mode_history,
    const std::vector<std::string>& modes,
    double dirichlet_alpha = 1.0,
    double sticky_kappa = 0.0
);

/**
 * @brief One-step HMM belief prediction:  p_{t+1}(j) = sum_i T(i,j) p_t(i)  ( = T^T p_t ).
 *
 * @param belief Current mode belief (mode_id -> prob).
 * @param transition M x M row-stochastic transition matrix (rows/cols indexed by `modes`).
 * @param modes Mode ordering matching `transition`.
 * @return Predicted belief over the same modes (normalized).
 */
ModeDistribution predict_mode_belief(
    const ModeDistribution& belief,
    const Eigen::MatrixXd& transition,
    const std::vector<std::string>& modes
);

/**
 * @brief Bayesian mode-belief update:  posterior_j proportional to (T^T prior)_j * likelihood_j.
 *
 * Propagates `prior` one step through `transition` (the predictive), multiplies by the per-mode
 * `likelihood`, and normalizes. If the normalizer is <= `floor` (degenerate / all-zero
 * likelihood), returns the predictive belief instead of dividing by zero.
 */
ModeDistribution update_mode_belief(
    const ModeDistribution& prior,
    const Eigen::MatrixXd& transition,
    const std::vector<std::string>& modes,
    const ModeDistribution& likelihood,
    double floor = 1e-12
);

}  // namespace dro_mpc

#endif  // SCENARIO_MPC_MODE_WEIGHTS_HPP
