/**
 * @file scenario_sampler.hpp
 * @brief Scenario sampling for scenario MPC.
 *
 * The controller defaults to DirectTrajectorySampler, which consumes supplied
 * prediction paths using the reference scenario_module Gaussian database.
 * The following legacy sampling approaches are selected by multimodal_bilevel:
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

/// Predictor input, matching MPCPlanner::PredictionStep. Radii are consumed
/// literally as noise standard deviations, then integrated and multiplied by dt
/// exactly once by the sampler, just like the reference Safe-Horizon method.
struct GaussianPredictionStep {
    Eigen::Vector2d position = Eigen::Vector2d::Zero();
    double angle = 0.0;
    double major_radius = 1.0;
    double minor_radius = 1.0;
};

struct GaussianPredictionMode {
    std::string mode_id;
    double probability = 1.0;
    // Exactly N supplied means, preserved without shifting or extrapolation.
    // Input steps[k] maps to output steps[k+1]; output step 0 is measured state.
    std::vector<GaussianPredictionStep> steps;
};

using ObstaclePredictions = std::map<int, std::vector<GaussianPredictionMode>>;

/// scenario_module::Sampler::IntegrateAndTranslateToMeanAndVariance semantics:
/// a persistent Gaussian database, per-stage batch picks/permutations shared
/// across obstacles, a held supplied mode, and accumulated velocity noise * dt.
class DirectTrajectorySampler {
public:
    std::vector<Scenario> sample(
        const std::map<int, ObstacleState>& obstacles,
        const ObstaclePredictions& predictions,
        int horizon, int num_scenarios, double dt, int batch_count,
        std::mt19937& rng, int scenario_id_offset = 0);

    void reset() { standard_samples_.clear(); }

private:
    std::vector<std::vector<Eigen::Vector2d>> standard_samples_;
    friend struct DirectTrajectorySamplerTestAccess;
};

/// Simple predictor for callers without externally supplied paths. It produces
/// one constant-velocity mean path, with velocity-noise standard deviation sigma.
ObstaclePredictions constant_velocity_predictions(
    const std::map<int, ObstacleState>& obstacles, int horizon, double dt, double sigma);

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
 * @param scenario_id_offset First identifier assigned to this batch.  This is
 *        required when a certified controller augments an existing sample set:
 *        support is tracked by joint scenario identifier, so appended samples
 *        must not reuse identifiers from the original batch.
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
    std::mt19937* rng = nullptr,
    int scenario_id_offset = 0,
    // One forced draw per available mode per obstacle, then residual IID draws.
    // Held modes only; S >= each mode count. No IID scenario certificate.
    bool stratified = false
);

}  // namespace dro_mpc

#endif  // DRO_MPC_SCENARIO_SAMPLER_HPP
