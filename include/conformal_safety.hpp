/**
 * @file conformal_safety.hpp
 * @brief Conformal safety wrappers for multi-step collision risk (Section 6.1).
 *
 * Mode-conditional conformal prediction sets C_{t,m} and boundary-aware
 * scenario allocation for scenario MPC.
 */

#ifndef SCENARIO_MPC_CONFORMAL_SAFETY_HPP
#define SCENARIO_MPC_CONFORMAL_SAFETY_HPP

#include "types.hpp"
#include <vector>
#include <map>
#include <string>

namespace scenario_mpc {

/**
 * @brief Nonconformity score: default L2 distance between predicted and actual (2D position).
 */
inline double default_nonconformity_score(const Eigen::Vector2d& pred, const Eigen::Vector2d& actual) {
    return (pred - actual).norm();
}

/**
 * @brief Compute (1-delta)-quantile of residuals for each mode (mode-conditional calibration).
 *
 * @param residuals_per_mode  mode_id -> list of residuals r = rho(pred, actual)
 * @param delta               miscoverage level (target coverage >= 1-delta)
 * @return mode_id -> conformal quantile q_m
 */
std::map<std::string, double> conformal_quantiles_per_mode(
    const std::map<std::string, std::vector<double>>& residuals_per_mode,
    double delta
);

/**
 * @brief Compute conformal quantile from a single vector of residuals (global calibration).
 */
double conformal_quantile(const std::vector<double>& residuals, double delta);

/**
 * @brief Boundary score for mode m: max over t and over xi in C_{t,m} of g_t(x_t, xi).
 *
 * For a ball C_{t,m} = { xi : ||xi - mean_t,m|| <= q_m }, the worst-case constraint
 * value is: mean_constraint_value + q_m * ||a_t|| (linearization). Here we use
 * the provided mean constraint value and gradient norm to approximate max over ball.
 *
 * @param constraint_value_at_mean  g_t(x_t, mean_t) for each (t, mode)
 * @param constraint_gradient_norm  ||nabla_xi g_t|| at mean (for ball radius effect)
 * @param radius_per_mode           q_m for each mode
 * @param mode_ids                  list of mode ids
 * @return mode_id -> boundary score s_m
 */
std::map<std::string, double> boundary_scores(
    const std::map<std::string, std::vector<double>>& constraint_value_at_mean,
    const std::map<std::string, std::vector<double>>& constraint_gradient_norm,
    const std::map<std::string, double>& radius_per_mode,
    const std::vector<std::string>& mode_ids
);

/**
 * @brief Allocate scenario counts S_m proportional to exp(alpha * s_m), minimum 1 per mode.
 *
 * @param boundary_scores  mode_id -> s_m
 * @param total_S          total number of scenarios
 * @param alpha            temperature for softmax
 * @return mode_id -> S_m (sum <= total_S, each >= 1 if possible)
 */
std::map<std::string, int> allocate_scenarios_boundary_aware(
    const std::map<std::string, double>& boundary_scores,
    int total_S,
    double alpha = 1.0
);

/**
 * @brief Conformal tube radius at timestep t for mode m: radius of set C_{t,m}.
 */
inline double conformal_tube_radius(double quantile_qm) {
    return quantile_qm;
}

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_CONFORMAL_SAFETY_HPP
