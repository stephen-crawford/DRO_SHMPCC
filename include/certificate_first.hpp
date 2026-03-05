/**
 * @file certificate_first.hpp
 * @brief Certificate-first learning: safety certificates as first-class outputs (Section 7.1).
 */

#ifndef SCENARIO_MPC_CERTIFICATE_FIRST_HPP
#define SCENARIO_MPC_CERTIFICATE_FIRST_HPP

#include <vector>
#include <cmath>

namespace scenario_mpc {

/**
 * @brief Certificate schema: tube C_t = ball(mean_t, r_t) with confidence delta.
 */
struct Certificate {
    std::vector<double> radii;  ///< r_t for each timestep (tube radius)
    double delta = 0.1;         ///< Miscoverage level (coverage >= 1-delta)
};

/**
 * @brief Compute certificate radii from calibration residuals (conformal quantile per step).
 */
std::vector<double> certificate_radii_from_residuals(
    const std::vector<std::vector<double>>& residuals_per_timestep,
    double delta
);

/**
 * @brief Certificate volume (sum of r_t^2) as proxy for conservativeness.
 */
double certificate_volume(const Certificate& cert);

/**
 * @brief Tighten constraint offset for robust constraint over ball: b_robust = b_nominal - r * ||a||.
 */
double tighten_offset(double b_nominal, double constraint_norm_a, double radius);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_CERTIFICATE_FIRST_HPP
