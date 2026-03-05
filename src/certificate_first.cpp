/**
 * @file certificate_first.cpp
 * @brief Implementation of certificate-first schema (Section 7.1).
 */

#include "certificate_first.hpp"
#include "conformal_safety.hpp"
#include <algorithm>

namespace scenario_mpc {

std::vector<double> certificate_radii_from_residuals(
    const std::vector<std::vector<double>>& residuals_per_timestep,
    double delta) {
    std::vector<double> radii;
    for (const auto& res : residuals_per_timestep) {
        radii.push_back(conformal_quantile(res, delta));
    }
    return radii;
}

double certificate_volume(const Certificate& cert) {
    double vol = 0.0;
    for (double r : cert.radii) vol += r * r;
    return vol;
}

double tighten_offset(double b_nominal, double constraint_norm_a, double radius) {
    return b_nominal - radius * constraint_norm_a;
}

}  // namespace scenario_mpc
