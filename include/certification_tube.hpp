#pragma once
#include "collision_constraints.hpp"
#include <cmath>
#include <limits>

namespace dro_mpc {
// Fixed for a whole decision, including every recovery/retry. Future stages only.
inline double certification_tube_displacement(
    const std::vector<EgoState>& reference, const std::vector<EgoState>& candidate,
    int discs, double length) {
    if (reference.size() < 2 || candidate.size() != reference.size())
        return std::numeric_limits<double>::infinity();
    double maximum = 0;
    for (std::size_t k = 1; k < reference.size(); ++k) {
        const auto a = compute_ego_disc_positions(reference[k], discs, length);
        const auto b = compute_ego_disc_positions(candidate[k], discs, length);
        for (int j = 0; j < discs; ++j) {
            const double distance = (a[j] - b[j]).norm();
            if (!std::isfinite(distance)) return std::numeric_limits<double>::infinity();
            maximum = std::max(maximum, distance);
        }
    }
    return maximum;
}

// Conditional Gaussian diagnostic: strict disc overlap, fixed unit normal, and
// ||c-c_ref|| <= tube_radius. Degenerate Gaussians retain the strict inequality.
inline double tube_gaussian_probability(const Eigen::Vector2d& center,
    const Eigen::Vector2d& mean, const Eigen::Matrix2d& covariance,
    double collision_radius, double tube_radius) {
    const Eigen::Vector2d diff = mean - center;
    const double distance = diff.norm();
    const Eigen::Vector2d normal = distance > 0 ? Eigen::Vector2d(diff / distance)
                                               : Eigen::Vector2d(1, 0);
    const double projected_mean = collision_radius - normal.dot(diff) + tube_radius;
    const double variance = normal.dot(covariance * normal);
    if (!std::isfinite(projected_mean) || !std::isfinite(variance) || variance < 0)
        return 1.0; // An unusable model must never produce an optimistic bound.
    return variance == 0 ? double(projected_mean > 0)
        : 0.5 * std::erfc(-projected_mean / std::sqrt(2 * variance));
}
}
