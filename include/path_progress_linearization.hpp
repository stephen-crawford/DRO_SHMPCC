#ifndef DRO_MPC_PATH_PROGRESS_LINEARIZATION_HPP
#define DRO_MPC_PATH_PROGRESS_LINEARIZATION_HPP

#include "reference_path.hpp"
#include <algorithm>
#include <limits>

namespace dro_mpc {

struct PathProgressLinearization {
    Eigen::RowVector2d position = Eigen::RowVector2d::Zero();
    double previous_progress = 0.0;
};

// Derivative of the active segment in ReferencePath::find_closest_point(p, min_s).
// Preserve its segment order, clipping and strict nearest-distance tie rule.
// At a clipping kink, select the interior segment derivative. At a nearest-
// segment switch no unique derivative exists; SQP uses the selected branch.
inline PathProgressLinearization linearize_path_progress(
    const ReferencePath& path, const Eigen::Vector2d& position, double previous_s) {
    PathProgressLinearization result;
    const auto& points = path.points();
    if (points.size() < 2) return result;
    const double min_s = std::clamp(previous_s, 0.0, path.total_length());
    const double min_s_derivative =
        previous_s >= 0.0 && previous_s <= path.total_length() ? 1.0 : 0.0;
    double minimum_distance = std::numeric_limits<double>::max();
    for (std::size_t i = 1; i < points.size(); ++i) {
        const auto& p0 = points[i - 1];
        const auto& p1 = points[i];
        const double ds = p1.s - p0.s;
        if (ds <= 0.0 || p1.s < min_s) continue;
        const double t_min = std::clamp((min_s - p0.s) / ds, 0.0, 1.0);
        const Eigen::Vector2d segment = p1.position - p0.position;
        const double norm_sq = segment.squaredNorm();
        const double raw_t = norm_sq > 1e-18
            ? segment.dot(position - p0.position) / norm_sq : t_min;
        const double t = std::clamp(raw_t, t_min, 1.0);
        const double distance = (position - (p0.position + t * segment)).squaredNorm();
        if (distance < minimum_distance) {
            minimum_distance = distance;
            result = PathProgressLinearization{};
            if (norm_sq > 1e-18 && raw_t >= t_min && raw_t <= 1.0) {
                result.position = ds * segment.transpose() / norm_sq;
            } else if (raw_t <= t_min && min_s >= p0.s && min_s <= p1.s) {
                result.previous_progress = min_s_derivative;
            }
        }
    }
    return result;
}

}  // namespace dro_mpc
#endif
