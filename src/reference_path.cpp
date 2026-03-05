/**
 * @file reference_path.cpp
 * @brief Implementation of reference path for contouring MPC.
 */

#include "reference_path.hpp"
#include <algorithm>
#include <limits>

namespace scenario_mpc {

ReferencePath ReferencePath::create_straight(
    const Eigen::Vector2d& start,
    const Eigen::Vector2d& end,
    int num_points
) {
    ReferencePath path;
    path.path_type_ = PathType::STRAIGHT;

    Eigen::Vector2d direction = end - start;
    double length = direction.norm();
    path.total_length_ = length;

    if (length < 1e-6) {
        // Degenerate case
        path.points_.emplace_back(start, 0, 0, 0);
        return path;
    }

    double heading = std::atan2(direction.y(), direction.x());
    direction /= length;

    path.points_.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
        double t = static_cast<double>(i) / (num_points - 1);
        double s = t * length;
        Eigen::Vector2d pos = start + t * (end - start);
        path.points_.emplace_back(pos, heading, 0.0, s);
    }

    return path;
}

ReferencePath ReferencePath::create_s_curve(
    double length,
    double amplitude,
    int num_points
) {
    ReferencePath path;
    path.path_type_ = PathType::S_CURVE;

    path.points_.reserve(num_points);

    // S-curve: y = A * sin(2*pi*x/L)
    // We parameterize by x and compute arc length numerically

    std::vector<double> x_vals(num_points);
    std::vector<double> y_vals(num_points);
    std::vector<double> s_vals(num_points);

    double dx = length / (num_points - 1);

    // Compute positions
    for (int i = 0; i < num_points; ++i) {
        double x = i * dx;
        double y = amplitude * std::sin(2 * M_PI * x / length);
        x_vals[i] = x;
        y_vals[i] = y;
    }

    // Compute arc length
    s_vals[0] = 0;
    for (int i = 1; i < num_points; ++i) {
        double dx_seg = x_vals[i] - x_vals[i-1];
        double dy_seg = y_vals[i] - y_vals[i-1];
        s_vals[i] = s_vals[i-1] + std::sqrt(dx_seg*dx_seg + dy_seg*dy_seg);
    }

    path.total_length_ = s_vals.back();

    // Compute headings and curvatures
    for (int i = 0; i < num_points; ++i) {
        double x = x_vals[i];

        // dy/dx = A * (2*pi/L) * cos(2*pi*x/L)
        double dydx = amplitude * (2 * M_PI / length) * std::cos(2 * M_PI * x / length);
        double heading = std::atan2(dydx, 1.0);

        // d2y/dx2 = -A * (2*pi/L)^2 * sin(2*pi*x/L)
        double d2ydx2 = -amplitude * std::pow(2 * M_PI / length, 2) * std::sin(2 * M_PI * x / length);

        // Curvature: k = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
        double curvature = d2ydx2 / std::pow(1 + dydx*dydx, 1.5);

        Eigen::Vector2d pos(x_vals[i], y_vals[i]);
        path.points_.emplace_back(pos, heading, curvature, s_vals[i]);
    }

    return path;
}

ReferencePath ReferencePath::create_circle(
    const Eigen::Vector2d& center,
    double radius,
    double start_angle,
    double end_angle,
    int num_points
) {
    ReferencePath path;
    path.path_type_ = PathType::CIRCLE;

    double angle_span = end_angle - start_angle;
    path.total_length_ = std::abs(radius * angle_span);
    double curvature = 1.0 / radius;

    path.points_.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
        double t = static_cast<double>(i) / (num_points - 1);
        double angle = start_angle + t * angle_span;
        double s = std::abs(radius * t * angle_span);

        Eigen::Vector2d pos = center + radius * Eigen::Vector2d(std::cos(angle), std::sin(angle));
        double heading = angle + M_PI / 2;  // Tangent is perpendicular to radius

        if (angle_span < 0) {
            heading = angle - M_PI / 2;
            curvature = -1.0 / radius;
        }

        path.points_.emplace_back(pos, heading, curvature, s);
    }

    return path;
}

PathPoint ReferencePath::get_point_at(double s) const {
    if (points_.empty()) {
        return PathPoint();
    }

    // Clamp s to valid range
    s = std::max(0.0, std::min(s, total_length_));

    // Binary search for segment
    auto it = std::lower_bound(points_.begin(), points_.end(), s,
        [](const PathPoint& p, double val) { return p.s < val; });

    if (it == points_.begin()) {
        return points_.front();
    }
    if (it == points_.end()) {
        return points_.back();
    }

    // Interpolate
    const PathPoint& p2 = *it;
    const PathPoint& p1 = *(it - 1);

    double t = (s - p1.s) / (p2.s - p1.s + 1e-9);
    return interpolate(p1, p2, t);
}

Eigen::Vector2d ReferencePath::get_position_at(double s) const {
    return get_point_at(s).position;
}

double ReferencePath::get_heading_at(double s) const {
    return get_point_at(s).heading;
}

double ReferencePath::find_closest_point(const Eigen::Vector2d& position) const {
    if (points_.empty()) {
        return 0;
    }

    double min_dist = std::numeric_limits<double>::max();
    double best_s = 0;

    for (const auto& p : points_) {
        double dist = (position - p.position).norm();
        if (dist < min_dist) {
            min_dist = dist;
            best_s = p.s;
        }
    }

    return best_s;
}

double ReferencePath::find_closest_point(const Eigen::Vector2d& position, double min_s) const {
    if (points_.empty()) {
        return min_s;
    }

    double min_dist = std::numeric_limits<double>::max();
    double best_s = min_s;

    for (const auto& p : points_) {
        if (p.s < min_s) {
            continue;
        }
        double dist = (position - p.position).norm();
        if (dist < min_dist) {
            min_dist = dist;
            best_s = p.s;
        }
    }

    return best_s;
}

double ReferencePath::compute_lateral_offset(const Eigen::Vector2d& position, double s) const {
    PathPoint p = get_point_at(s);

    Eigen::Vector2d to_pos = position - p.position;
    Eigen::Vector2d normal(-std::sin(p.heading), std::cos(p.heading));

    return to_pos.dot(normal);
}

Eigen::Vector2d ReferencePath::get_position_at_fraction(double fraction) const {
    return get_position_at(fraction * total_length_);
}

PathPoint ReferencePath::interpolate(const PathPoint& p1, const PathPoint& p2, double t) const {
    PathPoint result;
    result.position = (1 - t) * p1.position + t * p2.position;
    result.s = (1 - t) * p1.s + t * p2.s;
    result.curvature = (1 - t) * p1.curvature + t * p2.curvature;

    // Interpolate heading carefully (handle wrap-around)
    double h1 = p1.heading;
    double h2 = p2.heading;
    double diff = h2 - h1;
    while (diff > M_PI) diff -= 2 * M_PI;
    while (diff < -M_PI) diff += 2 * M_PI;
    result.heading = h1 + t * diff;

    return result;
}

}  // namespace scenario_mpc
