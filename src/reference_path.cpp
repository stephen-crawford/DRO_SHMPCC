/**
 * @file reference_path.cpp
 * @brief Implementation of reference path for contouring MPC.
 */

#include "reference_path.hpp"
#include <algorithm>
#include <Eigen/LU>
#include <limits>

namespace dro_mpc {

namespace {

constexpr double kGeometryEpsilon = 1e-9;

double cross_2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a.x() * b.y() - a.y() * b.x();
}

struct CubicSegment {
    Eigen::Vector2d a;
    Eigen::Vector2d b;
    Eigen::Vector2d c;
    Eigen::Vector2d d;
    double parameter_length = 0.0;

    Eigen::Vector2d position(double u) const {
        return ((d * u + c) * u + b) * u + a;
    }

    Eigen::Vector2d first_derivative(double u) const {
        return (3.0 * d * u + 2.0 * c) * u + b;
    }

    Eigen::Vector2d second_derivative(double u) const {
        return 6.0 * d * u + 2.0 * c;
    }
};

struct GeometrySample {
    Eigen::Vector2d position;
    Eigen::Vector2d first_derivative;
    Eigen::Vector2d second_derivative;
};

std::vector<Eigen::Vector2d> unique_waypoints(
    const std::vector<Eigen::Vector2d>& waypoints, bool closed_loop
) {
    std::vector<Eigen::Vector2d> result;
    result.reserve(waypoints.size());
    for (const auto& waypoint : waypoints) {
        if (result.empty() ||
            (waypoint - result.back()).norm() > kGeometryEpsilon) {
            result.push_back(waypoint);
        }
    }
    if (closed_loop && result.size() > 1 &&
        (result.front() - result.back()).norm() <= kGeometryEpsilon) {
        result.pop_back();
    }
    return result;
}

std::vector<Eigen::Vector2d> resample_control_points(
    const std::vector<Eigen::Vector2d>& knots,
    double control_point_spacing,
    bool closed_loop
) {
    if (!(control_point_spacing > kGeometryEpsilon) || knots.size() < 2) {
        return knots;
    }

    const size_t segment_count = closed_loop ? knots.size() : knots.size() - 1;
    std::vector<Eigen::Vector2d> controls;
    controls.reserve(knots.size());
    for (size_t i = 0; i < segment_count; ++i) {
        const Eigen::Vector2d& start = knots[i];
        const Eigen::Vector2d& end = knots[(i + 1) % knots.size()];
        if (controls.empty()) controls.push_back(start);

        const double length = (end - start).norm();
        const int pieces = std::max(
            1, static_cast<int>(std::ceil(length / control_point_spacing)));
        for (int piece = 1; piece <= pieces; ++piece) {
            // The closed path stores its seam implicitly; do not duplicate the
            // first control as the final control.
            if (closed_loop && i + 1 == segment_count && piece == pieces) {
                continue;
            }
            controls.push_back(start + (end - start) *
                (static_cast<double>(piece) / static_cast<double>(pieces)));
        }
    }
    return controls;
}

std::vector<CubicSegment> make_cubic_segments(
    const std::vector<Eigen::Vector2d>& knots, bool closed_loop
) {
    const int knot_count = static_cast<int>(knots.size());
    const int segment_count = closed_loop ? knot_count : knot_count - 1;
    if (knot_count < 2 || (closed_loop && knot_count < 3)) return {};

    std::vector<double> h(segment_count);
    for (int i = 0; i < segment_count; ++i) {
        const int next = (i + 1) % knot_count;
        h[i] = (knots[next] - knots[i]).norm();
        if (!(h[i] > kGeometryEpsilon)) return {};
    }

    // Solve the natural (open) or periodic (closed) cubic-spline equations.
    // This is a minimum-bending-energy interpolation in the chord-length
    // parameter, giving C2 position/heading/curvature continuity.
    Eigen::MatrixXd system = Eigen::MatrixXd::Zero(knot_count, knot_count);
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(knot_count, 2);
    if (closed_loop) {
        for (int i = 0; i < knot_count; ++i) {
            const int prev = (i + knot_count - 1) % knot_count;
            const int next = (i + 1) % knot_count;
            const double h_prev = h[prev];
            const double h_next = h[i];
            system(i, prev) += h_prev;
            system(i, i) += 2.0 * (h_prev + h_next);
            system(i, next) += h_next;
            const Eigen::Vector2d value = 3.0 * (
                (knots[next] - knots[i]) / h_next -
                (knots[i] - knots[prev]) / h_prev);
            rhs.row(i) = value.transpose();
        }
    } else {
        system(0, 0) = 1.0;
        system(knot_count - 1, knot_count - 1) = 1.0;
        for (int i = 1; i < knot_count - 1; ++i) {
            const double h_prev = h[i - 1];
            const double h_next = h[i];
            system(i, i - 1) = h_prev;
            system(i, i) = 2.0 * (h_prev + h_next);
            system(i, i + 1) = h_next;
            const Eigen::Vector2d value = 3.0 * (
                (knots[i + 1] - knots[i]) / h_next -
                (knots[i] - knots[i - 1]) / h_prev);
            rhs.row(i) = value.transpose();
        }
    }

    const Eigen::MatrixXd second_coefficients = system.fullPivLu().solve(rhs);
    if (!second_coefficients.allFinite()) return {};

    std::vector<Eigen::Vector2d> c(knot_count);
    for (int i = 0; i < knot_count; ++i) {
        c[i] = second_coefficients.row(i).transpose();
    }

    std::vector<CubicSegment> segments;
    segments.reserve(segment_count);
    for (int i = 0; i < segment_count; ++i) {
        const int next = (i + 1) % knot_count;
        CubicSegment segment;
        segment.a = knots[i];
        segment.c = c[i];
        segment.parameter_length = h[i];
        segment.b = (knots[next] - knots[i]) / h[i] -
            h[i] * (c[next] + 2.0 * c[i]) / 3.0;
        segment.d = (c[next] - c[i]) / (3.0 * h[i]);
        segments.push_back(segment);
    }
    return segments;
}

}  // namespace

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

    path.points_.reserve(num_points);
    if (num_points == 1) {
        path.points_.emplace_back(start, heading, 0.0, 0.0);
        return path;
    }

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

    if (num_points == 1) {
        path.points_.emplace_back(Eigen::Vector2d(x_vals[0], y_vals[0]), 0.0, 0.0, s_vals[0]);
        return path;
    }
    
    // Compute headings and curvatures
    for (int i = 0; i < num_points; ++i) {
        double x = x_vals[i];

        // dy/dx = A * (2*pi/L) * cos(2*pi*x/L)
        double dydx = amplitude * (2 * M_PI / length) * std::cos(2 * M_PI * x / length);
        double heading = std::atan2(dydx, 1.0);

        // d2y/dx2 = -A * (2*pi/L)^2 * sin(2*pi*x/L)
        double d2ydx2 = -amplitude * std::pow(2 * M_PI / length, 2) * std::sin(2 * M_PI * x / length);

        // Curvature: k = d2y/dx2 / (1 + (dy/dx)^2)^(3/2)
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
    const double turns = std::abs(angle_span) / (2.0 * M_PI);
    path.closed_loop_ = turns >= 1.0 &&
        std::abs(turns - std::round(turns)) <= 1e-9;
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
        heading = std::atan2(std::sin(heading), std::cos(heading));

        path.points_.emplace_back(pos, heading, curvature, s);
    }

    return path;
}

ReferencePath ReferencePath::create_polyline(
    const std::vector<Eigen::Vector2d>& waypoints
) {
    ReferencePath path;
    path.path_type_ = PathType::CUSTOM;
    if (waypoints.empty()) return path;
    if (waypoints.size() == 1) {
        path.points_.emplace_back(waypoints.front(), 0.0, 0.0, 0.0);
        return path;
    }

    path.points_.reserve(waypoints.size());
    double arc_length = 0.0;
    for (size_t i = 0; i < waypoints.size(); ++i) {
        Eigen::Vector2d tangent;
        if (i == 0) tangent = waypoints[1] - waypoints[0];
        else if (i + 1 == waypoints.size()) tangent = waypoints[i] - waypoints[i - 1];
        else tangent = waypoints[i + 1] - waypoints[i - 1];
        const double heading = std::atan2(tangent.y(), tangent.x());
        if (i > 0) arc_length += (waypoints[i] - waypoints[i - 1]).norm();
        path.points_.emplace_back(waypoints[i], heading, 0.0, arc_length);
    }
    path.total_length_ = arc_length;
    return path;
}

ReferencePath ReferencePath::create_waypoint_trajectory(
    const std::vector<Eigen::Vector2d>& waypoints,
    const WaypointTrajectoryOptions& options
) {
    const auto source_knots = unique_waypoints(waypoints, options.closed_loop);
    const auto knots = resample_control_points(
        source_knots, options.control_point_spacing, options.closed_loop);
    if (knots.empty()) return ReferencePath();
    if (knots.size() == 1 || (options.closed_loop && knots.size() < 3)) {
        ReferencePath path = create_polyline(knots);
        path.path_type_ = PathType::WAYPOINT_TRAJECTORY;
        return path;
    }

    const auto segments = make_cubic_segments(knots, options.closed_loop);
    if (segments.empty()) {
        ReferencePath path = create_polyline(knots);
        path.path_type_ = PathType::WAYPOINT_TRAJECTORY;
        return path;
    }

    const double sample_spacing = std::max(1e-3, options.sample_spacing);
    std::vector<GeometrySample> samples;
    for (size_t i = 0; i < segments.size(); ++i) {
        const CubicSegment& segment = segments[i];
        const int steps = std::max(
            1, static_cast<int>(std::ceil(segment.parameter_length / sample_spacing)));
        for (int step = 0; step <= steps; ++step) {
            if (!samples.empty() && step == 0) continue;
            const double u = segment.parameter_length *
                static_cast<double>(step) / static_cast<double>(steps);
            samples.push_back({
                segment.position(u), segment.first_derivative(u),
                segment.second_derivative(u)});
        }
    }

    ReferencePath path;
    path.path_type_ = PathType::WAYPOINT_TRAJECTORY;
    path.closed_loop_ = options.closed_loop;
    path.points_.reserve(samples.size());
    double arc_length = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i > 0) {
            arc_length += (samples[i].position - samples[i - 1].position).norm();
        }

        Eigen::Vector2d tangent = samples[i].first_derivative;
        if (tangent.squaredNorm() <= kGeometryEpsilon * kGeometryEpsilon) {
            if (i + 1 < samples.size()) {
                tangent = samples[i + 1].position - samples[i].position;
            } else if (i > 0) {
                tangent = samples[i].position - samples[i - 1].position;
            }
        }

        const double tangent_norm = tangent.norm();
        const double derivative_norm = samples[i].first_derivative.norm();
        const double heading = tangent_norm > kGeometryEpsilon
            ? std::atan2(tangent.y(), tangent.x()) : 0.0;
        const double curvature = derivative_norm > kGeometryEpsilon
            ? cross_2d(samples[i].first_derivative,
                       samples[i].second_derivative) /
                std::pow(derivative_norm, 3)
            : 0.0;
        path.points_.emplace_back(
            samples[i].position, heading,
            std::isfinite(curvature) ? curvature : 0.0, arc_length);
    }
    path.total_length_ = arc_length;
    return path;
}

ReferencePath ReferencePath::with_time_parameterization(
    const TrajectoryTimingOptions& options
) const {
    ReferencePath result = *this;
    result.has_timing_ = false;
    result.total_duration_ = 0.0;
    for (auto& point : result.points_) {
        point.time = 0.0;
        point.speed = 0.0;
    }

    if (result.points_.size() < 2 ||
        !(options.nominal_speed > kGeometryEpsilon)) {
        return result;
    }

    const size_t point_count = result.points_.size();
    std::vector<double> speeds(point_count, options.nominal_speed);
    const double max_lateral_acceleration =
        std::max(0.0, options.max_lateral_acceleration);
    const double max_angular_velocity =
        std::max(0.0, options.max_angular_velocity);
    for (size_t i = 0; i < point_count; ++i) {
        const double curvature = std::abs(result.points_[i].curvature);
        if (curvature <= kGeometryEpsilon) continue;
        if (max_lateral_acceleration > 0.0) {
            speeds[i] = std::min(
                speeds[i], std::sqrt(max_lateral_acceleration / curvature));
        }
        if (max_angular_velocity > 0.0) {
            speeds[i] = std::min(
                speeds[i], max_angular_velocity / curvature);
        }
    }

    const bool closed_loop = result.closed_loop_;
    if (!closed_loop && options.initial_speed >= 0.0) {
        speeds.front() = std::min(speeds.front(), options.initial_speed);
    }

    const double max_acceleration = std::max(0.0, options.max_acceleration);
    const double max_deceleration = std::max(0.0, options.max_deceleration);
    const auto constrain_forward = [&] {
        if (!(max_acceleration > 0.0)) return;
        for (size_t i = 1; i < point_count; ++i) {
            const double ds = std::max(0.0,
                result.points_[i].s - result.points_[i - 1].s);
            const double reachable = std::sqrt(std::max(
                0.0, speeds[i - 1] * speeds[i - 1] + 2.0 * max_acceleration * ds));
            speeds[i] = std::min(speeds[i], reachable);
        }
    };
    const auto constrain_backward = [&] {
        if (!(max_deceleration > 0.0)) return;
        for (size_t i = point_count - 1; i > 0; --i) {
            const double ds = std::max(0.0,
                result.points_[i].s - result.points_[i - 1].s);
            const double reachable = std::sqrt(std::max(
                0.0, speeds[i] * speeds[i] + 2.0 * max_deceleration * ds));
            speeds[i - 1] = std::min(speeds[i - 1], reachable);
        }
    };

    constrain_forward();
    constrain_backward();
    if (closed_loop) {
        // The final sample duplicates the first point.  Repeating the two
        // passes after tying their speeds removes a discontinuity at the seam.
        for (int pass = 0; pass < 2; ++pass) {
            const double seam_speed = std::min(speeds.front(), speeds.back());
            speeds.front() = seam_speed;
            speeds.back() = seam_speed;
            constrain_forward();
            constrain_backward();
        }
        const double seam_speed = std::min(speeds.front(), speeds.back());
        speeds.front() = seam_speed;
        speeds.back() = seam_speed;
    }

    double time = 0.0;
    result.points_.front().speed = speeds.front();
    result.points_.front().time = time;
    for (size_t i = 1; i < point_count; ++i) {
        const double ds = std::max(0.0,
            result.points_[i].s - result.points_[i - 1].s);
        const double average_speed = std::max(
            kGeometryEpsilon, 0.5 * (speeds[i - 1] + speeds[i]));
        time += ds / average_speed;
        result.points_[i].speed = speeds[i];
        result.points_[i].time = time;
    }
    result.total_duration_ = time;
    result.has_timing_ = true;
    return result;
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

    const double segment_length = p2.s - p1.s;
    if (segment_length <= kGeometryEpsilon) {
        return p2;
    }
    const double t = std::clamp((s - p1.s) / segment_length, 0.0, 1.0);
    return interpolate(p1, p2, t);
}

Eigen::Vector2d ReferencePath::get_position_at(double s) const {
    return get_point_at(s).position;
}

double ReferencePath::get_heading_at(double s) const {
    return get_point_at(s).heading;
}

double ReferencePath::get_time_at(double s) const {
    return get_point_at(s).time;
}

double ReferencePath::get_speed_at(double s) const {
    return get_point_at(s).speed;
}

double ReferencePath::find_closest_point(const Eigen::Vector2d& position) const {
    return find_closest_point_projection(position, 0.0);
}

double ReferencePath::find_closest_point(const Eigen::Vector2d& position, double min_s) const {
    return find_closest_point_projection(position, min_s);
}

double ReferencePath::find_closest_point_projection(
    const Eigen::Vector2d& position, double min_s) const {
    if (points_.empty()) {
        return std::max(0.0, min_s);
    }

    min_s = std::clamp(min_s, 0.0, total_length_);
    double min_dist = std::numeric_limits<double>::max();
    double best_s = min_s;

    if (points_.size() == 1) {
        return points_.front().s >= min_s ? points_.front().s : min_s;
    }

    for (size_t i = 1; i < points_.size(); ++i) {
        const PathPoint& p0 = points_[i - 1];
        const PathPoint& p1 = points_[i];
        const double ds = p1.s - p0.s;
        if (ds <= 0.0 || p1.s < min_s) {
            continue;
        }

        const double t_min = std::clamp((min_s - p0.s) / ds, 0.0, 1.0);
        const Eigen::Vector2d segment = p1.position - p0.position;
        const double segment_norm_sq = segment.squaredNorm();
        double t = t_min;
        if (segment_norm_sq > 1e-18) {
            t = std::clamp(
                segment.dot(position - p0.position) / segment_norm_sq,
                t_min, 1.0);
        }

        const Eigen::Vector2d foot = p0.position + t * segment;
        const double distance_sq = (position - foot).squaredNorm();
        if (distance_sq < min_dist) {
            min_dist = distance_sq;
            best_s = p0.s + t * ds;
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

double ReferencePath::wrap_arc_length(double s) const {
    if (!(total_length_ > kGeometryEpsilon)) return 0.0;
    if (!closed_loop_) return std::clamp(s, 0.0, total_length_);
    double wrapped = std::fmod(s, total_length_);
    if (wrapped < 0.0) wrapped += total_length_;
    return wrapped;
}

PathPoint ReferencePath::interpolate(const PathPoint& p1, const PathPoint& p2, double t) const {
    PathPoint result;
    result.position = (1 - t) * p1.position + t * p2.position;
    result.s = (1 - t) * p1.s + t * p2.s;
    result.curvature = (1 - t) * p1.curvature + t * p2.curvature;
    result.time = (1 - t) * p1.time + t * p2.time;
    result.speed = (1 - t) * p1.speed + t * p2.speed;

    // Interpolate heading carefully (handle wrap-around)
    double h1 = p1.heading;
    double h2 = p2.heading;
    double diff = h2 - h1;
    while (diff > M_PI) diff -= 2 * M_PI;
    while (diff < -M_PI) diff += 2 * M_PI;
    const double interpolated_heading = h1 + t * diff;
    result.heading = std::atan2(
        std::sin(interpolated_heading), std::cos(interpolated_heading));

    return result;
}

}  // namespace dro_mpc
