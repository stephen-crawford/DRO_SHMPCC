#include "experiment_harness.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dro_mpc {
namespace {

WaypointTrajectoryOptions route_options(
    const EnvironmentExperimentConfig& config, bool closed_loop = false
) {
    WaypointTrajectoryOptions options;
    options.sample_spacing = std::max(1e-3, config.path_sample_spacing);
    options.control_point_spacing = std::max(
        0.0, config.path_control_point_spacing);
    options.closed_loop = closed_loop;
    return options;
}

ReferencePath build_route(
    const std::vector<Eigen::Vector2d>& waypoints,
    const EnvironmentExperimentConfig& config,
    bool closed_loop = false
) {
    return ReferencePath::create_waypoint_trajectory(
        waypoints, route_options(config, closed_loop));
}

std::vector<Eigen::Vector2d> make_s_curve_waypoints(
    const EnvironmentExperimentConfig& config
) {
    const double length = std::max(1.0, config.s_curve_length);
    const int count = std::max(4, config.s_curve_points);
    std::vector<Eigen::Vector2d> waypoints;
    waypoints.reserve(count);
    for (int i = 0; i < count; ++i) {
        const double fraction = static_cast<double>(i) /
            static_cast<double>(count - 1);
        const double x = fraction * length;
        const double y = config.s_curve_amplitude *
            std::sin(2.0 * M_PI * fraction);
        waypoints.emplace_back(x, y);
    }
    return waypoints;
}

std::vector<Eigen::Vector2d> make_roundabout_waypoints(
    const EnvironmentExperimentConfig& config
) {
    const double radius = std::max(1.0, config.roundabout_radius);
    const double control_spacing =
        std::max(0.25, config.path_control_point_spacing);
    const int count = std::max(12, static_cast<int>(std::ceil(
        2.0 * M_PI * radius / control_spacing)));
    std::vector<Eigen::Vector2d> waypoints;
    waypoints.reserve(count);
    const Eigen::Vector2d center(0.0, radius);
    for (int i = 0; i < count; ++i) {
        const double angle = -M_PI_2 + 2.0 * M_PI *
            static_cast<double>(i) / static_cast<double>(count);
        waypoints.push_back(center + radius *
            Eigen::Vector2d(std::cos(angle), std::sin(angle)));
    }
    return waypoints;
}

std::vector<Eigen::Vector2d> generated_route_waypoints(
    EnvironmentType environment,
    const EnvironmentExperimentConfig& config
) {
    const double length = std::max(10.0, config.road_length);
    const double half_length = 0.5 * length;
    const double lane = std::max(0.1, config.lane_width);
    const double box = std::max(1.0, config.intersection_box_size);
    const double corner = std::min(0.5 * box,
        std::max(0.1, config.corner_radius));
    const double ramp = std::max(1.0, config.ramp_length);
    const double merge = std::max(1.0, config.merge_length);
    const double carriageway_offset =
        config.lane_count > 2 ? -0.5 * lane : 0.0;

    switch (environment) {
        case EnvironmentType::S_CURVE:
            return make_s_curve_waypoints(config);
        case EnvironmentType::TWO_LANE_HIGHWAY:
        case EnvironmentType::FOUR_LANE_HIGHWAY:
        case EnvironmentType::OVERTAKE_SLOW_LEAD:
        case EnvironmentType::ONCOMING:
        case EnvironmentType::NARROW_CORRIDOR:
            return {{0.0, carriageway_offset}, {length, carriageway_offset}};
        case EnvironmentType::FOUR_WAY_INTERSECTION:
        case EnvironmentType::INTERSECTION:
            return {{-0.5 * lane, -half_length},
                    {-0.5 * lane, half_length}};
        case EnvironmentType::T_INTERSECTION:
            return {{0.0, -half_length}, {0.0, -corner},
                    {corner, 0.0}, {half_length, 0.0}};
        case EnvironmentType::ENTER_RAMP:
            return {{0.0, -1.5 * lane}, {0.5 * ramp, -lane},
                    {ramp, 0.0}, {ramp + merge + half_length, 0.0}};
        case EnvironmentType::EXIT_RAMP:
            return {{0.0, 0.0}, {merge, 0.0},
                    {merge + ramp, -lane},
                    {merge + ramp + half_length, -1.5 * lane}};
        case EnvironmentType::TWO_LANE_ROUNDABOUT:
        case EnvironmentType::FOUR_LANE_ROUNDABOUT:
            return make_roundabout_waypoints(config);
    }
    return {{0.0, 0.0}, {length, 0.0}};
}

bool generated_route_is_closed(EnvironmentType environment) {
    return environment == EnvironmentType::TWO_LANE_ROUNDABOUT ||
        environment == EnvironmentType::FOUR_LANE_ROUNDABOUT;
}

}  // namespace

ReferencePath build_environment_reference_path(
    EnvironmentType environment,
    const EnvironmentExperimentConfig& config
) {
    // Explicit in-memory geometry is intentionally the highest-priority API.
    if (config.custom_ref_path.has_value()) return config.custom_ref_path.value();

    // YAML waypoint routes are the next layer.  This makes the generator's
    // input data explicit and avoids each environment duplicating its own
    // sampling/heading/curvature calculations.
    if (!config.path_waypoints.empty()) {
        return build_route(
            config.path_waypoints, config, config.path_closed_loop);
    }

    return build_route(
        generated_route_waypoints(environment, config), config,
        generated_route_is_closed(environment));
}

std::vector<ReferencePath> build_environment_road_centerlines(
    EnvironmentType environment,
    const EnvironmentExperimentConfig& config
) {
    std::vector<ReferencePath> roads;
    roads.push_back(build_environment_reference_path(environment, config));
    const double length = std::max(10.0, config.road_length);
    const double half_length = 0.5 * length;

    switch (environment) {
        case EnvironmentType::FOUR_WAY_INTERSECTION:
        case EnvironmentType::INTERSECTION:
            roads.push_back(build_route(
                {{-half_length, 0.0}, {half_length, 0.0}}, config));
            break;
        case EnvironmentType::T_INTERSECTION:
            roads.push_back(build_route(
                {{-half_length, 0.0}, {half_length, 0.0}}, config));
            roads.push_back(build_route(
                {{0.0, -half_length}, {0.0, 0.0}}, config));
            break;
        case EnvironmentType::ENTER_RAMP:
        case EnvironmentType::EXIT_RAMP:
            roads.push_back(build_route(
                {{0.0, 0.0}, {length, 0.0}}, config));
            break;
        default:
            break;
    }
    return roads;
}

}  // namespace dro_mpc
