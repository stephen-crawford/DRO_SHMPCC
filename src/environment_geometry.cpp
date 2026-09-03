#include "experiment_harness.hpp"

#include <algorithm>
#include <cmath>

namespace dro_mpc {

ReferencePath build_environment_reference_path(
    EnvironmentType environment,
    const EnvironmentExperimentConfig& config
) {
    if (config.custom_ref_path.has_value()) return config.custom_ref_path.value();

    const double length = std::max(10.0, config.road_length);
    const double half_length = 0.5 * length;
    const double lane = std::max(0.1, config.lane_width);
    const double box = std::max(1.0, config.intersection_box_size);
    const double corner = std::min(0.5 * box, std::max(0.1, config.corner_radius));
    const double ramp = std::max(1.0, config.ramp_length);
    const double merge = std::max(1.0, config.merge_length);
    const double carriageway_offset = config.lane_count > 2 ? -0.5 * lane : 0.0;

    switch (environment) {
        case EnvironmentType::S_CURVE:
            return ReferencePath::create_s_curve(
                config.s_curve_length, config.s_curve_amplitude, config.s_curve_points);
        case EnvironmentType::TWO_LANE_HIGHWAY:
        case EnvironmentType::FOUR_LANE_HIGHWAY:
        case EnvironmentType::OVERTAKE_SLOW_LEAD:
        case EnvironmentType::ONCOMING:
        case EnvironmentType::NARROW_CORRIDOR:
            return ReferencePath::create_straight(
                Eigen::Vector2d(0.0, carriageway_offset),
                Eigen::Vector2d(length, carriageway_offset));
        case EnvironmentType::FOUR_WAY_INTERSECTION:
        case EnvironmentType::INTERSECTION:
            return ReferencePath::create_straight(
                Eigen::Vector2d(-0.5 * lane, -half_length),
                Eigen::Vector2d(-0.5 * lane, half_length));
        case EnvironmentType::T_INTERSECTION:
            return ReferencePath::create_polyline({
                Eigen::Vector2d(0.0, -half_length), Eigen::Vector2d(0.0, -corner),
                Eigen::Vector2d(corner, 0.0), Eigen::Vector2d(half_length, 0.0)});
        case EnvironmentType::ENTER_RAMP:
            return ReferencePath::create_polyline({
                Eigen::Vector2d(0.0, -1.5 * lane), Eigen::Vector2d(0.5 * ramp, -lane),
                Eigen::Vector2d(ramp, 0.0), Eigen::Vector2d(ramp + merge + half_length, 0.0)});
        case EnvironmentType::EXIT_RAMP:
            return ReferencePath::create_polyline({
                Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(merge, 0.0),
                Eigen::Vector2d(merge + ramp, -lane),
                Eigen::Vector2d(merge + ramp + half_length, -1.5 * lane)});
        case EnvironmentType::TWO_LANE_ROUNDABOUT:
        case EnvironmentType::FOUR_LANE_ROUNDABOUT:
            return ReferencePath::create_circle(
                Eigen::Vector2d(0.0, config.roundabout_radius),
                std::max(1.0, config.roundabout_radius), -M_PI_2, 3.0 * M_PI_2);
    }
    return ReferencePath::create_straight(Eigen::Vector2d::Zero(), Eigen::Vector2d(length, 0.0));
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
            roads.push_back(ReferencePath::create_straight(
                Eigen::Vector2d(-half_length, 0.0), Eigen::Vector2d(half_length, 0.0)));
            break;
        case EnvironmentType::T_INTERSECTION:
            roads.push_back(ReferencePath::create_straight(
                Eigen::Vector2d(-half_length, 0.0), Eigen::Vector2d(half_length, 0.0)));
            roads.push_back(ReferencePath::create_straight(
                Eigen::Vector2d(0.0, -half_length), Eigen::Vector2d(0.0, 0.0)));
            break;
        case EnvironmentType::ENTER_RAMP:
        case EnvironmentType::EXIT_RAMP:
            roads.push_back(ReferencePath::create_straight(
                Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(length, 0.0)));
            break;
        default:
            break;
    }
    return roads;
}

}  // namespace dro_mpc
