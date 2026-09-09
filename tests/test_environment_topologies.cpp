/**
 * @file test_environment_topologies.cpp
 * @brief Focused geometric contract for every built-in environment route.
 *
 * Runtime route visualization belongs to the harness artifact bundle. This
 * test deliberately verifies geometry without writing cwd-relative files.
 */

#include "experiment_harness.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace dro_mpc;

namespace {

struct Case {
    EnvironmentType type;
    const char* name;
    bool expects_lateral_variation;
    bool expects_closed_route;
};

int failures = 0;

void check(bool condition, const std::string& message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

}  // namespace

int main() {
    EnvironmentExperimentConfig config;
    config.road_length = 90.0;
    config.lane_width = 3.6;
    config.intersection_box_size = 24.0;
    config.corner_radius = 9.0;
    config.ramp_length = 55.0;
    config.merge_length = 35.0;
    config.roundabout_radius = 20.0;
    config.s_curve_length = 50.0;
    config.s_curve_amplitude = 5.0;

    const std::vector<Case> cases = {
        {EnvironmentType::T_INTERSECTION, "T intersection", true, false},
        {EnvironmentType::FOUR_WAY_INTERSECTION, "Four-way intersection", true, false},
        {EnvironmentType::S_CURVE, "S curve", true, false},
        {EnvironmentType::TWO_LANE_ROUNDABOUT, "Two-lane roundabout", true, true},
        {EnvironmentType::FOUR_LANE_ROUNDABOUT, "Four-lane roundabout", true, true},
        {EnvironmentType::TWO_LANE_HIGHWAY, "Two-lane highway", false, false},
        {EnvironmentType::FOUR_LANE_HIGHWAY, "Four-lane highway", false, false},
        {EnvironmentType::ENTER_RAMP, "Enter ramp", true, false},
        {EnvironmentType::EXIT_RAMP, "Exit ramp", true, false},
        {EnvironmentType::OVERTAKE_SLOW_LEAD, "Overtake", false, false},
        {EnvironmentType::NARROW_CORRIDOR, "Narrow corridor", false, false},
        {EnvironmentType::ONCOMING, "Oncoming", false, false},
        {EnvironmentType::INTERSECTION, "Intersection", true, false},
    };

    for (const Case& test_case : cases) {
        const ReferencePath path = build_environment_reference_path(test_case.type, config);
        const std::vector<ReferencePath> roads =
            build_environment_road_centerlines(test_case.type, config);
        const auto& points = path.points();
        double min_y = points.front().position.y();
        double max_y = min_y;
        for (const auto& point : points) {
            min_y = std::min(min_y, point.position.y());
            max_y = std::max(max_y, point.position.y());
        }
        check(path.num_points() >= 2 && path.total_length() > 1.0,
              std::string(test_case.name) + " has a usable route");
        check(path.path_type() == ReferencePath::PathType::WAYPOINT_TRAJECTORY,
              std::string(test_case.name) + " uses canonical waypoint construction");
        check(test_case.expects_lateral_variation ? (max_y - min_y > 0.5) :
                                                   (max_y - min_y < 1e-8),
              std::string(test_case.name) + " has the expected lateral shape");
        const double endpoint_gap =
            (points.front().position - points.back().position).norm();
        check(test_case.expects_closed_route ? endpoint_gap < 1e-8 : endpoint_gap >= 1e-8,
              std::string(test_case.name) + (test_case.expects_closed_route
                  ? " closes at its starting point" : " remains an open route"));
        const bool is_intersection = test_case.type == EnvironmentType::T_INTERSECTION ||
            test_case.type == EnvironmentType::FOUR_WAY_INTERSECTION ||
            test_case.type == EnvironmentType::INTERSECTION;
        check(!is_intersection || roads.size() >= 2,
              std::string(test_case.name) + " includes perpendicular road centerlines");
    }

    return failures == 0 ? 0 : 1;
}
