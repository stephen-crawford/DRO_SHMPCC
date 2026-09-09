#include "experiment_config_yaml.hpp"
#include "experiment_harness.hpp"
#include "reference_path.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

double heading_delta(double a, double b) {
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

double nearest_sample_distance(
    const ReferencePath& path, const Eigen::Vector2d& waypoint
) {
    double distance = std::numeric_limits<double>::infinity();
    for (const auto& point : path.points()) {
        distance = std::min(distance, (point.position - waypoint).norm());
    }
    return distance;
}

}  // namespace

int main() {
    const std::vector<Eigen::Vector2d> waypoints = {
        {0.0, 0.0}, {8.0, 0.0}, {12.0, 4.0}, {20.0, 4.0}};
    WaypointTrajectoryOptions geometry;
    geometry.sample_spacing = 0.10;
    const ReferencePath path =
        ReferencePath::create_waypoint_trajectory(waypoints, geometry);

    check(path.path_type() == ReferencePath::PathType::WAYPOINT_TRAJECTORY &&
              path.num_points() > waypoints.size() && path.total_length() > 20.0,
          "waypoint input produces a sampled polynomial trajectory");
    check((path.get_position_at(0.0) - waypoints.front()).norm() < 1e-12 &&
              (path.get_position_at(path.total_length()) - waypoints.back()).norm() < 1e-10,
          "polynomial trajectory preserves endpoint waypoints");

    bool has_all_waypoints = true;
    bool strictly_increasing_s = true;
    bool finite_geometry = true;
    double largest_heading_step = 0.0;
    for (const auto& waypoint : waypoints) {
        has_all_waypoints = has_all_waypoints &&
            nearest_sample_distance(path, waypoint) < 1e-10;
    }
    for (size_t i = 0; i < path.points().size(); ++i) {
        const PathPoint& point = path.points()[i];
        finite_geometry = finite_geometry &&
            std::isfinite(point.heading) && std::isfinite(point.curvature) &&
            std::isfinite(point.s);
        if (i > 0) {
            strictly_increasing_s = strictly_increasing_s &&
                point.s > path.points()[i - 1].s;
            largest_heading_step = std::max(largest_heading_step,
                std::abs(heading_delta(point.heading, path.points()[i - 1].heading)));
        }
    }
    check(has_all_waypoints, "trajectory interpolates every supplied waypoint");
    check(strictly_increasing_s && finite_geometry,
          "trajectory has finite geometry and a strictly increasing arc length");
    check(largest_heading_step < 0.15,
          "C2 trajectory removes polyline heading discontinuities");

    WaypointTrajectoryOptions sparse_controls;
    sparse_controls.sample_spacing = 100.0;
    WaypointTrajectoryOptions dense_controls = sparse_controls;
    dense_controls.control_point_spacing = 2.0;
    const std::vector<Eigen::Vector2d> control_test_waypoints = {
        {0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}};
    const ReferencePath sparse_control_path =
        ReferencePath::create_waypoint_trajectory(
            control_test_waypoints, sparse_controls);
    const ReferencePath dense_control_path =
        ReferencePath::create_waypoint_trajectory(
            control_test_waypoints, dense_controls);
    check(dense_control_path.num_points() > sparse_control_path.num_points(),
          "control-point spacing changes waypoint trajectory resolution");

    const ReferencePath repeated =
        ReferencePath::create_waypoint_trajectory(waypoints, geometry);
    bool deterministic = repeated.num_points() == path.num_points() &&
        std::abs(repeated.total_length() - path.total_length()) < 1e-12;
    for (size_t i = 0; deterministic && i < path.points().size(); ++i) {
        deterministic = (repeated.points()[i].position - path.points()[i].position).norm() < 1e-12 &&
            std::abs(repeated.points()[i].heading - path.points()[i].heading) < 1e-12 &&
            std::abs(repeated.points()[i].curvature - path.points()[i].curvature) < 1e-12;
    }
    check(deterministic, "waypoint trajectory construction is deterministic");

    TrajectoryTimingOptions timing;
    timing.nominal_speed = 5.0;
    timing.initial_speed = 1.0;
    timing.max_acceleration = 1.2;
    timing.max_deceleration = 1.8;
    timing.max_lateral_acceleration = 1.0;
    timing.max_angular_velocity = 0.5;
    const ReferencePath timed = path.with_time_parameterization(timing);
    bool finite_timing = timed.has_time_parameterization() &&
        timed.total_duration() > 0.0;
    bool obeys_curvature_limits = true;
    bool obeys_longitudinal_limits = true;
    for (size_t i = 0; i < timed.points().size(); ++i) {
        const PathPoint& point = timed.points()[i];
        finite_timing = finite_timing && std::isfinite(point.time) &&
            std::isfinite(point.speed) && point.speed > 0.0 &&
            point.speed <= timing.nominal_speed + 1e-9;
        const double curvature = std::abs(point.curvature);
        obeys_curvature_limits = obeys_curvature_limits &&
            point.speed * point.speed * curvature <=
                timing.max_lateral_acceleration + 1e-6 &&
            point.speed * curvature <= timing.max_angular_velocity + 1e-6;
        if (i > 0) {
            const PathPoint& previous = timed.points()[i - 1];
            finite_timing = finite_timing && point.time > previous.time;
            const double ds = point.s - previous.s;
            const double acceleration =
                (point.speed * point.speed - previous.speed * previous.speed) /
                (2.0 * ds);
            obeys_longitudinal_limits = obeys_longitudinal_limits &&
                acceleration <= timing.max_acceleration + 1e-6 &&
                acceleration >= -timing.max_deceleration - 1e-6;
        }
    }
    check(finite_timing &&
              std::abs(timed.get_time_at(timed.total_length()) - timed.total_duration()) < 1e-9,
          "time parameterization is finite and monotone");
    check(obeys_curvature_limits && obeys_longitudinal_limits,
          "time parameterization respects lateral, angular, and longitudinal limits");

    WaypointTrajectoryOptions closed_geometry;
    closed_geometry.sample_spacing = 0.10;
    closed_geometry.closed_loop = true;
    const ReferencePath closed = ReferencePath::create_waypoint_trajectory(
        {{0.0, 0.0}, {10.0, 0.0}, {10.0, 10.0}, {0.0, 10.0}}, closed_geometry);
    const PathPoint closed_start = closed.get_point_at(0.0);
    const PathPoint closed_end = closed.get_point_at(closed.total_length());
    check((closed_start.position - closed_end.position).norm() < 1e-10 &&
              std::abs(heading_delta(closed_start.heading, closed_end.heading)) < 1e-9 &&
              std::abs(closed_start.curvature - closed_end.curvature) < 1e-9,
          "closed waypoint trajectories have periodic position, heading, and curvature");
    const Eigen::Vector2d closed_goal = rollout_tracking_goal(
        closed, 0.0, 2.0, 10, 0.1);
    check(closed.is_closed_loop() &&
              std::abs(closed.wrap_arc_length(closed.total_length() + 1.0) - 1.0) < 1e-10 &&
              (closed_goal - closed_start.position).norm() > 0.1,
          "closed-loop tracking uses a wrapped lookahead rather than the start point");
    check((rollout_tracking_goal(path, 0.0, 2.0, 10, 0.1) -
               path.get_position_at(path.total_length())).norm() < 1e-12,
          "open-path tracking retains its terminal goal");

    const char* overlay_path = "/tmp/dro_shmpcc_waypoint_trajectory.yaml";
    {
        std::ofstream overlay(overlay_path);
        overlay << "environment: enter_ramp\n"
                << "path_waypoints: [0,0; 8,0; 12,3; 20,3]\n"
                << "path_closed_loop: false\n"
                << "path_sample_spacing: 0.20\n"
                << "path_control_point_spacing: 2.5\n"
                << "path_max_lateral_acceleration: 1.7\n";
    }
    const ExperimentConfig yaml_cfg =
        yaml_config::load_experiment_config(overlay_path, true);
    std::remove(overlay_path);
    const ReferencePath yaml_path = build_environment_reference_path(
        yaml_cfg.environment.type, yaml_cfg.environment);
    check(yaml_cfg.environment.path_waypoints.size() == 4 &&
              yaml_cfg.environment.path_sample_spacing == 0.20 &&
              yaml_cfg.environment.path_control_point_spacing == 2.5 &&
              yaml_cfg.environment.path_max_lateral_acceleration == 1.7,
          "trajectory YAML settings are inherited into environment configuration");
    check((yaml_path.get_position_at(0.0) - Eigen::Vector2d(0.0, 0.0)).norm() < 1e-12 &&
              (yaml_path.get_position_at(yaml_path.total_length()) -
                  Eigen::Vector2d(20.0, 3.0)).norm() < 1e-10 &&
              nearest_sample_distance(yaml_path, Eigen::Vector2d(12.0, 3.0)) < 1e-10,
          "YAML path_waypoints override generated environment route geometry");

    EnvironmentExperimentConfig precedence = yaml_cfg.environment;
    precedence.custom_ref_path = ReferencePath::create_straight(
        Eigen::Vector2d(-1.0, 2.0), Eigen::Vector2d(3.0, 2.0));
    const ReferencePath explicit_path = build_environment_reference_path(
        precedence.type, precedence);
    check(explicit_path.path_type() == ReferencePath::PathType::STRAIGHT &&
              (explicit_path.get_position_at(0.0) - Eigen::Vector2d(-1.0, 2.0)).norm() < 1e-12,
          "explicit C++ reference path retains precedence over YAML waypoints");

    return failures == 0 ? 0 : 1;
}
