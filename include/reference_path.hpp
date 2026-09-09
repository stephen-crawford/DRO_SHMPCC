/**
 * @file reference_path.hpp
 * @brief Reference path for contouring MPC.
 *
 * Implements geometric reference paths and waypoint-generated polynomial
 * trajectories for contouring MPC.
 */

#ifndef DRO_MPC_REFERENCE_PATH_HPP
#define DRO_MPC_REFERENCE_PATH_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <string>

namespace dro_mpc {

/**
 * @brief Point on reference path with position and tangent.
 */
struct PathPoint {
    Eigen::Vector2d position;    // Position [x, y]
    double heading;              // Tangent angle [rad]
    double curvature;            // Path curvature [1/m]
    double s;                    // Arc length parameter
    double time;                 // Optional trajectory time [s]
    double speed;                // Optional speed reference [m/s]

    PathPoint() : heading(0), curvature(0), s(0), time(0), speed(0) {}
    PathPoint(const Eigen::Vector2d& pos, double h, double k, double arc_length,
              double time_seconds = 0.0, double speed_mps = 0.0)
        : position(pos), heading(h), curvature(k), s(arc_length),
          time(time_seconds), speed(speed_mps) {}
};

/**
 * @brief Geometry settings for a waypoint-to-trajectory conversion.
 *
 * The generated curve is a C2 piecewise-cubic interpolation of the supplied
 * waypoints. `sample_spacing` controls the arc-length lookup resolution used
 * by the existing MPCC interface. A positive `control_point_spacing` inserts
 * collinear controls so no source-polyline segment is longer than that value.
 */
struct WaypointTrajectoryOptions {
    double sample_spacing = 0.25;
    double control_point_spacing = 0.0;
    bool closed_loop = false;
};

/**
 * @brief Dynamic limits used to time-parameterize an already geometric path.
 *
 * Non-positive lateral-acceleration or angular-velocity limits disable the
 * corresponding cap.  This keeps geometry construction independent of the
 * vehicle model while allowing callers to derive timing from that model.
 */
struct TrajectoryTimingOptions {
    double nominal_speed = 1.5;
    double initial_speed = -1.0;
    double max_acceleration = 3.0;
    double max_deceleration = 5.0;
    double max_lateral_acceleration = 0.0;
    double max_angular_velocity = 0.0;
};

/**
 * @brief Reference path for contouring MPC.
 */
class ReferencePath {
public:
    enum class PathType {
        STRAIGHT,
        S_CURVE,
        CIRCLE,
        CUSTOM,
        WAYPOINT_TRAJECTORY
    };

    ReferencePath()
        : total_length_(0), total_duration_(0), path_type_(PathType::STRAIGHT),
          has_timing_(false), closed_loop_(false) {}

    /**
     * @brief Create a straight line path.
     * @param start Start position
     * @param end End position
     * @param num_points Number of discretization points
     */
    static ReferencePath create_straight(
        const Eigen::Vector2d& start,
        const Eigen::Vector2d& end,
        int num_points = 100
    );

    /**
     * @brief Create an S-curve path.
     * @param length Total path length
     * @param amplitude S-curve amplitude
     * @param num_points Number of discretization points
     */
    static ReferencePath create_s_curve(
        double length = 25.0,
        double amplitude = 3.0,
        int num_points = 100
    );

    /**
     * @brief Create a circular arc path.
     * @param center Center of circle
     * @param radius Circle radius
     * @param start_angle Start angle [rad]
     * @param end_angle End angle [rad]
     * @param num_points Number of discretization points
     */
    static ReferencePath create_circle(
        const Eigen::Vector2d& center,
        double radius,
        double start_angle,
        double end_angle,
        int num_points = 100
    );

    /// Create a raw, piecewise-linear path through ordered waypoints. This is
    /// retained for exact legacy tests; use create_waypoint_trajectory() for a
    /// smooth route supplied as waypoints.
    static ReferencePath create_polyline(
        const std::vector<Eigen::Vector2d>& waypoints
    );

    /**
     * @brief Construct a C2 piecewise-polynomial trajectory through waypoints.
     *
     * This is the waypoint -> smooth polynomial -> sampled arc-length pipeline
     * used by generated environments.  When closed_loop is true, the final
     * sampled point equals the first and the tangent/curvature are periodic.
     */
    static ReferencePath create_waypoint_trajectory(
        const std::vector<Eigen::Vector2d>& waypoints,
        const WaypointTrajectoryOptions& options = WaypointTrajectoryOptions()
    );

    /**
     * @brief Return a copy carrying a feasible speed/time parameterization.
     *
     * The path geometry and arc-length coordinates are unchanged.  A
     * forward/backward pass enforces longitudinal acceleration/deceleration
     * caps after curvature and angular-rate speed limits are applied.
     */
    ReferencePath with_time_parameterization(
        const TrajectoryTimingOptions& options
    ) const;

    /**
     * @brief Get point on path at arc length s.
     * @param s Arc length parameter [0, total_length]
     * @return PathPoint at s
     */
    PathPoint get_point_at(double s) const;

    /**
     * @brief Get position at arc length s.
     * @param s Arc length parameter
     * @return Position [x, y]
     */
    Eigen::Vector2d get_position_at(double s) const;

    /**
     * @brief Get heading at arc length s.
     * @param s Arc length parameter
     * @return Heading angle [rad]
     */
    double get_heading_at(double s) const;

    /// Get the time profile value at arc length s (zero for untimed paths).
    double get_time_at(double s) const;

    /// Get the speed profile value at arc length s (zero for untimed paths).
    double get_speed_at(double s) const;

    /**
     * @brief Find closest point on path to given position.
     * @param position Query position
     * @return Arc length of closest point
     */
    double find_closest_point(const Eigen::Vector2d& position) const;

    /**
     * @brief Find closest point on path at or ahead of min_s (monotonic).
     * Prevents progress regression on S-curves by only searching forward.
     * @param position Query position
     * @param min_s Minimum arc length to search from
     * @return Arc length of closest point (>= min_s)
     */
    double find_closest_point(const Eigen::Vector2d& position, double min_s) const;

    /**
     * @brief Compute lateral offset from path.
     * @param position Query position
     * @param s Arc length of reference point
     * @return Signed lateral offset (positive = left)
     */
    double compute_lateral_offset(const Eigen::Vector2d& position, double s) const;

    /**
     * @brief Get position at fraction of path.
     * @param fraction Fraction [0, 1]
     * @return Position
     */
    Eigen::Vector2d get_position_at_fraction(double fraction) const;

    /// Total path length
    double total_length() const { return total_length_; }

    /// Duration of the optional time parameterization.
    double total_duration() const { return total_duration_; }

    /// Whether this path has an associated speed/time profile.
    bool has_time_parameterization() const { return has_timing_; }

    /// Whether the path has an explicit periodic geometric seam.
    bool is_closed_loop() const { return closed_loop_; }

    /// Normalize an arc-length coordinate onto a periodic path. Open paths are
    /// clamped to their endpoint so callers can use one uniform interface.
    double wrap_arc_length(double s) const;

    /// Path type
    PathType path_type() const { return path_type_; }

    /// Number of waypoints
    size_t num_points() const { return points_.size(); }

    /// Get all path points
    const std::vector<PathPoint>& points() const { return points_; }

private:
    std::vector<PathPoint> points_;
    double total_length_;
    double total_duration_;
    PathType path_type_;
    bool has_timing_;
    bool closed_loop_;

    double find_closest_point_projection(
        const Eigen::Vector2d& position, double min_s) const;

    /// Interpolate between two points
    PathPoint interpolate(const PathPoint& p1, const PathPoint& p2, double t) const;
};

}  // namespace dro_mpc

#endif  // SCENARIO_MPC_REFERENCE_PATH_HPP
