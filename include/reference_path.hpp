/**
 * @file reference_path.hpp
 * @brief Reference path for contouring MPC.
 *
 * Implements path representations including:
 * - Straight lines
 * - S-curves
 * - Custom waypoint paths
 */

#ifndef SCENARIO_MPC_REFERENCE_PATH_HPP
#define SCENARIO_MPC_REFERENCE_PATH_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <string>

namespace scenario_mpc {

/**
 * @brief Point on reference path with position and tangent.
 */
struct PathPoint {
    Eigen::Vector2d position;    ///< Position [x, y]
    double heading;              ///< Tangent angle [rad]
    double curvature;            ///< Path curvature [1/m]
    double s;                    ///< Arc length parameter

    PathPoint() : heading(0), curvature(0), s(0) {}
    PathPoint(const Eigen::Vector2d& pos, double h, double k, double s)
        : position(pos), heading(h), curvature(k), s(s) {}
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
        CUSTOM
    };

    ReferencePath() : total_length_(0), path_type_(PathType::STRAIGHT) {}

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

    /// Path type
    PathType path_type() const { return path_type_; }

    /// Number of waypoints
    size_t num_points() const { return points_.size(); }

    /// Get all path points
    const std::vector<PathPoint>& points() const { return points_; }

private:
    std::vector<PathPoint> points_;
    double total_length_;
    PathType path_type_;

    /// Interpolate between two points
    PathPoint interpolate(const PathPoint& p1, const PathPoint& p2, double t) const;
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_REFERENCE_PATH_HPP
