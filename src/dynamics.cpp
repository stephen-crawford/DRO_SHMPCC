/**
 * @file dynamics.cpp
 * @brief Implementation of ego vehicle dynamics and obstacle mode models.
 */

#include "dynamics.hpp"
#include "reference_path.hpp"
#include <cmath>
#include <algorithm>

namespace scenario_mpc {

EgoDynamics::EgoDynamics(double dt) : dt_(dt) {}

Eigen::Vector4d EgoDynamics::continuous_dynamics(const Eigen::Vector4d& state,
                                                  const Eigen::Vector2d& input) const {
    // Extract state components
    double theta = state(2);
    double v = state(3);

    // Extract inputs
    double a = input(0);
    double w = input(1);

    Eigen::Vector4d deriv;
    deriv(0) = v * std::cos(theta);  // dx/dt
    deriv(1) = v * std::sin(theta);  // dy/dt
    deriv(2) = w;                     // dtheta/dt
    deriv(3) = a;                     // dv/dt

    return deriv;
}

Eigen::Vector4d EgoDynamics::discrete_dynamics(const Eigen::Vector4d& state,
                                                const Eigen::Vector2d& input,
                                                double dt) const {
    if (dt < 0) {
        dt = dt_;
    }

    // RK4 integration
    Eigen::Vector4d k1 = continuous_dynamics(state, input);
    Eigen::Vector4d k2 = continuous_dynamics(state + dt / 2 * k1, input);
    Eigen::Vector4d k3 = continuous_dynamics(state + dt / 2 * k2, input);
    Eigen::Vector4d k4 = continuous_dynamics(state + dt * k3, input);

    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

EgoState EgoDynamics::propagate(const EgoState& state, const EgoInput& input,
                                 double dt) const {
    Eigen::Vector4d x = state.to_array();
    Eigen::Vector2d u = input.to_array();
    Eigen::Vector4d x_next = discrete_dynamics(x, u, dt);
    return EgoState::from_array(x_next);
}

std::vector<EgoState> EgoDynamics::rollout(const EgoState& initial_state,
                                            const std::vector<EgoInput>& inputs,
                                            double dt) const {
    std::vector<EgoState> states;
    states.reserve(inputs.size() + 1);
    states.push_back(initial_state);

    EgoState current = initial_state;
    for (const auto& u : inputs) {
        current = propagate(current, u, dt);
        states.push_back(current);
    }

    return states;
}

std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 2>>
EgoDynamics::get_jacobians(const Eigen::Vector4d& state,
                           const Eigen::Vector2d& input) const {
    double theta = state(2);
    double v = state(3);
    double dt = dt_;

    // Jacobian w.r.t. state (using RK4 approximation)
    Eigen::Matrix4d A = Eigen::Matrix4d::Identity();
    A(0, 2) = -v * std::sin(theta) * dt;
    A(0, 3) = std::cos(theta) * dt;
    A(1, 2) = v * std::cos(theta) * dt;
    A(1, 3) = std::sin(theta) * dt;

    // Jacobian w.r.t. input
    Eigen::Matrix<double, 4, 2> B = Eigen::Matrix<double, 4, 2>::Zero();
    B(2, 1) = dt;  // theta depends on w
    B(3, 0) = dt;  // v depends on a

    return {A, B};
}

double EgoDynamics::compute_spline_update(
    const EgoState& prev_state,
    const EgoState& next_state,
    const ReferencePath& path,
    double dt
) {
    double s = prev_state.s;
    if (s < 0) return -1.0;  // Not tracking spline

    double path_length = path.total_length();
    double s_clamped = std::clamp(s, 0.0, path_length);

    // Get path geometry at current spline position
    PathPoint pp = path.get_point_at(s_clamped);
    double ph = pp.heading;
    Eigen::Vector2d t_vec(std::cos(ph), std::sin(ph));    // tangent
    Eigen::Vector2d n_vec(-std::sin(ph), std::cos(ph));   // normal (left)

    // Displacement from current to integrated position
    Eigen::Vector2d dp = next_state.position() - prev_state.position();

    // Project displacement onto tangent and normal
    double vt = t_vec.dot(dp);   // tangential component
    double vn = n_vec.dot(dp);   // normal component

    // Contour error at current position
    Eigen::Vector2d pos_diff = prev_state.position() - pp.position;
    double contour_error = n_vec.dot(pos_diff);

    // Path curvature at current position
    double curvature = std::max(std::abs(pp.curvature), 1e-5);
    double R = std::min(1.0 / curvature, 1e4);  // Cap radius at 10km

    // Method 1: Curvature-aware formula (from C++/Python reference)
    // s_new = s + R * atan2(vt, R - contour_error - vn)
    double denominator = std::max(R - contour_error - vn, 1e-6);
    double theta_update = std::atan2(vt, denominator);
    theta_update = std::clamp(theta_update, -0.5, 0.5);
    double ds_curvature = R * theta_update;

    // Method 2: Direct tangential projection
    // vt directly represents progress along the path tangent
    double ds_tangential = vt;

    // Blend both methods (60% tangential, 40% curvature-aware)
    // Matching Python reference blend weights
    double ds = 0.6 * ds_tangential + 0.4 * ds_curvature;

    // Clamp update to reasonable range
    double v = std::max(prev_state.v, 0.0);
    double ds_max = v * dt * 5.0;
    double ds_min = (vt > 1e-6) ? -v * dt * 0.1 : 0.0;
    ds = std::clamp(ds, ds_min, ds_max);

    // Prevent backward progress when not moving forward along path
    if (vt <= 1e-6) {
        ds = std::max(ds, 0.0);
    }

    double s_new = s + ds;
    return std::max(s_new, 0.0);
}

std::vector<EgoState> EgoDynamics::rollout_with_spline(
    const EgoState& initial_state,
    const std::vector<EgoInput>& inputs,
    const ReferencePath& path,
    double dt) const
{
    std::vector<EgoState> states;
    states.reserve(inputs.size() + 1);

    // Initialize spline if not set: project onto path
    EgoState start = initial_state;
    if (!start.has_spline()) {
        start.s = path.find_closest_point(start.position());
    }
    states.push_back(start);

    EgoState current = start;
    for (const auto& u : inputs) {
        // RK4 integration of [x, y, theta, v]
        EgoState next = propagate(current, u, dt);

        // Algebraic spline update
        next.s = compute_spline_update(current, next, path,
                                        (dt > 0) ? dt : dt_);

        states.push_back(next);
        current = next;
    }

    return states;
}

std::map<std::string, ModeModel> create_obstacle_mode_models(double dt) {
    std::map<std::string, ModeModel> modes;

    // Constant velocity mode
    Eigen::Matrix4d A_cv;
    A_cv << 1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1;

    Eigen::Vector4d b_cv = Eigen::Vector4d::Zero();

    Eigen::MatrixXd G_cv(4, 2);
    G_cv << 0.5 * dt * dt, 0,
            0, 0.5 * dt * dt,
            dt, 0,
            0, dt;
    G_cv *= 0.5;  // Scale process noise

    modes["constant_velocity"] = ModeModel(
        "constant_velocity", A_cv, b_cv, G_cv, "Constant velocity motion"
    );

    // Decelerating mode
    Eigen::Matrix4d A_dec = A_cv;
    Eigen::Vector4d b_dec;
    b_dec << 0, 0, -0.5 * dt, -0.5 * dt;  // Deceleration

    modes["decelerating"] = ModeModel(
        "decelerating", A_dec, b_dec, G_cv, "Decelerating motion"
    );

    // Left turn mode
    double omega = 0.3;  // Turn rate [rad/s]
    double cos_w = std::cos(omega * dt);
    double sin_w = std::sin(omega * dt);

    Eigen::Matrix4d A_left;
    A_left << 1, 0, dt * cos_w, -dt * sin_w,
              0, 1, dt * sin_w, dt * cos_w,
              0, 0, cos_w, -sin_w,
              0, 0, sin_w, cos_w;
    Eigen::Vector4d b_left = Eigen::Vector4d::Zero();

    modes["turn_left"] = ModeModel(
        "turn_left", A_left, b_left, G_cv, "Left turning motion"
    );

    // Right turn mode
    Eigen::Matrix4d A_right;
    A_right << 1, 0, dt * cos_w, dt * sin_w,
               0, 1, -dt * sin_w, dt * cos_w,
               0, 0, cos_w, sin_w,
               0, 0, -sin_w, cos_w;
    Eigen::Vector4d b_right = Eigen::Vector4d::Zero();

    modes["turn_right"] = ModeModel(
        "turn_right", A_right, b_right, G_cv, "Right turning motion"
    );

    // Lane change left
    Eigen::Matrix4d A_lc = A_cv;
    Eigen::Vector4d b_lc_left;
    b_lc_left << 0, 0.3 * dt, 0, 0;  // Lateral drift left

    modes["lane_change_left"] = ModeModel(
        "lane_change_left", A_lc, b_lc_left, G_cv, "Lane change left"
    );

    // Lane change right
    Eigen::Vector4d b_lc_right;
    b_lc_right << 0, -0.3 * dt, 0, 0;  // Lateral drift right

    modes["lane_change_right"] = ModeModel(
        "lane_change_right", A_lc, b_lc_right, G_cv, "Lane change right"
    );

    return modes;
}

}  // namespace scenario_mpc
