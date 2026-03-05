/**
 * @file dynamics.cpp
 * @brief Implementation of ego vehicle dynamics and obstacle mode models.
 */

#include "dynamics.hpp"
#include <cmath>

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
