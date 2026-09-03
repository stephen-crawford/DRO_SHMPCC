/**
 * @file dynamics.cpp
 * @brief Implementation of ego vehicle dynamics and obstacle mode models.
 */

#include "dynamics.hpp"
#include "reference_path.hpp"
#include <cmath>
#include <algorithm>

namespace dro_mpc {

EgoDynamics::EgoDynamics(double dt) : dt_(dt) {}

EgoDynamics::EgoDynamics(const EgoDynamicsConfig& model_spec, double dt)
    : model(model_spec), dt_(dt) {}

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

Eigen::Matrix4d EgoDynamics::continuous_state_jacobian(
    const Eigen::Vector4d& state,
    const Eigen::Vector2d& /*input*/
) const {
    const double theta = state(2);
    const double v = state(3);

    Eigen::Matrix4d J = Eigen::Matrix4d::Zero();
    J(0, 2) = -v * std::sin(theta);
    J(0, 3) = std::cos(theta);
    J(1, 2) = v * std::cos(theta);
    J(1, 3) = std::sin(theta);
    // dtheta/dt = w and dv/dt = a carry no state dependence, so rows 2-3 stay zero.
    return J;
}

Eigen::Matrix<double, 4, 2> EgoDynamics::continuous_input_jacobian(
    const Eigen::Vector4d& /*state*/,
    const Eigen::Vector2d& /*input*/
) const {
    Eigen::Matrix<double, 4, 2> F = Eigen::Matrix<double, 4, 2>::Zero();
    F(2, 1) = 1.0;  // dtheta/dt = w
    F(3, 0) = 1.0;  // dv/dt = a
    return F;
}

std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 2>>
EgoDynamics::get_jacobians(const Eigen::Vector4d& state,
                           const Eigen::Vector2d& input,
                           double dt) const {

    const double h = dt < 0 ? dt_ : dt;
    const Eigen::Matrix4d I = Eigen::Matrix4d::Identity();

    // Stage states; these MUST mirror discrete_dynamics() exactly or the returned
    // Jacobians are the derivative of a different map than the one being rolled out.
    const Eigen::Vector4d k1 = continuous_dynamics(state, input);
    const Eigen::Vector4d s2 = state + (h / 2) * k1;
    const Eigen::Vector4d k2 = continuous_dynamics(s2, input);
    const Eigen::Vector4d s3 = state + (h / 2) * k2;
    const Eigen::Vector4d k3 = continuous_dynamics(s3, input);
    const Eigen::Vector4d s4 = state + h * k3;

    const Eigen::Matrix4d J1 = continuous_state_jacobian(state, input);
    const Eigen::Matrix4d J2 = continuous_state_jacobian(s2, input);
    const Eigen::Matrix4d J3 = continuous_state_jacobian(s3, input);
    const Eigen::Matrix4d J4 = continuous_state_jacobian(s4, input);

    // d(k_i)/d(state): each stage state depends on the previous stage slope.
    const Eigen::Matrix4d K1 = J1;
    const Eigen::Matrix4d K2 = J2 * (I + (h / 2) * K1);
    const Eigen::Matrix4d K3 = J3 * (I + (h / 2) * K2);
    const Eigen::Matrix4d K4 = J4 * (I + h * K3);
    const Eigen::Matrix4d A = I + (h / 6) * (K1 + 2 * K2 + 2 * K3 + K4);

    // d(k_i)/d(input): u enters each stage directly (F_i) AND through the stage state.
    const Eigen::Matrix<double, 4, 2> F1 = continuous_input_jacobian(state, input);
    const Eigen::Matrix<double, 4, 2> F2 = continuous_input_jacobian(s2, input);
    const Eigen::Matrix<double, 4, 2> F3 = continuous_input_jacobian(s3, input);
    const Eigen::Matrix<double, 4, 2> F4 = continuous_input_jacobian(s4, input);

    const Eigen::Matrix<double, 4, 2> L1 = F1;
    const Eigen::Matrix<double, 4, 2> L2 = J2 * ((h / 2) * L1) + F2;
    const Eigen::Matrix<double, 4, 2> L3 = J3 * ((h / 2) * L2) + F3;
    const Eigen::Matrix<double, 4, 2> L4 = J4 * (h * L3) + F4;
    const Eigen::Matrix<double, 4, 2> B = (h / 6) * (L1 + 2 * L2 + 2 * L3 + L4);

    return {A, B};
}

double EgoDynamics::compute_spline_update(
    const EgoState& prev_state,
    const EgoState& next_state,
    const ReferencePath& path,
    double dt
) {
    (void)dt;
    if (prev_state.s < 0.0) return -1.0;

    // Progress is the monotone closest point on the actual discretized path.
    // This is the same geometric evaluation used by the controller outside
    // the dynamics rollout, rather than a chord/curvature approximation.
    return path.find_closest_point(next_state.position(), prev_state.s);
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

    // Longitudinal speed variants.  Scaling velocity preserves the current
    // travel direction, unlike a fixed world-frame acceleration bias.
    Eigen::Matrix4d A_acc = A_cv;
    A_acc(2, 2) = 1.08;
    A_acc(3, 3) = 1.08;
    modes["accelerating"] = ModeModel(
        "accelerating", A_acc, b_cv, G_cv, "Accelerating along current velocity"
    );

    Eigen::Matrix4d A_dec = A_cv;
    A_dec(2, 2) = 0.82;
    A_dec(3, 3) = 0.82;
    modes["decelerating"] = ModeModel(
        "decelerating", A_dec, b_cv, G_cv, "Decelerating along current velocity"
    );

    Eigen::Matrix4d A_stop = A_cv;
    A_stop(2, 2) = 0.15;
    A_stop(3, 3) = 0.15;
    modes["stop"] = ModeModel(
        "stop", A_stop, b_cv, G_cv, "Rapidly braking to a stop"
    );

    const auto add_turn = [&](const std::string& id, double omega) {
        const double cosine = std::cos(omega * dt);
        const double sine = std::sin(omega * dt);
        Eigen::Matrix4d A_turn;
        A_turn << 1, 0, dt * cosine, -dt * sine,
                  0, 1, dt * sine,  dt * cosine,
                  0, 0, cosine,     -sine,
                  0, 0, sine,       cosine;
        modes[id] = ModeModel(
            id, A_turn, b_cv, G_cv,
            omega > 0.0 ? "Left turning motion" : "Right turning motion");
    };
    add_turn("turn_left", 0.3);
    add_turn("turn_right", -0.3);
    add_turn("turn_left_sharp", 0.65);
    add_turn("turn_right_sharp", -0.65);

    const auto add_lane_change = [&](const std::string& id, double lateral_step) {
        modes[id] = ModeModel(id, A_cv, b_cv, G_cv,
                              lateral_step > 0.0 ? "Lane change left" : "Lane change right");
        modes[id].body_lateral_displacement = lateral_step;
    };
    add_lane_change("lane_change_left", 0.3 * dt);
    add_lane_change("lane_change_right", -0.3 * dt);
    add_lane_change("lane_change_left_fast", 0.7 * dt);
    add_lane_change("lane_change_right_fast", -0.7 * dt);

    return modes;
}

std::vector<std::string> select_obstacle_mode_ids(
    const std::vector<std::string>& regular_candidates,
    const std::string& rare_mode,
    const std::map<std::string, ModeModel>& mode_catalog,
    bool randomize,
    int requested_regular_modes,
    std::mt19937& rng
) {
    std::vector<std::string> selected;
    for (const auto& mode_id : regular_candidates) {
        if (mode_catalog.count(mode_id) != 0 &&
            std::find(selected.begin(), selected.end(), mode_id) == selected.end()) {
            selected.push_back(mode_id);
        }
    }
    if (selected.empty() && mode_catalog.count("constant_velocity") != 0) {
        selected.push_back("constant_velocity");
    }
    if (randomize) {
        std::shuffle(selected.begin(), selected.end(), rng);
        const int count = std::max(1, requested_regular_modes);
        selected.resize(std::min(static_cast<size_t>(count), selected.size()));
    }
    if (!rare_mode.empty() && mode_catalog.count(rare_mode) != 0 &&
        std::find(selected.begin(), selected.end(), rare_mode) == selected.end()) {
        selected.push_back(rare_mode);
    }
    return selected;
}

}  // namespace dro_mpc
