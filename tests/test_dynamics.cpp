#include "dynamics.hpp"

#include <cmath>
#include <iostream>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

bool close(double lhs, double rhs, double tolerance = 1e-9) {
    return std::abs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
    constexpr double dt = 0.2;
    EgoDynamics dynamics(dt);

    {
        const Eigen::Vector4d state(1.0, -2.0, 0.0, 3.0);
        const Eigen::Vector2d input(2.0, 0.0);
        const Eigen::Vector4d next = dynamics.discrete_dynamics(state, input);
        check(close(next.x(), 1.64) && close(next.y(), -2.0) &&
                  close(next.z(), 0.0) && close(next.w(), 3.4),
              "RK4 is exact for straight constant-acceleration motion");
    }

    {
        const Eigen::Vector4d state(0.0, 0.0, 0.3, 2.0);
        const Eigen::Vector2d input(0.0, 0.5);
        const Eigen::Vector4d next = dynamics.discrete_dynamics(state, input);
        const double theta_next = state(2) + input(1) * dt;
        const double expected_x = state(3) / input(1) *
            (std::sin(theta_next) - std::sin(state(2)));
        const double expected_y = state(3) / input(1) *
            (std::cos(state(2)) - std::cos(theta_next));
        check(close(next.x(), expected_x, 2e-7) && close(next.y(), expected_y, 2e-7) &&
                  close(next.z(), theta_next, 1e-12) && close(next.w(), state(3), 1e-12),
              "RK4 agrees with the analytic constant-turn solution");
    }

    {
        const Eigen::Vector4d state(0.4, -0.7, 0.6, 1.3);
        const Eigen::Vector2d input(-0.4, 0.25);
        const auto [A, B] = dynamics.get_jacobians(state, input);
        constexpr double h = 1e-6;
        Eigen::Matrix4d finite_A;
        Eigen::Matrix<double, 4, 2> finite_B;
        for (int i = 0; i < 4; ++i) {
            Eigen::Vector4d plus = state;
            Eigen::Vector4d minus = state;
            plus(i) += h;
            minus(i) -= h;
            finite_A.col(i) = (dynamics.discrete_dynamics(plus, input) -
                               dynamics.discrete_dynamics(minus, input)) / (2.0 * h);
        }
        for (int i = 0; i < 2; ++i) {
            Eigen::Vector2d plus = input;
            Eigen::Vector2d minus = input;
            plus(i) += h;
            minus(i) -= h;
            finite_B.col(i) = (dynamics.discrete_dynamics(state, plus) -
                               dynamics.discrete_dynamics(state, minus)) / (2.0 * h);
        }
        check((A - finite_A).cwiseAbs().maxCoeff() < 2e-6 &&
                  (B - finite_B).cwiseAbs().maxCoeff() < 2e-6,
              "analytic RK4 Jacobians agree with central finite differences");
    }

    {
        const std::vector<EgoInput> inputs = {
            EgoInput(0.0, 0.0), EgoInput(1.0, 0.0), EgoInput(-1.0, 0.0)};
        const auto rollout = dynamics.rollout(EgoState(0.0, 0.0, 0.0, 1.0), inputs);
        check(rollout.size() == inputs.size() + 1 &&
                  close(rollout.front().x, 0.0) && rollout.back().x > rollout.front().x,
              "rollout retains its initial state and applies every input once");
    }

    {
        const auto modes = create_obstacle_mode_models(0.1);
        const auto& constant_velocity = modes.at("constant_velocity");
        const ObstacleState initial(0.0, 0.0, 2.0, -1.0);
        const ObstacleState next = constant_velocity.propagate(initial);
        check(close(next.x, 0.2) && close(next.y, -0.1) &&
                  close(next.vx, 2.0) && close(next.vy, -1.0),
              "constant-velocity obstacle model has the configured discrete kinematics");
    }

    std::cout << (failures == 0 ? "ALL DYNAMICS TESTS PASSED\n"
                                : "DYNAMICS TESTS FAILED\n");
    return failures == 0 ? 0 : 1;
}
