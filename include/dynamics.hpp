/**
 * @file dynamics.hpp
 * @brief Ego vehicle dynamics model.
 *
 * Implements the unicycle model with acceleration and steering rate inputs.
 */

#ifndef SCENARIO_MPC_DYNAMICS_HPP
#define SCENARIO_MPC_DYNAMICS_HPP

#include "types.hpp"
#include <cmath>
#include <map>

namespace scenario_mpc {

/**
 * @brief Unicycle dynamics model for ego vehicle.
 *
 * State: x = [x, y, theta, v]
 * Input: u = [a, w] (acceleration, angular velocity)
 *
 * Continuous dynamics:
 *     dx/dt = v * cos(theta)
 *     dy/dt = v * sin(theta)
 *     dtheta/dt = w
 *     dv/dt = a
 */
class EgoDynamics {
public:
    static constexpr int STATE_DIM = 4;  ///< [x, y, theta, v]
    static constexpr int INPUT_DIM = 2;  ///< [a, w]

    /**
     * @brief Initialize dynamics model.
     * @param dt Timestep for discrete integration [s]
     */
    explicit EgoDynamics(double dt = 0.1);

    /**
     * @brief Compute continuous-time state derivative.
     * @param state State vector [x, y, theta, v]
     * @param input Input vector [a, w]
     * @return State derivative [dx, dy, dtheta, dv]
     */
    Eigen::Vector4d continuous_dynamics(const Eigen::Vector4d& state,
                                        const Eigen::Vector2d& input) const;

    /**
     * @brief Compute discrete-time state update using RK4 integration.
     * @param state Current state [x, y, theta, v]
     * @param input Control input [a, w]
     * @param dt Timestep (uses member dt if not provided)
     * @return Next state after dt
     */
    Eigen::Vector4d discrete_dynamics(const Eigen::Vector4d& state,
                                      const Eigen::Vector2d& input,
                                      double dt = -1) const;

    /**
     * @brief Propagate ego state forward one timestep.
     * @param state Current ego state
     * @param input Control input
     * @param dt Timestep (uses member dt if not provided)
     * @return Next ego state
     */
    EgoState propagate(const EgoState& state, const EgoInput& input,
                       double dt = -1) const;

    /**
     * @brief Roll out trajectory from initial state with given inputs.
     * @param initial_state Starting ego state
     * @param inputs List of EgoInput for each timestep
     * @param dt Timestep (uses member dt if not provided)
     * @return List of EgoState including initial state (length N+1)
     */
    std::vector<EgoState> rollout(const EgoState& initial_state,
                                  const std::vector<EgoInput>& inputs,
                                  double dt = -1) const;

    /**
     * @brief Compute Jacobians of discrete dynamics for linearization.
     * @param state State vector [x, y, theta, v]
     * @param input Input vector [a, w]
     * @return Pair of (A, B) where x_next approx A @ x + B @ u + c
     */
    std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 2>>
    get_jacobians(const Eigen::Vector4d& state, const Eigen::Vector2d& input) const;

    double dt() const { return dt_; }

private:
    double dt_;  ///< Timestep for discrete integration
};

/**
 * @brief Create standard obstacle mode models.
 * @param dt Timestep for dynamics
 * @return Map of mode_id to ModeModel
 */
std::map<std::string, ModeModel> create_obstacle_mode_models(double dt = 0.1);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_DYNAMICS_HPP
