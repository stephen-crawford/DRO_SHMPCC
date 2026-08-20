/**
 * @file dynamics.hpp
 * @brief Ego vehicle dynamics model.
 *
 * Implements the unicycle model with acceleration and steering rate inputs.
 */

#ifndef SCENARIO_MPC_DYNAMICS_HPP
#define SCENARIO_MPC_DYNAMICS_HPP

#include "types.hpp"
#include "config.hpp"
#include <cmath>
#include <map>

namespace dro_mpc {

// Forward declaration
class ReferencePath;

/**
 * @brief Unicycle dynamics model for ego vehicle.
 *
 * State: x = [x, y, theta, v] (integrated via RK4)
 * Spline: s (arc length, updated algebraically after integration)
 * Input: u = [a, w] (acceleration, angular velocity)
 *
 * Continuous dynamics (RK4-integrated):
 *     dx/dt = v * cos(theta)
 *     dy/dt = v * sin(theta)
 *     dtheta/dt = w
 *     dv/dt = a
 *
 * Algebraic spline update (after RK4):
 *     s_new = s + R * atan2(vt, R - e_c - vn)
 * where R = 1/curvature, vt/vn are tangential/normal displacement
 * components, and e_c is the contouring error.
 */
class EgoDynamics {
public:

    /// Motion-model spec + kinematic limits this integrator realizes.
    /// The limits are properties of the dynamics model (SLMPC-style), not the
    /// vehicle geometry. Accessible as ego_dynamics_.model.max_velocity, etc.
    EgoDynamicsConfig model;

    static constexpr int STATE_DIM = 4;  // [x, y, theta, v]
    static constexpr int INPUT_DIM = 2;  // [a, w]

    /**
     * @brief Initialize dynamics model with just a timestep (default limits).
     * @param dt Timestep for discrete integration [s]
     */
    explicit EgoDynamics(double dt = 0.1);

    /**
     * @brief Initialize from a dynamics-model spec (model kind + limits).
     * @param model Dynamics model specification (limits enforced on this model)
     * @param dt Timestep for discrete integration [s]
     */
    EgoDynamics(const EgoDynamicsConfig& model, double dt);

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
     * @brief Propagate ego state forward one timestep (4D dynamics only).
     * @param state Current ego state
     * @param input Control input
     * @param dt Timestep (uses member dt if not provided)
     * @return Next ego state (s remains unchanged)
     */
    EgoState propagate(const EgoState& state, const EgoInput& input,
                       double dt = -1) const;

    /**
     * @brief Roll out trajectory from initial state with given inputs (4D dynamics only).
     * @param initial_state Starting ego state
     * @param inputs List of EgoInput for each timestep
     * @param dt Timestep (uses member dt if not provided)
     * @return List of EgoState including initial state (length N+1)
     */
    std::vector<EgoState> rollout(const EgoState& initial_state,
                                  const std::vector<EgoInput>& inputs,
                                  double dt = -1) const;

    /**
     * @brief Roll out trajectory with algebraic spline update.
     *
     * Integrates [x,y,theta,v] via RK4, then updates the spline parameter
     * s algebraically using the curvature-aware formula from the Python
     * reference (ContouringSecondOrderUnicycleModel.model_discrete_dynamics).
     *
     * @param initial_state Starting ego state (with valid s >= 0)
     * @param inputs List of EgoInput for each timestep
     * @param path Reference path for spline update
     * @param dt Timestep (uses member dt if not provided)
     * @return List of EgoState including initial state (length N+1)
     */
    std::vector<EgoState> rollout_with_spline(
        const EgoState& initial_state,
        const std::vector<EgoInput>& inputs,
        const ReferencePath& path,
        double dt = -1) const;

    /**
     * @brief Compute Jacobians of discrete dynamics for linearization.
     * @param state State vector [x, y, theta, v]
     * @param input Input vector [a, w]
     * @return Pair of (A, B) where x_next approx A @ x + B @ u + c
     */
    std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 2>>
    get_jacobians(const Eigen::Vector4d& state, const Eigen::Vector2d& input) const;

    /**
     * @brief Compute algebraic spline update for one timestep.
     *
     * Uses curvature-aware formula: s_new = s + R * atan2(vt, R - e_c - vn)
     * with blending between curvature-aware and direct tangential projection.
     * Reference: Python ContouringSecondOrderUnicycleModel.model_discrete_dynamics
     *
     * @param prev_state Previous ego state (before integration)
     * @param next_state Next ego state (after RK4 integration)
     * @param path Reference path
     * @param dt Timestep
     * @return Updated spline parameter s_new
     */
    static double compute_spline_update(
        const EgoState& prev_state,
        const EgoState& next_state,
        const ReferencePath& path,
        double dt);

    double dt() const { return dt_; }

private:
    double dt_;  // Timestep for discrete integration
};

/**
 * @brief Create standard obstacle mode models.
 * @param dt Timestep for dynamics
 * @return Map of mode_id to ModeModel
 */
std::map<std::string, ModeModel> create_obstacle_mode_models(double dt = 0.1);

}  // namespace dro_mpc

#endif  // SCENARIO_MPC_DYNAMICS_HPP
