/**
 * @file dynamics.hpp
 * @brief Ego vehicle dynamics model.
 *
 * Implements the unicycle model with acceleration and angular velocity inputs.
 */

#ifndef DRO_MPC_DYNAMICS_HPP
#define DRO_MPC_DYNAMICS_HPP

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
    //  Accessible as ego_dynamics_.model.max_velocity, etc.
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
     * s algebraically
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
     *
     * These are the EXACT derivatives of discrete_dynamics(), i.e. of the RK4 map,
     * obtained by propagating the variational equations through the four stages:
     *
     *     K_1 = J_1,  K_i = J_i (I + c_i h K_{i-1}),   A = I + (h/6) sum w_i K_i,
     *     L_1 = F_1,  L_i = J_i (c_i h L_{i-1}) + F_i, B = (h/6) sum w_i L_i,
     *
     * with J_i = df/dx and F_i = df/du evaluated at stage state i, c = (-, 1/2, 1/2, 1)
     * and w = (1, 2, 2, 1). 
     *
     * @param state State vector [x, y, theta, v]
     * @param input Input vector [a, w]
     * @return Pair of (A, B) where x_next approx A @ x + B @ u + c
     */
    std::pair<Eigen::Matrix4d, Eigen::Matrix<double, 4, 2>>
    get_jacobians(const Eigen::Vector4d& state, const Eigen::Vector2d& input, double dt = -1) const;

    /**
     * @brief Compute algebraic spline update for one timestep.
     *
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
    /// df/dx of continuous_dynamics() at (state, input).
    Eigen::Matrix4d continuous_state_jacobian(const Eigen::Vector4d& state,
                                              const Eigen::Vector2d& input) const;

    /// df/du of continuous_dynamics() at (state, input).
    Eigen::Matrix<double, 4, 2> continuous_input_jacobian(
        const Eigen::Vector4d& state, const Eigen::Vector2d& input) const;

    double dt_;  // Timestep for discrete integration
};

/**
 * @brief Create standard obstacle mode models.
 * @param dt Timestep for dynamics
 * @return Map of mode_id to ModeModel
 */
std::map<std::string, ModeModel> create_obstacle_mode_models(double dt = 0.1);

}  // namespace dro_mpc

#endif  // DRO_MPC_DYNAMICS_HPP
