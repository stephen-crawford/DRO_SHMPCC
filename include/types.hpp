/**
 * @file types.hpp
 * @brief Core data structures for Adaptive Scenario-Based MPC.
 *
 * Following the mathematical formulation:
 * - Section 2: State Representations
 * - Section 3: Mode and Dynamics Models
 * - Section 4: Mode History and Weights
 * - Section 5: Trajectory and Scenario Structures
 */

#ifndef SCENARIO_MPC_TYPES_HPP
#define SCENARIO_MPC_TYPES_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <optional>
#include <cmath>

namespace scenario_mpc {

// =============================================================================
// Section 2: State Representations
// =============================================================================

/**
 * @brief Ego vehicle state: x_ego = (x, y, theta, v, s)
 *
 * The first 4 components [x, y, theta, v] are integrated via RK4.
 * The spline parameter s (arc length along reference path) is updated
 * algebraically after integration, following the Python reference
 * (ContouringSecondOrderUnicycleModel).
 */
struct EgoState {
    double x;      ///< Position x-coordinate [m]
    double y;      ///< Position y-coordinate [m]
    double theta;  ///< Heading angle [rad]
    double v;      ///< Velocity magnitude [m/s]
    double s;      ///< Spline/arc-length parameter along reference path [m] (-1 = not tracking)

    EgoState() : x(0), y(0), theta(0), v(0), s(-1) {}
    EgoState(double x, double y, double theta, double v)
        : x(x), y(y), theta(theta), v(v), s(-1) {}
    EgoState(double x, double y, double theta, double v, double s)
        : x(x), y(y), theta(theta), v(v), s(s) {}

    /// Convert to Eigen vector [x, y, theta, v] (dynamics-only, 4D)
    Eigen::Vector4d to_array() const {
        return Eigen::Vector4d(x, y, theta, v);
    }

    /// Create from Eigen vector (dynamics-only, 4D; preserves s=-1)
    static EgoState from_array(const Eigen::Vector4d& arr) {
        return EgoState(arr(0), arr(1), arr(2), arr(3));
    }

    /// Create from Eigen vector with spline parameter
    static EgoState from_array(const Eigen::Vector4d& arr, double s) {
        return EgoState(arr(0), arr(1), arr(2), arr(3), s);
    }

    /// Get position as 2D vector [x, y]
    Eigen::Vector2d position() const {
        return Eigen::Vector2d(x, y);
    }

    /// Whether this state has a valid spline parameter
    bool has_spline() const { return s >= 0; }
};

/**
 * @brief Ego vehicle control input: u = (a, delta)
 */
struct EgoInput {
    double a;      ///< Acceleration [m/s^2]
    double delta;  ///< Steering angle or angular velocity [rad or rad/s]

    EgoInput() : a(0), delta(0) {}
    EgoInput(double a, double delta) : a(a), delta(delta) {}

    /// Convert to Eigen vector [a, delta]
    Eigen::Vector2d to_array() const {
        return Eigen::Vector2d(a, delta);
    }

    /// Create from Eigen vector
    static EgoInput from_array(const Eigen::Vector2d& arr) {
        return EgoInput(arr(0), arr(1));
    }
};

/**
 * @brief Obstacle state: x_obs = (x, y, vx, vy)
 */
struct ObstacleState {
    double x;   ///< Position x-coordinate [m]
    double y;   ///< Position y-coordinate [m]
    double vx;  ///< Velocity x-component [m/s]
    double vy;  ///< Velocity y-component [m/s]

    ObstacleState() : x(0), y(0), vx(0), vy(0) {}
    ObstacleState(double x, double y, double vx, double vy)
        : x(x), y(y), vx(vx), vy(vy) {}

    /// Convert to Eigen vector [x, y, vx, vy]
    Eigen::Vector4d to_array() const {
        return Eigen::Vector4d(x, y, vx, vy);
    }

    /// Create from Eigen vector
    static ObstacleState from_array(const Eigen::Vector4d& arr) {
        return ObstacleState(arr(0), arr(1), arr(2), arr(3));
    }

    /// Get position as 2D vector [x, y]
    Eigen::Vector2d position() const {
        return Eigen::Vector2d(x, y);
    }

    /// Get velocity as 2D vector [vx, vy]
    Eigen::Vector2d velocity() const {
        return Eigen::Vector2d(vx, vy);
    }
};

// =============================================================================
// Section 3: Mode and Dynamics Models
// =============================================================================

/**
 * @brief Mode-dependent dynamics model for obstacle prediction.
 *
 * Dynamics: x_{k+1} = A @ x_k + b + G @ w_k
 * where w_k ~ N(0, I) is process noise.
 */
struct ModeModel {
    std::string mode_id;          ///< Unique identifier for this mode
    Eigen::Matrix4d A;            ///< State transition matrix (4x4)
    Eigen::Vector4d b;            ///< Bias/drift vector (4,)
    Eigen::MatrixXd G;            ///< Process noise matrix (4 x n_noise)
    std::string description;      ///< Human-readable description

    ModeModel() : A(Eigen::Matrix4d::Identity()), b(Eigen::Vector4d::Zero()),
                  G(Eigen::MatrixXd::Zero(4, 2)) {}

    ModeModel(const std::string& mode_id, const Eigen::Matrix4d& A,
              const Eigen::Vector4d& b, const Eigen::MatrixXd& G,
              const std::string& description = "")
        : mode_id(mode_id), A(A), b(b), G(G), description(description) {}

    /// Propagate state one timestep forward
    ObstacleState propagate(const ObstacleState& state,
                           const Eigen::VectorXd* noise = nullptr) const {
        Eigen::Vector4d x = state.to_array();
        Eigen::Vector4d x_next = A * x + b;
        if (noise != nullptr) {
            x_next += G * (*noise);
        }
        return ObstacleState::from_array(x_next);
    }

    /// Dimension of process noise
    int noise_dim() const {
        return static_cast<int>(G.cols());
    }
};

// =============================================================================
// Section 4: Mode History and Weights
// =============================================================================

/**
 * @brief Track observed modes for an obstacle over time.
 */
struct ModeHistory {
    int obstacle_id;                                    ///< Unique obstacle identifier
    int obstacle_class = 0;                             ///< Class identifier (shared across obstacles)
    std::map<std::string, ModeModel> available_modes;   ///< Mode ID to ModeModel
    std::vector<std::pair<int, std::string>> observed_modes;  ///< (timestep, mode_id)
    int max_history = 100;                              ///< Maximum history length

    ModeHistory() : obstacle_id(0) {}
    ModeHistory(int obstacle_id, const std::map<std::string, ModeModel>& modes,
                int obstacle_class = 0)
        : obstacle_id(obstacle_id), obstacle_class(obstacle_class),
          available_modes(modes) {}

    /// Record a mode observation at the given timestep
    void record_observation(int timestep, const std::string& mode_id) {
        observed_modes.emplace_back(timestep, mode_id);
        // Trim history if too long
        if (static_cast<int>(observed_modes.size()) > max_history) {
            observed_modes.erase(observed_modes.begin(),
                observed_modes.begin() + (observed_modes.size() - max_history));
        }
    }

    /// Count occurrences of each mode in history
    std::map<std::string, int> get_mode_counts() const {
        std::map<std::string, int> counts;
        for (const auto& [mode_id, _] : available_modes) {
            counts[mode_id] = 0;
        }
        for (const auto& [_, mode_id] : observed_modes) {
            counts[mode_id]++;
        }
        return counts;
    }

    /// Get the n most recent observed modes
    std::vector<std::string> get_recent_modes(int n) const {
        std::vector<std::string> recent;
        int start = std::max(0, static_cast<int>(observed_modes.size()) - n);
        for (size_t i = start; i < observed_modes.size(); ++i) {
            recent.push_back(observed_modes[i].second);
        }
        return recent;
    }
};

// =============================================================================
// Section 5: Trajectory and Scenario Structures
// =============================================================================

/**
 * @brief Single step of an obstacle trajectory prediction.
 */
struct PredictionStep {
    int k;                        ///< Timestep index
    Eigen::Vector2d mean;         ///< Mean position [x, y]
    Eigen::Matrix2d covariance;   ///< Position covariance (2x2)

    PredictionStep() : k(0), mean(Eigen::Vector2d::Zero()),
                       covariance(Eigen::Matrix2d::Zero()) {}
    PredictionStep(int k, const Eigen::Vector2d& mean, const Eigen::Matrix2d& cov)
        : k(k), mean(mean), covariance(cov) {}
};

/**
 * @brief Predicted trajectory for a single obstacle over the horizon.
 */
struct ObstacleTrajectory {
    int obstacle_id;                   ///< Unique obstacle identifier
    std::string mode_id;               ///< Mode used for this trajectory
    std::vector<PredictionStep> steps; ///< Prediction steps over horizon
    double probability = 1.0;          ///< Probability/weight of this trajectory

    ObstacleTrajectory() : obstacle_id(0) {}
    ObstacleTrajectory(int obstacle_id, const std::string& mode_id,
                       const std::vector<PredictionStep>& steps,
                       double probability = 1.0)
        : obstacle_id(obstacle_id), mode_id(mode_id), steps(steps),
          probability(probability) {}

    /// Prediction horizon length
    int horizon() const {
        return static_cast<int>(steps.size()) - 1;
    }

    /// Get mean position at timestep k
    Eigen::Vector2d get_mean_at(int k) const {
        return steps[k].mean;
    }

    /// Get position covariance at timestep k
    Eigen::Matrix2d get_covariance_at(int k) const {
        return steps[k].covariance;
    }
};

/**
 * @brief A scenario is a collection of obstacle trajectories.
 */
struct Scenario {
    int scenario_id;                                      ///< Unique scenario identifier
    std::map<int, ObstacleTrajectory> trajectories;       ///< obstacle_id -> trajectory
    double probability = 1.0;                             ///< Combined probability
    bool is_injected = false;                             ///< True if DRO worst-case injected (never prune)

    Scenario() : scenario_id(0) {}
    Scenario(int scenario_id, const std::map<int, ObstacleTrajectory>& trajectories,
             double probability = 1.0)
        : scenario_id(scenario_id), trajectories(trajectories),
          probability(probability) {}

    /// Number of obstacles in this scenario
    int num_obstacles() const {
        return static_cast<int>(trajectories.size());
    }

    /// Get obstacle mean position at timestep k
    Eigen::Vector2d get_obstacle_position_at(int obstacle_id, int k) const {
        return trajectories.at(obstacle_id).get_mean_at(k);
    }
};

/**
 * @brief First and second moments of obstacle trajectory distribution.
 */
struct TrajectoryMoments {
    int obstacle_id;                           ///< Unique obstacle identifier
    Eigen::MatrixX2d means;                    ///< Mean positions (N+1, 2)
    std::vector<Eigen::Matrix2d> covariances;  ///< Position covariances (N+1)

    TrajectoryMoments() : obstacle_id(0) {}

    /// Prediction horizon length
    int horizon() const {
        return static_cast<int>(means.rows()) - 1;
    }

    /// Get mean position at timestep k
    Eigen::Vector2d get_mean_at(int k) const {
        return means.row(k);
    }

    /// Get position covariance at timestep k
    Eigen::Matrix2d get_covariance_at(int k) const {
        return covariances[k];
    }
};

// =============================================================================
// Additional Helper Types
// =============================================================================

/**
 * @brief Mode weight computation strategies.
 */
enum class WeightType {
    UNIFORM,          ///< Equal weights for all modes
    RECENCY,          ///< Exponential decay weighting recent observations
    FREQUENCY,        ///< Weights based on observation frequency
    WASSERSTEIN,      ///< OT-based inverse Wasserstein distance weights
    TEMPERATURE,      ///< Temperature-scaled frequency: w'_m = exp(log(w_m)/T)
    EPSILON_GREEDY    ///< Epsilon-greedy: w'_m = (1-eps)*w_m + eps/M
};

/**
 * @brief Linearized collision avoidance constraint.
 *
 * Form: a^T @ p_ego >= b
 */
struct CollisionConstraint {
    int k;                    ///< Timestep index
    int obstacle_id;          ///< Obstacle this constraint is for
    int scenario_id;          ///< Scenario this constraint belongs to
    Eigen::Vector2d a;        ///< Constraint normal vector (2,)
    double b;                 ///< Constraint offset (scalar)
    Eigen::Vector2d linearization_point = Eigen::Vector2d::Zero();  ///< Ego disc position at linearization

    CollisionConstraint() : k(0), obstacle_id(0), scenario_id(0), b(0) {}
    CollisionConstraint(int k, int obstacle_id, int scenario_id,
                        const Eigen::Vector2d& a, double b)
        : k(k), obstacle_id(obstacle_id), scenario_id(scenario_id), a(a), b(b) {}

    /**
     * @brief Evaluate constraint: positive = satisfied, negative = violated.
     * @param ego_position Ego position [x, y]
     * @return Constraint value (a^T @ p - b)
     */
    double evaluate(const Eigen::Vector2d& ego_position) const {
        return a.dot(ego_position) - b;
    }
};

/**
 * @brief Result from MPC optimization.
 */
struct MPCResult {
    bool success;                           ///< Whether optimization succeeded
    std::vector<EgoState> ego_trajectory;   ///< Planned ego states over horizon
    std::vector<EgoInput> control_inputs;   ///< Planned control inputs
    std::vector<int> active_scenarios;      ///< Scenarios with binding constraints
    double solve_time = 0.0;                ///< Optimization solve time [s]
    double cost = std::numeric_limits<double>::infinity();  ///< Optimal cost value
    int safe_horizon = -1;              ///< Truncated safe horizon used (-1 = full)
    int num_dro_injected = 0;           ///< Number of DRO worst-case scenarios injected
    double constraint_construction_time = 0.0;  ///< Time for constraint building [s]
    double qp_solve_time = 0.0;                 ///< Time for QP/SQP solve [s]

    MPCResult() : success(false) {}

    /// Get first control input for execution
    std::optional<EgoInput> first_input() const {
        if (!control_inputs.empty()) {
            return control_inputs[0];
        }
        return std::nullopt;
    }
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_TYPES_HPP
