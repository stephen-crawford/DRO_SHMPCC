/**
 * @file contouring_mpc.hpp
 * @brief Contouring MPC with Safe Horizon Adaptive Mode Sampling.
 *
 * Implements the full scenario-based MPC with:
 * - Contouring path following objective
 * - Road boundary constraints
 * - Adaptive mode-based obstacle prediction
 * - Multiple dynamic obstacles with different behavior modes
 */

#ifndef SCENARIO_MPC_CONTOURING_MPC_HPP
#define SCENARIO_MPC_CONTOURING_MPC_HPP

#include "mpc_controller.hpp"
#include "reference_path.hpp"
#include <functional>

namespace scenario_mpc {

/**
 * @brief Obstacle behavior modes.
 */
enum class ObstacleBehavior {
    CONSTANT_VELOCITY,  ///< Constant velocity motion
    TURN_LEFT,          ///< Turning left
    TURN_RIGHT,         ///< Turning right
    DECELERATING,       ///< Slowing down
    ACCELERATING,       ///< Speeding up
    LANE_CHANGE_LEFT,   ///< Lane change left
    LANE_CHANGE_RIGHT,  ///< Lane change right
    PATH_INTERSECT      ///< Moving back and forth across path
};

/**
 * @brief Configuration for a single obstacle.
 */
struct ObstacleConfiguration {
    int obstacle_id;
    std::string name;
    ObstacleBehavior initial_mode;
    std::vector<std::string> available_modes;
    double mode_switch_probability = 0.05;
    double radius = 0.35;

    // Initial state (set by placement function)
    Eigen::Vector2d initial_position;
    Eigen::Vector2d initial_velocity;

    // Uncertainty parameters
    double position_std = 0.15;
    double uncertainty_growth = 0.08;
};

/**
 * @brief Configuration for Contouring MPC with Safe Horizon.
 */
struct ContouringMPCConfig : public ScenarioMPCConfig {
    // Contouring parameters
    double road_width = 7.0;          ///< Total road width [m]
    double contour_weight = 1.0;      ///< Weight for contouring error
    double lag_weight = 0.1;          ///< Weight for lag error
    double terminal_weight = 10.0;    ///< Terminal contouring weight
    double terminal_angle_weight = 1.0;  ///< Terminal angle alignment weight

    // Safe Horizon parameters
    double epsilon_p = 0.15;          ///< Violation probability tolerance
    double beta_confidence = 0.1;     ///< Confidence level
    int min_scenarios = 50;           ///< Minimum scenarios
    int max_scenarios = 500;          ///< Maximum scenarios

    // Adaptive mode sampling
    bool enable_adaptive_mode_sampling = true;
    double mode_recency_decay = 0.1;  ///< Lambda parameter for recency
    double mode_frequency_smoothing = 1.0;  ///< Alpha parameter for frequency

    // Path parameters
    double path_length = 25.0;        ///< Reference path length
    double s_curve_amplitude = 3.0;   ///< S-curve amplitude

    /// Compute required scenarios using Theorem 1
    int compute_safe_horizon_scenarios() const {
        // S >= (2/epsilon) * (ln(1/beta) + d)
        // where d = horizon * (n_x + n_u)
        int d = horizon * 6;  // 4 state + 2 input
        return std::max(min_scenarios, std::min(max_scenarios,
            static_cast<int>(std::ceil(2.0 / epsilon_p * (std::log(1.0 / beta_confidence) + d)))
        ));
    }
};

/**
 * @brief Result from Contouring MPC.
 */
struct ContouringMPCResult : public MPCResult {
    double path_progress = 0.0;       ///< Progress along path [m]
    double lateral_error = 0.0;       ///< Lateral deviation from path [m]
    double heading_error = 0.0;       ///< Heading error [rad]
    int num_scenarios_used = 0;       ///< Number of scenarios in optimization
    int num_active_constraints = 0;   ///< Number of active collision constraints
    std::vector<double> obstacle_distances;  ///< Distance to each obstacle
};

/**
 * @brief Contouring MPC with Safe Horizon Adaptive Mode Sampling.
 */
class ContouringMPC {
public:
    /**
     * @brief Initialize the contouring MPC controller.
     * @param config Configuration parameters
     */
    explicit ContouringMPC(const ContouringMPCConfig& config);

    /**
     * @brief Set the reference path.
     * @param path Reference path for contouring
     */
    void set_reference_path(const ReferencePath& path);

    /**
     * @brief Add an obstacle with configuration.
     * @param config Obstacle configuration
     */
    void add_obstacle(const ObstacleConfiguration& config);

    /**
     * @brief Place obstacles along the path.
     * @param configs Obstacle configurations
     * @param path_fractions Fractions along path [0, 1] for each obstacle
     * @param lateral_offsets Lateral offset from centerline [m]
     * @param velocities_toward_path Velocities toward path centerline [m/s]
     */
    void place_obstacles_along_path(
        std::vector<ObstacleConfiguration>& configs,
        const std::vector<double>& path_fractions,
        const std::vector<double>& lateral_offsets,
        const std::vector<double>& velocities_toward_path
    );

    /**
     * @brief Update obstacle state and mode observation.
     * @param obstacle_id Obstacle identifier
     * @param position Current position
     * @param velocity Current velocity
     * @param observed_mode Observed behavior mode
     */
    void update_obstacle(
        int obstacle_id,
        const Eigen::Vector2d& position,
        const Eigen::Vector2d& velocity,
        const std::string& observed_mode
    );

    /**
     * @brief Solve the contouring MPC problem.
     * @param ego_state Current ego vehicle state
     * @param current_path_progress Current progress along path
     * @param reference_velocity Desired velocity
     * @return ContouringMPCResult with trajectory and diagnostics
     */
    ContouringMPCResult solve(
        const EgoState& ego_state,
        double current_path_progress,
        double reference_velocity = 2.0
    );

    /**
     * @brief Simulate obstacle motion for one timestep.
     * @param obstacle_id Obstacle identifier
     * @param dt Timestep
     * @return New obstacle state
     */
    ObstacleState simulate_obstacle_step(int obstacle_id, double dt);

    /**
     * @brief Get current obstacle states.
     */
    const std::map<int, ObstacleState>& obstacles() const { return obstacles_; }

    /**
     * @brief Get obstacle configuration.
     */
    const ObstacleConfiguration& get_obstacle_config(int obstacle_id) const {
        return obstacle_configs_.at(obstacle_id);
    }

    /**
     * @brief Get reference path.
     */
    const ReferencePath& reference_path() const { return reference_path_; }

    /**
     * @brief Get configuration.
     */
    const ContouringMPCConfig& config() const { return config_; }

    /**
     * @brief Get current sampled scenarios.
     */
    const std::vector<Scenario>& scenarios() const { return base_controller_.scenarios(); }

    /**
     * @brief Get base controller for accessing mode histories.
     */
    const AdaptiveScenarioMPC& base_controller() const { return base_controller_; }

    /**
     * @brief Reset controller state.
     */
    void reset();

private:
    ContouringMPCConfig config_;
    ReferencePath reference_path_;
    AdaptiveScenarioMPC base_controller_;

    std::map<int, ObstacleState> obstacles_;
    std::map<int, ObstacleConfiguration> obstacle_configs_;
    std::map<int, ObstacleBehavior> current_modes_;

    std::mt19937 rng_;
    int iteration_count_ = 0;

    /**
     * @brief Compute contouring cost for a trajectory.
     */
    double compute_contouring_cost(
        const std::vector<EgoState>& trajectory,
        double start_progress
    ) const;

    /**
     * @brief Check road boundary constraints.
     */
    bool check_road_constraints(const EgoState& state, double path_progress) const;

    /**
     * @brief Get goal position based on path progress.
     */
    Eigen::Vector2d get_goal_from_progress(double current_progress, double lookahead) const;

    /**
     * @brief Convert obstacle behavior to mode string.
     */
    static std::string behavior_to_mode_string(ObstacleBehavior behavior);

    /**
     * @brief Possibly switch obstacle mode based on probability.
     */
    void maybe_switch_obstacle_mode(int obstacle_id);
};

/**
 * @brief Create default obstacle configurations for testing.
 *
 * Creates 4 obstacles with different behavior modes:
 * - Obstacle 0: Constant velocity
 * - Obstacle 1: Turning
 * - Obstacle 2: Decelerating
 * - Obstacle 3: Lane changing
 *
 * @return Vector of obstacle configurations
 */
std::vector<ObstacleConfiguration> create_default_obstacle_configs();

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_CONTOURING_MPC_HPP
