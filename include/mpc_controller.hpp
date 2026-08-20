/**
 * @file mpc_controller.hpp
 * @brief Adaptive Scenario-Based MPC Controller.
 *
 * Implements Algorithm 2: AdaptiveScenarioMPC
 *
 * Main control loop that:
 * 1. Samples scenarios from obstacle predictions
 * 2. Computes linearized collision constraints
 * 3. Solves the scenario-constrained optimization
 * 4. Updates mode histories with observations
 * 5. Prunes inactive scenarios
 */

#ifndef SCENARIO_MPC_MPC_CONTROLLER_HPP
#define SCENARIO_MPC_MPC_CONTROLLER_HPP

#include "types.hpp"
#include <map>
#include "config.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "collision_constraints.hpp"
#include "qp_solver.hpp"
#include "wasserstein_dro.hpp"
#include "reference_path.hpp"
#include <random>
#include <chrono>
#include <optional>
#include <vector>

namespace dro_mpc {

/**
 * @brief Statistics from MPC controller.
 */
struct MPCStatistics {
    int iteration_count = 0;
    double avg_solve_time = 0.0;
    double max_solve_time = 0.0;
    int num_obstacles = 0;
    int num_scenarios = 0;
};

/**
 * @brief Adaptive Scenario-Based Model Predictive Controller.
 *
 * Implements Algorithm 2.
 */
class AdaptiveScenarioMPC {
public:
    /**
     * @brief Initialize the MPC controller.
     * @param config Configuration parameters
     */
    explicit AdaptiveScenarioMPC(const RuntimeConfig& config);

    /**
     * @brief Initialize mode history for a new obstacle.
     * @param obstacle_id Unique obstacle identifier
     * @param obstacle_class Obstacle class identifier
     * @param available_modes Optional custom modes (uses defaults if empty)
     */
    void initialize_obstacle(
        int obstacle_id,
        int obstacle_class,
        const std::map<std::string, ModeModel>& available_modes = {}
    );

    /**
     * @brief Record a mode observation for an obstacle.
     * @param obstacle_id Obstacle identifier
     * @param obstacle_class Obstacle class identifier
     * @param observed_mode Observed mode ID
     * @param timestep Optional timestep (uses iteration count if -1)
     */
    void update_mode_observation(
        int obstacle_id,
        int obstacle_class,
        const std::string& observed_mode,
        int timestep = -1
    );


    /**
     * @brief Solve the MPC problem.
     *
     * Algorithm 2: AdaptiveScenarioMPC
     *
     * @param ego_state Current ego vehicle state
     * @param obstacles Current obstacle states
     * @param goal Goal position [x, y] for terminal cost
     * @param reference_velocity Desired velocity
     * @param path_progress Current progress along reference path (-1 to disable progress-aware features)
     * @param path_length Total reference path length (-1 to disable)
     * @return MPCResult with optimal trajectory and controls
     */
    MPCResult solve(
        const EgoState& ego_state,
        const std::map<int, ObstacleState>& obstacles,
        const Eigen::Vector2d& goal,
        double reference_velocity = 2.0,
        double path_progress = -1.0,
        double path_length = -1.0
    );

    /**
     * @brief Get controller statistics.
     * @return MPCStatistics struct
     */
    MPCStatistics get_statistics() const;

    /**
     * @brief Reset the controller state.
     *
     * Clears mode histories, scenarios, and statistics.
     */
    void reset();

    /// Get the configuration
    const RuntimeConfig& config() const { return config_; }

    /// Get current scenarios
    const std::vector<Scenario>& scenarios() const { return scenarios_; }

    /// Get DRO module (for diagnostics)
    const WassersteinDRO& dro() const { return dro_; }

    /// Set reference path for MPCC contouring/lag cost (Paper Eq. 6).
    void set_reference_path(const ReferencePath& path);
    /// Clear reference path (disables MPCC cost terms).
    void clear_reference_path();

    /// Set custom per-obstacle mode weights (e.g. from OT predictor).
    /// When set, these override the internal weight_type computation for
    /// scenario sampling. Cleared after each solve() call.
    void set_custom_mode_weights(int obstacle_id,
                                 const std::map<std::string, double>& weights);
    /// Clear all custom mode weights.
    void clear_custom_mode_weights();

    /// Mutable config access (for per-step adjustments like safety_margin).
    RuntimeConfig& mutable_config() { return config_; }

    /// Inject an extra scenario for the next solve() call.
    /// Pre-injected scenarios are added after normal sampling and marked
    /// is_injected=true so they survive pruning. Cleared after solve().
    void inject_scenario(const Scenario& scenario);

    /// Clear all pre-injected scenarios.
    void clear_injected_scenarios();

    /// Update a mode model's dynamics parameters (b, G) for all obstacles.
    void update_mode_model(const std::string& mode_id,
                           const Eigen::Vector4d& b_new,
                           const Eigen::Matrix4d& G_new);

private:
    /**
     * @brief Initialize reference trajectory for constraint linearization.
     * Uses previous solution shifted forward, or straight-line to goal.
     */
    void initialize_reference_trajectory(
        const EgoState& ego_state,
        const Eigen::Vector2d& goal,
        double reference_velocity
    );

    /**
     * @brief Generate straight-line reference trajectory.
     */
    std::vector<EgoState> generate_straight_line_trajectory(
        const EgoState& start,
        const Eigen::Vector2d& goal,
        double reference_velocity
    );

    /**
     * @brief Solve the scenario-constrained optimization problem.
     * Uses simple optimization (no CasADi dependency).
     */
    MPCResult solve_optimization(
        const EgoState& ego_state,
        const Eigen::Vector2d& goal,
        double reference_velocity,
        const std::vector<CollisionConstraint>& constraints,
        double path_progress = -1.0,
        double path_length = -1.0,
        int cost_horizon = -1
    );

    /**
     * @brief Solve optimization using SQP with ADMM QP subproblems.
     */
    MPCResult solve_optimization_sqp(
        const EgoState& ego_state,
        const Eigen::Vector2d& goal,
        double reference_velocity,
        const std::vector<CollisionConstraint>& constraints,
        double path_progress = -1.0,
        double path_length = -1.0,
        int cost_horizon = -1
    );

    /**
     * @brief Build condensed QP subproblem for SQP iteration.
     *
     * Condenses dynamics to express positions as linear function of inputs,
     * then maps collision constraints into input space. MPCC contouring/lag
     * objectives are applied to all steps 1..N. Collision constraints are
     * pre-filtered by the safe horizon in solve() before reaching this method.
     *
     * @param cost_horizon Reserved for future use (-1 = full horizon).
     *        Constraint truncation is handled upstream in solve().
     */
    QPProblem build_condensed_qp(
        const std::vector<EgoState>& x_ref,
        const std::vector<EgoInput>& u_ref,
        const Eigen::Vector2d& goal,
        double reference_velocity,
        const std::vector<CollisionConstraint>& constraints,
        double path_progress = -1.0,
        double path_length = -1.0,
        int cost_horizon = -1
    );

    /**
     * @brief Apply simple constraint avoidance by adjusting inputs.
     */
    std::pair<std::vector<EgoState>, std::vector<EgoInput>> apply_simple_avoidance(
        const EgoState& ego_state,
        std::vector<EgoState> trajectory,
        std::vector<EgoInput> inputs,
        const std::vector<CollisionConstraint>& constraints
    );

    /**
     * @brief Generate safe fallback trajectory (gentle braking).
     */
    MPCResult generate_safe_fallback(const EgoState& ego_state);

    RuntimeConfig config_;
    EgoDynamics ego_dynamics_;
    ADMMSolver qp_solver_;
    std::map<std::string, ModeModel> default_modes_;
    std::map<int, ModeHistory> mode_histories_;
    std::map<int, int> obstacle_classes_;  // obstacle_id -> obstacle_class
    std::vector<Scenario> scenarios_;
    WassersteinDRO dro_;
    std::vector<EgoState> reference_trajectory_;
    std::mt19937 rng_;
    std::vector<double> solve_times_;
    int iteration_count_ = 0;

    std::optional<ReferencePath> reference_path_;

    /// Custom per-obstacle mode weights (set externally, e.g. from OT predictor).
    std::map<int, std::map<std::string, double>> custom_per_obstacle_weights_;

    /// Pre-injected scenarios for the next solve() call.
    std::vector<Scenario> pre_injected_scenarios_;
};

}  // namespace dro_mpc

#endif  // SCENARIO_MPC_MPC_CONTROLLER_HPP
