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
#include "scenario_pruning.hpp"
#include "qp_solver.hpp"
#include "wasserstein_dro.hpp"
#include <random>
#include <chrono>
#include <optional>
#include <vector>

namespace scenario_mpc {

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
    explicit AdaptiveScenarioMPC(const ScenarioMPCConfig& config);

    /**
     * @brief Initialize mode history for a new obstacle.
     * @param obstacle_id Unique obstacle identifier
     * @param available_modes Optional custom modes (uses defaults if empty)
     */
    void initialize_obstacle(
        int obstacle_id,
        const std::map<std::string, ModeModel>& available_modes = {}
    );

    /**
     * @brief Record a mode observation for an obstacle.
     * @param obstacle_id Obstacle identifier
     * @param observed_mode Observed mode ID
     * @param timestep Optional timestep (uses iteration count if -1)
     */
    void update_mode_observation(
        int obstacle_id,
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
     * @param goal Goal position [x, y]
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
    const ScenarioMPCConfig& config() const { return config_; }

    /// Get current scenarios
    const std::vector<Scenario>& scenarios() const { return scenarios_; }

    /// Get DRO module (for diagnostics)
    const WassersteinDRO& dro() const { return dro_; }

    /// Set DRO epsilon override for next solve (e.g. adaptive shift). Clear after solve.
    void set_dro_epsilon_override(std::optional<double> rho);
    void clear_dro_epsilon_override();

    /// Custom mode weights for next solve (Conformal/Hazard/Bandit). If set, sampling uses these instead of history.
    void set_custom_mode_weights(int obstacle_id, const std::map<std::string, double>& weights);
    void clear_custom_mode_weights();

    /// Certificate radii for constraint tightening (Certificate-First §7.1). Applied after linearization.
    void set_certificate_radii(const std::vector<double>& radii);
    void clear_certificate_radii();

    /// Scenario compiler: set scenarios and skip sampling on next solve.
    void set_scenarios(const std::vector<Scenario>& scenarios);
    /// After set_scenarios(), next solve() uses current scenarios only (no sampling). Cleared after that solve.
    void set_use_current_scenarios_next(bool use);
    /// Sample N scenarios and set them for next solve (for compiler initial set). Call before solve().
    void sample_and_set_scenarios(const std::map<int, ObstacleState>& obstacles, int N);

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
        double path_length = -1.0
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
        double path_length = -1.0
    );

    /**
     * @brief Build condensed QP subproblem for SQP iteration.
     *
     * Condenses dynamics to express positions as linear function of inputs,
     * then maps collision constraints into input space.
     */
    QPProblem build_condensed_qp(
        const std::vector<EgoState>& x_ref,
        const std::vector<EgoInput>& u_ref,
        const Eigen::Vector2d& goal,
        double reference_velocity,
        const std::vector<CollisionConstraint>& constraints,
        double path_progress = -1.0,
        double path_length = -1.0
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

    ScenarioMPCConfig config_;
    EgoDynamics ego_dynamics_;
    ADMMSolver qp_solver_;
    std::map<std::string, ModeModel> default_modes_;
    std::map<int, ModeHistory> mode_histories_;
    std::vector<Scenario> scenarios_;
    WassersteinDRO dro_;
    std::vector<EgoState> reference_trajectory_;
    std::mt19937 rng_;
    std::vector<double> solve_times_;
    int iteration_count_ = 0;

    std::map<int, std::map<std::string, double>> custom_per_obstacle_weights_;
    std::vector<double> certificate_radii_;
    bool use_current_scenarios_next_ = false;
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_MPC_CONTROLLER_HPP
