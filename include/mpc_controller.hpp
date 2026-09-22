/**
 * @file mpc_controller.hpp
 * @brief Scenario MPC controller.
 *
 * Main control loop that:
 * 1. Samples scenarios from obstacle predictions
 * 2. Computes linearized collision constraints
 * 3. Solves the scenario-constrained optimization
 * 4. Updates mode histories with observations
 * 5. Prunes inactive scenarios
 */

#ifndef DRO_MPC_MPC_CONTROLLER_HPP
#define DRO_MPC_MPC_CONTROLLER_HPP

#include "types.hpp"
#include <map>
#include "config.hpp"
#include "dynamics.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "collision_constraints.hpp"
#include "qp_solver.hpp"
#include "dro.hpp"
#include "reference_path.hpp"
#include <random>
#include <chrono>
#include <optional>
#include <vector>
#include <set>
#include <limits>

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
 * @brief Model Predictive Controller.
 *
 * Implements MPC controller.
 */
class MPCController {
public:
    /**
     * @brief Initialize the MPC controller.
     * @param config Configuration parameters
     */
    explicit MPCController(const RuntimeConfig& config);

    /**
     * @brief Initialize mode history for a new obstacle.
     * @param obstacle_id Unique obstacle identifier
    * @param obstacle_class_id Obstacle class identifier
     * @param available_modes Optional custom modes (uses defaults if empty)
     */
    void initialize_obstacle(
        int obstacle_id,
        int obstacle_class_id,
        const std::map<std::string, ModeModel>& available_modes = {}
    );

    /**
     * @brief Record a mode observation for an obstacle.
     * @param obstacle_id Obstacle identifier
    * @param obstacle_class_id Obstacle class identifier
     * @param observed_mode Observed mode ID
     * @param timestep Optional timestep (uses iteration count if -1)
     */
    void update_mode_observation(
        int obstacle_id,
        int obstacle_class_id,
        const std::string& observed_mode,
        int timestep = -1
    );


    /**
     * @brief Solve the MPC problem.
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

    /// Optional diagnostic copy of retained disc-space half-spaces supplied to optimization.
    void set_capture_linearized_constraints(bool enabled) { capture_linearized_constraints_ = enabled; }
    const std::vector<CollisionConstraint>& last_linearized_constraints() const {
        return last_linearized_constraints_;
    }

    /// Get current scenarios
    const std::vector<Scenario>& scenarios() const { return scenarios_; }

    /// Per-obstacle DRO diagnostics produced by the most recent solve.
    const std::map<int, DROResult>& last_dro_results() const {
        return last_dro_results_;
    }

    /// Get DRO module (for diagnostics)
    const DRO& dro() const { return dro_; }

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

    /// Update a mode model's dynamics parameters (b, G) for all obstacles.
    void update_mode_model(const std::string& mode_id,
                           const Eigen::Vector4d& b_new,
                           const Eigen::Matrix4d& G_new);

  
    const std::set<int>& last_removed_scenario_ids() const {
        return last_removed_scenario_ids_;
    }

    enum class HomotopySide {
        Auto = 0,
        Left = 1,
        Right = -1,
        Path = 2
    };

    struct HomotopyRecoveryCandidate {
        bool success = false;

        HomotopySide side =
            HomotopySide::Auto;

        std::vector<EgoState> anchor;
        std::vector<CollisionConstraint> constraints;

        std::vector<EgoInput> inputs;
        std::vector<EgoState> trajectory;

        double score =
            std::numeric_limits<double>::infinity();
    };


private:
    // One complete solve, including the existing homotopy/braking recovery.
    MPCResult solve_attempt(const EgoState& ego_state,
        const std::map<int, ObstacleState>& obstacles,
        const Eigen::Vector2d& goal, double reference_velocity,
        double path_progress, double path_length, bool use_dro);

    // Standalone counterfactual tool may copy pre-decision state into isolated
    // controllers. It never mutates the live controller or its RNG.
    friend class CounterfactualProbe;

    bool capture_linearized_constraints_ = false;
    std::vector<CollisionConstraint> last_linearized_constraints_;

    // ------------------------------------------------------------------------
    // Recursive-feasibility state
    // ------------------------------------------------------------------------

    /// Exact input sequence returned by the previous feasible MPC solve.
    std::vector<EgoInput> last_feasible_controls_;

    /// True once a complete dynamically feasible input sequence is available.
    bool has_feasible_backup_ = false;

    /// Scenarios removed in the most recent solve to preserve the shifted
    /// feasible candidate. These count toward Safe-Horizon support.
    std::set<int> last_removed_scenario_ids_;

    /**
     * @brief Initialize reference trajectory for constraint linearization.
     * Uses previous solution shifted forward, or straight-line to goal.
     */
    void initialize_reference_trajectory(
        const EgoState& ego_state,
        const Eigen::Vector2d& goal,
        double reference_velocity
    );

    /// Build the cold-start numerical trajectory used by the MPC linearization.
    std::vector<EgoState> generate_straight_line_trajectory(
        const EgoState& start,
        const Eigen::Vector2d& goal,
        double reference_velocity
    );

    MPCResult solve_optimization_sqp(
    const EgoState& ego_state,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    const std::vector<CollisionConstraint>& constraints,
    const std::set<int>& pre_support_scenarios,
    const std::vector<EgoInput>& feasible_warmstart_inputs,
    double path_progress = -1.0,
    double path_length = -1.0
    );

    /**
     * @brief Build condensed QP subproblem for SQP iteration.
     *
     * Condenses dynamics to express positions as linear function of inputs,
     * then maps collision constraints into input space. MPCC contouring/lag
     * objectives and Safe-Horizon collision constraints are applied over all
     * steps 1..N.  Safe Horizon is a support-bound certificate, not a
     * temporal constraint filter.
     *
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
     * @brief Generate safe fallback trajectory (gentle braking).
     */
    MPCResult generate_safe_fallback(const EgoState& ego_state);

    bool trajectory_is_deterministically_admissible(
    const std::vector<EgoState>& trajectory,
    const std::vector<EgoInput>& inputs
    ) const;  
    
    bool project_onto_linearized_set(
        const Eigen::VectorXd& target,
        const Eigen::MatrixXd& C,
        const Eigen::VectorXd& d,
        const Eigen::VectorXd& lb,
        const Eigen::VectorXd& ub,
        Eigen::VectorXd& projection
    );

    bool douglas_rachford_control_projection(
        const QPProblem& full_qp,
        int num_collision_constraints,
        Eigen::VectorXd& projected_delta_u
    );

    bool restore_dynamically_feasible_plan(
        const EgoState& ego_state,
        const Eigen::Vector2d& goal,
        double reference_velocity,
        const std::vector<CollisionConstraint>& constraints,
        const std::vector<EgoInput>& seed_inputs,
        std::vector<EgoInput>& restored_inputs,
        std::vector<EgoState>& restored_trajectory,
        double path_progress,
        double path_length
    );

    struct ConflictInfo {
        int first_k = -1;
        int last_k = -1;
        int obstacle_id = -1;
    };

    ConflictInfo find_backup_conflict_window(
        const std::vector<EgoState>& backup,
        const std::vector<Scenario>& scenarios,
        double combined_radius
    ) const;

    std::vector<EgoState> make_homotopy_seed(
        const std::vector<EgoState>& backup,
        int conflict_obstacle_id,
        HomotopySide side,
        int first_conflict_k,
        int last_conflict_k,
        double combined_radius
    ) const;

    double score_homotopy_recovery(
        const std::vector<EgoState>& trajectory,
        const std::vector<EgoInput>& inputs,
        const std::vector<EgoInput>& backup_inputs
    ) const;

    std::vector<EgoInput> make_braking_seed_inputs(
    const EgoState& ego_state,
    const std::vector<EgoInput>& backup_inputs,
    const std::vector<EgoState>& backup_trajectory
    ) const;

    RuntimeConfig config_;
    EgoDynamics ego_dynamics_;
    AcadosQPSolver qp_solver_;
    std::map<std::string, ModeModel> default_modes_;
    std::map<int, ModeHistory> mode_histories_;
    std::map<int, int> obstacle_class_ids_;  // obstacle_id -> obstacle_class_id
    std::vector<Scenario> scenarios_;
    DRO dro_;
    std::map<int, DROResult> last_dro_results_;
    std::vector<EgoState> reference_trajectory_;
    std::mt19937 rng_;
    std::vector<double> solve_times_;
    int iteration_count_ = 0;

    std::optional<ReferencePath> reference_path_;

    /// Custom per-obstacle mode weights (set externally, e.g. from OT predictor).
    std::map<int, std::map<std::string, double>> custom_per_obstacle_weights_;

};

}  // namespace dro_mpc

#endif  // DRO_MPC_MPC_CONTROLLER_HPP
