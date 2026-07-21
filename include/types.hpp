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

#ifndef DRO_MPC_TYPES_HPP
#define DRO_MPC_TYPES_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <optional>
#include <cmath>

namespace dro_mpc {

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
    TEMPERATURE,      ///< Temperature-scaled frequency: w'_m = exp(log(w_m)/T)
    EPSILON_GREEDY    ///< Epsilon-greedy: w'_m = (1-eps)*w_m + eps/M
};

/**
 * @brief Linearized collision avoidance constraint.
 *
 * Form: a^T @ p_ego >= b
 *
 * Built from a fixed numerical half-space
 *     n^T c_d <= n^T x_obs - R
 * via a = -n and b = -upper_bound. The normal is frozen before the QP solve.
 */
struct CollisionConstraint {
    int k;                    ///< Timestep index
    int obstacle_id;          ///< Obstacle this constraint is for
    int scenario_id;          ///< Scenario this constraint belongs to
    Eigen::Vector2d a;        ///< Constraint normal vector (2,)
    double b;                 ///< Constraint offset (scalar)
    Eigen::Vector2d linearization_point = Eigen::Vector2d::Zero();  ///< Ego disc position at linearization
    int disc_index = 0;       ///< Disc used for this constraint (Case B Jacobian)
    double disc_offset = 0.0; ///< Longitudinal disc offset ℓ_d used at linearization

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

/**
 * @brief Ground cost type for the Wasserstein ball transport matrix.
 */
enum class DROGroundCostType {
    W1_METRIC,      ///< Gaussian W1 metric
    W2_BURES,       ///< Gaussian W2 Bures metric (default)
    ZERO_ONE,       ///< D[i][j] = (i!=j) ? 1 : 0
    EUCLIDEAN_MEAN  ///< ||mu_i - mu_j|| averaged over horizon
};

/**
 * @brief Which risk functional the per-mode risk score r[m] reports.
 *
 * The SURROGATE_* family is the CDC'26 formulation: the violation is LINEARISED
 * (projected on the ego->obstacle mean direction, which by Cauchy-Schwarz upper
 * -bounds the true Euclidean violation, hence "conservative"), evaluated per
 * (step, disc), and aggregated with a max. That max is NOT a risk measure of the
 * trajectory: max_k VaR(V_k) <= VaR(max_k V_k). It is a per-step score under a
 * planner that enforces a JOINT bound -- the two disagree about what "risk" means.
 *
 * Two errors run in OPPOSITE directions and partially cancel:
 *   - linearisation makes the surrogate too LARGE (Vtil >= V pointwise),
 *   - max-of-marginals makes it too SMALL (max_k VaR <= VaR of max_k).
 * Which wins is scenario-dependent. For iid steps the aggregation gap is big
 * (~1.9x at 15 steps), but a real rollout is strongly correlated through A, so the
 * effective number of independent steps is far below N_s and the gap collapses.
 * Measured on the canonical 6-mode scenario, linearisation dominates and the
 * surrogate sits ~2-5% ABOVE the true joint VaR -- i.e. conservative, the safe
 * direction. Do not assume that sign holds elsewhere; it is not a theorem.
 *
 * The JOINT_* family fixes both: it is the true risk measure of the joint-horizon
 * EUCLIDEAN collision violation
 *
 *     V := max_{k=1..N_s} max_d [ R - ||x_k - c_{d,k}|| ]_+ ,
 *
 * with x_k sampled from the mode's own linear-Gaussian rollout
 * x_{k+1} = A x_k + b + G w_k, so the temporal correlation induced by A is
 * carried exactly. No linearisation, no per-step decoupling. There is no closed
 * form (the distance is non-Gaussian and the steps are dependent), so it is
 * estimated by Monte Carlo with common random numbers across modes.
 */
enum class DRORiskMeasure {
    SURROGATE_VAR,   ///< DEFAULT, bit-for-bit master/CDC'26: per-step linearised VaR, max over (k,d)
    SURROGATE_CVAR,  ///< per-step linearised CVaR, correct clamp order (closed form), max over (k,d)
    SURROGATE_VAR_BONFERRONI,  ///< per-step linearised VaR at the union-bound-corrected level (see below)
    JOINT_VAR,       ///< joint-horizon VaR of Euclidean collision over the whole horizon (MC)
    JOINT_CVAR       ///< joint-horizon CVaR of Euclidean collision over the whole horizon (MC)
};

/*
 * SURROGATE_VAR_BONFERRONI -- the only CLOSED-FORM option that actually carries a
 * joint-horizon guarantee.
 *
 * Plain SURROGATE_VAR has none. max_k VaR_a(V_k) is a max of per-step quantiles; the
 * trajectory violates if ANY step does, so the tails union and the per-step level
 * buys much less jointly (15 steps at 95% each can be ~46% jointly). It sits above
 * the true joint VaR on the canonical scenario only because the linearisation slack
 * happens to exceed the aggregation deficit -- an accident of that geometry, not a
 * bound. Tighten the linearisation and the sign flips.
 *
 * Inflating the per-step level to alpha' = 1 - (1-alpha)/(N_s*D) repairs it in one
 * line, with t := max_{k,d} VaR_{alpha'}(Vtil_{k,d}):
 *
 *   P[max_{k,d} V_{k,d} > t] <= sum_{k,d} P[V_{k,d} > t]     (union bound)
 *                            <= sum_{k,d} P[Vtil_{k,d} > t]  (V <= Vtil, Cauchy-Schwarz)
 *                            <= N_s*D*(1-alpha') = 1-alpha.
 *
 * So t upper-bounds the true joint-horizon VaR of EUCLIDEAN collision -- for free,
 * no Monte Carlo, same cost as SURROGATE_VAR. This is the risk-allocation argument
 * de Groot 2023 uses to bound joint collision probability, applied to the risk score
 * rather than the constraint, which closes the per-step/joint mismatch between the
 * two. Verified against 1M-rollout ground truth: dominates on every mode.
 *
 * Cost: conservatism. z: 1.645 -> 2.713 at N_s=15, D=1, alpha=0.95, and the level
 * loosens linearly in N_s*D (D=4 discs -> z ~ 2.94).
 */

/**
 * @brief Symmetric Dirichlet prior on a categorical distribution.
 *
 * The pseudocount `a` gives the posterior-predictive mean
 *
 *     p_m = (n_m + a) / sum_j (n_j + a),
 *
 * and a > 0 is what prevents a known-but-unobserved mode from receiving exactly
 * zero mass. That matters more than it looks: sampling S times from a
 * distribution that assigns 0 to mode m yields mode m exactly 0 times for EVERY
 * S, so a zero there is unrecoverable by any scenario budget.
 *
 * There is no universally minimax choice of `a`, so this is an explicit modelling
 * decision rather than a default to be asserted:
 *
 *  - LAPLACE (a = 1). Optimal when the true distribution is drawn from a uniform
 *    prior on the simplex (Dirichlet(1)).
 *  - KRICHEVSKY_TROFIMOV (a = 1/2). Jeffreys prior; ASYMPTOTICALLY minimax for
 *    CUMULATIVE regret, and optimal under a Dirichlet(1/2) prior.
 *    Krichevsky & Trofimov, "The performance of universal encoding",
 *    IEEE Trans. Inf. Theory, 1981.
 *  - PERKS (a = 1/M). Keeps the total pseudocount mass at 1 independent of the
 *    mode-set size, so the prior does not strengthen as M grows.
 *
 * IMPORTANT (Orlitsky & Suresh, "Competitive Distribution Estimation: Why is
 * Good-Turing Good", NeurIPS 2015, arXiv:1503.07940): NEITHER Laplace nor
 * Krichevsky-Trofimov is asymptotically minimax over the full range of possible
 * sequences. Any claim of the form "a = 1 is the principled default" is false.
 * Whichever is chosen, the paper must name the criterion it is optimal under.
 */
enum class DirichletPrior {
    LAPLACE,              ///< a = 1      (uniform prior on the simplex)
    KRICHEVSKY_TROFIMOV,  ///< a = 1/2    (Jeffreys; asymptotically minimax, cumulative regret)
    PERKS                 ///< a = 1/M    (total prior mass 1, M-invariant)
};

/**
 * @brief Prior hyperparameters for the Bayesian mode-belief estimator.
 *
 * Shared vocabulary: config.hpp names these to forward them into the samplers;
 * mode_weights.hpp consumes them. Both already include types.hpp.
 *
 * Design note -- why the transition matrix and not the marginal:
 *   Each ROW of the transition matrix is conditionally multinomial given the
 *   current mode, so a Dirichlet prior per row is exactly conjugate. Smoothing
 *   the MARGINAL counts is not on that footing: a mode-observation history is a
 *   single trajectory of a Markov chain, not i.i.d. draws, so the marginal
 *   estimator is mis-specified for a switching process. The transition matrix is
 *   the object the data actually identifies.
 *
 * Design note -- why self_persistence_prior (theta) and NOT a raw sticky bonus:
 *   The sticky-HMM self-transition bias adds kappa to the diagonal of the row
 *   prior: alpha_ij = alpha + kappa * 1{i=j}. Fox, Sudderth, Jordan & Willsky
 *   ("A sticky HDP-HMM with application to speaker diarization", Annals of
 *   Applied Statistics 5(2A), 2011, arXiv:0905.2592) parameterise this by the
 *   RATIO theta = kappa/(alpha+kappa) -- the prior mean of the transition-matrix
 *   diagonal -- and LEARN it (Gamma priors on alpha+kappa), rather than fixing
 *   kappa. Raw kappa is the wrong knob for two reasons:
 *
 *     1. It is not M-invariant. With the finite row prior above,
 *            E[T_ii] = (alpha + kappa) / (M*alpha + kappa),
 *        so a fixed kappa silently changes meaning as the mode set grows. The
 *        previous hardcoded (alpha=1, kappa=2) gives E[T_ii] = 0.60 at M=3 but
 *        0.43 at M=5.
 *     2. It has no physical reading. theta does: expected dwell time in a mode
 *        is 1/(1 - theta) steps, which is directly comparable to the obstacle's
 *        true switching rate. The old (alpha=1, kappa=2) at M=5 encodes a dwell
 *        of ~1.75 steps; against a simulator with switch_prob = 0.2 (dwell 5)
 *        that prior actively fights the data.
 *
 *   So theta is the stated modelling assumption and kappa is DERIVED from it:
 *        kappa = alpha * (theta*M - 1) / (1 - theta),
 *   which inverts E[T_ii] = theta exactly. Requires theta > 1/M for kappa > 0;
 *   theta = 1/M gives kappa = 0 (uniform row prior, no stickiness).
 */
struct ModeBeliefConfig {
    /// Which symmetric Dirichlet prior to use for the initial belief AND for the
    /// per-row transition prior. Default KRICHEVSKY_TROFIMOV: it is the choice
    /// with a named optimality criterion (asymptotic minimax, cumulative regret).
    DirichletPrior prior = DirichletPrior::KRICHEVSKY_TROFIMOV;

    /// theta: prior mean of the transition-matrix diagonal, E[T_ii]. Equivalently
    /// the prior probability a mode persists one more step; expected dwell time is
    /// 1/(1 - theta). Set from the application's known/assumed switching rate:
    /// theta = 1 - switch_prob. Values <= 1/M disable the sticky bias (kappa = 0).
    /// Default 0.0 => no stickiness, i.e. no unstated assumption is smuggled in.
    double self_persistence_prior = 0.0;

    /// Symmetric Dirichlet pseudocount `a` for a mode set of size M.
    double alpha(int num_modes) const {
        const int M = (num_modes > 0) ? num_modes : 1;
        switch (prior) {
            case DirichletPrior::LAPLACE:             return 1.0;
            case DirichletPrior::KRICHEVSKY_TROFIMOV: return 0.5;
            case DirichletPrior::PERKS:               return 1.0 / static_cast<double>(M);
        }
        return 0.5;
    }

    /// Sticky self-transition pseudocount kappa, DERIVED from theta so that
    /// E[T_ii] = (alpha + kappa)/(M*alpha + kappa) = self_persistence_prior.
    /// Returns 0 when theta <= 1/M (no stickiness) or theta >= 1 (degenerate).
    double kappa(int num_modes) const {
        const int M = (num_modes > 0) ? num_modes : 1;
        const double theta = self_persistence_prior;
        const double uniform_diag = 1.0 / static_cast<double>(M);
        if (!(theta > uniform_diag) || theta >= 1.0) return 0.0;
        const double a = alpha(M);
        return a * (theta * static_cast<double>(M) - 1.0) / (1.0 - theta);
    }
};

}  // namespace dro_mpc

#endif  // SCENARIO_MPC_TYPES_HPP
