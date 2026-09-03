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
 * [x, y, theta, v] integrated via RK4.
 * Spline parameter s (arc length along reference path) is updated
 * algebraically after integration.
 */
struct EgoState {
    double x;      // Position x-coordinate [m]
    double y;      // Position y-coordinate [m]
    double theta;  // Heading angle [rad]
    double v;      // Velocity magnitude [m/s]
    double s;      // Spline/arc-length parameter along reference path [m] (-1 = not tracking)

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
 * @brief Ego vehicle control input: u = (a, omega)
 */
struct EgoInput {
    double a;      // Acceleration [m/s^2]
    double omega;  // Angular velocity [rad/s]

    EgoInput() : a(0), omega(0) {}
    EgoInput(double a, double omega) : a(a), omega(omega) {}

    /// Convert to Eigen vector [a, omega]
    Eigen::Vector2d to_array() const {
        return Eigen::Vector2d(a, omega);
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
    double x;   // Position x-coordinate [m]
    double y;   // Position y-coordinate [m]
    double vx;  // Velocity x-component [m/s]
    double vy;  // Velocity y-component [m/s]

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
    std::string mode_id;          // Unique identifier for this mode
    Eigen::Matrix4d A;            // State transition matrix (4x4)
    Eigen::Vector4d b;            // Bias/drift vector (4,)
    Eigen::MatrixXd G;            // Process noise matrix (4 x n_noise)
    std::string description;      // Human-readable description
    double body_lateral_displacement = 0.0;  // Body-frame lateral displacement [m/step]

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
        if (body_lateral_displacement != 0.0) {
            const double speed = std::hypot(x(2), x(3));
            const Eigen::Vector2d lateral = speed > 1e-9
                ? Eigen::Vector2d(-x(3), x(2)) / speed
                : Eigen::Vector2d(0.0, 1.0);
            x_next(0) += body_lateral_displacement * lateral(0);
            x_next(1) += body_lateral_displacement * lateral(1);
        }
        if (noise != nullptr) {
            x_next += G * (*noise);
        }
        return ObstacleState::from_array(x_next);
    }

    /// Dimension of process noise
    int noise_dim() const {
        return static_cast<int>(G.cols());
    }

    /// Propagate full state covariance Sigma_x with the discrete Lyapunov recursion.
    /// PredictionStep stores only Sigma_x's position block.
    void propagate_covariance(Eigen::Matrix4d& state_covariance) const {
        state_covariance = A * state_covariance * A.transpose() + G * G.transpose();
    }
};

// =============================================================================
// Section 4: Mode History and Weights
// =============================================================================

/**
 * @brief Track observed modes for an obstacle over time.
 */
struct ModeHistory {
    int obstacle_id;                                    // Unique obstacle identifier
    int obstacle_class_id = 0;                          // Class identifier (shared across obstacles)
    std::map<std::string, ModeModel> available_modes;   // Mode ID to ModeModel
    std::vector<std::pair<int, std::string>> observed_modes;  // (timestep, mode_id)
    int max_history_length = 100;                       // Maximum history length

    ModeHistory() : obstacle_id(0) {}
    ModeHistory(int obstacle_id, const std::map<std::string, ModeModel>& modes,
                int obstacle_class_id = 0)
            : obstacle_id(obstacle_id), obstacle_class_id(obstacle_class_id),
          available_modes(modes) {}

    /// Record a mode observation at the given timestep
    void record_observation(int timestep, const std::string& mode_id) {
        observed_modes.emplace_back(timestep, mode_id);
        // Trim history if too long
        if (static_cast<int>(observed_modes.size()) > max_history_length) {
            observed_modes.erase(observed_modes.begin(),
            observed_modes.begin() + (observed_modes.size() - max_history_length));
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
    int k;                        // Timestep index
    Eigen::Vector2d mean;         // Mean position [x, y]
    Eigen::Matrix2d covariance;   // Stored position covariance Sigma_pos (2x2)

    PredictionStep() : k(0), mean(Eigen::Vector2d::Zero()),
                       covariance(Eigen::Matrix2d::Zero()) {}
    PredictionStep(int k, const Eigen::Vector2d& mean, const Eigen::Matrix2d& cov)
        : k(k), mean(mean), covariance(cov) {}
};

/**
 * @brief Predicted trajectory for a single obstacle over the horizon.
 */
struct ObstacleTrajectory {
    int obstacle_id;                   // Unique obstacle identifier
    std::string mode_id;               // Mode used for this trajectory
    std::vector<PredictionStep> steps; // Prediction steps over horizon

    ObstacleTrajectory() : obstacle_id(0) {}
    ObstacleTrajectory(int obstacle_id, const std::string& mode_id,
                       const std::vector<PredictionStep>& steps)
        : obstacle_id(obstacle_id), mode_id(mode_id), steps(steps) {}

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
    int scenario_id;                                      // Unique scenario identifier
    std::map<int, ObstacleTrajectory> trajectories;       // obstacle_id -> trajectory
    
    Scenario() : scenario_id(0) {}
    Scenario(int scenario_id, const std::map<int, ObstacleTrajectory>& trajectories)
        : scenario_id(scenario_id), trajectories(trajectories) {}

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
 * @brief Linearized collision avoidance constraint.
 *
 * Form: a^T @ p_ego >= b
 *
 * Built from a fixed numerical half-space
 *     n^T c_d <= n^T x_obs - R
 * via a = -n and b = -upper_bound. The normal is frozen before the QP solve.
 */
struct CollisionConstraint {
    int k;                    // Timestep index
    int obstacle_id;          // Obstacle this constraint is for
    int scenario_id;          // Scenario this constraint belongs to
    Eigen::Vector2d a;        // Constraint normal vector (2,)
    double b;                 // Constraint offset (scalar)
    Eigen::Vector2d linearization_point = Eigen::Vector2d::Zero();  // Ego disc position at linearization
    int disc_index = 0;       // Disc used for this constraint (Case B Jacobian)
    double disc_offset = 0.0; // Longitudinal disc offset ℓ_d used at linearization

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
    bool success;                           // Whether optimization succeeded
    std::vector<EgoState> ego_trajectory;   // Planned ego states over horizon
    std::vector<EgoInput> control_inputs;   // Planned control inputs
    std::vector<int> active_scenarios;      // Scenarios with binding constraints
    double solve_time = 0.0;                // Optimization solve time [s]
    double cost = std::numeric_limits<double>::infinity();  // Optimal cost value
    int safe_horizon = -1;              // Truncated safe horizon used (-1 = full)
    double constraint_construction_time = 0.0;  // Time for constraint building [s]
    double qp_solve_time = 0.0;                 // Time for QP/SQP solve [s]
    int num_dro_injected = 0;           // Always 0; DRO now resamples from q* only
    /// Largest ambiguity radius used across obstacles during this solve.
    double ambiguity_radius_used = 0.0;

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
 * @brief Ground cost D[i][j] between two MODES for the Wasserstein ball.
 * No intrinsic distance between the labels "turn_left" and "decelerating" 
 * Options below give one by embedding mode m into space of predicted
 * distributions, m ↦ {N(mu^m_k, Sigma^m_k)}_k, and measuring distance there.
 * Modes are close when they imply similar futures.
 * All options collapse the horizon the same way, D_ij = (1/N) sum_k d(P^i_k, P^j_k), which is safe because
 * a mean of metrics is a metric.
 *
 *   W2_BURES (default): d = W2 between the two position Gaussians,
 *     W2^2 = ||mu_i - mu_j||^2 + Bures^2(Sigma_i, Sigma_j), with
 *     Bures^2 = tr Si + tr Sj - 2 tr((Si^1/2 Sj Si^1/2)^1/2). The stored value is
 *     the SQUARE ROOT — the metric, not the squared cost, which is not a metric.
 *     Closed form, deterministic, and the canonical metric between Gaussians.
 *
 *   W1_METRIC: d = normalised SLICED W1. Multivariate W1 has no closed form
 *     for dim >= 2, so project both Gaussians onto directions theta (where
 *     the 1D W1 is closed form, a folded-normal mean) and average. Scaled so equal covariances give exactly ||mu_i - mu_j||,
 *     matching the others. Relative to W2_BURES it penalises covariance
 *     mismatch LESS, since W1 grows linearly rather than quadratically in the
 *     displacement.
 *
 *   EUCLIDEAN_MEAN: d = ||mu_i - mu_j||. Mean geometry only;. Exactly equals W2_BURES and W1_METRIC whenever the
 *     modes share a covariance trajectory, which is the case whenever they differ
 *     only through the affine offset b (b does not enter the covariance recursion).
 *
 *   ZERO_ONE: D_ij = (i != j). The discrete metric ⇒ W_D = total variation. Every
 *     mode equidistant, so the geometry of the mode set is discarded and the worst
 *     case moves mass to the highest-risk mode regardless of dynamic plausibility.
 *     
 */
enum class DROGroundCostType {
    W1_METRIC,      // Normalised sliced-W1 between mode Gaussians (least pessimistic metric)
    W2_BURES,       // W2 Bures distance between mode Gaussians (default); OUTER ball is still W1
    ZERO_ONE,       // D[i][j] = (i!=j) ? 1 : 0  ⇒  W_D = total variation
    EUCLIDEAN_MEAN  // ||mu_i - mu_j|| averaged over horizon (mode-to-mode)
};

/**
 * @brief Divergence-based ambiguity families over the discrete mode simplex.
 *
 * Table I of Schuurmans & Patrinos, "A General Framework for Learning-Based
 * Distributionally Robust MPC of Markov Jump Systems," arXiv:2106.00561 (2021).
 * Each family defines the ambiguity ball A = {p ∈ Δ_d : D(p̂,p) ≤ r} through a
 * distance D(p̂,p), a FINITE-SAMPLE radius r(m,β) guaranteeing p⋆ ∈ A w.p. ≥ 1−β,
 * and a conic reformulation of the worst-case expectation max_{p∈A} ⟨p, r_risk⟩:
 *
 *   family            D(p̂,p)                            radius r(m,β)   cone
 *   TOTAL_VARIATION   ‖p−p̂‖₁                            2√r_TV         linear
 *   KULLBACK_LEIBLER  D_KL(p̂,p)=Σ p̂_i log(p̂_i/p_i)      r_KL           exponential
 *   JENSEN_SHANNON    ½[D_KL(p̂,M)+D_KL(p,M)], M=(p̂+p)/2  ½ r_KL        exponential
 *   HELLINGER         Σ(√p_i−√p̂_i)²                      r_KL           quadratic
 *   WASSERSTEIN       min_Π{Σ Π_ij K_ij : Π1=p, Πᵀ1=p̂}   maxK·√r_TV    linear
 *
 * with  r_TV(m,β) = (d·ln2 − ln β)/(2m)   [Bretagnolle–Huber–Carol, Eq. 12]
 *       r_KL(m,β) = (d·ln m − ln β)/m     [method of types,          Eq. 13]
 * where m = mode-observation count, d = #modes, K_ij = dist(i,j).
 *
 * WHY TV AND WASSERSTEIN SHARE A RADIUS: the calibrated true-W1 radius used
 * elsewhere in this codebase IS the WASSERSTEIN row — it is the TV concentration
 * r_TV lifted to W₁ by the transport diameter, since
 *   maxK·√r_TV = ½·diam(K)·(2√r_TV) = ½·diam(K)·ε_n(β).
 * So TOTAL_VARIATION and WASSERSTEIN rest on the SAME BHC concentration and differ
 * only by the ½·diam(K) diameter scaling; W₁ = TV × ½·diam(K). All five families
 * are implemented in schuurmans_ambiguity.hpp.
 */
enum class AmbiguityDivergence {
    TOTAL_VARIATION,   // ‖p−p̂‖₁; radius 2√r_TV; linear cone
    KULLBACK_LEIBLER,  // D_KL(p̂,p); radius r_KL; exponential cone
    JENSEN_SHANNON,    // symmetrized KL; radius ½ r_KL; exponential cone
    HELLINGER,         // Σ(√p−√p̂)²; radius r_KL; quadratic cone
    WASSERSTEIN        // W₁ with ground cost K; radius maxK·√r_TV; linear cone (DEFAULT)
};

/**
 * @brief Which risk functional the per-mode risk score r[m] reports.
 *
 */
 
// Three families, in increasing fidelity (and cost):
//  SURROGATE_* — LINEARISED per-step violation: the true Euclidean violation
//    [R-||x_k-c||]_+ is replaced by its projection on the ego->obstacle MEAN
//    direction, modelled as Gaussian with directional sigma. By Cauchy-Schwarz this
//    projection is an UPPER bound of the true violation, so the surrogate is
//    closed-form (no Monte Carlo) and conservative. "Surrogate" names the
//    linearisation, not an invalid bound.
//  BONFERRONI_VAR — PROPER Bonferroni VaR: the union-bound level correction
//    a' = 1-(1-a)/(N_s*D) applied to the TRUE per-step EUCLIDEAN VaR (Monte Carlo
//    on each step's marginal Gaussian, no projection). Removes the Cauchy-Schwarz
//    slack of SURROGATE_VAR_BONFERRONI while keeping the per-step decoupling; still
//    a valid upper bound on the joint-horizon VaR (union bound).
//  MIXTURE_* — risk measure of the SEQUENCE-MIXTURE surrogate: conditional on a
//    sampled mode sequence the surrogate makes V Gaussian, so V is a K-component
//    Gaussian mixture and its VaR/CVaR are available semi-analytically (see below).
//    Coherent (CVaR) and JOINT over the mode chain, at K rollouts instead of K
//    noise-sampled rollouts -- the in-loop-affordable coherent option.
//  JOINT_* — the TRUE joint-horizon risk measure of V = max_{k,d}[R-||x_k-c||]_+
//    over the WHOLE horizon, x_k from the mode's correlated rollout (exact temporal
//    correlation), estimated by Monte Carlo.
enum class DRORiskMeasure {
    SURROGATE_VAR,   // per-step LINEARISED VaR, max over (k,d) (CDC'26 legacy)
    SURROGATE_CVAR,  // per-step LINEARISED CVaR, correct clamp order (closed form), max over (k,d)
    SURROGATE_VAR_BONFERRONI,  // LINEARISED VaR at the union-corrected level a'=1-(1-a)/(N_s*D)
    BONFERRONI_VAR,  // PROPER Bonferroni VaR: union correction on the TRUE per-step Euclidean VaR (MC)
    MIXTURE_VAR,     // VaR  of the sequence-mixture surrogate (semi-analytic, switching-aware)
    MIXTURE_CVAR,    // CVaR of the sequence-mixture surrogate (semi-analytic, COHERENT, switching-aware)
    JOINT_VAR,       // joint-horizon VaR of Euclidean collision over the whole horizon (MC)
    JOINT_CVAR       // joint-horizon CVaR of Euclidean collision over the whole horizon (MC)
};

/*
 * MIXTURE_* -- why it exists, and what it fixes.
 *
 * For a Markov-jump obstacle the per-mode score r[m] must be a risk measure of the
 * JOINT uncertainty (mode sequence, process noise) given a start in mode m. The
 * original switching estimator was
 *
 *     r[m] = E_seq[ max_{k,d} VaR_alpha^noise( Vtil_{k,d} | seq ) ],
 *
 * which is an EXPECTATION over sequences wrapped around a VaR over noise. That is
 * not a risk measure of the joint law, and it understates. For CVaR the direction
 * is provable from the dual representation CVaR_a(X) = max_{dQ/dP <= 1/(1-a)} E_Q[X]:
 * let Q_s attain the conditional max on sequence s and glue them,
 * dQ/dP := sum_s 1{seq=s} dQ_s/dP(.|s). That Q is globally feasible, so
 *
 *     E_seq[ CVaR_a(X | seq) ] = E_Q[X] <= CVaR_a(X).
 *
 * Averaging over sequences discards exactly the between-sequence tail: if 5% of
 * sequences are catastrophic (a sustained aggressive mode), the mean dilutes them
 * by 20x while CVaR_0.95 puts full weight on them.
 *
 * THE FIX (semi-analytic, this family). Conditional on a sampled sequence s the
 * surrogate gives V | s ~ N(mu_s, sigma_s^2) (the dominant (k,d) pair), so V is a
 * K-component Gaussian mixture with equal weights. Then
 *
 *   VaR:  solve  (1/K) sum_s Phi((q - mu_s)/sigma_s) = alpha   for q  (bisection;
 *         the mixture CDF is continuous and strictly increasing), then clamp [q]_+.
 *   CVaR: Rockafellar-Uryasev,  CVaR_a(Z) = min_q { q + E[(Z-q)_+]/(1-a) },
 *         attained at q* = VaR_a(Z). For Z = [V]_+ the minimiser q* >= 0, and for
 *         q >= 0 we have ([V]_+ - q)_+ = (V - q)_+, so the clamp is handled EXACTLY:
 *
 *           CVaR_a([V]_+) = q* + (1/(1-a)) * (1/K) sum_s E[(V_s - q*)_+],
 *           E[(X-q)_+] = (mu-q) Phi((mu-q)/sigma) + sigma phi((mu-q)/sigma).
 *
 *         No clamp-order bug is possible here (contrast cvar_clamped_gaussian, which
 *         is the K=1 special case of this formula).
 *
 * Taking the max over (k,d) INSIDE the conditional and the risk measure OUTSIDE is
 * also the correct order, so MIXTURE_* needs no Bonferroni level correction -- that
 * apparatus exists only to patch up max-of-marginals.
 *
 * DEGENERACIES (both checked by test_switching_risk):
 *   - transition = I (or absent) and K = 1  =>  a single Gaussian, so MIXTURE_VAR
 *     reduces to SURROGATE_VAR and MIXTURE_CVAR to SURROGATE_CVAR exactly.
 *   - JOINT_* with a transition matrix samples (sequence, noise) jointly and is the
 *     exact reference this family approximates; at transition = I it reduces to the
 *     held-mode JOINT_* estimator.
 *
 * COST. MIXTURE_* is K sequence rollouts + one 1-D bisection, with the noise handled
 * in closed form; JOINT_* needs joint_risk_samples noise-sampled rollouts per mode.
 * MIXTURE_* is the affordable coherent option; JOINT_* is the offline reference.
 *
 * RESIDUAL APPROXIMATION. MIXTURE_* inherits the surrogate's Cauchy-Schwarz slack
 * (conservative) and summarises V|s by its dominant (k,d) pair rather than the true
 * max of correlated Gaussians (anti-conservative). JOINT_* has neither.
 */

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
    LAPLACE,              // a = 1      (uniform prior on the simplex)
    KRICHEVSKY_TROFIMOV,  // a = 1/2    (Jeffreys; asymptotically minimax, cumulative regret)
    PERKS                 // a = 1/M    (total prior mass 1, M-invariant)
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

#endif  
