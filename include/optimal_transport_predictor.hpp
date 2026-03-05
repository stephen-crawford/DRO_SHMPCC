/**
 * @file optimal_transport_predictor.hpp
 * @brief Optimal Transport Predictor for Learning Obstacle Dynamics.
 *
 * Implements statistical learning of obstacle dynamics using optimal transport theory.
 * Key features:
 * - Sinkhorn algorithm for efficient Wasserstein distance computation
 * - Empirical distribution learning from observed trajectories
 * - Wasserstein barycenter computation for multi-modal predictions
 * - Adaptive uncertainty quantification based on distributional distance
 *
 * Reference: Computational Optimal Transport (Peyré & Cuturi, 2019)
 */

#ifndef SCENARIO_MPC_OPTIMAL_TRANSPORT_PREDICTOR_HPP
#define SCENARIO_MPC_OPTIMAL_TRANSPORT_PREDICTOR_HPP

#include "types.hpp"
#include <deque>
#include <set>
#include <functional>

namespace scenario_mpc {

// =============================================================================
// Trajectory Observation and Buffer
// =============================================================================

/**
 * @brief Single trajectory observation for an obstacle.
 */
struct TrajectoryObservation {
    int timestep;               ///< Timestep when observation was made
    Eigen::Vector2d position;   ///< Position [x, y]
    Eigen::Vector2d velocity;   ///< Velocity [vx, vy]
    Eigen::Vector2d acceleration; ///< Acceleration [ax, ay]
    std::string mode_id;        ///< Optional mode label

    TrajectoryObservation()
        : timestep(0), position(Eigen::Vector2d::Zero()),
          velocity(Eigen::Vector2d::Zero()),
          acceleration(Eigen::Vector2d::Zero()) {}

    TrajectoryObservation(int timestep, const Eigen::Vector2d& position,
                          const Eigen::Vector2d& velocity,
                          const Eigen::Vector2d& acceleration,
                          const std::string& mode_id = "")
        : timestep(timestep), position(position), velocity(velocity),
          acceleration(acceleration), mode_id(mode_id) {}

    /// Full state vector [x, y, vx, vy, ax, ay]
    Eigen::VectorXd state() const {
        Eigen::VectorXd s(6);
        s << position, velocity, acceleration;
        return s;
    }

    /// Dynamics state vector [x, y, vx, vy] for ModeModel compatibility
    Eigen::Vector4d dynamics_state() const {
        Eigen::Vector4d s;
        s << position, velocity;
        return s;
    }
};

/**
 * @brief Circular buffer storing observed trajectory snippets for an obstacle.
 *
 * Used to build empirical distributions of dynamics.
 */
class TrajectoryBuffer {
public:
    TrajectoryBuffer(int obstacle_id = 0, size_t max_length = 200)
        : obstacle_id_(obstacle_id), max_length_(max_length) {}

    /// Add a new observation to the buffer
    void add_observation(const TrajectoryObservation& obs) {
        observations_.push_back(obs);
        if (observations_.size() > max_length_) {
            observations_.pop_front();
        }
    }

    /// Get the n most recent observations
    std::vector<TrajectoryObservation> get_recent(size_t n) const {
        std::vector<TrajectoryObservation> recent;
        size_t start = observations_.size() > n ? observations_.size() - n : 0;
        for (size_t i = start; i < observations_.size(); ++i) {
            recent.push_back(observations_[i]);
        }
        return recent;
    }

    /// Get all velocity observations as Nx2 matrix
    Eigen::MatrixX2d get_velocity_samples() const {
        Eigen::MatrixX2d samples(observations_.size(), 2);
        for (size_t i = 0; i < observations_.size(); ++i) {
            samples.row(i) = observations_[i].velocity.transpose();
        }
        return samples;
    }

    /// Get all acceleration observations as Nx2 matrix
    Eigen::MatrixX2d get_acceleration_samples() const {
        Eigen::MatrixX2d samples(observations_.size(), 2);
        for (size_t i = 0; i < observations_.size(); ++i) {
            samples.row(i) = observations_[i].acceleration.transpose();
        }
        return samples;
    }

    /// Get all state observations as Nx6 matrix
    Eigen::MatrixXd get_state_samples() const {
        Eigen::MatrixXd samples(observations_.size(), 6);
        for (size_t i = 0; i < observations_.size(); ++i) {
            samples.row(i) = observations_[i].state().transpose();
        }
        return samples;
    }

    size_t size() const { return observations_.size(); }
    bool empty() const { return observations_.empty(); }
    int obstacle_id() const { return obstacle_id_; }

    /// Access observations for iteration
    const std::deque<TrajectoryObservation>& observations() const { return observations_; }

private:
    int obstacle_id_;
    size_t max_length_;
    std::deque<TrajectoryObservation> observations_;
};

// =============================================================================
// Empirical Distribution
// =============================================================================

/**
 * @brief Empirical probability distribution from samples.
 *
 * Supports discrete distributions with equal or weighted samples.
 * Used for optimal transport computations.
 */
class EmpiricalDistribution {
public:
    EmpiricalDistribution() : samples_(0, 2), weights_(0) {}

    /**
     * @brief Create empirical distribution from samples.
     *
     * @param samples N x d matrix of samples
     * @param weights Optional weights (uniform if not provided)
     */
    static EmpiricalDistribution from_samples(
        const Eigen::MatrixXd& samples,
        const Eigen::VectorXd& weights = Eigen::VectorXd());

    /// Number of samples
    int n_samples() const { return static_cast<int>(samples_.rows()); }

    /// Dimension of samples
    int dim() const { return n_samples() > 0 ? static_cast<int>(samples_.cols()) : 0; }

    /// Check if distribution has no samples
    bool is_empty() const { return n_samples() == 0; }

    /// Get samples matrix
    const Eigen::MatrixXd& samples() const { return samples_; }

    /// Get weights vector
    const Eigen::VectorXd& weights() const { return weights_; }

    /// Weighted mean of the distribution
    Eigen::VectorXd mean() const;

    /// Weighted covariance matrix
    Eigen::MatrixXd covariance() const;

private:
    Eigen::MatrixXd samples_;  ///< N x d matrix of samples
    Eigen::VectorXd weights_;  ///< N-dimensional weight vector (sums to 1)
};

// =============================================================================
// Ground Cost Types
// =============================================================================

/**
 * @brief Ground cost types for OT distance computation.
 */
enum class GroundCostType {
    SQUARED_EUCLIDEAN,  ///< ||v_i - v_j||^2 (default, full W2 geometry)
    MANHATTAN,          ///< L1 distance: |v_ix - v_jx| + |v_iy - v_jy|
    FLAT,               ///< All pairs cost 1.0 (no spatial geometry)
    MEAN_ONLY,          ///< Bypass OT, use L2 between distribution means
    DIRECTIONAL,        ///< Cosine distance: 1 - cos(v_i, v_j)
    RANDOM_PERMUTED,    ///< Randomly permute rows/cols of W2 cost matrix (breaks semantic alignment)
    CONSTANT            ///< All entries = constant value (degenerate geometry)
};

// =============================================================================
// Sinkhorn Algorithm for Optimal Transport
// =============================================================================

/**
 * @brief Compute pairwise cost matrix between source and target samples.
 *
 * @param source N x d array of source samples
 * @param target M x d array of target samples
 * @param p Power for cost (p=2 for squared Euclidean)
 * @return N x M cost matrix
 */
Eigen::MatrixXd compute_cost_matrix(
    const Eigen::MatrixXd& source,
    const Eigen::MatrixXd& target,
    int p = 2,
    GroundCostType cost_type = GroundCostType::SQUARED_EUCLIDEAN);

/**
 * @brief Result of Sinkhorn algorithm.
 */
struct SinkhornResult {
    Eigen::MatrixXd transport_plan;  ///< N x M transport plan matrix
    double sinkhorn_distance;         ///< Sinkhorn distance value
    int iterations;                   ///< Number of iterations used
    bool converged;                   ///< Whether algorithm converged
};

/**
 * @brief Sinkhorn-Knopp algorithm for entropy-regularized optimal transport.
 *
 * Solves: min_P <C, P> + epsilon * H(P)
 * s.t. P @ 1 = a, P.T @ 1 = b, P >= 0
 *
 * Reference: Cuturi, M. (2013) "Sinkhorn Distances"
 *
 * @param source_weights Source marginal (a)
 * @param target_weights Target marginal (b)
 * @param cost_matrix C[i,j] = cost of transporting source[i] to target[j]
 * @param epsilon Entropy regularization parameter
 * @param max_iterations Maximum Sinkhorn iterations
 * @param convergence_threshold Convergence criterion
 * @return SinkhornResult containing transport plan and distance
 */
SinkhornResult sinkhorn_algorithm(
    const Eigen::VectorXd& source_weights,
    const Eigen::VectorXd& target_weights,
    const Eigen::MatrixXd& cost_matrix,
    double epsilon = 0.1,
    int max_iterations = 100,
    double convergence_threshold = 1e-6);

/**
 * @brief Compute (regularized) Wasserstein distance between two distributions.
 *
 * W_p(mu, nu) = (inf_P sum_{i,j} C[i,j] * P[i,j])^(1/p)
 *
 * @param source Source distribution
 * @param target Target distribution
 * @param epsilon Sinkhorn regularization
 * @param p Wasserstein order (typically 2)
 * @return Wasserstein distance (or 0 if either distribution is empty)
 */
double wasserstein_distance(
    const EmpiricalDistribution& source,
    const EmpiricalDistribution& target,
    double epsilon = 0.1,
    int p = 2,
    GroundCostType cost_type = GroundCostType::SQUARED_EUCLIDEAN);

// =============================================================================
// Wasserstein Barycenter for Multi-Modal Predictions
// =============================================================================

/**
 * @brief Compute Wasserstein barycenter of multiple distributions.
 *
 * Finds: argmin_nu sum_i w_i * W_2(mu_i, nu)^2
 *
 * Uses iterative Bregman projection algorithm.
 *
 * Reference: Cuturi & Doucet (2014) "Fast Computation of Wasserstein Barycenters"
 *
 * @param distributions List of input distributions
 * @param weights Barycentric weights (should sum to 1)
 * @param n_support Number of support points in output
 * @param epsilon Sinkhorn regularization
 * @param max_iterations Maximum iterations
 * @param convergence_threshold Convergence criterion
 * @return Barycenter distribution
 */
EmpiricalDistribution wasserstein_barycenter(
    const std::vector<EmpiricalDistribution>& distributions,
    const std::vector<double>& weights,
    int n_support = 50,
    double epsilon = 0.1,
    int max_iterations = 50,
    double convergence_threshold = 1e-4);

// =============================================================================
// Mode Distribution
// =============================================================================

/**
 * @brief Weight computation strategies for OT predictor.
 */
enum class OTWeightType {
    WASSERSTEIN,  ///< Inverse Wasserstein distance weights
    LIKELIHOOD,   ///< Likelihood-based weights
    UNIFORM       ///< Equal weights
};

/**
 * @brief Distribution of dynamics for a specific mode.
 */
struct ModeDistribution {
    std::string mode_id;                      ///< Mode identifier
    EmpiricalDistribution velocity_dist;      ///< Velocity distribution
    EmpiricalDistribution acceleration_dist;  ///< Acceleration distribution
    int observation_count = 0;                ///< Number of observations
    int last_updated = 0;                     ///< Last update timestep

    ModeDistribution() = default;
    ModeDistribution(const std::string& mode_id,
                     const EmpiricalDistribution& vel_dist,
                     const EmpiricalDistribution& acc_dist,
                     int count = 0, int last = 0)
        : mode_id(mode_id), velocity_dist(vel_dist),
          acceleration_dist(acc_dist), observation_count(count),
          last_updated(last) {}
};

// =============================================================================
// Prediction Step for OT Predictor
// =============================================================================

/**
 * @brief Prediction step with elliptical uncertainty representation.
 */
struct OTPredictionStep {
    Eigen::Vector2d position;  ///< Mean position [x, y]
    double angle;              ///< Heading angle [rad]
    double major_radius;       ///< Major axis of uncertainty ellipse
    double minor_radius;       ///< Minor axis of uncertainty ellipse

    OTPredictionStep()
        : position(Eigen::Vector2d::Zero()), angle(0),
          major_radius(0.3), minor_radius(0.3) {}

    OTPredictionStep(const Eigen::Vector2d& pos, double angle,
                     double major_r, double minor_r)
        : position(pos), angle(angle), major_radius(major_r),
          minor_radius(minor_r) {}
};

// =============================================================================
// Optimal Transport Predictor
// =============================================================================

/**
 * @brief Optimal Transport-based predictor for obstacle dynamics.
 *
 * Key features:
 * 1. Learns empirical distributions of obstacle dynamics from observations
 * 2. Uses Wasserstein distance to compare predicted vs actual trajectories
 * 3. Computes Wasserstein barycenters for multi-modal predictions
 * 4. Provides adaptive uncertainty based on distributional distance
 *
 * Integration with AdaptiveModeSampler:
 * - Can provide OT-based mode weights as alternative to frequency/recency
 * - Validates mode predictions against observed dynamics
 * - Adjusts uncertainty based on distribution mismatch
 */
class OptimalTransportPredictor {
public:
    /**
     * @brief Initialize the Optimal Transport Predictor.
     *
     * @param dt Timestep in seconds
     * @param buffer_size Maximum observations per obstacle
     * @param sinkhorn_epsilon Regularization for Sinkhorn algorithm
     * @param min_samples_for_ot Minimum samples needed for OT computation
     * @param uncertainty_scale Scaling factor for uncertainty estimates
     * @param weight_type Strategy for computing mode weights
     */
    OptimalTransportPredictor(
        double dt = 0.1,
        size_t buffer_size = 200,
        double sinkhorn_epsilon = 0.1,
        int min_samples_for_ot = 10,
        double uncertainty_scale = 1.0,
        OTWeightType weight_type = OTWeightType::WASSERSTEIN,
        GroundCostType ground_cost = GroundCostType::SQUARED_EUCLIDEAN);

    /**
     * @brief Record an observation of obstacle state.
     *
     * @param obstacle_id Obstacle identifier
     * @param position Current position [x, y]
     * @param mode_id Optional mode label for the observation
     */
    void observe(int obstacle_id, const Eigen::Vector2d& position,
                 const std::string& mode_id = "");

    /// Advance the current timestep
    void advance_timestep() { ++current_timestep_; }

    /// Get current timestep
    int current_timestep() const { return current_timestep_; }

    /**
     * @brief Compute mode weights using optimal transport distance.
     *
     * Uses inverse Wasserstein distance from observed dynamics to mode
     * reference distributions.
     *
     * @param obstacle_id Obstacle identifier
     * @param available_modes List of available mode IDs
     * @param reference_velocity Optional recent velocity observation
     * @return Dictionary mapping mode_id to weight (normalized to sum to 1)
     */
    std::map<std::string, double> compute_mode_weights(
        int obstacle_id,
        const std::vector<std::string>& available_modes,
        const Eigen::Vector2d* reference_velocity = nullptr);

    /**
     * @brief Generate trajectory prediction using Wasserstein barycenter.
     *
     * Combines mode-specific predictions using OT barycenter for
     * smooth multi-modal prediction.
     *
     * @param obstacle_id Obstacle identifier
     * @param current_position Current [x, y] position
     * @param current_velocity Current [vx, vy] velocity
     * @param horizon Number of prediction steps
     * @param mode_weights Optional mode weights (computed if nullptr)
     * @return List of OTPredictionStep objects
     */
    std::vector<OTPredictionStep> predict_trajectory(
        int obstacle_id,
        const Eigen::Vector2d& current_position,
        const Eigen::Vector2d& current_velocity,
        int horizon,
        const std::map<std::string, double>* mode_weights = nullptr);

    /**
     * @brief Compute Wasserstein distance between predicted and actual trajectories.
     *
     * @param obstacle_id Obstacle identifier
     * @param predicted_trajectory List of predicted positions
     * @param actual_trajectory List of actual positions
     * @return Wasserstein distance between trajectories
     */
    double compute_prediction_error(
        int obstacle_id,
        const std::vector<Eigen::Vector2d>& predicted_trajectory,
        const std::vector<Eigen::Vector2d>& actual_trajectory);

    /**
     * @brief Compute adaptive uncertainty scaling based on prediction error.
     *
     * @param obstacle_id Obstacle identifier
     * @param prediction_error Recent Wasserstein prediction error
     * @return Uncertainty multiplier
     */
    double adapt_uncertainty(int obstacle_id, double prediction_error);

    /**
     * @brief Set reference distribution for a mode (prior knowledge).
     *
     * @param mode_id Mode identifier
     * @param velocity_samples N x 2 matrix of velocity samples
     * @param acceleration_samples Optional N x 2 matrix of acceleration samples
     */
    void set_reference_distribution(
        const std::string& mode_id,
        const Eigen::MatrixX2d& velocity_samples,
        const Eigen::MatrixX2d* acceleration_samples = nullptr);

    /// Get set of modes with learned distributions for an obstacle
    std::set<std::string> get_learned_modes(int obstacle_id) const;

    /// Get statistics for a mode's learned distribution
    std::optional<std::map<std::string, double>> get_mode_distribution_stats(
        int obstacle_id, const std::string& mode_id) const;

    /**
     * @brief Estimate mode dynamics (b, G) from observed trajectory data.
     *
     * Computes residuals: r_k = x_{k+1} - A_prior * x_k
     * Then b_learned = mean(r), G_learned = cholesky(cov(r) + reg*I)
     *
     * @param obstacle_id Obstacle identifier
     * @param mode_id Mode to estimate for
     * @param A_prior Prior state transition matrix (4x4)
     * @param dt Timestep
     * @return Optional pair (b_learned, G_learned), nullopt if insufficient data
     */
    std::optional<std::pair<Eigen::Vector4d, Eigen::Matrix4d>> estimate_mode_dynamics(
        int obstacle_id,
        const std::string& mode_id,
        const Eigen::Matrix4d& A_prior,
        double dt);

    /// Reset learned distributions (keep reference distributions)
    void reset();

    /// Reset all state including reference distributions
    void reset_all();

    /// Check if obstacle has trajectory buffer
    bool has_obstacle(int obstacle_id) const {
        return trajectory_buffers_.find(obstacle_id) != trajectory_buffers_.end();
    }

    /// Get number of observations for obstacle
    size_t get_observation_count(int obstacle_id) const {
        auto it = trajectory_buffers_.find(obstacle_id);
        return it != trajectory_buffers_.end() ? it->second.size() : 0;
    }

    /// Get weight type
    OTWeightType weight_type() const { return weight_type_; }

    /// Get timestep
    double dt() const { return dt_; }

private:
    /// Update the learned distribution for a specific mode
    void update_mode_distribution(
        int obstacle_id,
        const std::string& mode_id,
        const Eigen::Vector2d& velocity,
        const Eigen::Vector2d& acceleration);

    /// Get velocity distribution for a mode (learned or reference)
    const EmpiricalDistribution* get_mode_velocity_distribution(
        int obstacle_id, const std::string& mode_id) const;

    /// Generate prediction using a specific mode's learned dynamics
    std::vector<OTPredictionStep> predict_with_mode(
        int obstacle_id,
        const std::string& mode_id,
        const Eigen::Vector2d& position,
        const Eigen::Vector2d& velocity,
        int horizon);

    /// Simple constant velocity prediction
    std::vector<OTPredictionStep> constant_velocity_prediction(
        const Eigen::Vector2d& position,
        const Eigen::Vector2d& velocity,
        int horizon);

    /// Combine multiple predictions using Wasserstein barycenter
    std::vector<OTPredictionStep> combine_predictions_barycenter(
        const std::vector<std::vector<OTPredictionStep>>& predictions_list,
        const std::vector<double>& weights,
        int horizon);

    double dt_;                     ///< Timestep
    size_t buffer_size_;            ///< Maximum buffer size per obstacle
    double sinkhorn_epsilon_;       ///< Sinkhorn regularization parameter
    int min_samples_for_ot_;        ///< Minimum samples for OT computation
    double uncertainty_scale_;      ///< Uncertainty scaling factor
    OTWeightType weight_type_;      ///< Mode weight computation strategy
    GroundCostType ground_cost_;    ///< Ground cost type for OT distance

    /// Per-obstacle trajectory buffers
    std::map<int, TrajectoryBuffer> trajectory_buffers_;

    /// Per-obstacle, per-mode learned distributions
    std::map<int, std::map<std::string, ModeDistribution>> mode_distributions_;

    /// Reference mode distributions (prior knowledge)
    std::map<std::string, ModeDistribution> reference_distributions_;

    /// Current timestep
    int current_timestep_ = 0;

    /// Previous observations for velocity/acceleration computation
    std::map<int, std::pair<Eigen::Vector2d, Eigen::Vector2d>> prev_observations_;
};

// =============================================================================
// Factory function for creating OT predictor with standard mode references
// =============================================================================

/**
 * @brief Create OT predictor with reference distributions for standard modes.
 *
 * @param dt Timestep
 * @param base_speed Base obstacle speed for generating reference samples
 * @param buffer_size Maximum buffer size
 * @param sinkhorn_epsilon Sinkhorn regularization
 * @return Configured OptimalTransportPredictor
 */
OptimalTransportPredictor create_ot_predictor_with_standard_modes(
    double dt = 0.1,
    double base_speed = 0.5,
    size_t buffer_size = 200,
    double sinkhorn_epsilon = 0.1);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_OPTIMAL_TRANSPORT_PREDICTOR_HPP
