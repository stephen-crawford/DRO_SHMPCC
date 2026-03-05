/**
 * @file optimal_transport_predictor.cpp
 * @brief Implementation of Optimal Transport Predictor for Learning Obstacle Dynamics.
 */

#include "optimal_transport_predictor.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

namespace scenario_mpc {

// =============================================================================
// EmpiricalDistribution Implementation
// =============================================================================

EmpiricalDistribution EmpiricalDistribution::from_samples(
    const Eigen::MatrixXd& samples,
    const Eigen::VectorXd& weights) {

    EmpiricalDistribution dist;
    int n = static_cast<int>(samples.rows());

    if (n == 0) {
        dist.samples_ = Eigen::MatrixXd(0, samples.cols() > 0 ? samples.cols() : 2);
        dist.weights_ = Eigen::VectorXd(0);
        return dist;
    }

    dist.samples_ = samples;

    if (weights.size() == 0) {
        // Uniform weights
        dist.weights_ = Eigen::VectorXd::Constant(n, 1.0 / n);
    } else {
        // Normalize provided weights
        dist.weights_ = weights / weights.sum();
    }

    return dist;
}

Eigen::VectorXd EmpiricalDistribution::mean() const {
    if (n_samples() == 0) {
        return Eigen::VectorXd(0);
    }

    Eigen::VectorXd result = Eigen::VectorXd::Zero(dim());
    for (int i = 0; i < n_samples(); ++i) {
        result += weights_(i) * samples_.row(i).transpose();
    }
    return result;
}

Eigen::MatrixXd EmpiricalDistribution::covariance() const {
    if (n_samples() < 2) {
        if (dim() > 0) {
            return Eigen::MatrixXd::Identity(dim(), dim());
        }
        return Eigen::MatrixXd::Identity(1, 1);
    }

    Eigen::VectorXd mu = mean();
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(dim(), dim());

    for (int i = 0; i < n_samples(); ++i) {
        Eigen::VectorXd centered = samples_.row(i).transpose() - mu;
        cov += weights_(i) * centered * centered.transpose();
    }

    // Bias correction factor for weighted samples
    double w_sum = weights_.sum();
    double w_sum_sq = weights_.squaredNorm();
    double correction = w_sum / (w_sum * w_sum - w_sum_sq);

    if (std::isfinite(correction) && correction > 0) {
        cov *= correction;
    }

    return cov;
}

// =============================================================================
// Cost Matrix Computation
// =============================================================================

Eigen::MatrixXd compute_cost_matrix(
    const Eigen::MatrixXd& source,
    const Eigen::MatrixXd& target,
    int p,
    GroundCostType cost_type) {

    int n = static_cast<int>(source.rows());
    int m = static_cast<int>(target.rows());

    if (n == 0 || m == 0) {
        return Eigen::MatrixXd(n, m);
    }

    Eigen::MatrixXd cost(n, m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            switch (cost_type) {
                case GroundCostType::SQUARED_EUCLIDEAN: {
                    double dist = (source.row(i) - target.row(j)).norm();
                    cost(i, j) = (p == 2) ? dist * dist : std::pow(dist, p);
                    break;
                }
                case GroundCostType::MANHATTAN: {
                    double l1 = (source.row(i) - target.row(j)).cwiseAbs().sum();
                    cost(i, j) = (p == 2) ? l1 * l1 : std::pow(l1, p);
                    break;
                }
                case GroundCostType::FLAT:
                    cost(i, j) = 1.0;
                    break;
                case GroundCostType::DIRECTIONAL: {
                    double dot = source.row(i).dot(target.row(j));
                    double n1 = source.row(i).norm();
                    double n2 = target.row(j).norm();
                    double cos_sim = (n1 > 1e-8 && n2 > 1e-8) ?
                        dot / (n1 * n2) : 0.0;
                    cost(i, j) = 1.0 - cos_sim;  // cosine distance [0, 2]
                    break;
                }
                case GroundCostType::MEAN_ONLY:
                    // MEAN_ONLY is handled at a higher level; fall back to L2
                    cost(i, j) = (source.row(i) - target.row(j)).squaredNorm();
                    break;
                case GroundCostType::RANDOM_PERMUTED:
                case GroundCostType::CONSTANT:
                    // Handled after the loop
                    cost(i, j) = (source.row(i) - target.row(j)).squaredNorm();
                    break;
            }
        }
    }

    // Post-process for special cost types
    if (cost_type == GroundCostType::RANDOM_PERMUTED) {
        // Compute W2 cost matrix first, then randomly permute rows and columns
        // This breaks the semantic alignment while preserving the distribution of costs
        std::mt19937 perm_rng(12345);  // Fixed seed for reproducibility
        std::vector<int> row_perm(n), col_perm(m);
        std::iota(row_perm.begin(), row_perm.end(), 0);
        std::iota(col_perm.begin(), col_perm.end(), 0);
        std::shuffle(row_perm.begin(), row_perm.end(), perm_rng);
        std::shuffle(col_perm.begin(), col_perm.end(), perm_rng);

        Eigen::MatrixXd permuted(n, m);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                permuted(i, j) = cost(row_perm[i], col_perm[j]);
            }
        }
        cost = permuted;
    } else if (cost_type == GroundCostType::CONSTANT) {
        // All entries set to 1.0 (constant cost, degenerate geometry)
        cost.setConstant(1.0);
    }

    return cost;
}

// =============================================================================
// Sinkhorn Algorithm Implementation
// =============================================================================

SinkhornResult sinkhorn_algorithm(
    const Eigen::VectorXd& source_weights,
    const Eigen::VectorXd& target_weights,
    const Eigen::MatrixXd& cost_matrix,
    double epsilon,
    int max_iterations,
    double convergence_threshold) {

    SinkhornResult result;
    int n = static_cast<int>(cost_matrix.rows());
    int m = static_cast<int>(cost_matrix.cols());

    // Handle edge cases
    if (n == 0 || m == 0) {
        result.transport_plan = Eigen::MatrixXd(n, m);
        result.sinkhorn_distance = 0.0;
        result.iterations = 0;
        result.converged = true;
        return result;
    }

    // Kernel matrix K = exp(-C/epsilon)
    Eigen::MatrixXd K = (-cost_matrix / epsilon).array().exp().matrix();

    // Numerical stability: clamp very small values
    const double min_val = 1e-300;
    K = K.cwiseMax(min_val);

    // Initialize scaling vectors
    Eigen::VectorXd u = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd v = Eigen::VectorXd::Ones(m);

    // Sinkhorn iterations
    result.converged = false;
    for (int iter = 0; iter < max_iterations; ++iter) {
        Eigen::VectorXd u_prev = u;

        // Update u: u = a / (K @ v)
        Eigen::VectorXd Kv = K * v;
        u = source_weights.array() / Kv.cwiseMax(min_val).array();

        // Update v: v = b / (K.T @ u)
        Eigen::VectorXd Ktu = K.transpose() * u;
        v = target_weights.array() / Ktu.cwiseMax(min_val).array();

        // Check convergence
        double max_change = (u - u_prev).cwiseAbs().maxCoeff();
        if (max_change < convergence_threshold) {
            result.converged = true;
            result.iterations = iter + 1;
            break;
        }
        result.iterations = iter + 1;
    }

    // Compute transport plan: P = diag(u) @ K @ diag(v)
    result.transport_plan = u.asDiagonal() * K * v.asDiagonal();

    // Sinkhorn distance: <C, P>
    result.sinkhorn_distance = (cost_matrix.array() * result.transport_plan.array()).sum();

    return result;
}

// =============================================================================
// Wasserstein Distance
// =============================================================================

double wasserstein_distance(
    const EmpiricalDistribution& source,
    const EmpiricalDistribution& target,
    double epsilon,
    int p,
    GroundCostType cost_type) {

    if (source.is_empty() || target.is_empty()) {
        return 0.0;
    }

    Eigen::MatrixXd cost_matrix = compute_cost_matrix(
        source.samples(), target.samples(), p, cost_type);

    SinkhornResult result = sinkhorn_algorithm(
        source.weights(), target.weights(), cost_matrix, epsilon);

    // Return p-th root for W_p distance
    return std::pow(result.sinkhorn_distance, 1.0 / p);
}

// =============================================================================
// Wasserstein Barycenter
// =============================================================================

EmpiricalDistribution wasserstein_barycenter(
    const std::vector<EmpiricalDistribution>& distributions,
    const std::vector<double>& weights,
    int n_support,
    double epsilon,
    int max_iterations,
    double convergence_threshold) {

    if (distributions.empty()) {
        return EmpiricalDistribution::from_samples(
            Eigen::MatrixXd(0, 2), Eigen::VectorXd(0));
    }

    if (distributions.size() == 1) {
        return distributions[0];
    }

    // Normalize weights
    std::vector<double> normalized_weights = weights;
    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total > 0) {
        for (auto& w : normalized_weights) {
            w /= total;
        }
    } else {
        double uniform = 1.0 / weights.size();
        for (auto& w : normalized_weights) {
            w = uniform;
        }
    }

    // Get dimension from first non-empty distribution
    int dim = 0;
    for (const auto& dist : distributions) {
        if (!dist.is_empty()) {
            dim = dist.dim();
            break;
        }
    }

    if (dim == 0) {
        return EmpiricalDistribution::from_samples(
            Eigen::MatrixXd(0, 2), Eigen::VectorXd(0));
    }

    // Initialize barycenter support from weighted combination
    std::vector<Eigen::VectorXd> all_samples;
    std::mt19937 rng(42);  // Fixed seed for reproducibility

    for (size_t i = 0; i < distributions.size(); ++i) {
        if (distributions[i].is_empty()) continue;

        int n_from_dist = std::max(1, static_cast<int>(n_support * normalized_weights[i]));
        const auto& dist = distributions[i];

        // Sample from distribution
        std::discrete_distribution<> sampler(
            dist.weights().data(),
            dist.weights().data() + dist.weights().size());

        for (int j = 0; j < n_from_dist; ++j) {
            int idx = sampler(rng);
            all_samples.push_back(dist.samples().row(idx).transpose());
        }
    }

    if (all_samples.empty()) {
        return EmpiricalDistribution::from_samples(
            Eigen::MatrixXd(0, dim), Eigen::VectorXd(0));
    }

    // Truncate to n_support samples
    if (static_cast<int>(all_samples.size()) > n_support) {
        all_samples.resize(n_support);
    }

    Eigen::MatrixXd barycenter_samples(all_samples.size(), dim);
    for (size_t i = 0; i < all_samples.size(); ++i) {
        barycenter_samples.row(i) = all_samples[i].transpose();
    }

    Eigen::VectorXd barycenter_weights =
        Eigen::VectorXd::Constant(barycenter_samples.rows(),
                                   1.0 / barycenter_samples.rows());

    // Iterative refinement using fixed-point iterations
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Eigen::MatrixXd samples_prev = barycenter_samples;

        // Compute transport plans to each distribution
        Eigen::MatrixXd transport_updates =
            Eigen::MatrixXd::Zero(barycenter_samples.rows(), dim);

        for (size_t i = 0; i < distributions.size(); ++i) {
            if (distributions[i].is_empty()) continue;

            Eigen::MatrixXd cost_matrix = compute_cost_matrix(
                barycenter_samples, distributions[i].samples(), 2);

            SinkhornResult result = sinkhorn_algorithm(
                barycenter_weights, distributions[i].weights(),
                cost_matrix, epsilon);

            // Barycentric update: weighted average of transport targets
            Eigen::VectorXd row_sums = result.transport_plan.rowwise().sum();
            row_sums = row_sums.cwiseMax(1e-10);

            Eigen::MatrixXd update =
                (result.transport_plan * distributions[i].samples()).array().colwise() /
                row_sums.array();

            transport_updates += normalized_weights[i] * update;
        }

        barycenter_samples = transport_updates;

        // Check convergence
        double change = (barycenter_samples - samples_prev).cwiseAbs().maxCoeff();
        if (change < convergence_threshold) {
            break;
        }
    }

    return EmpiricalDistribution::from_samples(barycenter_samples, barycenter_weights);
}

// =============================================================================
// OptimalTransportPredictor Implementation
// =============================================================================

OptimalTransportPredictor::OptimalTransportPredictor(
    double dt,
    size_t buffer_size,
    double sinkhorn_epsilon,
    int min_samples_for_ot,
    double uncertainty_scale,
    OTWeightType weight_type,
    GroundCostType ground_cost)
    : dt_(dt), buffer_size_(buffer_size), sinkhorn_epsilon_(sinkhorn_epsilon),
      min_samples_for_ot_(min_samples_for_ot), uncertainty_scale_(uncertainty_scale),
      weight_type_(weight_type), ground_cost_(ground_cost), current_timestep_(0) {}

void OptimalTransportPredictor::observe(
    int obstacle_id,
    const Eigen::Vector2d& position,
    const std::string& mode_id) {

    // Initialize buffer if needed
    if (trajectory_buffers_.find(obstacle_id) == trajectory_buffers_.end()) {
        trajectory_buffers_.emplace(obstacle_id,
            TrajectoryBuffer(obstacle_id, buffer_size_));
        prev_observations_[obstacle_id] = {position, Eigen::Vector2d::Zero()};
    }

    // Compute velocity and acceleration from finite differences
    auto& [prev_pos, prev_vel] = prev_observations_[obstacle_id];

    Eigen::Vector2d velocity = (position - prev_pos) / dt_;
    Eigen::Vector2d acceleration = (velocity - prev_vel) / dt_;

    // Create and store observation
    TrajectoryObservation obs(current_timestep_, position, velocity,
                               acceleration, mode_id);
    trajectory_buffers_.at(obstacle_id).add_observation(obs);

    // Update previous observation
    prev_pos = position;
    prev_vel = velocity;

    // Update mode distribution if mode is provided
    if (!mode_id.empty()) {
        update_mode_distribution(obstacle_id, mode_id, velocity, acceleration);
    }
}

void OptimalTransportPredictor::update_mode_distribution(
    int obstacle_id,
    const std::string& mode_id,
    const Eigen::Vector2d& velocity,
    const Eigen::Vector2d& acceleration) {

    if (mode_distributions_.find(obstacle_id) == mode_distributions_.end()) {
        mode_distributions_[obstacle_id] = {};
    }

    auto& mode_dists = mode_distributions_[obstacle_id];

    if (mode_dists.find(mode_id) == mode_dists.end()) {
        // Initialize new mode distribution
        Eigen::MatrixXd vel_samples(1, 2);
        vel_samples.row(0) = velocity.transpose();
        Eigen::MatrixXd acc_samples(1, 2);
        acc_samples.row(0) = acceleration.transpose();

        mode_dists[mode_id] = ModeDistribution(
            mode_id,
            EmpiricalDistribution::from_samples(vel_samples),
            EmpiricalDistribution::from_samples(acc_samples),
            1, current_timestep_);
    } else {
        // Update existing distribution with new sample
        auto& mode_dist = mode_dists[mode_id];

        // Add new samples to existing distributions
        int n_old = mode_dist.velocity_dist.n_samples();
        int max_samples = static_cast<int>(buffer_size_ / 2);

        Eigen::MatrixXd new_vel_samples(std::min(n_old + 1, max_samples), 2);
        Eigen::MatrixXd new_acc_samples(std::min(n_old + 1, max_samples), 2);

        // Keep recent samples
        int start_idx = std::max(0, n_old + 1 - max_samples);
        int copy_count = std::min(n_old, max_samples - 1);

        for (int i = 0; i < copy_count; ++i) {
            new_vel_samples.row(i) =
                mode_dist.velocity_dist.samples().row(start_idx + i);
            new_acc_samples.row(i) =
                mode_dist.acceleration_dist.samples().row(start_idx + i);
        }

        // Add new sample
        new_vel_samples.row(new_vel_samples.rows() - 1) = velocity.transpose();
        new_acc_samples.row(new_acc_samples.rows() - 1) = acceleration.transpose();

        mode_dist.velocity_dist = EmpiricalDistribution::from_samples(new_vel_samples);
        mode_dist.acceleration_dist = EmpiricalDistribution::from_samples(new_acc_samples);
        mode_dist.observation_count++;
        mode_dist.last_updated = current_timestep_;
    }
}

std::map<std::string, double> OptimalTransportPredictor::compute_mode_weights(
    int obstacle_id,
    const std::vector<std::string>& available_modes,
    const Eigen::Vector2d* reference_velocity) {

    std::map<std::string, double> weights;

    if (weight_type_ == OTWeightType::UNIFORM || available_modes.empty()) {
        double uniform = 1.0 / std::max(1, static_cast<int>(available_modes.size()));
        for (const auto& mode : available_modes) {
            weights[mode] = uniform;
        }
        return weights;
    }

    // Get recent velocity samples
    auto it = trajectory_buffers_.find(obstacle_id);
    if (it == trajectory_buffers_.end() ||
        static_cast<int>(it->second.size()) < min_samples_for_ot_) {
        // Not enough samples, use uniform
        double uniform = 1.0 / std::max(1, static_cast<int>(available_modes.size()));
        for (const auto& mode : available_modes) {
            weights[mode] = uniform;
        }
        return weights;
    }

    // Build empirical distribution from recent observations
    auto recent_obs = it->second.get_recent(min_samples_for_ot_);
    Eigen::MatrixXd observed_velocities(recent_obs.size(), 2);
    for (size_t i = 0; i < recent_obs.size(); ++i) {
        observed_velocities.row(i) = recent_obs[i].velocity.transpose();
    }
    EmpiricalDistribution observed_dist =
        EmpiricalDistribution::from_samples(observed_velocities);

    // Compute distance to each mode's distribution
    for (const auto& mode_id : available_modes) {
        const EmpiricalDistribution* mode_dist =
            get_mode_velocity_distribution(obstacle_id, mode_id);

        if (mode_dist == nullptr || mode_dist->is_empty()) {
            weights[mode_id] = 1.0;
            continue;
        }

        double dist;
        if (ground_cost_ == GroundCostType::MEAN_ONLY) {
            // Bypass OT: use L2 distance between distribution means
            Eigen::VectorXd obs_mean = observed_dist.samples().colwise().mean();
            Eigen::VectorXd mode_mean = mode_dist->samples().colwise().mean();
            dist = (obs_mean - mode_mean).norm();
        } else {
            // Full OT with specified ground cost
            dist = wasserstein_distance(
                observed_dist, *mode_dist, sinkhorn_epsilon_, 2, ground_cost_);
        }

        // Convert distance to weight: higher distance -> lower weight
        // Using exponential kernel: w = exp(-distance / scale)
        weights[mode_id] = std::exp(-dist / (uncertainty_scale_ + 1e-6));
    }

    // Normalize weights
    double total = 0.0;
    for (const auto& [_, w] : weights) {
        total += w;
    }

    if (total > 0) {
        for (auto& [_, w] : weights) {
            w /= total;
        }
    } else {
        double uniform = 1.0 / std::max(1, static_cast<int>(available_modes.size()));
        for (auto& [_, w] : weights) {
            w = uniform;
        }
    }

    return weights;
}

const EmpiricalDistribution* OptimalTransportPredictor::get_mode_velocity_distribution(
    int obstacle_id, const std::string& mode_id) const {

    // Check learned distributions first
    auto obs_it = mode_distributions_.find(obstacle_id);
    if (obs_it != mode_distributions_.end()) {
        auto mode_it = obs_it->second.find(mode_id);
        if (mode_it != obs_it->second.end()) {
            return &mode_it->second.velocity_dist;
        }
    }

    // Fall back to reference distributions
    auto ref_it = reference_distributions_.find(mode_id);
    if (ref_it != reference_distributions_.end()) {
        return &ref_it->second.velocity_dist;
    }

    return nullptr;
}

std::vector<OTPredictionStep> OptimalTransportPredictor::predict_trajectory(
    int obstacle_id,
    const Eigen::Vector2d& current_position,
    const Eigen::Vector2d& current_velocity,
    int horizon,
    const std::map<std::string, double>* mode_weights) {

    // Get available modes
    std::vector<std::string> available_modes;

    auto obs_it = mode_distributions_.find(obstacle_id);
    if (obs_it != mode_distributions_.end()) {
        for (const auto& [mode_id, _] : obs_it->second) {
            available_modes.push_back(mode_id);
        }
    }

    if (available_modes.empty()) {
        for (const auto& [mode_id, _] : reference_distributions_) {
            available_modes.push_back(mode_id);
        }
    }

    if (available_modes.empty()) {
        return constant_velocity_prediction(current_position, current_velocity, horizon);
    }

    // Compute mode weights if not provided
    std::map<std::string, double> weights;
    if (mode_weights == nullptr) {
        weights = compute_mode_weights(obstacle_id, available_modes);
    } else {
        weights = *mode_weights;
    }

    // Generate prediction for each mode
    std::vector<std::vector<OTPredictionStep>> mode_predictions;
    std::vector<double> active_weights;

    for (const auto& mode_id : available_modes) {
        double weight = weights.count(mode_id) > 0 ? weights.at(mode_id) : 0.0;
        if (weight < 0.01) continue;  // Skip negligible modes

        auto pred = predict_with_mode(
            obstacle_id, mode_id, current_position, current_velocity, horizon);
        mode_predictions.push_back(std::move(pred));
        active_weights.push_back(weight);
    }

    if (mode_predictions.empty()) {
        return constant_velocity_prediction(current_position, current_velocity, horizon);
    }

    if (mode_predictions.size() == 1) {
        return mode_predictions[0];
    }

    // Combine predictions using Wasserstein barycenter
    return combine_predictions_barycenter(mode_predictions, active_weights, horizon);
}

std::vector<OTPredictionStep> OptimalTransportPredictor::predict_with_mode(
    int obstacle_id,
    const std::string& mode_id,
    const Eigen::Vector2d& position,
    const Eigen::Vector2d& velocity,
    int horizon) {

    std::vector<OTPredictionStep> predictions;
    predictions.reserve(horizon + 1);

    // Get mode distribution
    const ModeDistribution* mode_dist = nullptr;

    auto obs_it = mode_distributions_.find(obstacle_id);
    if (obs_it != mode_distributions_.end()) {
        auto mode_it = obs_it->second.find(mode_id);
        if (mode_it != obs_it->second.end()) {
            mode_dist = &mode_it->second;
        }
    }

    if (mode_dist == nullptr) {
        auto ref_it = reference_distributions_.find(mode_id);
        if (ref_it != reference_distributions_.end()) {
            mode_dist = &ref_it->second;
        }
    }

    Eigen::Vector2d pos = position;
    Eigen::Vector2d vel = velocity;

    for (int k = 0; k <= horizon; ++k) {
        double major_r, minor_r;

        // Compute uncertainty from distribution spread
        if (mode_dist != nullptr && !mode_dist->velocity_dist.is_empty()) {
            Eigen::MatrixXd vel_cov = mode_dist->velocity_dist.covariance();
            if (vel_cov.rows() >= 2 && vel_cov.cols() >= 2) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(
                    vel_cov.block<2, 2>(0, 0));
                Eigen::Vector2d eigvals = solver.eigenvalues();
                major_r = std::sqrt(std::max(eigvals(0), eigvals(1))) *
                          (1 + 0.1 * k) * uncertainty_scale_;
                minor_r = std::sqrt(std::min(eigvals(0), eigvals(1))) *
                          (1 + 0.1 * k) * uncertainty_scale_;
            } else {
                major_r = 0.3 * (1 + 0.1 * k);
                minor_r = 0.3 * (1 + 0.1 * k);
            }
        } else {
            major_r = 0.3 * (1 + 0.1 * k);
            minor_r = 0.3 * (1 + 0.1 * k);
        }

        // Compute heading from velocity
        double speed = vel.norm();
        double angle = (speed > 0.01) ? std::atan2(vel(1), vel(0)) : 0.0;

        predictions.emplace_back(pos, angle, major_r, minor_r);

        // Propagate position
        pos = pos + vel * dt_;

        // Update velocity based on mode dynamics
        if (mode_dist != nullptr && !mode_dist->velocity_dist.is_empty()) {
            Eigen::VectorXd mean_vel = mode_dist->velocity_dist.mean();
            if (mean_vel.size() >= 2) {
                vel = 0.95 * vel + 0.05 * mean_vel.head<2>();
            }
        }
    }

    return predictions;
}

std::vector<OTPredictionStep> OptimalTransportPredictor::constant_velocity_prediction(
    const Eigen::Vector2d& position,
    const Eigen::Vector2d& velocity,
    int horizon) {

    std::vector<OTPredictionStep> predictions;
    predictions.reserve(horizon + 1);

    Eigen::Vector2d pos = position;

    for (int k = 0; k <= horizon; ++k) {
        double speed = velocity.norm();
        double angle = (speed > 0.01) ? std::atan2(velocity(1), velocity(0)) : 0.0;

        double major_r = 0.3 * (1 + 0.1 * k);
        double minor_r = 0.3 * (1 + 0.1 * k);

        predictions.emplace_back(pos, angle, major_r, minor_r);

        pos = pos + velocity * dt_;
    }

    return predictions;
}

std::vector<OTPredictionStep> OptimalTransportPredictor::combine_predictions_barycenter(
    const std::vector<std::vector<OTPredictionStep>>& predictions_list,
    const std::vector<double>& weights,
    int horizon) {

    std::vector<OTPredictionStep> combined;
    combined.reserve(horizon + 1);

    // Normalize weights
    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::vector<double> norm_weights = weights;
    if (total > 0) {
        for (auto& w : norm_weights) {
            w /= total;
        }
    }

    for (int k = 0; k <= horizon; ++k) {
        // Collect positions and uncertainties at this timestep
        std::vector<Eigen::Vector2d> positions;
        std::vector<double> major_radii;
        std::vector<double> minor_radii;
        std::vector<double> angles;
        std::vector<double> active_weights;

        for (size_t i = 0; i < predictions_list.size(); ++i) {
            if (k < static_cast<int>(predictions_list[i].size())) {
                positions.push_back(predictions_list[i][k].position);
                major_radii.push_back(predictions_list[i][k].major_radius);
                minor_radii.push_back(predictions_list[i][k].minor_radius);
                angles.push_back(predictions_list[i][k].angle);
                active_weights.push_back(norm_weights[i]);
            }
        }

        if (positions.empty()) continue;

        // Compute barycenter position (weighted average)
        Eigen::Vector2d barycenter_pos = Eigen::Vector2d::Zero();
        double weight_sum = 0.0;
        for (size_t i = 0; i < positions.size(); ++i) {
            barycenter_pos += active_weights[i] * positions[i];
            weight_sum += active_weights[i];
        }
        if (weight_sum > 0) {
            barycenter_pos /= weight_sum;
        }

        // Average uncertainty
        double avg_major = 0.0, avg_minor = 0.0;
        for (size_t i = 0; i < major_radii.size(); ++i) {
            avg_major += active_weights[i] * major_radii[i];
            avg_minor += active_weights[i] * minor_radii[i];
        }
        if (weight_sum > 0) {
            avg_major /= weight_sum;
            avg_minor /= weight_sum;
        }

        // Weighted circular mean for angle
        double sin_sum = 0.0, cos_sum = 0.0;
        for (size_t i = 0; i < angles.size(); ++i) {
            sin_sum += active_weights[i] * std::sin(angles[i]);
            cos_sum += active_weights[i] * std::cos(angles[i]);
        }
        double avg_angle = std::atan2(sin_sum, cos_sum);

        // Add uncertainty from mode disagreement
        double pos_spread = 0.0;
        for (size_t i = 0; i < positions.size(); ++i) {
            pos_spread += active_weights[i] *
                          (positions[i] - barycenter_pos).squaredNorm();
        }
        pos_spread = std::sqrt(pos_spread / std::max(weight_sum, 1e-10));
        double spread_factor = 1.0 + pos_spread * 0.5;

        combined.emplace_back(barycenter_pos, avg_angle,
                               avg_major * spread_factor,
                               avg_minor * spread_factor);
    }

    return combined;
}

double OptimalTransportPredictor::compute_prediction_error(
    int obstacle_id,
    const std::vector<Eigen::Vector2d>& predicted_trajectory,
    const std::vector<Eigen::Vector2d>& actual_trajectory) {

    if (predicted_trajectory.empty() || actual_trajectory.empty()) {
        return 0.0;
    }

    size_t min_len = std::min(predicted_trajectory.size(), actual_trajectory.size());

    Eigen::MatrixXd pred_array(min_len, 2);
    Eigen::MatrixXd actual_array(min_len, 2);

    for (size_t i = 0; i < min_len; ++i) {
        pred_array.row(i) = predicted_trajectory[i].transpose();
        actual_array.row(i) = actual_trajectory[i].transpose();
    }

    EmpiricalDistribution pred_dist = EmpiricalDistribution::from_samples(pred_array);
    EmpiricalDistribution actual_dist = EmpiricalDistribution::from_samples(actual_array);

    return wasserstein_distance(pred_dist, actual_dist, sinkhorn_epsilon_);
}

double OptimalTransportPredictor::adapt_uncertainty(
    int obstacle_id, double prediction_error) {

    // Sigmoid-like scaling: error -> multiplier in [1, 3]
    double base_scale = 1.0;
    double error_scale = 2.0 * (1.0 / (1.0 + std::exp(-prediction_error + 1.0)));

    return base_scale + error_scale;
}

void OptimalTransportPredictor::set_reference_distribution(
    const std::string& mode_id,
    const Eigen::MatrixX2d& velocity_samples,
    const Eigen::MatrixX2d* acceleration_samples) {

    EmpiricalDistribution vel_dist =
        EmpiricalDistribution::from_samples(velocity_samples);

    EmpiricalDistribution acc_dist;
    if (acceleration_samples != nullptr) {
        acc_dist = EmpiricalDistribution::from_samples(*acceleration_samples);
    } else {
        acc_dist = EmpiricalDistribution::from_samples(Eigen::MatrixXd::Zero(1, 2));
    }

    reference_distributions_[mode_id] = ModeDistribution(
        mode_id, vel_dist, acc_dist,
        static_cast<int>(velocity_samples.rows()),
        current_timestep_);
}

std::set<std::string> OptimalTransportPredictor::get_learned_modes(int obstacle_id) const {
    std::set<std::string> modes;

    auto it = mode_distributions_.find(obstacle_id);
    if (it != mode_distributions_.end()) {
        for (const auto& [mode_id, _] : it->second) {
            modes.insert(mode_id);
        }
    }

    return modes;
}

std::optional<std::map<std::string, double>>
OptimalTransportPredictor::get_mode_distribution_stats(
    int obstacle_id, const std::string& mode_id) const {

    auto obs_it = mode_distributions_.find(obstacle_id);
    if (obs_it == mode_distributions_.end()) {
        return std::nullopt;
    }

    auto mode_it = obs_it->second.find(mode_id);
    if (mode_it == obs_it->second.end()) {
        return std::nullopt;
    }

    const auto& mode_dist = mode_it->second;
    std::map<std::string, double> stats;

    stats["observation_count"] = static_cast<double>(mode_dist.observation_count);
    stats["last_updated"] = static_cast<double>(mode_dist.last_updated);

    if (!mode_dist.velocity_dist.is_empty()) {
        Eigen::VectorXd vel_mean = mode_dist.velocity_dist.mean();
        if (vel_mean.size() >= 2) {
            stats["velocity_mean_x"] = vel_mean(0);
            stats["velocity_mean_y"] = vel_mean(1);
        }
    }

    return stats;
}

std::optional<std::pair<Eigen::Vector4d, Eigen::Matrix4d>>
OptimalTransportPredictor::estimate_mode_dynamics(
    int obstacle_id,
    const std::string& mode_id,
    const Eigen::Matrix4d& A_prior,
    double /*dt*/) {

    auto buf_it = trajectory_buffers_.find(obstacle_id);
    if (buf_it == trajectory_buffers_.end()) return std::nullopt;

    const auto& observations = buf_it->second.observations();
    if (observations.size() < 2) return std::nullopt;

    // Collect residuals from consecutive pairs where mode matches
    std::vector<Eigen::Vector4d> residuals;
    for (size_t i = 0; i + 1 < observations.size(); ++i) {
        const auto& obs_k = observations[i];
        const auto& obs_k1 = observations[i + 1];

        if (obs_k.mode_id != mode_id && obs_k1.mode_id != mode_id)
            continue;

        Eigen::Vector4d x_k = obs_k.dynamics_state();
        Eigen::Vector4d x_k1 = obs_k1.dynamics_state();
        Eigen::Vector4d predicted = A_prior * x_k;
        residuals.push_back(x_k1 - predicted);
    }

    if (residuals.size() < 5) return std::nullopt;

    // b_learned = mean(residuals)
    Eigen::Vector4d b_learned = Eigen::Vector4d::Zero();
    for (const auto& r : residuals) b_learned += r;
    b_learned /= static_cast<double>(residuals.size());

    // cov(residuals) + regularization
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
    for (const auto& r : residuals) {
        Eigen::Vector4d diff = r - b_learned;
        cov += diff * diff.transpose();
    }
    cov /= static_cast<double>(residuals.size() - 1);
    cov += 1e-6 * Eigen::Matrix4d::Identity();

    // G_learned = cholesky(cov)
    Eigen::LLT<Eigen::Matrix4d> llt(cov);
    Eigen::Matrix4d G_learned;
    if (llt.info() == Eigen::Success) {
        G_learned = llt.matrixL();
    } else {
        // Fallback: diagonal approximation
        G_learned = Eigen::Matrix4d::Zero();
        for (int i = 0; i < 4; ++i)
            G_learned(i, i) = std::sqrt(std::max(cov(i, i), 1e-8));
    }

    return std::make_pair(b_learned, G_learned);
}

void OptimalTransportPredictor::reset() {
    trajectory_buffers_.clear();
    mode_distributions_.clear();
    prev_observations_.clear();
    current_timestep_ = 0;
}

void OptimalTransportPredictor::reset_all() {
    reset();
    reference_distributions_.clear();
}

// =============================================================================
// Factory Function
// =============================================================================

OptimalTransportPredictor create_ot_predictor_with_standard_modes(
    double dt,
    double base_speed,
    size_t buffer_size,
    double sinkhorn_epsilon) {

    OptimalTransportPredictor predictor(
        dt, buffer_size, sinkhorn_epsilon,
        10,   // min_samples_for_ot
        1.0,  // uncertainty_scale
        OTWeightType::WASSERSTEIN);

    int n_samples = 100;
    double noise_std = 0.1;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, noise_std);

    auto generate_samples = [&](double mean_x, double mean_y) {
        Eigen::MatrixX2d samples(n_samples, 2);
        for (int i = 0; i < n_samples; ++i) {
            samples(i, 0) = mean_x + noise(rng);
            samples(i, 1) = mean_y + noise(rng);
        }
        return samples;
    };

    // Constant velocity - forward motion
    predictor.set_reference_distribution("constant_velocity",
        generate_samples(base_speed, 0.0));

    // Decelerating - reduced forward speed
    predictor.set_reference_distribution("decelerating",
        generate_samples(base_speed * 0.5, 0.0));

    // Accelerating - increased forward speed
    predictor.set_reference_distribution("accelerating",
        generate_samples(base_speed * 1.5, 0.0));

    // Turn left - forward with positive lateral component
    double turn_rate = 0.8;
    predictor.set_reference_distribution("turn_left",
        generate_samples(base_speed * std::cos(turn_rate),
                         base_speed * std::sin(turn_rate)));

    // Turn right - forward with negative lateral component
    predictor.set_reference_distribution("turn_right",
        generate_samples(base_speed * std::cos(-turn_rate),
                         base_speed * std::sin(-turn_rate)));

    // Lane change left
    predictor.set_reference_distribution("lane_change_left",
        generate_samples(base_speed, 0.3));

    // Lane change right
    predictor.set_reference_distribution("lane_change_right",
        generate_samples(base_speed, -0.3));

    return predictor;
}

}  // namespace scenario_mpc
