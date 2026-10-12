#include "scenario_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace dro_mpc {

ObstaclePredictions constant_velocity_predictions(
    const std::map<int, ObstacleState>& obstacles, int horizon, double dt, double sigma) {
    ObstaclePredictions predictions;
    for (const auto& [id, state] : obstacles) {
        GaussianPredictionMode mode;
        mode.mode_id = "constant_velocity";
        for (int k = 1; k <= horizon; ++k) {
            mode.steps.push_back({state.position() + k * dt * state.velocity(),
                                  std::atan2(state.vy, state.vx), sigma, sigma});
        }
        predictions[id].push_back(std::move(mode));
    }
    return predictions;
}

std::vector<Scenario> DirectTrajectorySampler::sample(
    const std::map<int, ObstacleState>& obstacles,
    const ObstaclePredictions& predictions,
    int horizon, int num_scenarios, double dt, int batch_count,
    std::mt19937& rng, int scenario_id_offset) {
    if (horizon <= 0 || num_scenarios <= 0 || batch_count <= 0 ||
        !std::isfinite(dt) || dt <= 0.0 || scenario_id_offset < 0 ||
        scenario_id_offset > std::numeric_limits<int>::max() - num_scenarios)
        throw std::invalid_argument("invalid direct trajectory sampling dimensions / timestep / IDs");
    if (predictions.size() != obstacles.size())
        throw std::invalid_argument("direct predictions must cover exactly the current obstacles");

    struct ModePlan {
        std::vector<Eigen::Matrix2d> covariance;
        std::vector<Eigen::Matrix2d> factor;
    };
    std::map<int, std::vector<ModePlan>> plans;
    for (const auto& [id, state] : obstacles) {
        if (!state.to_array().allFinite() || !predictions.count(id) || predictions.at(id).empty())
            throw std::invalid_argument("missing direct obstacle prediction / invalid state");
        double total_probability = 0.0;
        std::set<std::string> mode_ids;
        for (const auto& mode : predictions.at(id)) {
            if (mode.mode_id.empty() || !mode_ids.insert(mode.mode_id).second ||
                !std::isfinite(mode.probability) || mode.probability < 0.0 ||
                mode.steps.size() != static_cast<std::size_t>(horizon))
                throw std::invalid_argument("invalid direct prediction mode / probability / horizon");
            total_probability += mode.probability;
            ModePlan plan;
            for (const auto& step : mode.steps) {
                if (!step.position.allFinite() || !std::isfinite(step.angle) ||
                    !std::isfinite(step.major_radius) || !std::isfinite(step.minor_radius) ||
                    step.major_radius < 0.0 || step.minor_radius < 0.0 ||
                    ((step.major_radius == 0.0) != (step.minor_radius == 0.0)))
                    throw std::invalid_argument("direct noise radii must both be positive or both zero");
                // RosTools::rotationMatrixFromHeading(-angle) is the active
                // rotation below (RosTools itself uses a passive rotation).
                Eigen::Matrix2d rotation;
                rotation << std::cos(step.angle), -std::sin(step.angle),
                            std::sin(step.angle),  std::cos(step.angle);
                const Eigen::Vector2d variance(step.major_radius * step.major_radius,
                                               step.minor_radius * step.minor_radius);
                const Eigen::Matrix2d covariance = rotation * variance.asDiagonal() * rotation.transpose();
                if (!covariance.allFinite())
                    throw std::invalid_argument("direct prediction noise covariance is not finite");
                Eigen::Matrix2d factor = Eigen::Matrix2d::Zero();
                if (step.major_radius > 0.0) {
                    Eigen::LLT<Eigen::Matrix2d> cholesky(covariance);
                    if (cholesky.info() != Eigen::Success)
                        throw std::invalid_argument("direct prediction noise covariance is not positive definite");
                    factor = cholesky.matrixL();
                }
                plan.covariance.push_back(covariance);
                plan.factor.push_back(factor);
            }
            plans[id].push_back(std::move(plan));
        }
        // Do not silently normalize a malformed predictor distribution.
        if (std::abs(total_probability - 1.0) > 1e-12)
            throw std::invalid_argument("direct prediction probabilities must sum to one");
    }

    std::vector<Scenario> scenarios;
    scenarios.reserve(num_scenarios);
    for (int s = 0; s < num_scenarios; ++s)
        scenarios.emplace_back(scenario_id_offset + s, std::map<int, ObstacleTrajectory>{});
    if (obstacles.empty()) return scenarios;

    // Safe-Horizon reference Init/SampleStandardNormal: no pruning/truncation.
    // Retain this bank between solves, rebuilding only if its dimensions change.
    if (standard_samples_.size() != static_cast<std::size_t>(batch_count) ||
        standard_samples_.front().size() != static_cast<std::size_t>(num_scenarios)) {
        standard_samples_.assign(batch_count, std::vector<Eigen::Vector2d>(num_scenarios));
        std::normal_distribution<double> normal(0.0, 1.0);
        for (auto& batch : standard_samples_) {
            for (auto& sample : batch) {
                sample.x() = normal(rng);
                sample.y() = normal(rng);
            }
        }
    }

    // The reference uses a separate random_device-seeded shuffle generator.
    // Seed that separate stream from the caller's RNG for replayable experiments.
    std::mt19937 shuffle_rng(rng());
    std::uniform_int_distribution<int> batch_distribution(0, batch_count - 1);
    std::vector<int> batch_pick(horizon);
    for (auto& batch : batch_pick) batch = batch_distribution(rng);
    std::vector<std::vector<int>> shuffle_indices(horizon, std::vector<int>(num_scenarios));
    for (auto& indices : shuffle_indices) {
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), shuffle_rng);
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (const auto& [id, state] : obstacles) {
        const auto& modes = predictions.at(id);
        for (int s = 0; s < num_scenarios; ++s) {
            const double draw = uniform(rng);
            double cumulative = 0.0;
            std::size_t selected = modes.size() - 1;
            for (std::size_t m = 0; m < modes.size(); ++m) {
                cumulative += modes[m].probability;
                if (draw <= cumulative) { selected = m; break; }
            }
            const auto& mode = modes[selected];
            const auto& plan = plans.at(id)[selected];
            std::vector<PredictionStep> steps;
            steps.emplace_back(0, state.position(), Eigen::Matrix2d::Zero());
            Eigen::Vector2d accumulated_noise = Eigen::Vector2d::Zero();
            // Nominal independent-increment position covariance metadata;
            // database reuse induces dependence not represented by this matrix.
            Eigen::Matrix2d accumulated_covariance = Eigen::Matrix2d::Zero();
            for (int k = 0; k < horizon; ++k) {
                const auto& standard = standard_samples_[batch_pick[k]][shuffle_indices[k][s]];
                accumulated_noise += plan.factor[k] * standard;
                accumulated_covariance += plan.covariance[k];
                steps.emplace_back(k + 1, mode.steps[k].position + dt * accumulated_noise,
                                   dt * dt * accumulated_covariance);
            }
            scenarios[s].trajectories.emplace(id, ObstacleTrajectory(id, mode.mode_id, steps));
        }
    }
    return scenarios;
}

}  // namespace dro_mpc
