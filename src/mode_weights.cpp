/**
 * @file mode_weights.cpp
 * @brief Implementation of mode weight computation.
 */

#include "mode_weights.hpp"
#include <cmath>
#include <algorithm>

namespace scenario_mpc {

namespace {

/**
 * @brief Compute recency-based weights (Eq. 5).
 *
 * w_m = sum_{t: m_t = m} lambda^(T - t)
 * Recent observations are weighted more heavily.
 */
std::map<std::string, double> compute_recency_weights(
    const ModeHistory& mode_history,
    const std::vector<std::string>& modes,
    double decay,
    int current_timestep
) {
    std::map<std::string, double> weights;
    for (const auto& mode_id : modes) {
        weights[mode_id] = 0.0;
    }

    for (const auto& [timestep, mode_id] : mode_history.observed_modes) {
        if (weights.find(mode_id) != weights.end()) {
            // Exponential decay based on how old the observation is
            int age = current_timestep - timestep;
            weights[mode_id] += std::pow(decay, age);
        }
    }

    return weights;
}

/**
 * @brief Compute frequency-based weights (Eq. 6).
 *
 * w_m = n_m / sum_j n_j
 * where n_m is the number of times mode m was observed.
 */
std::map<std::string, double> compute_frequency_weights(
    const ModeHistory& mode_history,
    const std::vector<std::string>& modes
) {
    auto counts = mode_history.get_mode_counts();
    std::map<std::string, double> weights;

    for (const auto& mode_id : modes) {
        weights[mode_id] = static_cast<double>(counts[mode_id]);
    }

    return weights;
}

}  // anonymous namespace

std::map<std::string, double> compute_mode_weights(
    const ModeHistory& mode_history,
    WeightType weight_type,
    double recency_decay,
    int current_timestep
) {
    std::vector<std::string> modes;
    for (const auto& [mode_id, _] : mode_history.available_modes) {
        modes.push_back(mode_id);
    }

    int num_modes = static_cast<int>(modes.size());
    if (num_modes == 0) {
        return {};
    }

    std::map<std::string, double> weights;

    switch (weight_type) {
        case WeightType::UNIFORM:
            // Eq. 4: w_m = 1/M for all modes
            for (const auto& mode_id : modes) {
                weights[mode_id] = 1.0 / num_modes;
            }
            break;

        case WeightType::RECENCY:
            // Eq. 5: w_m = sum_{t: m_t = m} lambda^(T - t)
            weights = compute_recency_weights(
                mode_history, modes, recency_decay, current_timestep
            );
            break;

        case WeightType::FREQUENCY:
            // Eq. 6: w_m = n_m / sum_j n_j
            weights = compute_frequency_weights(mode_history, modes);
            break;

        case WeightType::TEMPERATURE:
            // Temperature scaling (T=0.5): w'_m = exp(log(w_m)/T), sharpens distribution
            {
                const double T = 0.5;
                auto freq_w = compute_frequency_weights(mode_history, modes);
                double freq_total = 0;
                for (auto& [_, w] : freq_w) freq_total += w;
                if (freq_total > 0) for (auto& [_, w] : freq_w) w /= freq_total;
                for (const auto& m : modes) {
                    double w = freq_total > 0 ? freq_w[m] : 1.0 / num_modes;
                    // Avoid log(0): clamp to small positive value
                    w = std::max(w, 1e-10);
                    weights[m] = std::exp(std::log(w) / T);
                }
            }
            break;

        case WeightType::EPSILON_GREEDY:
            // Epsilon-greedy (eps=0.3): w'_m = (1-eps)*w_m + eps/M
            {
                const double eps = 0.3;
                auto freq_w = compute_frequency_weights(mode_history, modes);
                double freq_total = 0;
                for (auto& [_, w] : freq_w) freq_total += w;
                if (freq_total > 0) for (auto& [_, w] : freq_w) w /= freq_total;
                for (const auto& m : modes) {
                    double w = freq_total > 0 ? freq_w[m] : 1.0 / num_modes;
                    weights[m] = (1.0 - eps) * w + eps / num_modes;
                }
            }
            break;
    }

    // Gate: only modes observed at least once may have nonzero weight.
    // This ensures progressive discovery — unobserved modes are not sampled.
    auto counts = mode_history.get_mode_counts();
    for (auto& [mode_id, w] : weights) {
        if (counts[mode_id] == 0) {
            w = 0.0;
        }
    }

    // Normalize weights to sum to 1
    double total = 0.0;
    for (const auto& [_, w] : weights) {
        total += w;
    }

    if (total > 0) {
        for (auto& [_, w] : weights) {
            w /= total;
        }
    } else {
        // No modes observed yet — return empty to signal stationary treatment.
        // Callers generate a hold-position trajectory for this obstacle.
        return {};
    }

    return weights;
}

std::vector<std::string> sample_mode_sequence(
    const std::map<std::string, double>& mode_weights,
    int horizon,
    std::mt19937& rng
) {
    std::vector<std::string> sequence;
    sequence.reserve(horizon);

    for (int k = 0; k < horizon; ++k) {
        sequence.push_back(sample_mode_from_weights(mode_weights, rng));
    }

    return sequence;
}

std::string sample_mode_from_weights(
    const std::map<std::string, double>& mode_weights,
    std::mt19937& rng
) {
    std::vector<std::string> modes;
    std::vector<double> weights;

    for (const auto& [mode_id, w] : mode_weights) {
        modes.push_back(mode_id);
        weights.push_back(w);
    }

    // Normalize weights
    double total = 0.0;
    for (double w : weights) {
        total += w;
    }
    for (double& w : weights) {
        w /= total;
    }

    // Sample using discrete distribution
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    int idx = dist(rng);

    return modes[idx];
}

Eigen::MatrixXd compute_mode_transition_matrix(
    const ModeHistory& mode_history,
    const std::vector<std::string>& modes
) {
    int num_modes = static_cast<int>(modes.size());
    std::map<std::string, int> mode_to_idx;
    for (int i = 0; i < num_modes; ++i) {
        mode_to_idx[modes[i]] = i;
    }

    // Count transitions
    Eigen::MatrixXd counts = Eigen::MatrixXd::Zero(num_modes, num_modes);
    const auto& observations = mode_history.observed_modes;

    for (size_t i = 0; i + 1 < observations.size(); ++i) {
        const std::string& mode_from = observations[i].second;
        const std::string& mode_to = observations[i + 1].second;

        if (mode_to_idx.find(mode_from) != mode_to_idx.end() &&
            mode_to_idx.find(mode_to) != mode_to_idx.end()) {
            int idx_from = mode_to_idx[mode_from];
            int idx_to = mode_to_idx[mode_to];
            counts(idx_from, idx_to) += 1.0;
        }
    }

    // Normalize rows to get probabilities
    Eigen::MatrixXd transition_matrix(num_modes, num_modes);

    for (int i = 0; i < num_modes; ++i) {
        double row_sum = counts.row(i).sum();
        if (row_sum > 0) {
            transition_matrix.row(i) = counts.row(i) / row_sum;
        } else {
            // For rows with no observations, use uniform distribution
            transition_matrix.row(i).setConstant(1.0 / num_modes);
        }
    }

    return transition_matrix;
}

}  // namespace scenario_mpc
