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
    const std::vector<std::string>& modes,
    double dirichlet_alpha = 0.0
) {
    auto counts = mode_history.get_mode_counts();
    std::map<std::string, double> weights;

    // (n_m + a): the Dirichlet pseudocount keeps every library mode strictly positive.
    for (const auto& mode_id : modes) {
        weights[mode_id] = static_cast<double>(counts[mode_id]) + dirichlet_alpha;
    }

    return weights;
}

}  // anonymous namespace

std::map<std::string, double> compute_mode_weights(
    const ModeHistory& mode_history,
    WeightType weight_type,
    double recency_decay,
    int current_timestep,
    double dirichlet_alpha
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
            // Dirichlet posterior-predictive mean: w_m = (n_m + a) / (N + M a).
            weights = compute_frequency_weights(mode_history, modes, dirichlet_alpha);
            break;

        case WeightType::TEMPERATURE:
            // Temperature scaling (T=0.5): w'_m = exp(log(w_m)/T), sharpens distribution
            {
                const double T = 0.5;
                auto freq_w = compute_frequency_weights(mode_history, modes, dirichlet_alpha);
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
                auto freq_w = compute_frequency_weights(mode_history, modes, dirichlet_alpha);
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
    const std::vector<std::string>& modes,
    double dirichlet_alpha,
    double sticky_kappa
) {
    int num_modes = static_cast<int>(modes.size());
    std::map<std::string, int> mode_to_idx;
    for (int i = 0; i < num_modes; ++i) {
        mode_to_idx[modes[i]] = i;
    }

    // Count observed i->j transitions.
    Eigen::MatrixXd counts = Eigen::MatrixXd::Zero(num_modes, num_modes);
    const auto& observations = mode_history.observed_modes;
    for (size_t i = 0; i + 1 < observations.size(); ++i) {
        auto f = mode_to_idx.find(observations[i].second);
        auto t = mode_to_idx.find(observations[i + 1].second);
        if (f != mode_to_idx.end() && t != mode_to_idx.end()) {
            counts(f->second, t->second) += 1.0;
        }
    }

    // Dirichlet + sticky posterior mean:
    //   T(i,j) = (N[i][j] + a + kappa*[i==j]) / (rowsum_i + M a + kappa).
    // With a > 0 every entry is strictly positive; a never-observed source row is the pure
    // prior, whose diagonal equals E[T_ii] = (a + kappa)/(M a + kappa).
    Eigen::MatrixXd T(num_modes, num_modes);
    const double M = static_cast<double>(num_modes);
    for (int i = 0; i < num_modes; ++i) {
        const double row_sum = counts.row(i).sum();
        const double Z = row_sum + M * dirichlet_alpha + sticky_kappa;
        for (int j = 0; j < num_modes; ++j) {
            const double sticky = (i == j) ? sticky_kappa : 0.0;
            T(i, j) = (counts(i, j) + dirichlet_alpha + sticky) / Z;
        }
    }
    return T;
}

std::vector<std::string> sample_mode_sequence(
    const ModeDistribution& initial_belief,
    const Eigen::MatrixXd& transition,
    const std::vector<std::string>& modes,
    int horizon,
    std::mt19937& rng,
    bool predict_before_first_sample
) {
    const int M = static_cast<int>(modes.size());
    std::vector<std::string> seq;
    if (M == 0 || horizon <= 0) return seq;
    seq.reserve(horizon);

    // If the first sampled mode governs the interval [t, t+1], draw it from the one-step
    // predictive p_{t+1|t} = T^T p_t; otherwise seed directly from the current belief.
    const ModeDistribution seed =
        predict_before_first_sample ? predict_mode_belief(initial_belief, transition, modes)
                                    : initial_belief;

    // mode_0 ~ seed (uniform fallback if empty/degenerate).
    std::vector<double> w0(M);
    double s0 = 0.0;
    for (int i = 0; i < M; ++i) {
        auto it = seed.find(modes[i]);
        w0[i] = (it != seed.end() && it->second > 0.0) ? it->second : 0.0;
        s0 += w0[i];
    }
    if (!(s0 > 0.0)) std::fill(w0.begin(), w0.end(), 1.0);
    int cur = std::discrete_distribution<int>(w0.begin(), w0.end())(rng);
    seq.push_back(modes[cur]);

    // mode_k ~ T[mode_{k-1}, :].
    for (int k = 1; k < horizon; ++k) {
        std::vector<double> wr(M);
        double sr = 0.0;
        for (int j = 0; j < M; ++j) { wr[j] = transition(cur, j); sr += wr[j]; }
        if (!(sr > 0.0)) std::fill(wr.begin(), wr.end(), 1.0);
        cur = std::discrete_distribution<int>(wr.begin(), wr.end())(rng);
        seq.push_back(modes[cur]);
    }
    return seq;
}

ModeDistribution predict_mode_belief(
    const ModeDistribution& belief,
    const Eigen::MatrixXd& transition,
    const std::vector<std::string>& modes
) {
    const int M = static_cast<int>(modes.size());
    Eigen::VectorXd p(M);
    for (int i = 0; i < M; ++i) {
        auto it = belief.find(modes[i]);
        p(i) = (it != belief.end()) ? it->second : 0.0;
    }
    Eigen::VectorXd pn = transition.transpose() * p;   // p_{t+1} = T^T p_t
    double s = pn.sum();
    if (!(s > 0.0)) s = 1.0;                            // degenerate guard
    ModeDistribution out;
    for (int j = 0; j < M; ++j) out[modes[j]] = pn(j) / s;
    return out;
}

ModeDistribution update_mode_belief(
    const ModeDistribution& prior,
    const Eigen::MatrixXd& transition,
    const std::vector<std::string>& modes,
    const ModeDistribution& likelihood,
    double floor
) {
    ModeDistribution predictive = predict_mode_belief(prior, transition, modes);
    // Divide likelihoods by their maximum before multiplying: Gaussian innovation
    // densities can be tiny and underflow. This leaves the Bayesian posterior unchanged
    // but conditions the arithmetic (the shared 1/max cancels in the normalization).
    double max_like = 0.0;
    for (const auto& m : modes) {
        auto it = likelihood.find(m);
        if (it != likelihood.end() && std::isfinite(it->second) && it->second > 0.0)
            max_like = std::max(max_like, it->second);
    }
    if (max_like <= floor) {
        return predictive;   // no informative likelihood: fall back to the predictive belief
    }
    ModeDistribution post;
    double Z = 0.0;
    for (const auto& m : modes) {
        auto it = likelihood.find(m);
        double like = (it != likelihood.end() && std::isfinite(it->second) && it->second > 0.0)
                          ? it->second : 0.0;
        // Floor the scaled likelihood so a single observation cannot hard-zero a known mode.
        const double val = predictive[m] * std::max(like / max_like, floor);
        post[m] = val;
        Z += val;
    }
    if (Z <= 0.0) return predictive;
    for (auto& [m, w] : post) w /= Z;
    return post;
}

}  // namespace scenario_mpc
