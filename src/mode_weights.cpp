/**
 * @file mode_weights.cpp
 * @brief Implementation of mode weight computation.
 */

#include "mode_weights.hpp"
#include <cmath>
#include <algorithm>

namespace dro_mpc {

namespace {

/**
 * @brief Compute weights
 */
std::map<std::string, double> compute_frequency_weights(
    const ModeHistory& mode_history,
    const std::vector<std::string>& modes,
    double dirichlet_alpha = 0.5
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
    const ModeBeliefConfig& belief
) {
    std::vector<std::string> modes;
    for (const auto& [mode_id, _] : mode_history.available_modes) {
        modes.push_back(mode_id);
    }

    int num_modes = static_cast<int>(modes.size());
    if (num_modes == 0) {
        return {};
    }

    // Nominal belief = Dirichlet posterior-predictive mean
    //   p_m = (n_m + a) / (N + M a),   a = belief.alpha(M),
    // so every library mode keeps strictly positive mass (no zero-mask on
    // unobserved modes). The belief KIND (DIRICHLET vs STICKY) shares this
    // marginal; stickiness enters only through the Markov transition prior
    // (ModeBeliefConfig::kappa), which the mode-sequence samplers apply.
    auto weights = compute_frequency_weights(mode_history, modes, belief.alpha(num_modes));

    // Normalize to sum to 1
    double total = 0.0;
    for (const auto& [_, w] : weights) {
        total += w;
    }
    if (total > 0) {
        for (auto& [_, w] : weights) {
            w /= total;
        }
    } else {
        // No modes and no prior mass — stationary treatment by caller.
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

}  // namespace dro_mpc
