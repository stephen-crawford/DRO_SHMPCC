/**
 * @file conformal_safety.cpp
 * @brief Implementation of conformal safety wrappers (Section 6.1).
 */

#include "conformal_safety.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace scenario_mpc {

double conformal_quantile(const std::vector<double>& residuals, double delta) {
    if (residuals.empty()) return 0.0;
    if (delta <= 0.0 || delta >= 1.0) return 0.0;
    std::vector<double> sorted = residuals;
    std::sort(sorted.begin(), sorted.end());
    // (1-delta)-quantile: index n = ceil((n_cal)*(1-delta)) - 1, 0-based
    int n = static_cast<int>(residuals.size());
    int idx = static_cast<int>(std::ceil((n + 1) * (1.0 - delta))) - 1;
    idx = std::clamp(idx, 0, n - 1);
    return sorted[idx];
}

std::map<std::string, double> conformal_quantiles_per_mode(
    const std::map<std::string, std::vector<double>>& residuals_per_mode,
    double delta) {
    std::map<std::string, double> out;
    for (const auto& [mode_id, residuals] : residuals_per_mode) {
        out[mode_id] = conformal_quantile(residuals, delta);
    }
    return out;
}

std::map<std::string, double> boundary_scores(
    const std::map<std::string, std::vector<double>>& constraint_value_at_mean,
    const std::map<std::string, std::vector<double>>& constraint_gradient_norm,
    const std::map<std::string, double>& radius_per_mode,
    const std::vector<std::string>& mode_ids) {
    std::map<std::string, double> scores;
    for (const auto& mode_id : mode_ids) {
        auto it_val = constraint_value_at_mean.find(mode_id);
        auto it_grad = constraint_gradient_norm.find(mode_id);
        auto it_r = radius_per_mode.find(mode_id);
        if (it_val == constraint_value_at_mean.end() || it_grad == constraint_gradient_norm.end() ||
            it_r == radius_per_mode.end())
            continue;
        const auto& vals = it_val->second;
        const auto& grads = it_grad->second;
        double r = it_r->second;
        double sm = -1e30;
        for (size_t t = 0; t < vals.size() && t < grads.size(); ++t) {
            // Worst case over ball: value + radius * gradient_norm
            double worst = vals[t] + r * grads[t];
            if (worst > sm) sm = worst;
        }
        scores[mode_id] = (vals.empty() ? 0.0 : sm);
    }
    return scores;
}

std::map<std::string, int> allocate_scenarios_boundary_aware(
    const std::map<std::string, double>& boundary_scores,
    int total_S,
    double alpha) {
    std::map<std::string, int> out;
    if (boundary_scores.empty() || total_S <= 0) return out;

    double max_s = -1e30;
    for (const auto& [_, s] : boundary_scores) {
        if (s > max_s) max_s = s;
    }
    // exp(alpha * (s_m - max)) for numerical stability
    std::vector<double> exp_s;
    std::vector<std::string> ids;
    double sum = 0.0;
    for (const auto& [mode_id, s] : boundary_scores) {
        double es = std::exp(alpha * (s - max_s));
        exp_s.push_back(es);
        ids.push_back(mode_id);
        sum += es;
    }
    if (sum <= 0.0) {
        for (const auto& id : ids) out[id] = 1;
        return out;
    }

    int assigned = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        int Sm = static_cast<int>(std::floor(total_S * (exp_s[i] / sum)));
        if (Sm < 1) Sm = 1;
        out[ids[i]] = Sm;
        assigned += Sm;
    }
    // Adjust so sum equals total_S
    while (assigned < total_S) {
        double best = -1;
        size_t best_i = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            double frac = (exp_s[i] / sum) - static_cast<double>(out[ids[i]]) / total_S;
            if (frac > best) { best = frac; best_i = i; }
        }
        out[ids[best_i]]++;
        assigned++;
    }
    while (assigned > total_S) {
        for (auto& [id, cnt] : out) {
            if (cnt > 1) { cnt--; assigned--; break; }
        }
    }
    return out;
}

}  // namespace scenario_mpc
