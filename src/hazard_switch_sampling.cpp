/**
 * @file hazard_switch_sampling.cpp
 * @brief Implementation of hazard-triggered switch-aware sampling (Section 6.2).
 */

#include "hazard_switch_sampling.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace scenario_mpc {

namespace {
double sigmoid(double x) {
    if (x >= 0) return 1.0 / (1.0 + std::exp(-x));
    double e = std::exp(x);
    return e / (1.0 + e);
}
}  // namespace

void HazardModel::set_mode_coefficients(const std::map<std::string, double>& scale_per_mode) {
    scale_per_mode_ = scale_per_mode;
}

double HazardModel::hazard(const std::string& mode_id, const HazardFeatures& h) const {
    auto it = scale_per_mode_.find(mode_id);
    double scale = (it != scale_per_mode_.end()) ? it->second : 1.0;
    // Simple formula: higher hazard when time_since_last_switch is small (recent switch or imminent)
    double inv_tts = (h.time_since_last_switch > 0.1) ? (1.0 / h.time_since_last_switch) : 10.0;
    double x = scale * (inv_tts + 0.1 * h.time_to_conflict + 0.1 * h.relative_speed);
    return sigmoid(x);
}

std::map<std::string, double> reweight_by_hazard(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& hazard_per_mode,
    double eta) {
    std::map<std::string, double> out;
    double sum = 0.0;
    for (const auto& [mode_id, pi] : nominal_weights) {
        double lam = 0.0;
        auto it = hazard_per_mode.find(mode_id);
        if (it != hazard_per_mode.end()) lam = it->second;
        double w = pi * std::exp(eta * lam);
        out[mode_id] = w;
        sum += w;
    }
    if (sum > 0.0) {
        for (auto& [_, w] : out) w /= sum;
    }
    return out;
}

std::map<std::string, int> allocate_with_hazard_trigger(
    const std::map<std::string, double>& reweighted_probs,
    int total_S,
    double max_hazard,
    double tau,
    int exploration_floor) {
    std::map<std::string, int> out;
    if (reweighted_probs.empty() || total_S <= 0) return out;

    bool use_floor = (exploration_floor > 0 && max_hazard > tau);
    int M = static_cast<int>(reweighted_probs.size());

    if (use_floor) {
        int floor_total = M * exploration_floor;
        if (floor_total >= total_S) {
            for (const auto& [mode_id, _] : reweighted_probs)
                out[mode_id] = exploration_floor;
            return out;
        }
        for (const auto& [mode_id, prob] : reweighted_probs) {
            int base = static_cast<int>(std::floor((total_S - floor_total) * prob)) + exploration_floor;
            out[mode_id] = std::max(1, base);
        }
    } else {
        for (const auto& [mode_id, prob] : reweighted_probs) {
            int Sm = static_cast<int>(std::floor(total_S * prob));
            out[mode_id] = std::max(1, Sm);
        }
    }

    int sum = 0;
    for (const auto& [_, Sm] : out) sum += Sm;
    while (sum < total_S) {
        for (auto& [_, Sm] : out) { Sm++; sum++; if (sum >= total_S) break; }
    }
    while (sum > total_S) {
        for (auto& [id, Sm] : out) {
            if (Sm > 1) { Sm--; sum--; break; }
        }
    }
    return out;
}

}  // namespace scenario_mpc
