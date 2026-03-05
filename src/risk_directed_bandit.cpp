/**
 * @file risk_directed_bandit.cpp
 * @brief Implementation of risk-directed UCB allocation (Section 6.3).
 */

#include "risk_directed_bandit.hpp"
#include <cmath>
#include <algorithm>

namespace scenario_mpc {

void RiskDirectedBandit::set_modes(const std::vector<std::string>& mode_ids) {
    for (const auto& id : mode_ids) {
        state_[id] = BanditModeState{0.0, 0};
    }
}

void RiskDirectedBandit::update(int timestep, const std::string& mode_id, bool violation_observed) {
    t_ = std::max(t_, timestep + 1);
    auto it = state_.find(mode_id);
    if (it == state_.end()) return;
    auto& s = it->second;
    double v = violation_observed ? 1.0 : 0.0;
    s.R_hat = (s.R_hat * s.n + v) / (s.n + 1);
    s.n++;
}

double RiskDirectedBandit::ucb(const std::string& mode_id) const {
    auto it = state_.find(mode_id);
    if (it == state_.end()) return 0.0;
    const auto& s = it->second;
    int n = std::max(1, s.n);
    double exploration = (t_ > 0) ? beta_ * std::sqrt(std::log(static_cast<double>(t_ + 1)) / n) : 1.0;
    return s.R_hat + exploration;
}

std::map<std::string, int> RiskDirectedBandit::allocate(int total_S) const {
    std::map<std::string, int> out;
    if (state_.empty() || total_S <= 0) return out;
    double sum_ucb = 0.0;
    for (const auto& [id, _] : state_) sum_ucb += ucb(id);
    if (sum_ucb <= 0.0) {
        int per_mode = total_S / static_cast<int>(state_.size());
        for (const auto& [id, _] : state_) out[id] = per_mode;
        return out;
    }
    int assigned = 0;
    for (const auto& [id, _] : state_) {
        int Sm = static_cast<int>(std::floor(total_S * (ucb(id) / sum_ucb)));
        out[id] = std::max(1, Sm);
        assigned += out[id];
    }
    while (assigned < total_S) {
        double best = -1;
        std::string best_id;
        for (const auto& [id, _] : state_) {
            double u = ucb(id);
            if (u > best) { best = u; best_id = id; }
        }
        out[best_id]++;
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
