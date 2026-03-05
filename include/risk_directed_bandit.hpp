/**
 * @file risk_directed_bandit.hpp
 * @brief Risk-directed scenario allocation via UCB over mode-conditioned violation rate (Section 6.3).
 */

#ifndef SCENARIO_MPC_RISK_DIRECTED_BANDIT_HPP
#define SCENARIO_MPC_RISK_DIRECTED_BANDIT_HPP

#include "types.hpp"
#include <map>
#include <string>

namespace scenario_mpc {

/**
 * @brief Per-mode bandit state: empirical violation rate and sample count.
 */
struct BanditModeState {
    double R_hat = 0.0;   ///< Empirical mean violation indicator
    int n = 0;            ///< Number of samples from this mode
};

/**
 * @brief UCB allocator: maintain state per mode, compute UCB_m = R_hat_m + beta * sqrt(log(t)/n_m).
 */
class RiskDirectedBandit {
public:
    explicit RiskDirectedBandit(double beta = 1.0) : beta_(beta), t_(0) {}
    void set_modes(const std::vector<std::string>& mode_ids);
    void update(int timestep, const std::string& mode_id, bool violation_observed);
    double ucb(const std::string& mode_id) const;
    std::map<std::string, int> allocate(int total_S) const;

private:
    double beta_;
    int t_;
    std::map<std::string, BanditModeState> state_;
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_RISK_DIRECTED_BANDIT_HPP
