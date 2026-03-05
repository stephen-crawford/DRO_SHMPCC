/**
 * @file hazard_switch_sampling.hpp
 * @brief Hazard/change-point models for switch-aware scenario allocation (Section 6.2).
 */

#ifndef SCENARIO_MPC_HAZARD_SWITCH_SAMPLING_HPP
#define SCENARIO_MPC_HAZARD_SWITCH_SAMPLING_HPP

#include "types.hpp"
#include <vector>
#include <map>
#include <string>

namespace scenario_mpc {

/**
 * @brief Feature vector for hazard model (time-to-conflict, relative speed, etc.).
 */
struct HazardFeatures {
    double time_to_conflict = 0.0;   ///< Estimated time to closest approach
    double relative_speed = 0.0;    ///< Relative velocity magnitude
    double time_since_last_switch = 1e6;  ///< Steps since last observed switch
    double entropy_of_modes = 0.0;  ///< Entropy of current mode distribution
};

/**
 * @brief Logistic hazard: lambda_m = sigmoid(theta_m^T h).
 * We use a simple scalar form: lambda_m = sigmoid(scale_m * (time_since_last_switch^{-1} + ...)).
 */
class HazardModel {
public:
    HazardModel() = default;
    void set_mode_coefficients(const std::map<std::string, double>& scale_per_mode);
    double hazard(const std::string& mode_id, const HazardFeatures& h) const;

private:
    std::map<std::string, double> scale_per_mode_;
};

/**
 * @brief Reweight mode probabilities by hazard: tilde_pi_m ∝ pi_m * exp(eta * lambda_m).
 */
std::map<std::string, double> reweight_by_hazard(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& hazard_per_mode,
    double eta
);

/**
 * @brief Allocate scenario counts S_m from reweighted probs; optional exploration floor when max hazard > tau.
 *
 * @param reweighted_probs  tilde_pi_m (sum 1)
 * @param total_S           total scenarios
 * @param max_hazard        max_m lambda_m (if > tau, apply exploration floor)
 * @param tau               threshold for "switch imminent"
 * @param exploration_floor minimum scenarios per mode when spike active
 * @return mode_id -> S_m
 */
std::map<std::string, int> allocate_with_hazard_trigger(
    const std::map<std::string, double>& reweighted_probs,
    int total_S,
    double max_hazard,
    double tau = 0.5,
    int exploration_floor = 1
);

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_HAZARD_SWITCH_SAMPLING_HPP
