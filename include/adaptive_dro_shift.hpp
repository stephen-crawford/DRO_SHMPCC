/**
 * @file adaptive_dro_shift.hpp
 * @brief Distribution shift detection and adaptive Wasserstein radius rho(t) (Section 6.5).
 */

#ifndef SCENARIO_MPC_ADAPTIVE_DRO_SHIFT_HPP
#define SCENARIO_MPC_ADAPTIVE_DRO_SHIFT_HPP

#include <vector>
#include <deque>
#include <cmath>

namespace scenario_mpc {

/**
 * @brief Adaptive DRO radius from online residual drift.
 *
 * rho(t) = rho_min + k * max(0, d_bar_t - mu_0)
 * where d_bar_t is rolling mean of residuals (e.g. prediction error).
 */
class AdaptiveDROShift {
public:
    AdaptiveDROShift(double rho_min = 0.01, double k = 1.0, double mu_0 = 0.0, size_t window = 20);
    void push_residual(double residual);
    double get_rho() const;
    void set_baseline_mean(double mu_0);
    void set_rho_min(double rho_min);
    void set_scale(double k);

private:
    double rho_min_;
    double k_;
    double mu_0_;
    size_t window_;
    std::deque<double> residuals_;
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_ADAPTIVE_DRO_SHIFT_HPP
