/**
 * @file adaptive_dro_shift.cpp
 * @brief Implementation of adaptive DRO radius from shift detection (Section 6.5).
 */

#include "adaptive_dro_shift.hpp"
#include <algorithm>
#include <numeric>

namespace scenario_mpc {

AdaptiveDROShift::AdaptiveDROShift(double rho_min, double k, double mu_0, size_t window)
    : rho_min_(rho_min), k_(k), mu_0_(mu_0), window_(window) {}

void AdaptiveDROShift::push_residual(double residual) {
    residuals_.push_back(residual);
    if (residuals_.size() > window_) {
        residuals_.pop_front();
    }
}

double AdaptiveDROShift::get_rho() const {
    if (residuals_.empty()) return rho_min_;
    double sum = std::accumulate(residuals_.begin(), residuals_.end(), 0.0);
    double d_bar = sum / static_cast<double>(residuals_.size());
    double delta = d_bar - mu_0_;
    return rho_min_ + k_ * std::max(0.0, delta);
}

void AdaptiveDROShift::set_baseline_mean(double mu_0) { mu_0_ = mu_0; }
void AdaptiveDROShift::set_rho_min(double rho_min) { rho_min_ = rho_min; }
void AdaptiveDROShift::set_scale(double k) { k_ = k; }

}  // namespace scenario_mpc
