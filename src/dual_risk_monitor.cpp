/**
 * @file dual_risk_monitor.cpp
 * @brief Implementation of dual-variable risk monitor (Section 6.8).
 */

#include "dual_risk_monitor.hpp"
#include <algorithm>
#include <numeric>

namespace scenario_mpc {

double DualRiskMonitor::sigmoid(double x) {
    if (x >= 0) return 1.0 / (1.0 + std::exp(-x));
    double e = std::exp(x);
    return e / (1.0 + e);
}

DualRiskMonitor::DualRiskMonitor(double theta0, double theta1, double theta2, double tau)
    : theta0_(theta0), theta1_(theta1), theta2_(theta2), tau_(tau) {}

double DualRiskMonitor::predict_risk(const MPCTraceFeatures& features) const {
    double margin_violation = 0.0;
    for (double g : features.constraint_margins) {
        if (g < 0) margin_violation += (-g);
    }
    double sum_duals = std::accumulate(
        features.dual_magnitudes.begin(), features.dual_magnitudes.end(), 0.0
    );
    double x = theta0_ + theta1_ * margin_violation + theta2_ * sum_duals;
    return sigmoid(x);
}

bool DualRiskMonitor::should_trigger_intervention(const MPCTraceFeatures& features) const {
    return predict_risk(features) > tau_;
}

void DualRiskMonitor::set_threshold(double tau) { tau_ = tau; }

}  // namespace scenario_mpc
