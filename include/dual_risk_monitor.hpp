/**
 * @file dual_risk_monitor.hpp
 * @brief Risk certificates from MPC dual variables / constraint activity (Section 6.8).
 */

#ifndef SCENARIO_MPC_DUAL_RISK_MONITOR_HPP
#define SCENARIO_MPC_DUAL_RISK_MONITOR_HPP

#include <vector>
#include <cmath>

namespace scenario_mpc {

/**
 * @brief Features extracted from MPC trace (margins and duals).
 */
struct MPCTraceFeatures {
    std::vector<double> constraint_margins;  ///< g_i (positive = satisfied)
    std::vector<double> dual_magnitudes;      ///< |lambda_i|
    int solver_iterations = 0;
};

/**
 * @brief Simple risk predictor: r_hat = sigmoid(theta0 + theta1 * margin_violation + theta2 * sum_duals).
 */
class DualRiskMonitor {
public:
    DualRiskMonitor(double theta0 = 0.0, double theta1 = 1.0, double theta2 = 0.1, double tau = 0.5);
    double predict_risk(const MPCTraceFeatures& features) const;
    bool should_trigger_intervention(const MPCTraceFeatures& features) const;
    void set_threshold(double tau);

private:
    double theta0_, theta1_, theta2_, tau_;
    static double sigmoid(double x);
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_DUAL_RISK_MONITOR_HPP
