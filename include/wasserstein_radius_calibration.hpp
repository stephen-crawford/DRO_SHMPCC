// include/wasserstein_radius_calibration.hpp

#ifndef DRO_MPC_WASSERSTEIN_RADIUS_CALIBRATION_HPP
#define DRO_MPC_WASSERSTEIN_RADIUS_CALIBRATION_HPP

#include <vector>

namespace dro_mpc {

struct FiniteSampleWassersteinRadius {
    double rho = 0.0;

    // Simultaneous confidence polytope diagnostics.
    std::vector<double> lower;
    std::vector<double> upper;

    int sample_count = 0;
    int vertex_count = 0;
};

FiniteSampleWassersteinRadius
finite_sample_wasserstein_radius(
    const std::vector<int>& counts,
    const std::vector<double>& center,
    const std::vector<std::vector<double>>& ground_cost,
    double beta
);

}  // namespace dro_mpc

#endif