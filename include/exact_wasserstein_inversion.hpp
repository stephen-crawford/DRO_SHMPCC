// include/exact_wasserstein_inversion.hpp

#ifndef DRO_MPC_EXACT_WASSERSTEIN_INVERSION_HPP
#define DRO_MPC_EXACT_WASSERSTEIN_INVERSION_HPP

#include <vector>

namespace dro_mpc {

struct ExactWassersteinInversionResult {
    // Largest W_D(center, p) among candidate distributions that are not
    // rejected by the exact multinomial Wasserstein test.
    double rho_grid = 0.0;

    // Candidate distribution attaining rho_grid.
    std::vector<double> maximizing_distribution;

    // Number of realized categorical observations.
    int sample_count = 0;

    // Number of possible multinomial count vectors.
    std::size_t multinomial_outcome_count = 0;

    // Number of candidate true distributions tested.
    std::size_t candidate_count = 0;

    // Number of candidate distributions accepted by the exact test.
    std::size_t accepted_candidate_count = 0;

    // Grid denominator L:
    // p_i = k_i / L, sum_i k_i = L.
    int grid_denominator = 0;

    double beta = 0.0;
};

/**
 * @brief Exact multinomial-test inversion on a simplex lattice.
 *
 * For every candidate categorical distribution p on the lattice,
 * computes the exact p-value
 *
 *   pv(p; n_obs)
 *     = P_p[
 *          W_D(N/m, p)
 *          >=
 *          W_D(n_obs/m, p)
 *       ]
 *
 * by summing over every multinomial count vector N.
 *
 * Candidates satisfying pv > beta are retained, and rho_grid is the
 * largest W_D(center,p) among retained candidates.
 *
 * IMPORTANT:
 * The multinomial tail probability is exact for each tested p.
 * The inversion over the continuous simplex is only as fine as the
 * configured lattice.
 */
ExactWassersteinInversionResult
exact_multinomial_wasserstein_inversion(
    const std::vector<int>& observed_counts,
    const std::vector<double>& center,
    const std::vector<std::vector<double>>& ground_cost,
    double beta,
    int grid_denominator
);

}  // namespace dro_mpc

#endif