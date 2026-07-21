// True Wasserstein-metric reweighting via the primal optimal-transport LP.
//
// The existing WassersteinDRO recovers Q* with a dual + greedy deterministic
// transport (each source ships ALL its mass to one destination, then two extreme
// plans are convex-mixed to meet the rho budget). That is a heuristic primal
// recovery. This module instead solves the genuine primal OT linear program
//
//     max_{pi >= 0}  sum_{i,j} pi_ij * r_j
//     s.t.  sum_j pi_ij = p_i           for all i   (source marginals fixed)
//           sum_{i,j} pi_ij * D_ij <= rho            (Wasserstein budget)
//
// and returns Q*_j = sum_i pi_ij. Unlike the greedy recovery, the LP permits a
// single source to SPLIT its mass fractionally across destinations, which is the
// true OT optimum when several (symmetric) modes tie at the optimal dual price.
//
// Solved with a self-contained two-phase dense simplex (Bland's rule for
// anti-cycling). Problem size here is tiny (M^2+1 vars, M+1 rows), so this is
// exact and fast.
#ifndef SCENARIO_MPC_PRIMAL_OT_HPP
#define SCENARIO_MPC_PRIMAL_OT_HPP

#include <map>
#include <string>
#include <vector>

namespace dro_mpc {

struct PrimalOTResult {
    std::map<std::string, double> q;                 ///< Q*_j = sum_i pi_ij
    std::vector<std::vector<double>> plan;           ///< pi_ij transport plan
    double expected_risk = 0.0;                      ///< sum_j r_j Q*_j (= LP optimum)
    double transport_cost = 0.0;                     ///< sum_ij D_ij pi_ij
    bool solved = false;                             ///< simplex converged
};

/// Solve the primal OT LP. mode_ids fixes the index order for p, r, D.
PrimalOTResult solve_primal_ot(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids,
    double rho);

}  // namespace dro_mpc

#endif  // SCENARIO_MPC_PRIMAL_OT_HPP
