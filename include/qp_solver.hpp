/**
 * @file qp_solver.hpp
 * @brief ADMM-based Quadratic Programming solver.
 *
 * Solves QPs of the form:
 *   min  0.5 * x^T H x + g^T x
 *   s.t. C x >= d          (inequality constraints)
 *        lb <= x <= ub     (box constraints)
 *
 * Uses Alternating Direction Method of Multipliers (ADMM) — no external
 * solver dependencies, Eigen only.
 */

#ifndef SCENARIO_MPC_QP_SOLVER_HPP
#define SCENARIO_MPC_QP_SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Cholesky>

namespace scenario_mpc {

/**
 * @brief QP problem data.
 */
struct QPProblem {
    Eigen::MatrixXd H;   ///< n x n PSD Hessian
    Eigen::VectorXd g;   ///< n gradient
    Eigen::MatrixXd C;   ///< m x n inequality matrix (Cx >= d)
    Eigen::VectorXd d;   ///< m RHS of inequality constraints
    Eigen::VectorXd lb;  ///< n lower bounds (can be -inf)
    Eigen::VectorXd ub;  ///< n upper bounds (can be +inf)
};

/**
 * @brief QP solver result.
 */
struct QPResult {
    Eigen::VectorXd x;        ///< Optimal primal variable
    bool converged = false;    ///< Whether solver converged
    int iterations = 0;        ///< Number of ADMM iterations used
    double primal_residual = 0.0;
    double dual_residual = 0.0;
};

/**
 * @brief ADMM solver settings.
 */
struct QPSettings {
    double rho = 1.0;             ///< ADMM penalty parameter
    int max_iterations = 200;     ///< Maximum ADMM iterations
    double abs_tol = 1e-4;        ///< Absolute convergence tolerance
    double rel_tol = 1e-3;        ///< Relative convergence tolerance
    bool adaptive_rho = true;     ///< Enable adaptive rho scaling
    double rho_min = 1e-6;        ///< Minimum rho for adaptive scaling
    double rho_max = 1e6;         ///< Maximum rho for adaptive scaling
};

/**
 * @brief ADMM-based QP solver.
 *
 * Reformulates the QP with slack variables:
 *   min  0.5 x^T H x + g^T x
 *   s.t. A_aug x = z,  z in K
 *
 * where A_aug = [C; I], K = {z : z_ineq >= d, lb <= z_box <= ub}
 *
 * ADMM iterations:
 *   x = (H + rho * A^T A)^{-1} (-g + rho * A^T (z - lambda))
 *   z = proj_K(A x + lambda)
 *   lambda += A x - z
 */
class ADMMSolver {
public:
    /**
     * @brief Solve a QP problem.
     * @param prob QP problem data
     * @param settings Solver settings
     * @return QPResult with optimal solution
     */
    QPResult solve(const QPProblem& prob, const QPSettings& settings = {});

    /**
     * @brief Provide a warm-start primal solution.
     * @param x0 Initial guess for x
     */
    void warm_start(const Eigen::VectorXd& x0);

private:
    Eigen::VectorXd x_warm_;
    bool has_warm_start_ = false;
};

}  // namespace scenario_mpc

#endif  // SCENARIO_MPC_QP_SOLVER_HPP
