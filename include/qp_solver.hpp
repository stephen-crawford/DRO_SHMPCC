/**
 * @file qp_solver.hpp
 * @brief acados/HPIPM dense quadratic-programming interface.
 *
 * Solves QPs of the form:
 *   min  0.5 * x^T H x + g^T x
 *   s.t. C x >= d          (inequality constraints)
 *        lb <= x <= ub     (box constraints)
 *
 * The controller keeps its existing SQP linearization and condensed problem
 * construction. This interface delegates each QP subproblem to acados's
 * deterministic dense HPIPM backend.
 */

#ifndef SCENARIO_MPC_QP_SOLVER_HPP
#define SCENARIO_MPC_QP_SOLVER_HPP

#include <Eigen/Dense>

#include <memory>
#include <string>

namespace dro_mpc {

/** @brief QP problem data. */
struct QPProblem {
    Eigen::MatrixXd H;   ///< n x n positive-semidefinite Hessian.
    Eigen::VectorXd g;   ///< n gradient.
    Eigen::MatrixXd C;   ///< m x n inequality matrix (C x >= d).
    Eigen::VectorXd d;   ///< m inequality right-hand side.
    Eigen::VectorXd lb;  ///< n lower bounds; +/- infinity disables a side.
    Eigen::VectorXd ub;  ///< n upper bounds; +/- infinity disables a side.
};

/** @brief QP solver result. */
struct QPResult {
    Eigen::VectorXd x;        ///< Best primal solution returned by HPIPM.
    bool converged = false;   ///< True only for an acados success status.
    int iterations = 0;       ///< HPIPM interior-point iterations used.
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    int status = -1;          ///< acados return status (0 is success).
};

/**
 * @brief Runtime settings shared with acados HPIPM.
 *
 * `max_iterations` maps to HPIPM's interior-point iteration cap. The scalar
 * tolerance is applied consistently to stationarity, equality, inequality,
 * and complementarity residuals, so the existing YAML `qp_tolerance` remains
 * the single source of truth for the QP solve.
 */
struct QPSettings {
    int max_iterations = 200;
    double tolerance = 1e-4;
};

/**
 * @brief acados dense-QP solver backed by HPIPM.
 *
 * Workspaces are cached by dimensions and iteration cap. Constraint counts
 * legitimately vary as dominance pruning changes the active scenario set, so
 * each distinct condensed shape gets one reusable acados workspace. The
 * backend consumes no random numbers and starts each QP deterministically.
 */
class AcadosQPSolver {
public:
    AcadosQPSolver();
    ~AcadosQPSolver();

    AcadosQPSolver(const AcadosQPSolver&) = delete;
    AcadosQPSolver& operator=(const AcadosQPSolver&) = delete;
    AcadosQPSolver(AcadosQPSolver&&) = delete;
    AcadosQPSolver& operator=(AcadosQPSolver&&) = delete;

    QPResult solve(const QPProblem& problem, const QPSettings& settings = {});

    /// Release cached acados workspaces while leaving controller RNG state untouched.
    void clear();

    static const char* backend_name();
    static std::string backend_identity();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dro_mpc

#endif  // SCENARIO_MPC_QP_SOLVER_HPP
