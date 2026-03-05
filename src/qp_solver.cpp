/**
 * @file qp_solver.cpp
 * @brief ADMM-based QP solver implementation.
 */

#include "qp_solver.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace scenario_mpc {

void ADMMSolver::warm_start(const Eigen::VectorXd& x0) {
    x_warm_ = x0;
    has_warm_start_ = true;
}

QPResult ADMMSolver::solve(const QPProblem& prob, const QPSettings& settings) {
    const int n = static_cast<int>(prob.H.rows());   // decision variables
    const int m = static_cast<int>(prob.C.rows());    // inequality constraints
    const int m_total = m + n;  // inequality + box constraints

    QPResult result;

    // Handle degenerate case: no constraints
    if (m == 0 && prob.lb.size() == 0) {
        // Unconstrained QP: x = -H^{-1} g
        Eigen::LLT<Eigen::MatrixXd> llt(prob.H);
        if (llt.info() != Eigen::Success) {
            // Fallback: add regularization
            Eigen::MatrixXd H_reg = prob.H + 1e-8 * Eigen::MatrixXd::Identity(n, n);
            llt.compute(H_reg);
        }
        result.x = llt.solve(-prob.g);
        result.converged = true;
        result.iterations = 0;
        return result;
    }

    // Build augmented constraint matrix A_aug = [C; I]
    // z = A_aug * x, where z_ineq = C*x must be >= d, z_box = x must be in [lb, ub]
    Eigen::MatrixXd A_aug(m_total, n);
    A_aug.topRows(m) = prob.C;
    A_aug.bottomRows(n) = Eigen::MatrixXd::Identity(n, n);

    // Set up bounds for box constraints
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
    if (prob.lb.size() == n) lb = prob.lb;
    if (prob.ub.size() == n) ub = prob.ub;

    // Initialize ADMM variables
    double rho = settings.rho;

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    if (has_warm_start_ && x_warm_.size() == n) {
        x = x_warm_;
    }

    Eigen::VectorXd z = Eigen::VectorXd::Zero(m_total);
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m_total);

    // Pre-compute the KKT matrix factorization: (H + rho * A^T A)
    auto factorize = [&](double rho_val) -> Eigen::LLT<Eigen::MatrixXd> {
        Eigen::MatrixXd KKT = prob.H + rho_val * A_aug.transpose() * A_aug;
        // Add small regularization for numerical stability
        KKT.diagonal().array() += 1e-8;
        return Eigen::LLT<Eigen::MatrixXd>(KKT);
    };

    auto llt = factorize(rho);

    // Projection onto the constraint set K
    // z_ineq >= d (inequality constraints), lb <= z_box <= ub (box constraints)
    auto project_K = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
        Eigen::VectorXd proj = v;
        // Inequality part: z_ineq >= d
        for (int i = 0; i < m; ++i) {
            proj(i) = std::max(proj(i), prob.d(i));
        }
        // Box part: lb <= z_box <= ub
        for (int i = 0; i < n; ++i) {
            proj(m + i) = std::clamp(proj(m + i), lb(i), ub(i));
        }
        return proj;
    };

    // ADMM iteration loop
    for (int iter = 0; iter < settings.max_iterations; ++iter) {
        Eigen::VectorXd z_prev = z;

        // x-update: x = (H + rho * A^T A)^{-1} (-g + rho * A^T (z - lambda))
        Eigen::VectorXd rhs = -prob.g + rho * A_aug.transpose() * (z - lambda);
        x = llt.solve(rhs);

        // z-update: z = proj_K(A*x + lambda)
        Eigen::VectorXd Ax = A_aug * x;
        z = project_K(Ax + lambda);

        // lambda-update: lambda += A*x - z
        Eigen::VectorXd r = Ax - z;  // primal residual
        lambda += r;

        // Compute residuals
        Eigen::VectorXd s = rho * A_aug.transpose() * (z - z_prev);  // dual residual
        double primal_res = r.norm();
        double dual_res = s.norm();

        // Check convergence
        double eps_pri = settings.abs_tol * std::sqrt(m_total)
                       + settings.rel_tol * std::max(Ax.norm(), z.norm());
        double eps_dual = settings.abs_tol * std::sqrt(n)
                        + settings.rel_tol * (rho * A_aug.transpose() * lambda).norm();

        if (primal_res < eps_pri && dual_res < eps_dual) {
            result.x = x;
            result.converged = true;
            result.iterations = iter + 1;
            result.primal_residual = primal_res;
            result.dual_residual = dual_res;
            has_warm_start_ = false;
            return result;
        }

        // Adaptive rho update
        if (settings.adaptive_rho && iter > 0 && iter % 10 == 0) {
            double ratio = primal_res / (dual_res + 1e-10);
            if (ratio > 10.0) {
                rho = std::min(rho * 2.0, settings.rho_max);
                llt = factorize(rho);
                // Scale dual variable to account for rho change
                lambda *= 0.5;
            } else if (ratio < 0.1) {
                rho = std::max(rho / 2.0, settings.rho_min);
                llt = factorize(rho);
                lambda *= 2.0;
            }
        }
    }

    // Did not converge — return best iterate
    result.x = x;
    result.converged = false;
    result.iterations = settings.max_iterations;
    result.primal_residual = (A_aug * x - z).norm();
    result.dual_residual = 0.0;
    has_warm_start_ = false;
    return result;
}

}  // namespace scenario_mpc
