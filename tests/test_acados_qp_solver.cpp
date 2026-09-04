/**
 * @file test_acados_qp_solver.cpp
 * @brief Regression tests for the acados/HPIPM dense-QP adapter.
 */

#include "qp_solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace dro_mpc;

namespace {

bool near(double actual, double expected, double tolerance = 1e-6) {
    return std::abs(actual - expected) <= tolerance;
}

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

}  // namespace

int main() {
    QPSettings settings;
    settings.max_iterations = 100;
    settings.tolerance = 1e-8;

    AcadosQPSolver solver;

    // min 0.5 ||x - [1, 2]||^2 subject to x0 + x1 >= 4 and box bounds.
    // The unique optimum is [1.5, 2.5], which exercises the one-sided
    // general-constraint mask used by the scenario collision halfspaces.
    QPProblem constrained;
    constrained.H = Eigen::Matrix2d::Identity();
    constrained.g = Eigen::Vector2d(-1.0, -2.0);
    constrained.C.resize(1, 2);
    constrained.C << 1.0, 1.0;
    constrained.d = Eigen::VectorXd::Constant(1, 4.0);
    constrained.lb = Eigen::Vector2d(0.0, 0.0);
    constrained.ub = Eigen::Vector2d(2.0, 3.0);

    const QPResult first = solver.solve(constrained, settings);
    if (!first.converged || first.status != 0)
        return fail("constrained acados QP did not converge");
    if (!near(first.x(0), 1.5) || !near(first.x(1), 2.5))
        return fail("constrained QP solution is incorrect");
    if (first.x.sum() < 4.0 - 1e-7)
        return fail("one-sided lower general constraint was not enforced");

    // Reusing the cached workspace must remain deterministic and consume no RNG.
    const QPResult second = solver.solve(constrained, settings);
    if (!second.converged || !second.x.isApprox(first.x, 1e-12))
        return fail("cached acados solve is not reproducible");

    // Exercise masked box sides: only x0's upper bound is active.
    QPProblem boxed;
    boxed.H = Eigen::Matrix2d::Identity();
    boxed.g = Eigen::Vector2d(-3.0, 1.0);
    boxed.C.resize(0, 2);
    boxed.d.resize(0);
    boxed.lb = Eigen::Vector2d(0.0, -std::numeric_limits<double>::infinity());
    boxed.ub = Eigen::Vector2d(2.0, std::numeric_limits<double>::infinity());

    const QPResult bounded = solver.solve(boxed, settings);
    if (!bounded.converged || !near(bounded.x(0), 2.0) || !near(bounded.x(1), -1.0))
        return fail("masked box bounds are incorrect");

    solver.clear();
    const QPResult after_clear = solver.solve(constrained, settings);
    if (!after_clear.converged || !after_clear.x.isApprox(first.x, 1e-12))
        return fail("clearing the acados workspace changed the deterministic result");
    if (std::string(AcadosQPSolver::backend_name()) != "acados_hpipm_dense_qp" ||
        AcadosQPSolver::backend_identity().empty())
        return fail("acados backend identity is not available for rollout logs");

    std::cout << "PASS: acados HPIPM dense-QP adapter is correct and reproducible\n";
    return 0;
}
