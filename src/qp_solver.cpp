/**
 * @file qp_solver.cpp
 * @brief acados/HPIPM implementation of the controller's dense QP interface.
 */

#include "qp_solver.hpp"

#include <acados/dense_qp/dense_qp_hpipm.h>
#include <acados/utils/types.h>
#include <acados_c/dense_qp_interface.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#ifndef DRO_MPC_ACADOS_GIT_COMMIT
#define DRO_MPC_ACADOS_GIT_COMMIT "unknown"
#endif

namespace dro_mpc {
namespace {

bool finite_or_infinite(const Eigen::VectorXd& values) {
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        if (std::isnan(values(i))) return false;
    }
    return true;
}

QPResult invalid_problem_result(int n) {
    QPResult result;
    result.x = Eigen::VectorXd::Zero(std::max(0, n));
    result.primal_residual = std::numeric_limits<double>::infinity();
    result.dual_residual = std::numeric_limits<double>::infinity();
    return result;
}

bool has_finite_lower_bound(const Eigen::VectorXd& lower, int i) {
    return lower.size() != 0 && std::isfinite(lower(i));
}

bool has_finite_upper_bound(const Eigen::VectorXd& upper, int i) {
    return upper.size() != 0 && std::isfinite(upper(i));
}

}  // namespace

struct AcadosQPSolver::Impl {
    struct Workspace {
        dense_qp_dims dims{};
        qp_solver_config* config = nullptr;
        dense_qp_in* input = nullptr;
        dense_qp_out* output = nullptr;
        void* options = nullptr;
        dense_qp_solver* solver = nullptr;
        std::vector<int> box_indices;

        Workspace(int num_variables, int num_box_bounds,
                  int num_general_constraints, const QPSettings& settings) {
            dims.nv = num_variables;
            dims.ne = 0;
            dims.nb = num_box_bounds;
            dims.ng = num_general_constraints;
            dims.ns = 0;

            dense_qp_solver_plan plan{};
            plan.qp_solver = DENSE_QP_HPIPM;
            config = dense_qp_config_create(&plan);
            if (config == nullptr) return;

            input = dense_qp_in_create(config, &dims);
            options = dense_qp_opts_create(config, &dims);
            output = dense_qp_out_create(config, &dims);
            if (input == nullptr || options == nullptr || output == nullptr) return;

            configure(settings);
            solver = dense_qp_create(config, &dims, options);

            box_indices.resize(num_box_bounds);
            for (int i = 0; i < num_box_bounds; ++i) box_indices[i] = i;
        }

        ~Workspace() {
            std::free(solver);
            std::free(output);
            std::free(options);
            std::free(input);
            std::free(config);
        }

        bool valid() const {
            return config != nullptr && input != nullptr && output != nullptr &&
                   options != nullptr && solver != nullptr;
        }

        void configure(const QPSettings& settings) {
            if (options == nullptr) return;
            auto* hpipm_options = static_cast<dense_qp_hpipm_opts*>(options);
            if (hpipm_options->hpipm_opts == nullptr) return;

            const int max_iterations = std::max(1, settings.max_iterations);
            const double tolerance = settings.tolerance > 0.0
                ? settings.tolerance : 1e-6;
            auto* ipm_options = hpipm_options->hpipm_opts;
            ipm_options->iter_max = max_iterations;
            ipm_options->stat_max = max_iterations;
            ipm_options->res_g_max = tolerance;
            ipm_options->res_b_max = tolerance;
            ipm_options->res_d_max = tolerance;
            ipm_options->res_m_max = tolerance;
            hpipm_options->print_level = 0;
        }
    };

    using WorkspaceKey = std::tuple<int, int, int, int>;
    std::map<WorkspaceKey, std::unique_ptr<Workspace>> workspaces;

    Workspace& workspace_for(int num_variables, int num_box_bounds,
                             int num_general_constraints,
                             const QPSettings& settings) {
        const int max_iterations = std::max(1, settings.max_iterations);
        const WorkspaceKey key{
            num_variables, num_box_bounds, num_general_constraints, max_iterations};
        auto [it, inserted] = workspaces.try_emplace(
            key, std::make_unique<Workspace>(
                num_variables, num_box_bounds, num_general_constraints, settings));
        if (!inserted) it->second->configure(settings);
        return *it->second;
    }
};

AcadosQPSolver::AcadosQPSolver() : impl_(std::make_unique<Impl>()) {}
AcadosQPSolver::~AcadosQPSolver() = default;

QPResult AcadosQPSolver::solve(const QPProblem& problem,
                               const QPSettings& settings) {
    const int n = static_cast<int>(problem.H.rows());
    const int m = static_cast<int>(problem.C.rows());
    if (n <= 0 || problem.H.cols() != n || problem.g.size() != n ||
        problem.C.cols() != n || problem.d.size() != m ||
        (problem.lb.size() != 0 && problem.lb.size() != n) ||
        (problem.ub.size() != 0 && problem.ub.size() != n) ||
        !problem.H.allFinite() || !problem.g.allFinite() ||
        !problem.C.allFinite() || !problem.d.allFinite() ||
        !finite_or_infinite(problem.lb) || !finite_or_infinite(problem.ub)) {
        return invalid_problem_result(n);
    }

    const bool use_box_bounds = problem.lb.size() == n || problem.ub.size() == n;
    const int num_box_bounds = use_box_bounds ? n : 0;
    Impl::Workspace& workspace = impl_->workspace_for(
        n, num_box_bounds, m, settings);
    if (!workspace.valid()) return invalid_problem_result(n);

    // HPIPM consumes column-major arrays; Eigen::MatrixXd has the same layout.
    Eigen::MatrixXd H = 0.5 * (problem.H + problem.H.transpose());
    Eigen::VectorXd g = problem.g;
    Eigen::MatrixXd C = problem.C;
    Eigen::VectorXd d = problem.d;
    Eigen::VectorXd lower = Eigen::VectorXd::Zero(num_box_bounds);
    Eigen::VectorXd upper = Eigen::VectorXd::Zero(num_box_bounds);
    Eigen::VectorXd lower_mask = Eigen::VectorXd::Zero(num_box_bounds);
    Eigen::VectorXd upper_mask = Eigen::VectorXd::Zero(num_box_bounds);
    for (int i = 0; i < num_box_bounds; ++i) {
        if (has_finite_lower_bound(problem.lb, i)) {
            lower(i) = problem.lb(i);
            lower_mask(i) = 1.0;
        }
        if (has_finite_upper_bound(problem.ub, i)) {
            upper(i) = problem.ub(i);
            upper_mask(i) = 1.0;
        }
    }

    // C x >= d is represented as a lower general bound. The upper side is
    // explicitly masked out: using a merely large finite upper bound gives
    // HPIPM an unnecessary barrier term and can cause avoidable MAXITER exits.
    Eigen::VectorXd general_upper = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd lower_general_mask = Eigen::VectorXd::Ones(m);
    Eigen::VectorXd upper_general_mask = Eigen::VectorXd::Zero(m);

    d_dense_qp_set_all(
        H.data(), g.data(),
        nullptr, nullptr,
        num_box_bounds > 0 ? workspace.box_indices.data() : nullptr,
        num_box_bounds > 0 ? lower.data() : nullptr,
        num_box_bounds > 0 ? upper.data() : nullptr,
        m > 0 ? C.data() : nullptr,
        m > 0 ? d.data() : nullptr,
        m > 0 ? general_upper.data() : nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        workspace.input);

    if (num_box_bounds > 0) {
        d_dense_qp_set_lb_mask(lower_mask.data(), workspace.input);
        d_dense_qp_set_ub_mask(upper_mask.data(), workspace.input);
    }
    if (m > 0) {
        d_dense_qp_set_lg_mask(lower_general_mask.data(), workspace.input);
        d_dense_qp_set_ug_mask(upper_general_mask.data(), workspace.input);
    }

    QPResult result;
    result.x.resize(n);
    result.status = dense_qp_solve(workspace.solver, workspace.input, workspace.output);
    d_dense_qp_sol_get_v(workspace.output, result.x.data());

    qp_info* info = nullptr;
    dense_qp_out_get(workspace.output, "qp_info", &info);
    if (info != nullptr) result.iterations = info->num_iter;

    double residuals[4] = {0.0, 0.0, 0.0, 0.0};
    dense_qp_inf_norm_residuals(
        &workspace.dims, workspace.input, workspace.output, residuals);
    result.dual_residual = std::max(residuals[0], residuals[3]);
    result.primal_residual = std::max(residuals[1], residuals[2]);
    result.converged = (result.status == ACADOS_SUCCESS);
    return result;
}

void AcadosQPSolver::clear() {
    impl_->workspaces.clear();
}

const char* AcadosQPSolver::backend_name() {
    return "acados_hpipm_dense_qp";
}

std::string AcadosQPSolver::backend_identity() {
    return std::string("acados-hpipm@") + DRO_MPC_ACADOS_GIT_COMMIT;
}

}  // namespace dro_mpc
