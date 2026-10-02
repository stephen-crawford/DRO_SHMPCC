    #include "primal_ot.hpp"

    #include <cmath>
    #include <algorithm>
    #include <limits>
    #include <numeric>
    
    namespace dro_mpc {
    
    namespace {
    
    // Dense Big-M primal simplex: minimize c.x s.t. A x = b (b >= 0), x >= 0.
    // `basis` gives the initial basic variable index per row (must form a feasible
    // identity basis with the given b). Bland's rule guarantees termination.
    // Returns the optimal x (size n); `ok` reports convergence.
    std::vector<double> simplex_min(
        std::vector<std::vector<double>> A,   // m x n  (mutated: becomes B^{-1}A)
        std::vector<double> b,                // m       (mutated: becomes B^{-1}b)
        const std::vector<double>& c,         // n
        std::vector<int> basis,               // m
        bool& ok)
    {
        const int m = static_cast<int>(A.size());
        const int n = static_cast<int>(c.size());
        const double EPS = 1e-9;
    
        // Objective (reduced-cost) row d_j = c_j - z_j, with RHS = -objective.
        std::vector<double> d(n, 0.0);
        for (int j = 0; j < n; ++j) d[j] = c[j];
        double obj_rhs = 0.0;
        // Fold basic costs in so basic columns have reduced cost 0.
        for (int i = 0; i < m; ++i) {
            double cb = c[basis[i]];
            if (cb == 0.0) continue;
            for (int j = 0; j < n; ++j) d[j] -= cb * A[i][j];
            obj_rhs -= cb * b[i];
        }
    
        for (int iter = 0; iter < 20000; ++iter) {
            // Entering: smallest index j with d_j < -EPS (Bland).
            int enter = -1;
            for (int j = 0; j < n; ++j) {
                if (d[j] < -EPS) { enter = j; break; }
            }
            if (enter < 0) { ok = true; break; }  // optimal
    
            // Ratio test: min b_i / A[i][enter] over A[i][enter] > EPS.
            int leave = -1;
            double best_ratio = std::numeric_limits<double>::infinity();
            for (int i = 0; i < m; ++i) {
                double a = A[i][enter];
                if (a > EPS) {
                    double ratio = b[i] / a;
                    // Bland tie-break: prefer smaller basis variable index.
                    if (ratio < best_ratio - EPS ||
                        (std::abs(ratio - best_ratio) <= EPS &&
                         (leave < 0 || basis[i] < basis[leave]))) {
                        best_ratio = ratio;
                        leave = i;
                    }
                }
            }
            if (leave < 0) { ok = false; return {}; }  // unbounded (shouldn't happen)
    
            // Pivot on (leave, enter).
            double piv = A[leave][enter];
            for (int j = 0; j < n; ++j) A[leave][j] /= piv;
            b[leave] /= piv;
            for (int i = 0; i < m; ++i) {
                if (i == leave) continue;
                double f = A[i][enter];
                if (f == 0.0) continue;
                for (int j = 0; j < n; ++j) A[i][j] -= f * A[leave][j];
                b[i] -= f * b[leave];
            }
            double fo = d[enter];
            if (fo != 0.0) {
                for (int j = 0; j < n; ++j) d[j] -= fo * A[leave][j];
                obj_rhs -= fo * b[leave];
            }
            basis[leave] = enter;
        }
    
        std::vector<double> x(n, 0.0);
        for (int i = 0; i < m; ++i) x[basis[i]] = b[i];
        return x;
    }
    
    }  // namespace
        
    PrimalOTResult solve_primal_ot(
        const std::map<std::string, double>& nominal_weights,
        const std::map<std::string, double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        const std::vector<std::string>& mode_ids,
        double rho)
    {
        const int M = static_cast<int>(mode_ids.size());

        if (M == 0)
            return {};

        std::vector<double> p(M);
        std::vector<double> risk(M);

        for (int i = 0; i < M; ++i) {
            p[i] = nominal_weights.at(mode_ids[i]);
            risk[i] = risk_vector.at(mode_ids[i]);
        }

        // Run the single authoritative vector implementation.
        PrimalOTResult out =
            solve_primal_ot(p, risk, D, rho);

        if (!out.solved)
            return out;

        // Recover named worst-case probabilities q_j.
        for (const auto& id : mode_ids)
            out.q[id] = 0.0;

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                out.q[mode_ids[j]] += out.plan[i][j];
            }
        }

        return out;
    }

    PrimalOTResult solve_primal_ot(
        const std::vector<double>& nominal_weights,
        const std::vector<double>& risk_vector,
        const std::vector<std::vector<double>>& D,
        double rho,
        const std::vector<double>& destination_lower_bounds)
    {
        PrimalOTResult out;

        const int M =
            static_cast<int>(nominal_weights.size());

        // ---------------------------------------------------------
        // Input validation
        // ---------------------------------------------------------

        if (M == 0 ||
            static_cast<int>(risk_vector.size()) != M ||
            static_cast<int>(D.size()) != M ||
            !std::isfinite(rho) || rho < 0.0 ||
            (!destination_lower_bounds.empty() &&
             static_cast<int>(destination_lower_bounds.size()) != M)) {
            return out;
        }

        for (const auto& row : D) {
            if (static_cast<int>(row.size()) != M)
                return out;
        }

        // ---------------------------------------------------------
        // Normalize nominal source probabilities
        // ---------------------------------------------------------

        std::vector<double> p = nominal_weights;
        const std::vector<double>& risk = risk_vector;

        double p_tot =
            std::accumulate(p.begin(), p.end(), 0.0);

        if (!std::isfinite(p_tot) ||
            p_tot <= 0.0) {
            return out;
        }

        for (double& pi : p) {
            if (!std::isfinite(pi) || pi < 0.0)
                return out;

            pi /= p_tot;
        }

        // ---------------------------------------------------------
        // LP variables
        //
        // pi_ij  : transport plan
        // slack  : unused Wasserstein budget
        // a_i    : Big-M artificial source variables
        // ---------------------------------------------------------

        const int PI = 0;
        const int SLACK = M * M;
        const int L = destination_lower_bounds.empty() ? 0 : M;
        const int SURPLUS0 = M * M + 1;
        const int ART0 = SURPLUS0 + L;

        const int n =
            M * M + 1 + M + 2 * L;

        const int mrows =
            M + 1 + L;

        const double BIG_M = 1e7;

        // ---------------------------------------------------------
        // Objective
        //
        // simplex_min minimizes:
        //
        //   -sum_ij risk[j] * pi_ij
        //
        // which is equivalent to maximizing the worst-case risk.
        // ---------------------------------------------------------

        std::vector<double> c(n, 0.0);

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                c[PI + i * M + j] =
                    -risk[j];
            }
        }

        for (int i = 0; i < M + L; ++i)
            c[ART0 + i] = BIG_M;

        // ---------------------------------------------------------
        // Equality-form LP
        // ---------------------------------------------------------

        std::vector<std::vector<double>> A(
            mrows,
            std::vector<double>(n, 0.0));

        std::vector<double> b(
            mrows,
            0.0);

        // Source marginal:
        //
        // sum_j pi_ij + a_i = p_hat_i
        for (int i = 0; i < M; ++i) {

            for (int j = 0; j < M; ++j)
                A[i][PI + i * M + j] = 1.0;

            A[i][ART0 + i] = 1.0;
            b[i] = p[i];
        }

        // Wasserstein budget:
        //
        // sum_ij D_ij pi_ij + slack = rho
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {

                if (!std::isfinite(D[i][j]) ||
                    D[i][j] < 0.0) {
                    return out;
                }

                A[M][PI + i * M + j] =
                    D[i][j];
            }
        }

        A[M][SLACK] = 1.0;
        b[M] = rho;

        // ---------------------------------------------------------
        // Initial feasible Big-M basis
        // ---------------------------------------------------------

        std::vector<int> basis(mrows);

        for (int i = 0; i < M; ++i)
            basis[i] = ART0 + i;

        basis[M] = SLACK;

        // Destination floor: sum_i pi_ij - surplus_j + artificial_j = lower_j.
        for (int j = 0; j < L; ++j) {
            const double lower = destination_lower_bounds[j];
            if (!std::isfinite(lower) || lower < 0.0 || lower > 1.0)
                return out;
            const int row = M + 1 + j;
            for (int i = 0; i < M; ++i) A[row][i * M + j] = 1.0;
            A[row][SURPLUS0 + j] = -1.0;
            A[row][ART0 + M + j] = 1.0;
            b[row] = lower;
            basis[row] = ART0 + M + j;
        }

        // ---------------------------------------------------------
        // Solve LP
        // ---------------------------------------------------------

        bool ok = false;

        std::vector<double> x =
            simplex_min(
                A,
                b,
                c,
                basis,
                ok);

        if (!ok || x.empty())
            return out;

        // ---------------------------------------------------------
        // Artificial variables MUST vanish.
        // Otherwise the original OT problem was not satisfied.
        // ---------------------------------------------------------

        double artificial_mass = 0.0;

        for (int i = 0; i < M + L; ++i) {
            artificial_mass +=
                std::max(0.0, x[ART0 + i]);
        }

        if (artificial_mass > 1e-8)
            return out;

        // ---------------------------------------------------------
        // Extract transport plan
        // ---------------------------------------------------------

        out.plan.assign(
            M,
            std::vector<double>(M, 0.0));

        std::vector<double> q(M, 0.0);

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {

                double v =
                    x[PI + i * M + j];

                if (v < -1e-9)
                    return PrimalOTResult{};

                if (std::abs(v) < 1e-12)
                    v = 0.0;

                out.plan[i][j] = v;

                q[j] += v;

                out.transport_cost +=
                    v * D[i][j];
            }
        }

        // ---------------------------------------------------------
        // Validate source marginal
        // ---------------------------------------------------------

        for (int i = 0; i < M; ++i) {

            double row_sum = 0.0;

            for (int j = 0; j < M; ++j)
                row_sum += out.plan[i][j];

            if (std::abs(row_sum - p[i]) > 1e-7)
                return PrimalOTResult{};
        }

        // ---------------------------------------------------------
        // Validate resulting worst-case distribution
        // ---------------------------------------------------------

        double q_sum = 0.0;

        for (double qj : q) {
            if (!std::isfinite(qj) ||
                qj < -1e-9) {
                return PrimalOTResult{};
            }

            q_sum += qj;
        }

        if (std::abs(q_sum - 1.0) > 1e-7)
            return PrimalOTResult{};

        // ---------------------------------------------------------
        // Verify Wasserstein budget
        // ---------------------------------------------------------

        if (!std::isfinite(out.transport_cost) ||
            out.transport_cost > rho + 1e-7) {
            return PrimalOTResult{};
        }

        // ---------------------------------------------------------
        // Compute exact ambiguous expectation
        //
        // rho[xi] = q*^T xi
        // ---------------------------------------------------------

        out.expected_risk = 0.0;

        for (int j = 0; j < M; ++j)
            out.expected_risk += risk[j] * q[j];

        for (int j = 0; j < L; ++j) {
            if (q[j] < destination_lower_bounds[j] - 1e-8)
                return PrimalOTResult{};
        }
        out.solved = true;

        return out;
    }
    
    DominatingOTResult solve_dominating_ot(
        const std::vector<double>& p, const std::vector<double>& risk,
        const std::vector<std::vector<double>>& D,
        double rho, double risk_threshold)
    {
        DominatingOTResult out;
        const size_t M = p.size();
        if (M == 0 || risk.size() != M || !std::isfinite(risk_threshold) ||
            risk_threshold <= 0.0) return out;
        double total = 0.0;
        for (size_t j = 0; j < M; ++j) {
            if (!std::isfinite(p[j]) || p[j] <= 0.0 ||
                !std::isfinite(risk[j]) || risk[j] < 0.0) return out;
            total += p[j];
        }
        if (std::abs(total - 1.0) > 1e-10) return out;
        out.envelope.resize(M);
        for (size_t j = 0; j < M; ++j) {
            std::vector<double> coordinate(M, 0.0);
            coordinate[j] = 1.0;
            const auto maximum = solve_primal_ot(p, coordinate, D, rho);
            if (!maximum.solved) return out;
            out.envelope[j] = maximum.expected_risk;
            out.nominal_domination = std::max(out.nominal_domination,
                                             out.envelope[j] / p[j]);
        }
        std::vector<double> lower(M);
        for (size_t j = 0; j < M; ++j) {
            lower[j] = out.envelope[j] / out.nominal_domination;
            if (risk[j] >= risk_threshold) lower[j] = std::max(lower[j], p[j]);
        }
        if (*std::min_element(risk.begin(), risk.end()) ==
            *std::max_element(risk.begin(), risk.end())) {
            out.transport.plan.assign(M, std::vector<double>(M, 0.0));
            for (size_t j = 0; j < M; ++j) out.transport.plan[j][j] = p[j];
            out.transport.expected_risk = std::inner_product(p.begin(), p.end(), risk.begin(), 0.0);
            out.transport.solved = true;
        } else {
            out.transport = solve_primal_ot(p, risk, D, rho, lower);
        }
        if (!out.transport.solved) return out;
        for (size_t j = 0; j < M; ++j) {
            double q = 0.0;
            for (size_t i = 0; i < M; ++i) q += out.transport.plan[i][j];
            if (!(q > 0.0)) { out.transport = {}; return out; }
            out.domination = std::max(out.domination, out.envelope[j] / q);
        }
        return out;
    }

    }  // namespace dro_mpc