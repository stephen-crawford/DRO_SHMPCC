    #include "primal_ot.hpp"

    #include <cmath>
    #include <limits>
    
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
        PrimalOTResult out;
        const int M = static_cast<int>(mode_ids.size());
        if (M == 0) return out;
    
        // Normalize source marginals.
        std::vector<double> p(M), r(M);
        double p_tot = 0.0;
        for (int i = 0; i < M; ++i) { p[i] = nominal_weights.at(mode_ids[i]); p_tot += p[i]; }
        if (p_tot > 0.0) for (int i = 0; i < M; ++i) p[i] /= p_tot;
        for (int j = 0; j < M; ++j) r[j] = risk_vector.at(mode_ids[j]);
    
        // Variable layout: pi_ij at col i*M+j; slack at M*M; artificial a_i at M*M+1+i.
        const int PI = 0;
        const int SLACK = M * M;
        const int ART0 = M * M + 1;
        const int n = M * M + 1 + M;   // total variables
        const int mrows = M + 1;       // M source rows + 1 budget row
    
        const double BIG_M = 1e7;
        std::vector<double> c(n, 0.0);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                c[PI + i * M + j] = -r[j];       // maximize sum r_j pi_ij
        for (int i = 0; i < M; ++i) c[ART0 + i] = BIG_M;
    
        std::vector<std::vector<double>> A(mrows, std::vector<double>(n, 0.0));
        std::vector<double> b(mrows, 0.0);
        // Source rows: sum_j pi_ij + a_i = p_i.
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) A[i][PI + i * M + j] = 1.0;
            A[i][ART0 + i] = 1.0;
            b[i] = p[i];
        }
        // Budget row: sum_ij D_ij pi_ij + slack = rho.
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                A[M][PI + i * M + j] = D[i][j];
        A[M][SLACK] = 1.0;
        b[M] = rho;
    
        std::vector<int> basis(mrows);
        for (int i = 0; i < M; ++i) basis[i] = ART0 + i;   // artificials
        basis[M] = SLACK;                                  // slack basic for budget
    
        bool ok = false;
        std::vector<double> x = simplex_min(A, b, c, basis, ok);
        if (!ok || x.empty()) { out.solved = false; return out; }
    
        out.solved = true;
        out.plan.assign(M, std::vector<double>(M, 0.0));
        for (const auto& id : mode_ids) out.q[id] = 0.0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                double v = x[PI + i * M + j];
                if (v < 1e-12) v = 0.0;
                out.plan[i][j] = v;
                out.q[mode_ids[j]] += v;
                out.transport_cost += v * D[i][j];
            }
        }
        for (int j = 0; j < M; ++j) out.expected_risk += r[j] * out.q[mode_ids[j]];
        return out;
    }
    
    }  // namespace dro_mpc