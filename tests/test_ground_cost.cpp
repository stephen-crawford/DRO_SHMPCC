// Ground cost D[i][j] for the Wasserstein ball over the mode simplex.
//
// D must be a METRIC for rho to mean "radius": if the triangle inequality fails,
// the transport LP silently routes mass through intermediate modes and prices the
// ball with the metric closure of D rather than D itself. These tests pin that
// down for all four DROGroundCostType values, plus the properties that make the
// taxonomy interpretable:
//
//   (A) The 1D Gaussian W1 closed form (a folded-normal mean) agrees with direct
//       numerical integration of \int |F_1 - F_2|.
//   (B) Sliced-W1 quadrature is converged: the integrand is smooth and pi-periodic,
//       so the midpoint rule converges spectrally and 32 directions is exact to
//       machine noise.
//   (C) Metric axioms (D_ii = 0, symmetry, triangle inequality) hold for every
//       ground cost -- checked on modes with DISTINCT covariances so the Bures and
//       sliced-W1 terms are actually exercised.
//   (D) Normalisation: equal covariances collapse W2_BURES and W1_METRIC onto
//       EUCLIDEAN_MEAN exactly, so the three share a scale and diam(D) is comparable.
//   (E) Ordering: EUCLIDEAN_MEAN <= W1_METRIC <= W2_BURES, i.e. W1 is the less
//       pessimistic of the two Gaussian metrics.
//   (F) A mode missing from mode_models must NOT get a zero row -- that would be
//       free transport into and out of it. It is backfilled at diam(D), the largest
//       value that keeps the triangle inequality.
#include "wasserstein_dro.hpp"
#include "dynamics.hpp"
#include "types.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <random>

using namespace dro_mpc;

static int fails = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++fails;
}

namespace {

// Mode set whose covariance trajectories genuinely differ, by giving each mode a
// distinct process-noise gain G. The shipped create_obstacle_mode_models() shares
// one G across all modes, so the covariance half of W2/W1 is inert there; these
// tests need it live.
std::map<std::string, ModeModel> distinct_cov_modes(double dt) {
    std::map<std::string, ModeModel> modes;
    Eigen::Matrix4d A;
    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;

    const struct { const char* id; double bx, by, gain; } spec[] = {
        {"tight",     0.0,  0.0, 0.25},
        {"medium",    0.0,  0.3, 1.00},
        {"wide",      0.0, -0.3, 3.00},
        {"very_wide", -0.5, 0.0, 6.00},
    };
    for (const auto& s : spec) {
        Eigen::MatrixXd G(4, 2);
        G << 0.5 * dt * dt, 0,
             0, 0.5 * dt * dt,
             dt, 0,
             0, dt;
        G *= s.gain;
        Eigen::Vector4d b;
        b << 0, s.by * dt, s.bx * dt, 0;
        modes[s.id] = ModeModel(s.id, A, b, G, s.id);
    }
    return modes;
}

std::vector<std::vector<double>> ground_cost(
    DROGroundCostType type,
    const std::map<std::string, ModeModel>& modes,
    const std::vector<std::string>& ids)
{
    DROConfig cfg;
    cfg.ground_cost_type = type;
    WassersteinDRO dro(cfg);
    dro.set_observation_count(200);

    std::map<std::string, double> nominal;
    for (const auto& id : ids) nominal[id] = 1.0 / static_cast<double>(ids.size());

    ObstacleState obs(6.0, 1.6, -1.0, -0.35);
    std::vector<EgoState> ego;
    for (int k = 0; k <= 12; ++k) {
        EgoState e; e.x = 0.4 * k; e.y = 0.0; e.theta = 0.0; e.v = 4.0;
        ego.push_back(e);
    }
    return dro.compute_worst_case_weights(nominal, obs, modes, ego, 12,
                                          0.5, 0.5, 0.2, -1, 1, 4.0)
             .transport_cost_matrix;
}

struct MetricReport { double diag, asym, tri, min_off, diam; };

MetricReport metric_report(const std::vector<std::vector<double>>& D) {
    const int M = static_cast<int>(D.size());
    MetricReport r{0.0, 0.0, -1e300, 1e300, 0.0};
    for (int i = 0; i < M; ++i) {
        r.diag = std::max(r.diag, std::abs(D[i][i]));
        for (int j = 0; j < M; ++j) {
            r.asym = std::max(r.asym, std::abs(D[i][j] - D[j][i]));
            r.diam = std::max(r.diam, D[i][j]);
            if (i != j) r.min_off = std::min(r.min_off, D[i][j]);
            for (int k = 0; k < M; ++k)
                r.tri = std::max(r.tri, D[i][j] - (D[i][k] + D[k][j]));
        }
    }
    return r;
}

// ground_cost_name returns by value, so bind the string before taking c_str().
std::string name_of(DROGroundCostType t) { return ground_cost_name(t); }

}  // namespace

int main() {
    const double dt = 0.1;
    auto modes = distinct_cov_modes(dt);
    std::vector<std::string> ids;
    for (const auto& [id, _] : modes) ids.push_back(id);

    // ---- (A) 1D Gaussian W1 closed form vs direct integration ---------------
    std::printf("\n(A) 1D Gaussian W1: folded-normal closed form vs \\int |F1 - F2|\n");
    {
        // W1(P,Q) = \int |F_P(x) - F_Q(x)| dx for 1D distributions.
        auto numeric_w1 = [](double m1, double s1, double m2, double s2) {
            const double lo = std::min(m1 - 12 * s1, m2 - 12 * s2);
            const double hi = std::max(m1 + 12 * s1, m2 + 12 * s2);
            const int N = 2000000;
            const double h = (hi - lo) / N;
            auto cdf = [](double x, double m, double s) {
                return 0.5 * std::erfc(-(x - m) / (s * M_SQRT2));
            };
            double acc = 0.0;
            for (int i = 0; i < N; ++i) {
                const double x = lo + (i + 0.5) * h;
                acc += std::abs(cdf(x, m1, s1) - cdf(x, m2, s2));
            }
            return acc * h;
        };
        // Closed form, replicated here since the .cpp helper has internal linkage.
        auto closed_w1 = [](double m1, double s1, double m2, double s2) {
            const double d = m1 - m2, s = std::abs(s1 - s2);
            if (s < 1e-12) return std::abs(d);
            return s * std::sqrt(2.0 / M_PI) * std::exp(-(d * d) / (2 * s * s))
                 + d * std::erf(d / (s * M_SQRT2));
        };
        const double cases[][4] = {
            {0.0, 1.0, 0.0, 1.0},    // identical
            {0.0, 1.0, 2.0, 1.0},    // equal std  -> |d|
            {0.0, 1.0, 0.0, 3.0},    // equal mean -> |ds| sqrt(2/pi)
            {1.0, 0.5, -2.0, 2.5},   // both differ
            {-3.0, 4.0, 1.5, 0.75},
        };
        double worst = 0.0;
        for (const auto& c : cases) {
            const double a = closed_w1(c[0], c[1], c[2], c[3]);
            const double b = numeric_w1(c[0], c[1], c[2], c[3]);
            worst = std::max(worst, std::abs(a - b));
            std::printf("    N(%5.2f,%4.2f) vs N(%5.2f,%4.2f):  closed=%.8f  numeric=%.8f\n",
                        c[0], c[1], c[2], c[3], a, b);
        }
        std::printf("    max |closed - numeric| = %.3e\n", worst);
        check(worst < 1e-5, "1D Gaussian W1 closed form matches numerical integration");
    }

    // ---- (B) sliced-W1 quadrature convergence, in BOTH regimes --------------
    // Distinct covariances => integrand smooth and pi-periodic => spectral.
    // Equal covariances    => integrand is |<theta,dmu>|, kinked => O(n^-2), and
    // no finite positive-weight rule removes that (a finite sum of |<theta_l,v>|
    // is a zonotope gauge, which cannot equal the Euclidean norm). The production
    // direction count is chosen against the second, worse rate.
    std::printf("\n(B) sliced-W1 quadrature convergence\n");
    {
        const Eigen::Vector2d m1(0.0, 0.0), m2(1.3, -0.7);
        Eigen::Matrix2d S1, S2, Sa;
        S1 <<  0.40, 0.15, 0.15, 0.10;      // anisotropic, different orientations
        S2 <<  0.09, -0.05, -0.05, 0.55;
        Sa <<  0.20, 0.03, 0.03, 0.12;      // shared by both arguments below
        auto closed_w1 = [](double a, double s1, double b, double s2) {
            const double d = a - b, s = std::abs(s1 - s2);
            if (s < 1e-12) return std::abs(d);
            return s * std::sqrt(2.0 / M_PI) * std::exp(-(d * d) / (2 * s * s))
                 + d * std::erf(d / (s * M_SQRT2));
        };
        auto sliced = [&](int n, const Eigen::Matrix2d& A, const Eigen::Matrix2d& B) {
            double acc = 0.0;
            for (int l = 0; l < n; ++l) {
                const double t = M_PI * (l + 0.5) / n;
                const Eigen::Vector2d th(std::cos(t), std::sin(t));
                acc += closed_w1(th.dot(m1), std::sqrt(th.dot(A * th)),
                                 th.dot(m2), std::sqrt(th.dot(B * th)));
            }
            return (M_PI / 2.0) * acc / n;
        };

        const double ref_generic = sliced(1 << 16, S1, S2);
        const double ref_degen = (m1 - m2).norm();   // exact in the equal-cov case
        std::printf("    %6s %14s %12s %14s %12s\n",
                    "n", "distinct-cov", "|err|", "equal-cov", "|err|");
        for (int n : {8, 32, 128, 512}) {
            std::printf("    %6d %14.10f %12.2e %14.10f %12.2e\n",
                        n, sliced(n, S1, S2), std::abs(sliced(n, S1, S2) - ref_generic),
                        sliced(n, Sa, Sa), std::abs(sliced(n, Sa, Sa) - ref_degen));
        }
        check(std::abs(sliced(512, S1, S2) - ref_generic) < 1e-12,
              "distinct covariances: spectral, machine precision by n=512");
        check(std::abs(sliced(128, Sa, Sa) - ref_degen) < 1e-4,
              "equal covariances: O(n^-2), within 1e-4 at the production n=128");
    }

    // ---- (C) metric axioms, on modes with genuinely distinct covariances ----
    std::printf("\n(C) metric axioms (mode set has DISTINCT per-mode G, so Bures/W1 are live)\n");
    for (auto t : {DROGroundCostType::ZERO_ONE, DROGroundCostType::EUCLIDEAN_MEAN,
                   DROGroundCostType::W1_METRIC, DROGroundCostType::W2_BURES}) {
        const auto D = ground_cost(t, modes, ids);
        const auto r = metric_report(D);
        const std::string label = name_of(t);
        std::printf("    %-15s D_ii=%.1e  asym=%.1e  worst_tri=%+.2e  min_off=%.4f  diam=%.4f\n",
                    label.c_str(), r.diag, r.asym, r.tri, r.min_off, r.diam);
        char msg[160];
        std::snprintf(msg, sizeof msg, "%s is a metric (zero diagonal, symmetric, triangle)",
                      label.c_str());
        check(r.diag < 1e-12 && r.asym < 1e-12 && r.tri < 1e-9 && r.min_off > 0.0, msg);
    }

    // ---- (D) normalisation: equal covariances => all three coincide ---------
    // W2_BURES collapses EXACTLY (the Bures term is analytically zero when the
    // covariances match). W1_METRIC collapses only to the angular quadrature
    // tolerance, since its degenerate integrand is the kinked |<theta,dmu>|.
    std::printf("\n(D) shared normalisation: equal covariance trajectories collapse\n"
                "    W2_BURES and W1_METRIC onto EUCLIDEAN_MEAN\n");
    {
        // Shipped modes all share one G, so their covariance trajectories agree.
        auto shared = create_obstacle_mode_models(dt);
        std::vector<std::string> sids;
        for (const auto& [id, _] : shared) sids.push_back(id);

        const auto De = ground_cost(DROGroundCostType::EUCLIDEAN_MEAN, shared, sids);
        const auto D1 = ground_cost(DROGroundCostType::W1_METRIC, shared, sids);
        const auto D2 = ground_cost(DROGroundCostType::W2_BURES, shared, sids);
        double g1 = 0.0, g2 = 0.0;
        for (size_t i = 0; i < sids.size(); ++i)
            for (size_t j = 0; j < sids.size(); ++j) {
                g1 = std::max(g1, std::abs(D1[i][j] - De[i][j]));
                g2 = std::max(g2, std::abs(D2[i][j] - De[i][j]));
            }
        std::printf("    max |W1 - EUCL| = %.3e     max |W2 - EUCL| = %.3e\n", g1, g2);
        check(g1 < 1e-4, "W1_METRIC reduces to EUCLIDEAN_MEAN to quadrature tolerance");
        check(g2 < 1e-6, "W2_BURES reduces to EUCLIDEAN_MEAN exactly (Bures term vanishes)");
    }

    // ---- (E) ordering EUCLIDEAN <= W1 <= W2 ---------------------------------
    std::printf("\n(E) ordering: EUCLIDEAN_MEAN <= W1_METRIC <= W2_BURES\n");
    {
        const auto De = ground_cost(DROGroundCostType::EUCLIDEAN_MEAN, modes, ids);
        const auto D1 = ground_cost(DROGroundCostType::W1_METRIC, modes, ids);
        const auto D2 = ground_cost(DROGroundCostType::W2_BURES, modes, ids);
        const int M = static_cast<int>(ids.size());
        bool ok_lo = true, ok_hi = true;
        std::printf("    %-12s %-12s %10s %10s %10s\n", "from", "to", "EUCL", "W1", "W2");
        for (int i = 0; i < M; ++i)
            for (int j = i + 1; j < M; ++j) {
                std::printf("    %-12s %-12s %10.5f %10.5f %10.5f\n",
                            ids[i].c_str(), ids[j].c_str(), De[i][j], D1[i][j], D2[i][j]);
                ok_lo &= (D1[i][j] >= De[i][j] - 1e-9);
                ok_hi &= (D2[i][j] >= D1[i][j] - 1e-9);
            }
        check(ok_lo, "W1_METRIC >= EUCLIDEAN_MEAN (W1 per slice dominates the mean gap)");
        check(ok_hi, "W2_BURES >= W1_METRIC (W2 penalises covariance mismatch more)");
    }

    // ---- (F) a mode absent from mode_models must not get free transport -----
    std::printf("\n(F) mode present in mode_ids but ABSENT from mode_models\n");
    {
        auto partial = modes;
        partial.erase("very_wide");                 // still referenced via `ids`
        const auto D = ground_cost(DROGroundCostType::W2_BURES, partial, ids);
        const auto r = metric_report(D);

        int missing = -1;
        for (size_t i = 0; i < ids.size(); ++i) if (ids[i] == "very_wide") missing = (int)i;
        double row_max = 0.0, row_min = 1e300;
        for (size_t j = 0; j < ids.size(); ++j) {
            if ((int)j == missing) continue;
            row_max = std::max(row_max, D[missing][j]);
            row_min = std::min(row_min, D[missing][j]);
        }
        std::printf("    missing-mode row: min=%.4f max=%.4f (diam of valid submatrix=%.4f)\n",
                    row_min, row_max, r.diam);
        std::printf("    worst triangle violation = %+.2e\n", r.tri);
        check(row_min > 0.0, "missing mode is NOT free to transport into (row is not zero)");
        check(r.tri < 1e-9, "backfilling at diam(D) keeps D a metric");
    }

    // ---- (G) separation: the costs are pseudometrics, and that is fine ------
    // Two mode ids with identical dynamics have identical position tubes, so
    // D_ij = 0 for i != j. Pin this down so nobody "fixes" it by flooring the
    // off-diagonal, which would break the triangle inequality. It is harmless
    // because identical tubes also give identical risk, making the free transport
    // that the zero entry permits a no-op on <q, r>.
    std::printf("\n(G) separation: aliased modes give D_ij = 0 (pseudometric, harmless)\n");
    {
        auto aliased = create_obstacle_mode_models(dt);
        const auto& cv = aliased.at("constant_velocity");
        aliased["cv_duplicate"] = ModeModel("cv_duplicate", cv.A, cv.b, cv.G, "alias");
        std::vector<std::string> aids;
        for (const auto& [id, _] : aliased) aids.push_back(id);

        DROConfig cfg;
        cfg.ground_cost_type = DROGroundCostType::W2_BURES;
        WassersteinDRO dro(cfg);
        dro.set_observation_count(200);
        std::map<std::string, double> nominal;
        for (const auto& id : aids) nominal[id] = 1.0 / static_cast<double>(aids.size());
        ObstacleState obs(6.0, 1.6, -1.0, -0.35);
        std::vector<EgoState> ego;
        for (int k = 0; k <= 12; ++k) {
            EgoState e; e.x = 0.4 * k; e.y = 0.0; e.theta = 0.0; e.v = 4.0; ego.push_back(e);
        }
        const auto res = dro.compute_worst_case_weights(nominal, obs, aliased, ego, 12,
                                                        0.5, 0.5, 0.2, -1, 1, 4.0);
        int a = -1, b = -1;
        for (size_t i = 0; i < aids.size(); ++i) {
            if (aids[i] == "constant_velocity") a = static_cast<int>(i);
            if (aids[i] == "cv_duplicate")      b = static_cast<int>(i);
        }
        const double d_ab = res.transport_cost_matrix[a][b];
        const double r_a = res.risk_per_mode.at("constant_velocity");
        const double r_b = res.risk_per_mode.at("cv_duplicate");
        const auto r = metric_report(res.transport_cost_matrix);
        std::printf("    D[cv][cv_duplicate] = %.3e     r[cv]=%.6f  r[dup]=%.6f\n",
                    d_ab, r_a, r_b);
        std::printf("    worst triangle violation with the alias present = %+.2e\n", r.tri);
        check(d_ab < 1e-9, "aliased modes are at distance zero (pseudometric, as documented)");
        check(std::abs(r_a - r_b) < 1e-12, "...and carry equal risk, so the zero entry is a no-op");
        check(r.tri < 1e-9, "triangle inequality survives the zero off-diagonal entry");
    }

    std::printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASSED",
                fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
