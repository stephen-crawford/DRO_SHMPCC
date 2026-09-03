// test_pruning_and_linearization: verifies
//   (A) de Groot 2023 (arXiv:2307.01070) Definition-2 geometric dominance pruning
//       (prune_dominated_scenarios): dominance is certified by an EXACT closed-form
//       bounded-domain half-space implication test on the actual (a, b) per ego disc,
//       so it is sound (never over-prunes). Checks: a genuinely redundant (farther,
//       same-side, collinear) scenario is removed; opposite-side scenarios are BOTH
//       kept
//   (B) fixed collision half-space construction (compute_linearized_constraints /
//       make_collision_halfspace): unit normal, boundary at the safety radius, sign.
//   (C) heading-Jacobian constraint linearization (linearize_disc_halfspace).
#include "collision_constraints.hpp"
#include "types.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace dro_mpc;
static int fails = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++fails;
}
static bool approx(double a, double b, double t = 1e-6) { return std::abs(a - b) < t; }

// Build a scenario with one obstacle at the given per-step positions (zero covariance).
static Scenario make_scenario(int id, int obs_id,
                              const std::vector<Eigen::Vector2d>& positions) {
    ObstacleTrajectory traj;
    traj.obstacle_id = obs_id;
    traj.mode_id = "m";
    for (size_t k = 0; k < positions.size(); ++k)
        traj.steps.emplace_back(static_cast<int>(k), positions[k], Eigen::Matrix2d::Zero());
    return Scenario(id, {{obs_id, traj}});
}

static bool has_scenario(const std::vector<Scenario>& v, int id) {
    for (const auto& s : v) if (s.scenario_id == id) return true;
    return false;
}

int main() {
    // Ego reference: straight line along +x, obstacle offsets are lateral (in y).
    std::vector<EgoState> ego;
    for (int k = 0; k <= 5; ++k) ego.emplace_back(static_cast<double>(k), 0.0, 0.0, 1.0);
    auto lat = [&](double y) {
        std::vector<Eigen::Vector2d> p;
        for (int k = 0; k <= 5; ++k) p.emplace_back(static_cast<double>(k), y);
        return p;
    };

    std::printf("=== (A) de Groot Definition-2 dominance pruning ===\n");
    // "close" (y=1) is more constraining EVERYWHERE than "far" (y=3), same side:
    // Theta_close subset Theta_far => far is redundant => far removed, close kept.
    Scenario close = make_scenario(1, 0, lat(1.0));
    Scenario far   = make_scenario(2, 0, lat(3.0));
    {
        auto out = prune_dominated_scenarios({close, far}, ego);
        check(out.size() == 1 && has_scenario(out, 1),
              "dominated (farther, same-side) scenario is removed; closer kept");
    }
    // Opposite sides (y=+1 vs y=-2): neither dominates => both kept (non-distorting).
    Scenario other = make_scenario(3, 0, lat(-2.0));
    {
        auto out = prune_dominated_scenarios({close, other}, ego);
        check(out.size() == 2, "opposite-direction scenarios are BOTH kept (no false prune)");
    }
    // Degenerate inputs.
    {
        auto out1 = prune_dominated_scenarios({close}, ego);
        auto out0 = prune_dominated_scenarios({}, ego);
        check(out1.size() == 1 && out0.empty(), "single/empty scenario sets pass through");
    }
    // (#3) Different obstacle-ID sets must NOT dominate even when geometry would nest:
    // 'close' constrains obstacle 0, 'far7' constrains obstacle 7. Neither implies the
    // other's constraint (they are on different obstacles), so both are kept.
    Scenario far7 = make_scenario(5, /*obs_id=*/7, lat(3.0));
    {
        auto out = prune_dominated_scenarios({close, far7}, ego);
        check(out.size() == 2, "scenarios over different obstacle sets are both kept (#3)");
    }
    // Non-distortion: the constraints of the pruned set imply the removed scenario's
    // constraint at the reference (the kept 'close' is tighter than the removed 'far').
    {
        auto cons_all   = compute_linearized_constraints(ego, {close, far}, 0.5, 0.3, 0.2, 1, 1.5);
        auto pruned     = prune_dominated_scenarios({close, far}, ego);
        auto cons_prune = compute_linearized_constraints(ego, pruned, 0.5, 0.3, 0.2, 1, 1.5);
        // At each ego reference point, the tightest clearance is unchanged by pruning.
        double tight_all = 1e9, tight_prune = 1e9;
        for (const auto& c : cons_all)   tight_all   = std::min(tight_all,   c.evaluate(ego[c.k].position()));
        for (const auto& c : cons_prune) tight_prune = std::min(tight_prune, c.evaluate(ego[c.k].position()));
        check(approx(tight_all, tight_prune, 1e-9),
              "pruning is non-distorting: tightest active clearance unchanged");
    }

    std::printf("\n=== (B) collision half-space construction ===\n");
    // Single obstacle at (5,0); ego reference disc at (0,0); R = 0.5+0.3+0.2 = 1.0.
    Scenario onaxis = make_scenario(10, 0, {Eigen::Vector2d(5.0, 0.0)});
    std::vector<EgoState> ego0 = { EgoState(0.0, 0.0, 0.0, 1.0) };
    auto cons = compute_linearized_constraints(ego0, {onaxis}, 0.5, 0.3, 0.2, 1, 1.5);
    check(cons.size() == 1, "one halfspace per (scenario, step, disc)");
    if (!cons.empty()) {
        const auto& c = cons.front();
        check(approx(c.a.norm(), 1.0), "constraint normal is a unit vector");
        // Normal points from obstacle toward the ego reference => -x direction.
        check(c.a.x() < -0.99 && approx(c.a.y(), 0.0), "normal points obstacle->ego (-x)");
        // evaluate(p) = a.p - b >= 0 when safe. Reference (0,0) is safe (deep).
        check(c.evaluate(Eigen::Vector2d(0.0, 0.0)) > 0.0, "ego reference satisfies the halfspace");
        // Boundary sits exactly R=1.0 from the obstacle: ego_x = 5 - R = 4.
        check(approx(c.evaluate(Eigen::Vector2d(4.0, 0.0)), 0.0, 1e-6),
              "boundary is exactly one safety radius (R=1) from the obstacle");
        // At the obstacle centre the constraint is violated.
        check(c.evaluate(Eigen::Vector2d(5.0, 0.0)) < 0.0, "obstacle centre violates the halfspace");
    }
    // Degenerate coincident geometry uses the deterministic fallback normal (no NaN).
    {
        auto hs = make_collision_halfspace(Eigen::Vector2d(1.0, 1.0), Eigen::Vector2d(1.0, 1.0), 1.0);
        check(hs.normal.allFinite() && approx(hs.normal.norm(), 1.0) && hs.used_fallback_normal,
              "coincident obstacle/disc falls back to a finite unit normal");
    }

    std::printf("\n=== (C) heading-Jacobian constraint linearization ===\n");
    // linearize_disc_halfspace maps a fixed half-space through J_d(theta):
    //   c_d ~ c_bar + J_d (x - x_bar),  J_d = [[1,0,-l sin th],[0,1, l cos th]],
    // so coefficients = normal^T J_d = [n_x, n_y, n_x(-l sin th) + n_y(l cos th)].
    {
        LinearizedCollisionHalfspace hs;
        hs.normal = Eigen::Vector2d(0.6, 0.8);   // unit
        hs.upper_bound = 2.0;
        const double th = 0.5, ell = 1.3, px = 2.0, py = -1.0;
        hs.reference_disc_center = linearize_disc_center(
            EgoState(px, py, th, 0.0), ell).center;
        auto aff = linearize_disc_halfspace(hs, px, py, th, ell);
        const double exp0 = hs.normal.x();
        const double exp1 = hs.normal.y();
        const double exp2 = hs.normal.x() * (-ell * std::sin(th)) + hs.normal.y() * (ell * std::cos(th));
        check(approx(aff.coefficients(0), exp0) && approx(aff.coefficients(1), exp1),
              "position coefficients equal the fixed normal");
        check(approx(aff.coefficients(2), exp2),
              "theta coefficient is n^T [-l sin th, l cos th] (heading Jacobian)");
        // Zero longitudinal offset => the disc is the center => no heading term.
        auto centered_hs = hs;
        centered_hs.reference_disc_center = Eigen::Vector2d(px, py);
        auto aff0 = linearize_disc_halfspace(centered_hs, px, py, th, 0.0);
        check(approx(aff0.coefficients(2), 0.0),
              "zero disc offset => zero theta coefficient (single-disc case)");
    }

    std::printf("\n=== (D) QP row anchoring (linearize_constraint_at_state) ===\n");
    // An SQP row is a valid first-order model only if its constant and its gradient
    // are taken at the SAME point. build_condensed_qp differentiates about x_ref, so
    // the row's value must be the clearance at x_ref -- NOT at the (earlier)
    // trajectory the normals were frozen on. Regression guard for that bug.
    {
        // Reference the constraints are built from.
        std::vector<EgoState> ref;
        for (int k = 0; k <= 4; ++k) ref.emplace_back(0.15 * k, 0.0, 0.0, 1.5);
        Scenario sc = make_scenario(20, 0, {
            Eigen::Vector2d(1.20, 1.30), Eigen::Vector2d(1.22, 1.24),
            Eigen::Vector2d(1.24, 1.18), Eigen::Vector2d(1.26, 1.12),
            Eigen::Vector2d(1.28, 1.06)});

        for (int num_discs : {1, 3}) {
            const double L = 4.0;
            auto cs = compute_linearized_constraints(ref, {sc}, 0.5, 0.5, 0.2, num_discs, L);

            // The SQP iterate drifts from the construction reference by roughly one
            // receding-horizon step plus the Douglas-Rachford projection.
            std::vector<EgoState> iter;
            for (int k = 0; k <= 4; ++k)
                iter.emplace_back(0.15 * k + 0.15, 0.12, 0.05, 1.5);

            double worst_anchor = 0.0, worst_grad = 0.0, worst_self = 0.0;
            for (const auto& c : cs) {
                const EgoState& x = iter[c.k];
                const auto row = linearize_constraint_at_state(c, x);

                // (1) The row's value IS the clearance the feasibility check reports,
                // so the QP model, the line search and the final check agree.
                const Eigen::Vector2d u(std::cos(x.theta), std::sin(x.theta));
                const double truth = c.evaluate(x.position() + c.disc_offset * u);
                worst_anchor = std::max(worst_anchor, std::abs(row.value - truth));

                // (2) The gradient is the true derivative of a^T c_d(x).
                const double h = 1e-6;
                for (int dim = 0; dim < 3; ++dim) {
                    EgoState hi = x, lo = x;
                    (dim == 0 ? hi.x : dim == 1 ? hi.y : hi.theta) += h;
                    (dim == 0 ? lo.x : dim == 1 ? lo.y : lo.theta) -= h;
                    auto cd = [&](const EgoState& e) {
                        return c.evaluate(e.position() + c.disc_offset *
                                          Eigen::Vector2d(std::cos(e.theta), std::sin(e.theta)));
                    };
                    worst_grad = std::max(worst_grad,
                        std::abs(row.gradient(dim) - (cd(hi) - cd(lo)) / (2 * h)));
                }

                // (3) Evaluated AT the construction reference the anchor must coincide
                // with the frozen disc centre -- the old behaviour, still correct there.
                const auto row0 = linearize_constraint_at_state(c, ref[c.k]);
                worst_self = std::max(worst_self,
                    std::abs(row0.value - (c.a.dot(c.linearization_point) - c.b)));
            }
            char msg[192];
            std::snprintf(msg, sizeof msg,
                "num_discs=%d: row value == exact nonlinear clearance at the iterate "
                "(max err %.2e)", num_discs, worst_anchor);
            check(worst_anchor < 1e-12, msg);
            std::snprintf(msg, sizeof msg,
                "num_discs=%d: gradient matches finite differences (max err %.2e)",
                num_discs, worst_grad);
            check(worst_grad < 1e-5, msg);
            std::snprintf(msg, sizeof msg,
                "num_discs=%d: at the construction reference the anchor reproduces "
                "linearization_point (max err %.2e)", num_discs, worst_self);
            check(worst_self < 1e-12, msg);
        }
    }

    std::printf("\n=== (E) per-step reachable ball ===\n");
    {
        // Growth of 0 must reproduce the constant-ball behaviour exactly.
        auto flat = prune_dominated_scenarios({close, far}, ego, 1.0, 1, 0.0, 1.0e6, 0.0);
        auto dflt = prune_dominated_scenarios({close, far}, ego);
        check(flat.size() == dflt.size(), "growth = 0 reproduces the previous default");

        // A tighter (sound) ball can only certify MORE implications, never fewer.
        auto tight = prune_dominated_scenarios({close, far, other}, ego, 1.0, 1, 0.0, 0.0, 0.8);
        auto loose = prune_dominated_scenarios({close, far, other}, ego, 1.0, 1, 0.0, 1.0e6, 0.0);
        check(tight.size() <= loose.size(), "tighter reachable ball prunes at least as much");
        // ...but still never prunes genuinely opposite-side scenarios.
        check(has_scenario(tight, 1) && has_scenario(tight, 3),
              "opposite-side scenarios survive even the tight ball (no false prune)");
    }

    std::printf("\n%s (%d checks failed)\n",
                fails == 0 ? "ALL PRUNING/CONSTRAINT TESTS PASSED" : "SOME TESTS FAILED", fails);
    return fails == 0 ? 0 : 1;
}
