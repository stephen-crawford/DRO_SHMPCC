// test_projection: locks in the Euclidean warmstart projection onto collision
// half-spaces (project_warmstart_to_safety / project_state_to_collision_halfspace).
//
// Contract (consistent with the half-space notation block in collision_constraints.cpp):
//   a constraint is a^T p >= b, evaluate(p) = a^T p - b is the signed clearance
//   (>= 0 safe). A violating warmstart point is translated by the CLOSEST correction
//   onto the boundary, along +a (AWAY from the obstacle):
//       correction = -(a^T c_d - b)/||a||^2 * a.
//   Multi-disc: the corrected quantity is the DISC CENTER c_d = p + ℓ_d·(cosθ, sinθ),
//   not the vehicle center. Feasible points are untouched; the sweep is idempotent.
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
static bool approx(double a, double b, double t = 1e-9) { return std::abs(a - b) < t; }

template <typename F>
static bool throws_invalid(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

// Half-space a^T p >= b with a=(-1,0), b=-4  <=>  p_x <= 4  (obstacle at x=5, R=1).
static CollisionConstraint left_of_four(int k) {
    CollisionConstraint c(k, /*obs=*/0, /*scen=*/0, Eigen::Vector2d(-1.0, 0.0), -4.0);
    return c;
}

int main() {
    std::printf("=== (A) violating point projected exactly onto the boundary ===\n");
    {
        std::vector<EgoState> traj = { EgoState(6.0, 0.0, 0.0, 1.0) };  // p_x=6 > 4 (violated)
        auto c = left_of_four(0);
        check(c.evaluate(Eigen::Vector2d(6.0, 0.0)) < 0.0, "precondition: warmstart violates the halfspace");
        int n = project_warmstart_to_safety(traj, {c});
        check(n == 1, "exactly one projection performed");
        check(approx(traj[0].x, 4.0) && approx(traj[0].y, 0.0),
              "projected onto the boundary p_x = 4 (closest point)");
        check(approx(c.evaluate(traj[0].position()), 0.0),
              "signed clearance is ~0 at the boundary (now feasible)");
    }

    std::printf("\n=== (B) closest correction purely along +a; pushes a colliding point out to R ===\n");
    {
        auto c = left_of_four(0);
        // On-axis: point INSIDE the R=1 disc of the obstacle at (5,0) (dist 0.5 < R).
        // Projection along +a=(-1,0) lands it on the tangent boundary x=4 (dist = R = 1).
        std::vector<EgoState> t1 = { EgoState(4.5, 0.0, 0.0, 1.0) };
        project_warmstart_to_safety(t1, {c});
        check(approx(t1[0].x, 4.0), "on-axis: corrected to boundary x=4 (moved exactly the depth 0.5)");
        check(approx((t1[0].position() - Eigen::Vector2d(5.0, 0.0)).norm(), 1.0),
              "on-axis colliding point pushed out to exactly clearance R=1");
        // Off-axis: correction is purely along a (x only); y is invariant; still moves away.
        std::vector<EgoState> t2 = { EgoState(4.5, 2.0, 0.0, 1.0) };
        double before = (t2[0].position() - Eigen::Vector2d(5.0, 0.0)).norm();
        project_warmstart_to_safety(t2, {c});
        double after = (t2[0].position() - Eigen::Vector2d(5.0, 0.0)).norm();
        check(approx(t2[0].x, 4.0) && approx(t2[0].y, 2.0),
              "off-axis: only x corrected (motion purely along a), y unchanged");
        check(after > before, "off-axis projection still increases distance to the obstacle");
    }

    std::printf("\n=== (C) feasible points untouched + idempotence ===\n");
    {
        std::vector<EgoState> traj = { EgoState(0.0, 0.0, 0.0, 1.0) };  // p_x=0 <= 4 (safe, deep)
        auto c = left_of_four(0);
        int n = project_warmstart_to_safety(traj, {c});
        check(n == 0 && approx(traj[0].x, 0.0), "already-feasible point is not moved");
        // Project a violating point, then project again -> second pass is a no-op.
        std::vector<EgoState> traj2 = { EgoState(6.0, 0.0, 0.0, 1.0) };
        project_warmstart_to_safety(traj2, {c});
        int n2 = project_warmstart_to_safety(traj2, {c});
        check(n2 == 0 && approx(traj2[0].x, 4.0), "projection is idempotent (converged is a fixed point)");
    }

    std::printf("\n=== (D) multi-disc: the DISC CENTER is corrected, not the vehicle center ===\n");
    {
        // Front disc offset ℓ=1, heading 0 => disc center = position + (1,0). Require the
        // disc (not the center) to satisfy p_x <= 4, so the center must land at x = 3.
        std::vector<EgoState> traj = { EgoState(5.0, 0.0, 0.0, 1.0) };  // disc at x=6 (violates)
        CollisionConstraint c = left_of_four(0);
        c.disc_offset = 1.0; c.disc_index = 1;
        int n = project_warmstart_to_safety(traj, {c});
        check(n == 1 && approx(traj[0].x, 3.0),
              "vehicle center moved to x=3 so the front disc center sits on the boundary x=4");
    }

    std::printf("\n=== (E) cyclic projection converges for multiple half-spaces ===\n");
    {
        // p_x <= 4 (a=(-1,0)) AND p_y <= 2 (a=(0,-1), b=-2). Point (6,5) violates both.
        std::vector<EgoState> traj = { EgoState(6.0, 5.0, 0.0, 1.0) };
        CollisionConstraint cx = left_of_four(0);
        CollisionConstraint cy(0, 1, 0, Eigen::Vector2d(0.0, -1.0), -2.0);
        project_warmstart_to_safety(traj, {cx, cy}, /*sweeps=*/10);
        check(cx.evaluate(traj[0].position()) >= -1e-6 && cy.evaluate(traj[0].position()) >= -1e-6,
              "both half-spaces satisfied after cyclic sweeps");
        check(approx(traj[0].x, 4.0, 1e-6) && approx(traj[0].y, 2.0, 1e-6),
              "converged to the corner of the feasible box (4, 2)");
    }

    std::printf("\n=== (F) out-of-range constraints skipped; other steps untouched ===\n");
    {
        std::vector<EgoState> traj = { EgoState(6.0, 0.0, 0.0, 1.0), EgoState(6.0, 0.0, 0.0, 1.0) };
        auto c_bad_neg = left_of_four(-1);   // k < 0 -> ignored
        auto c_bad_big = left_of_four(5);    // k >= size -> ignored
        int n = project_warmstart_to_safety(traj, {c_bad_neg, c_bad_big});
        check(n == 0 && approx(traj[0].x, 6.0) && approx(traj[1].x, 6.0),
              "constraints with out-of-range k are skipped (trajectory unchanged)");
        // Only step 1 is constrained -> only trajectory[1] moves.
        auto c1 = left_of_four(1);
        project_warmstart_to_safety(traj, {c1});
        check(approx(traj[0].x, 6.0) && approx(traj[1].x, 4.0),
              "only the constrained step is projected; other steps left alone");
    }

    std::printf("\n=== (G) argument validation ===\n");
    {
        std::vector<EgoState> traj = { EgoState(6.0, 0.0, 0.0, 1.0) };
        auto c = left_of_four(0);
        check(throws_invalid([&]{ project_warmstart_to_safety(traj, {c}, /*sweeps=*/0); }),
              "max_projection_sweeps <= 0 throws");
        check(throws_invalid([&]{ project_warmstart_to_safety(traj, {c}, 10, /*tol=*/-1.0); }),
              "negative tolerance throws");
    }

    std::printf("\n%s (%d checks failed)\n",
                fails == 0 ? "ALL PROJECTION TESTS PASSED" : "SOME TESTS FAILED", fails);
    return fails == 0 ? 0 : 1;
}
