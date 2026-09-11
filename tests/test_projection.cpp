// test_projection.cpp
//
// Tests the collision geometry and fixed collision half-space representation
// used by Safe-Horizon MPC.
//
// Convention:
//
//   n = (x_obs - c_bar) / ||x_obs - c_bar||
//
// gives the fixed geometric half-space
//
//   n^T c <= n^T x_obs - R.
//
// CollisionConstraint stores the equivalent form
//
//   a^T c >= b,
//   a = -n,
//   b = -(n^T x_obs - R),
//
// so evaluate(c) = a^T c - b is the signed half-space clearance:
//
//   evaluate(c) >= 0  -> safe
//   evaluate(c) <  0  -> violated.
//
// This test intentionally does NOT test project_warmstart_to_safety().
// Geometric warmstart projection is no longer part of the condensed SQP path.

#include "collision_constraints.hpp"
#include "types.hpp"

#include <cmath>
#include <cstdio>
#include <optional>
#include <stdexcept>

using namespace dro_mpc;

namespace {

int fails = 0;

void check(bool ok, const char* message)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", message);

    if (!ok) {
        ++fails;
    }
}

bool approx(double a, double b, double tolerance = 1e-9)
{
    return std::abs(a - b) <= tolerance;
}

template <typename F>
bool throws_invalid(F&& function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }

    return false;
}

}  // namespace


int main()
{
    std::printf(
        "=== (A) fixed collision half-space construction ===\n");

    {
        /*
         * Reference disc center: (0, 0)
         * Obstacle:              (5, 0)
         * Safety radius:          1
         *
         * Therefore
         *
         *   n = (1, 0)
         *
         * and the safe half-space is
         *
         *   x <= 4.
         */

        const Eigen::Vector2d obstacle(5.0, 0.0);
        const Eigen::Vector2d reference_center(0.0, 0.0);

        const auto halfspace =
            make_collision_halfspace(
                obstacle,
                reference_center,
                1.0);

        check(
            approx(halfspace.normal.x(), 1.0) &&
            approx(halfspace.normal.y(), 0.0),
            "normal points from reference disc center toward obstacle");

        check(
            approx(halfspace.upper_bound, 4.0),
            "upper bound produces x <= 4");

        check(
            !halfspace.used_fallback_normal,
            "nondegenerate geometry does not use fallback normal");
    }


    std::printf(
        "\n=== (B) conversion to CollisionConstraint is consistent ===\n");

    {
        auto halfspace =
            make_collision_halfspace(
                Eigen::Vector2d(5.0, 0.0),
                Eigen::Vector2d(0.0, 0.0),
                1.0);

        halfspace.horizon_step = 3;
        halfspace.obstacle_id = 7;
        halfspace.scenario_id = 11;

        const CollisionConstraint constraint =
            halfspace_to_collision_constraint(halfspace);

        /*
         * x <= 4
         *
         * becomes
         *
         * (-1, 0)^T p >= -4.
         */

        check(
            approx(constraint.a.x(), -1.0) &&
            approx(constraint.a.y(), 0.0),
            "stored normal uses equivalent a = -n convention");

        check(
            approx(constraint.b, -4.0),
            "stored offset is b = -upper_bound");

        check(
            constraint.k == 3 &&
            constraint.obstacle_id == 7 &&
            constraint.scenario_id == 11,
            "constraint metadata is preserved");

        check(
            constraint.evaluate(
                Eigen::Vector2d(0.0, 0.0)) > 0.0,
            "point well inside safe half-space has positive clearance");

        check(
            approx(
                constraint.evaluate(
                    Eigen::Vector2d(4.0, 0.0)),
                0.0),
            "boundary point has zero clearance");

        check(
            constraint.evaluate(
                Eigen::Vector2d(5.0, 0.0)) < 0.0,
            "point beyond boundary has negative clearance");
    }


    std::printf(
        "\n=== (C) ego disc-center geometry ===\n");

    {
        /*
         * Vehicle center = (3, 2)
         * heading = 0
         * ell = 1
         *
         * disc center = (4, 2).
         */

        const EgoState state(
            3.0,
            2.0,
            0.0,
            1.0);

        const auto disc =
            linearize_disc_center(
                state,
                1.0);

        check(
            approx(disc.center.x(), 4.0) &&
            approx(disc.center.y(), 2.0),
            "disc center includes longitudinal offset");

        /*
         * At theta = 0 and ell = 1:
         *
         * dc/d[x,y,theta] =
         *
         * [1  0  0]
         * [0  1  1].
         */

        check(
            approx(disc.jacobian(0, 0), 1.0) &&
            approx(disc.jacobian(0, 1), 0.0) &&
            approx(disc.jacobian(0, 2), 0.0) &&
            approx(disc.jacobian(1, 0), 0.0) &&
            approx(disc.jacobian(1, 1), 1.0) &&
            approx(disc.jacobian(1, 2), 1.0),
            "disc-center Jacobian matches analytical expression");
    }


    std::printf(
        "\n=== (D) heading rotates an offset disc correctly ===\n");

    {
        const EgoState state(
            3.0,
            2.0,
            M_PI / 2.0,
            1.0);

        const auto disc =
            linearize_disc_center(
                state,
                1.0);

        check(
            approx(disc.center.x(), 3.0, 1e-9) &&
            approx(disc.center.y(), 3.0, 1e-9),
            "90-degree heading rotates longitudinal disc offset");
    }


    std::printf(
        "\n=== (E) CollisionConstraint uses disc center, not vehicle center ===\n");

    {
        /*
         * Fixed half-space:
         *
         *     disc_x <= 4.
         *
         * Vehicle center x = 3 with ell = 1 means
         * disc_x = 4, exactly on the boundary.
         */

        CollisionConstraint constraint(
            0,
            /* obstacle_id = */ 0,
            /* scenario_id = */ 0,
            Eigen::Vector2d(-1.0, 0.0),
            -4.0);

        constraint.disc_index = 1;
        constraint.disc_offset = 1.0;

        const EgoState state(
            3.0,
            0.0,
            0.0,
            1.0);

        const Eigen::Vector2d center =
            compute_collision_disc_center(
                state,
                constraint);

        check(
            approx(center.x(), 4.0),
            "collision geometry evaluates the offset disc center");

        check(
            approx(
                constraint.evaluate(center),
                0.0),
            "offset disc lies exactly on collision half-space boundary");
    }


    std::printf(
        "\n=== (F) SQP collision row is anchored consistently ===\n");

    {
        /*
         * Constraint:
         *
         *     disc_x <= 4
         *
         * represented as
         *
         *     (-1,0)^T disc >= -4.
         */

        CollisionConstraint constraint(
            0,
            0,
            0,
            Eigen::Vector2d(-1.0, 0.0),
            -4.0);

        constraint.disc_offset = 1.0;
        constraint.disc_index = 1;

        const EgoState state(
            3.0,
            0.0,
            0.0,
            1.0);

        const DiscConstraintRow row =
            linearize_constraint_at_state(
                constraint,
                state);

        check(
            approx(row.value, 0.0),
            "SQP row value equals exact signed clearance at anchor");

        /*
         * At theta = 0:
         *
         * J =
         *
         * [1 0 0]
         * [0 1 1]
         *
         * and a = (-1,0), hence
         *
         * a^T J = (-1,0,0).
         */

        check(
            approx(row.gradient(0), -1.0) &&
            approx(row.gradient(1),  0.0) &&
            approx(row.gradient(2),  0.0),
            "SQP row gradient matches a^T J");
    }


    std::printf(
        "\n=== (G) fixed half-space affine linearization is consistent ===\n");

    {
        /*
         * Reference vehicle:
         *
         *   center = (3,0)
         *   theta  = 0
         *   ell    = 1
         *
         * Therefore reference disc center is (4,0).
         *
         * Obstacle = (5,0), R = 1,
         * so the reference disc lies exactly on x = 4.
         */

        const EgoState state(
            3.0,
            0.0,
            0.0,
            1.0);

        const double ell = 1.0;

        const Eigen::Vector2d disc_center =
            linearize_disc_center(
                state,
                ell).center;

        const auto halfspace =
            make_collision_halfspace(
                Eigen::Vector2d(5.0, 0.0),
                disc_center,
                1.0);

        const auto affine =
            linearize_disc_halfspace(
                halfspace,
                state.x,
                state.y,
                state.theta,
                ell);

        check(
            approx(affine.coefficients(0), 1.0) &&
            approx(affine.coefficients(1), 0.0) &&
            approx(affine.coefficients(2), 0.0),
            "affine disc row has expected coefficients");

        /*
         * At this reference:
         *
         *     x <= 3
         *
         * for the VEHICLE CENTER because the front disc is one
         * meter ahead.
         */

        check(
            approx(affine.upper_bound, 3.0),
            "affine row accounts for longitudinal disc offset");
    }


    std::printf(
        "\n=== (H) coincident geometry uses deterministic fallback normal ===\n");

    {
        const Eigen::Vector2d point(5.0, 2.0);

        const auto halfspace =
            make_collision_halfspace(
                point,
                point,
                1.0,
                std::optional<Eigen::Vector2d>(
                    Eigen::Vector2d::UnitY()));

        check(
            halfspace.used_fallback_normal,
            "coincident obstacle/reference geometry uses fallback");

        check(
            approx(halfspace.normal.x(), 0.0) &&
            approx(halfspace.normal.y(), 1.0),
            "supplied fallback normal orientation is preserved");
    }


    std::printf(
        "\n=== (I) argument validation ===\n");

    {
        check(
            throws_invalid([] {
                make_collision_halfspace(
                    Eigen::Vector2d(1.0, 0.0),
                    Eigen::Vector2d(0.0, 0.0),
                    1.0,
                    std::nullopt,
                    0.0);
            }),
            "nonpositive direction_epsilon throws");

        check(
            throws_invalid([] {
                compute_ego_disc_positions(
                    EgoState(0.0, 0.0, 0.0, 0.0),
                    0,
                    4.0);
            }),
            "num_discs <= 0 throws");

        check(
            throws_invalid([] {
                compute_ego_disc_positions(
                    EgoState(0.0, 0.0, 0.0, 0.0),
                    1,
                    -1.0);
            }),
            "negative vehicle length throws");
    }


    std::printf(
        "\n%s (%d checks failed)\n",
        fails == 0
            ? "ALL COLLISION HALF-SPACE TESTS PASSED"
            : "SOME COLLISION HALF-SPACE TESTS FAILED",
        fails);

    return fails == 0 ? 0 : 1;
}