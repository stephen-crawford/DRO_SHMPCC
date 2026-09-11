// test_pruning_and_linearization: verifies
//   (A) free-space polygon reduction:
//       - redundant sampled half-spaces are removed,
//       - active polygon facets are retained with source metadata,
//       - reduced and full constraint sets are equivalent inside the certified
//         reachable domain.
//   (B) fixed collision half-space construction
//       (compute_linearized_constraints / make_collision_halfspace).
//   (C) heading-Jacobian constraint linearization
//       (linearize_disc_halfspace).
//   (D) SQP-row anchoring
//       (linearize_constraint_at_state).
//   (E) reachable-domain construction from vehicle limits and anchor shift.
//
// The old prune_dominated_scenarios() tests were intentionally removed.
// Safe-Horizon now preserves the complete scenario draw, constructs all sampled
// collision half-spaces, and reduces those half-spaces to the facets of a
// free-space polygon for each (prediction step, ego disc).

#include "collision_constraints.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace dro_mpc;

namespace {

int fails = 0;

void check(bool ok, const char* msg)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) {
        ++fails;
    }
}

bool approx(double a, double b, double tolerance = 1e-6)
{
    return std::abs(a - b) <= tolerance;
}

Scenario make_scenario(
    int scenario_id,
    int obstacle_id,
    const std::vector<Eigen::Vector2d>& positions)
{
    ObstacleTrajectory trajectory;
    trajectory.obstacle_id = obstacle_id;
    trajectory.mode_id = "m";

    for (size_t k = 0; k < positions.size(); ++k) {
        trajectory.steps.emplace_back(
            static_cast<int>(k),
            positions[k],
            Eigen::Matrix2d::Zero());
    }

    return Scenario(scenario_id, {{obstacle_id, trajectory}});
}

CollisionConstraint make_constraint(
    int k,
    int scenario_id,
    const Eigen::Vector2d& a,
    double b,
    int obstacle_id = 0,
    int disc_index = 0,
    double disc_offset = 0.0)
{
    CollisionConstraint constraint(k, obstacle_id, scenario_id, a, b);
    constraint.disc_index = disc_index;
    constraint.disc_offset = disc_offset;
    return constraint;
}

bool satisfies(
    const std::vector<CollisionConstraint>& constraints,
    const Eigen::Vector2d& point,
    double tolerance = 1e-9)
{
    for (const auto& constraint : constraints) {
        if (constraint.evaluate(point) < -tolerance) {
            return false;
        }
    }
    return true;
}

template <typename F>
bool throws_runtime(F&& f)
{
    try {
        f();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

}  // namespace

int main()
{
    std::printf("=== (A) free-space polygon reduction ===\n");

    const std::vector<EgoState> dynamic_reference = {
        EgoState(0.0, 0.0, 0.0, 0.0),
        EgoState(2.0, 2.0, 0.0, 0.0)
    };
    const std::vector<EgoState> anchor_reference = dynamic_reference;

    constexpr int num_discs = 1;
    constexpr double vehicle_length = 0.0;
    constexpr double max_abs_velocity = 1.0;
    constexpr double dt = 1.0;

    {
        const std::vector<CollisionConstraint> all = {
            make_constraint(1, 10, Eigen::Vector2d(1.0, 0.0), 0.0),
            make_constraint(1, 11, Eigen::Vector2d(1.0, 0.0), 1.0)
        };

        const auto reduced = reduce_to_free_space_polytopes(
            all,
            anchor_reference,
            dynamic_reference,
            num_discs,
            vehicle_length,
            max_abs_velocity,
            dt,
            20);

        check(reduced.size() == 1,
              "redundant parallel half-space is removed");

        if (reduced.size() == 1) {
            check(approx(reduced.front().a.x(), 1.0) &&
                  approx(reduced.front().a.y(), 0.0) &&
                  approx(reduced.front().b, 1.0),
                  "more restrictive facet x >= 1 is retained");
            check(reduced.front().scenario_id == 11,
                  "retained facet preserves its source scenario ID");
        }
    }

    {
        const std::vector<CollisionConstraint> all = {
            make_constraint(1, 20, Eigen::Vector2d(1.0, 0.0), 0.0),
            make_constraint(1, 21, Eigen::Vector2d(1.0, 0.0), 1.0),
            make_constraint(1, 22, Eigen::Vector2d(0.0, 1.0), 1.0)
        };

        const auto reduced = reduce_to_free_space_polytopes(
            all,
            anchor_reference,
            dynamic_reference,
            num_discs,
            vehicle_length,
            max_abs_velocity,
            dt,
            20);

        check(reduced.size() == 2,
              "independent active polygon facets are retained");

        bool found_x = false;
        bool found_y = false;
        for (const auto& constraint : reduced) {
            if (approx(constraint.a.x(), 1.0) &&
                approx(constraint.a.y(), 0.0) &&
                approx(constraint.b, 1.0)) {
                found_x = true;
            }
            if (approx(constraint.a.x(), 0.0) &&
                approx(constraint.a.y(), 1.0) &&
                approx(constraint.b, 1.0)) {
                found_y = true;
            }
        }

        check(found_x, "x >= 1 facet retained");
        check(found_y, "y >= 1 facet retained");

        bool equivalent = true;
        for (double x = 0.0; x <= 4.0 && equivalent; x += 0.05) {
            for (double y = 0.0; y <= 4.0; y += 0.05) {
                const Eigen::Vector2d p(x, y);
                if (satisfies(all, p) != satisfies(reduced, p)) {
                    std::printf(
                        "    mismatch at (%.3f, %.3f): full=%d reduced=%d\n",
                        x,
                        y,
                        static_cast<int>(satisfies(all, p)),
                        static_cast<int>(satisfies(reduced, p)));
                    equivalent = false;
                    break;
                }
            }
        }

        check(equivalent,
              "full and reduced half-spaces are equivalent inside reachable domain");
    }

    {
        const std::vector<CollisionConstraint> all = {
            make_constraint(1, 30, Eigen::Vector2d(1.0, 0.0), 1.0),
            make_constraint(1, 31, Eigen::Vector2d(0.0, 1.0), 1.0)
        };

        check(throws_runtime([&] {
                  (void)reduce_to_free_space_polytopes(
                      all,
                      anchor_reference,
                      dynamic_reference,
                      num_discs,
                      vehicle_length,
                      max_abs_velocity,
                      dt,
                      1);
              }),
              "facet limit throws instead of truncating the polygon");
    }

    std::printf("\n=== (B) collision half-space construction ===\n");

    {
        const Scenario onaxis = make_scenario(
            10, 0, {Eigen::Vector2d(5.0, 0.0)});
        const std::vector<EgoState> ego0 = {
            EgoState(0.0, 0.0, 0.0, 1.0)
        };

        const auto constraints = compute_linearized_constraints(
            ego0, {onaxis}, 0.5, 0.3, 0.2, 1, 1.5);

        check(constraints.size() == 1,
              "one half-space per (scenario, step, disc)");

        if (!constraints.empty()) {
            const auto& c = constraints.front();
            check(approx(c.a.norm(), 1.0),
                  "constraint normal is a unit vector");
            check(c.a.x() < -0.99 && approx(c.a.y(), 0.0),
                  "stored normal points obstacle -> ego (-x)");
            check(c.evaluate(Eigen::Vector2d(0.0, 0.0)) > 0.0,
                  "ego reference satisfies the half-space");
            check(approx(c.evaluate(Eigen::Vector2d(4.0, 0.0)), 0.0, 1e-6),
                  "boundary is exactly one safety radius from obstacle");
            check(c.evaluate(Eigen::Vector2d(5.0, 0.0)) < 0.0,
                  "obstacle centre violates the half-space");
        }
    }

    {
        const auto halfspace = make_collision_halfspace(
            Eigen::Vector2d(1.0, 1.0),
            Eigen::Vector2d(1.0, 1.0),
            1.0);

        check(halfspace.normal.allFinite() &&
              approx(halfspace.normal.norm(), 1.0) &&
              halfspace.used_fallback_normal,
              "coincident obstacle/disc uses a finite unit fallback normal");
    }

    std::printf("\n=== (C) heading-Jacobian constraint linearization ===\n");

    {
        LinearizedCollisionHalfspace halfspace;
        halfspace.normal = Eigen::Vector2d(0.6, 0.8);
        halfspace.upper_bound = 2.0;

        const double theta = 0.5;
        const double ell = 1.3;
        const double px = 2.0;
        const double py = -1.0;

        halfspace.reference_disc_center = linearize_disc_center(
            EgoState(px, py, theta, 0.0), ell).center;

        const auto affine = linearize_disc_halfspace(
            halfspace, px, py, theta, ell);

        const double expected_0 = halfspace.normal.x();
        const double expected_1 = halfspace.normal.y();
        const double expected_2 =
            halfspace.normal.x() * (-ell * std::sin(theta)) +
            halfspace.normal.y() * (ell * std::cos(theta));

        check(approx(affine.coefficients(0), expected_0) &&
              approx(affine.coefficients(1), expected_1),
              "position coefficients equal the fixed normal");
        check(approx(affine.coefficients(2), expected_2),
              "theta coefficient equals normal^T times heading Jacobian");

        auto centered_halfspace = halfspace;
        centered_halfspace.reference_disc_center = Eigen::Vector2d(px, py);
        const auto centered_affine = linearize_disc_halfspace(
            centered_halfspace, px, py, theta, 0.0);

        check(approx(centered_affine.coefficients(2), 0.0),
              "zero disc offset gives zero theta coefficient");
    }

    std::printf("\n=== (D) QP row anchoring ===\n");

    {
        std::vector<EgoState> reference;
        for (int k = 0; k <= 4; ++k) {
            reference.emplace_back(
                0.15 * static_cast<double>(k), 0.0, 0.0, 1.5);
        }

        const Scenario scenario = make_scenario(
            20,
            0,
            {
                Eigen::Vector2d(1.20, 1.30),
                Eigen::Vector2d(1.22, 1.24),
                Eigen::Vector2d(1.24, 1.18),
                Eigen::Vector2d(1.26, 1.12),
                Eigen::Vector2d(1.28, 1.06)
            });

        for (int discs : {1, 3}) {
            constexpr double length = 4.0;
            const auto constraints = compute_linearized_constraints(
                reference, {scenario}, 0.5, 0.5, 0.2, discs, length);

            std::vector<EgoState> iterate;
            for (int k = 0; k <= 4; ++k) {
                iterate.emplace_back(
                    0.15 * static_cast<double>(k) + 0.15,
                    0.12,
                    0.05,
                    1.5);
            }

            double worst_anchor_error = 0.0;
            double worst_gradient_error = 0.0;
            double worst_reference_error = 0.0;

            for (const auto& constraint : constraints) {
                const EgoState& x = iterate[constraint.k];
                const auto row = linearize_constraint_at_state(constraint, x);

                const Eigen::Vector2d heading(
                    std::cos(x.theta), std::sin(x.theta));
                const double truth = constraint.evaluate(
                    x.position() + constraint.disc_offset * heading);

                worst_anchor_error = std::max(
                    worst_anchor_error, std::abs(row.value - truth));

                constexpr double h = 1e-6;
                for (int dim = 0; dim < 3; ++dim) {
                    EgoState high = x;
                    EgoState low = x;

                    if (dim == 0) {
                        high.x += h;
                        low.x -= h;
                    } else if (dim == 1) {
                        high.y += h;
                        low.y -= h;
                    } else {
                        high.theta += h;
                        low.theta -= h;
                    }

                    const auto clearance = [&](const EgoState& state) {
                        return constraint.evaluate(
                            state.position() +
                            constraint.disc_offset * Eigen::Vector2d(
                                std::cos(state.theta),
                                std::sin(state.theta)));
                    };

                    const double finite_difference =
                        (clearance(high) - clearance(low)) / (2.0 * h);

                    worst_gradient_error = std::max(
                        worst_gradient_error,
                        std::abs(row.gradient(dim) - finite_difference));
                }

                const auto reference_row = linearize_constraint_at_state(
                    constraint, reference[constraint.k]);

                worst_reference_error = std::max(
                    worst_reference_error,
                    std::abs(
                        reference_row.value -
                        (constraint.a.dot(constraint.linearization_point) -
                         constraint.b)));
            }

            char msg[192];

            std::snprintf(
                msg,
                sizeof msg,
                "num_discs=%d: row value equals exact clearance at SQP iterate "
                "(max err %.2e)",
                discs,
                worst_anchor_error);
            check(worst_anchor_error < 1e-12, msg);

            std::snprintf(
                msg,
                sizeof msg,
                "num_discs=%d: gradient matches finite differences "
                "(max err %.2e)",
                discs,
                worst_gradient_error);
            check(worst_gradient_error < 1e-5, msg);

            std::snprintf(
                msg,
                sizeof msg,
                "num_discs=%d: construction reference reproduces linearization point "
                "(max err %.2e)",
                discs,
                worst_reference_error);
            check(worst_reference_error < 1e-12, msg);
        }
    }

    std::printf("\n=== (E) dynamic reachable-domain expansion ===\n");

    {
        const std::vector<EgoState> centered_reference = {
            EgoState(0.0, 0.0, 0.0, 0.0),
            EgoState(2.0, 0.0, 0.0, 0.0)
        };

        const std::vector<CollisionConstraint> all = {
            make_constraint(1, 40, Eigen::Vector2d(1.0, 0.0), 1.0)
        };

        const auto tight = reduce_to_free_space_polytopes(
            all,
            centered_reference,
            centered_reference,
            1,
            0.0,
            0.0,
            1.0,
            20);

        const auto wide = reduce_to_free_space_polytopes(
            all,
            centered_reference,
            centered_reference,
            1,
            0.0,
            1.0,
            1.0,
            20);

        check(tight.empty(),
              "zero reachable motion makes distant half-space redundant");
        check(wide.size() == 1,
              "dynamic velocity bound expands domain and exposes active facet");
    }

    {
        const std::vector<EgoState> dynamic = {
            EgoState(0.0, 0.0, 0.0, 0.0),
            EgoState(0.0, 0.0, 0.0, 0.0)
        };
        const std::vector<EgoState> anchor = {
            EgoState(0.0, 0.0, 0.0, 0.0),
            EgoState(2.0, 0.0, 0.0, 0.0)
        };
        const std::vector<CollisionConstraint> all = {
            make_constraint(1, 41, Eigen::Vector2d(1.0, 0.0), 1.0)
        };

        const auto reduced = reduce_to_free_space_polytopes(
            all,
            anchor,
            dynamic,
            1,
            0.0,
            0.0,
            1.0,
            20);

        check(reduced.size() == 1,
              "anchor displacement contributes to reachable-domain radius");
    }

    std::printf(
        "\n%s (%d checks failed)\n",
        fails == 0
            ? "ALL FREE-SPACE/CONSTRAINT TESTS PASSED"
            : "SOME FREE-SPACE/CONSTRAINT TESTS FAILED",
        fails);

    return fails == 0 ? 0 : 1;
}
