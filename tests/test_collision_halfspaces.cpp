/**
 * @file test_collision_halfspaces.cpp
 * @brief Mathematical validation of fixed collision half-spaces.
 */

#include "collision_constraints.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace scenario_mpc;

namespace {

int g_failures = 0;

void expect(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        ++g_failures;
    } else {
        std::cout << "PASS: " << msg << "\n";
    }
}

void test_clearance_identity() {
    const Eigen::Vector2d obs(5.0, 1.0);
    const Eigen::Vector2d disc(1.0, 1.0);
    const double R = 1.5;

    auto hs = make_collision_halfspace(obs, disc, R);

    const double lhs_at_reference =
        hs.normal.dot(hs.reference_disc_center);
    const double expected_clearance =
        hs.reference_distance - hs.safety_radius;
    const double signed_halfspace_clearance =
        hs.upper_bound - lhs_at_reference;

    expect(
        std::abs(signed_halfspace_clearance - expected_clearance) < 1e-8,
        "halfspace clearance identity at reference"
    );
    expect(
        std::abs(hs.normal.norm() - 1.0) < 1e-12,
        "normal is unit length"
    );
    expect(
        std::abs(hs.normal.dot(obs - disc) - hs.reference_distance) < 1e-8,
        "normal aligns with obstacle-minus-disc"
    );
}

void test_halfspace_implies_circular_separation() {
    const Eigen::Vector2d obs(4.0, 0.0);
    const Eigen::Vector2d disc_ref(0.0, 0.0);
    const double R = 1.0;
    auto hs = make_collision_halfspace(obs, disc_ref, R);

    // A point on the halfspace boundary in the safe direction.
    const Eigen::Vector2d p_boundary =
        obs - hs.normal * R;
    expect(
        hs.normal.dot(p_boundary) <= hs.upper_bound + 1e-12,
        "boundary point satisfies halfspace"
    );
    expect(
        (obs - p_boundary).norm() + 1e-12 >= R,
        "boundary point has Euclidean clearance >= R"
    );

    // An interior-safe point farther away.
    const Eigen::Vector2d p_safe =
        obs - hs.normal * (R + 2.0);
    expect(
        hs.normal.dot(p_safe) <= hs.upper_bound + 1e-12,
        "safe point satisfies halfspace"
    );
    expect(
        (obs - p_safe).norm() >= R,
        "safe point has Euclidean clearance >= R"
    );
}

void test_fallback_normal() {
    const Eigen::Vector2d obs(1.0, 1.0);
    const Eigen::Vector2d disc(1.0, 1.0);  // coincident
    const Eigen::Vector2d fallback(0.0, 1.0);
    auto hs = make_collision_halfspace(obs, disc, 0.5, fallback);
    expect(hs.used_fallback_normal, "uses fallback when coincident");
    expect(
        std::abs(hs.normal.y() - 1.0) < 1e-12 &&
            std::abs(hs.normal.x()) < 1e-12,
        "fallback normal is UnitY"
    );
}

void test_legacy_conversion_sign() {
    const Eigen::Vector2d obs(3.0, 0.0);
    const Eigen::Vector2d disc(0.0, 0.0);
    const double R = 1.0;
    auto hs = make_collision_halfspace(obs, disc, R);
    auto c = halfspace_to_collision_constraint(hs);

    // At the reference disc, clearance should be positive (dist - R = 2).
    const double value = c.evaluate(disc);
    expect(std::abs(value - (3.0 - R)) < 1e-8,
           "legacy a^T p - b equals clearance at reference");
}

void test_disc_offset_matches_positions() {
    EgoState state(0.0, 0.0, M_PI / 2.0, 1.0);
    const int num_discs = 3;
    const double length = 4.0;
    auto discs = compute_ego_disc_positions(state, num_discs, length);
    for (int d = 0; d < num_discs; ++d) {
        double ell = get_disc_longitudinal_offset(d, num_discs, length);
        Eigen::Vector2d expected =
            state.position()
            + ell * Eigen::Vector2d(std::cos(state.theta), std::sin(state.theta));
        expect((discs[d] - expected).norm() < 1e-12,
               "disc placement matches longitudinal offset");
    }
}

void test_affine_disc_linearization_at_reference() {
    LinearizedCollisionHalfspace hs;
    hs.normal = Eigen::Vector2d(1.0, 0.0);
    hs.upper_bound = 2.0;
    hs.reference_disc_center = Eigen::Vector2d(1.0, 0.5);

    const double px = 0.0, py = 0.5, theta = 0.0, ell = 1.0;
    auto affine = linearize_disc_halfspace(hs, px, py, theta, ell);

    // At the reference pose, n^T J x_bar should recover the absolute form.
    Eigen::Vector3d xbar(px, py, theta);
    const double lhs = affine.coefficients.dot(xbar);
    expect(lhs <= affine.upper_bound + 1e-10,
           "reference pose satisfies affine disc constraint");

    // Moving toward the obstacle (+x) increases n^T c.
    Eigen::Vector3d x_toward = xbar + Eigen::Vector3d(0.5, 0.0, 0.0);
    expect(affine.coefficients.dot(x_toward) > lhs,
           "moving toward obstacle increases linearized lhs");
}

}  // namespace

int main() {
    try {
        test_clearance_identity();
        test_halfspace_implies_circular_separation();
        test_fallback_normal();
        test_legacy_conversion_sign();
        test_disc_offset_matches_positions();
        test_affine_disc_linearization_at_reference();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    if (g_failures > 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "All collision halfspace tests passed.\n";
    return 0;
}
