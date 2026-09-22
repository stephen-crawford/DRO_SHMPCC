#include "dro.hpp"
#include "dynamics.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::cout
        << (condition ? "PASS: " : "FAIL: ")
        << message
        << '\n';

    if (!condition) {
        ++failures;
    }
}

bool close(
    double a,
    double b,
    double tolerance = 1e-8)
{
    return std::abs(a - b) <= tolerance;
}

double map_mass(
    const std::map<std::string, double>& probabilities)
{
    double mass = 0.0;

    for (const auto& [_, probability] :
         probabilities) {
        mass += probability;
    }

    return mass;
}

double expected_risk(
    const std::map<std::string, double>& probabilities,
    const std::map<std::string, double>& risk)
{
    double value = 0.0;

    for (const auto& [mode, probability] :
         probabilities) {
        value +=
            probability *
            risk.at(mode);
    }

    return value;
}

/*
 * Production DRO API wrapper.
 *
 * The synthetic geometry uses one ego disc. The wrapper makes the newer
 * compute_worst_case_weights() signature explicit while keeping the individual
 * tests focused on the scientific property being checked.
 */
DROResult compute_test_dro(
    DRO& dro,
    const std::map<std::string, double>& nominal,
    const std::map<std::string, int>& counts,
    const ObstacleState& obstacle,
    const std::map<std::string, ModeModel>& modes,
    const std::vector<EgoState>& ego_reference,
    int horizon,
    double ego_radius,
    double obstacle_radius,
    double safety_margin)
{
    return dro.compute_worst_case_weights(
        nominal,
        counts,
        obstacle,
        modes,
        ego_reference,
        horizon,
        ego_radius,
        obstacle_radius,
        safety_margin,
        /*risk_horizon=*/-1,
        /*num_discs=*/1,
        /*ego_length=*/0.0,
        /*transition=*/nullptr);
}

/*
 * Build two intentionally simple deterministic behavior models.
 *
 * Obstacle state:
 *
 *     [x, y, vx, vy]
 *
 * "stay":
 *     leaves the obstacle at its current lateral location.
 *
 * "cross":
 *     translates the obstacle by -1 m in y at every prediction step.
 *
 * Starting from y=1:
 *
 *     stay  -> y = 1, 1, 1, ...
 *     cross -> y = 0,-1,-2, ...
 *
 * There is no process noise. This lets the test isolate:
 *
 *     predicted interaction geometry
 *          -> risk score
 *          -> worst-case probability redistribution.
 */
std::map<std::string, ModeModel>
synthetic_modes()
{
    const Eigen::Matrix4d A =
        Eigen::Matrix4d::Identity();

    const Eigen::MatrixXd G =
        Eigen::MatrixXd::Zero(
            4,
            2);

    const Eigen::Vector4d stay_b =
        Eigen::Vector4d::Zero();

    Eigen::Vector4d cross_b =
        Eigen::Vector4d::Zero();

    cross_b(1) = -1.0;

    std::map<std::string, ModeModel>
        modes;

    modes.emplace(
        "stay",
        ModeModel(
            "stay",
            A,
            stay_b,
            G,
            "synthetic_stay"));

    modes.emplace(
        "cross",
        ModeModel(
            "cross",
            A,
            cross_b,
            G,
            "synthetic_cross"));

    return modes;
}

std::vector<EgoState>
stationary_reference(
    double x,
    double y,
    int horizon)
{
    std::vector<EgoState> reference;

    reference.reserve(
        horizon + 1);

    for (int k = 0;
         k <= horizon;
         ++k) {

        reference.emplace_back(
            x,
            y,
            0.0,
            0.0);
    }

    return reference;
}

DRO fixed_radius_solver(double rho) {
    DROConfig config;

    config.radius_calibration.use_calibrated_radius = false;
    config.base_radius = rho;
    config.min_radius = 0.0;
    config.max_radius = std::max(1.0, rho);

    config.ground_cost_type =
        DROGroundCostType::ZERO_ONE;

    // Make this test specifically exercise exact primal OT rather than
    // depending on whatever allocator is configured by default.
    config.radius_calibration.use_entropic_allocator = false;
    config.radius_calibration.use_primal_ot = true;

    DRO dro(config);

    return dro;
}

}  // namespace

int main() {
    constexpr int horizon = 4;

    constexpr double ego_radius =
        0.4;

    constexpr double obstacle_radius =
        0.3;

    constexpr double safety_margin =
        0.1;

    const auto modes =
        synthetic_modes();

    /*
     * Obstacle begins at:
     *
     *     (x,y) = (2,1)
     *
     * Combined safety radius:
     *
     *     0.4 + 0.3 + 0.1 = 0.8 m
     */
    const ObstacleState obstacle(
        2.0,
        1.0,
        0.0,
        0.0);

    // ==================================================================
    // Test A
    //
    // CROSS is rare but dangerous.
    //
    // Ego is stationary at (2,0).
    //
    // stay:
    //      obstacle remains one metre away -> outside 0.8 m radius.
    //
    // cross:
    //      obstacle reaches (2,0) on its first prediction step.
    //
    // Expected:
    //
    //      risk(cross) > risk(stay)
    //      q*(cross)   > p(cross)
    // ==================================================================
    {
        const std::map<std::string, double>
            nominal{
                {"stay", 0.99},
                {"cross", 0.01},
            };

        const std::map<std::string, int>
            counts{
                {"stay", 99},
                {"cross", 1},
            };

        const auto ego =
            stationary_reference(
                2.0,
                0.0,
                horizon);

        DRO dro =
            fixed_radius_solver(
                0.25);

        const DROResult result =
            compute_test_dro(
                dro,
                nominal,
                counts,
                obstacle,
                modes,
                ego,
                horizon,
                ego_radius,
                obstacle_radius,
                safety_margin);

        const double stay_risk =
            result.risk_per_mode.at(
                "stay");

        const double cross_risk =
            result.risk_per_mode.at(
                "cross");

        const double q_stay =
            result.worst_case_weights.at(
                "stay");

        const double q_cross =
            result.worst_case_weights.at(
                "cross");

        std::cout
            << "\nRare crossing-mode case\n"
            << "  stay risk      = "
            << stay_risk
            << '\n'
            << "  cross risk     = "
            << cross_risk
            << '\n'
            << "  p(cross)       = "
            << nominal.at("cross")
            << '\n'
            << "  q*(cross)      = "
            << q_cross
            << '\n'
            << "  delta q(cross) = "
            << (
                   q_cross -
                   nominal.at("cross"))
            << '\n'
            << "  rho            = "
            << result.rho_used
            << '\n'
            << "  transport cost = "
            << result.implied_transport_cost
            << '\n';

        check(
            cross_risk >
                stay_risk + 1e-9,
            "live risk scorer identifies crossing mode as more dangerous");

        check(
            q_cross >
                nominal.at("cross") +
                    1e-9,
            "DRO upweights rare crossing mode when its predicted geometry is dangerous");

        check(
            q_stay <
                nominal.at("stay") -
                    1e-9,
            "DRO moves probability away from the safer mode");

        check(
            close(
                map_mass(
                    result.worst_case_weights),
                1.0),
            "live DRO worst-case distribution is normalized");

        const double nominal_expected =
            expected_risk(
                nominal,
                result.risk_per_mode);

        const double robust_expected =
            expected_risk(
                result.worst_case_weights,
                result.risk_per_mode);

        check(
            robust_expected >=
                nominal_expected -
                    1e-9,
            "live DRO adversary does not reduce expected interaction risk");

        check(
            result.implied_transport_cost <=
                result.rho_used +
                    1e-8,
            "live adversarial distribution respects Wasserstein transport budget");
    }

    // ==================================================================
    // Test B
    //
    // Reverse the geometry while keeping the exact same mode names.
    //
    // Ego is stationary at (2,1).
    //
    // stay:
    //      obstacle remains on top of the ego.
    //
    // cross:
    //      obstacle moves immediately away toward y=0,-1,...
    //
    // Also reverse the nominal frequencies so STAY is now the rare mode.
    //
    // This verifies that DRO responds to interaction geometry rather than
    // a hard-coded mode name/index.
    // ==================================================================
    {
        const std::map<std::string, double>
            nominal{
                {"stay", 0.01},
                {"cross", 0.99},
            };

        const std::map<std::string, int>
            counts{
                {"stay", 1},
                {"cross", 99},
            };

        const auto ego =
            stationary_reference(
                2.0,
                1.0,
                horizon);

        DRO dro =
            fixed_radius_solver(
                0.25);

        const DROResult result =
            compute_test_dro(
                dro,
                nominal,
                counts,
                obstacle,
                modes,
                ego,
                horizon,
                ego_radius,
                obstacle_radius,
                safety_margin);

        const double stay_risk =
            result.risk_per_mode.at(
                "stay");

        const double cross_risk =
            result.risk_per_mode.at(
                "cross");

        const double q_stay =
            result.worst_case_weights.at(
                "stay");

        std::cout
            << "\nReversed-geometry case\n"
            << "  stay risk      = "
            << stay_risk
            << '\n'
            << "  cross risk     = "
            << cross_risk
            << '\n'
            << "  p(stay)        = "
            << nominal.at("stay")
            << '\n'
            << "  q*(stay)       = "
            << q_stay
            << '\n'
            << "  delta q(stay)  = "
            << (
                   q_stay -
                   nominal.at("stay"))
            << '\n'
            << "  rho            = "
            << result.rho_used
            << '\n';

        check(
            stay_risk >
                cross_risk + 1e-9,
            "risk scorer reverses mode ordering when interaction geometry reverses");

        check(
            q_stay >
                nominal.at("stay") +
                    1e-9,
            "DRO upweights rare stay mode after stay becomes dangerous");

        const double nominal_expected =
            expected_risk(
                nominal,
                result.risk_per_mode);

        const double robust_expected =
            expected_risk(
                result.worst_case_weights,
                result.risk_per_mode);

        check(
            robust_expected >=
                nominal_expected -
                    1e-9,
            "risk-selective adversary increases expectation after geometry reversal");
    }

    // ==================================================================
    // Test C
    //
    // Same dangerous CROSS geometry as Test A, but rho=0.
    //
    // This validates the complete production compute_worst_case_weights()
    // path rather than only the standalone primal OT implementation.
    // ==================================================================
    {
        const std::map<std::string, double>
            nominal{
                {"stay", 0.99},
                {"cross", 0.01},
            };

        const std::map<std::string, int>
            counts{
                {"stay", 99},
                {"cross", 1},
            };

        const auto ego =
            stationary_reference(
                2.0,
                0.0,
                horizon);

        DRO dro =
            fixed_radius_solver(
                0.0);

        const DROResult result =
            compute_test_dro(
                dro,
                nominal,
                counts,
                obstacle,
                modes,
                ego,
                horizon,
                ego_radius,
                obstacle_radius,
                safety_margin);

        check(
            close(
                result.worst_case_weights.at(
                    "stay"),
                nominal.at("stay")) &&
                close(
                    result.worst_case_weights.at(
                        "cross"),
                    nominal.at("cross")),
            "rho=0 leaves empirical probabilities unchanged through full live DRO path");

        check(
            std::abs(
                result.rho_used) <
                1e-12,
            "live DRO reports zero radius for zero-radius experiment");
    }

    // ==================================================================
    // Test D
    //
    // Increase the ambiguity radius while holding EVERYTHING else fixed.
    //
    // Because CROSS is uniquely more dangerous in this geometry:
    //
    //      q_cross(rho_large) >= q_cross(rho_small)
    //
    // and:
    //
    //      R*(rho_large) >= R*(rho_small)
    // ==================================================================
    {
        const std::map<std::string, double>
            nominal{
                {"stay", 0.99},
                {"cross", 0.01},
            };

        const std::map<std::string, int>
            counts{
                {"stay", 99},
                {"cross", 1},
            };

        const auto ego =
            stationary_reference(
                2.0,
                0.0,
                horizon);

        DRO small_solver =
            fixed_radius_solver(
                0.05);

        DRO large_solver =
            fixed_radius_solver(
                0.25);

        const DROResult small =
            compute_test_dro(
                small_solver,
                nominal,
                counts,
                obstacle,
                modes,
                ego,
                horizon,
                ego_radius,
                obstacle_radius,
                safety_margin);

        const DROResult large =
            compute_test_dro(
                large_solver,
                nominal,
                counts,
                obstacle,
                modes,
                ego,
                horizon,
                ego_radius,
                obstacle_radius,
                safety_margin);

        const double small_risk =
            expected_risk(
                small.worst_case_weights,
                small.risk_per_mode);

        const double large_risk =
            expected_risk(
                large.worst_case_weights,
                large.risk_per_mode);

        std::cout
            << "\nRadius-response case\n"
            << "  q_cross(rho=0.05) = "
            << small.worst_case_weights.at(
                   "cross")
            << '\n'
            << "  q_cross(rho=0.25) = "
            << large.worst_case_weights.at(
                   "cross")
            << '\n'
            << "  risk(rho=0.05)    = "
            << small_risk
            << '\n'
            << "  risk(rho=0.25)    = "
            << large_risk
            << '\n';

        check(
            large.worst_case_weights.at(
                "cross") >=
                small.worst_case_weights.at(
                    "cross") -
                    1e-9,
            "larger live DRO radius does not reduce probability on unique dangerous mode");

        check(
            large_risk >=
                small_risk -
                    1e-9,
            "larger live DRO radius does not reduce worst-case interaction risk");
    }

    std::cout
        << '\n'
        << (
               failures == 0
                   ? "ALL DRO RISK-RESPONSE TESTS PASSED"
                   : "DRO RISK-RESPONSE TESTS FAILED")
        << '\n';

    return failures == 0 ? 0 : 1;
}