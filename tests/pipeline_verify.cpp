// pipeline_verify:
// Verify the WDRO Safe-Horizon scenario-MPC pipeline end-to-end.
//
// Stages checked:
//   1. Reference path, obstacle state, nominal mode belief, and risk scoring.
//   2. WDRO worst-case reweighting.
//   3. Scenario sampling from q*.
//   4. Safe-Horizon collision geometry:
//        dynamic reference
//          -> collision-feasible anchor reference
//          -> all sampled half-spaces
//          -> free-space polygon facet reduction.
//   5. Full controller solve.
//   6. First control availability.
//
// The old whole-scenario prune_dominated_scenarios() stage is intentionally
// absent. The updated implementation preserves the complete sampled scenario
// set and reduces redundant COLLISION HALF-SPACES instead.

#include "mpc_controller.hpp"
#include "dro.hpp"
#include "mode_weights.hpp"
#include "scenario_sampler.hpp"
#include "collision_constraints.hpp"
#include "dynamics.hpp"
#include "reference_path.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <string>
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

bool satisfies_group(
    const std::vector<CollisionConstraint>& constraints,
    int k,
    int disc_index,
    const Eigen::Vector2d& point,
    double tolerance = 1e-9)
{
    for (const auto& constraint : constraints) {
        if (constraint.k != k ||
            constraint.disc_index != disc_index) {
            continue;
        }

        if (constraint.evaluate(point) < -tolerance) {
            return false;
        }
    }

    return true;
}

}  // namespace


int main()
{
    constexpr double dt = 0.1;
    constexpr int horizon = 15;

    auto mode_models =
        create_obstacle_mode_models(dt);

    std::vector<std::string> modes;
    for (const auto& [mode_name, model] : mode_models) {
        (void)model;
        modes.push_back(mode_name);
    }

    const int mode_count =
        static_cast<int>(modes.size());

    // Ego drives +x toward an obstacle ahead drifting slowly +x.
    const ObstacleState obstacle(
        5.0,
        0.0,
        0.5,
        0.0);

    std::vector<EgoState> ego_reference;
    for (int k = 0; k <= horizon; ++k) {
        ego_reference.emplace_back(
            static_cast<double>(k) * 0.34,
            0.0,
            0.0,
            1.5);
    }

    // Non-uniform mode history: mostly constant_velocity, with some
    // decelerating and turn_left observations.
    ModeHistory history(
        0,
        mode_models);

    for (int t = 0; t < 30; ++t) {
        const std::string mode =
            (t % 2 == 0)
                ? "constant_velocity"
                : ((t % 5 == 0)
                    ? "decelerating"
                    : "turn_left");

        history.record_observation(
            t,
            history.obstacle_id,
            mode);
    }


    // -------------------------------------------------------------------------
    // STEP 1
    // -------------------------------------------------------------------------

    std::printf(
        "=== STEP 1: reference path, obstacle, nominal belief, "
        "and Bonferroni-VaR risk ===\n");

    const ReferencePath path =
        ReferencePath::create_s_curve(
            25.0,
            3.0,
            200);

    const double s0 =
        path.find_closest_point(
            Eigen::Vector2d(0.0, 0.0));

    check(
        std::isfinite(s0),
        "reference-path spline position computed for ego");

    check(
        obstacle.position().x() == 5.0,
        "obstacle position determined");

    const auto nominal =
        compute_mode_weights(history);

    double nominal_sum = 0.0;
    bool all_positive = true;

    for (const auto& [mode, probability] : nominal) {
        (void)mode;
        nominal_sum += probability;

        if (probability <= 0.0) {
            all_positive = false;
        }
    }

    check(
        static_cast<int>(nominal.size()) == mode_count &&
        all_positive,
        "nominal belief gives every mode positive Dirichlet mass");

    check(
        std::abs(nominal_sum - 1.0) < 1e-9,
        "nominal belief normalized to one");

    auto risk_of =
        [&](DRORiskMeasure risk_measure) {
            DROConfig config;
            config.radius_calibration.risk_measure =
                risk_measure;

            DRO dro(config);

            return dro.compute_worst_case_weights(
                nominal,
                history.get_mode_counts(),
                obstacle,
                mode_models,
                ego_reference,
                horizon,
                0.5,
                0.35,
                0.2);
        };

    DROResult bonferroni =
        risk_of(
            DRORiskMeasure::SURROGATE_VAR_BONFERRONI);

    DROResult plain_var =
        risk_of(
            DRORiskMeasure::SURROGATE_VAR);

    double risk_min = 1e9;
    double risk_max = -1e9;
    std::string dangerous_mode;

    for (const auto& [mode, risk] :
         bonferroni.risk_per_mode) {

        if (risk > risk_max) {
            risk_max = risk;
            dangerous_mode = mode;
        }

        risk_min =
            std::min(
                risk_min,
                risk);
    }

    check(
        risk_max - risk_min > 1e-6,
        "per-mode risk scores are differentiated");

    double bonferroni_sum = 0.0;
    double var_sum = 0.0;

    for (const auto& [mode, risk] :
         bonferroni.risk_per_mode) {
        (void)mode;
        bonferroni_sum += risk;
    }

    for (const auto& [mode, risk] :
         plain_var.risk_per_mode) {
        (void)mode;
        var_sum += risk;
    }

    check(
        std::abs(bonferroni_sum - var_sum) > 1e-9,
        "Bonferroni VaR differs from plain VaR");

    std::printf(
        "    most-dangerous mode = %s (r=%.4f); "
        "risk range [%.4f, %.4f]\n",
        dangerous_mode.c_str(),
        risk_max,
        risk_min,
        risk_max);


    // -------------------------------------------------------------------------
    // STEP 2
    // -------------------------------------------------------------------------

    std::printf(
        "=== STEP 2: WDRO reweighting shifts probability mass "
        "toward high-risk modes ===\n");

    const double q_danger =
        bonferroni.worst_case_weights.at(
            dangerous_mode);

    const double p_danger =
        nominal.at(
            dangerous_mode);

    double q_sum = 0.0;
    bool q_valid = true;

    for (const auto& [mode, probability] :
         bonferroni.worst_case_weights) {

        (void)mode;
        q_sum += probability;

        if (probability < -1e-9 ||
            probability > 1.0 + 1e-9) {
            q_valid = false;
        }
    }

    check(
        q_valid &&
        std::abs(q_sum - 1.0) < 1e-6,
        "q* is a valid probability distribution");

    check(
        q_danger > p_danger + 1e-6,
        "q* up-weights the most-dangerous mode");

    check(
        bonferroni.rho_used > 0.0,
        "ambiguity radius rho_used is positive");

    std::printf(
        "    q*[%s]=%.4f vs nominal=%.4f (rho=%.4f)\n",
        dangerous_mode.c_str(),
        q_danger,
        p_danger,
        bonferroni.rho_used);


    // -------------------------------------------------------------------------
    // STEP 3
    // -------------------------------------------------------------------------

    std::printf(
        "=== STEP 3: sample scenarios from q* and verify empirical "
        "mode frequency ===\n");

    std::mt19937 rng(12345);

    const std::map<int, ObstacleState> obstacles{
        {0, obstacle}
    };

    const std::map<int, ModeHistory> histories{
        {0, history}
    };

    constexpr int sample_count = 400;

    const std::map<
        int,
        std::map<std::string, double>> q_map{
            {0, bonferroni.worst_case_weights}
        };

    const auto scenarios_q =
        sample_scenarios(
            obstacles,
            histories,
            &q_map,
            horizon,
            sample_count,
            {},
            nullptr,
            &rng);

    std::map<std::string, int> q_counts;

    for (const auto& scenario : scenarios_q) {
        const auto it =
            scenario.trajectories.find(0);

        if (it != scenario.trajectories.end()) {
            ++q_counts[it->second.mode_id];
        }
    }

    const double empirical_q_danger =
        q_counts[dangerous_mode] /
        static_cast<double>(
            scenarios_q.size());

    check(
        static_cast<int>(scenarios_q.size()) ==
        sample_count,
        "sampler returned S scenarios");

    check(
        std::abs(
            empirical_q_danger -
            q_danger) < 0.06,
        "empirical dangerous-mode frequency matches q*");

    const std::map<
        int,
        std::map<std::string, double>> p_map{
            {0, nominal}
        };

    const auto scenarios_p =
        sample_scenarios(
            obstacles,
            histories,
            &p_map,
            horizon,
            sample_count,
            {},
            nullptr,
            &rng);

    std::map<std::string, int> p_counts;

    for (const auto& scenario : scenarios_p) {
        const auto it =
            scenario.trajectories.find(0);

        if (it != scenario.trajectories.end()) {
            ++p_counts[it->second.mode_id];
        }
    }

    const double empirical_p_danger =
        p_counts[dangerous_mode] /
        static_cast<double>(
            scenarios_p.size());

    check(
        empirical_q_danger >
        empirical_p_danger + 0.03,
        "q*-sampling over-represents the dangerous mode "
        "relative to nominal sampling");

    std::printf(
        "    dangerous-mode frequency: "
        "q*=%.3f nominal=%.3f target=%.3f\n",
        empirical_q_danger,
        empirical_p_danger,
        q_danger);


    // -------------------------------------------------------------------------
    // STEP 4
    // -------------------------------------------------------------------------

    std::printf(
        "=== STEP 4: Safe-Horizon anchor, sampled half-spaces, "
        "and free-space polygon reduction ===\n");

    constexpr double ego_radius = 0.5;
    constexpr double obstacle_radius = 0.35;
    constexpr double safety_margin = 0.1;
    constexpr int num_discs = 1;
    constexpr double vehicle_length = 1.5;

    const double combined_radius =
        ego_radius +
        obstacle_radius +
        safety_margin;

    /*
     * Preserve the dynamically consistent reference. The collision anchor is
     * a separate copy because prepare_safe_horizon_anchors() is a geometric
     * operation and is not itself a dynamics rollout.
     */
    const auto dynamic_reference =
        ego_reference;

    auto anchor_reference =
        dynamic_reference;

    const bool anchor_ok =
        prepare_safe_horizon_anchors(
            anchor_reference,
            scenarios_q,
            combined_radius,
            num_discs,
            vehicle_length);

    check(
        anchor_ok,
        "Safe-Horizon collision anchor is feasible against sampled circles");

    const auto all_constraints =
        compute_linearized_constraints(
            anchor_reference,
            scenarios_q,
            ego_radius,
            obstacle_radius,
            safety_margin,
            num_discs,
            vehicle_length);

    bool all_constraints_proper = true;

    for (const auto& constraint :
         all_constraints) {

        if (std::abs(
                constraint.a.norm() -
                1.0) > 1e-6 ||
            !std::isfinite(
                constraint.b)) {

            all_constraints_proper =
                false;
            break;
        }
    }

    check(
        !all_constraints.empty(),
        "complete sampled half-space set generated");

    check(
        all_constraints_proper,
        "every sampled constraint is a finite unit-normal half-space");

    /*
     * Use the same type of conservative bound as the controller:
     * an absolute hard speed limit plus the anchor displacement term handled
     * internally by reduce_to_free_space_polytopes().
     *
     * 3 m/s is deliberately conservative relative to this test trajectory.
     */
    constexpr double max_abs_velocity = 3.0;

    const auto reduced_constraints =
        reduce_to_free_space_polytopes(
            all_constraints,
            anchor_reference,
            dynamic_reference,
            num_discs,
            vehicle_length,
            max_abs_velocity,
            dt,
            20);

    check(
        reduced_constraints.size() <=
        all_constraints.size(),
        "free-space reduction never increases the collision-row count");

    check(
        !reduced_constraints.empty(),
        "free-space reduction retains active collision facets");

    std::printf(
        "    scenarios preserved: %zu\n",
        scenarios_q.size());

    std::printf(
        "    collision rows: %zu -> %zu after free-space reduction\n",
        all_constraints.size(),
        reduced_constraints.size());

    /*
     * Verify the key reduction invariant on one representative (k, disc)
     * reachable square:
     *
     *     D_k^d ∩ H_all = D_k^d ∩ H_reduced.
     *
     * This test has one ego disc, so ell = 0.
     */
    constexpr int test_k = 8;
    constexpr int test_disc = 0;

    const auto anchor_discs =
        compute_ego_disc_positions(
            anchor_reference[test_k],
            num_discs,
            vehicle_length);

    const auto dynamic_discs =
        compute_ego_disc_positions(
            dynamic_reference[test_k],
            num_discs,
            vehicle_length);

    const Eigen::Vector2d domain_center =
        anchor_discs[test_disc];

    const double anchor_shift =
        (anchor_discs[test_disc] -
         dynamic_discs[test_disc]).norm();

    const double disc_offset =
        std::abs(
            get_disc_longitudinal_offset(
                test_disc,
                num_discs,
                vehicle_length));

    const double rho =
        2.0 *
        max_abs_velocity *
        static_cast<double>(test_k) *
        dt
        + 2.0 * disc_offset
        + anchor_shift
        + 1e-6;

    bool equivalent_on_domain = true;

    /*
     * Sample the exact square used as the clipping domain.
     * A coarse grid is enough for a pipeline regression check; dedicated
     * polygon unit tests should exercise exact facet geometry more aggressively.
     */
    constexpr int grid_count = 20;

    for (int ix = 0;
         ix <= grid_count &&
         equivalent_on_domain;
         ++ix) {

        const double x =
            domain_center.x() -
            rho +
            2.0 * rho *
                static_cast<double>(ix) /
                static_cast<double>(grid_count);

        for (int iy = 0;
             iy <= grid_count;
             ++iy) {

            const double y =
                domain_center.y() -
                rho +
                2.0 * rho *
                    static_cast<double>(iy) /
                    static_cast<double>(grid_count);

            const Eigen::Vector2d point(
                x,
                y);

            const bool full_safe =
                satisfies_group(
                    all_constraints,
                    test_k,
                    test_disc,
                    point);

            const bool reduced_safe =
                satisfies_group(
                    reduced_constraints,
                    test_k,
                    test_disc,
                    point);

            if (full_safe != reduced_safe) {
                std::printf(
                    "    free-space mismatch at k=%d "
                    "point=(%.3f, %.3f): "
                    "full=%d reduced=%d\n",
                    test_k,
                    x,
                    y,
                    static_cast<int>(
                        full_safe),
                    static_cast<int>(
                        reduced_safe));

                equivalent_on_domain =
                    false;
                break;
            }
        }
    }

    check(
        equivalent_on_domain,
        "full and reduced half-spaces agree inside the certified "
        "reachable domain");


    // -------------------------------------------------------------------------
    // STEP 5 + 6
    // -------------------------------------------------------------------------

    std::printf(
        "=== STEP 5+6: full controller solve and first-control output ===\n");

    RuntimeConfig config;
    config.dro.enabled = true;

    config.mpc.sampling.set_manual_sample_count(
        40);

    config.mpc.ego.num_discs = 1;
    config.mpc.ego.length = 1.5;

    MPCController controller(config);

    controller.set_reference_path(
        path);

    const EgoState ego(
        0.0,
        0.0,
        0.0,
        1.5);

    const Eigen::Vector2d goal(
        25.0,
        0.0);

    const MPCResult result =
        controller.solve(
            ego,
            obstacles,
            goal,
            1.5,
            0.0,
            path.total_length());

    check(
        result.success,
        "controller.solve() succeeded end-to-end with DRO enabled");

    check(
        result.first_input().has_value(),
        "controller produced a first control input");

    check(
        !controller.scenarios().empty(),
        "controller populated its scenario set");


    // -------------------------------------------------------------------------
    // DIAGNOSTIC
    // -------------------------------------------------------------------------

    std::printf(
        "=== DIAGNOSTIC: sampled support under raw-LP vs entropic allocator ===\n");

    auto sampled_mode_support =
        [&](const DROResult& result) -> int {

            const std::map<
                int,
                std::map<std::string, double>> weights{
                    {0, result.worst_case_weights}
                };

            std::mt19937 local_rng(7);

            const auto scenarios =
                sample_scenarios(
                    obstacles,
                    histories,
                    &weights,
                    horizon,
                    sample_count,
                    {},
                    nullptr,
                    &local_rng);

            std::map<std::string, int> counts;

            for (const auto& scenario : scenarios) {
                const auto it =
                    scenario.trajectories.find(0);

                if (it != scenario.trajectories.end()) {
                    ++counts[it->second.mode_id];
                }
            }

            int support = 0;

            for (const auto& [mode, count] :
                 counts) {
                (void)mode;

                if (count > 0) {
                    ++support;
                }
            }

            return support;
        };

    const int raw_support =
        sampled_mode_support(
            bonferroni);

    DROConfig entropic_config;
    entropic_config.radius_calibration.risk_measure =
        DRORiskMeasure::SURROGATE_VAR_BONFERRONI;

    entropic_config.radius_calibration.use_entropic_allocator =
        true;

    entropic_config.radius_calibration.entropic_tau =
        0.05;

    DRO entropic_dro(
        entropic_config);

    const DROResult entropic =
        entropic_dro.compute_worst_case_weights(
            nominal,
            history.get_mode_counts(),
            obstacle,
            mode_models,
            ego_reference,
            horizon,
            0.5,
            0.35,
            0.2);
    const int entropic_support =
        sampled_mode_support(
            entropic);

    std::printf(
        "    distinct modes represented in S sampled scenarios:\n");

    std::printf(
        "      raw-LP q*            : %d / %d modes\n",
        raw_support,
        mode_count);

    std::printf(
        "      entropic q* (tau=.05): %d / %d modes\n",
        entropic_support,
        mode_count);

    /*
     * Keep this as a diagnostic rather than a hard pass/fail assertion:
     * allocator behaviour can legitimately change as the DRO implementation
     * evolves, whereas Steps 1-6 above define the pipeline contract.
     */


    std::printf(
        "\n%s (%d checks failed)\n",
        fails == 0
            ? "ALL PIPELINE STAGES VERIFIED"
            : "SOME PIPELINE STAGES FAILED",
        fails);

    return fails == 0 ? 0 : 1;
}
