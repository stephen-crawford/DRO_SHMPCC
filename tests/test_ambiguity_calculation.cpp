#include "dro.hpp"
#include "mode_count_test_utils.hpp"
#include "dynamics.hpp"
#include "experiment_config_yaml.hpp"
#include "mpc_controller.hpp"
#include "schuurmans_ambiguity.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits> 

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) {
        ++failures;
    }
}

/*
 * Keep the production DRO API details in one place.
 *
 * These tests use a single ego disc, so ego_length is irrelevant and is set
 * to zero. risk_horizon=-1 means use the normal configured/default behavior.
 */
DROResult compute_test_dro(
    DRO& dro,
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, int>& mode_counts,
    const ObstacleState& obstacle,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<EgoState>& ego_reference,
    int horizon,
    double ego_radius,
    double obstacle_radius,
    double safety_margin)
{
    return dro.compute_worst_case_weights(
        nominal_weights,
        mode_counts,
        obstacle,
        mode_models,
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

double expectation(
    const std::vector<double>& probabilities,
    const std::vector<double>& values)
{
    if (probabilities.size() != values.size()) {
        throw std::runtime_error("expectation size mismatch");
    }

    return std::inner_product(
        probabilities.begin(),
        probabilities.end(),
        values.begin(),
        0.0);
}

bool normalized_nonnegative(
    const std::vector<double>& probabilities,
    double tolerance = 1e-8)
{
    const double mass = std::accumulate(
        probabilities.begin(),
        probabilities.end(),
        0.0);

    if (std::abs(mass - 1.0) > tolerance) {
        return false;
    }

    for (const double p : probabilities) {
        if (!std::isfinite(p) || p < -tolerance) {
            return false;
        }
    }

    return true;
}

}  // namespace

int main() {
    DROConfig default_config;

    check(
        default_config.radius_calibration.divergence ==
            AmbiguityDivergence::WASSERSTEIN,
        "Wasserstein is the default ambiguity set");

    check(
        default_config.radius_calibration.use_entropic_allocator == false,
        "entropic allocation is disabled by default");

    const ExperimentConfig yaml_config =
        yaml_config::load_experiment_config(
            "configs/default.yaml",
            true);

    check(
        yaml_config.dro.solver.radius_calibration.divergence ==
            AmbiguityDivergence::WASSERSTEIN,
        "YAML configuration inherits the Wasserstein default");

    check(
        yaml_config.dro.solver.radius_calibration.use_entropic_allocator,
        "YAML configuration inherits entropic allocation");

    check(
        yaml_config.dro.solver.radius_calibration.risk_scoring_model ==
            DRORiskScoringModel::INHERIT_RISK_MEASURE,
        "YAML configuration inherits the default risk-scoring profile");

    // ------------------------------------------------------------------
    // Fixed-rho plumbing.
    // ------------------------------------------------------------------
    ExperimentConfig fixed_rho_config;

    fixed_rho_config.dro.fixed_rho = 0.23;
    fixed_rho_config.dro.apply_fixed_rho();

    check(
        !fixed_rho_config.dro.solver.radius_calibration.use_calibrated_radius &&
            std::abs(
                fixed_rho_config.dro.solver.base_radius - 0.23) <
                1e-12 &&
            fixed_rho_config.dro.solver.min_radius <= 0.23 &&
            fixed_rho_config.dro.solver.max_radius >= 0.23,
        "fixed_rho pins the effective ambiguity radius independently of calibration");

    {
        const auto all_mode_models =
            create_obstacle_mode_models(0.1);

        // Keep the live mode support aligned with the tested distribution.
        const std::map<std::string, ModeModel> mode_models{
            {
                "constant_velocity",
                all_mode_models.at("constant_velocity"),
            },
            {
                "turn_left",
                all_mode_models.at("turn_left"),
            },
        };

        const std::map<std::string, double> nominal_weights{
            {"constant_velocity", 0.5},
            {"turn_left", 0.5},
        };

        // Counts are consistent with 20 observed episodes and the 50/50
        // empirical distribution.
        const std::map<std::string, int> mode_counts{
            {"constant_velocity", 10},
            {"turn_left", 10},
        };

        const std::vector<EgoState> ego_reference{
            EgoState(0.0, 0.0, 0.0, 1.0),
            EgoState(0.1, 0.0, 0.0, 1.0),
            EgoState(0.2, 0.0, 0.0, 1.0),
        };

        DRO fixed_rho_solver(
            fixed_rho_config.dro.solver);


        const DROResult fixed_result =
            compute_test_dro(
                fixed_rho_solver,
                nominal_weights,
                mode_counts,
                ObstacleState(
                    2.0,
                    0.0,
                    -0.1,
                    0.0),
                mode_models,
                ego_reference,
                /*horizon=*/2,
                /*ego_radius=*/0.4,
                /*obstacle_radius=*/0.3,
                /*safety_margin=*/0.1);

        check(
            std::abs(
                fixed_result.rho_used - 0.23) <
                1e-12,
            "the live DRO solve reports the requested fixed rho");

        check(
            fixed_result.radius_observation_count == 20 &&
                fixed_result.radius_mode_count == 2 &&
                std::abs(
                    fixed_result.rho_before_clamp -
                    0.23) <
                    1e-12 &&
                !fixed_result.rho_clamped_to_min &&
                !fixed_result.rho_clamped_to_max,
            "DRO diagnostics expose the evidence and clamp state behind rho");
    }

    // ------------------------------------------------------------------
    // Controller must calibrate from mode episodes, not raw tick count.
    // ------------------------------------------------------------------
    {
        RuntimeConfig controller_config;

        controller_config.mpc.type =
            MPCType::MPC;

        controller_config.mpc.sync_from_type();

        controller_config.mpc.horizon = 2;

        controller_config.mpc.sampling
            .set_manual_sample_count(4);

        controller_config.dro.enabled = true;
        controller_config.random_seed = 117u;

        MPCController controller(
            controller_config);

        const auto all_modes =
            create_obstacle_mode_models(0.1);

        const std::map<std::string, ModeModel> modes{
            {
                "constant_velocity",
                all_modes.at("constant_velocity"),
            },
            {
                "turn_left",
                all_modes.at("turn_left"),
            },
        };

        controller.initialize_obstacle(
            3,
            0,
            modes);

        for (int t = 0; t < 8; ++t) {
            controller.update_mode_observation(
                3,
                0,
                "constant_velocity",
                t);
        }

        (void)controller.solve(
            EgoState(
                0.0,
                0.0,
                0.0,
                1.0),
            {
                {
                    3,
                    ObstacleState(
                        5.0,
                        3.0,
                        0.0,
                        0.0),
                },
            },
            Eigen::Vector2d(
                1.0,
                0.0),
            1.0);

        const auto& last =
            controller.last_dro_results();

        check(
        last.count(3) == 1 &&
            last.at(3).radius_observation_count == 8 &&
            last.at(3).radius_mode_count == 2,
        "controller calibrates rho from realized obstacle-timestep observations");
    }

    // ------------------------------------------------------------------
    // Live calibrated Wasserstein radius must contract with evidence.
    //
    // This test deliberately constructs a fresh DROConfig instead of copying
    // the production/default configuration, so fixed-radius settings or YAML
    // policy cannot silently disable calibration.
    // ------------------------------------------------------------------
    {
        const auto mode_models =
            create_obstacle_mode_models(0.1);

        std::map<std::string, double> nominal_weights;
        std::map<std::string, int> mode_counts;

        for (const auto& [mode_id, _] : mode_models) {
            nominal_weights[mode_id] =
                1.0 / static_cast<double>(mode_models.size());

            mode_counts[mode_id] = 1;
        }

        std::vector<EgoState> ego_reference;

        for (int k = 0; k <= 15; ++k) {
            ego_reference.emplace_back(
                0.34 * k,
                0.0,
                0.0,
                1.5);
        }

        // IMPORTANT: construct from scratch.
        DROConfig calibrated_config;

        calibrated_config.radius_calibration.use_calibrated_radius = true;
        calibrated_config.radius_calibration.divergence =
            AmbiguityDivergence::WASSERSTEIN;

        calibrated_config.radius_calibration.confidence_beta = 0.05;
        calibrated_config.radius_calibration.calibration_scale = 1.0;

        // These are irrelevant to the special Wasserstein branch, but setting
        // them explicitly makes the intended test configuration unambiguous.
        calibrated_config.base_radius = 0.123456;
        calibrated_config.min_radius = 0.0;
        calibrated_config.max_radius = 1e9;

        DRO early_evidence(calibrated_config);
        DRO mature_evidence(calibrated_config);

        // Verify the configuration actually survived construction.
        check(
            early_evidence.config()
                .radius_calibration
                .use_calibrated_radius,
            "early DRO instance has calibrated radius enabled");

        check(
            mature_evidence.config()
                .radius_calibration
                .use_calibrated_radius,
            "mature DRO instance has calibrated radius enabled");

        check(
            early_evidence.config()
                .radius_calibration
                .divergence ==
                AmbiguityDivergence::WASSERSTEIN,
            "early DRO instance uses Wasserstein calibration");

        check(
            mature_evidence.config()
                .radius_calibration
                .divergence ==
                AmbiguityDivergence::WASSERSTEIN,
            "mature DRO instance uses Wasserstein calibration");


        const DROResult early =
            compute_test_dro(
                early_evidence,
                nominal_weights,
                test::uniform_mode_counts(nominal_weights, 1),
                ObstacleState(
                    5.0,
                    0.0,
                    0.5,
                    0.0),
                mode_models,
                ego_reference,
                /*horizon=*/15,
                /*ego_radius=*/0.5,
                /*obstacle_radius=*/0.35,
                /*safety_margin=*/0.2);

        const DROResult mature =
            compute_test_dro(
                mature_evidence,
                nominal_weights,
                test::uniform_mode_counts(nominal_weights, 10000),
                ObstacleState(
                    5.0,
                    0.0,
                    0.5,
                    0.0),
                mode_models,
                ego_reference,
                /*horizon=*/15,
                /*ego_radius=*/0.5,
                /*obstacle_radius=*/0.35,
                /*safety_margin=*/0.2);

        std::cout
            << "\nCalibrated-radius diagnostic\n"
            << "  configured use_calibrated_radius = "
            << calibrated_config.radius_calibration.use_calibrated_radius
            << '\n'
            << "  early config flag = "
            << early_evidence.config()
                .radius_calibration
                .use_calibrated_radius
            << '\n'
            << "  mature config flag = "
            << mature_evidence.config()
                .radius_calibration
                .use_calibrated_radius
            << '\n'
            << "  early observations = "
            << early.radius_observation_count
            << '\n'
            << "  mature observations = "
            << mature.radius_observation_count
            << '\n'
            << "  early D_max = "
            << early.transport_diameter
            << '\n'
            << "  mature D_max = "
            << mature.transport_diameter
            << '\n'
            << "  early rho used/raw = "
            << early.rho_used
            << " / "
            << early.rho_before_clamp
            << "  clamped_max="
            << early.rho_clamped_to_max
            << '\n'
            << "  mature rho used/raw = "
            << mature.rho_used
            << " / "
            << mature.rho_before_clamp
            << "  clamped_max="
            << mature.rho_clamped_to_max
            << '\n';

        check(
            early.radius_observation_count == 1 &&
                mature.radius_observation_count == 10000,
            "live radius diagnostics retain requested evidence counts");

        check(
            std::abs(
                early.transport_diameter -
                mature.transport_diameter) < 1e-12,
            "early and mature radius tests use the same transport geometry");

        check(
            mature.rho_before_clamp <
                early.rho_before_clamp,
            "uncapped calibrated Wasserstein radius contracts with additional evidence");

        check(
            mature.rho_used <
                early.rho_used,
            "effective calibrated Wasserstein radius contracts with additional evidence");
    }

    // ------------------------------------------------------------------
    // Independent seed streams.
    // ------------------------------------------------------------------
    const SeedBundle seeds =
        derive_seeds(42u, 0);

    check(
        seeds.env != seeds.scenario &&
            seeds.env != seeds.predictor &&
            seeds.predictor != seeds.scenario,
        "plant, predictor, and controller seeds are distinct");

    const SeedBundle repeat =
        derive_seeds(42u, 0);

    check(
        seeds.env == repeat.env &&
            seeds.predictor == repeat.predictor &&
            seeds.scenario == repeat.scenario,
        "derived seeds are reproducible from the master seed");

    // ------------------------------------------------------------------
    // Common ambiguity-family tests.
    // ------------------------------------------------------------------
    const std::vector<double> nominal{
        0.55,
        0.30,
        0.15,
    };

    const std::vector<double> risk{
        0.1,
        0.4,
        0.9,
    };

    const std::vector<std::vector<double>>
        ground_cost{
            {
                0.0,
                0.5,
                1.0,
            },
            {
                0.5,
                0.0,
                0.5,
            },
            {
                1.0,
                0.5,
                0.0,
            },
        };

    const std::vector<AmbiguityDivergence>
        families{
            AmbiguityDivergence::WASSERSTEIN,
            AmbiguityDivergence::TOTAL_VARIATION,
            AmbiguityDivergence::KULLBACK_LEIBLER,
            AmbiguityDivergence::JENSEN_SHANNON,
            AmbiguityDivergence::HELLINGER,
        };

    const double nominal_risk =
        expectation(
            nominal,
            risk);

    for (const auto family : families) {
        const std::string family_name =
            schuurmans::divergence_name(
                family);

        const double radius =
            schuurmans::ambiguity_radius(
                family,
                static_cast<int>(
                    nominal.size()),
                200,
                0.05,
                1.0);

        const auto worst_case =
            schuurmans::worst_case_expectation(
                family,
                nominal,
                risk,
                radius,
                &ground_cost);

        check(
            normalized_nonnegative(
                worst_case.p),
            (
                "normalized nonnegative worst-case distribution for " +
                family_name)
                .c_str());

        // Wasserstein is recovered through the production OT path; the
        // Schuurmans helper's common interface is still exercised here.
        check(
            family ==
                    AmbiguityDivergence::WASSERSTEIN ||
                worst_case.feasible,
            (
                "valid configured ambiguity-set result for " +
                family_name)
                .c_str());

        const double robust_risk =
            expectation(
                worst_case.p,
                risk);

        check(
            robust_risk >=
                nominal_risk - 1e-8,
            (
                "worst-case expected risk is no smaller than nominal for " +
                family_name)
                .c_str());
    }

    // ------------------------------------------------------------------
    // Fixed-radius nesting.
    //
    // If the ambiguity set with radius rho1 is contained in the set with
    // rho2 >= rho1, maximizing the same risk objective over the larger set
    // cannot reduce the optimum.
    // ------------------------------------------------------------------
    for (const auto family : families) {
        const std::string family_name =
            schuurmans::divergence_name(
                family);

        const auto rho0 =
            schuurmans::worst_case_expectation(
                family,
                nominal,
                risk,
                0.0,
                &ground_cost);

        const auto rho_small =
            schuurmans::worst_case_expectation(
                family,
                nominal,
                risk,
                0.01,
                &ground_cost);

        const auto rho_medium =
            schuurmans::worst_case_expectation(
                family,
                nominal,
                risk,
                0.05,
                &ground_cost);

        const auto rho_large =
            schuurmans::worst_case_expectation(
                family,
                nominal,
                risk,
                0.20,
                &ground_cost);

        const double r0 =
            expectation(
                rho0.p,
                risk);

        const double r_small =
            expectation(
                rho_small.p,
                risk);

        const double r_medium =
            expectation(
                rho_medium.p,
                risk);

        const double r_large =
            expectation(
                rho_large.p,
                risk);

        check(
            std::abs(
                r0 - nominal_risk) <
                1e-7,
            (
                "zero ambiguity radius recovers nominal expectation for " +
                family_name)
                .c_str());

        check(
            r_small >= r0 - 1e-8 &&
                r_medium >=
                    r_small - 1e-8 &&
                r_large >=
                    r_medium - 1e-8,
            (
                "worst-case risk is nondecreasing with radius for " +
                family_name)
                .c_str());
    }

    // ------------------------------------------------------------------
    // Calibrated radius should contract as independent evidence grows.
    // ------------------------------------------------------------------
    for (const auto family : families) {
        const std::string family_name =
            schuurmans::divergence_name(
                family);

        const double r20 =
            schuurmans::ambiguity_radius(
                family,
                4,
                20,
                0.05,
                1.0);

        const double r200 =
            schuurmans::ambiguity_radius(
                family,
                4,
                200,
                0.05,
                1.0);

        const double r2000 =
            schuurmans::ambiguity_radius(
                family,
                4,
                2000,
                0.05,
                1.0);

        check(
            r20 >= r200 - 1e-12 &&
                r200 >= r2000 - 1e-12,
            (
                "ambiguity radius contracts with evidence for " +
                family_name)
                .c_str());
    }

    // ------------------------------------------------------------------
    // Smaller beta means a stronger confidence requirement and therefore
    // should not produce a smaller calibrated ambiguity radius.
    // ------------------------------------------------------------------
    for (const auto family : families) {
        const std::string family_name =
            schuurmans::divergence_name(
                family);

        const double beta001 =
            schuurmans::ambiguity_radius(
                family,
                4,
                200,
                0.01,
                1.0);

        const double beta005 =
            schuurmans::ambiguity_radius(
                family,
                4,
                200,
                0.05,
                1.0);

        const double beta010 =
            schuurmans::ambiguity_radius(
                family,
                4,
                200,
                0.10,
                1.0);

        check(
            beta001 >= beta005 - 1e-12 &&
                beta005 >= beta010 - 1e-12,
            (
                "stronger confidence gives no smaller radius for " +
                family_name)
                .c_str());
    }

    std::cout
        << (
               failures == 0
                   ? "ALL AMBIGUITY TESTS PASSED\n"
                   : "AMBIGUITY TESTS FAILED\n");

    return failures == 0 ? 0 : 1;
}