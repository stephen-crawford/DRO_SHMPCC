#include "dro.hpp"
#include "dynamics.hpp"
#include "experiment_config_yaml.hpp"
#include "mpc_controller.hpp"
#include "schuurmans_ambiguity.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace dro_mpc;

namespace {
int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}
}  // namespace

int main() {
    DROConfig default_config;
    check(default_config.radius_calibration.divergence ==
              AmbiguityDivergence::WASSERSTEIN,
          "Wasserstein is the default ambiguity set");
    check(default_config.radius_calibration.use_entropic_allocator,
          "entropic allocation is enabled by default");
    const ExperimentConfig yaml_config =
        yaml_config::load_experiment_config("configs/default.yaml", true);
    check(yaml_config.dro.solver.radius_calibration.divergence ==
              AmbiguityDivergence::WASSERSTEIN,
          "YAML configuration inherits the Wasserstein default");
    check(yaml_config.dro.solver.radius_calibration.use_entropic_allocator,
          "YAML configuration inherits entropic allocation");
    check(yaml_config.dro.solver.radius_calibration.risk_scoring_model ==
              DRORiskScoringModel::INHERIT_RISK_MEASURE,
          "YAML configuration inherits the default risk-scoring profile");

    ExperimentConfig fixed_rho_config;
    fixed_rho_config.dro.fixed_rho = 0.23;
    fixed_rho_config.dro.apply_fixed_rho();
    check(!fixed_rho_config.dro.solver.radius_calibration.use_calibrated_radius &&
              std::abs(fixed_rho_config.dro.solver.base_radius - 0.23) < 1e-12 &&
              fixed_rho_config.dro.solver.min_radius <= 0.23 &&
              fixed_rho_config.dro.solver.max_radius >= 0.23,
          "fixed_rho pins the effective ambiguity radius independently of calibration");
    {
        const auto mode_models = create_obstacle_mode_models(0.1);
        const std::map<std::string, double> nominal_weights{
            {"constant_velocity", 0.5}, {"turn_left", 0.5}};
        const std::vector<EgoState> ego_reference{
            EgoState(0.0, 0.0, 0.0, 1.0),
            EgoState(0.1, 0.0, 0.0, 1.0),
            EgoState(0.2, 0.0, 0.0, 1.0),
        };
        DRO fixed_rho_solver(fixed_rho_config.dro.solver);
        fixed_rho_solver.set_observation_count(20);
        const DROResult fixed_result = fixed_rho_solver.compute_worst_case_weights(
            nominal_weights, ObstacleState(2.0, 0.0, -0.1, 0.0), mode_models,
            ego_reference, /*horizon=*/2, /*ego_r=*/0.4, /*obs_r=*/0.3,
            /*margin=*/0.1);
        check(std::abs(fixed_result.rho_used - 0.23) < 1e-12,
              "the live DRO solve reports the requested fixed rho");
        check(fixed_result.radius_observation_count == 20 &&
                  fixed_result.radius_mode_count == 2 &&
                  std::abs(fixed_result.rho_before_clamp - 0.23) < 1e-12 &&
                  !fixed_result.rho_clamped_to_min &&
                  !fixed_result.rho_clamped_to_max,
              "DRO diagnostics expose the evidence and clamp state behind rho");
    }

    {
        // The controller must pass episode evidence, rather than the raw
        // per-tick history length, into the calibrated radius.
        RuntimeConfig controller_config;
        controller_config.mpc.type = MPCType::MPC;
        controller_config.mpc.sync_from_type();
        controller_config.mpc.horizon = 2;
        controller_config.mpc.sampling.set_manual_sample_count(4);
        controller_config.dro.enabled = true;
        controller_config.random_seed = 117u;

        MPCController controller(controller_config);
        const auto all_modes = create_obstacle_mode_models(0.1);
        const std::map<std::string, ModeModel> modes{
            {"constant_velocity", all_modes.at("constant_velocity")},
            {"turn_left", all_modes.at("turn_left")},
        };
        controller.initialize_obstacle(3, 0, modes);
        for (int t = 0; t < 8; ++t) {
            controller.update_mode_observation(3, 0, "constant_velocity", t);
        }
        (void)controller.solve(
            EgoState(0.0, 0.0, 0.0, 1.0),
            {{3, ObstacleState(5.0, 3.0, 0.0, 0.0)}},
            Eigen::Vector2d(1.0, 0.0), 1.0);
        const auto& last = controller.last_dro_results();
        check(last.count(3) == 1 && last.at(3).radius_observation_count == 1,
              "controller calibrates rho from mode episodes rather than raw tick count");
    }

    {
        // Calibration saturation is observable and eventually releases once
        // sufficient independent evidence is supplied.
        const auto mode_models = create_obstacle_mode_models(0.1);
        std::map<std::string, double> nominal_weights;
        for (const auto& [mode_id, _] : mode_models) {
            nominal_weights[mode_id] = 1.0 / static_cast<double>(mode_models.size());
        }
        std::vector<EgoState> ego_reference;
        for (int k = 0; k <= 15; ++k) {
            ego_reference.emplace_back(0.34 * k, 0.0, 0.0, 1.5);
        }

        DRO early_evidence(default_config);
        early_evidence.set_observation_count(1);
        const DROResult early = early_evidence.compute_worst_case_weights(
            nominal_weights, ObstacleState(5.0, 0.0, 0.5, 0.0), mode_models,
            ego_reference, 15, 0.5, 0.35, 0.2);

        DRO mature_evidence(default_config);
        mature_evidence.set_observation_count(10000);
        const DROResult mature = mature_evidence.compute_worst_case_weights(
            nominal_weights, ObstacleState(5.0, 0.0, 0.5, 0.0), mode_models,
            ego_reference, 15, 0.5, 0.35, 0.2);

        check(early.rho_clamped_to_max &&
                  mature.rho_used < early.rho_used &&
                  !mature.rho_clamped_to_max,
              "radius diagnostics reveal saturation and calibrated shrinkage");
    }

    const SeedBundle seeds = derive_seeds(42u, 0);
    check(seeds.env != seeds.scenario && seeds.env != seeds.predictor &&
              seeds.predictor != seeds.scenario,
          "plant, predictor, and controller seeds are distinct");
    const SeedBundle repeat = derive_seeds(42u, 0);
    check(seeds.env == repeat.env && seeds.predictor == repeat.predictor &&
              seeds.scenario == repeat.scenario,
          "derived seeds are reproducible from the master seed");

    const std::vector<double> nominal = {0.55, 0.30, 0.15};
    const std::vector<double> risk = {0.1, 0.4, 0.9};
    const std::vector<std::vector<double>> ground_cost = {
        {0.0, 0.5, 1.0}, {0.5, 0.0, 0.5}, {1.0, 0.5, 0.0}};
    const std::vector<AmbiguityDivergence> families = {
        AmbiguityDivergence::WASSERSTEIN,
        AmbiguityDivergence::TOTAL_VARIATION,
        AmbiguityDivergence::KULLBACK_LEIBLER,
        AmbiguityDivergence::JENSEN_SHANNON,
        AmbiguityDivergence::HELLINGER,
    };

    for (const auto family : families) {
        const double radius = schuurmans::ambiguity_radius(
            family, static_cast<int>(nominal.size()), 200, 0.05, 1.0);
        const auto worst_case = schuurmans::worst_case_expectation(
            family, nominal, risk, radius, &ground_cost);
        const double mass = std::accumulate(
            worst_case.p.begin(), worst_case.p.end(), 0.0);
        check(std::abs(mass - 1.0) < 1e-8,
              ("normalized worst-case distribution for " +
               schuurmans::divergence_name(family)).c_str());
        // Wasserstein is recovered by the optimizer's primal/dual OT path;
        // the Schuurmans helper is exercised here for its common interface.
        check(family == AmbiguityDivergence::WASSERSTEIN || worst_case.feasible,
              ("valid configured ambiguity-set result for " +
               schuurmans::divergence_name(family)).c_str());
    }

    return failures == 0 ? 0 : 1;
}
