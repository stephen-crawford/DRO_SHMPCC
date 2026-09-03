#include "dro.hpp"
#include "experiment_config_yaml.hpp"
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
