#include "experiment_harness.hpp"

#include <cmath>
#include <iostream>
#include <random>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

bool close(double lhs, double rhs, double tolerance = 1e-12) {
    return std::abs(lhs - rhs) <= tolerance;
}

}  // namespace

int main() {
    {
        const auto [lower, upper] = wilson_ci(50, 100);
        check(close(lower + upper, 1.0, 1e-12) && lower < 0.5 && upper > 0.5 &&
                  wilson_ci(0, 0) == std::pair<double, double>{0.0, 1.0},
              "Wilson intervals are bounded and symmetric at a 50% rate");
    }

    check(close(percentile({1.0, 3.0, 5.0, 9.0}, 50.0), 4.0) &&
              close(percentile({1.0, 3.0, 5.0, 9.0}, -10.0), 1.0) &&
              close(percentile({1.0, 3.0, 5.0, 9.0}, 120.0), 9.0),
          "percentile interpolation and endpoint clamping are numerically exact");

    {
        std::mt19937 first_rng(42);
        std::mt19937 second_rng(42);
        const std::vector<bool> base{true, false, true, false};
        const std::vector<bool> comparison{false, false, false, false};
        const auto first = bootstrap_paired_delta(base, comparison, 500, &first_rng);
        const auto second = bootstrap_paired_delta(base, comparison, 500, &second_rng);
        check(close(first.mean_delta, second.mean_delta) &&
                  close(first.ci_low, second.ci_low) && close(first.ci_high, second.ci_high) &&
                  first.ci_low <= first.mean_delta && first.mean_delta <= first.ci_high,
              "paired bootstrap is reproducible with a caller-owned RNG");
    }

    {
        const EffectSizes effects = compute_effect_sizes(0.4, 0.2);
        check(close(mcnemar_chi2(10, 2), 49.0 / 12.0) && close(mcnemar_chi2(0, 0), 0.0) &&
                  close(effects.abs_delta, 0.2) && close(effects.rel_delta, 0.5) &&
                  close(effects.risk_ratio, 0.5) && effects.cohens_h > 0.0,
              "effect-size and McNemar utilities satisfy their closed-form values");
    }

    {
        const SeedBundle first = derive_seeds(1234, 7);
        const SeedBundle second = derive_seeds(1234, 7);
        const SeedBundle next = derive_seeds(1234, 8);
        check(first.env == second.env && first.predictor == second.predictor &&
                  first.scenario == second.scenario && first.env != first.predictor &&
                  first.predictor != first.scenario && first.env != next.env,
              "derived plant, predictor, and controller seeds are stable and distinct");
    }

    std::cout << (failures == 0 ? "ALL EXPERIMENT-STATISTICS TESTS PASSED\n"
                                : "EXPERIMENT-STATISTICS TESTS FAILED\n");
    return failures == 0 ? 0 : 1;
}
