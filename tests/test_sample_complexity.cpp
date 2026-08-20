// test_sample_complexity: verifies the de Groot (arXiv:2307.01070) scenario-theoretic
// sample-complexity calculation implemented in RuntimeConfig.
//
//  (A) The realized-risk map ε(n) = 1 - (β/(S·C(S,n)))^{1/(S-n)} (de Groot Eq. 8):
//      monotone increasing in support n, decreasing in sample size S, ε(n)=1 for n≥S.
//  (B) compute_required_scenarios (de Groot Alg. 1 line 2 = bisection of Eq. 8)
//      reproduces the paper's published numbers: at ε=0.1, β=1e-6 with support limit
//      n̄=2, S(R=0)≈290 and S(R=20)≈1250 (paper reports S∈{290,…,1250} for R∈{0,…,20}).
//  (C) The exact bound is a genuine certificate and is TIGHTER than the old closed-form
//      Alamo/Campi upper bound it replaced.
#include "config.hpp"
#include <cstdio>
#include <cmath>

using namespace dro_mpc;
static int fails = 0;
static void check(bool ok, const char* msg) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", msg);
    if (!ok) ++fails;
}

// Build a config with the given ε (via confidence) and β.
static RuntimeConfig cfg_with(double eps, double beta) {
    RuntimeConfig c;
    c.mpc.sampling.one_minus_chance_constraint_violation_probability = 1.0 - eps;
    c.mpc.sampling.chance_of_certificate_violation = beta;
    return c;
}

int main() {
    std::printf("=== (A) de Groot Eq. 8 risk map ε(n) ===\n");
    {
        const double beta = 1e-6;
        // Monotone increasing in support n (more support => higher risk).
        bool mono = true;
        for (int n = 1; n < 40; ++n)
            if (!(RuntimeConfig::degroot_violation_risk(500, n, beta) >
                  RuntimeConfig::degroot_violation_risk(500, n - 1, beta))) mono = false;
        check(mono, "ε(n) is monotonically increasing in the support size n");
        // Decreasing in S (more samples => lower risk at fixed support).
        check(RuntimeConfig::degroot_violation_risk(1000, 5, beta) <
              RuntimeConfig::degroot_violation_risk(200, 5, beta),
              "ε(n) decreases as the sample size S grows");
        // Under-sampled: n >= S => no guarantee.
        check(RuntimeConfig::degroot_violation_risk(10, 10, beta) == 1.0 &&
              RuntimeConfig::degroot_violation_risk(10, 15, beta) == 1.0,
              "ε(n)=1 (no guarantee) when support n >= sample size S");
    }

    std::printf("\n=== (B) compute_required_scenarios reproduces de Groot's paper numbers ===\n");
    {
        // Paper: ε=0.1, β=1e-6, support limit n̄=2, R∈{0,2,...,20} -> S∈{290,390,...,1250}.
        RuntimeConfig c = cfg_with(0.1, 1e-6);
        int S0  = c.compute_required_scenarios(/*nbar=*/2, /*removal=*/0);
        int S20 = c.compute_required_scenarios(/*nbar=*/2, /*removal=*/20);
        std::printf("    n̄=2: S(R=0)=%d (paper 290), S(R=20)=%d (paper 1250)\n", S0, S20);
        check(std::abs(S0 - 290) <= 3, "S(R=0) matches paper's 290 within rounding");
        check(std::abs(S20 - 1250) <= 3, "S(R=20) matches paper's 1250 within rounding");
        // The returned S actually certifies the requested support (ε(n̄+R) ≤ ε).
        check(RuntimeConfig::degroot_violation_risk(S0, 2, 1e-6) <= 0.1 &&
              RuntimeConfig::degroot_violation_risk(S0 - 1, 2, 1e-6) > 0.1,
              "returned S is the SMALLEST sample size certifying ε(n̄) ≤ ε");
        // Monotone in the removal budget.
        check(c.compute_required_scenarios(2, 0) < c.compute_required_scenarios(2, 10),
              "required S grows with the removal budget R (support n̄+R)");
    }

    std::printf("\n=== (C) exact bound is a valid, tighter certificate ===\n");
    {
        // At the code defaults (ε=0.05, β=0.01, n̄=5) the exact bound is ~781 and is
        // strictly below the old closed-form Alamo/Campi upper bound (~932).
        RuntimeConfig c = cfg_with(0.05, 0.01);
        int S_exact = c.compute_required_scenarios(5);
        double closed_form = (2.0 / 0.05) * std::log(1.0 / 0.01)
                           + 2.0 * 5 + (2.0 * 5 / 0.05) * std::log(2.0 / 0.05);
        std::printf("    n̄=5: exact=%d, old closed-form=%d\n",
                    S_exact, static_cast<int>(std::ceil(closed_form)));
        check(S_exact <= static_cast<int>(std::ceil(closed_form)),
              "exact de Groot bound <= old closed-form upper bound (tighter)");
        check(RuntimeConfig::degroot_violation_risk(S_exact, 5, 0.01) <= 0.05,
              "exact bound genuinely certifies ε(n̄) ≤ ε at the defaults");
        // Reference values (de Groot-style Eq. 8 inversion, ε=0.05, β=0.01), and the
        // steep sensitivity to the total support that motivates counting removal.
        check(S_exact == 781, "n̄=5 -> S*=781 (reference)");
        check(c.compute_required_scenarios(6) == 895 &&
              c.compute_required_scenarios(7) == 1009 &&
              c.compute_required_scenarios(9) == 1237,
              "n̄=6/7/9 -> S*=895/1009/1237 (each extra support scenario costs ~114 samples)");
        // compute_effective_epsilon is the inverse: the risk realized by S scenarios.
        check(c.compute_effective_epsilon(S_exact, 5) <= 0.05 + 1e-12,
              "compute_effective_epsilon(S_required, n̄) <= ε (consistent inverse)");
    }

    std::printf("\n%s (%d checks failed)\n",
                fails == 0 ? "ALL SAMPLE-COMPLEXITY TESTS PASSED" : "SOME TESTS FAILED", fails);
    return fails == 0 ? 0 : 1;
}
