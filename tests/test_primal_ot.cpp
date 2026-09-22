#include "primal_ot.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

bool close(double lhs, double rhs, double tolerance = 1e-9) {
    return std::abs(lhs - rhs) <= tolerance;
}

double expected_risk(
    const std::map<std::string, double>& q,
    const std::map<std::string, double>& risk)
{
    double value = 0.0;
    for (const auto& [mode, probability] : q) {
        value += probability * risk.at(mode);
    }
    return value;
}

double total_mass(const std::map<std::string, double>& q) {
    double value = 0.0;
    for (const auto& [_, probability] : q) {
        value += probability;
    }
    return value;
}

}  // namespace

int main() {
    const std::vector<std::string> ids{"A", "B"};
    const std::vector<std::vector<double>> cost{
        {0.0, 1.0},
        {1.0, 0.0},
    };

    // ---------------------------------------------------------------------
    // Existing fractional transport test.
    // ---------------------------------------------------------------------
    {
        const PrimalOTResult result = solve_primal_ot(
            {{"A", 1.0}, {"B", 0.0}},
            {{"A", 0.0}, {"B", 10.0}},
            cost, ids, 0.4);

        check(result.solved &&
                  close(result.q.at("A"), 0.6) &&
                  close(result.q.at("B"), 0.4) &&
                  close(result.transport_cost, 0.4) &&
                  close(result.expected_risk, 4.0),
              "OT splits source mass fractionally at an active transport budget");

        check(close(result.plan[0][0] + result.plan[0][1], 1.0) &&
                  close(result.plan[1][0] + result.plan[1][1], 0.0),
              "OT plan preserves each source marginal");
    }

    // ---------------------------------------------------------------------
    // rho = 0 must recover the empirical distribution exactly.
    // ---------------------------------------------------------------------
    {
        const std::map<std::string, double> nominal{
            {"A", 0.3},
            {"B", 0.7},
        };

        const std::map<std::string, double> risk{
            {"A", 1.0},
            {"B", 3.0},
        };

        const PrimalOTResult result =
            solve_primal_ot(nominal, risk, cost, ids, 0.0);

        check(result.solved &&
                  close(result.q.at("A"), 0.3) &&
                  close(result.q.at("B"), 0.7) &&
                  close(result.transport_cost, 0.0) &&
                  close(result.expected_risk, 2.4),
              "zero radius leaves the nominal distribution unchanged");
    }

    // ---------------------------------------------------------------------
    // Existing full transport / determinism test.
    // ---------------------------------------------------------------------
    {
        const std::map<std::string, double> nominal{
            {"A", 0.4},
            {"B", 0.6},
        };

        const std::map<std::string, double> risk{
            {"A", 1.0},
            {"B", 3.0},
        };

        const PrimalOTResult first =
            solve_primal_ot(nominal, risk, cost, ids, 1.0);

        const PrimalOTResult second =
            solve_primal_ot(nominal, risk, cost, ids, 1.0);

        check(first.solved &&
                  close(first.q.at("A"), 0.0) &&
                  close(first.q.at("B"), 1.0) &&
                  close(first.transport_cost, 0.4) &&
                  close(first.expected_risk, 3.0),
              "slack budget transports all low-risk movable mass to the argmax risk mode");

        check(first.plan == second.plan &&
                  first.q == second.q &&
                  close(first.expected_risk, second.expected_risk),
              "primal OT is deterministic for a fixed problem");
    }

    // ---------------------------------------------------------------------
    // New: rare high-risk mode must receive adversarial mass.
    // Also test monotonicity with ambiguity radius.
    // ---------------------------------------------------------------------
    {
        const std::vector<std::string> mode_ids{
            "safe", "m1", "m2", "danger"
        };

        // Zero-one metric over modes.
        const std::vector<std::vector<double>> unit_cost{
            {0.0, 1.0, 1.0, 1.0},
            {1.0, 0.0, 1.0, 1.0},
            {1.0, 1.0, 0.0, 1.0},
            {1.0, 1.0, 1.0, 0.0},
        };

        const std::map<std::string, double> nominal{
            {"safe",   0.97},
            {"m1",     0.01},
            {"m2",     0.01},
            {"danger", 0.01},
        };

        const std::map<std::string, double> risk{
            {"safe",   0.0},
            {"m1",     0.0},
            {"m2",     0.0},
            {"danger", 1.0},
        };

        const auto rho0 =
            solve_primal_ot(nominal, risk, unit_cost, mode_ids, 0.0);

        const auto rho005 =
            solve_primal_ot(nominal, risk, unit_cost, mode_ids, 0.05);

        const auto rho020 =
            solve_primal_ot(nominal, risk, unit_cost, mode_ids, 0.20);

        check(rho0.solved &&
                  close(rho0.q.at("danger"), 0.01),
              "rho=0 leaves rare dangerous-mode probability unchanged");

        check(rho005.solved &&
                  rho005.q.at("danger") > nominal.at("danger") + 1e-12,
              "positive ambiguity radius upweights a rare uniquely dangerous mode");

        check(rho020.solved &&
                  rho020.q.at("danger") >= rho005.q.at("danger") - 1e-12,
              "larger ambiguity radius does not reduce mass on the uniquely highest-risk mode");

        check(rho0.expected_risk <= rho005.expected_risk + 1e-12 &&
                  rho005.expected_risk <= rho020.expected_risk + 1e-12,
              "worst-case expected risk is nondecreasing with ambiguity radius");

        check(close(total_mass(rho005.q), 1.0) &&
                  close(total_mass(rho020.q), 1.0),
              "adversarial distributions remain normalized");

        check(rho005.expected_risk >= expected_risk(nominal, risk) - 1e-12,
              "OT worst-case expected risk is no smaller than nominal expected risk");
    }

    // ---------------------------------------------------------------------
    // New: changing which mode is risky must change where the mass goes.
    // This catches mode-order or hard-coded-index mistakes.
    // ---------------------------------------------------------------------
    {
        const std::vector<std::string> mode_ids{
            "safe", "m1", "m2", "danger"
        };

        const std::vector<std::vector<double>> unit_cost{
            {0.0, 1.0, 1.0, 1.0},
            {1.0, 0.0, 1.0, 1.0},
            {1.0, 1.0, 0.0, 1.0},
            {1.0, 1.0, 1.0, 0.0},
        };

        const std::map<std::string, double> nominal{
            {"safe",   0.25},
            {"m1",     0.25},
            {"m2",     0.25},
            {"danger", 0.25},
        };

        const std::map<std::string, double> risk_safe{
            {"safe",   1.0},
            {"m1",     0.0},
            {"m2",     0.0},
            {"danger", 0.0},
        };

        const std::map<std::string, double> risk_danger{
            {"safe",   0.0},
            {"m1",     0.0},
            {"m2",     0.0},
            {"danger", 1.0},
        };

        const auto safe_high =
            solve_primal_ot(
                nominal, risk_safe, unit_cost, mode_ids, 0.20);

        const auto danger_high =
            solve_primal_ot(
                nominal, risk_danger, unit_cost, mode_ids, 0.20);

        check(safe_high.solved &&
                  safe_high.q.at("safe") > nominal.at("safe"),
              "OT moves probability to safe-id when that id carries highest risk");

        check(danger_high.solved &&
                  danger_high.q.at("danger") > nominal.at("danger"),
              "OT moves probability to danger-id when that id carries highest risk");

        check(safe_high.q.at("safe") >
                  danger_high.q.at("safe") + 1e-12 &&
                  danger_high.q.at("danger") >
                  safe_high.q.at("danger") + 1e-12,
              "adversarial mass follows risk values rather than fixed mode ordering");
    }

    std::cout << (
        failures == 0
            ? "ALL PRIMAL-OT TESTS PASSED\n"
            : "PRIMAL-OT TESTS FAILED\n");

    return failures == 0 ? 0 : 1;
}