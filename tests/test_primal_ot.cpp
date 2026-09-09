#include "primal_ot.hpp"

#include <cmath>
#include <iostream>

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

}  // namespace

int main() {
    const std::vector<std::string> ids{"A", "B"};
    const std::vector<std::vector<double>> cost{{0.0, 1.0}, {1.0, 0.0}};

    {
        const PrimalOTResult result = solve_primal_ot(
            {{"A", 1.0}, {"B", 0.0}}, {{"A", 0.0}, {"B", 10.0}},
            cost, ids, 0.4);
        check(result.solved && close(result.q.at("A"), 0.6) &&
                  close(result.q.at("B"), 0.4) && close(result.transport_cost, 0.4) &&
                  close(result.expected_risk, 4.0),
              "OT splits source mass fractionally at an active transport budget");
        check(close(result.plan[0][0] + result.plan[0][1], 1.0) &&
                  close(result.plan[1][0] + result.plan[1][1], 0.0),
              "OT plan preserves each source marginal");
    }

    {
        const std::map<std::string, double> nominal{{"A", 0.3}, {"B", 0.7}};
        const std::map<std::string, double> risk{{"A", 1.0}, {"B", 3.0}};
        const PrimalOTResult result = solve_primal_ot(nominal, risk, cost, ids, 0.0);
        check(result.solved && close(result.q.at("A"), 0.3) &&
                  close(result.q.at("B"), 0.7) && close(result.transport_cost, 0.0) &&
                  close(result.expected_risk, 2.4),
              "zero radius leaves the nominal distribution unchanged");
    }

    {
        const std::map<std::string, double> nominal{{"A", 0.4}, {"B", 0.6}};
        const std::map<std::string, double> risk{{"A", 1.0}, {"B", 3.0}};
        const PrimalOTResult first = solve_primal_ot(nominal, risk, cost, ids, 1.0);
        const PrimalOTResult second = solve_primal_ot(nominal, risk, cost, ids, 1.0);
        check(first.solved && close(first.q.at("A"), 0.0) && close(first.q.at("B"), 1.0) &&
                  close(first.transport_cost, 0.4) && close(first.expected_risk, 3.0),
              "slack budget transports all low-risk movable mass to the argmax risk mode");
        check(first.plan == second.plan && first.q == second.q &&
                  close(first.expected_risk, second.expected_risk),
              "primal OT is deterministic for a fixed problem");
    }

    std::cout << (failures == 0 ? "ALL PRIMAL-OT TESTS PASSED\n"
                                : "PRIMAL-OT TESTS FAILED\n");
    return failures == 0 ? 0 : 1;
}
