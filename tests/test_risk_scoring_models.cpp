// Direct comparison coverage for the standardized DRO risk-scoring layer.
//
// The test deliberately holds the W2-Bures ground cost and radius inputs fixed
// while varying only r[m]. It reports numerical deltas from a higher-sample
// JOINT_CVAR reference, measured risk-vector time, and the mathematical
// properties carried in DROResult::risk_diagnostics.
#include "dro.hpp"
#include "dynamics.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
    if (!condition) ++failures;
}

bool approx(double a, double b, double tolerance = 1e-11) {
    return std::abs(a - b) <= tolerance *
        std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

bool same_map(const std::map<std::string, double>& a,
              const std::map<std::string, double>& b,
              double tolerance = 1e-11) {
    if (a.size() != b.size()) return false;
    for (const auto& [id, value] : a) {
        const auto it = b.find(id);
        if (it == b.end() || !approx(value, it->second, tolerance)) return false;
    }
    return true;
}

bool same_matrix(const std::vector<std::vector<double>>& a,
                 const std::vector<std::vector<double>>& b,
                 double tolerance = 1e-12) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j)
            if (!approx(a[i][j], b[i][j], tolerance)) return false;
    }
    return true;
}

double risk_mae(const std::map<std::string, double>& a,
                const std::map<std::string, double>& b) {
    if (a.empty() || a.size() != b.size()) return 0.0;
    double sum = 0.0;
    for (const auto& [id, value] : a) sum += std::abs(value - b.at(id));
    return sum / static_cast<double>(a.size());
}

double weight_l1(const std::map<std::string, double>& a,
                 const std::map<std::string, double>& b) {
    double sum = 0.0;
    for (const auto& [id, value] : a) {
        const auto it = b.find(id);
        sum += std::abs(value - (it == b.end() ? 0.0 : it->second));
    }
    for (const auto& [id, value] : b)
        if (a.find(id) == a.end()) sum += std::abs(value);
    return sum;
}

struct Fixture {
    ObstacleState obstacle{6.0, 1.6, -1.0, -0.35};
    std::map<std::string, ModeModel> modes = create_obstacle_mode_models(0.1);
    std::vector<std::string> ids{
        "constant_velocity", "decelerating", "turn_left", "turn_right"};
    std::vector<EgoState> ego;
    int horizon = 12;
};

Fixture make_fixture() {
    Fixture fixture;
    for (int k = 0; k <= fixture.horizon; ++k) {
        EgoState state;
        state.x = 0.4 * static_cast<double>(k);
        state.y = 0.0;
        state.theta = 0.0;
        state.v = 4.0;
        fixture.ego.push_back(state);
    }
    return fixture;
}

std::map<std::string, double> uniform_nominal(const Fixture& fixture) {
    std::map<std::string, double> nominal;
    const double weight = 1.0 / static_cast<double>(fixture.ids.size());
    for (const auto& id : fixture.ids) nominal[id] = weight;
    return nominal;
}

Eigen::MatrixXd sticky_chain(int modes, double persistence = 0.75) {
    Eigen::MatrixXd transition(modes, modes);
    const double off_diagonal = (1.0 - persistence) / std::max(1, modes - 1);
    for (int i = 0; i < modes; ++i)
        for (int j = 0; j < modes; ++j)
            transition(i, j) = (i == j) ? persistence : off_diagonal;
    return transition;
}

DROResult evaluate(
    const Fixture& fixture,
    DRORiskScoringModel scoring_model,
    DRORiskMeasure configured_measure,
    int joint_samples,
    const Eigen::MatrixXd* transition = nullptr,
    uint64_t seed = 0xA51CE55ULL
) {
    DROConfig config;
    config.ground_cost_type = DROGroundCostType::W2_BURES;
    config.min_radius = 0.0;
    config.max_radius = 2.0;
    config.base_radius = 0.12;
    config.radius_calibration.use_calibrated_radius = false;
    // Use the production-default entropic allocator in this comparison so
    // differences in score magnitude, not just ranking, appear in q-star.
    config.radius_calibration.use_entropic_allocator = true;
    config.radius_calibration.use_primal_ot = true;
    config.radius_calibration.risk_measure = configured_measure;
    config.radius_calibration.risk_scoring_model = scoring_model;
    config.radius_calibration.alpha_one_sided = 0.95;
    config.radius_calibration.joint_risk_samples = joint_samples;
    config.radius_calibration.joint_risk_seed = seed;
    config.radius_calibration.mixture_sequence_samples = 256;

    DRO dro(config);
    dro.set_observation_count(120);
    return dro.compute_worst_case_weights(
        uniform_nominal(fixture), fixture.obstacle, fixture.modes, fixture.ego,
        fixture.horizon, /*ego radius=*/0.5, /*obstacle radius=*/0.35,
        /*margin=*/0.1, /*risk horizon=*/-1, /*discs=*/1,
        /*vehicle length=*/4.0, transition);
}

struct ProfileCase {
    const char* name;
    DRORiskScoringModel profile;
    DRORiskMeasure expected_effective_measure;
};

}  // namespace

int main() {
    const Fixture fixture = make_fixture();
    const DRORiskMeasure configured_measure = DRORiskMeasure::MIXTURE_CVAR;
    const std::vector<ProfileCase> profiles = {
        {"inherit_mixture_cvar", DRORiskScoringModel::INHERIT_RISK_MEASURE,
         DRORiskMeasure::MIXTURE_CVAR},
        {"certified_surrogate", DRORiskScoringModel::CERTIFIED_SURROGATE,
         DRORiskMeasure::SURROGATE_VAR_BONFERRONI},
        {"euclidean_bonferroni_var", DRORiskScoringModel::EUCLIDEAN_BONFERRONI_VAR,
         DRORiskMeasure::BONFERRONI_VAR},
        {"euclidean_joint_var", DRORiskScoringModel::EUCLIDEAN_JOINT_VAR,
         DRORiskMeasure::JOINT_VAR},
        {"euclidean_joint_cvar", DRORiskScoringModel::EUCLIDEAN_JOINT_CVAR,
         DRORiskMeasure::JOINT_CVAR},
    };

    // A higher-sample true-Euclidean joint-CVaR calculation is the comparison
    // reference; it deliberately uses an independent deterministic stream.
    const DROResult oracle = evaluate(
        fixture, DRORiskScoringModel::EUCLIDEAN_JOINT_CVAR,
        DRORiskMeasure::SURROGATE_VAR, /*samples=*/24000,
        /*transition=*/nullptr, /*seed=*/0xC0FFEE42ULL);

    struct Evaluation { ProfileCase profile; DROResult result; };
    std::vector<Evaluation> evaluations;
    evaluations.reserve(profiles.size());

    std::printf("\nRisk-score comparison (fixed W2-Bures D and rho)\n");
    std::printf("model,effective,true_euclidean,joint_horizon,switching,"
                "population_bound,coherent,finite_sampling,risk_ms,"
                "risk_mae_to_joint_cvar,q_l1_to_joint_cvar\n");

    for (const auto& profile : profiles) {
        const DROResult first = evaluate(
            fixture, profile.profile, configured_measure, /*samples=*/2048);
        const DROResult repeat = evaluate(
            fixture, profile.profile, configured_measure, /*samples=*/2048);

        const auto& diagnostics = first.risk_diagnostics;
        const auto& properties = diagnostics.properties;
        std::printf("%s,%s,%d,%d,%d,%d,%d,%d,%.4f,%.8f,%.8f\n",
                    profile.name,
                    risk_measure_name(diagnostics.effective_measure).c_str(),
                    properties.evaluates_true_euclidean_violation ? 1 : 0,
                    properties.covers_joint_horizon ? 1 : 0,
                    properties.supports_mode_switching ? 1 : 0,
                    properties.population_joint_var_upper_bound ? 1 : 0,
                    properties.coherent_joint_risk ? 1 : 0,
                    properties.uses_finite_sampling ? 1 : 0,
                    diagnostics.evaluation_seconds * 1000.0,
                    risk_mae(first.risk_per_mode, oracle.risk_per_mode),
                    weight_l1(first.worst_case_weights, oracle.worst_case_weights));

        check(diagnostics.effective_measure == profile.expected_effective_measure,
              "risk-scoring profile resolves to its expected existing implementation");
        check(diagnostics.evaluation_seconds >= 0.0,
              "risk diagnostics expose nonnegative evaluation time");
        check(same_map(first.risk_per_mode, repeat.risk_per_mode) &&
                  same_map(first.worst_case_weights, repeat.worst_case_weights),
              "fixed profile, seed, and inputs produce reproducible risk and q-star");
        evaluations.push_back({profile, first});
    }

    const DROResult& baseline = evaluations.front().result;
    bool geometry_invariant = true;
    bool rho_invariant = true;
    for (const auto& evaluation : evaluations) {
        geometry_invariant = geometry_invariant && same_matrix(
            baseline.transport_cost_matrix, evaluation.result.transport_cost_matrix);
        rho_invariant = rho_invariant && approx(
            baseline.rho_used, evaluation.result.rho_used);
    }
    check(geometry_invariant,
          "all profiles retain identical W2-Bures transport geometry");
    check(rho_invariant,
          "all profiles retain the same ambiguity radius");

    const auto& certified = evaluations[1].result.risk_diagnostics.properties;
    const auto& bonferroni = evaluations[2].result.risk_diagnostics.properties;
    const auto& joint_cvar = evaluations[4].result.risk_diagnostics.properties;
    check(certified.population_joint_var_upper_bound &&
              !certified.evaluates_true_euclidean_violation &&
              !certified.uses_finite_sampling,
          "certified surrogate advertises its analytic population joint-VaR bound");
    check(bonferroni.population_joint_var_upper_bound &&
              bonferroni.evaluates_true_euclidean_violation &&
              bonferroni.uses_finite_sampling,
          "Euclidean Bonferroni advertises its population bound and MC error source");
    check(joint_cvar.evaluates_true_euclidean_violation &&
              joint_cvar.covers_joint_horizon && joint_cvar.coherent_joint_risk,
          "Euclidean joint CVaR advertises true-distance, joint, coherent semantics");

    std::printf("\nSwitching semantics\n");
    const Eigen::MatrixXd transition = sticky_chain(
        static_cast<int>(fixture.ids.size()));
    bool certified_rejected = false;
    bool inherited_surrogate_rejected = false;
    try {
        (void)evaluate(fixture, DRORiskScoringModel::CERTIFIED_SURROGATE,
                       configured_measure, 1024, &transition);
    } catch (const std::invalid_argument&) {
        certified_rejected = true;
    }
    try {
        (void)evaluate(fixture, DRORiskScoringModel::INHERIT_RISK_MEASURE,
                       DRORiskMeasure::SURROGATE_CVAR, 1024, &transition);
    } catch (const std::invalid_argument&) {
        inherited_surrogate_rejected = true;
    }
    check(certified_rejected,
          "held-mode certified surrogate rejects a Markov transition matrix");
    check(inherited_surrogate_rejected,
          "inherited surrogate CVaR cannot silently become joint VaR under switching");

    const DROResult explicit_joint_var = evaluate(
        fixture, DRORiskScoringModel::EUCLIDEAN_JOINT_VAR,
        configured_measure, 2048, &transition);
    const DROResult explicit_joint_cvar = evaluate(
        fixture, DRORiskScoringModel::EUCLIDEAN_JOINT_CVAR,
        configured_measure, 2048, &transition);

    bool cvar_ge_var = true;
    for (const auto& [id, risk] : explicit_joint_cvar.risk_per_mode)
        cvar_ge_var = cvar_ge_var &&
            risk + 1e-12 >= explicit_joint_var.risk_per_mode.at(id);
    check(cvar_ge_var,
          "switching-aware joint CVaR is at least joint VaR with common samples");

    std::printf("\n%s (%d checks failed)\n",
                failures == 0 ? "ALL RISK-SCORING TESTS PASSED"
                              : "SOME RISK-SCORING TESTS FAILED",
                failures);
    return failures == 0 ? 0 : 1;
}
