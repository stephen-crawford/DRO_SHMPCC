#include "experiment_config_yaml.hpp"
#include "mpc_controller.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace dro_mpc;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    RuntimeConfig invalid;
    invalid.mpc.nominal_resampling_baseline = true;
    bool rejected = false;
    try { invalid.validate(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "resampling flag must require hybrid recovery semantics");

    const ModeModel safe("safe", Eigen::Matrix4d::Identity(), Eigen::Vector4d::Zero(),
                         Eigen::MatrixXd::Zero(4, 2));
    const ModeModel threat("threat", Eigen::Matrix4d::Zero(), Eigen::Vector4d::Zero(),
                           Eigen::MatrixXd::Zero(4, 2));
    RuntimeConfig config;
    config.mpc.type = MPCType::SH_MPCC_DRO_FALLBACK;
    config.mpc.nominal_resampling_baseline = true;
    config.mpc.horizon = 4;
    config.mpc.sampling.set_manual_sample_count(8);
    config.mpc.ego.num_discs = 1;
    config.mpc.ego.length = 0;
    bool observed_resample_success = false;
    // Exercise a genuinely different second nominal draw, not an empty-road retry.
    for (unsigned seed = 1; seed <= 40 && !observed_resample_success; ++seed) {
        config.random_seed = seed;
        MPCController controller(config);
        require(!controller.config().dro.enabled, "nominal ablation must remain DRO-disabled after normalization");
        controller.initialize_obstacle(0, 0, {{"safe", safe}, {"threat", threat}});
        for (int i = 0; i < 90; ++i) controller.update_mode_observation(0, 0, "safe", i);
        for (int i = 90; i < 100; ++i) controller.update_mode_observation(0, 0, "threat", i);
        const auto start = std::chrono::steady_clock::now();
        const auto result = controller.solve(EgoState(0, 0, 0, 0),
            {{0, ObstacleState(5, 0, 0, 0)}}, Eigen::Vector2d(10, 0));
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        require(result.solve_time > 0 && result.solve_time <= elapsed + .001,
                "reported controller-cycle time must fit enclosing wall clock");
        require(controller.last_dro_results().empty(), "nominal retry must not compute DRO weights");
        require(controller.get_statistics().iteration_count == 1, "two attempts must count as one decision");
        if (result.nominal_fallback_attempted && result.success) {
            require(result.used_nominal_fallback && result.sampled_constraints_satisfied && result.first_input().has_value(),
                    "resampled nominal result must pass unchanged acceptance checks");
            observed_resample_success = true;
            std::cout << "PASS nominal resampling recovered: seed=" << seed
                      << " cycle_ms=" << 1000*result.solve_time << " external_ms=" << 1000*elapsed << '\n';
        }
    }
    require(observed_resample_success, "fixture must exercise failure followed by nominal resampling success");
    config.random_seed = 77;
    MPCController blocked(config);
    blocked.initialize_obstacle(0,0,{{"threat",threat}});
    const auto failure = blocked.solve(EgoState(0,0,0,0),{{0,ObstacleState(5,0,0,0)}},Eigen::Vector2d(10,0));
    require(failure.nominal_fallback_attempted && !failure.success && !failure.used_nominal_fallback,
            "two inadmissible nominal attempts must remain rejected");
    std::cout << "PASS both nominal attempts rejected\n";

    DROConfig risk_config;
    risk_config.radius_calibration.use_calibrated_radius = false;
    risk_config.radius_calibration.use_entropic_allocator = false;
    risk_config.radius_calibration.risk_measure = DRORiskMeasure::SURROGATE_VAR_BONFERRONI;
    risk_config.radius_calibration.alpha_one_sided = .5; // N=D=1 => z=0; isolates the mean.
    DRO dro(risk_config);
    std::cout << std::setprecision(17);
    for (const auto& displacement : {Eigen::Vector2d(0,0), Eigen::Vector2d(-5e-13,0),
                                     Eigen::Vector2d(0,5e-13), Eigen::Vector2d(.2,.3)}) {
        const auto r = dro.compute_worst_case_weights({{"safe",1.}}, {{"safe",100}},
            ObstacleState(displacement.x(),displacement.y(),0,0), {{"safe",safe}},
            std::vector<EgoState>(2,EgoState(0,0,0,0)),1,.5,.35,.1);
        const Eigen::Vector2d normal = displacement.norm()<1e-12 ? Eigen::Vector2d(1,0) :
                                                                  Eigen::Vector2d(displacement.normalized());
        const double expected = .95-normal.dot(displacement);
        const double observed = r.risk_per_mode.at("safe");
        require(std::abs(observed-expected)<2e-15, "surrogate must retain projected mean in fallback region");
        auto mixture_config = risk_config;
        mixture_config.radius_calibration.risk_measure = DRORiskMeasure::MIXTURE_VAR;
        DRO mixture(mixture_config);
        const auto mixture_result = mixture.compute_worst_case_weights({{"safe",1.}}, {{"safe",100}},
            ObstacleState(displacement.x(),displacement.y(),0,0), {{"safe",safe}},
            std::vector<EgoState>(2,EgoState(0,0,0,0)),1,.5,.35,.1);
        require(std::abs(mixture_result.risk_per_mode.at("safe")-expected)<2e-15,
                "held-mode mixture must retain the same projected Gaussian mean");
        std::cout << "PASS projected mean diff=(" << displacement.x() << ',' << displacement.y()
                  << ") observed=" << observed << " expected=" << expected << '\n';
    }
    // Only stage N is dangerous; shortening risk evaluation would miss it.
    std::vector<EgoState> ego(5,EgoState(0,0,0,0));
    ego.back() = EgoState(5,0,0,0);
    const auto full = dro.compute_worst_case_weights({{"safe",1.}},{{"safe",100}},
        ObstacleState(5,0,0,0),{{"safe",safe}},ego,4,.5,.35,.1,4);
    const auto short_risk = dro.compute_worst_case_weights({{"safe",1.}},{{"safe",100}},
        ObstacleState(5,0,0,0),{{"safe",safe}},ego,4,.5,.35,.1,1);
    require(full.risk_per_mode.at("safe")>.94 && short_risk.risk_per_mode.at("safe")==0,
            "full-horizon risk must include a threat appearing only at the final stage");
    std::cout << "PASS horizon diagnostic: full=" << full.risk_per_mode.at("safe")
              << " first_stage_only=" << short_risk.risk_per_mode.at("safe") << '\n';
}
