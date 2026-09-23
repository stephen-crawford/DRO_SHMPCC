#include "mpc_controller.hpp"
#include <iostream>
#include <stdexcept>
using namespace dro_mpc;

static void require(bool ok,const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    for (bool nominal : {true,false}) {
        RuntimeConfig cfg;
        cfg.mpc.type=MPCType::SH_MPCC_DRO_FALLBACK;
        cfg.mpc.nominal_resampling_baseline=nominal;
        cfg.mpc.horizon=4;
        cfg.mpc.sampling.set_manual_sample_count(8);
        cfg.mpc.ego.num_discs=1;cfg.mpc.ego.length=0;
        cfg.random_seed=2;
        ModeModel safe("safe",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
        ModeModel threat("threat",Eigen::Matrix4d::Zero(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
        MPCController ordinary(cfg),observed(cfg);
        observed.set_capture_attempt_diagnostics(true);
        for (auto* controller:{&ordinary,&observed}) {
            controller->initialize_obstacle(0,0,{{"safe",safe},{"threat",threat}});
            for (int i=0;i<100;++i) controller->update_mode_observation(0,0,i<90 ? "safe" : "threat",i);
        }
        // A second decision checks that diagnostics did not consume RNG state.
        for (int step=0;step<2;++step) {
            const auto a=ordinary.solve(EgoState(0,0,0,0),{{0,ObstacleState(5,0,0,0)}},Eigen::Vector2d(10,0));
            const auto b=observed.solve(EgoState(0,0,0,0),{{0,ObstacleState(5,0,0,0)}},Eigen::Vector2d(10,0));
            require(a.attempt_diagnostics.empty(),"diagnostics must default off");
            require(a.success==b.success && a.nominal_fallback_attempted==b.nominal_fallback_attempted,
                    "observing attempts changed controller outcome");
            require(a.control_inputs.size()==b.control_inputs.size(),"control lengths changed");
            for (size_t i=0;i<a.control_inputs.size();++i)
                require(a.control_inputs[i].a==b.control_inputs[i].a && a.control_inputs[i].omega==b.control_inputs[i].omega,
                        "observing attempts changed controls");
            require(b.attempt_diagnostics.size()==1+static_cast<size_t>(b.nominal_fallback_attempted),"missing attempt");
            require(b.attempt_diagnostics.back().success==b.success,"final status mismatch");
            for (const auto& attempt:b.attempt_diagnostics) {
                int count=0;
                for (const auto& [mode,n]:attempt.initial_mode_counts.at(0)) count+=n;
                require(count==8 && attempt.sampled_scenarios==8,"incorrect draw accounting");
                require(attempt.qp_calls>=0 && attempt.elapsed_seconds>=0,"invalid effort accounting");
                if (nominal) require(!attempt.dro_enabled,"nominal baseline performed DRO");
            }
            if (nominal && step==0)
                require(b.attempt_diagnostics.size()==2 && !b.attempt_diagnostics[0].success && b.success,
                        "fixture must retain evidence of failed first and successful second draw");
        }
    }
    std::cout << "PASS: diagnostics preserve controls, outcomes and successive RNG draws; both attempts retained\n";
}
