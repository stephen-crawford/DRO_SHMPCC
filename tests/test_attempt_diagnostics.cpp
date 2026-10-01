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
                require(attempt.trajectory_generation_seconds>0, "missing sampler wall time");
                if (nominal) require(!attempt.dro_enabled,"nominal baseline performed DRO");
            }
            if (nominal && step==0)
                require(b.attempt_diagnostics.size()==2 && !b.attempt_diagnostics[0].success && b.success,
                        "fixture must retain evidence of failed first and successful second draw");
        }
    }
    {
        RuntimeConfig cfg;
        cfg.dro.enabled=true;
        cfg.mpc.safe_horizon_enabled=false;
        cfg.mpc.wdro_stratified_sampling=true;
        cfg.mpc.ego.num_discs=1;cfg.mpc.ego.length=0;
        cfg.mpc.sampling.set_manual_sample_count(8);
        cfg.mpc.horizon=4;
        cfg.random_seed=77;
        cfg.validate();
        for (int flag=0;flag<5;++flag) {
            auto invalid=cfg;
            if (flag==0) invalid.mpc.safe_horizon_enabled=true;
            if (flag==1) invalid.mpc.sampling.automatically_compute_sample_size=true;
            if (flag==2) invalid.mpc.sampling.markov_jump_system=true;
            if (flag==3) invalid.dro.enabled=false;
            if (flag==4) invalid.mpc.type=MPCType::SH_MPCC_DRO_FALLBACK;
            bool rejected=false;
            try { invalid.validate(); } catch (const std::invalid_argument&) { rejected=true; }
            require(rejected,"stratified mode accepted an incompatible guarantee/configuration");
        }
        MPCController controller(cfg);
        controller.set_capture_attempt_diagnostics(true);
        const ModeModel safe("safe",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),Eigen::MatrixXd::Zero(4,2));
        const ModeModel threat("threat",Eigen::Matrix4d::Identity(),Eigen::Vector4d(-.2,0,0,0),Eigen::MatrixXd::Zero(4,2));
        controller.initialize_obstacle(0,0,{{"safe",safe},{"threat",threat}});
        for (int i=0;i<100;++i) controller.update_mode_observation(0,0,i<99 ? "safe" : "threat",i);
        const auto result=controller.solve(EgoState(0,0,0,0),{{0,ObstacleState(5,0,0,0)}},Eigen::Vector2d(10,0));
        require(!result.sample_count_sufficient,"stratified batch must not claim sufficient certified samples");
        require(result.attempt_diagnostics.size()==1,"stratified raw arm must not retry");
        const auto& counts=result.attempt_diagnostics.front().initial_mode_counts.at(0);
        require(counts.at("safe")>=1 && counts.at("threat")>=1 && counts.at("safe")+counts.at("threat")==8,
                "live stratified controller did not retain all modes at the requested budget");
        std::cout << "PASS: live stratified counts safe=" << counts.at("safe") << " threat=" << counts.at("threat")
                  << "; certificate sample sufficiency=false\n";
    }
    std::cout << "PASS: diagnostics preserve controls, outcomes and successive RNG draws; both attempts retained\n";
}
