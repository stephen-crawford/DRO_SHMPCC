#include "mpc_controller.hpp"
#include "certification_tube.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace dro_mpc;

// Paired closed-loop pilot: identical seeds, mode histories, measured obstacle
// sequence, horizon and scenario budget. These synthetic histories are not CP data.
int main(int argc, char** argv) {
    if (argc != 5) throw std::invalid_argument("usage: certification_tube_experiment OUTPUT.csv SCENARIOS SEEDS CYCLES");
    const int scenarios = std::stoi(argv[2]);
    const int seeds = std::stoi(argv[3]), cycles = std::stoi(argv[4]);
    if (scenarios < 1 || seeds < 1 || cycles < 1) throw std::invalid_argument("counts must be positive");
    std::ofstream out(argv[1]);
    if (!out) throw std::runtime_error("cannot open output");
    std::ofstream geometry(std::filesystem::path(argv[1]).parent_path() / "disc_geometry.csv");
    if (!geometry) throw std::runtime_error("cannot open geometry output");
    geometry << std::setprecision(17)
        << "seed,repeat,arm,radius,cycle,stage,disc,x,y,reference_x,reference_y,distance\n";
    std::ofstream attempts(std::filesystem::path(argv[1]).parent_path() / "attempts.csv");
    std::ofstream predictions(std::filesystem::path(argv[1]).parent_path() / "predictions.csv");
    if (!attempts || !predictions) throw std::runtime_error("cannot open decomposition output");
    attempts << std::setprecision(17)
        << "seed,repeat,arm,radius,cycle,attempt,dro_enabled,success,mode,p,q,rho,risk_score,n_bar,removal_budget,total_support_cap,observed_final_support\n";
    predictions << std::setprecision(17)
        << "mode,stage,mean_x,mean_y,cov_xx,cov_xy,cov_yx,cov_yy,collision_radius\n";
    out << std::setprecision(17);
    out << "seed,repeat,arm,radius,cycle,success,active,rejected,max_displacement,nominal_retry,used_fallback,sampled_scenarios,certificate_status,terminal_x,solve_seconds,mode,b,p,q\n";
    for (int seed=77;seed<77+seeds;++seed) for (int repeat=0;repeat<2;++repeat)
    for (bool wdro : {false,true}) for (double radius : {0.,.1,.2,.3,.4,.5,.75}) {
        RuntimeConfig cfg;
        cfg.random_seed = seed;
        cfg.mpc.type = MPCType::SH_MPCC_DRO_FALLBACK;
        cfg.mpc.nominal_resampling_baseline = !wdro;
        cfg.mpc.horizon = 8;
        cfg.mpc.sampling.set_manual_sample_count(scenarios);
        cfg.mpc.certification_tube_radius = radius;
        cfg.mpc.ego.length = 2;
        cfg.mpc.ego.num_discs = 3;
        MPCController controller(cfg);
        controller.set_capture_attempt_diagnostics(true);
        controller.set_reference_path(ReferencePath::create_straight({0,0},{30,0}));
        Eigen::MatrixXd noise = Eigen::MatrixXd::Zero(4,2);
        noise(0,0) = .02; noise(1,1) = .02;
        ModeModel safe("safe",Eigen::Matrix4d::Identity(),Eigen::Vector4d::Zero(),noise);
        ModeModel cut("cut",Eigen::Matrix4d::Identity(),Eigen::Vector4d(0,-.2,0,0),noise);
        if (seed==77 && repeat==0 && !wdro && radius==0) {
            for (const auto& model : {safe,cut}) {
                ObstacleState mean(2.8,2.5,0,0);
                Eigen::Matrix4d cov=Eigen::Matrix4d::Zero();
                for (int k=1;k<=cfg.mpc.horizon;++k) {
                    mean=model.propagate(mean); model.propagate_covariance(cov);
                    predictions << model.mode_id << ',' << k << ',' << mean.x << ',' << mean.y << ','
                        << cov(0,0) << ',' << cov(0,1) << ',' << cov(1,0) << ',' << cov(1,1) << ','
                        << cfg.mpc.ego.radius+cfg.obstacle_radius+cfg.mpc.constraints.safety_margin << '\n';
                }
            }
        }
        controller.initialize_obstacle(0,0,{{"safe",safe},{"cut",cut}});
        for (int i=0;i<100;++i) controller.update_mode_observation(0,0,i<95 ? "safe":"cut",i);
        EgoState ego(0,0,0,1);
        for (int cycle=0;cycle<cycles;++cycle) {
            const auto r=controller.solve(ego,{{0,ObstacleState(2.8,2.5,0,0)}},{30,0});
            for (std::size_t i=0;i<r.attempt_diagnostics.size();++i) {
                const auto& a=r.attempt_diagnostics[i];
                for (const auto& [mode,p] : a.nominal_weights.at(0)) {
                    attempts << seed << ',' << repeat << ',' << (wdro?"wdro":"nominal_resampling") << ','
                        << radius << ',' << cycle << ',' << i << ',' << a.dro_enabled << ',' << a.success << ','
                        << mode << ',' << p << ',' << a.sampling_weights.at(0).at(mode) << ',';
                    if (a.radii.count(0)) attempts << a.radii.at(0);
                    attempts << ',';
                    if (a.risk_scores.count(0)) attempts << a.risk_scores.at(0).at(mode);
                    attempts << ',' << cfg.mpc.constraints.support_cap_n_bar << ','
                        << cfg.mpc.constraints.scenario_removal_budget << ',' << cfg.support_limit()
                        << ',' << r.support_size << '\n';
                }
            }
            for (std::size_t k=1;k<r.ego_trajectory.size();++k) {
                const auto centers=compute_ego_disc_positions(r.ego_trajectory[k],3,2);
                for (int j=0;j<3;++j) {
                    geometry << seed << ',' << repeat << ',' << (wdro?"wdro":"nominal_resampling") << ','
                        << radius << ',' << cycle << ',' << k << ',' << j << ',' << centers[j].x() << ',' << centers[j].y() << ',';
                    if (r.certification_tube_active) {
                        const auto reference=compute_ego_disc_positions(r.certification_tube_reference[k],3,2);
                        geometry << reference[j].x() << ',' << reference[j].y() << ',' << (centers[j]-reference[j]).norm();
                    } else geometry << ",,";
                    geometry << '\n';
                }
            }
            for (const auto& mode : {std::string("safe"),std::string("cut")}) {
                out << seed << ',' << repeat << ',' << (wdro?"wdro":"nominal_resampling") << ','
                    << radius << ',' << cycle << ',' << r.success << ',' << r.certification_tube_active << ','
                    << r.certification_tube_rejected << ',' << r.certification_tube_max_displacement << ','
                    << r.nominal_fallback_attempted << ',' << r.used_fallback << ',' << r.sampled_scenarios << ','
                    << safe_horizon_certificate_status_name(r.certificate_status) << ','
                    << (r.ego_trajectory.empty()?ego.x:r.ego_trajectory.back().x) << ',' << r.solve_time << ',' << mode << ',';
                if (r.certification_tube_active) out << r.certification_tube_mode_bounds.at(0).at(mode);
                out << ',';
                if (!r.attempt_diagnostics.empty()) {
                    const auto& a=r.attempt_diagnostics.back();
                    if (a.nominal_weights.count(0)) out << a.nominal_weights.at(0).at(mode);
                    out << ',';
                    if (a.sampling_weights.count(0)) out << a.sampling_weights.at(0).at(mode);
                } else out << ',';
                out << '\n';
            }
            if (!r.success) break; // Never execute a rejected plan.
            ego = r.ego_trajectory.at(1);
        }
    }
}
