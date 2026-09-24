#include "mpc_controller.hpp"
#include "certification_tube.hpp"
#include <iostream>
#include <stdexcept>

using namespace dro_mpc;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
namespace dro_mpc {
class CertificationTubeTestAccess {
public:
    static void check_qp(MPCController& controller) {
        const EgoState start(0,0,.2,1);
        const std::vector<EgoInput> inputs(4, EgoInput(0,0));
        const auto nominal=controller.ego_dynamics_.rollout(start,inputs);
        controller.certification_reference_=nominal;
        for (auto& state : controller.certification_reference_) state.y += .1;
        const auto qp=controller.build_condensed_qp(nominal,inputs,{10,0},2,{});
        const int first=qp.C.rows()-12;
        for (int col=0;col<8;++col) {
            auto plus=inputs,minus=inputs;
            const double h=1e-6;
            if (col%2) { plus[col/2].omega+=h; minus[col/2].omega-=h; }
            else { plus[col/2].a+=h; minus[col/2].a-=h; }
            const auto xp=controller.ego_dynamics_.rollout(start,plus);
            const auto xm=controller.ego_dynamics_.rollout(start,minus);
            for (int k=1;k<=4;++k) for (int disc=0;disc<3;++disc) {
                const auto center=compute_ego_disc_positions(controller.certification_reference_[k],3,2)[disc];
                const auto p=compute_ego_disc_positions(xp[k],3,2)[disc];
                const auto m=compute_ego_disc_positions(xm[k],3,2)[disc];
                const double derivative=((p-center).squaredNorm()-(m-center).squaredNorm())/(2*h);
                const int row=first+(k-1)*3+disc;
                require(std::abs(qp.C(row,col)+derivative)<1e-8, "tube QP sign/condensed Jacobian mismatch");
                require(std::abs(qp.d(row)-(.01-.25*.25))<1e-14, "tube QP right hand side mismatch");
            }
        }
    }
    static void check_guard(MPCController& controller) {
        controller.certification_reference_ = std::vector<EgoState>(5, EgoState(0,0,0,0));
        for (bool fallback : {false, true}) {
            MPCResult result;
            result.success = true;
            result.used_fallback = fallback;
            result.used_nominal_fallback = fallback;
            result.ego_trajectory = controller.certification_reference_;
            result.ego_trajectory.back().theta = 1.0; // center fixed, end discs outside
            result.certificate_status = SafeHorizonCertificateStatus::CERTIFIED;
            result.certified_horizon = 4;
            controller.enforce_certification_tube(result);
            require(!result.success && result.certification_tube_rejected && result.certified_horizon == -1,
                "all output types must reject exact disc excursion and suppress certification");
        }
        MPCResult result;
        result.success = true;
        result.ego_trajectory = controller.certification_reference_;
        result.ego_trajectory.back().x = .25;
        controller.enforce_certification_tube(result);
        require(result.success, "closed tube boundary must be accepted");
        result.ego_trajectory.back().x = .25001;
        controller.enforce_certification_tube(result);
        require(!result.success, "cumulative displacement uses frozen reference");
    }
};
}
int main() {
    RuntimeConfig cfg;
    cfg.mpc.horizon = 4;
    cfg.mpc.sampling.set_manual_sample_count(8);
    cfg.mpc.certification_tube_radius = .25;
    cfg.mpc.ego.num_discs = 3;
    cfg.mpc.ego.length = 2;
    MPCController controller(cfg);
    CertificationTubeTestAccess::check_qp(controller);
    CertificationTubeTestAccess::check_guard(controller);
    controller.reset();
    auto first = controller.solve(EgoState(0,0,0,0), {}, {10,0});
    require(first.success && !first.certification_tube_active, "cold solve must use original budget without a tube");
    auto second = controller.solve(first.ego_trajectory[1], {}, {10,0});
    require(second.certification_tube_active, "accepted prior plan must activate tube");
    require(!second.success || second.certification_tube_max_displacement <= .25, "runtime acceptance must enforce tube");
    require(first.sampled_scenarios == second.sampled_scenarios, "tube experiment must preserve budget");
    RuntimeConfig retry_cfg = cfg;
    retry_cfg.mpc.type = MPCType::SH_MPCC_DRO_FALLBACK;
    retry_cfg.mpc.ego.num_discs = 1;
    retry_cfg.mpc.ego.length = 0;
    retry_cfg.dro.fixed_rho = 100;
    retry_cfg.random_seed = 77;
    MPCController retry(retry_cfg);
    const EgoState ego(0,0,0,0);
    require(retry.solve(ego, {}, {10,0}).success, "retry fixture cold start");
    ModeModel safe("safe", Eigen::Matrix4d::Identity(), Eigen::Vector4d::Zero(), Eigen::MatrixXd::Zero(4,2));
    ModeModel threat("threat", Eigen::Matrix4d::Zero(), Eigen::Vector4d::Zero(), Eigen::MatrixXd::Zero(4,2));
    retry.initialize_obstacle(0,0,{{"safe",safe},{"threat",threat}});
    for (int i=0;i<1000;++i) retry.update_mode_observation(0,0,"safe",i);
    const auto fallback = retry.solve(ego, {{0,ObstacleState(5,0,0,0)}}, {10,0});
    require(fallback.certification_tube_active && fallback.nominal_fallback_attempted && fallback.success,
        "actual nominal retry must retain active tube and produce an admissible plan");
    require(fallback.certification_tube_mode_bounds.at(0).at("safe") == 0 &&
            fallback.certification_tube_mode_bounds.at(0).at("threat") == 1,
        "pre-sampling held-mode union bounds");
    require(fallback.certification_tube_max_displacement <= .25, "nominal retry must remain in original tube");
    safe.body_lateral_displacement = .1;
    retry.initialize_obstacle(0,0,{{"safe",safe}});
    const auto unsupported = retry.solve(ego, {{0,ObstacleState(5,0,0,0)}}, {10,0});
    require(unsupported.certification_tube_mode_bounds.at(0).at("safe") == 1,
        "nonlinear mode must not receive affine Gaussian bound");
    auto broken = second.certification_tube_reference;
    broken.pop_back();
    require(std::isinf(certification_tube_displacement(second.certification_tube_reference, broken,3,2)),
        "incomplete horizon must fail closed");
    const Eigen::Matrix2d zero = Eigen::Matrix2d::Zero();
    require(tube_gaussian_probability({0,0},{1.25,0},zero,1,.25) == 0, "strict degenerate boundary");
    require(tube_gaussian_probability({0,0},{1.249,0},zero,1,.25) == 1, "inside degenerate boundary");
    const double bound = tube_gaussian_probability({0,0},{2,0},Eigen::Matrix2d::Identity(),1,.25);
    require(std::abs(bound - .2266273523768682) < 1e-14, "Gaussian tube tail must equal Phi(-.75)");
    // The projected tail at every sampled center must be below the uniform tail.
    for (int i=0;i<360;++i) {
        Eigen::Vector2d center(.25*std::cos(i),.25*std::sin(i));
        const double tail=.5*std::erfc(-(1-2+center.x())/std::sqrt(2.));
        require(tail <= bound, "tube bound must dominate candidate fixed-normal tails");
    }
    std::cout << "PASS tube: exact boundary, heading, incomplete horizon, all-output guard, cold/warm solves, unchanged S; Gaussian bound="
              << bound << " warm_success=" << second.success << " displacement=" << second.certification_tube_max_displacement << '\n';
}
