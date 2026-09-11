#include "mpc_controller.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace dro_mpc;
int main() {
    RuntimeConfig cfg;
    MPCController controller(cfg);
    EgoDynamics dynamics(cfg.mpc.ego.dynamics, cfg.mpc.dt);
    for (double speed : {0.0, 0.05, 0.1, 8.4112725111552965e-5,
                         0.15992795966233597, 1.8527553298944672, 4.0}) {
        EgoState current(0,0,0,speed);
        const auto fallback = controller.generate_safe_fallback(current);
        for (size_t k=0;k<fallback.control_inputs.size();++k) {
            const auto& input=fallback.control_inputs[k];
            const auto next=dynamics.propagate(current,input);
            if (input.a < -1.0 || input.a > 0.0 || input.omega != 0.0 ||
                next.v < 0.0 || next.v > current.v ||
                (next.to_array()-fallback.ego_trajectory[k+1].to_array()).norm()!=0.0)
                throw std::runtime_error("braking reverses, accelerates, or mismatches dynamics");
            if (speed==0.0 && (input.a!=0.0 || next.x!=0.0 || next.v!=0.0))
                throw std::runtime_error("stopped fallback did not hold");
            current=next;
        }
        const double expected=std::max(0.0,speed-cfg.mpc.horizon*cfg.mpc.dt);
        if (std::abs(current.v-expected)>1e-12)
            throw std::runtime_error("unexpected final braking speed");
        std::cout << "v0=" << speed << " terminal_v=" << current.v << '\n';
    }
}
