#include "collision_constraints.hpp"
#include <iostream>
#include <stdexcept>
using namespace dro_mpc;
int main() {
    std::vector<EgoState> original{{0,0,0,1}, {1.8,0,0,1}, {2.0,0,0,1}};
    Scenario scenario;
    ObstacleTrajectory obstacle;
    obstacle.steps.resize(3);
    for (auto& step : obstacle.steps) step.mean = Eigen::Vector2d(2,0);
    scenario.trajectories[0] = obstacle;
    std::vector<Scenario> scenarios(5, scenario);
    auto anchors = original;
    prepare_safe_horizon_anchors(anchors, scenarios, 0.95, 1, 0.0);
    if ((anchors[0].to_array() - original[0].to_array()).norm() != 0)
        throw std::runtime_error("initial state moved");
    auto constraints = compute_linearized_constraints(anchors, scenarios, .5,.35,.1,1,0);
    for (const auto& con : constraints) {
        if (con.k == 0) continue;
        if (!anchors[con.k].to_array().allFinite() ||
            con.evaluate(anchors[con.k].position()) < -1e-9 ||
            anchors[con.k].y <= 0 || con.a.y() <= 0)
            throw std::runtime_error("stationary obstacle did not produce lateral feasible anchors");
    }
    auto empty = original;
    prepare_safe_horizon_anchors(empty, {}, .95, 1, 0);
    for (size_t k=0;k<empty.size();++k)
        if ((empty[k].to_array()-original[k].to_array()).norm()!=0)
            throw std::runtime_error("empty scene changed");
    std::cout << "PASS: stationary obstacle, exact center, lateral normals, initial state, empty scene\n";
}
