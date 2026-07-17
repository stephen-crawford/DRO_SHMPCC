// velocity_bound_probe: closed-loop check that the hard velocity bounds hold.
// Drives the controller with a reference velocity well above max_velocity so the
// upper bound is binding, on an obstacle-free straight path, and asserts the
// realized speed never leaves [min_velocity, max_velocity].
#include "mpc_controller.hpp"
#include "dynamics.hpp"
#include "reference_path.hpp"
#include <cstdio>
#include <cmath>

using namespace scenario_mpc;

int main() {
    ScenarioMPCConfig cfg;                 // defaults: max_velocity=4.0, min_velocity=0.0
    const double tol = 1e-6;

    AdaptiveScenarioMPC controller(cfg);
    ReferencePath path = ReferencePath::create_straight({0.0, 0.0}, {60.0, 0.0}, 200);
    controller.set_reference_path(path);

    EgoDynamics dyn(cfg.dt);
    EgoState ego(0.0, 0.0, 0.0, 0.0);      // start at rest
    std::map<int, ObstacleState> obstacles;  // no obstacles: bound must still hold
    Eigen::Vector2d goal(60.0, 0.0);

    const double v_target = 10.0;          // >> max_velocity, so the cap should bind
    double v_max_seen = -1e9, v_min_seen = 1e9;
    int steps = 120, fails = 0;

    for (int t = 0; t < steps; ++t) {
        double s = path.find_closest_point(ego.position());
        MPCResult r = controller.solve(ego, obstacles, goal, v_target, s, path.total_length());
        if (r.success && r.first_input().has_value()) {
            ego = dyn.propagate(ego, r.first_input().value());
        }
        v_max_seen = std::max(v_max_seen, ego.v);
        v_min_seen = std::min(v_min_seen, ego.v);
        if (ego.v > cfg.max_velocity + 1e-3 || ego.v < cfg.min_velocity - 1e-3) {
            std::printf("  VIOLATION at step %d: v=%.6f (bounds [%.2f, %.2f])\n",
                        t, ego.v, cfg.min_velocity, cfg.max_velocity);
            ++fails;
        }
    }

    std::printf("velocity_bound_probe: v in [%.4f, %.4f], bounds [%.2f, %.2f], target=%.1f\n",
                v_min_seen, v_max_seen, cfg.min_velocity, cfg.max_velocity, v_target);
    // The cap must actually engage (target is 2.5x the cap) and never be exceeded.
    bool cap_engaged = v_max_seen > cfg.max_velocity - 0.5;
    bool respected   = (fails == 0) && (v_max_seen <= cfg.max_velocity + 1e-3)
                                    && (v_min_seen >= cfg.min_velocity - 1e-3);
    (void)tol;
    if (respected && cap_engaged) {
        std::printf("PASS: hard velocity bounds respected and upper cap engaged.\n");
        return 0;
    }
    std::printf("FAIL: respected=%d cap_engaged=%d fails=%d\n", respected, cap_engaged, fails);
    return 1;
}
