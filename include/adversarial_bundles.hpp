#pragma once
#include "config.hpp"
#include "mode_weights.hpp"
#include <random>

namespace dro_mpc {
struct BundleDesign {
    std::vector<std::string> modes;
    std::vector<double> lower, upper;
    std::vector<int> multiplicities;
    double amplification = 0;
    double threshold = 0;
    int required_bundles = 0;
    int raw_per_bundle = 0;
};
BundleDesign design_adversarial_bundles(const ModeHistory& history,
    const std::map<std::string,double>& q, const std::map<std::string,double>& scores,
    const RuntimeConfig& config);
std::vector<Scenario> sample_adversarial_bundles(int obstacle_id,
    const ObstacleState& state, const ModeHistory& history,
    const BundleDesign& design, int horizon, int bundles, std::mt19937& rng);
int scenario_group_count(const std::vector<Scenario>& scenarios);
// Only exactly equal normals and disc maps within the SAME bundle are compared.
// No angular tolerance, score ranking, reachable-ball assumption or facet cap.
std::vector<CollisionConstraint> prune_bundle_constraints_exact(
    const std::vector<CollisionConstraint>& constraints);
}
