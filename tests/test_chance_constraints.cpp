/**
 * @file test_chance_constraints.cpp
 * @brief Compare WDRO vs TopRisk under Gaussian chance constraints.
 *
 * With chance constraints, the constraint bound includes z_alpha * sqrt(a^T Sigma a),
 * making it sensitive to directional variance.  Different modes have different
 * covariance matrices, so the constraint tightening varies by mode — which should
 * make WDRO's transport-aware mode selection matter.
 *
 * Tests:
 *   - Half-space (baseline) vs chance constraints at z=1.0, 1.645, 2.0, 2.5
 *   - S-curve and Tight-S paths
 *   - Switch prob 0.2 and 0.4
 *
 * Output: figures/chance_constraints/
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <filesystem>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"
#include "reference_path.hpp"
#include "dynamics.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string DIR = "figures/chance_constraints/";
static constexpr int    HORIZON         = DEFAULT_HORIZON;
static constexpr double DT              = DEFAULT_DT;
static constexpr int    ROLLOUT_STEPS   = DEFAULT_ROLLOUT_STEPS;

static const std::vector<std::string> BASE_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};

struct Metrics {
    std::string method;
    std::vector<bool> collisions;
    std::vector<double> min_clearances, total_progress, solve_times_ms;
    int n() const { return (int)collisions.size(); }
    int cc() const { int c=0; for (bool b : collisions) if (b) c++; return c; }
    double cr() const { return n()>0 ? double(cc())/n() : 0; }
    std::pair<double,double> cci() const { return wilson_ci(cc(), n()); }
    double mc() const { return min_clearances.empty() ? 0 : std::accumulate(min_clearances.begin(), min_clearances.end(), 0.0)/min_clearances.size(); }
    double p5() const {
        if (min_clearances.empty()) return 0;
        auto v=min_clearances; std::sort(v.begin(),v.end());
        return v[std::max(0,(int)(0.05*(v.size()-1)))];
    }
    double mp() const { return total_progress.empty() ? 0 : std::accumulate(total_progress.begin(), total_progress.end(), 0.0)/total_progress.size(); }
    double ms() const { return solve_times_ms.empty() ? 0 : std::accumulate(solve_times_ms.begin(), solve_times_ms.end(), 0.0)/solve_times_ms.size(); }
    void add(const RolloutRecord& rec) {
        collisions.push_back(rec.collision);
        min_clearances.push_back(rec.min_clearance);
        total_progress.push_back(rec.total_progress);
        solve_times_ms.push_back(rec.avg_solve_ms);
    }
};

static double elapsed_sec(std::chrono::steady_clock::time_point s) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
}

static ObstacleState obs_oncoming(const ReferencePath& path, double frac,
                                   std::mt19937& rng, double speed = 0.8) {
    double s = frac * path.total_length();
    PathPoint pp = path.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
    std::uniform_real_distribution<double> lat(-0.15, 0.15), spd(-0.05, 0.05);
    Eigen::Vector2d pos = pp.position + lat(rng) * n;
    double v = speed + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -v * t.x(), -v * t.y());
}

struct MethodDef {
    std::string name;
    InjectionMode mode;
    int K;
};

static const std::vector<MethodDef> METHODS = {
    {"Base",              InjectionMode::NONE,               0},
    {"WDRO-sampling",     InjectionMode::QSTAR_SAMPLE,       0},
    {"WDRO-inject-K1",    InjectionMode::DRO,                1},
    {"WDRO-inject-K2",    InjectionMode::DRO,                2},
    {"TopRisk-K1",        InjectionMode::TOP_RISK_INJECT,    1},
    {"TopRisk-K2",        InjectionMode::TOP_RISK_INJECT,    2},
    {"DiverseRisk-K1",    InjectionMode::DIVERSE_RISK_INJECT,1},
};

int main() {
    fs::create_directories(DIR);

    std::cout << "========================================\n";
    std::cout << "Chance Constraint Comparison\n";
    std::cout << "  Constraint types: half-space, chance(z=1.0,1.645,2.0,2.5)\n";
    std::cout << "  Paths: S-curve, Tight-S\n";
    std::cout << "  Switch probs: 0.2, 0.4\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();
    const int N = 1000;

    struct ConstraintConfig {
        std::string label;
        bool chance;
        double z_alpha;
    };

    std::vector<ConstraintConfig> constraint_configs = {
        {"halfspace",    false, 0.0},
        {"chance-z1.0",  true,  1.0},
        {"chance-z1.645", true, 1.6449},
        {"chance-z2.0",  true,  2.0},
        {"chance-z2.5",  true,  2.5},
    };

    struct PathConfig {
        std::string name;
        ReferencePath path;
        double switch_prob;
    };

    auto scurve = ReferencePath::create_s_curve(25.0, 3.0, 200);
    auto tight  = ReferencePath::create_s_curve(20.0, 5.0, 200);

    std::vector<PathConfig> path_configs = {
        {"S-curve-sp0.2",  scurve, 0.2},
        {"S-curve-sp0.4",  scurve, 0.4},
        {"Tight-S-sp0.2",  tight,  0.2},
        {"Tight-S-sp0.4",  tight,  0.4},
    };

    std::string filepath = DIR + "chance_constraints.csv";
    std::ofstream ofs(filepath);
    ofs << "constraint_type,path_config,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& cc : constraint_configs) {
        for (const auto& pc : path_configs) {
            std::cout << "\n--- " << cc.label << " / " << pc.name << " ---\n";

            for (const auto& md : METHODS) {
                std::cout << "  " << md.name << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();
                Metrics met; met.method = md.name;

                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        std::hash<std::string>{}(cc.label + pc.name) / 1000 +
                        static_cast<int>(md.mode) * 100000 + md.K * 10000 + i + 700000000);
                    std::mt19937 rng(seed);

                    ExperimentConfig cfg;
                    cfg.horizon = HORIZON;
                    cfg.num_scenarios = DEFAULT_BASE_SCENARIOS;
                    cfg.switch_prob = pc.switch_prob;
                    cfg.rollout_steps = ROLLOUT_STEPS;
                    cfg.obs_modes = BASE_MODES;
                    cfg.rare_mode = "lane_change_left";
                    cfg.rare_switch_prob = 0.1;
                    cfg.num_discs = 1;
                    cfg.vehicle_length = 1.5;
                    cfg.safe_horizon_enabled = true;
                    cfg.safe_horizon_min = 3;
                    cfg.path_completion_termination = true;
                    cfg.path_completion_fraction = 0.95;
                    cfg.weight_type = WeightType::FREQUENCY;
                    cfg.enable_dro = (md.mode != InjectionMode::NONE);
                    cfg.injection_mode = md.mode;
                    cfg.dro_injection_count = md.K;
                    cfg.method_name = md.name;
                    cfg.ablation = cfg.enable_dro ? AblationVariant::DRO_FULL : AblationVariant::NO_INJECTION;
                    cfg.num_obstacles = 1;
                    cfg.obstacles_per_class = 1;
                    cfg.use_chance_constraints = cc.chance;
                    cfg.chance_z_alpha = cc.z_alpha;
                    cfg.custom_ref_path = pc.path;
                    cfg.custom_initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
                    cfg.initial_obstacle_states = {obs_oncoming(pc.path, 0.55, rng)};

                    met.add(run_experiment_rollout(cfg, seed));
                }

                auto [lo, hi] = met.cci();
                ofs << cc.label << "," << pc.name << "," << md.name << ","
                    << std::fixed << std::setprecision(6) << met.cr() << "," << lo << "," << hi << ","
                    << std::setprecision(4) << met.mc() << "," << met.p5() << ","
                    << met.mp() << "," << std::setprecision(3) << met.ms() << "," << N << "\n";
                ofs.flush();

                std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                          << "% [" << (lo*100) << "," << (hi*100) << "]"
                          << " clr=" << std::setprecision(2) << met.mc()
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }

    ofs.close();
    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Written: " << filepath << "\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0)
              << total << "s (" << std::setprecision(1) << total/60.0 << " min)\n";
    std::cout << "========================================\n";
    return 0;
}
