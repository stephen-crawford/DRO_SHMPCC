/**
 * @file test_constraint_hypotheses.cpp
 * @brief Test WDRO vs TopRisk under different constraint regimes.
 *
 * Previous tests showed no WDRO advantage with default half-space constraints.
 * Key insight: half-space constraints project everything onto a single normal
 * direction, losing the distributional shape that WDRO exploits.
 *
 * Hypotheses:
 *   H6: Contouring constraints — narrow road forces ego onto the path,
 *       reducing evasion options and making mode selection matter more.
 *   H7: High adversarial sigma — injected scenarios pushed further into
 *       the covariance tail (sigma_scale=3,5). WDRO's transport-aware
 *       selection should pick modes whose tails differ most from TopRisk's.
 *   H8: Multi-disc model — D=3 discs gives 3x the constraints per scenario.
 *       More constraint surfaces means the full distribution shape matters
 *       more than just the closest-point normal.
 *   H9: Combined (narrow road + multi-disc + high sigma) — stack all effects.
 *   H10: Tight safety margin — large safety_margin makes constraints bind
 *        earlier, amplifying any selection difference.
 *
 * Output: figures/constraint_hyp/
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

static const std::string DIR = "figures/constraint_hyp/";
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

// ============================================================================
// Hypothesis configurations
// ============================================================================

struct HypConfig {
    std::string hyp_name;
    std::string cond_name;
    // MPC config overrides
    bool contouring;
    double road_width;
    int num_discs;
    double vehicle_length;
    double adv_sigma;
    double safety_margin;  // extra safety margin on top of default
};

static std::vector<HypConfig> make_configs() {
    std::vector<HypConfig> cfgs;

    // H6: Contouring constraints with narrow road
    for (double rw : {2.5, 3.0, 4.0}) {
        cfgs.push_back({"H6", "road=" + std::to_string((int)(rw*10)/10) + "." + std::to_string((int)(rw*10)%10) + "m",
                         true, rw, 1, 1.5, 1.5, 0.2});
    }

    // H7: High adversarial sigma
    for (double sig : {1.5, 3.0, 5.0, 8.0}) {
        cfgs.push_back({"H7", "sigma=" + std::to_string((int)sig) + (sig == 1.5 ? ".5" : ""),
                         false, 7.0, 1, 1.5, sig, 0.2});
    }

    // H8: Multi-disc model
    for (int nd : {1, 3, 5}) {
        cfgs.push_back({"H8", "discs=" + std::to_string(nd),
                         false, 7.0, nd, 4.0, 1.5, 0.2});
    }

    // H9: Combined (narrow road + 3 discs + sigma=3)
    cfgs.push_back({"H9", "combined-tight",
                     true, 2.5, 3, 4.0, 3.0, 0.3});
    cfgs.push_back({"H9", "combined-medium",
                     true, 3.5, 3, 4.0, 3.0, 0.2});

    // H10: Tight safety margin sweep
    for (double sm : {0.1, 0.3, 0.5, 0.8}) {
        cfgs.push_back({"H10", "margin=" + std::to_string((int)(sm*10)/10) + "." + std::to_string((int)(sm*10)%10),
                         false, 7.0, 1, 1.5, 1.5, sm});
    }

    return cfgs;
}

int main() {
    fs::create_directories(DIR);

    std::cout << "========================================\n";
    std::cout << "Constraint Hypotheses Test\n";
    std::cout << "  H6:  Contouring constraints (narrow road)\n";
    std::cout << "  H7:  High adversarial sigma\n";
    std::cout << "  H8:  Multi-disc model\n";
    std::cout << "  H9:  Combined constraints\n";
    std::cout << "  H10: Safety margin sweep\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();
    const int N = 1000;

    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);
    auto configs = make_configs();

    std::string filepath = DIR + "constraint_hyp.csv";
    std::ofstream ofs(filepath);
    ofs << "hypothesis,condition,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& hc : configs) {
        std::cout << "\n--- " << hc.hyp_name << ": " << hc.cond_name << " ---\n";

        for (const auto& md : METHODS) {
            std::cout << "  " << md.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = md.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(hc.hyp_name + hc.cond_name) / 1000 +
                    static_cast<int>(md.mode) * 100000 + md.K * 10000 + i + 600000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg;
                cfg.horizon = HORIZON;
                cfg.num_scenarios = DEFAULT_BASE_SCENARIOS;
                cfg.switch_prob = 0.2;
                cfg.rollout_steps = ROLLOUT_STEPS;
                cfg.obs_modes = BASE_MODES;
                cfg.rare_mode = "lane_change_left";
                cfg.rare_switch_prob = 0.1;
                cfg.num_discs = hc.num_discs;
                cfg.vehicle_length = hc.vehicle_length;
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
                cfg.enable_contouring_constraints = hc.contouring;
                cfg.road_width = hc.road_width;
                cfg.adversarial_sigma_scale = hc.adv_sigma;
                cfg.custom_ref_path = path;
                cfg.custom_initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);
                cfg.initial_obstacle_states = {obs_oncoming(path, 0.55, rng)};

                met.add(run_experiment_rollout(cfg, seed));
            }

            auto [lo, hi] = met.cci();
            ofs << hc.hyp_name << "," << hc.cond_name << "," << md.name << ","
                << std::fixed << std::setprecision(6) << met.cr() << "," << lo << "," << hi << ","
                << std::setprecision(4) << met.mc() << "," << met.p5() << ","
                << met.mp() << "," << std::setprecision(3) << met.ms() << "," << N << "\n";
            ofs.flush();

            std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                      << "% [" << (lo*100) << "," << (hi*100) << "]"
                      << " clr=" << std::setprecision(2) << met.mc()
                      << " t=" << std::setprecision(1) << met.ms()
                      << "ms (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
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
