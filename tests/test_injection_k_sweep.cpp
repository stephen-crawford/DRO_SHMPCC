/**
 * @file test_injection_k_sweep.cpp
 * @brief Sweep injection count K from 1..6 for WDRO-inject, TopRisk, DiverseRisk.
 *
 * The hypothesis: at higher K, WDRO's transport-aware selection should
 * outperform TopRisk's greedy risk-ranked selection because TopRisk
 * will inject redundant (similar) modes while WDRO spreads mass across
 * geometrically diverse threats.
 *
 * Conditions tested:
 *   - S-curve oncoming (standard)
 *   - Tight-S oncoming (harder path)
 *   - Multi-obstacle (2 obs, S-curve)
 *   - High switching (sp=0.4, S-curve)
 *
 * Output: figures/k_sweep/
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

static const std::string DIR = "figures/k_sweep/";

static constexpr int    HORIZON         = DEFAULT_HORIZON;
static constexpr double DT              = DEFAULT_DT;
static constexpr int    NUM_SCENARIOS   = DEFAULT_BASE_SCENARIOS;
static constexpr int    ROLLOUT_STEPS   = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;

static const std::vector<std::string> BASE_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Metrics
// ============================================================================

struct Metrics {
    std::string method;
    std::vector<bool> collisions;
    std::vector<int> missed_mode_steps, total_mode_checks;
    std::vector<int> rare_mode_active, rare_mode_missed;
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
        missed_mode_steps.push_back(rec.missed_mode_steps);
        total_mode_checks.push_back(rec.total_mode_checks);
        rare_mode_active.push_back(rec.rare_mode_active);
        rare_mode_missed.push_back(rec.rare_mode_missed);
        min_clearances.push_back(rec.min_clearance);
        total_progress.push_back(rec.total_progress);
        solve_times_ms.push_back(rec.avg_solve_ms);
    }
};

static double elapsed_sec(std::chrono::steady_clock::time_point s) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
}

// ============================================================================
// Helpers
// ============================================================================

static ExperimentConfig make_base_config(double switch_prob = 0.2,
                                          double rare_prob = 0.1,
                                          int num_obs = 1) {
    ExperimentConfig cfg;
    cfg.horizon = HORIZON;
    cfg.num_scenarios = NUM_SCENARIOS;
    cfg.switch_prob = switch_prob;
    cfg.rollout_steps = ROLLOUT_STEPS;
    cfg.obs_modes = BASE_MODES;
    cfg.rare_mode = RARE_MODE;
    cfg.rare_switch_prob = rare_prob;
    cfg.num_discs = NUM_DISCS;
    cfg.vehicle_length = VEHICLE_LENGTH;
    cfg.safe_horizon_enabled = true;
    cfg.safe_horizon_min = SAFE_HORIZON_MIN;
    cfg.path_completion_termination = true;
    cfg.path_completion_fraction = 0.95;
    cfg.weight_type = WeightType::FREQUENCY;
    cfg.enable_dro = false;
    cfg.injection_mode = InjectionMode::NONE;
    cfg.ablation = AblationVariant::NO_INJECTION;
    cfg.num_obstacles = num_obs;
    cfg.obstacles_per_class = 1;
    return cfg;
}

static ObstacleState obs_oncoming(const ReferencePath& path, double frac,
                                   std::mt19937& rng, double speed = 0.8) {
    double s = frac * path.total_length();
    PathPoint pp = path.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
    std::uniform_real_distribution<double> lat(-0.3, 0.3), spd(-0.1, 0.1);
    Eigen::Vector2d pos = pp.position + lat(rng) * n;
    double v = speed + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -v * t.x(), -v * t.y());
}

// ============================================================================
// Injection methods to sweep
// ============================================================================

struct InjMethod {
    std::string label;        // e.g. "WDRO-inject"
    InjectionMode mode;
};

static const std::vector<InjMethod> INJ_METHODS = {
    {"WDRO-inject",     InjectionMode::DRO},
    {"TopRisk",         InjectionMode::TOP_RISK_INJECT},
    {"DiverseRisk",     InjectionMode::DIVERSE_RISK_INJECT},
};

// ============================================================================
// Conditions to test
// ============================================================================

struct Condition {
    std::string name;
    ReferencePath path;
    EgoState ego;
    double switch_prob;
    int num_obs;
    std::vector<double> obs_fracs;
    double obs_speed;
};

static std::vector<Condition> make_conditions() {
    auto scurve = ReferencePath::create_s_curve(25.0, 3.0, 200);
    auto tight  = ReferencePath::create_s_curve(20.0, 5.0, 200);

    return {
        {"S-curve",      scurve, EgoState(0,0,0,1.5), 0.2, 1, {0.55}, 0.8},
        {"Tight-S",      tight,  EgoState(0,0,0,1.5), 0.2, 1, {0.55}, 0.8},
        {"2obs-S-curve", scurve, EgoState(0,0,0,1.5), 0.2, 2, {0.40, 0.60}, 0.8},
        {"HighSwitch",   scurve, EgoState(0,0,0,1.5), 0.4, 1, {0.55}, 0.8},
    };
}

// ============================================================================
// Main sweep
// ============================================================================

int main() {
    fs::create_directories(DIR);

    std::cout << "========================================\n";
    std::cout << "Injection K-Sweep\n";
    std::cout << "  Methods: WDRO-inject, TopRisk, DiverseRisk\n";
    std::cout << "  K values: 1..6\n";
    std::cout << "  + Base and WDRO-sampling baselines\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    const std::vector<int> K_VALUES = {1, 2, 3, 4, 5, 6};

    auto conditions = make_conditions();

    std::string filepath = DIR + "k_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "condition,method,K,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& cond : conditions) {
        std::cout << "\n--- Condition: " << cond.name << " ---\n";

        // Base (no injection)
        {
            std::cout << "  Base: ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = "Base";

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(cond.name) / 1000 + i + 300000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg = make_base_config(cond.switch_prob, 0.1, cond.num_obs);
                cfg.custom_ref_path = cond.path;
                cfg.custom_initial_ego = cond.ego;

                std::vector<ObstacleState> obs;
                for (double f : cond.obs_fracs)
                    obs.push_back(obs_oncoming(cond.path, f, rng, cond.obs_speed));
                cfg.initial_obstacle_states = obs;

                met.add(run_experiment_rollout(cfg, seed));
            }

            auto [lo, hi] = met.cci();
            ofs << cond.name << ",Base,0,"
                << std::fixed << std::setprecision(6) << met.cr() << "," << lo << "," << hi << ","
                << std::setprecision(4) << met.mc() << "," << met.p5() << ","
                << met.mp() << "," << std::setprecision(3) << met.ms() << "," << N << "\n";
            ofs.flush();
            std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                      << "% (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }

        // WDRO-sampling (no K parameter)
        {
            std::cout << "  WDRO-sampling: ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = "WDRO-sampling";

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(cond.name) / 1000 + i + 301000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg = make_base_config(cond.switch_prob, 0.1, cond.num_obs);
                cfg.enable_dro = true;
                cfg.injection_mode = InjectionMode::QSTAR_SAMPLE;
                cfg.method_name = "WDRO-sampling";
                cfg.custom_ref_path = cond.path;
                cfg.custom_initial_ego = cond.ego;

                std::vector<ObstacleState> obs;
                for (double f : cond.obs_fracs)
                    obs.push_back(obs_oncoming(cond.path, f, rng, cond.obs_speed));
                cfg.initial_obstacle_states = obs;

                met.add(run_experiment_rollout(cfg, seed));
            }

            auto [lo, hi] = met.cci();
            ofs << cond.name << ",WDRO-sampling,0,"
                << std::fixed << std::setprecision(6) << met.cr() << "," << lo << "," << hi << ","
                << std::setprecision(4) << met.mc() << "," << met.p5() << ","
                << met.mp() << "," << std::setprecision(3) << met.ms() << "," << N << "\n";
            ofs.flush();
            std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                      << "% (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }

        // Sweep K for each injection method
        for (const auto& inj : INJ_METHODS) {
            for (int K : K_VALUES) {
                std::string label = inj.label + "-K" + std::to_string(K);
                std::cout << "  " << label << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();
                Metrics met; met.method = label;

                for (int i = 0; i < N; ++i) {
                    if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        std::hash<std::string>{}(cond.name) / 1000 +
                        static_cast<int>(inj.mode) * 1000000 +
                        K * 100000 + i + 302000000);
                    std::mt19937 rng(seed);

                    ExperimentConfig cfg = make_base_config(cond.switch_prob, 0.1, cond.num_obs);
                    cfg.enable_dro = true;
                    cfg.injection_mode = inj.mode;
                    cfg.dro_injection_count = K;
                    cfg.method_name = label;
                    cfg.custom_ref_path = cond.path;
                    cfg.custom_initial_ego = cond.ego;

                    std::vector<ObstacleState> obs;
                    for (double f : cond.obs_fracs)
                        obs.push_back(obs_oncoming(cond.path, f, rng, cond.obs_speed));
                    cfg.initial_obstacle_states = obs;

                    met.add(run_experiment_rollout(cfg, seed));
                }

                auto [lo, hi] = met.cci();
                ofs << cond.name << "," << inj.label << "," << K << ","
                    << std::fixed << std::setprecision(6) << met.cr() << "," << lo << "," << hi << ","
                    << std::setprecision(4) << met.mc() << "," << met.p5() << ","
                    << met.mp() << "," << std::setprecision(3) << met.ms() << "," << N << "\n";
                ofs.flush();
                std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                          << "% clr=" << std::setprecision(2) << met.mc()
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }

    ofs.close();
    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Written: " << filepath << "\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total
              << "s (" << std::setprecision(1) << total/60.0 << " min)\n";
    std::cout << "========================================\n";
    return 0;
}
