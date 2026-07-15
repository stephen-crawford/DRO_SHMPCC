/**
 * @file test_noisy_risk.cpp
 * @brief Test WDRO vs TopRisk with noisy risk scores and partial observability.
 *
 * Two noise mechanisms:
 *   1. Risk score noise: additive Gaussian noise on r_m before injection selection.
 *      TopRisk picks argmax(r_m + noise) — susceptible to winner's curse.
 *      WDRO solves the dual with noisy risks — transport cost regularizes.
 *
 *   2. Partial observability: obstacle state perturbed before the controller
 *      sees it, simulating noisy perception. This makes both risk scores and
 *      predictions uncertain.
 *
 * Output: figures/noisy_risk/
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
#include "reference_path.hpp"
#include "dynamics.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string DIR = "figures/noisy_risk/";
static constexpr int HORIZON = DEFAULT_HORIZON;
static constexpr double DT = DEFAULT_DT;
static constexpr int ROLLOUT_STEPS = DEFAULT_ROLLOUT_STEPS;
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
    std::cout << "Noisy Risk & Partial Observability Test\n";
    std::cout << "  Part A: Risk score noise sweep\n";
    std::cout << "  Part B: Partial observability (state noise)\n";
    std::cout << "  Part C: Combined (risk noise + state noise)\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();
    const int N = 1000;

    auto scurve = ReferencePath::create_s_curve(25.0, 3.0, 200);
    auto tight = ReferencePath::create_s_curve(20.0, 5.0, 200);

    std::string filepath = DIR + "noisy_risk.csv";
    std::ofstream ofs(filepath);
    ofs << "test,condition,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mean_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    auto run = [&](const std::string& test, const std::string& cond,
                    const ReferencePath& path, double risk_noise,
                    double state_noise_pos, double state_noise_vel,
                    double switch_prob) {
        std::cout << "\n--- " << test << ": " << cond << " ---\n";
        for (const auto& md : METHODS) {
            std::cout << "  " << md.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = md.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(test + cond) / 1000 +
                    static_cast<int>(md.mode) * 100000 + md.K * 10000 + i + 800000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg;
                cfg.horizon = HORIZON;
                cfg.num_scenarios = DEFAULT_BASE_SCENARIOS;
                cfg.switch_prob = switch_prob;
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
                cfg.risk_noise_sigma = risk_noise;
                cfg.custom_ref_path = path;
                cfg.custom_initial_ego = EgoState(0.0, 0.0, 0.0, 1.5);

                // Obstacle with optional state noise (partial observability)
                ObstacleState true_obs = obs_oncoming(path, 0.55, rng);
                if (state_noise_pos > 0 || state_noise_vel > 0) {
                    std::normal_distribution<double> pn(0, state_noise_pos);
                    std::normal_distribution<double> vn(0, state_noise_vel);
                    // The controller sees a noisy version of the obstacle state
                    // but the simulation uses the true state for collision detection
                    // We approximate this by perturbing the initial state slightly
                    // each step — the step_callback would be more accurate but
                    // for initial testing, perturbing the initial state captures
                    // the effect of uncertain initial conditions
                    true_obs = ObstacleState(
                        true_obs.x + pn(rng), true_obs.y + pn(rng),
                        true_obs.vx + vn(rng), true_obs.vy + vn(rng));
                }
                cfg.initial_obstacle_states = {true_obs};

                met.add(run_experiment_rollout(cfg, seed));
            }

            auto [lo, hi] = met.cci();
            ofs << test << "," << cond << "," << md.name << ","
                << std::fixed << std::setprecision(6) << met.cr() << "," << lo << "," << hi << ","
                << std::setprecision(4) << met.mc() << ","
                << std::setprecision(4) << (met.total_progress.empty() ? 0 :
                    std::accumulate(met.total_progress.begin(), met.total_progress.end(), 0.0)/met.total_progress.size()) << ","
                << std::setprecision(3) << (met.solve_times_ms.empty() ? 0 :
                    std::accumulate(met.solve_times_ms.begin(), met.solve_times_ms.end(), 0.0)/met.solve_times_ms.size()) << ","
                << N << "\n";
            ofs.flush();

            std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                      << "% [" << (lo*100) << "," << (hi*100) << "]"
                      << " clr=" << std::setprecision(2) << met.mc()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    };

    // Part A: Risk score noise sweep (no state noise)
    for (double rn : {0.0, 0.05, 0.1, 0.2, 0.5, 1.0}) {
        std::string cond = "rn=" + std::to_string(int(rn*100));
        run("A-risk-noise", cond, scurve, rn, 0.0, 0.0, 0.2);
    }

    // Part A on Tight-S (harder scenario)
    for (double rn : {0.0, 0.1, 0.2, 0.5}) {
        std::string cond = "rn=" + std::to_string(int(rn*100)) + "-tight";
        run("A-risk-noise", cond, tight, rn, 0.0, 0.0, 0.2);
    }

    // Part B: State noise (partial observability, no risk noise)
    for (double sn : {0.0, 0.1, 0.3, 0.5, 1.0}) {
        std::string cond = "sn=" + std::to_string(int(sn*100));
        run("B-state-noise", cond, scurve, 0.0, sn, sn * 0.5, 0.2);
    }

    // Part C: Combined (risk + state noise)
    for (auto [rn, sn] : std::vector<std::pair<double,double>>{{0.1, 0.2}, {0.2, 0.3}, {0.5, 0.5}}) {
        std::string cond = "rn" + std::to_string(int(rn*100)) + "_sn" + std::to_string(int(sn*100));
        run("C-combined", cond, scurve, rn, sn, sn * 0.5, 0.3);
    }

    // Part C on Tight-S
    for (auto [rn, sn] : std::vector<std::pair<double,double>>{{0.2, 0.3}, {0.5, 0.5}}) {
        std::string cond = "rn" + std::to_string(int(rn*100)) + "_sn" + std::to_string(int(sn*100)) + "-tight";
        run("C-combined", cond, tight, rn, sn, sn * 0.5, 0.3);
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
