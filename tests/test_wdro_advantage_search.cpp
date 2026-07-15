/**
 * @file test_wdro_advantage_search.cpp
 * @brief Search for conditions where WDRO outperforms TopRisk.
 *
 * Five hypothesis tests, each targeting a different theoretical failure
 * mode of greedy risk-ranked injection:
 *
 *   H1: Low scenario budget (S=5,8,12) — injection has more relative weight
 *   H2: Rare-but-riskiest mode — TopRisk injects an implausible mode
 *   H3: Adversarial switching — obstacle picks controller's weakest mode
 *   H4: High process noise — noisy risk estimates destabilize TopRisk
 *   H5: Biased belief — controller's mode weights are deliberately wrong
 *
 * Output: figures/wdro_advantage/
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
#include <functional>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"
#include "reference_path.hpp"
#include "dynamics.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string DIR = "figures/wdro_advantage/";
static constexpr int    HORIZON         = DEFAULT_HORIZON;
static constexpr double DT              = DEFAULT_DT;
static constexpr int    ROLLOUT_STEPS   = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;

static const std::vector<std::string> BASE_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};

// ============================================================================
// Metrics
// ============================================================================

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

// Shared method definitions
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

static void write_row(std::ofstream& ofs, const std::string& hyp,
                       const std::string& cond, const Metrics& m, int N) {
    auto [lo, hi] = m.cci();
    ofs << hyp << "," << cond << "," << m.method << ","
        << std::fixed << std::setprecision(6) << m.cr() << "," << lo << "," << hi << ","
        << std::setprecision(4) << m.mc() << "," << m.p5() << ","
        << m.mp() << "," << std::setprecision(3) << m.ms() << "," << N << "\n";
    ofs.flush();
}

static void print_result(const Metrics& m, int N, double secs) {
    auto [lo, hi] = m.cci();
    std::cout << N << " coll=" << std::setprecision(1) << (m.cr()*100)
              << "% [" << (lo*100) << "," << (hi*100) << "]"
              << " clr=" << std::setprecision(2) << m.mc()
              << " (" << std::setprecision(0) << secs << "s)\n";
}

// ============================================================================
// H1: Low Scenario Budget
//     With S=5 base scenarios, each injection is 20% of the constraint set.
//     TopRisk may duplicate what sampling already covers.
// ============================================================================

static void run_h1(std::ofstream& ofs) {
    std::cout << "\n=== H1: Low Scenario Budget ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    const std::vector<int> BUDGETS = {5, 8, 12, 20};
    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    for (int S : BUDGETS) {
        std::string cond = "S=" + std::to_string(S);
        for (const auto& md : METHODS) {
            std::cout << "  " << cond << " " << md.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = md.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(S * 1000000 +
                    static_cast<int>(md.mode) * 100000 + md.K * 10000 + i + 500000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg;
                cfg.horizon = HORIZON;
                cfg.num_scenarios = S;  // KEY: low budget
                cfg.switch_prob = 0.2;
                cfg.rollout_steps = ROLLOUT_STEPS;
                cfg.obs_modes = BASE_MODES;
                cfg.rare_mode = "lane_change_left";
                cfg.rare_switch_prob = 0.1;
                cfg.num_discs = NUM_DISCS;
                cfg.vehicle_length = VEHICLE_LENGTH;
                cfg.safe_horizon_enabled = true;
                cfg.safe_horizon_min = SAFE_HORIZON_MIN;
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
                cfg.initial_obstacle_states = {obs_oncoming(path, 0.55, rng)};
                met.add(run_experiment_rollout(cfg, seed));
            }
            write_row(ofs, "H1", cond, met, N);
            print_result(met, N, elapsed_sec(t1));
        }
    }
    std::cout << "  H1 done (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// H2: Rare Mode is Riskiest
//     The obstacle mostly does constant_velocity, but the riskiest mode
//     (hard turn toward ego) almost never fires.  TopRisk always injects
//     the rare risky mode; WDRO should recognize CV is the real threat
//     within its Wasserstein ball.
// ============================================================================

static void run_h2(std::ofstream& ofs) {
    std::cout << "\n=== H2: Rare Mode is Riskiest ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    // Very low rare_switch_prob so the obstacle almost never uses the rare mode
    const std::vector<double> RARE_PROBS = {0.01, 0.03, 0.05, 0.10};
    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    for (double rp : RARE_PROBS) {
        std::string cond = "rare=" + std::to_string(int(rp * 100)) + "%";
        for (const auto& md : METHODS) {
            std::cout << "  " << cond << " " << md.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = md.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(rp * 10000) * 100000 +
                    static_cast<int>(md.mode) * 10000 + md.K * 1000 + i + 510000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg;
                cfg.horizon = HORIZON;
                cfg.num_scenarios = DEFAULT_BASE_SCENARIOS;
                cfg.switch_prob = 0.05;  // Low switching — obstacle mostly stays in one mode
                cfg.rollout_steps = ROLLOUT_STEPS;
                cfg.obs_modes = {"constant_velocity", "decelerating"};  // Only safe modes known
                cfg.rare_mode = "turn_left";  // Riskiest mode is "rare"
                cfg.rare_switch_prob = rp;
                cfg.num_discs = NUM_DISCS;
                cfg.vehicle_length = VEHICLE_LENGTH;
                cfg.safe_horizon_enabled = true;
                cfg.safe_horizon_min = SAFE_HORIZON_MIN;
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
                cfg.initial_obstacle_states = {obs_oncoming(path, 0.55, rng)};
                met.add(run_experiment_rollout(cfg, seed));
            }
            write_row(ofs, "H2", cond, met, N);
            print_result(met, N, elapsed_sec(t1));
        }
    }
    std::cout << "  H2 done (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// H3: Adversarial Mode Switching
//     At each step, the obstacle switches to whichever mode the controller
//     has the LOWEST weight for — the mode it's least prepared for.
//     WDRO's ball should provide better coverage of low-probability modes.
// ============================================================================

static void run_h3(std::ofstream& ofs) {
    std::cout << "\n=== H3: Adversarial Switching ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    // Adversarial callback: every few steps, switch to the mode with lowest weight
    auto make_adversarial_cb = [](int switch_interval) {
        return [switch_interval](
            int step, int obs_id, ObstacleSim& obs_sim,
            AdaptiveScenarioMPC& controller, std::mt19937& rng
        ) {
            if (obs_id != 0) return;
            if (step > 0 && step % switch_interval == 0) {
                // Find mode with lowest weight in controller's belief
                const auto& scenarios = controller.scenarios();
                std::map<std::string, int> mode_counts;
                for (const auto& [mid, _] : obs_sim.mode_models)
                    mode_counts[mid] = 0;
                for (const auto& sc : scenarios) {
                    auto it = sc.trajectories.find(0);
                    if (it != sc.trajectories.end())
                        mode_counts[it->second.mode_id]++;
                }
                // Switch to least-represented mode
                std::string least_mode = obs_sim.current_mode;
                int min_count = std::numeric_limits<int>::max();
                for (const auto& [mid, cnt] : mode_counts) {
                    if (cnt < min_count) {
                        min_count = cnt;
                        least_mode = mid;
                    }
                }
                obs_sim.current_mode = least_mode;
            }
        };
    };

    const std::vector<int> INTERVALS = {3, 5, 8};

    for (int interval : INTERVALS) {
        std::string cond = "every-" + std::to_string(interval) + "-steps";
        for (const auto& md : METHODS) {
            std::cout << "  " << cond << " " << md.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = md.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    interval * 1000000 +
                    static_cast<int>(md.mode) * 100000 + md.K * 10000 + i + 520000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg;
                cfg.horizon = HORIZON;
                cfg.num_scenarios = DEFAULT_BASE_SCENARIOS;
                cfg.switch_prob = 0.0;  // Disable random switching; adversarial only
                cfg.rollout_steps = ROLLOUT_STEPS;
                cfg.obs_modes = BASE_MODES;
                cfg.rare_mode = "lane_change_left";
                cfg.rare_switch_prob = 0.0;
                cfg.num_discs = NUM_DISCS;
                cfg.vehicle_length = VEHICLE_LENGTH;
                cfg.safe_horizon_enabled = true;
                cfg.safe_horizon_min = SAFE_HORIZON_MIN;
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
                cfg.initial_obstacle_states = {obs_oncoming(path, 0.55, rng)};
                cfg.step_callback = make_adversarial_cb(interval);
                met.add(run_experiment_rollout(cfg, seed));
            }
            write_row(ofs, "H3", cond, met, N);
            print_result(met, N, elapsed_sec(t1));
        }
    }
    std::cout << "  H3 done (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// H4: High Process Noise
//     Scale the noise matrix G so risk estimates are noisier.
//     TopRisk's greedy pick may be unstable when r_m values are noisy.
// ============================================================================

static void run_h4(std::ofstream& ofs) {
    std::cout << "\n=== H4: High Process Noise ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    const std::vector<double> NOISE_SCALES = {1.0, 2.0, 4.0, 8.0};
    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    for (double ns : NOISE_SCALES) {
        std::string cond = "noise=" + std::to_string(int(ns)) + "x";

        // Create custom modes with scaled noise
        auto custom_modes = create_obstacle_mode_models(DT);
        for (auto& [id, model] : custom_modes) {
            model.G *= ns;
        }

        for (const auto& md : METHODS) {
            std::cout << "  " << cond << " " << md.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = md.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(ns * 100) * 1000000 +
                    static_cast<int>(md.mode) * 100000 + md.K * 10000 + i + 530000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg;
                cfg.horizon = HORIZON;
                cfg.num_scenarios = DEFAULT_BASE_SCENARIOS;
                cfg.switch_prob = 0.2;
                cfg.rollout_steps = ROLLOUT_STEPS;
                cfg.obs_modes = BASE_MODES;
                cfg.rare_mode = "lane_change_left";
                cfg.rare_switch_prob = 0.1;
                cfg.num_discs = NUM_DISCS;
                cfg.vehicle_length = VEHICLE_LENGTH;
                cfg.safe_horizon_enabled = true;
                cfg.safe_horizon_min = SAFE_HORIZON_MIN;
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
                cfg.initial_obstacle_states = {obs_oncoming(path, 0.55, rng)};
                cfg.custom_mode_models = custom_modes;
                met.add(run_experiment_rollout(cfg, seed));
            }
            write_row(ofs, "H4", cond, met, N);
            print_result(met, N, elapsed_sec(t1));
        }
    }
    std::cout << "  H4 done (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// H5: Biased Nominal Belief
//     Force the controller to start with a wrong belief (100% constant_velocity)
//     while the obstacle is actually turning.  WDRO's Wasserstein ball should
//     correct the biased belief better than TopRisk which ignores the belief.
//     Actually — TopRisk ignoring the belief might be BETTER here.
//     So we also test: controller believes turn_left but obstacle does CV.
//     The "correct" injection depends on whether you trust the belief.
// ============================================================================

static void run_h5(std::ofstream& ofs) {
    std::cout << "\n=== H5: Biased Belief (rapid switching from wrong start) ===\n";
    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    // Rapid switching: obstacle changes mode every 2-3 steps.
    // With short mode history, the controller's belief lags behind reality.
    // WDRO's ball should hedge against belief lag.
    const std::vector<double> SWITCH_PROBS = {0.4, 0.6, 0.8};

    for (double sp : SWITCH_PROBS) {
        std::string cond = "sp=" + std::to_string(int(sp * 100)) + "%";
        for (const auto& md : METHODS) {
            std::cout << "  " << cond << " " << md.name << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = md.name;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(sp * 1000) * 1000000 +
                    static_cast<int>(md.mode) * 100000 + md.K * 10000 + i + 540000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg;
                cfg.horizon = HORIZON;
                cfg.num_scenarios = DEFAULT_BASE_SCENARIOS;
                cfg.switch_prob = sp;  // Very high switching
                cfg.rollout_steps = ROLLOUT_STEPS;
                cfg.obs_modes = BASE_MODES;
                cfg.rare_mode = "lane_change_left";
                cfg.rare_switch_prob = 0.2;  // Rare mode also very active
                cfg.num_discs = NUM_DISCS;
                cfg.vehicle_length = VEHICLE_LENGTH;
                cfg.safe_horizon_enabled = true;
                cfg.safe_horizon_min = SAFE_HORIZON_MIN;
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
                cfg.initial_obstacle_states = {obs_oncoming(path, 0.55, rng)};
                met.add(run_experiment_rollout(cfg, seed));
            }
            write_row(ofs, "H5", cond, met, N);
            print_result(met, N, elapsed_sec(t1));
        }
    }
    std::cout << "  H5 done (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    fs::create_directories(DIR);

    std::set<std::string> filters;
    for (int i = 1; i < argc; ++i) filters.insert(argv[i]);
    bool run_all = filters.empty();

    std::cout << "========================================\n";
    std::cout << "WDRO Advantage Search\n";
    std::cout << "  H1: Low scenario budget\n";
    std::cout << "  H2: Rare mode is riskiest\n";
    std::cout << "  H3: Adversarial switching\n";
    std::cout << "  H4: High process noise\n";
    std::cout << "  H5: Rapid switching / biased belief\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    std::string filepath = DIR + "wdro_advantage.csv";
    std::ofstream ofs(filepath);
    ofs << "hypothesis,condition,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    if (run_all || filters.count("h1")) run_h1(ofs);
    if (run_all || filters.count("h2")) run_h2(ofs);
    if (run_all || filters.count("h3")) run_h3(ofs);
    if (run_all || filters.count("h4")) run_h4(ofs);
    if (run_all || filters.count("h5")) run_h5(ofs);

    ofs.close();
    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Written: " << filepath << "\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0)
              << total << "s (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";
    return 0;
}
