/**
 * @file test_multi_risk_injection.cpp
 * @brief Test injection algorithms when multiple geometrically-diverse
 *        high-risk modes exist simultaneously.
 *
 * Hypothesis: TopRisk greedily selects the top-K modes by risk score r_m.
 * When several modes have similar high risk but very different trajectories,
 * TopRisk may select redundant modes from the same "risk cluster" while
 * ignoring an equally dangerous cluster.  WDRO should spread injection
 * mass across diverse threats via transport-cost awareness.
 *
 * Design:
 *   - 10-mode obstacle model with two high-risk "clusters":
 *       Cluster A  (approach-left):  hard_left, drift_left, accel_left
 *       Cluster B  (approach-right): hard_right, drift_right, accel_right
 *     Plus 4 lower-risk modes: constant_velocity, decelerating, slow_left, slow_right
 *   - Obstacle placed directly ahead so both left-approach and right-approach
 *     clusters threaten the ego equally.
 *   - The risk scores within each cluster are nearly equal, so TopRisk's
 *     greedy ordering is arbitrary within a cluster but will tend to pick
 *     multiple modes from whichever cluster has the marginally highest r_m,
 *     leaving the other cluster uncovered.
 *
 * Output: figures/multi_risk/
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

static const std::string DIR = "figures/multi_risk/";

static constexpr int    HORIZON         = DEFAULT_HORIZON;
static constexpr double DT              = DEFAULT_DT;
static constexpr int    NUM_SCENARIOS   = DEFAULT_BASE_SCENARIOS;
static constexpr int    ROLLOUT_STEPS   = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;

// ============================================================================
// Rich mode set: 10 modes with two high-risk clusters
// ============================================================================

/// Build a turn mode with specified omega.
static ModeModel make_turn(const std::string& id, double dt, double omega,
                            double noise_scale = 0.5, const std::string& desc = "") {
    double cw = std::cos(omega * dt);
    double sw = std::sin(omega * dt);
    Eigen::Matrix4d A;
    A << 1, 0, dt*cw, -dt*sw,
         0, 1, dt*sw,  dt*cw,
         0, 0, cw,     -sw,
         0, 0, sw,      cw;
    Eigen::Vector4d b = Eigen::Vector4d::Zero();
    Eigen::MatrixXd G(4, 2);
    G << 0.5*dt*dt, 0, 0, 0.5*dt*dt, dt, 0, 0, dt;
    G *= noise_scale;
    return ModeModel(id, A, b, G, desc.empty() ? id : desc);
}

/// Build a lateral-drift mode (lane-change style).
static ModeModel make_drift(const std::string& id, double dt, double lat_vel,
                              double noise_scale = 0.5) {
    Eigen::Matrix4d A;
    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;
    Eigen::Vector4d b;
    b << 0, lat_vel * dt, 0, 0;
    Eigen::MatrixXd G(4, 2);
    G << 0.5*dt*dt, 0, 0, 0.5*dt*dt, dt, 0, 0, dt;
    G *= noise_scale;
    return ModeModel(id, A, b, G, id);
}

/// Build an accelerating + turning mode (compound).
static ModeModel make_accel_turn(const std::string& id, double dt,
                                   double omega, double accel,
                                   double noise_scale = 0.5) {
    double cw = std::cos(omega * dt);
    double sw = std::sin(omega * dt);
    Eigen::Matrix4d A;
    A << 1, 0, dt*cw, -dt*sw,
         0, 1, dt*sw,  dt*cw,
         0, 0, cw,     -sw,
         0, 0, sw,      cw;
    Eigen::Vector4d b;
    b << 0, 0, accel * dt * cw, accel * dt * sw;
    Eigen::MatrixXd G(4, 2);
    G << 0.5*dt*dt, 0, 0, 0.5*dt*dt, dt, 0, 0, dt;
    G *= noise_scale;
    return ModeModel(id, A, b, G, id);
}

/// Create the 10-mode model set.  Cluster A attacks from the left,
/// Cluster B attacks from the right.  Both are high-risk.
static std::map<std::string, ModeModel> make_rich_mode_set(double dt) {
    std::map<std::string, ModeModel> modes;

    // --- Low-risk modes (4) ---
    {   // constant_velocity
        Eigen::Matrix4d A;
        A << 1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1;
        Eigen::Vector4d b = Eigen::Vector4d::Zero();
        Eigen::MatrixXd G(4,2);
        G << 0.5*dt*dt,0, 0,0.5*dt*dt, dt,0, 0,dt;
        G *= 0.5;
        modes["cv"] = ModeModel("cv", A, b, G, "Constant velocity");
    }
    {   // decelerating
        Eigen::Matrix4d A;
        A << 1,0,dt,0, 0,1,0,dt, 0,0,1,0, 0,0,0,1;
        Eigen::Vector4d b;
        b << 0, 0, -0.4*dt, -0.4*dt;
        Eigen::MatrixXd G(4,2);
        G << 0.5*dt*dt,0, 0,0.5*dt*dt, dt,0, 0,dt;
        G *= 0.5;
        modes["decel"] = ModeModel("decel", A, b, G, "Decelerating");
    }
    // Gentle turns (low risk because omega is small)
    modes["slow_left"]  = make_turn("slow_left",  dt,  0.15, 0.4, "Gentle left");
    modes["slow_right"] = make_turn("slow_right", dt, -0.15, 0.4, "Gentle right");

    // --- Cluster A: approach-left (3 high-risk modes) ---
    // These all curve the obstacle toward the ego's left side
    modes["hard_left"]  = make_turn("hard_left",  dt,  0.6, 0.5, "Hard left turn");
    modes["drift_left"] = make_drift("drift_left", dt, 0.5, 0.5);
    modes["accel_left"] = make_accel_turn("accel_left", dt, 0.4, 0.3, 0.5);

    // --- Cluster B: approach-right (3 high-risk modes) ---
    // These all curve the obstacle toward the ego's right side
    modes["hard_right"]  = make_turn("hard_right",  dt, -0.6, 0.5, "Hard right turn");
    modes["drift_right"] = make_drift("drift_right", dt, -0.5, 0.5);
    modes["accel_right"] = make_accel_turn("accel_right", dt, -0.4, 0.3, 0.5);

    return modes;
}

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

// ============================================================================
// Obstacle placement: head-on so left and right clusters are symmetrically risky
// ============================================================================

static ObstacleState obs_headon(const ReferencePath& path, double frac,
                                 std::mt19937& rng, double speed = 0.9) {
    double s = frac * path.total_length();
    PathPoint pp = path.get_point_at(s);
    Eigen::Vector2d n(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d t(std::cos(pp.heading), std::sin(pp.heading));
    // Tiny lateral jitter to break perfect symmetry, but keep both clusters dangerous
    std::uniform_real_distribution<double> lat(-0.15, 0.15), spd(-0.05, 0.05);
    Eigen::Vector2d pos = pp.position + lat(rng) * n;
    double v = speed + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -v * t.x(), -v * t.y());
}

// ============================================================================
// Main
// ============================================================================

int main() {
    fs::create_directories(DIR);

    std::cout << "========================================\n";
    std::cout << "Multi-Risk Injection Comparison\n";
    std::cout << "  10 modes: 4 low-risk + 3 left-cluster + 3 right-cluster\n";
    std::cout << "  Methods: Base, WDRO-sampling, WDRO-inject, TopRisk, DiverseRisk\n";
    std::cout << "  K sweep: 1..5\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    const int N = 1000;
    const std::vector<int> K_VALUES = {1, 2, 3, 4, 5};

    // All 10 mode names (controller knows all of them)
    const std::vector<std::string> ALL_MODE_NAMES = {
        "cv", "decel", "slow_left", "slow_right",
        "hard_left", "drift_left", "accel_left",
        "hard_right", "drift_right", "accel_right"
    };

    // Register custom modes in the global mode model map
    auto custom_modes = make_rich_mode_set(DT);

    // Conditions: different obstacle placements and paths
    struct Condition {
        std::string name;
        ReferencePath path;
        EgoState ego;
        double switch_prob;
        int num_obs;
        std::vector<double> obs_fracs;
        double obs_speed;
    };

    auto scurve = ReferencePath::create_s_curve(25.0, 3.0, 200);
    auto tight  = ReferencePath::create_s_curve(20.0, 5.0, 200);
    auto straight = ReferencePath::create_straight(
        Eigen::Vector2d(0, 0), Eigen::Vector2d(25, 0), 200);

    std::vector<Condition> conditions = {
        {"Straight-head-on",  straight, EgoState(0,0,0,1.5), 0.3, 1, {0.55}, 0.9},
        {"S-curve-head-on",   scurve,   EgoState(0,0,0,1.5), 0.3, 1, {0.55}, 0.9},
        {"Tight-S-head-on",   tight,    EgoState(0,0,0,1.5), 0.3, 1, {0.55}, 0.9},
        {"2obs-straight",     straight, EgoState(0,0,0,1.5), 0.3, 2, {0.45, 0.60}, 0.9},
        {"High-switch",       scurve,   EgoState(0,0,0,1.5), 0.5, 1, {0.55}, 0.9},
    };

    struct InjMethod {
        std::string label;
        InjectionMode mode;
    };

    const std::vector<InjMethod> INJ_METHODS = {
        {"WDRO-inject",   InjectionMode::DRO},
        {"TopRisk",       InjectionMode::TOP_RISK_INJECT},
        {"DiverseRisk",   InjectionMode::DIVERSE_RISK_INJECT},
    };

    std::string filepath = DIR + "multi_risk.csv";
    std::ofstream ofs(filepath);
    ofs << "condition,method,K,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& cond : conditions) {
        std::cout << "\n--- " << cond.name << " ---\n";

        // Lambda to run one method config
        auto run_config = [&](const std::string& label, InjectionMode inj_mode,
                              int K, bool enable_dro) {
            std::cout << "  " << label << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();
            Metrics met; met.method = label;

            for (int i = 0; i < N; ++i) {
                if (i > 0 && i % 250 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(cond.name) / 1000 +
                    std::hash<std::string>{}(label) / 1000 +
                    K * 100000 + i + 400000000);
                std::mt19937 rng(seed);

                ExperimentConfig cfg;
                cfg.horizon = HORIZON;
                cfg.num_scenarios = NUM_SCENARIOS;
                cfg.switch_prob = cond.switch_prob;
                cfg.rollout_steps = ROLLOUT_STEPS;
                cfg.obs_modes = ALL_MODE_NAMES;
                cfg.rare_mode = "";  // no rare mode; all 10 are "known"
                cfg.rare_switch_prob = 0.0;
                cfg.num_discs = NUM_DISCS;
                cfg.vehicle_length = VEHICLE_LENGTH;
                cfg.safe_horizon_enabled = true;
                cfg.safe_horizon_min = SAFE_HORIZON_MIN;
                cfg.path_completion_termination = true;
                cfg.path_completion_fraction = 0.95;
                cfg.weight_type = WeightType::FREQUENCY;
                cfg.enable_dro = enable_dro;
                cfg.injection_mode = inj_mode;
                cfg.dro_injection_count = K;
                cfg.method_name = label;
                cfg.ablation = enable_dro ? AblationVariant::DRO_FULL : AblationVariant::NO_INJECTION;
                cfg.num_obstacles = cond.num_obs;
                cfg.obstacles_per_class = 1;
                cfg.custom_ref_path = cond.path;
                cfg.custom_initial_ego = cond.ego;
                cfg.custom_mode_models = custom_modes;

                std::vector<ObstacleState> obs;
                for (double f : cond.obs_fracs)
                    obs.push_back(obs_headon(cond.path, f, rng, cond.obs_speed));
                cfg.initial_obstacle_states = obs;

                met.add(run_experiment_rollout(cfg, seed));
            }

            auto [lo, hi] = met.cci();
            ofs << cond.name << "," << label << "," << K << ","
                << std::fixed << std::setprecision(6) << met.cr() << "," << lo << "," << hi << ","
                << std::setprecision(4) << met.mc() << "," << met.p5() << ","
                << met.mp() << "," << std::setprecision(3) << met.ms() << "," << N << "\n";
            ofs.flush();

            std::cout << N << " coll=" << std::setprecision(1) << (met.cr()*100)
                      << "% [" << (lo*100) << "," << (hi*100) << "]"
                      << " clr=" << std::setprecision(2) << met.mc()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        };

        // Baselines
        run_config("Base", InjectionMode::NONE, 0, false);
        run_config("WDRO-sampling", InjectionMode::QSTAR_SAMPLE, 0, true);

        // Injection K-sweep
        for (const auto& inj : INJ_METHODS) {
            for (int K : K_VALUES) {
                std::string label = inj.label + "-K" + std::to_string(K);
                run_config(label, inj.mode, K, true);
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
