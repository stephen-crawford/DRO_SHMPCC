/**
 * @file test_robustness.cpp
 * @brief Robustness experiments: novel unseen modes and mode misclassification.
 *
 * Tests how mode-level DRO and trajectory-level DRO handle:
 *   R1: Novel (unseen) modes — obstacle follows a "swerving" mode that is
 *       not in the controller's mode set. Trajectory DRO should be more
 *       robust since it works directly on trajectory particles, while
 *       mode DRO can only reweight known modes.
 *   R2: Mode misclassification — modes with similar trajectory realizations
 *       (e.g., slow_turn vs lane_change at low speed) are confused by the
 *       observer. Tests degradation from misclassification.
 *   R3: Novel mode severity sweep — vary how different the novel mode is
 *       from known modes (swerve amplitude 0.1 to 1.0).
 *   R4: Misclassification rate sweep — vary the probability that the
 *       observer reports the wrong (but similar) mode.
 *
 * Output: figures/robustness/
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
#include "scenario_sampler.hpp"
#include "mode_weights.hpp"
#include "reference_path.hpp"
#include "dynamics.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

// ============================================================================
// Constants
// ============================================================================

static const std::string OUTPUT_DIR = "figures/robustness/";

static constexpr int    HORIZON         = DEFAULT_HORIZON;
static constexpr double DT              = DEFAULT_DT;
static constexpr int    NUM_SCENARIOS   = DEFAULT_BASE_SCENARIOS;
static constexpr int    ROLLOUT_STEPS   = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;
static constexpr int    N_ROLLOUTS      = 500;

static const std::vector<std::string> BASE_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Novel Mode Definitions
// ============================================================================

/// Create a "swerving" mode: sinusoidal lateral oscillation.
/// The A matrix couples position and velocity with an oscillatory bias.
static ModeModel create_swerve_mode(double dt, double amplitude = 0.5) {
    Eigen::Matrix4d A;
    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1;

    Eigen::Vector4d b;
    b << 0, 0, 0, amplitude * dt;

    Eigen::MatrixXd G(4, 2);
    G << 0.5 * dt * dt, 0,
         0, 0.5 * dt * dt,
         dt, 0,
         0, dt;
    G *= 0.5;

    return ModeModel("swerving", A, b, G, "Swerving lateral oscillation");
}

/// Create a "sharp_swerve" mode: aggressive version.
static ModeModel create_sharp_swerve_mode(double dt, double amplitude = 1.0) {
    Eigen::Matrix4d A;
    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 0.95, 0,
         0, 0, 0, 0.95;

    Eigen::Vector4d b;
    b << 0, 0, 0, amplitude * dt;

    Eigen::MatrixXd G(4, 2);
    G << 0.5 * dt * dt, 0,
         0, 0.5 * dt * dt,
         dt, 0,
         0, dt;
    G *= 0.8;

    return ModeModel("sharp_swerve", A, b, G, "Sharp swerving maneuver");
}

/// Create a "spiraling" mode: tight circular motion with outward drift.
/// Uses a high turn rate (omega=1.2 rad/s, vs 0.3 for known turns) plus
/// a small centrifugal-like outward bias, producing a spiral trajectory
/// that is qualitatively unlike any mode in the controller's set.
static ModeModel create_spiral_mode(double dt) {
    double omega = 1.2;  // ~4x the known turn rate
    double cos_w = std::cos(omega * dt);
    double sin_w = std::sin(omega * dt);

    Eigen::Matrix4d A;
    A << 1, 0, dt * cos_w, -dt * sin_w,
         0, 1, dt * sin_w,  dt * cos_w,
         0, 0, cos_w,       -sin_w,
         0, 0, sin_w,        cos_w;

    // Outward drift bias: velocity grows slightly each step
    Eigen::Vector4d b;
    b << 0, 0, 0.15 * dt, 0.15 * dt;

    Eigen::MatrixXd G(4, 2);
    G << 0.5 * dt * dt, 0,
         0, 0.5 * dt * dt,
         dt, 0,
         0, dt;
    G *= 0.6;

    return ModeModel("spiral", A, b, G, "Tight spiraling motion");
}

/// Create a "hard braking" mode: sudden extreme deceleration.
/// Velocity decays by ~40% per step (A diagonal 0.6 on vx, vy) plus a
/// strong deceleration bias. Much more aggressive than the known
/// "decelerating" mode (-0.5*dt bias with no velocity decay).
static ModeModel create_hard_brake_mode(double dt) {
    Eigen::Matrix4d A;
    A << 1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 0.6, 0,   // 40% velocity loss per step
         0, 0, 0, 0.6;

    // Strong backward bias on top of the decay
    Eigen::Vector4d b;
    b << 0, 0, -1.5 * dt, -1.5 * dt;

    Eigen::MatrixXd G(4, 2);
    G << 0.5 * dt * dt, 0,
         0, 0.5 * dt * dt,
         dt, 0,
         0, dt;
    G *= 0.3;  // Low noise — braking is deterministic-ish

    return ModeModel("hard_brake", A, b, G, "Emergency hard braking");
}

/// Create a "U-turn" mode: very sharp reversal.
/// Uses omega=2.0 rad/s (near pi/dt for aggressive turn) so the obstacle
/// curves back toward the ego within a few steps. Completely unlike
/// the gentle omega=0.3 turns in the controller's mode set.
static ModeModel create_uturn_mode(double dt) {
    double omega = 2.0;  // ~7x the known turn rate
    double cos_w = std::cos(omega * dt);
    double sin_w = std::sin(omega * dt);

    Eigen::Matrix4d A;
    A << 1, 0, dt * cos_w, -dt * sin_w,
         0, 1, dt * sin_w,  dt * cos_w,
         0, 0, cos_w,       -sin_w,
         0, 0, sin_w,        cos_w;

    // Slight speed increase during turn (momentum)
    Eigen::Vector4d b;
    b << 0, 0, 0.1 * dt, 0;

    Eigen::MatrixXd G(4, 2);
    G << 0.5 * dt * dt, 0,
         0, 0.5 * dt * dt,
         dt, 0,
         0, dt;
    G *= 0.4;

    return ModeModel("uturn", A, b, G, "Sharp U-turn reversal");
}

/// Create a "skidding" mode: loss of traction with lateral slide.
/// Forward velocity decays while cross-coupling transfers energy to
/// lateral motion — the obstacle slides sideways. The A matrix has
/// off-diagonal terms (vx -> vy coupling) that no known mode has.
static ModeModel create_skid_mode(double dt) {
    Eigen::Matrix4d A;
    A << 1, 0, dt,  0,
         0, 1, 0,   dt,
         0, 0, 0.75, 0.4,   // vx decays, gains lateral component from vy
         0, 0, -0.3,  0.85;  // vy picks up from vx, decays slower

    // Lateral acceleration bias (sideways push)
    Eigen::Vector4d b;
    b << 0, 0, -0.3 * dt, 0.8 * dt;

    Eigen::MatrixXd G(4, 2);
    G << 0.5 * dt * dt, 0,
         0, 0.5 * dt * dt,
         dt, 0,
         0, dt;
    G *= 0.7;  // Higher noise — skidding is chaotic

    return ModeModel("skid", A, b, G, "Lateral skidding / loss of traction");
}

/// Create a "slow_turn" mode that is similar to lane_change_left
/// but implemented differently (different A matrix, similar trajectory).
static ModeModel create_slow_turn_mode(double dt) {
    double omega = 0.15;  // Slow turn rate
    double cos_w = std::cos(omega * dt);
    double sin_w = std::sin(omega * dt);

    Eigen::Matrix4d A;
    A << 1, 0, dt * cos_w, -dt * sin_w,
         0, 1, dt * sin_w, dt * cos_w,
         0, 0, cos_w, -sin_w,
         0, 0, sin_w, cos_w;

    Eigen::Vector4d b = Eigen::Vector4d::Zero();

    Eigen::MatrixXd G(4, 2);
    G << 0.5 * dt * dt, 0,
         0, 0.5 * dt * dt,
         dt, 0,
         0, dt;
    G *= 0.5;

    return ModeModel("slow_turn", A, b, G, "Slow turning (similar to lane change)");
}

// ============================================================================
// Trajectory DRO callback (reused from test_traj_dro_generalization.cpp)
// ============================================================================

static double compute_trajectory_risk(
    const ObstacleTrajectory& traj,
    const std::vector<EgoState>& ego_ref,
    double safety_radius, int num_discs, double vehicle_length,
    double z_alpha = 1.6449
) {
    double max_risk = 0.0;
    int H = std::min(static_cast<int>(traj.steps.size()),
                     static_cast<int>(ego_ref.size()));
    for (int k = 1; k < H; ++k) {
        const auto& step = traj.steps[k];
        const auto& ego = ego_ref[k];
        Eigen::Vector2d disc_pos = ego.position();
        Eigen::Vector2d diff = step.mean - disc_pos;
        double dist = diff.norm();
        if (dist < 1e-12) { max_risk = std::max(max_risk, safety_radius); continue; }
        Eigen::Vector2d n_dir = diff / dist;
        double var_dir = static_cast<double>(n_dir.transpose() * step.covariance * n_dir);
        double sigma_dir = std::sqrt(std::max(1e-12, var_dir));
        double risk = std::max(0.0, safety_radius + z_alpha * sigma_dir - dist);
        max_risk = std::max(max_risk, risk);
    }
    return max_risk;
}

static std::vector<std::vector<double>> compute_trajectory_transport_costs(
    const std::vector<ObstacleTrajectory>& trajectories
) {
    int S = static_cast<int>(trajectories.size());
    std::vector<std::vector<double>> D(S, std::vector<double>(S, 0.0));
    for (int i = 0; i < S; ++i)
        for (int j = i + 1; j < S; ++j) {
            int H = std::min(static_cast<int>(trajectories[i].steps.size()),
                             static_cast<int>(trajectories[j].steps.size()));
            double total = 0.0;
            for (int k = 0; k < H; ++k) {
                total += (trajectories[i].steps[k].mean - trajectories[j].steps[k].mean).squaredNorm();
            }
            double cost = (H > 0) ? total / H : 0.0;
            D[i][j] = cost; D[j][i] = cost;
        }
    return D;
}

struct TrajDROResult {
    std::vector<double> weights;
    int worst_trajectory_idx = -1;
};

static TrajDROResult solve_trajectory_dro(
    const std::vector<double>& risks,
    const std::vector<std::vector<double>>& D, double rho
) {
    int S = static_cast<int>(risks.size());
    TrajDROResult result;
    if (S == 0) return result;
    double w_nom = 1.0 / S;
    double max_risk = *std::max_element(risks.begin(), risks.end());
    double min_risk = *std::min_element(risks.begin(), risks.end());
    if (max_risk < 1e-12 || std::abs(max_risk - min_risk) < 1e-12) {
        result.weights.assign(S, w_nom);
        result.worst_trajectory_idx = static_cast<int>(
            std::max_element(risks.begin(), risks.end()) - risks.begin());
        return result;
    }
    double max_D = 0.0;
    for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) max_D = std::max(max_D, D[i][j]);
    if (max_D < 1e-12) max_D = 1.0;
    double lambda_max = std::max((max_risk - min_risk) / (max_D * 0.01), 10.0);

    auto eval_dual = [&](double lambda) {
        double obj = lambda * rho;
        for (int i = 0; i < S; ++i) {
            double mx = -1e18;
            for (int j = 0; j < S; ++j) mx = std::max(mx, risks[j] - lambda * D[i][j]);
            obj += w_nom * mx;
        }
        return obj;
    };
    auto build_plan = [&](double lambda) {
        std::vector<double> q(S, 0.0);
        for (int i = 0; i < S; ++i) {
            int bj = 0; double bv = risks[0] - lambda * D[i][0];
            for (int j = 1; j < S; ++j) { double v = risks[j] - lambda * D[i][j]; if (v > bv) { bv = v; bj = j; } }
            q[bj] += w_nom;
        }
        return q;
    };
    auto plan_cost = [&](double lambda) {
        double c = 0.0;
        for (int i = 0; i < S; ++i) {
            int bj = 0; double bv = risks[0] - lambda * D[i][0];
            for (int j = 1; j < S; ++j) { double v = risks[j] - lambda * D[i][j]; if (v > bv) { bv = v; bj = j; } }
            c += w_nom * D[i][bj];
        }
        return c;
    };

    double lo = 0.0, hi = lambda_max, best_lambda = 0.0, best_dual = eval_dual(0.0);
    for (int iter = 0; iter < 50; ++iter) {
        double mid = (lo + hi) / 2.0;
        double dv = eval_dual(mid);
        if (dv < best_dual) { best_dual = dv; best_lambda = mid; }
        if (plan_cost(mid) > rho) lo = mid; else hi = mid;
    }

    result.weights = build_plan(best_lambda);
    int bi = 0; double bw = result.weights[0];
    for (int i = 1; i < S; ++i) if (result.weights[i] > bw) { bw = result.weights[i]; bi = i; }
    result.worst_trajectory_idx = bi;
    return result;
}

static auto make_traj_dro_callback(double rho, int num_scenarios) {
    return [rho, num_scenarios](
        int step, int obs_id, ObstacleSim& obs_sim,
        AdaptiveScenarioMPC& controller, std::mt19937& rng
    ) {
        if (obs_id != 0) return;
        const auto& modes = obs_sim.mode_models;
        int horizon = controller.config().horizon;
        double safety_radius = controller.config().ego_radius + controller.config().obstacle_radius + controller.config().safety_margin;
        int n_discs = controller.config().num_discs;
        double veh_len = controller.config().vehicle_length;

        std::map<std::string, double> freq_weights;
        for (const auto& [mid, _] : modes) freq_weights[mid] = 0.0;
        int cnt = 0;
        for (const auto& sc : controller.scenarios()) {
            auto it = sc.trajectories.find(0);
            if (it != sc.trajectories.end()) { freq_weights[it->second.mode_id] += 1.0; cnt++; }
        }
        if (cnt > 0) { for (auto& [_, w] : freq_weights) w /= cnt; }
        else { for (auto& [_, w] : freq_weights) w = 1.0 / freq_weights.size(); }

        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<ObstacleTrajectory> particles;
        for (int s = 0; s < num_scenarios; ++s) {
            std::string sm = sample_mode_from_weights(freq_weights, rng);
            if (modes.find(sm) == modes.end()) continue;
            const ModeModel& mode = modes.at(sm);
            std::vector<PredictionStep> steps;
            Eigen::Vector4d x = obs_sim.state.to_array();
            Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
            steps.emplace_back(0, x.head<2>(), cov.block<2,2>(0,0));
            for (int k = 0; k < horizon; ++k) {
                Eigen::VectorXd noise(mode.noise_dim());
                for (int d = 0; d < mode.noise_dim(); ++d) noise(d) = nd(rng);
                x = mode.A * x + mode.b + mode.G * noise;
                cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();
                steps.emplace_back(k+1, x.head<2>(), cov.block<2,2>(0,0));
            }
            particles.emplace_back(0, sm, steps, freq_weights[sm]);
        }
        if (static_cast<int>(particles.size()) < 3) return;

        Eigen::Vector2d opos = obs_sim.state.position();
        double heading = std::atan2(opos.y(), opos.x());
        std::vector<EgoState> ego_ref;
        for (int k = 0; k <= horizon; ++k) {
            EgoState es; es.x = 1.5*k*0.1*std::cos(heading); es.y = 1.5*k*0.1*std::sin(heading);
            es.theta = heading; es.v = 1.5; ego_ref.push_back(es);
        }

        int S = static_cast<int>(particles.size());
        std::vector<double> risks(S);
        for (int s = 0; s < S; ++s)
            risks[s] = compute_trajectory_risk(particles[s], ego_ref, safety_radius, n_discs, veh_len);
        auto D = compute_trajectory_transport_costs(particles);
        auto dro = solve_trajectory_dro(risks, D, rho);
        if (dro.worst_trajectory_idx < 0) return;

        // Inject worst-case trajectory
        const auto& wt = particles[dro.worst_trajectory_idx];
        Scenario inj; inj.scenario_id = num_scenarios + 100 + step;
        inj.is_injected = true; inj.probability = 1.0;
        inj.trajectories[0] = wt; inj.trajectories[0].probability = 1.0;
        controller.inject_scenario(inj);

        // Also resample mode weights
        std::map<std::string, double> mw;
        for (const auto& [mid, _] : modes) mw[mid] = 0.0;
        for (int s = 0; s < S; ++s) mw[particles[s].mode_id] += dro.weights[s];
        double sw = 0; for (auto& [_, w] : mw) sw += w;
        if (sw > 1e-12) for (auto& [_, w] : mw) w /= sw;
        controller.reset_scenarios();
        controller.set_custom_mode_weights(0, mw);
    };
}

// ============================================================================
// Metrics
// ============================================================================

struct Metrics {
    std::string method;
    std::vector<bool> collisions;
    std::vector<int> missed_mode_steps, total_mode_checks;
    std::vector<int> rare_mode_active, rare_mode_missed;
    std::vector<double> min_clearances, total_progress, solve_times_ms;

    int n() const { return static_cast<int>(collisions.size()); }
    int cc() const { int c=0; for (bool b : collisions) if (b) c++; return c; }
    double cr() const { return n()>0 ? double(cc())/n() : 0; }
    std::pair<double,double> cci() const { return wilson_ci(cc(), n()); }
    int tm() const { return std::accumulate(missed_mode_steps.begin(), missed_mode_steps.end(), 0); }
    int tc() const { return std::accumulate(total_mode_checks.begin(), total_mode_checks.end(), 0); }
    double mr() const { int t=tc(); return t>0 ? double(tm())/t : 0; }
    double mc() const { return min_clearances.empty() ? 0 : std::accumulate(min_clearances.begin(), min_clearances.end(), 0.0) / min_clearances.size(); }
    double p5c() const {
        if (min_clearances.empty()) return 0;
        auto v = min_clearances;
        std::sort(v.begin(), v.end());
        return v[std::max(0, static_cast<int>(v.size() * 0.05) - 1)];
    }
    double mp() const { return total_progress.empty() ? 0 : std::accumulate(total_progress.begin(), total_progress.end(), 0.0) / total_progress.size(); }
    double ms() const { return solve_times_ms.empty() ? 0 : std::accumulate(solve_times_ms.begin(), solve_times_ms.end(), 0.0) / solve_times_ms.size(); }

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

static void write_csv_row(std::ofstream& ofs, const Metrics& m, int n, const std::string& prefix = "") {
    auto [clo, chi] = m.cci();
    ofs << prefix << m.method << ","
        << std::fixed << std::setprecision(6)
        << m.cr() << "," << clo << "," << chi << ","
        << m.mr() << ","
        << std::setprecision(4) << m.mc() << "," << m.p5c() << ","
        << m.mp() << ","
        << std::setprecision(3) << m.ms() << "," << n << "\n";
    ofs.flush();
}

// ============================================================================
// Method setup
// ============================================================================

enum class Method { BASE, MODE_DRO_SAMPLE, MODE_DRO_INJ_K1, MODE_DRO_INJ_K2, TOPRISK_K1, TOPRISK_K2 };

static std::string mname(Method m) {
    switch (m) {
        case Method::BASE:            return "Base";
        case Method::MODE_DRO_SAMPLE: return "WDRO-sampling";
        case Method::MODE_DRO_INJ_K1: return "WDRO-inject-K1";
        case Method::MODE_DRO_INJ_K2: return "WDRO-inject-K2";
        case Method::TOPRISK_K1:      return "TopRisk-K1";
        case Method::TOPRISK_K2:      return "TopRisk-K2";
    }
    return "?";
}

static ExperimentConfig make_base_config(double switch_prob = 0.2, double rare_prob = 0.1) {
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
    cfg.num_obstacles = 1;
    cfg.obstacles_per_class = 1;
    return cfg;
}

static ExperimentConfig apply_method(ExperimentConfig cfg, Method m) {
    cfg.method_name = mname(m);
    switch (m) {
        case Method::BASE: break;
        case Method::MODE_DRO_SAMPLE:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::QSTAR_SAMPLE;
            break;
        case Method::MODE_DRO_INJ_K1:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::DRO;
            cfg.dro_injection_count = 1;
            break;
        case Method::MODE_DRO_INJ_K2:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::DRO;
            cfg.dro_injection_count = 2;
            break;
        case Method::TOPRISK_K1:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::TOP_RISK_INJECT;
            cfg.dro_injection_count = 1;
            break;
        case Method::TOPRISK_K2:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::TOP_RISK_INJECT;
            cfg.dro_injection_count = 2;
            break;
    }
    return cfg;
}

static const std::vector<Method> METHODS = {
    Method::BASE, Method::MODE_DRO_SAMPLE, Method::MODE_DRO_INJ_K1,
    Method::MODE_DRO_INJ_K2, Method::TOPRISK_K1, Method::TOPRISK_K2
};

static ObstacleState obstacle_on_path(const ReferencePath& path, double frac,
                                       std::mt19937& rng) {
    double obs_s = frac * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d tangent(std::cos(pp.heading), std::sin(pp.heading));
    std::uniform_real_distribution<double> lat(-0.3, 0.3), spd(-0.2, 0.1);
    Eigen::Vector2d pos = pp.position + lat(rng) * normal;
    double v = 0.8 + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -v * tangent.x(), -v * tangent.y());
}

// ============================================================================
// R1: Novel (Unseen) Mode — Swerving
// ============================================================================

static void run_r1() {
    std::cout << "\n================================================================\n";
    std::cout << "  R1: Novel Mode — Obstacle swerves (mode not in controller set)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    // Novel mode: at step 10, force the obstacle into "swerving" mode.
    // The controller doesn't know this mode exists.
    struct NovelModeSetup {
        std::string name;
        ModeModel novel_mode;
        int switch_step;
    };

    std::vector<NovelModeSetup> setups = {
        {"Spiral",     create_spiral_mode(DT),      10},
        {"HardBrake",  create_hard_brake_mode(DT),   10},
        {"U-turn",     create_uturn_mode(DT),        10},
        {"Skid",       create_skid_mode(DT),         10},
    };

    std::string filepath = OUTPUT_DIR + "r1_novel_mode.csv";
    std::ofstream ofs(filepath);
    ofs << "novel_mode,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    // Also run "no novel mode" baseline for reference
    for (const auto& setup : setups) {
        for (auto method : METHODS) {
            std::cout << "  " << setup.name << " / " << mname(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = mname(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(setup.name) / 1000 +
                    static_cast<int>(method) * 100000 + i + 100000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, 0.1);
                cfg = apply_method(cfg, method);
                cfg.path_following_obstacles = true;
                cfg.initial_obstacle_states = {obstacle_on_path(ref_path, 0.55, env_rng)};

                // Install a callback that forces the novel mode at the switch step.
                // We wrap the existing callback (if any) so traj-DRO still runs.
                auto existing_cb = cfg.step_callback;
                ModeModel novel = setup.novel_mode;
                int sw_step = setup.switch_step;

                cfg.step_callback = [existing_cb, novel, sw_step](
                    int step, int obs_id, ObstacleSim& obs_sim,
                    AdaptiveScenarioMPC& controller, std::mt19937& rng
                ) {
                    // At the switch step, inject the novel mode into the obstacle
                    if (step == sw_step && obs_id == 0) {
                        obs_sim.mode_models[novel.mode_id] = novel;
                        obs_sim.current_mode = novel.mode_id;
                        // Don't add to available_modes so it can't switch away
                    }
                    // Run existing callback (traj-DRO) if present
                    if (existing_cb) existing_cb(step, obs_id, obs_sim, controller, rng);
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << setup.name << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.cr() * 100) << "% clr=" << std::setprecision(2)
                      << met.mc() << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }

    // Also run normal (no novel mode) for comparison
    for (auto method : METHODS) {
        std::cout << "  No-novel / " << mname(method) << ": ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();
        Metrics met; met.method = mname(method);
        for (int i = 0; i < N_ROLLOUTS; ++i) {
            if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(static_cast<int>(method) * 100000 + i + 101000000);
            std::mt19937 env_rng(seed);
            ExperimentConfig cfg = make_base_config(0.2, 0.1);
            cfg = apply_method(cfg, method);
            cfg.path_following_obstacles = true;
            cfg.initial_obstacle_states = {obstacle_on_path(ref_path, 0.55, env_rng)};
            met.add(run_experiment_rollout(cfg, seed));
        }
        ofs << "No-novel,";
        write_csv_row(ofs, met, N_ROLLOUTS);
        std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1) << (met.cr()*100) << "%"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// R2: Mode Misclassification
// ============================================================================

static void run_r2() {
    std::cout << "\n================================================================\n";
    std::cout << "  R2: Mode Misclassification (similar modes confused)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    // Misclassification: the obstacle is actually doing "turn_left" but
    // the observer reports "constant_velocity" with some probability.
    // We test misclassification rates from 0% to 80%.
    const std::vector<double> MISCLASS_RATES = {0.0, 0.1, 0.2, 0.4, 0.6, 0.8};

    // Which modes get confused: turn_left reported as constant_velocity
    const std::string TRUE_MODE = "turn_left";
    const std::string CONFUSED_AS = "constant_velocity";

    std::string filepath = OUTPUT_DIR + "r2_misclassification.csv";
    std::ofstream ofs(filepath);
    ofs << "misclass_rate,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double mc_rate : MISCLASS_RATES) {
        for (auto method : METHODS) {
            std::cout << "  mc=" << std::setprecision(1) << (mc_rate*100) << "% / "
                      << mname(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = mname(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(mc_rate * 1000) * 100000 +
                    static_cast<int>(method) * 10000 + i + 102000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.15, 0.1);
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {obstacle_on_path(ref_path, 0.55, env_rng)};

                // Install misclassification callback: after the normal mode observation,
                // sometimes report the wrong mode to the controller.
                auto existing_cb = cfg.step_callback;
                cfg.step_callback = [existing_cb, mc_rate, TRUE_MODE, CONFUSED_AS](
                    int step, int obs_id, ObstacleSim& obs_sim,
                    AdaptiveScenarioMPC& controller, std::mt19937& rng
                ) {
                    // If the obstacle is actually in TRUE_MODE, with probability mc_rate
                    // tell the controller it's in CONFUSED_AS instead.
                    if (obs_id == 0 && obs_sim.current_mode == TRUE_MODE) {
                        std::uniform_real_distribution<double> u(0, 1);
                        if (u(rng) < mc_rate) {
                            // Overwrite the mode observation with the wrong mode
                            controller.update_mode_observation(0, 0, CONFUSED_AS, step + 10);
                        }
                    }
                    if (existing_cb) existing_cb(step, obs_id, obs_sim, controller, rng);
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(1) << mc_rate << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.cr()*100) << "% (" << std::setprecision(0)
                      << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// R3: Novel Mode Severity Sweep
// ============================================================================

static void run_r3() {
    std::cout << "\n================================================================\n";
    std::cout << "  R3: Novel Mode Severity Sweep (swerve amplitude)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);
    const std::vector<double> AMPLITUDES = {0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0};

    std::string filepath = OUTPUT_DIR + "r3_novel_severity.csv";
    std::ofstream ofs(filepath);
    ofs << "swerve_amplitude,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double amp : AMPLITUDES) {
        for (auto method : METHODS) {
            std::cout << "  amp=" << std::setprecision(1) << amp << " / "
                      << mname(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met; met.method = mname(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(amp * 1000) * 100000 +
                    static_cast<int>(method) * 10000 + i + 103000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, 0.1);
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {obstacle_on_path(ref_path, 0.55, env_rng)};

                if (amp > 0.01) {
                    auto existing_cb = cfg.step_callback;
                    ModeModel novel = create_swerve_mode(DT, amp);
                    cfg.step_callback = [existing_cb, novel](
                        int step, int obs_id, ObstacleSim& obs_sim,
                        AdaptiveScenarioMPC& controller, std::mt19937& rng
                    ) {
                        if (step == 10 && obs_id == 0) {
                            obs_sim.mode_models[novel.mode_id] = novel;
                            obs_sim.current_mode = novel.mode_id;
                        }
                        if (existing_cb) existing_cb(step, obs_id, obs_sim, controller, rng);
                    };
                }

                met.add(run_experiment_rollout(cfg, seed));
            }

            ofs << std::fixed << std::setprecision(1) << amp << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.cr()*100) << "% (" << std::setprecision(0)
                      << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// R4: Misclassification with Different Confused Mode Pairs
// ============================================================================

static void run_r4() {
    std::cout << "\n================================================================\n";
    std::cout << "  R4: Misclassification — Different Confused Pairs\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    struct ConfusionPair {
        std::string label;
        std::string true_mode;
        std::string confused_as;
    };

    std::vector<ConfusionPair> pairs = {
        {"turn_left->cv",       "turn_left",        "constant_velocity"},
        {"turn_left->turn_right", "turn_left",      "turn_right"},
        {"decel->cv",           "decelerating",     "constant_velocity"},
        {"lane_change->cv",     "lane_change_left", "constant_velocity"},
        {"lane_change->turn_left", "lane_change_left", "turn_left"},
    };

    const double MC_RATE = 0.5;  // 50% misclassification

    std::string filepath = OUTPUT_DIR + "r4_confusion_pairs.csv";
    std::ofstream ofs(filepath);
    ofs << "confusion_pair,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& pair : pairs) {
        for (auto method : METHODS) {
            std::cout << "  " << pair.label << " / " << mname(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met; met.method = mname(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(pair.label) / 1000 +
                    static_cast<int>(method) * 10000 + i + 104000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.15, 0.1);
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {obstacle_on_path(ref_path, 0.55, env_rng)};

                auto existing_cb = cfg.step_callback;
                std::string tm = pair.true_mode;
                std::string ca = pair.confused_as;
                cfg.step_callback = [existing_cb, tm, ca, MC_RATE](
                    int step, int obs_id, ObstacleSim& obs_sim,
                    AdaptiveScenarioMPC& controller, std::mt19937& rng
                ) {
                    if (obs_id == 0 && obs_sim.current_mode == tm) {
                        std::uniform_real_distribution<double> u(0, 1);
                        if (u(rng) < MC_RATE) {
                            controller.update_mode_observation(0, 0, ca, step + 10);
                        }
                    }
                    if (existing_cb) existing_cb(step, obs_id, obs_sim, controller, rng);
                };

                met.add(run_experiment_rollout(cfg, seed));
            }

            ofs << pair.label << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.cr()*100) << "% (" << std::setprecision(0)
                      << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// R5: Novel Mode + Multi-Obstacle
// ============================================================================

static void run_r5() {
    std::cout << "\n================================================================\n";
    std::cout << "  R5: Novel Mode + Multiple Obstacles\n";
    std::cout << "      (Sharp swerve on obs #0, normal modes on others)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    // Obstacle placement fractions along the path
    struct ObsSetup {
        std::string label;
        int num_obs;
        std::vector<double> fracs;
    };
    std::vector<ObsSetup> setups = {
        {"1-obs", 1, {0.55}},
        {"2-obs", 2, {0.40, 0.60}},
        {"3-obs", 3, {0.35, 0.50, 0.65}},
    };

    // Novel mode: sharp swerve injected on obstacle 0 at step 10
    ModeModel novel = create_sharp_swerve_mode(DT, 1.0);
    constexpr int SWITCH_STEP = 10;

    std::string filepath = OUTPUT_DIR + "r5_novel_multi_obstacle.csv";
    std::ofstream ofs(filepath);
    ofs << "num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& setup : setups) {
        for (auto method : METHODS) {
            std::cout << "  " << setup.label << " / " << mname(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = mname(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    setup.num_obs * 1000000 +
                    static_cast<int>(method) * 100000 + i + 105000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, 0.1);
                cfg = apply_method(cfg, method);
                cfg.num_obstacles = setup.num_obs;
                cfg.obstacles_per_class = 1;
                cfg.path_following_obstacles = true;

                // Place obstacles along the path
                std::vector<ObstacleState> obs_states;
                for (double frac : setup.fracs) {
                    obs_states.push_back(obstacle_on_path(ref_path, frac, env_rng));
                }
                cfg.initial_obstacle_states = obs_states;

                // Install callback: force obstacle #0 into sharp swerve at SWITCH_STEP
                auto existing_cb = cfg.step_callback;
                cfg.step_callback = [existing_cb, novel, SWITCH_STEP](
                    int step, int obs_id, ObstacleSim& obs_sim,
                    AdaptiveScenarioMPC& controller, std::mt19937& rng
                ) {
                    // Only inject novel mode on obstacle 0
                    if (step == SWITCH_STEP && obs_id == 0) {
                        obs_sim.mode_models[novel.mode_id] = novel;
                        obs_sim.current_mode = novel.mode_id;
                    }
                    if (existing_cb) existing_cb(step, obs_id, obs_sim, controller, rng);
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << setup.num_obs << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.cr() * 100) << "% clr=" << std::setprecision(2)
                      << met.mc() << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0) << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    fs::create_directories(OUTPUT_DIR);

    std::set<std::string> filters;
    for (int i = 1; i < argc; ++i) filters.insert(argv[i]);
    bool run_all = filters.empty();

    std::cout << "========================================\n";
    std::cout << "Robustness Experiments\n";
    std::cout << "  R1: Novel unseen modes (swerving)\n";
    std::cout << "  R2: Mode misclassification sweep\n";
    std::cout << "  R3: Novel mode severity sweep\n";
    std::cout << "  R4: Confusion pair comparison\n";
    std::cout << "  R5: Novel mode + multi-obstacle\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("r1")) run_r1();
    if (run_all || filters.count("r2")) run_r2();
    if (run_all || filters.count("r3")) run_r3();
    if (run_all || filters.count("r4")) run_r4();
    if (run_all || filters.count("r5")) run_r5();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total << "s"
              << " (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";

    return 0;
}
