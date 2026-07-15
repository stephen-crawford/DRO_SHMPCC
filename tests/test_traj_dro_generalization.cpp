/**
 * @file test_traj_dro_generalization.cpp
 * @brief Generalization experiments comparing mode-level DRO vs trajectory-level DRO.
 *
 * Mirrors the G1–G5 generalization experiments but adds trajectory-level DRO methods:
 *   T1: Path geometry comparison (mode DRO vs traj DRO across path types)
 *   T2: Multi-obstacle scaling (1, 2, 3 obstacles)
 *   T3: Switching dynamics sweep (switch_prob 0.05 to 0.50)
 *   T4: Wasserstein radius sweep (rho = 0.01 to 0.5)
 *   T5: Path × obstacle interaction grid
 *
 * Methods compared:
 *   - Base (no DRO)
 *   - Mode-DRO(q*): resample scenarios from mode-level q*
 *   - Mode-DRO(inj): inject worst-case mode
 *   - Traj-DRO(q*): trajectory-level DRO, resample from traj q*
 *   - Traj-DRO(inj): trajectory-level DRO, inject worst-case trajectory
 *   - Traj-DRO(comb): trajectory-level DRO, resample + inject
 *
 * Output: figures/traj_dro/
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

static const std::string OUTPUT_DIR = "figures/traj_dro/";

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
// Trajectory-Level DRO Implementation (from test_trajectory_dro.cpp)
// ============================================================================

static double compute_trajectory_risk(
    const ObstacleTrajectory& traj,
    const std::vector<EgoState>& ego_ref,
    double safety_radius,
    int num_discs,
    double vehicle_length,
    double z_alpha = 1.6449
) {
    double max_risk = 0.0;
    int H = std::min(static_cast<int>(traj.steps.size()),
                     static_cast<int>(ego_ref.size()));

    for (int k = 1; k < H; ++k) {
        const auto& step = traj.steps[k];
        const auto& ego = ego_ref[k];

        for (int d = 0; d < num_discs; ++d) {
            Eigen::Vector2d disc_pos;
            if (num_discs > 1) {
                double theta = ego.theta;
                Eigen::Vector2d dir(std::cos(theta), std::sin(theta));
                double step_offset = vehicle_length / (num_discs - 1);
                double offset = -vehicle_length / 2.0 + d * step_offset;
                disc_pos = ego.position() + offset * dir;
            } else {
                disc_pos = ego.position();
            }

            Eigen::Vector2d diff = step.mean - disc_pos;
            double dist = diff.norm();
            if (dist < 1e-12) {
                max_risk = std::max(max_risk, safety_radius);
                continue;
            }

            Eigen::Vector2d n_dir = diff / dist;
            double var_dir = static_cast<double>(
                n_dir.transpose() * step.covariance * n_dir);
            double sigma_dir = std::sqrt(std::max(1e-12, var_dir));

            double risk = std::max(0.0,
                safety_radius + z_alpha * sigma_dir - dist);
            max_risk = std::max(max_risk, risk);
        }
    }
    return max_risk;
}

static std::vector<std::vector<double>> compute_trajectory_transport_costs(
    const std::vector<ObstacleTrajectory>& trajectories
) {
    int S = static_cast<int>(trajectories.size());
    std::vector<std::vector<double>> D(S, std::vector<double>(S, 0.0));

    for (int i = 0; i < S; ++i) {
        for (int j = i + 1; j < S; ++j) {
            int H = std::min(static_cast<int>(trajectories[i].steps.size()),
                             static_cast<int>(trajectories[j].steps.size()));
            double total = 0.0;
            for (int k = 0; k < H; ++k) {
                Eigen::Vector2d diff = trajectories[i].steps[k].mean
                                     - trajectories[j].steps[k].mean;
                total += diff.squaredNorm();
            }
            double cost = (H > 0) ? total / H : 0.0;
            D[i][j] = cost;
            D[j][i] = cost;
        }
    }
    return D;
}

struct TrajectoryDROResult {
    std::vector<double> weights;
    double optimal_lambda = 0.0;
    double worst_case_risk = 0.0;
    double rho_used = 0.0;
    int worst_trajectory_idx = -1;
};

static TrajectoryDROResult solve_trajectory_dro(
    const std::vector<double>& risks,
    const std::vector<std::vector<double>>& D,
    double rho
) {
    int S = static_cast<int>(risks.size());
    TrajectoryDROResult result;
    result.rho_used = rho;

    if (S == 0) return result;

    double w_nom = 1.0 / S;
    double max_risk = *std::max_element(risks.begin(), risks.end());
    double min_risk = *std::min_element(risks.begin(), risks.end());

    if (max_risk < 1e-12 || std::abs(max_risk - min_risk) < 1e-12) {
        result.weights.assign(S, w_nom);
        result.worst_case_risk = max_risk;
        result.worst_trajectory_idx = static_cast<int>(
            std::max_element(risks.begin(), risks.end()) - risks.begin());
        return result;
    }

    double max_D = 0.0;
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j)
            max_D = std::max(max_D, D[i][j]);
    if (max_D < 1e-12) max_D = 1.0;

    double lambda_max = (max_risk - min_risk) / (max_D > 1e-12 ? max_D * 0.01 : 1.0);
    lambda_max = std::max(lambda_max, 10.0);

    auto eval_dual = [&](double lambda) -> double {
        double obj = lambda * rho;
        for (int i = 0; i < S; ++i) {
            double max_val = -1e18;
            for (int j = 0; j < S; ++j) {
                double val = risks[j] - lambda * D[i][j];
                max_val = std::max(max_val, val);
            }
            obj += w_nom * max_val;
        }
        return obj;
    };

    auto build_plan = [&](double lambda) -> std::vector<double> {
        std::vector<double> q(S, 0.0);
        for (int i = 0; i < S; ++i) {
            int best_j = 0;
            double best_val = risks[0] - lambda * D[i][0];
            for (int j = 1; j < S; ++j) {
                double val = risks[j] - lambda * D[i][j];
                if (val > best_val) {
                    best_val = val;
                    best_j = j;
                }
            }
            q[best_j] += w_nom;
        }
        return q;
    };

    auto plan_cost = [&](double lambda) -> double {
        double cost = 0.0;
        for (int i = 0; i < S; ++i) {
            int best_j = 0;
            double best_val = risks[0] - lambda * D[i][0];
            for (int j = 1; j < S; ++j) {
                double val = risks[j] - lambda * D[i][j];
                if (val > best_val) {
                    best_val = val;
                    best_j = j;
                }
            }
            cost += w_nom * D[i][best_j];
        }
        return cost;
    };

    double lo = 0.0, hi = lambda_max;
    double best_lambda = 0.0;
    double best_dual = eval_dual(0.0);

    for (int iter = 0; iter < 50; ++iter) {
        double mid = (lo + hi) / 2.0;
        double dual_val = eval_dual(mid);
        if (dual_val < best_dual) {
            best_dual = dual_val;
            best_lambda = mid;
        }
        double tc = plan_cost(mid);
        if (tc > rho) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    result.optimal_lambda = best_lambda;
    result.worst_case_risk = best_dual;
    result.weights = build_plan(best_lambda);

    int best_idx = 0;
    double best_w = result.weights[0];
    for (int i = 1; i < S; ++i) {
        if (result.weights[i] > best_w) {
            best_w = result.weights[i];
            best_idx = i;
        }
    }
    result.worst_trajectory_idx = best_idx;

    return result;
}

// ============================================================================
// Metrics
// ============================================================================

struct Metrics {
    std::string method;
    std::vector<bool> collisions;
    std::vector<int> missed_mode_steps;
    std::vector<int> total_mode_checks;
    std::vector<int> rare_mode_active;
    std::vector<int> rare_mode_missed;
    std::vector<double> min_clearances;
    std::vector<double> total_progress;
    std::vector<double> solve_times_ms;

    int n() const { return static_cast<int>(collisions.size()); }
    int coll_count() const { int c = 0; for (bool b : collisions) if (b) c++; return c; }
    double coll_rate() const { return n() > 0 ? double(coll_count()) / n() : 0; }
    std::pair<double,double> coll_ci() const { return wilson_ci(coll_count(), n()); }

    int tot_missed() const { return std::accumulate(missed_mode_steps.begin(), missed_mode_steps.end(), 0); }
    int tot_checks() const { return std::accumulate(total_mode_checks.begin(), total_mode_checks.end(), 0); }
    double mm_rate() const { int tc = tot_checks(); return tc > 0 ? double(tot_missed()) / tc : 0; }
    std::pair<double,double> mm_ci() const { return wilson_ci(tot_missed(), tot_checks()); }

    int tot_rare_act() const { return std::accumulate(rare_mode_active.begin(), rare_mode_active.end(), 0); }
    int tot_rare_miss() const { return std::accumulate(rare_mode_missed.begin(), rare_mode_missed.end(), 0); }
    double rare_rate() const { int a = tot_rare_act(); return a > 0 ? double(tot_rare_miss()) / a : 0; }
    std::pair<double,double> rare_ci() const { return wilson_ci(tot_rare_miss(), tot_rare_act()); }

    double mean_clearance() const {
        if (min_clearances.empty()) return 0;
        return std::accumulate(min_clearances.begin(), min_clearances.end(), 0.0) / min_clearances.size();
    }
    double p5_clearance() const {
        if (min_clearances.empty()) return 0;
        auto v = min_clearances;
        std::sort(v.begin(), v.end());
        return v[std::max(0, (int)(0.05 * (v.size() - 1)))];
    }
    double mean_progress() const {
        if (total_progress.empty()) return 0;
        return std::accumulate(total_progress.begin(), total_progress.end(), 0.0) / total_progress.size();
    }
    double mean_solve_ms() const {
        if (solve_times_ms.empty()) return 0;
        return std::accumulate(solve_times_ms.begin(), solve_times_ms.end(), 0.0) / solve_times_ms.size();
    }

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

// ============================================================================
// Helpers
// ============================================================================

static double elapsed_sec(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

enum class TrajDROMethod {
    SHMPCC_BASE,
    MODE_DRO_SAMPLE,
    MODE_DRO_INJECT,
    TRAJ_DRO_RESAMPLE,
    TRAJ_DRO_INJECT,
    TRAJ_DRO_COMBINED
};

static std::string method_name(TrajDROMethod m) {
    switch (m) {
        case TrajDROMethod::SHMPCC_BASE:       return "Base";
        case TrajDROMethod::MODE_DRO_SAMPLE:   return "Mode-DRO(q*)";
        case TrajDROMethod::MODE_DRO_INJECT:   return "Mode-DRO(inj)";
        case TrajDROMethod::TRAJ_DRO_RESAMPLE: return "Traj-DRO(q*)";
        case TrajDROMethod::TRAJ_DRO_INJECT:   return "Traj-DRO(inj)";
        case TrajDROMethod::TRAJ_DRO_COMBINED: return "Traj-DRO(comb)";
    }
    return "?";
}

/// Build trajectory-DRO step callback
static std::function<void(int, int, ObstacleSim&, AdaptiveScenarioMPC&, std::mt19937&)>
make_traj_dro_callback(TrajDROMethod method, double rho, int num_scenarios) {
    return [method, rho, num_scenarios](
        int step, int obs_id, ObstacleSim& obs_sim,
        AdaptiveScenarioMPC& controller, std::mt19937& rng
    ) {
        if (obs_id != 0) return;

        const auto& modes = obs_sim.mode_models;
        int horizon = controller.config().horizon;
        double ego_r = controller.config().ego_radius;
        double obs_r = controller.config().obstacle_radius;
        double margin = controller.config().safety_margin;
        double safety_radius = ego_r + obs_r + margin;
        int n_discs = controller.config().num_discs;
        double veh_len = controller.config().vehicle_length;

        int S_particles = num_scenarios;
        std::vector<ObstacleTrajectory> traj_particles;
        traj_particles.reserve(S_particles);

        // Build mode weights from scenario frequency
        std::map<std::string, double> freq_weights;
        int total_obs_count = 0;
        for (const auto& [mode_id, _] : modes) {
            freq_weights[mode_id] = 0.0;
        }
        const auto& prev_scenarios = controller.scenarios();
        if (!prev_scenarios.empty()) {
            for (const auto& sc : prev_scenarios) {
                auto it = sc.trajectories.find(0);
                if (it != sc.trajectories.end()) {
                    freq_weights[it->second.mode_id] += 1.0;
                    total_obs_count++;
                }
            }
            if (total_obs_count > 0) {
                for (auto& [_, w] : freq_weights) w /= total_obs_count;
            }
        }
        if (total_obs_count == 0) {
            for (auto& [_, w] : freq_weights) {
                w = 1.0 / freq_weights.size();
            }
        }

        // Sample trajectory particles
        std::normal_distribution<double> normal_dist(0.0, 1.0);
        for (int s = 0; s < S_particles; ++s) {
            std::string sampled_mode = sample_mode_from_weights(freq_weights, rng);
            if (modes.find(sampled_mode) == modes.end()) continue;
            const ModeModel& mode = modes.at(sampled_mode);

            std::vector<PredictionStep> steps;
            steps.reserve(horizon + 1);

            Eigen::Vector4d x = obs_sim.state.to_array();
            Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
            steps.emplace_back(0, x.head<2>(), cov.block<2,2>(0,0));

            for (int k = 0; k < horizon; ++k) {
                Eigen::VectorXd noise(mode.noise_dim());
                for (int d = 0; d < mode.noise_dim(); ++d) {
                    noise(d) = normal_dist(rng);
                }
                x = mode.A * x + mode.b + mode.G * noise;
                cov = mode.A * cov * mode.A.transpose()
                    + mode.G * mode.G.transpose();
                steps.emplace_back(k + 1, x.head<2>(), cov.block<2,2>(0,0));
            }

            ObstacleTrajectory traj(0, sampled_mode, steps, freq_weights[sampled_mode]);
            traj_particles.push_back(std::move(traj));
        }

        if (static_cast<int>(traj_particles.size()) < 3) return;

        // Approximate ego reference trajectory
        Eigen::Vector2d obs_pos = obs_sim.state.position();
        double heading = std::atan2(obs_pos.y(), obs_pos.x());
        std::vector<EgoState> ego_ref;
        ego_ref.reserve(horizon + 1);
        for (int k = 0; k <= horizon; ++k) {
            EgoState es;
            es.x = 1.5 * k * 0.1 * std::cos(heading);
            es.y = 1.5 * k * 0.1 * std::sin(heading);
            es.theta = heading;
            es.v = 1.5;
            ego_ref.push_back(es);
        }

        // Compute per-trajectory risk
        int S = static_cast<int>(traj_particles.size());
        std::vector<double> risks(S);
        for (int s = 0; s < S; ++s) {
            risks[s] = compute_trajectory_risk(
                traj_particles[s], ego_ref, safety_radius, n_discs, veh_len);
        }

        // Compute transport costs and solve trajectory DRO
        auto D = compute_trajectory_transport_costs(traj_particles);
        auto dro_result = solve_trajectory_dro(risks, D, rho);
        if (dro_result.worst_trajectory_idx < 0) return;

        // Apply result
        if (method == TrajDROMethod::TRAJ_DRO_INJECT ||
            method == TrajDROMethod::TRAJ_DRO_COMBINED) {
            int worst_idx = dro_result.worst_trajectory_idx;
            const auto& worst_traj = traj_particles[worst_idx];

            Scenario inj_scenario;
            inj_scenario.scenario_id = num_scenarios + 100 + step;
            inj_scenario.is_injected = true;
            inj_scenario.probability = 1.0;
            inj_scenario.trajectories[0] = worst_traj;
            inj_scenario.trajectories[0].probability = 1.0;

            controller.inject_scenario(inj_scenario);
        }

        if (method == TrajDROMethod::TRAJ_DRO_RESAMPLE ||
            method == TrajDROMethod::TRAJ_DRO_COMBINED) {
            std::map<std::string, double> mode_weights;
            for (const auto& [mode_id, _] : modes) {
                mode_weights[mode_id] = 0.0;
            }
            for (int s = 0; s < S; ++s) {
                const std::string& mode = traj_particles[s].mode_id;
                mode_weights[mode] += dro_result.weights[s];
            }
            double sum_w = 0.0;
            for (auto& [_, w] : mode_weights) sum_w += w;
            if (sum_w > 1e-12) {
                for (auto& [_, w] : mode_weights) w /= sum_w;
            }

            controller.reset_scenarios();
            controller.set_custom_mode_weights(0, mode_weights);
        }
    };
}

static ExperimentConfig make_base_config(double switch_prob = 0.2, double rare_prob = 0.1,
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

static ExperimentConfig apply_method(ExperimentConfig cfg, TrajDROMethod method,
                                      double rho = 0.1) {
    cfg.method_name = method_name(method);

    switch (method) {
        case TrajDROMethod::SHMPCC_BASE:
            break;
        case TrajDROMethod::MODE_DRO_SAMPLE:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::QSTAR_SAMPLE;
            break;
        case TrajDROMethod::MODE_DRO_INJECT:
            cfg.enable_dro = true;
            cfg.injection_mode = InjectionMode::DRO;
            cfg.dro_injection_count = 1;
            break;
        case TrajDROMethod::TRAJ_DRO_RESAMPLE:
        case TrajDROMethod::TRAJ_DRO_INJECT:
        case TrajDROMethod::TRAJ_DRO_COMBINED:
            cfg.step_callback = make_traj_dro_callback(method, rho, NUM_SCENARIOS);
            break;
    }

    return cfg;
}

static void write_csv_row(std::ofstream& ofs, const Metrics& met, int n_rollouts,
                           const std::string& extra_prefix = "") {
    auto [clo, chi] = met.coll_ci();
    auto [mlo, mhi] = met.mm_ci();
    auto [rlo, rhi] = met.rare_ci();
    ofs << extra_prefix
        << met.method << ","
        << std::fixed << std::setprecision(6)
        << met.coll_rate() << "," << clo << "," << chi << ","
        << met.mm_rate() << "," << mlo << "," << mhi << ","
        << met.rare_rate() << "," << rlo << "," << rhi << ","
        << std::setprecision(4) << met.mean_clearance() << ","
        << met.p5_clearance() << "," << met.mean_progress() << ","
        << std::setprecision(3) << met.mean_solve_ms() << ","
        << n_rollouts << "\n";
    ofs.flush();
}

// ============================================================================
// Path/obstacle helpers (same as test_generalization.cpp)
// ============================================================================

struct PathSetup {
    std::string name;
    ReferencePath path;
    EgoState initial_ego;
};

static PathSetup make_straight_path() {
    auto path = ReferencePath::create_straight(
        Eigen::Vector2d(0, 0), Eigen::Vector2d(25, 0), 200);
    return {"Straight", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_standard_scurve() {
    auto path = ReferencePath::create_s_curve(25.0, 3.0, 200);
    return {"S-curve", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_tight_scurve() {
    auto path = ReferencePath::create_s_curve(20.0, 5.0, 200);
    return {"Tight-S", path, EgoState(0.0, 0.0, 0.0, 1.5)};
}

static PathSetup make_circle_path() {
    double radius = 10.0;
    auto path = ReferencePath::create_circle(
        Eigen::Vector2d(0, -radius), radius, M_PI / 2, 0.0, 200);
    Eigen::Vector2d start_pos = path.get_position_at(0.0);
    double start_heading = path.get_heading_at(0.0);
    return {"Circle", path, EgoState(start_pos.x(), start_pos.y(), start_heading, 1.5)};
}

static ObstacleState obstacle_on_path(const ReferencePath& path, double arc_fraction,
                                       std::mt19937& rng, bool oncoming = true) {
    double obs_s = arc_fraction * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d tangent(std::cos(pp.heading), std::sin(pp.heading));

    std::uniform_real_distribution<double> lat(-0.3, 0.3);
    std::uniform_real_distribution<double> spd(-0.2, 0.1);
    Eigen::Vector2d pos = pp.position + lat(rng) * normal;
    double v = 0.8 + spd(rng);
    double dir = oncoming ? -1.0 : 1.0;
    return ObstacleState(pos.x(), pos.y(), dir * v * tangent.x(), dir * v * tangent.y());
}

// ============================================================================
// Core method list
// ============================================================================

static const std::vector<TrajDROMethod> ALL_METHODS = {
    TrajDROMethod::SHMPCC_BASE,
    TrajDROMethod::MODE_DRO_SAMPLE,
    TrajDROMethod::MODE_DRO_INJECT,
    TrajDROMethod::TRAJ_DRO_RESAMPLE,
    TrajDROMethod::TRAJ_DRO_INJECT,
    TrajDROMethod::TRAJ_DRO_COMBINED
};

// Compact subset for large sweeps
static const std::vector<TrajDROMethod> COMPACT_METHODS = {
    TrajDROMethod::SHMPCC_BASE,
    TrajDROMethod::MODE_DRO_INJECT,
    TrajDROMethod::TRAJ_DRO_INJECT,
    TrajDROMethod::TRAJ_DRO_COMBINED
};

// ============================================================================
// T1: Path Geometry Comparison
// ============================================================================

static void run_t1() {
    std::cout << "\n================================================================\n";
    std::cout << "  T1: Path Geometry — Mode DRO vs Traj DRO\n";
    std::cout << "  Paths: Straight, S-curve, Tight-S, Circle\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    std::vector<PathSetup> paths = {
        make_straight_path(),
        make_standard_scurve(),
        make_tight_scurve(),
        make_circle_path(),
    };

    std::string filepath = OUTPUT_DIR + "t1_path_geometry.csv";
    std::ofstream ofs(filepath);
    ofs << "path,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& ps : paths) {
        for (auto method : ALL_METHODS) {
            std::cout << "  " << ps.name << " / " << method_name(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(ps.name) / 1000 +
                    static_cast<int>(method) * 100000 + i + 60000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);
                cfg.custom_ref_path = ps.path;
                cfg.custom_initial_ego = ps.initial_ego;
                cfg.initial_obstacle_states = {
                    obstacle_on_path(ps.path, 0.55, env_rng, true)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << ps.name << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// T2: Multi-Obstacle Scaling
// ============================================================================

static void run_t2() {
    std::cout << "\n================================================================\n";
    std::cout << "  T2: Multi-Obstacle Scaling (1, 2, 3 obstacles)\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> OBS_COUNTS = {1, 2, 3};
    const std::vector<std::vector<double>> OBS_FRACS = {
        {0.50}, {0.40, 0.60}, {0.30, 0.45, 0.65},
    };

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::string filepath = OUTPUT_DIR + "t2_multi_obstacle.csv";
    std::ofstream ofs(filepath);
    ofs << "num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (size_t oi = 0; oi < OBS_COUNTS.size(); ++oi) {
        int n_obs = OBS_COUNTS[oi];
        const auto& fracs = OBS_FRACS[oi];

        for (auto method : ALL_METHODS) {
            std::cout << "  " << n_obs << " obs / " << method_name(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(n_obs * 1000000 +
                    static_cast<int>(method) * 100000 + i + 70000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, 0.1, n_obs);
                cfg = apply_method(cfg, method);

                std::vector<ObstacleState> obs_states;
                for (double frac : fracs) {
                    obs_states.push_back(obstacle_on_path(ref_path, frac, env_rng, true));
                }
                cfg.initial_obstacle_states = obs_states;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << n_obs << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.coll_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// T3: Switching Dynamics Sweep
// ============================================================================

static void run_t3() {
    std::cout << "\n================================================================\n";
    std::cout << "  T3: Switching Dynamics Sweep\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> SWITCH_PROBS = {0.05, 0.10, 0.20, 0.35, 0.50};
    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::string filepath = OUTPUT_DIR + "t3_switch_dynamics.csv";
    std::ofstream ofs(filepath);
    ofs << "switch_prob,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double sp : SWITCH_PROBS) {
        for (auto method : COMPACT_METHODS) {
            std::cout << "  sp=" << std::setprecision(2) << sp << " / "
                      << method_name(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(sp * 1000) * 1000000 +
                    static_cast<int>(method) * 100000 + i + 80000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(sp, 0.1);
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {
                    obstacle_on_path(ref_path, 0.55, env_rng, true)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(2) << sp << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.coll_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// T4: Wasserstein Radius Sweep
// ============================================================================

static void run_t4() {
    std::cout << "\n================================================================\n";
    std::cout << "  T4: Wasserstein Radius (rho) Sweep\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> RHOS = {0.01, 0.05, 0.1, 0.2, 0.5};
    const std::vector<TrajDROMethod> METHODS = {
        TrajDROMethod::MODE_DRO_INJECT,
        TrajDROMethod::TRAJ_DRO_INJECT,
        TrajDROMethod::TRAJ_DRO_COMBINED
    };

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::string filepath = OUTPUT_DIR + "t4_rho_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "rho,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    // Baseline (rho-independent)
    {
        std::cout << "  Base (baseline): ";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();
        Metrics met;
        met.method = "Base";
        for (int i = 0; i < N_ROLLOUTS; ++i) {
            if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
            unsigned seed = static_cast<unsigned>(i + 90000000);
            std::mt19937 env_rng(seed);
            ExperimentConfig cfg = make_base_config();
            cfg = apply_method(cfg, TrajDROMethod::SHMPCC_BASE);
            cfg.initial_obstacle_states = {obstacle_on_path(ref_path, 0.55, env_rng, true)};
            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }
        ofs << "0.00,";
        write_csv_row(ofs, met, N_ROLLOUTS);
        std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                  << (met.coll_rate() * 100) << "%"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    for (double rho : RHOS) {
        for (auto method : METHODS) {
            std::cout << "  rho=" << std::fixed << std::setprecision(2) << rho
                      << " / " << method_name(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(rho * 1000) * 1000000 +
                    static_cast<int>(method) * 100000 + i + 91000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method, rho);
                cfg.initial_obstacle_states = {
                    obstacle_on_path(ref_path, 0.55, env_rng, true)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(2) << rho << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.coll_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// T5: Path × Obstacle Interaction Grid
// ============================================================================

static void run_t5() {
    std::cout << "\n================================================================\n";
    std::cout << "  T5: Path x Obstacle Interaction\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    std::vector<PathSetup> paths = {
        make_straight_path(),
        make_standard_scurve(),
        make_tight_scurve(),
        make_circle_path(),
    };

    const std::vector<int> OBS_COUNTS = {1, 2, 3};
    const std::vector<std::vector<double>> OBS_FRACS = {
        {0.50}, {0.40, 0.60}, {0.30, 0.45, 0.65},
    };

    std::string filepath = OUTPUT_DIR + "t5_path_obstacle_grid.csv";
    std::ofstream ofs(filepath);
    ofs << "path,num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& ps : paths) {
        for (size_t oi = 0; oi < OBS_COUNTS.size(); ++oi) {
            int n_obs = OBS_COUNTS[oi];
            const auto& fracs = OBS_FRACS[oi];

            for (auto method : COMPACT_METHODS) {
                std::cout << "  " << ps.name << "/" << n_obs << "obs / "
                          << method_name(method) << ": ";
                std::cout.flush();
                auto t1 = std::chrono::steady_clock::now();

                Metrics met;
                met.method = method_name(method);

                for (int i = 0; i < N_ROLLOUTS; ++i) {
                    if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                    unsigned seed = static_cast<unsigned>(
                        std::hash<std::string>{}(ps.name) / 1000 +
                        n_obs * 1000000 +
                        static_cast<int>(method) * 100000 + i + 95000000);
                    std::mt19937 env_rng(seed);

                    ExperimentConfig cfg = make_base_config(0.2, 0.1, n_obs);
                    cfg = apply_method(cfg, method);
                    cfg.custom_ref_path = ps.path;
                    cfg.custom_initial_ego = ps.initial_ego;

                    std::vector<ObstacleState> obs_states;
                    for (double frac : fracs) {
                        obs_states.push_back(obstacle_on_path(ps.path, frac, env_rng, true));
                    }
                    cfg.initial_obstacle_states = obs_states;

                    RolloutRecord rec = run_experiment_rollout(cfg, seed);
                    met.add(rec);
                }

                ofs << ps.name << "," << n_obs << ",";
                write_csv_row(ofs, met, N_ROLLOUTS);

                std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                          << (met.coll_rate() * 100) << "%"
                          << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
            }
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// Obstacle placement helpers for challenging environments
// ============================================================================

static ObstacleState obstacle_high_speed(const ReferencePath& path, double arc_fraction,
                                          std::mt19937& rng, double speed = 1.8) {
    double obs_s = arc_fraction * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d tangent(std::cos(pp.heading), std::sin(pp.heading));

    std::uniform_real_distribution<double> lat(-0.3, 0.3);
    std::uniform_real_distribution<double> spd(-0.1, 0.1);
    Eigen::Vector2d pos = pp.position + lat(rng) * normal;
    double v = speed + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -v * tangent.x(), -v * tangent.y());
}

static ObstacleState obstacle_crossing(const ReferencePath& path, double arc_fraction,
                                        std::mt19937& rng, double speed = 1.0) {
    double obs_s = arc_fraction * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));

    std::uniform_real_distribution<double> offset(1.5, 3.0);
    std::uniform_real_distribution<double> spd(-0.1, 0.1);
    double side = (rng() % 2 == 0) ? 1.0 : -1.0;
    Eigen::Vector2d pos = pp.position + side * offset(rng) * normal;
    double v = speed + spd(rng);
    return ObstacleState(pos.x(), pos.y(), -side * v * normal.x(), -side * v * normal.y());
}

static ObstacleState obstacle_diagonal(const ReferencePath& path, double arc_fraction,
                                        std::mt19937& rng, double speed = 1.0) {
    double obs_s = arc_fraction * path.total_length();
    PathPoint pp = path.get_point_at(obs_s);
    Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
    Eigen::Vector2d tangent(std::cos(pp.heading), std::sin(pp.heading));

    std::uniform_real_distribution<double> lat(1.0, 2.0);
    std::uniform_real_distribution<double> spd(-0.1, 0.1);
    double side = (rng() % 2 == 0) ? 1.0 : -1.0;
    Eigen::Vector2d pos = pp.position + side * lat(rng) * normal + 2.0 * tangent;
    double v = speed + spd(rng);
    Eigen::Vector2d vel_dir = (-tangent + (-side) * normal).normalized();
    return ObstacleState(pos.x(), pos.y(), v * vel_dir.x(), v * vel_dir.y());
}

// ============================================================================
// T6: Challenging Environment Variants (Mode DRO vs Traj DRO)
// ============================================================================

struct ChallengeEnv {
    std::string name;
    std::function<std::vector<ObstacleState>(const ReferencePath&, std::mt19937&)> make_obstacles;
    int num_obstacles;
};

static void run_t6() {
    std::cout << "\n================================================================\n";
    std::cout << "  T6: Challenging Environments — Mode DRO vs Traj DRO\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::vector<ChallengeEnv> envs = {
        {"HighSpeed-1.5",
         [](const ReferencePath& p, std::mt19937& rng) -> std::vector<ObstacleState> {
             return {obstacle_high_speed(p, 0.55, rng, 1.5)};
         }, 1},
        {"HighSpeed-2.0",
         [](const ReferencePath& p, std::mt19937& rng) -> std::vector<ObstacleState> {
             return {obstacle_high_speed(p, 0.55, rng, 2.0)};
         }, 1},
        {"Crossing",
         [](const ReferencePath& p, std::mt19937& rng) -> std::vector<ObstacleState> {
             return {obstacle_crossing(p, 0.50, rng, 1.0)};
         }, 1},
        {"Diagonal",
         [](const ReferencePath& p, std::mt19937& rng) -> std::vector<ObstacleState> {
             return {obstacle_diagonal(p, 0.50, rng, 1.0)};
         }, 1},
        {"Mixed-3obs",
         [](const ReferencePath& p, std::mt19937& rng) -> std::vector<ObstacleState> {
             return {obstacle_high_speed(p, 0.40, rng, 1.5),
                     obstacle_on_path(p, 0.55, rng, true),
                     obstacle_crossing(p, 0.60, rng, 0.8)};
         }, 3},
        {"HighSpeed-Dense",
         [](const ReferencePath& p, std::mt19937& rng) -> std::vector<ObstacleState> {
             return {obstacle_high_speed(p, 0.35, rng, 1.5),
                     obstacle_high_speed(p, 0.50, rng, 1.8),
                     obstacle_high_speed(p, 0.65, rng, 1.5)};
         }, 3},
    };

    std::string filepath = OUTPUT_DIR + "t6_challenging_envs.csv";
    std::ofstream ofs(filepath);
    ofs << "challenge,num_obstacles,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (const auto& env : envs) {
        for (auto method : ALL_METHODS) {
            std::cout << "  " << env.name << " / " << method_name(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    std::hash<std::string>{}(env.name) / 1000 +
                    static_cast<int>(method) * 100000 + i + 96000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config(0.2, 0.1, env.num_obstacles);
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = env.make_obstacles(ref_path, env_rng);

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << env.name << "," << env.num_obstacles << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
}

// ============================================================================
// T7: Obstacle Speed Sweep (Mode DRO vs Traj DRO)
// ============================================================================

static void run_t7() {
    std::cout << "\n================================================================\n";
    std::cout << "  T7: Speed Sweep — Mode DRO vs Traj DRO\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> SPEEDS = {0.5, 0.8, 1.0, 1.3, 1.5, 2.0};
    auto ref_path = ReferencePath::create_s_curve(25.0, 3.0, 200);

    std::string filepath = OUTPUT_DIR + "t7_speed_sweep.csv";
    std::ofstream ofs(filepath);
    ofs << "obstacle_speed,method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,mean_solve_ms,n_rollouts\n";

    for (double speed : SPEEDS) {
        for (auto method : COMPACT_METHODS) {
            std::cout << "  v=" << std::fixed << std::setprecision(1) << speed
                      << " / " << method_name(method) << ": ";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                if (i > 0 && i % 100 == 0) { std::cout << i << " "; std::cout.flush(); }
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(speed * 100) * 1000000 +
                    static_cast<int>(method) * 100000 + i + 97000000);
                std::mt19937 env_rng(seed);

                ExperimentConfig cfg = make_base_config();
                cfg = apply_method(cfg, method);
                cfg.initial_obstacle_states = {
                    obstacle_high_speed(ref_path, 0.55, env_rng, speed)
                };

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            ofs << std::fixed << std::setprecision(1) << speed << ",";
            write_csv_row(ofs, met, N_ROLLOUTS);

            std::cout << N_ROLLOUTS << " coll=" << std::setprecision(1)
                      << (met.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }
    ofs.close();
    std::cout << "  Written: " << filepath << " (" << std::setprecision(0)
              << elapsed_sec(t0) << "s)\n";
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
    std::cout << "Trajectory DRO Generalization Experiments\n";
    std::cout << "========================================\n";
    std::cout << "Methods:\n";
    for (auto m : ALL_METHODS) {
        std::cout << "  " << method_name(m) << "\n";
    }
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("t1")) run_t1();
    if (run_all || filters.count("t2")) run_t2();
    if (run_all || filters.count("t3")) run_t3();
    if (run_all || filters.count("t4")) run_t4();
    if (run_all || filters.count("t5")) run_t5();
    if (run_all || filters.count("t6")) run_t6();
    if (run_all || filters.count("t7")) run_t7();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total << "s"
              << " (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";

    return 0;
}
