/**
 * @file test_trajectory_dro.cpp
 * @brief Trajectory-level W2 DRO vs SHMPCC nominal baseline comparison.
 *
 * Instead of operating on discrete mode weights (the standard approach),
 * this test applies Wasserstein DRO directly to a pool of sampled obstacle
 * trajectories:
 *
 *   1. Sample S_pool candidate trajectories from the nominal dynamics
 *   2. Compute pairwise W2 (Bures) distances between trajectories
 *   3. Compute per-trajectory collision risk against the ego reference
 *   4. Solve the Kantorovich dual to find Q* over trajectories
 *   5. Inject the top-K highest-Q* trajectories as extra constraints
 *
 * Compared against:
 *   - SHMPCC-Nominal: standard scenario sampling from nominal mode weights + SH
 *   - Mode-DRO+SH:   existing mode-level DRO injection + SH (for reference)
 *
 * Outputs:
 *   paper_figures/trajectory_dro_results.csv   -- per-rollout data
 *   paper_figures/trajectory_dro_summary.csv   -- per-strategy Wilson CIs
 *   paper_figures/trajectory_dro_config.txt    -- experiment configuration
 *
 * Usage: ./test_trajectory_dro [max_rollouts] [batch_size]
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
#include <cassert>
#include <filesystem>
#include <sstream>

#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include "wasserstein_dro.hpp"
#include "scenario_sampler.hpp"
#include "mode_weights.hpp"

using namespace scenario_mpc;
namespace fs = std::filesystem;

static const std::string OUTPUT_DIR = "paper_figures/";

// ============================================================================
// Experiment parameters
// ============================================================================

static constexpr int    BATCH_SIZE       = 200;
static constexpr int    MIN_ROLLOUTS     = 400;
static constexpr int    MAX_ROLLOUTS_DEF = 2000;
static constexpr double CI_Z             = 1.96;
static constexpr int    ROLLOUT_STEPS    = DEFAULT_ROLLOUT_STEPS;
static constexpr int    HORIZON          = DEFAULT_HORIZON;
static constexpr int    NUM_SCENARIOS    = DEFAULT_BASE_SCENARIOS;
static constexpr double DT               = DEFAULT_DT;
static constexpr double SWITCH_PROB      = 0.1;
static constexpr double RARE_PROB        = 0.05;
static constexpr int    NUM_DISCS        = 1;
static constexpr double VEHICLE_LENGTH   = 1.5;
static constexpr double DRO_RHO_BASE    = 0.1;

// Trajectory-level DRO parameters
static constexpr int    TRAJ_POOL_SIZE    = 60;   // S_pool: candidate trajectories
static constexpr int    TRAJ_INJECT_K     = 3;    // top-K trajectories to inject
static constexpr double TRAJ_DRO_RHO     = 0.15;  // Wasserstein ball radius for trajectory DRO
static constexpr double TRAJ_ALPHA        = 0.95;  // one-sided quantile for risk

static const std::vector<std::string> OBS_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Trajectory-level W2 helpers (standalone, no WassersteinDRO private access)
// ============================================================================

namespace traj_dro {

/**
 * @brief Closed-form 2x2 matrix square root.
 * sqrt(M) = (M + sqrt(det(M))*I) / sqrt(tr(M) + 2*sqrt(det(M)))
 */
Eigen::Matrix2d matrix_sqrt_2x2(const Eigen::Matrix2d& M) {
    double det_M = M.determinant();
    double tr_M = M.trace();
    double sqrt_det = (det_M > 0) ? std::sqrt(det_M) : 0.0;
    double denom_sq = tr_M + 2.0 * sqrt_det;
    if (denom_sq < 1e-15) return Eigen::Matrix2d::Zero();
    double denom = std::sqrt(denom_sq);
    return (M + sqrt_det * Eigen::Matrix2d::Identity()) / denom;
}

/**
 * @brief Gaussian W2 distance between two 2D Gaussians (Bures metric).
 * W2 = sqrt(||mu1-mu2||^2 + Bures^2(cov1, cov2))
 */
double gaussian_w2_2d(
    const Eigen::Vector2d& mu1, const Eigen::Matrix2d& cov1,
    const Eigen::Vector2d& mu2, const Eigen::Matrix2d& cov2
) {
    double mean_dist_sq = (mu1 - mu2).squaredNorm();
    Eigen::Matrix2d sqrt_cov1 = matrix_sqrt_2x2(cov1);
    Eigen::Matrix2d M = sqrt_cov1 * cov2 * sqrt_cov1;
    Eigen::Matrix2d sqrt_M = matrix_sqrt_2x2(M);
    double bures_sq = cov1.trace() + cov2.trace() - 2.0 * sqrt_M.trace();
    bures_sq = std::max(0.0, bures_sq);
    return std::sqrt(mean_dist_sq + bures_sq);
}

/**
 * @brief W2 distance between two obstacle trajectories.
 * Average of per-step Gaussian W2 over the horizon (steps 1..N).
 */
double trajectory_w2(const ObstacleTrajectory& a, const ObstacleTrajectory& b) {
    int n = std::min(static_cast<int>(a.steps.size()),
                     static_cast<int>(b.steps.size()));
    if (n <= 1) return 0.0;
    double sum = 0.0;
    int count = 0;
    for (int k = 1; k < n; ++k) {
        sum += gaussian_w2_2d(a.steps[k].mean, a.steps[k].covariance,
                              b.steps[k].mean, b.steps[k].covariance);
        count++;
    }
    return count > 0 ? sum / count : 0.0;
}

/**
 * @brief Compute pairwise W2 distance matrix between trajectories.
 */
std::vector<std::vector<double>> compute_trajectory_w2_matrix(
    const std::vector<ObstacleTrajectory>& trajectories
) {
    int N = static_cast<int>(trajectories.size());
    std::vector<std::vector<double>> D(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double w2 = trajectory_w2(trajectories[i], trajectories[j]);
            D[i][j] = w2;
            D[j][i] = w2;
        }
    }
    return D;
}

/**
 * @brief Compute per-trajectory collision risk.
 *
 * For trajectory i at timestep k:
 *   approach_dir = (obs_mean - ego_pos) / dist
 *   sigma_dir = sqrt(approach_dir^T * cov_k * approach_dir)
 *   risk_k = max(0, R + z_alpha * sigma_dir - dist)
 * r[i] = max over k
 */
double trajectory_risk(
    const ObstacleTrajectory& traj,
    const std::vector<EgoState>& ego_ref,
    double safety_radius,
    double alpha = 0.95
) {
    // z_alpha lookup
    double z_alpha = 1.6449;  // 0.95
    if (std::abs(alpha - 0.90) < 1e-6) z_alpha = 1.2816;
    if (std::abs(alpha - 0.975) < 1e-6) z_alpha = 1.9600;
    if (std::abs(alpha - 0.99) < 1e-6) z_alpha = 2.3263;

    double max_risk = 0.0;
    int n_steps = static_cast<int>(traj.steps.size());

    for (int k = 1; k < n_steps; ++k) {
        const EgoState& ego = (k < static_cast<int>(ego_ref.size()))
            ? ego_ref[k] : ego_ref.back();
        Eigen::Vector2d ego_pos = ego.position();
        Eigen::Vector2d obs_pos = traj.steps[k].mean;
        Eigen::Matrix2d cov = traj.steps[k].covariance;

        Eigen::Vector2d diff = obs_pos - ego_pos;
        double dist = diff.norm();
        if (dist < 1e-12) {
            max_risk = std::max(max_risk, safety_radius);
            continue;
        }
        Eigen::Vector2d n = diff / dist;
        double var_dir = n.transpose() * cov * n;
        if (!std::isfinite(var_dir) || var_dir < 0.0) var_dir = 0.0;
        double sigma_dir = std::max(std::sqrt(var_dir), 1e-6);
        double risk = std::max(0.0, safety_radius + z_alpha * sigma_dir - dist);
        max_risk = std::max(max_risk, risk);
    }
    return max_risk;
}

/**
 * @brief Evaluate Kantorovich dual objective at a given lambda.
 * g(lambda) = lambda*rho + sum_i w_i * max_j(r_j - lambda*D[i][j])
 */
double evaluate_dual(
    double lambda,
    const std::vector<double>& weights,
    const std::vector<double>& risk,
    const std::vector<std::vector<double>>& D,
    double rho
) {
    int M = static_cast<int>(weights.size());
    double val = lambda * rho;
    for (int i = 0; i < M; ++i) {
        if (weights[i] < 1e-15) continue;
        double best = -1e30;
        for (int j = 0; j < M; ++j) {
            double candidate = risk[j] - lambda * D[i][j];
            if (candidate > best) best = candidate;
        }
        val += weights[i] * best;
    }
    return val;
}

/**
 * @brief Build deterministic transport plan at a given lambda.
 * For each source i, find j*(i) = argmax_j { r_j - lambda*D[i][j] }.
 * Returns q[j] = sum of weights moved to j, and the transport cost.
 */
struct TrajPlan {
    std::vector<double> q;
    double transport_cost = 0.0;
};

TrajPlan build_plan(
    double lambda,
    const std::vector<double>& weights,
    const std::vector<double>& risk,
    const std::vector<std::vector<double>>& D,
    bool prefer_high_cost  // tie-breaking: true = MAX_COST, false = MIN_COST
) {
    int M = static_cast<int>(weights.size());
    TrajPlan plan;
    plan.q.assign(M, 0.0);
    plan.transport_cost = 0.0;

    for (int i = 0; i < M; ++i) {
        if (weights[i] < 1e-15) continue;

        double best_val = -1e30;
        int best_j = i;

        for (int j = 0; j < M; ++j) {
            double candidate = risk[j] - lambda * D[i][j];
            if (candidate > best_val + 1e-12) {
                best_val = candidate;
                best_j = j;
            } else if (std::abs(candidate - best_val) < 1e-12) {
                // Tie-breaking
                if (prefer_high_cost && D[i][j] > D[i][best_j]) {
                    best_j = j;
                } else if (!prefer_high_cost && D[i][j] < D[i][best_j]) {
                    best_j = j;
                }
            }
        }

        plan.q[best_j] += weights[i];
        plan.transport_cost += weights[i] * D[i][best_j];
    }
    return plan;
}

/**
 * @brief Solve trajectory-level Kantorovich dual via binary search.
 * Returns Q* (redistributed weights over trajectories).
 */
std::vector<double> solve_trajectory_dro(
    const std::vector<double>& nominal_weights,
    const std::vector<double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    double rho
) {
    int M = static_cast<int>(nominal_weights.size());
    if (M == 0) return {};

    // Find max risk and max cost for lambda bounds
    double r_max = *std::max_element(risk_vector.begin(), risk_vector.end());
    if (r_max < 1e-12) return nominal_weights;  // no risk, Q* = P_hat

    double d_max = 0.0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            d_max = std::max(d_max, D[i][j]);
    if (d_max < 1e-12) d_max = 1.0;

    double lambda_max = 2.0 * r_max / std::max(1e-12, rho);
    lambda_max = std::max(lambda_max, r_max / d_max * 10.0);

    // Binary search on lambda to minimize dual
    double lo = 0.0, hi = lambda_max;
    double best_lambda = 0.0;
    double best_val = evaluate_dual(0.0, nominal_weights, risk_vector, D, rho);

    for (int iter = 0; iter < 80; ++iter) {
        double mid = (lo + hi) / 2.0;
        double val = evaluate_dual(mid, nominal_weights, risk_vector, D, rho);
        if (val < best_val) {
            best_val = val;
            best_lambda = mid;
        }
        // Check gradient direction via finite difference
        double eps = std::max(1e-10, mid * 1e-6);
        double val_plus = evaluate_dual(mid + eps, nominal_weights, risk_vector, D, rho);
        if (val_plus < val) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo < 1e-10) break;
    }

    // Recover feasible Q* via bracketing + mixing
    // Build two plans: low-cost (below budget) and high-cost (above budget)
    TrajPlan plan_lo = build_plan(lambda_max * 2.0, nominal_weights, risk_vector, D, false);
    TrajPlan plan_hi = build_plan(1e-12, nominal_weights, risk_vector, D, true);

    // Ensure plan_lo.cost <= rho <= plan_hi.cost
    if (plan_lo.transport_cost > rho) {
        // Both exceed budget -- fall back to nominal
        return nominal_weights;
    }
    if (plan_hi.transport_cost <= rho) {
        // Both within budget -- use the higher-risk plan
        return plan_hi.q;
    }

    // Refine bracket via binary search on lambda
    double lam_lo = 1e-12, lam_hi = lambda_max * 2.0;
    for (int iter = 0; iter < 40; ++iter) {
        double lam_mid = (lam_lo + lam_hi) / 2.0;
        TrajPlan p_min = build_plan(lam_mid, nominal_weights, risk_vector, D, false);
        TrajPlan p_max = build_plan(lam_mid, nominal_weights, risk_vector, D, true);

        if (p_min.transport_cost > rho) {
            lam_lo = lam_mid;
        } else if (p_max.transport_cost < rho) {
            lam_hi = lam_mid;
        } else {
            // Budget is bracketed: p_min.cost <= rho <= p_max.cost
            plan_lo = p_min;
            plan_hi = p_max;
            break;
        }

        // Update whichever bracket is tighter
        if (p_min.transport_cost <= rho && p_min.transport_cost > plan_lo.transport_cost) {
            plan_lo = p_min;
        }
        if (p_max.transport_cost >= rho && p_max.transport_cost < plan_hi.transport_cost) {
            plan_hi = p_max;
        }
    }

    // Convex mix: alpha = (rho - c_lo) / (c_hi - c_lo)
    double c_lo = plan_lo.transport_cost;
    double c_hi = plan_hi.transport_cost;
    double alpha = 0.5;
    if (c_hi - c_lo > 1e-12) {
        alpha = (rho - c_lo) / (c_hi - c_lo);
    }
    alpha = std::clamp(alpha, 0.0, 1.0);

    std::vector<double> q_star(M, 0.0);
    for (int j = 0; j < M; ++j) {
        q_star[j] = (1.0 - alpha) * plan_lo.q[j] + alpha * plan_hi.q[j];
    }
    return q_star;
}

/**
 * @brief Select top-K trajectories by Q* weight.
 * Returns indices sorted by descending Q* weight.
 */
std::vector<int> select_topk(const std::vector<double>& q_star, int K) {
    int M = static_cast<int>(q_star.size());
    std::vector<std::pair<double, int>> sorted;
    sorted.reserve(M);
    for (int i = 0; i < M; ++i) {
        if (q_star[i] > 1e-12) {
            sorted.push_back({q_star[i], i});
        }
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<>());

    int n = std::min(K, static_cast<int>(sorted.size()));
    std::vector<int> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
        result.push_back(sorted[i].second);
    }
    return result;
}

}  // namespace traj_dro

// ============================================================================
// Strategy definitions
// ============================================================================

enum class TrajectoryStrategy {
    SHMPCC_NOMINAL,    // Baseline: nominal scenario sampling + safe horizon
    TRAJ_W2_DRO_SH,   // New: trajectory-level W2 DRO + safe horizon
    MODE_DRO_SH,       // Reference: existing mode-level DRO injection + SH
};

struct StrategyDef {
    std::string name;
    TrajectoryStrategy strategy;
    bool use_ot;
    bool enable_dro;
    InjectionMode injection_mode;
    bool safe_horizon;
};

static const std::vector<StrategyDef> STRATEGIES = {
    {"SHMPCC-Nom+SH",     TrajectoryStrategy::SHMPCC_NOMINAL,  false, false, InjectionMode::NONE,         true},
    {"Traj-W2-DRO+SH",    TrajectoryStrategy::TRAJ_W2_DRO_SH,  false, false, InjectionMode::NONE,         true},
    {"Mode-DRO(inj)+SH",  TrajectoryStrategy::MODE_DRO_SH,      false, true,  InjectionMode::DRO,          true},
    {"OT+Mode-DRO+SH",    TrajectoryStrategy::MODE_DRO_SH,      true,  true,  InjectionMode::DRO,          true},
};

// ============================================================================
// Metrics
// ============================================================================

struct StrategyMetrics {
    std::string name;
    std::vector<bool> collisions;
    std::vector<double> min_clearances;
    std::vector<double> solve_times_avg;
    std::vector<double> clearance_5pct;
    std::vector<double> total_progress;
    std::vector<bool> completed_path;
    std::vector<int> dro_injected;
    std::vector<double> avg_safe_horizon;
    std::vector<int> missed_mode_steps;
    std::vector<int> total_mode_checks;

    int n() const { return static_cast<int>(collisions.size()); }

    int collision_count() const {
        int c = 0; for (bool b : collisions) if (b) c++; return c;
    }
    double collision_rate() const {
        return n() > 0 ? static_cast<double>(collision_count()) / n() : 0.0;
    }
    std::pair<double, double> collision_ci() const {
        return wilson_ci(collision_count(), n(), CI_Z);
    }

    int total_missed() const {
        return std::accumulate(missed_mode_steps.begin(), missed_mode_steps.end(), 0);
    }
    int total_checks() const {
        return std::accumulate(total_mode_checks.begin(), total_mode_checks.end(), 0);
    }
    double mode_coverage() const {
        int tc = total_checks();
        return tc > 0 ? 1.0 - static_cast<double>(total_missed()) / tc : 1.0;
    }
    std::pair<double, double> mode_coverage_ci() const {
        int tc = total_checks(), covered = tc - total_missed();
        return wilson_ci(covered, tc, CI_Z);
    }

    double mean_solve_ms() const {
        if (solve_times_avg.empty()) return 0;
        return std::accumulate(solve_times_avg.begin(), solve_times_avg.end(), 0.0)
               / solve_times_avg.size();
    }
    double mean_clearance_5pct() const {
        if (clearance_5pct.empty()) return 0;
        return std::accumulate(clearance_5pct.begin(), clearance_5pct.end(), 0.0)
               / clearance_5pct.size();
    }
    double completion_rate() const {
        int c = 0; for (bool b : completed_path) if (b) c++;
        return n() > 0 ? static_cast<double>(c) / n() : 0.0;
    }
    double mean_dro_injected() const {
        if (dro_injected.empty()) return 0;
        return std::accumulate(dro_injected.begin(), dro_injected.end(), 0.0)
               / dro_injected.size();
    }

    void add(const RolloutRecord& rec) {
        collisions.push_back(rec.collision);
        min_clearances.push_back(rec.min_clearance);
        solve_times_avg.push_back(rec.avg_solve_ms);
        clearance_5pct.push_back(rec.clearance_5pct);
        total_progress.push_back(rec.total_progress);
        completed_path.push_back(rec.completed_path);
        dro_injected.push_back(rec.total_dro_injected);
        avg_safe_horizon.push_back(rec.avg_safe_horizon);
        missed_mode_steps.push_back(rec.missed_mode_steps);
        total_mode_checks.push_back(rec.total_mode_checks);
    }
};

// ============================================================================
// Build ExperimentConfig for each strategy
// ============================================================================

static ExperimentConfig make_config(const StrategyDef& strat) {
    ExperimentConfig cfg;
    cfg.horizon = HORIZON;
    cfg.num_scenarios = NUM_SCENARIOS;
    cfg.switch_prob = SWITCH_PROB;
    cfg.rollout_steps = ROLLOUT_STEPS;
    cfg.obs_modes = OBS_MODES;
    cfg.rare_mode = RARE_MODE;
    cfg.rare_switch_prob = RARE_PROB;
    cfg.num_discs = NUM_DISCS;
    cfg.vehicle_length = VEHICLE_LENGTH;
    cfg.path_completion_termination = true;
    cfg.path_completion_fraction = PATH_COMPLETE_FRAC;
    cfg.method_name = strat.name;
    cfg.ablation = AblationVariant::NO_INJECTION;

    cfg.use_ot_predictor = strat.use_ot;
    cfg.weight_type = strat.use_ot ? WeightType::WASSERSTEIN : WeightType::FREQUENCY;
    cfg.ot_ground_cost = GroundCostType::SQUARED_EUCLIDEAN;

    cfg.enable_dro = strat.enable_dro;
    cfg.injection_mode = strat.injection_mode;
    cfg.eps_wass = DRO_RHO_BASE;

    cfg.safe_horizon_enabled = strat.safe_horizon;

    return cfg;
}

/**
 * @brief Make config for the trajectory-DRO strategy.
 *
 * The controller is set up with DRO disabled; the trajectory-level DRO
 * is implemented externally via the step_callback which generates a
 * trajectory pool, runs the Kantorovich dual, and injects the top-K
 * redistributed trajectories before solve().
 */
static ExperimentConfig make_traj_dro_config() {
    ExperimentConfig cfg;
    cfg.horizon = HORIZON;
    cfg.num_scenarios = NUM_SCENARIOS;
    cfg.switch_prob = SWITCH_PROB;
    cfg.rollout_steps = ROLLOUT_STEPS;
    cfg.obs_modes = OBS_MODES;
    cfg.rare_mode = RARE_MODE;
    cfg.rare_switch_prob = RARE_PROB;
    cfg.num_discs = NUM_DISCS;
    cfg.vehicle_length = VEHICLE_LENGTH;
    cfg.path_completion_termination = true;
    cfg.path_completion_fraction = PATH_COMPLETE_FRAC;
    cfg.method_name = "Traj-W2-DRO+SH";
    cfg.ablation = AblationVariant::NO_INJECTION;

    cfg.use_ot_predictor = false;
    cfg.weight_type = WeightType::FREQUENCY;
    cfg.enable_dro = false;           // DRO disabled at mode level
    cfg.injection_mode = InjectionMode::NONE;
    cfg.safe_horizon_enabled = true;

    // The step_callback implements trajectory-level DRO:
    // Before each solve(), it:
    //   1. Samples a pool of S_pool trajectories from the obstacle's nominal dynamics
    //   2. Computes pairwise W2 distances between trajectories
    //   3. Computes per-trajectory risk against ego reference trajectory
    //   4. Solves the Kantorovich dual to get Q*
    //   5. Injects top-K trajectories as additional scenarios
    cfg.step_callback = [](int step, int obstacle_id, ObstacleSim& obs_sim,
                           AdaptiveScenarioMPC& ctrl, std::mt19937& rng) {
        // Only run trajectory DRO for obstacle 0 (first obstacle)
        // and only after we have at least one mode observation
        if (obstacle_id != 0) return;
        if (step < 2) return;  // need warmup for reference trajectory

        const auto& config = ctrl.config();

        // Get obstacle current state from the sim
        ObstacleState obs_state = obs_sim.state;

        // Build obstacle map and mode history for sampling
        std::map<int, ObstacleState> obstacles;
        obstacles[0] = obs_state;

        // Create mode history for trajectory sampling
        ModeHistory mh(0, obs_sim.mode_models);
        // Add current mode as observation so weights are nonzero
        for (const auto& mode_id : obs_sim.available_modes) {
            mh.record_observation(step, mode_id);
        }

        auto mode_weights = compute_mode_weights(mh, WeightType::FREQUENCY);
        if (mode_weights.empty()) return;

        // Step 1: Sample S_pool candidate trajectories from nominal dynamics
        std::map<int, ModeHistory> histories;
        histories[0] = mh;

        auto pool_scenarios = sample_scenarios(
            obstacles, histories, config.horizon, TRAJ_POOL_SIZE,
            WeightType::FREQUENCY, 0.9, step, &rng
        );

        // Extract obstacle trajectories from scenarios
        std::vector<ObstacleTrajectory> traj_pool;
        traj_pool.reserve(pool_scenarios.size());
        for (const auto& sc : pool_scenarios) {
            auto it = sc.trajectories.find(0);
            if (it != sc.trajectories.end()) {
                traj_pool.push_back(it->second);
            }
        }
        if (traj_pool.size() < 3) return;

        // Build ego reference trajectory for risk computation
        // Use a straight line from ego's approximate current position
        const auto& prev_scenarios = ctrl.scenarios();
        std::vector<EgoState> ego_ref;
        // Simple: project forward at reference velocity
        // Use obstacle position to estimate ego position (rough)
        EgoState ego_approx;
        ego_approx.x = obs_state.x - 5.0;  // ego is behind obstacle
        ego_approx.y = obs_state.y;
        ego_approx.theta = 0.0;
        ego_approx.v = 2.0;
        ego_ref.resize(config.horizon + 1);
        for (int k = 0; k <= config.horizon; ++k) {
            ego_ref[k] = ego_approx;
            ego_ref[k].x += k * config.dt * ego_approx.v;
        }

        double safety_radius = config.ego_radius + config.obstacle_radius
                              + config.safety_margin;

        // Step 2: Compute pairwise W2 distance matrix
        auto D = traj_dro::compute_trajectory_w2_matrix(traj_pool);

        // Step 3: Compute per-trajectory risk
        int M = static_cast<int>(traj_pool.size());
        std::vector<double> risks(M);
        for (int i = 0; i < M; ++i) {
            risks[i] = traj_dro::trajectory_risk(
                traj_pool[i], ego_ref, safety_radius, TRAJ_ALPHA);
        }

        // Nominal weights: uniform over pool
        std::vector<double> nominal(M, 1.0 / M);

        // Step 4: Solve Kantorovich dual to get Q*
        auto q_star = traj_dro::solve_trajectory_dro(nominal, risks, D, TRAJ_DRO_RHO);

        // Step 5: Select top-K trajectories and inject them
        auto topk = traj_dro::select_topk(q_star, TRAJ_INJECT_K);

        int base_id = 10000 + step * 100;  // unique scenario IDs
        for (int idx : topk) {
            std::map<int, ObstacleTrajectory> trajs;
            ObstacleTrajectory t = traj_pool[idx];
            t.probability = q_star[idx];
            trajs[0] = t;

            Scenario sc(base_id++, trajs, q_star[idx]);
            sc.is_injected = true;  // immune to pruning
            ctrl.inject_scenario(sc);
        }
    };

    return cfg;
}

// ============================================================================
// CI overlap check
// ============================================================================

static bool cis_overlap(std::pair<double, double> a, std::pair<double, double> b) {
    return a.second >= b.first && b.second >= a.first;
}

// ============================================================================
// Output helpers
// ============================================================================

static std::string fmt_ci(double lo, double hi) {
    std::ostringstream s;
    s << "[" << std::fixed << std::setprecision(1)
      << (lo * 100.0) << "," << (hi * 100.0) << "]";
    return s.str();
}

static void write_config_file(const std::string& filepath, int total_rollouts,
                               int batch_size, int max_rollouts) {
    std::ofstream ofs(filepath);
    ofs << "# Trajectory-Level W2 DRO Experiment Configuration\n";
    ofs << "# Generated: " << __DATE__ << " " << __TIME__ << "\n\n";

    ofs << "[experiment]\n";
    ofs << "batch_size = " << batch_size << "\n";
    ofs << "min_rollouts = " << MIN_ROLLOUTS << "\n";
    ofs << "max_rollouts = " << max_rollouts << "\n";
    ofs << "actual_rollouts = " << total_rollouts << "\n";
    ofs << "ci_z = " << CI_Z << " (95% Wilson CI)\n\n";

    ofs << "[mpc]\n";
    ofs << "horizon = " << HORIZON << "\n";
    ofs << "dt = " << DT << "\n";
    ofs << "num_scenarios = " << NUM_SCENARIOS << "\n";
    ofs << "rollout_steps = " << ROLLOUT_STEPS << "\n";
    ofs << "num_discs = " << NUM_DISCS << "\n";
    ofs << "vehicle_length = " << VEHICLE_LENGTH << "\n\n";

    ofs << "[environment]\n";
    ofs << "path_type = s_curve\n";
    ofs << "switch_prob = " << SWITCH_PROB << "\n";
    ofs << "rare_mode_prob = " << RARE_PROB << "\n";
    ofs << "rare_mode = " << RARE_MODE << "\n\n";

    ofs << "[trajectory_dro]\n";
    ofs << "pool_size = " << TRAJ_POOL_SIZE << "\n";
    ofs << "inject_k = " << TRAJ_INJECT_K << "\n";
    ofs << "rho = " << TRAJ_DRO_RHO << "\n";
    ofs << "alpha = " << TRAJ_ALPHA << "\n";
    ofs << "w2_metric = gaussian_bures (averaged over horizon)\n\n";

    ofs << "[mode_dro]\n";
    ofs << "rho_base = " << DRO_RHO_BASE << "\n";
    ofs << "ground_cost = W2_BURES\n";
    ofs << "risk_mode = FULL\n\n";

    ofs << "[strategies]\n";
    for (size_t i = 0; i < STRATEGIES.size(); ++i) {
        const auto& s = STRATEGIES[i];
        ofs << i << ". " << s.name << "\n";
    }
    ofs.close();
}

static void write_detail_csv(
    const std::string& filepath,
    const std::vector<StrategyMetrics>& metrics
) {
    std::ofstream ofs(filepath);
    ofs << "strategy,rollout_idx,collision,min_clearance,clearance_5pct,"
        << "avg_solve_ms,total_progress,completed_path,dro_injected,"
        << "avg_safe_horizon,missed_mode_steps,total_mode_checks\n";

    for (const auto& m : metrics) {
        for (int i = 0; i < m.n(); ++i) {
            ofs << m.name << "," << i << ","
                << (m.collisions[i] ? 1 : 0) << ","
                << std::fixed << std::setprecision(4)
                << m.min_clearances[i] << ","
                << m.clearance_5pct[i] << ","
                << m.solve_times_avg[i] << ","
                << m.total_progress[i] << ","
                << (m.completed_path[i] ? 1 : 0) << ","
                << m.dro_injected[i] << ","
                << std::setprecision(2)
                << m.avg_safe_horizon[i] << ","
                << m.missed_mode_steps[i] << ","
                << m.total_mode_checks[i] << "\n";
        }
    }
    ofs.close();
}

static void write_summary_csv(
    const std::string& filepath,
    const std::vector<StrategyMetrics>& metrics
) {
    std::ofstream ofs(filepath);
    ofs << "strategy,n,"
        << "collision_rate,collision_ci_lo,collision_ci_hi,"
        << "mode_coverage,mode_cov_ci_lo,mode_cov_ci_hi,"
        << "mean_solve_ms,mean_clearance_5pct,completion_rate,"
        << "mean_dro_injected\n";

    for (const auto& m : metrics) {
        auto [clo, chi] = m.collision_ci();
        auto [mlo, mhi] = m.mode_coverage_ci();

        ofs << m.name << "," << m.n() << ","
            << std::fixed << std::setprecision(4)
            << m.collision_rate() << "," << clo << "," << chi << ","
            << m.mode_coverage() << "," << mlo << "," << mhi << ","
            << std::setprecision(2)
            << m.mean_solve_ms() << ","
            << std::setprecision(4)
            << m.mean_clearance_5pct() << ","
            << m.completion_rate() << ","
            << std::setprecision(1)
            << m.mean_dro_injected() << "\n";
    }
    ofs.close();
}

static void print_status(
    int batch_num,
    int total_rollouts,
    const std::vector<StrategyMetrics>& metrics,
    double batch_time_s
) {
    std::cout << "\n========== Batch " << batch_num
              << " (" << total_rollouts << " rollouts/strategy, "
              << std::fixed << std::setprecision(1) << batch_time_s << "s)"
              << " ==========\n";

    std::cout << "\n  Collision rate (95% Wilson CI):\n";
    for (const auto& m : metrics) {
        auto [lo, hi] = m.collision_ci();
        std::cout << "    " << std::setw(22) << std::left << m.name
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << std::right << (m.collision_rate() * 100.0) << "%  "
                  << std::setw(14) << fmt_ci(lo, hi) << "\n";
    }

    std::cout << "\n  Mode coverage (95% Wilson CI):\n";
    for (const auto& m : metrics) {
        auto [lo, hi] = m.mode_coverage_ci();
        std::cout << "    " << std::setw(22) << std::left << m.name
                  << std::fixed << std::setprecision(2)
                  << std::setw(7) << std::right << (m.mode_coverage() * 100.0) << "%  "
                  << std::setw(14) << fmt_ci(lo, hi) << "\n";
    }

    std::cout << "\n  Solve time (ms) / Clearance 5th%ile:\n";
    for (const auto& m : metrics) {
        std::cout << "    " << std::setw(22) << std::left << m.name
                  << std::fixed << std::setprecision(2)
                  << m.mean_solve_ms() << " ms  /  "
                  << std::setprecision(3) << m.mean_clearance_5pct() << " m\n";
    }

    // CI separation vs baseline (index 0)
    const auto& base = metrics[0];
    auto base_ci = base.collision_ci();
    std::cout << "\n  CI separation vs " << base.name << ":\n";
    int separated = 0;
    for (size_t i = 1; i < metrics.size(); ++i) {
        auto ci = metrics[i].collision_ci();
        bool sep = !cis_overlap(base_ci, ci);
        if (sep) separated++;
        std::cout << "    " << std::setw(22) << std::left << metrics[i].name
                  << (sep ? "SEPARATED" : "overlaps") << "\n";
    }
    std::cout << "  (" << separated << "/" << (metrics.size() - 1)
              << " separated from baseline)\n" << std::flush;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    fs::create_directories(OUTPUT_DIR);

    int max_rollouts = MAX_ROLLOUTS_DEF;
    int batch_size = BATCH_SIZE;

    for (int i = 1; i < argc; ++i) {
        if (i == 1 && std::atoi(argv[i]) > 0) {
            max_rollouts = std::atoi(argv[i]);
        } else if (i == 2 && std::atoi(argv[i]) > 0) {
            batch_size = std::atoi(argv[i]);
        }
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    std::cout << "================================================================\n";
    std::cout << "  Trajectory-Level W2 DRO vs SHMPCC Nominal Baseline\n";
    std::cout << "  " << STRATEGIES.size() << " strategies, max "
              << max_rollouts << " rollouts/strategy\n";
    std::cout << "  Traj pool=" << TRAJ_POOL_SIZE
              << "  inject_K=" << TRAJ_INJECT_K
              << "  rho=" << TRAJ_DRO_RHO << "\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "================================================================\n\n"
              << std::flush;

    // Initialize metrics
    std::vector<StrategyMetrics> metrics(STRATEGIES.size());
    for (size_t i = 0; i < STRATEGIES.size(); ++i) {
        metrics[i].name = STRATEGIES[i].name;
    }

    // Build configs
    std::vector<ExperimentConfig> configs(STRATEGIES.size());
    for (size_t i = 0; i < STRATEGIES.size(); ++i) {
        if (STRATEGIES[i].strategy == TrajectoryStrategy::TRAJ_W2_DRO_SH) {
            configs[i] = make_traj_dro_config();
        } else {
            configs[i] = make_config(STRATEGIES[i]);
        }
    }

    int total_rollouts = 0;
    bool early_stop = false;

    for (int batch = 0; total_rollouts < max_rollouts; ++batch) {
        auto batch_start = std::chrono::high_resolution_clock::now();

        int batch_end = std::min(total_rollouts + batch_size, max_rollouts);
        int batch_count = batch_end - total_rollouts;

        std::cout << "Running batch " << batch << ": rollouts "
                  << total_rollouts << ".." << (batch_end - 1)
                  << " (" << batch_count << " per strategy)...\n" << std::flush;

        for (size_t si = 0; si < STRATEGIES.size(); ++si) {
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int r = total_rollouts; r < batch_end; ++r) {
                SeedBundle seeds = derive_seeds(42u, r);
                auto rec = run_experiment_rollout(configs[si], seeds.env);
                metrics[si].add(rec);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double s = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "  " << std::setw(22) << std::left << STRATEGIES[si].name
                      << batch_count << " in "
                      << std::fixed << std::setprecision(1) << s << "s"
                      << "  coll=" << std::setprecision(1)
                      << (metrics[si].collision_rate() * 100.0) << "%\n" << std::flush;
        }

        total_rollouts = batch_end;

        auto batch_end_t = std::chrono::high_resolution_clock::now();
        double batch_s = std::chrono::duration<double>(batch_end_t - batch_start).count();
        print_status(batch, total_rollouts, metrics, batch_s);

        // Write intermediate results
        write_detail_csv(OUTPUT_DIR + "trajectory_dro_results.csv", metrics);
        write_summary_csv(OUTPUT_DIR + "trajectory_dro_summary.csv", metrics);
        write_config_file(OUTPUT_DIR + "trajectory_dro_config.txt",
                         total_rollouts, batch_size, max_rollouts);

        // Early stop when trajectory-DRO collision CI separated from baseline
        if (total_rollouts >= MIN_ROLLOUTS) {
            auto base_ci = metrics[0].collision_ci();
            int separated = 0;
            for (size_t i = 1; i < metrics.size(); ++i) {
                if (!cis_overlap(base_ci, metrics[i].collision_ci()))
                    separated++;
            }
            if (separated == static_cast<int>(metrics.size()) - 1) {
                early_stop = true;
                std::cout << "\n*** All collision CIs separated from baseline at "
                          << total_rollouts << " rollouts. ***\n" << std::flush;
                break;
            }
        }
    }

    // Final write
    write_detail_csv(OUTPUT_DIR + "trajectory_dro_results.csv", metrics);
    write_summary_csv(OUTPUT_DIR + "trajectory_dro_summary.csv", metrics);
    write_config_file(OUTPUT_DIR + "trajectory_dro_config.txt",
                     total_rollouts, batch_size, max_rollouts);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(total_end - total_start).count();

    // Final summary table
    std::cout << "\n================================================================\n";
    std::cout << "  Done in " << std::fixed << std::setprecision(1)
              << total_s << "s (" << total_s / 60.0 << " min), "
              << total_rollouts << " rollouts/strategy"
              << (early_stop ? " (early stop)" : " (cap)") << "\n\n";

    std::cout << "  " << std::setw(22) << std::left << "Strategy"
              << std::setw(8)  << "Coll%"
              << std::setw(15) << "CollCI"
              << std::setw(8)  << "MCov%"
              << std::setw(15) << "MCovCI"
              << std::setw(10) << "Solve ms"
              << std::setw(8)  << "DROInj"
              << "\n";
    std::cout << "  " << std::string(86, '-') << "\n";

    for (const auto& m : metrics) {
        auto [clo, chi] = m.collision_ci();
        auto [mlo, mhi] = m.mode_coverage_ci();

        std::cout << "  " << std::setw(22) << std::left << m.name
                  << std::fixed
                  << std::setprecision(1) << std::setw(8) << (m.collision_rate() * 100.0)
                  << std::setw(15) << fmt_ci(clo, chi)
                  << std::setprecision(1) << std::setw(8) << (m.mode_coverage() * 100.0)
                  << std::setw(15) << fmt_ci(mlo, mhi)
                  << std::setprecision(2) << std::setw(10) << m.mean_solve_ms()
                  << std::setprecision(1) << std::setw(8) << m.mean_dro_injected()
                  << "\n";
    }
    std::cout << "  " << std::string(86, '-') << "\n";

    std::cout << "\n  Trajectory-DRO parameters:\n";
    std::cout << "    Pool size (S_pool):  " << TRAJ_POOL_SIZE << "\n";
    std::cout << "    Inject K:            " << TRAJ_INJECT_K << "\n";
    std::cout << "    Wasserstein rho:     " << TRAJ_DRO_RHO << "\n";
    std::cout << "    W2 metric:           Gaussian Bures (avg over horizon)\n";
    std::cout << "    Risk quantile alpha: " << TRAJ_ALPHA << "\n";

    std::cout << "\n  Output files:\n";
    std::cout << "    " << OUTPUT_DIR << "trajectory_dro_results.csv\n";
    std::cout << "    " << OUTPUT_DIR << "trajectory_dro_summary.csv\n";
    std::cout << "    " << OUTPUT_DIR << "trajectory_dro_config.txt\n";
    std::cout << "================================================================\n";

    return 0;
}
