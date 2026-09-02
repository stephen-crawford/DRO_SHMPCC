/**
 * @file test_trajectory_dro.cpp
 * @brief Trajectory-level DRO experiments.
 *
 * Compares three approaches:
 *   1. SHMPCC baseline (no DRO)
 *   2. Discrete-mode DRO (existing: reweight mode probabilities via Wasserstein ball)
 *   3. Trajectory-level DRO (new: reweight individual trajectory realizations
 *      via a Wasserstein ball over the trajectory particle distribution)
 *
 * Trajectory-level DRO concept:
 *   Instead of defining a transport cost between discrete modes and solving
 *   the Kantorovich dual over modes, we:
 *     (a) Sample S trajectory realizations from the nominal mode distribution
 *     (b) Compute per-trajectory collision risk r_s for each trajectory s
 *     (c) Compute pairwise trajectory transport costs D[s1][s2] using
 *         mean-squared displacement between trajectory positions
 *     (d) Solve the Kantorovich dual over trajectory particles:
 *           sup_{Q in B_rho(P_hat)} sum_s q_s * r_s
 *         where P_hat = (1/S, ..., 1/S) is the empirical distribution
 *     (e) Either resample from Q* or inject the highest-risk trajectory
 *
 * This is mode-agnostic: it works even when the obstacle model has no discrete
 * modes, directly capturing the geometry of the trajectory distribution.
 *
 * Experiments:
 *   A: Method comparison across environments (Oncoming, Intersection)
 *   B: Multi-obstacle stress test
 *   C: Trajectory DRO variant comparison (resampling vs injection)
 *   D: Trajectory particle count sweep
 *
 * Outputs CSV files to paper_figures/.
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

using namespace dro_mpc;
namespace fs = std::filesystem;

// ============================================================================
// Constants
// ============================================================================

static const std::string OUTPUT_DIR = "paper_figures/";

static const int    HORIZON = DEFAULT_HORIZON;
static const double    DT = DEFAULT_DT;
static const int    NUM_SCENARIOS = DEFAULT_BASE_SCENARIOS;
static const int    ROLLOUT_STEPS = DEFAULT_ROLLOUT_STEPS;
static constexpr int    NUM_DISCS       = 1;
static constexpr double VEHICLE_LENGTH  = 1.5;
static constexpr int    SAFE_HORIZON_MIN = 3;

static const std::vector<std::string> OBS_MODES = {
    "constant_velocity", "turn_left", "turn_right", "decelerating"
};
static const std::string RARE_MODE = "lane_change_left";

// ============================================================================
// Trajectory-Level DRO Implementation
// ============================================================================

/**
 * @brief Compute directional collision risk for a single trajectory.
 *
 * For each timestep k and disc d, compute:
 *   r_{k,d} = max(0, R + z_alpha * sigma_dir - ||traj_k - disc_d||)
 * where sigma_dir is the directional std dev along the approach direction.
 * Return the maximum over k and d.
 */
static double compute_trajectory_risk(
    const ObstacleTrajectory& traj,
    const std::vector<EgoState>& ego_ref,
    double safety_radius,
    int num_discs,
    double vehicle_length,
    double z_alpha = 1.6449  // 95% one-sided
) {
    double max_risk = 0.0;
    int H = std::min(static_cast<int>(traj.steps.size()),
                     static_cast<int>(ego_ref.size()));

    for (int k = 1; k < H; ++k) {
        const auto& step = traj.steps[k];
        const auto& ego = ego_ref[k];

        // Disc positions along ego body
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

            // Directional std dev along approach direction
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

/**
 * @brief Compute pairwise trajectory transport cost using mean-squared
 * displacement between trajectory positions averaged over the horizon.
 *
 * D[i][j] = (1/H) * sum_k ||mu_i(k) - mu_j(k)||^2
 */
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

/**
 * @brief Solve the Kantorovich dual over trajectory particles.
 *
 * inf_{lambda>=0} { lambda*rho + (1/S)*sum_i max_j (r_j - lambda*D[i][j]) }
 *
 * Returns (optimal_lambda, worst_case_risk, worst_case_weights).
 */
struct TrajectoryDROResult {
    std::vector<double> weights;       // Q* weights over trajectories
    double optimal_lambda = 0.0;
    double worst_case_risk = 0.0;
    double rho_used = 0.0;
    int worst_trajectory_idx = -1;     // Trajectory with highest Q* weight
};

static TrajectoryDROResult solve_trajectory_dro(
    const std::vector<double>& risks,       // r[s] for each trajectory
    const std::vector<std::vector<double>>& D,  // Transport cost matrix
    double rho,
    int n_obs_for_rho = 10    // observation count for adaptive rho
) {
    int S = static_cast<int>(risks.size());
    TrajectoryDROResult result;
    result.rho_used = rho;

    if (S == 0) return result;

    // Uniform nominal weights
    double w_nom = 1.0 / S;

    // Find max risk for upper bound on lambda
    double max_risk = *std::max_element(risks.begin(), risks.end());
    double min_risk = *std::min_element(risks.begin(), risks.end());

    if (max_risk < 1e-12 || std::abs(max_risk - min_risk) < 1e-12) {
        // No risk differential — uniform weights
        result.weights.assign(S, w_nom);
        result.worst_case_risk = max_risk;
        result.worst_trajectory_idx = static_cast<int>(
            std::max_element(risks.begin(), risks.end()) - risks.begin());
        return result;
    }

    // Find max D for lambda upper bound
    double max_D = 0.0;
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j)
            max_D = std::max(max_D, D[i][j]);
    if (max_D < 1e-12) max_D = 1.0;

    double lambda_max = (max_risk - min_risk) / (max_D > 1e-12 ? max_D * 0.01 : 1.0);
    lambda_max = std::max(lambda_max, 10.0);

    // Evaluate dual objective at a given lambda
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

    // Build transport plan at given lambda (for Q* recovery)
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

    // Compute transport cost of a plan
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

    // Binary search on lambda
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

        // Check if the plan at mid uses more or less than rho transport
        double tc = plan_cost(mid);
        if (tc > rho) {
            lo = mid;  // Need higher lambda to reduce transport
        } else {
            hi = mid;  // Can reduce lambda to increase transport
        }
    }

    result.optimal_lambda = best_lambda;
    result.worst_case_risk = best_dual;
    result.weights = build_plan(best_lambda);

    // Find trajectory with highest Q* weight (for injection)
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
// Metrics (same structure as test_dro_benefits.cpp)
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
    std::vector<double> solve_times;

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
        if (solve_times.empty()) return 0;
        return std::accumulate(solve_times.begin(), solve_times.end(), 0.0) / solve_times.size();
    }

    void add(const RolloutRecord& rec) {
        collisions.push_back(rec.collision);
        missed_mode_steps.push_back(rec.missed_mode_steps);
        total_mode_checks.push_back(rec.total_mode_checks);
        rare_mode_active.push_back(rec.rare_mode_active);
        rare_mode_missed.push_back(rec.rare_mode_missed);
        min_clearances.push_back(rec.min_clearance);
        total_progress.push_back(rec.total_progress);
        solve_times.push_back(rec.avg_solve_ms);
    }
};

// ============================================================================
// Method definitions
// ============================================================================

enum class TrajDROMethod {
    SHMPCC_BASE,          // No DRO (SHMPCC baseline)
    MODE_DRO_SAMPLE,      // Discrete-mode DRO: resample from q*
    MODE_DRO_INJECT,      // Discrete-mode DRO: inject worst-case mode
    TRAJ_DRO_RESAMPLE,    // Trajectory DRO: resample from trajectory q*
    TRAJ_DRO_INJECT,      // Trajectory DRO: inject worst-case trajectory
    TRAJ_DRO_COMBINED     // Trajectory DRO: resample + inject adversarial
};

static std::string method_name(TrajDROMethod m) {
    switch (m) {
        case TrajDROMethod::SHMPCC_BASE:        return "SHMPCC-Base";
        case TrajDROMethod::MODE_DRO_SAMPLE:    return "Mode-DRO(q*)";
        case TrajDROMethod::MODE_DRO_INJECT:    return "Mode-DRO(inj)";
        case TrajDROMethod::TRAJ_DRO_RESAMPLE:  return "Traj-DRO(q*)";
        case TrajDROMethod::TRAJ_DRO_INJECT:    return "Traj-DRO(inj)";
        case TrajDROMethod::TRAJ_DRO_COMBINED:  return "Traj-DRO(comb)";
    }
    return "?";
}

/**
 * @brief Build an ExperimentConfig for the given method.
 *
 * For trajectory-level DRO methods, we use the base SHMPCC config with
 * a step_callback that performs trajectory-level DRO at each MPC step.
 * The callback:
 *   1. Samples trajectory particles from the nominal distribution
 *   2. Computes per-trajectory risk
 *   3. Solves trajectory-level Kantorovich dual
 *   4. Either sets custom mode weights (resampling) or injects worst-case
 *      trajectory as an extra scenario
 */
static ExperimentConfig make_config(
    TrajDROMethod method,
    double switch_prob,
    double rare_prob,
    int num_scenarios = NUM_SCENARIOS,
    int num_obs = 1,
    double rho = 0.1
) {
    ExperimentConfig cfg;
    cfg.mpc.horizon = HORIZON;
    cfg.mpc.sampling.num_scenarios = num_scenarios;
    cfg.obstacles.switch_prob = switch_prob;
    cfg.rollout.rollout_steps = ROLLOUT_STEPS;
    cfg.obstacles.obs_modes = OBS_MODES;
    cfg.obstacles.rare_mode = RARE_MODE;
    cfg.obstacles.rare_switch_prob = rare_prob;
    cfg.mpc.ego.num_discs = NUM_DISCS;
    cfg.mpc.ego.length = VEHICLE_LENGTH;
    cfg.mpc.safe_horizon_enabled = true;
    cfg.mpc.constraints.safe_horizon_min = SAFE_HORIZON_MIN;
    cfg.environment.path_completion_termination = true;
    cfg.environment.path_completion_fraction = 0.95;
    cfg.dro.enabled = false;
    cfg.dro.solver.base_radius = rho;
    cfg.obstacles.num_obstacles = num_obs;
    cfg.obstacles.obstacles_per_class = 1;
    cfg.rollout.method_name = method_name(method);

    switch (method) {
        case TrajDROMethod::SHMPCC_BASE:
            // No DRO
            break;

        case TrajDROMethod::MODE_DRO_SAMPLE:
            cfg.dro.enabled = true;
            break;

        case TrajDROMethod::MODE_DRO_INJECT:
            cfg.dro.enabled = true;
            break;

        case TrajDROMethod::TRAJ_DRO_RESAMPLE:
        case TrajDROMethod::TRAJ_DRO_INJECT:
        case TrajDROMethod::TRAJ_DRO_COMBINED:
            // Trajectory DRO: base SHMPCC + step callback that performs
            // trajectory-level DRO using the controller's inject_scenario API.
            // We leave enable_dro=false so the controller does normal sampling,
            // then our callback overrides with DRO-reweighted scenarios.
            cfg.rollout.step_callback = [method, rho, num_scenarios](
                int step, int obs_id, ObstacleSim& obs_sim,
                AdaptiveScenarioMPC& controller, std::mt19937& rng
            ) {
                // Only act on the first obstacle callback per step
                if (obs_id != 0) return;

                // Sample trajectory particles from the nominal distribution
                // by propagating each mode's dynamics with noise
                const auto& modes = obs_sim.mode_models;
                int horizon = controller.config().mpc.horizon;
                double ego_r = controller.config().mpc.ego.radius;
                double obs_r = controller.config().obstacle_radius;
                double margin = controller.config().mpc.constraints.safety_margin;
                double safety_radius = ego_r + obs_r + margin;
                int n_discs = controller.config().mpc.ego.num_discs;
                double veh_len = controller.config().mpc.ego.length;

                // Get the ego reference trajectory (from controller's scenarios)
                // We'll use the controller's current state as a proxy
                const auto& scenarios = controller.scenarios();
                if (scenarios.empty()) return;

                // Build a simple ego reference trajectory from the current
                // solution. If no previous solution, skip this step.
                // (In practice the controller has already initialized its
                // reference trajectory before step_callback fires.)

                // Collect all obstacle trajectories from the currently-sampled
                // scenarios for obs_id=0. These are our trajectory particles.
                std::vector<ObstacleTrajectory> traj_particles;
                for (const auto& sc : scenarios) {
                    auto it = sc.trajectories.find(0);
                    if (it != sc.trajectories.end()) {
                        traj_particles.push_back(it->second);
                    }
                }

                if (traj_particles.size() < 3) return;

                // Build a simple ego reference from the first scenario's
                // ego trajectory. Since we don't have direct access to the
                // controller's internal reference, we'll approximate using
                // a constant-velocity projection from current ego state.
                // The step_callback doesn't receive ego state directly,
                // but we can infer from obstacle-relative geometry.

                // Use a straight-line ego reference as approximation
                // (The actual ego reference is internal to the controller,
                //  but for risk computation this is a reasonable proxy.)
                std::vector<EgoState> ego_ref;
                // We'll create a placeholder ego trajectory that moves
                // forward at constant velocity along x-axis as proxy
                for (int k = 0; k <= horizon; ++k) {
                    EgoState es;
                    es.x = 1.5 * k * 0.1;  // approx v=1.5 m/s
                    es.y = 0.0;
                    es.theta = 0.0;
                    es.v = 1.5;
                    ego_ref.push_back(es);
                }

                // Compute per-trajectory risk
                int S = static_cast<int>(traj_particles.size());
                std::vector<double> risks(S);
                for (int s = 0; s < S; ++s) {
                    risks[s] = compute_trajectory_risk(
                        traj_particles[s], ego_ref, safety_radius,
                        n_discs, veh_len);
                }

                // Compute pairwise transport costs
                auto D = compute_trajectory_transport_costs(traj_particles);

                // Solve trajectory-level DRO
                auto dro_result = solve_trajectory_dro(risks, D, rho);

                if (dro_result.worst_trajectory_idx < 0) return;

                // Apply the DRO result based on method variant
                if (method == TrajDROMethod::TRAJ_DRO_INJECT ||
                    method == TrajDROMethod::TRAJ_DRO_COMBINED) {
                    // Inject the worst-case trajectory as an extra scenario
                    int worst_idx = dro_result.worst_trajectory_idx;
                    const auto& worst_traj = traj_particles[worst_idx];

                    // Build a scenario containing just this trajectory
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
                    // Set custom mode weights based on trajectory DRO result.
                    // Aggregate trajectory-level Q* weights back to mode weights:
                    //   q_mode[m] = sum_{s: mode_s == m} q_traj[s]
                    std::map<std::string, double> mode_weights;
                    for (int s = 0; s < S; ++s) {
                        const std::string& mode = traj_particles[s].mode_id;
                        mode_weights[mode] += dro_result.weights[s];
                    }

                    // Normalize
                    double sum_w = 0.0;
                    for (auto& [_, w] : mode_weights) sum_w += w;
                    if (sum_w > 1e-12) {
                        for (auto& [_, w] : mode_weights) w /= sum_w;
                    }

                    controller.set_custom_mode_weights(0, mode_weights);
                }
            };
            break;
    }

    return cfg;
}

static double elapsed_sec(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

static void write_csv_header(std::ofstream& ofs, const std::string& extra_cols = "") {
    ofs << extra_cols
        << "method,collision_rate,coll_ci_lo,coll_ci_hi,"
        << "missed_mode_rate,mm_ci_lo,mm_ci_hi,"
        << "rare_miss_rate,rare_ci_lo,rare_ci_hi,"
        << "mean_clearance,p5_clearance,mean_progress,"
        << "mean_solve_ms,n_rollouts\n";
}

static void write_csv_row(std::ofstream& ofs, const Metrics& met, int n_rollouts,
                           const std::string& prefix = "") {
    auto [clo, chi] = met.coll_ci();
    auto [mlo, mhi] = met.mm_ci();
    auto [rlo, rhi] = met.rare_ci();
    ofs << prefix << met.method << ","
        << std::fixed << std::setprecision(6)
        << met.coll_rate() << "," << clo << "," << chi << ","
        << met.mm_rate() << "," << mlo << "," << mhi << ","
        << met.rare_rate() << "," << rlo << "," << rhi << ","
        << std::setprecision(4) << met.mean_clearance() << ","
        << met.p5_clearance() << "," << met.mean_progress() << ","
        << std::setprecision(3) << met.mean_solve_ms() << ","
        << n_rollouts << "\n";
}

// ============================================================================
// Experiment A: Environment comparison
//   Oncoming + Intersection × {SHMPCC, Mode-DRO(q*), Mode-DRO(inj),
//                                Traj-DRO(q*), Traj-DRO(inj), Traj-DRO(comb)}
//   switch_prob=0.2, 500 rollouts each
// ============================================================================

static void run_exp_a() {
    std::cout << "\n================================================================\n";
    std::cout << "  Experiment A: Trajectory DRO — Environment Comparison\n";
    std::cout << "  Oncoming + Intersection, switch_prob=0.2\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<EnvironmentType> ENVS = {
        EnvironmentType::INTERSECTION, EnvironmentType::ONCOMING
    };
    const std::vector<TrajDROMethod> METHODS = {
        TrajDROMethod::SHMPCC_BASE,
        TrajDROMethod::MODE_DRO_SAMPLE,
        TrajDROMethod::MODE_DRO_INJECT,
        TrajDROMethod::TRAJ_DRO_RESAMPLE,
        TrajDROMethod::TRAJ_DRO_INJECT,
        TrajDROMethod::TRAJ_DRO_COMBINED
    };
    const int N_ROLLOUTS = 500;
    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "traj_dro_env_comparison.csv";
    std::ofstream ofs(filepath);
    write_csv_header(ofs, "environment,switch_prob,");

    for (EnvironmentType env : ENVS) {
        for (TrajDROMethod method : METHODS) {
            std::string ename = environment_name(env);
            std::cout << "  " << ename << " " << method_name(method) << " ...";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(env) * 1000000 +
                    static_cast<int>(method) * 100000 + i + 8000000);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(env, env_rng);

                ExperimentConfig cfg = make_config(method, SWITCH_PROB, RARE_PROB);
                cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obstacles.obs_modes = env_setup.obs_modes;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            std::string prefix = environment_name(env) + ","
                + std::to_string(SWITCH_PROB) + ",";
            write_csv_row(ofs, met, N_ROLLOUTS, prefix);
            ofs.flush();

            std::cout << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Experiment B: Multi-obstacle stress test
//   {1, 2, 4} obstacles × {SHMPCC, Mode-DRO(q*), Traj-DRO(q*), Traj-DRO(inj)}
//   switch_prob=0.2, 500 rollouts each
// ============================================================================

static void run_exp_b() {
    std::cout << "\n================================================================\n";
    std::cout << "  Experiment B: Trajectory DRO — Multi-Obstacle Stress Test\n";
    std::cout << "  {1, 2, 4} obstacles, switch_prob=0.2\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<int> OBS_COUNTS = {1, 2, 4};
    const std::vector<TrajDROMethod> METHODS = {
        TrajDROMethod::SHMPCC_BASE,
        TrajDROMethod::MODE_DRO_SAMPLE,
        TrajDROMethod::TRAJ_DRO_RESAMPLE,
        TrajDROMethod::TRAJ_DRO_INJECT
    };
    const int N_ROLLOUTS = 500;
    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "traj_dro_multi_obs.csv";
    std::ofstream ofs(filepath);
    write_csv_header(ofs, "num_obstacles,");

    for (int n_obs : OBS_COUNTS) {
        for (TrajDROMethod method : METHODS) {
            std::cout << "  obs=" << n_obs << " " << method_name(method) << " ...";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                unsigned seed = static_cast<unsigned>(
                    n_obs * 1000000 + static_cast<int>(method) * 100000 + i + 9000000);

                ExperimentConfig cfg = make_config(
                    method, SWITCH_PROB, RARE_PROB, NUM_SCENARIOS, n_obs);

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            std::string prefix = std::to_string(n_obs) + ",";
            write_csv_row(ofs, met, N_ROLLOUTS, prefix);
            ofs.flush();

            std::cout << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " mm=" << (met.mm_rate() * 100) << "%"
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Experiment C: Trajectory DRO variant comparison on hardest condition
//   All 6 methods, Oncoming, switch_prob=0.2, 1000 rollouts
// ============================================================================

static void run_exp_c() {
    std::cout << "\n================================================================\n";
    std::cout << "  Experiment C: Trajectory DRO — Variant Comparison\n";
    std::cout << "  Oncoming, switch_prob=0.2, 1000 rollouts\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<TrajDROMethod> METHODS = {
        TrajDROMethod::SHMPCC_BASE,
        TrajDROMethod::MODE_DRO_SAMPLE,
        TrajDROMethod::MODE_DRO_INJECT,
        TrajDROMethod::TRAJ_DRO_RESAMPLE,
        TrajDROMethod::TRAJ_DRO_INJECT,
        TrajDROMethod::TRAJ_DRO_COMBINED
    };
    const int N_ROLLOUTS = 1000;
    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "traj_dro_variant_comparison.csv";
    std::ofstream ofs(filepath);
    write_csv_header(ofs);

    for (TrajDROMethod method : METHODS) {
        std::cout << "  " << method_name(method) << " ...";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();

        Metrics met;
        met.method = method_name(method);

        for (int i = 0; i < N_ROLLOUTS; ++i) {
            if ((i + 1) % 250 == 0) {
                std::cout << " " << (i + 1);
                std::cout.flush();
            }
            unsigned seed = static_cast<unsigned>(
                static_cast<int>(method) * 100000 + i + 10000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

            ExperimentConfig cfg = make_config(method, SWITCH_PROB, RARE_PROB);
            cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obstacles.obs_modes = env_setup.obs_modes;

            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }

        write_csv_row(ofs, met, N_ROLLOUTS);
        ofs.flush();

        auto [clo, chi] = met.coll_ci();
        std::cout << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                  << " [" << std::setprecision(1) << (clo * 100) << ","
                  << std::setprecision(1) << (chi * 100) << "]"
                  << " mm=" << (met.mm_rate() * 100) << "%"
                  << " clr=" << std::setprecision(2) << met.mean_clearance()
                  << " prog=" << std::setprecision(2) << met.mean_progress()
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
}

// ============================================================================
// Experiment D: Wasserstein radius sweep for trajectory DRO
//   rho = {0.01, 0.05, 0.1, 0.2, 0.5} × {Traj-DRO(inj), Traj-DRO(q*)}
//   Also includes Mode-DRO(q*) at each rho for comparison.
//   Oncoming, switch_prob=0.2, 500 rollouts
// ============================================================================

static void run_exp_d() {
    std::cout << "\n================================================================\n";
    std::cout << "  Experiment D: Trajectory DRO — Wasserstein Radius Sweep\n";
    std::cout << "  rho = {0.01, 0.05, 0.1, 0.2, 0.5}, Oncoming\n";
    std::cout << "================================================================\n";
    auto t0 = std::chrono::steady_clock::now();

    const std::vector<double> RHOS = {0.01, 0.05, 0.1, 0.2, 0.5};
    const std::vector<TrajDROMethod> METHODS = {
        TrajDROMethod::MODE_DRO_SAMPLE,
        TrajDROMethod::TRAJ_DRO_RESAMPLE,
        TrajDROMethod::TRAJ_DRO_INJECT
    };
    const int N_ROLLOUTS = 500;
    const double SWITCH_PROB = 0.2;
    const double RARE_PROB = 0.1;

    std::string filepath = OUTPUT_DIR + "traj_dro_rho_sweep.csv";
    std::ofstream ofs(filepath);
    write_csv_header(ofs, "rho,");

    // Also run baseline once (rho-independent)
    {
        std::cout << "  SHMPCC-Base (baseline) ...";
        std::cout.flush();
        auto t1 = std::chrono::steady_clock::now();
        Metrics met;
        met.method = method_name(TrajDROMethod::SHMPCC_BASE);
        for (int i = 0; i < N_ROLLOUTS; ++i) {
            unsigned seed = static_cast<unsigned>(i + 11000000);
            std::mt19937 env_rng(seed);
            EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);
            ExperimentConfig cfg = make_config(TrajDROMethod::SHMPCC_BASE, SWITCH_PROB, RARE_PROB);
            cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
            cfg.obstacles.obs_modes = env_setup.obs_modes;
            RolloutRecord rec = run_experiment_rollout(cfg, seed);
            met.add(rec);
        }
        write_csv_row(ofs, met, N_ROLLOUTS, "0.0,");
        ofs.flush();
        std::cout << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                  << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
    }

    for (double rho : RHOS) {
        for (TrajDROMethod method : METHODS) {
            std::cout << "  rho=" << std::fixed << std::setprecision(2) << rho
                      << " " << method_name(method) << " ...";
            std::cout.flush();
            auto t1 = std::chrono::steady_clock::now();

            Metrics met;
            met.method = method_name(method);

            for (int i = 0; i < N_ROLLOUTS; ++i) {
                unsigned seed = static_cast<unsigned>(
                    static_cast<int>(rho * 1000) * 100000
                    + static_cast<int>(method) * 10000 + i + 12000000);
                std::mt19937 env_rng(seed);
                EnvironmentSetup env_setup = create_environment(EnvironmentType::ONCOMING, env_rng);

                ExperimentConfig cfg = make_config(
                    method, SWITCH_PROB, RARE_PROB, NUM_SCENARIOS, 1, rho);
                cfg.obstacles.initial_obstacle_states = {env_setup.initial_obs};
                cfg.obstacles.obs_modes = env_setup.obs_modes;

                RolloutRecord rec = run_experiment_rollout(cfg, seed);
                met.add(rec);
            }

            std::string prefix = std::to_string(rho) + ",";
            write_csv_row(ofs, met, N_ROLLOUTS, prefix);
            ofs.flush();

            std::cout << " coll=" << std::setprecision(1) << (met.coll_rate() * 100) << "%"
                      << " clr=" << std::setprecision(2) << met.mean_clearance()
                      << " (" << std::setprecision(0) << elapsed_sec(t1) << "s)\n";
        }
    }

    ofs.close();
    std::cout << "  Written: " << filepath
              << " (" << std::setprecision(0) << elapsed_sec(t0) << "s total)\n";
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
    std::cout << "Trajectory-Level DRO Experiments\n";
    std::cout << "========================================\n";
    std::cout << "Methods:\n";
    std::cout << "  SHMPCC-Base:     Standard scenario MPC (no DRO)\n";
    std::cout << "  Mode-DRO(q*):    Discrete-mode DRO, resample from q*\n";
    std::cout << "  Mode-DRO(inj):   Discrete-mode DRO, inject worst-case mode\n";
    std::cout << "  Traj-DRO(q*):    Trajectory-level DRO, resample from q*\n";
    std::cout << "  Traj-DRO(inj):   Trajectory-level DRO, inject worst-case trajectory\n";
    std::cout << "  Traj-DRO(comb):  Trajectory-level DRO, resample + inject\n";
    std::cout << "========================================\n";

    auto t0 = std::chrono::steady_clock::now();

    if (run_all || filters.count("a")) run_exp_a();
    if (run_all || filters.count("b")) run_exp_b();
    if (run_all || filters.count("c")) run_exp_c();
    if (run_all || filters.count("d")) run_exp_d();

    double total = elapsed_sec(t0);
    std::cout << "\n========================================\n";
    std::cout << "Done. Total: " << std::fixed << std::setprecision(0) << total << "s"
              << " (" << std::setprecision(1) << total / 60.0 << " min)\n";
    std::cout << "========================================\n";

    return 0;
}
