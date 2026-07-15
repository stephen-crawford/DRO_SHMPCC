/**
 * @file wasserstein_dro.cpp
 * @brief Implementation of Wasserstein DRO for scenario MPC.
 */

#include "wasserstein_dro.hpp"
#include "collision_constraints.hpp"
#include "primal_ot.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// One-sided normal quantile z_alpha for common alpha values.
double normal_quantile(double alpha) {
    if (std::abs(alpha - 0.90) < 1e-9)  return 1.2815515655446004;
    if (std::abs(alpha - 0.95) < 1e-9)  return 1.6448536269514722;
    if (std::abs(alpha - 0.975) < 1e-9) return 1.959963984540054;
    if (std::abs(alpha - 0.99) < 1e-9)  return 2.3263478740408408;
    // Fallback: 0.975 quantile
    return 1.959963984540054;
}

// CVaR (expected-shortfall) coefficient k_alpha = phi(z_alpha) / (1 - alpha) for a
// standard normal, z_alpha = Phi^{-1}(alpha). It is the tail-mean beyond the VaR
// quantile, so k_alpha > z_alpha (e.g. 2.063 vs 1.645 at alpha=0.95). Using it in
// place of the one-sided quantile makes the directional safety margin
// R + k_alpha * sigma_dir the coherent, tail-aware CVaR analogue of the VaR margin.
// Formal model: mpc-template-python/mpc_template/modules/samplers/CVAR_RISK_MODEL.md.
double cvar_coefficient(double alpha) {
    const double z = normal_quantile(alpha);
    const double two_pi = 6.283185307179586;
    const double phi = std::exp(-0.5 * z * z) / std::sqrt(two_pi);
    return phi / (1.0 - alpha);
}

// Standard normal pdf / cdf (cdf via erfc, no lookup table).
double normal_pdf(double t) {
    const double two_pi = 6.283185307179586;
    return std::exp(-0.5 * t * t) / std::sqrt(two_pi);
}
double normal_cdf(double t) {
    return 0.5 * std::erfc(-t / std::sqrt(2.0));
}

// CVaR_alpha of the CLAMPED Gaussian violation [V]_+ , V ~ N(mu, sigma^2).
//
// The naive form [CVaR_alpha(V)]_+ is WRONG: CVaR is a tail MEAN, not a quantile,
// so it does not commute with the clamp. Jensen gives [E[.]]_+ <= E[[.]_+], so the
// naive form UNDERSTATES, and does so exactly in the rare-but-dangerous regime --
// a mode that is safe at level alpha but has a real collision tail gets reported as
// risk 0 and receives zero WDRO mass. (VaR is a quantile and DOES commute, which is
// why the SURROGATE_VAR path can clamp last and still be exact.)
//
// Correct value, using CVaR_a(Z) = (1/(1-a)) * int_a^1 VaR_u(Z) du and
// VaR_u([V]_+) = [VaR_u(V)]_+ :
//   if VaR_alpha(V) = mu + z_a*sigma >= 0 : the clamp never binds in the tail, so
//       CVaR_alpha([V]_+) = CVaR_alpha(V) = mu + k_a*sigma
//   else, with t = -mu/sigma :
//       CVaR_alpha([V]_+) = ( sigma*phi(t) - (-mu)*(1 - Phi(t)) ) / (1 - alpha)
// The two branches agree at mu = -z_a*sigma (both give (k_a - z_a)*sigma), so the
// function is continuous. Verified against 20M-sample Monte Carlo to 4 decimals.
double cvar_clamped_gaussian(double mu, double sigma, double alpha) {
    if (sigma <= 0.0) return std::max(mu, 0.0);
    const double z = normal_quantile(alpha);
    if (mu + z * sigma >= 0.0) {
        return mu + cvar_coefficient(alpha) * sigma;
    }
    const double t = -mu / sigma;
    const double val = (sigma * normal_pdf(t) - (-mu) * (1.0 - normal_cdf(t))) / (1.0 - alpha);
    return std::max(val, 0.0);
}

// Empirical VaR_alpha: the alpha-quantile. `v` is modified (partially sorted).
double empirical_var(std::vector<double>& v, double alpha) {
    if (v.empty()) return 0.0;
    const size_t idx = static_cast<size_t>(
        std::clamp(alpha * static_cast<double>(v.size()), 0.0,
                   static_cast<double>(v.size() - 1)));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

// Empirical CVaR_alpha via Rockafellar-Uryasev:
//     CVaR_a(Z) = q + E[(Z - q)^+] / (1 - a),   q = VaR_a(Z).
// This is correct even when the distribution has an ATOM at q -- which is exactly
// the case here, since [.]_+ piles mass at 0 and the alpha-quantile often lands on
// it. The naive "mean of samples >= q" estimator silently returns E[Z] in that
// case (every sample satisfies Z >= 0) and is badly wrong.
double empirical_cvar(std::vector<double>& v, double alpha) {
    if (v.empty()) return 0.0;
    const double q = empirical_var(v, alpha);
    double excess = 0.0;
    for (double x : v) excess += std::max(x - q, 0.0);
    excess /= static_cast<double>(v.size());
    return q + excess / (1.0 - alpha);
}

// Safe unit vector: returns (1,0) if input is near-zero.
Eigen::Vector2d safe_unit(const Eigen::Vector2d& v, double eps = 1e-12) {
    const double n = v.norm();
    if (n < eps) return Eigen::Vector2d(1.0, 0.0);
    return v / n;
}

}  // anonymous namespace

namespace scenario_mpc {

WassersteinDRO::WassersteinDRO(const DROConfig& config)
    : config_(config) {}

DROResult WassersteinDRO::compute_worst_case_weights(
    const std::map<std::string, double>& nominal_weights,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<EgoState>& ego_ref_traj,
    int horizon,
    double ego_r,
    double obs_r,
    double margin,
    int risk_horizon,
    int num_discs,
    double vehicle_length
) {
    DROResult result;

    // Collect mode IDs in consistent order
    std::vector<std::string> mode_ids;
    for (const auto& [id, _] : nominal_weights) {
        mode_ids.push_back(id);
    }

    if (mode_ids.empty()) {
        result.worst_case_weights = nominal_weights;
        return result;
    }

    // Update entropy for adaptive rho
    entropy_ = 0.0;
    for (const auto& [_, w] : nominal_weights) {
        if (w > 1e-12) {
            entropy_ -= w * std::log(w);
        }
    }
    max_entropy_ = std::log(static_cast<double>(mode_ids.size()));
    if (max_entropy_ < 1e-12) max_entropy_ = 1.0;

    // Step 1: Compute transport cost matrix D[i][j]
    result.transport_cost_matrix = compute_transport_cost_matrix(
        obs_state, mode_models, mode_ids, horizon
    );

    // Step 2: Compute risk vector r[m] (truncated to risk_horizon if set)
    double safety_threshold = ego_r + obs_r + margin;
    int effective_risk_horizon = (risk_horizon > 0) ? risk_horizon : horizon;
    result.risk_per_mode = compute_risk_vector(
        obs_state, mode_models, mode_ids, ego_ref_traj,
        effective_risk_horizon, safety_threshold, num_discs, vehicle_length
    );

    // Step 3: Get adaptive Wasserstein radius rho
    double rho = get_adaptive_rho();
    result.rho_used = rho;

    // Step 4: Solve Kantorovich dual
    auto [opt_lambda, dual_val] = solve_kantorovich_dual(
        nominal_weights, result.risk_per_mode,
        result.transport_cost_matrix, mode_ids, rho
    );
    result.optimal_lambda = opt_lambda;
    result.worst_case_risk = dual_val;

    // Step 5: Recover Q*.
    //
    // Default path: dual-guided bracketing + plan mixing (a heuristic primal
    // recovery restricted to convex mixtures of two deterministic transport
    // plans). Opt-in path (env USE_PRIMAL_OT=1): the TRUE Wasserstein-metric
    // reweighting -- solve the primal OT LP exactly, allowing fractional
    // source-splits. See src/primal_ot.cpp.
    const char* use_primal_ot = std::getenv("USE_PRIMAL_OT");
    if (use_primal_ot && use_primal_ot[0] == '1') {
        PrimalOTResult ot = solve_primal_ot(
            nominal_weights, result.risk_per_mode,
            result.transport_cost_matrix, mode_ids, rho);
        if (ot.solved) {
            result.worst_case_weights = std::move(ot.q);
            result.implied_transport_cost = ot.transport_cost;
            result.recovery_feasible = (ot.transport_cost <= rho + 1e-6);
            return result;
        }
        // Fall through to the heuristic recovery if the LP failed.
    }

    auto recovery = recover_feasible_qstar(
        nominal_weights, result.risk_per_mode,
        result.transport_cost_matrix, mode_ids, rho
    );
    result.worst_case_weights = std::move(recovery.q_star);
    result.implied_transport_cost = recovery.implied_transport_cost;
    result.recovery_feasible = recovery.feasible;

    return result;
}

Scenario WassersteinDRO::generate_worst_case_scenario(
    const DROResult& dro_result,
    int obstacle_id,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    int horizon,
    int scenario_id
) {
    // Find mode m* = argmax_m Q*[m]  (highest weight under worst-case dist)
    std::string worst_mode;
    double max_weight = -1.0;
    for (const auto& [mode_id, w] : dro_result.worst_case_weights) {
        if (w > max_weight) {
            max_weight = w;
            worst_mode = mode_id;
        }
    }

    // If no risk or mode not found, return empty scenario
    if (worst_mode.empty() || dro_result.worst_case_risk < 1e-12 ||
        mode_models.find(worst_mode) == mode_models.end()) {
        return Scenario(scenario_id, {}, 0.0);
    }

    const ModeModel& mode = mode_models.at(worst_mode);

    // Forward-propagate deterministically (no noise = mean trajectory)
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    Eigen::Vector4d x = obs_state.to_array();
    Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();

    // k=0: current position
    steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

    for (int k = 0; k < horizon; ++k) {
        x = mode.A * x + mode.b;
        cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();
        steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));
    }

    // Build scenario with a single obstacle trajectory
    ObstacleTrajectory traj(obstacle_id, worst_mode, steps, max_weight);
    std::map<int, ObstacleTrajectory> trajs;
    trajs[obstacle_id] = traj;

    return Scenario(scenario_id, trajs, max_weight);
}

Scenario WassersteinDRO::generate_adversarial_scenario(
    const DROResult& dro_result,
    int obstacle_id,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<EgoState>& ego_ref,
    int horizon,
    int scenario_id,
    double sigma_scale
) {
    // Find worst-case mode m* = argmax_m Q*[m]
    std::string worst_mode;
    double max_weight = -1.0;
    for (const auto& [mode_id, w] : dro_result.worst_case_weights) {
        if (w > max_weight) {
            max_weight = w;
            worst_mode = mode_id;
        }
    }

    if (worst_mode.empty() || dro_result.worst_case_risk < 1e-12 ||
        mode_models.find(worst_mode) == mode_models.end()) {
        return Scenario(scenario_id, {}, 0.0);
    }

    const ModeModel& mode = mode_models.at(worst_mode);

    // Propagate mean trajectory and covariances
    auto means = propagate_mode_mean(obs_state, mode, horizon);
    auto covs = propagate_mode_covariance(mode, horizon);

    // Build adversarial trajectory: push obstacle toward ego along uncertain axis
    std::vector<PredictionStep> steps;
    steps.reserve(horizon + 1);

    // k=0: current position (no perturbation)
    steps.emplace_back(0, means[0], covs[0]);

    for (int k = 1; k <= horizon; ++k) {
        Eigen::Vector2d ego_pos;
        if (k < static_cast<int>(ego_ref.size())) {
            ego_pos = ego_ref[k].position();
        } else if (!ego_ref.empty()) {
            ego_pos = ego_ref.back().position();
        } else {
            // Fallback: just use mean trajectory
            steps.emplace_back(k, means[k], covs[k]);
            continue;
        }

        // Approach direction: from obstacle mean toward ego
        Eigen::Vector2d diff = ego_pos - means[k];
        double dist = diff.norm();

        if (dist < 1e-6) {
            // Already on top of ego, no direction to push
            steps.emplace_back(k, means[k], covs[k]);
            continue;
        }

        Eigen::Vector2d approach_dir = diff / dist;

        // Project covariance onto approach direction: sigma_along = sqrt(v^T * Cov * v)
        double var_along = (approach_dir.transpose() * covs[k] * approach_dir)(0, 0);
        double sigma_along = std::sqrt(std::max(0.0, var_along));

        // Adversarial position: push mean toward ego by sigma_scale * sigma_along
        Eigen::Vector2d adv_pos = means[k] + sigma_scale * sigma_along * approach_dir;

        steps.emplace_back(k, adv_pos, covs[k]);
    }

    ObstacleTrajectory traj(obstacle_id, worst_mode, steps, max_weight);
    std::map<int, ObstacleTrajectory> trajs;
    trajs[obstacle_id] = traj;

    return Scenario(scenario_id, trajs, max_weight);
}

int WassersteinDRO::effective_support(const DROResult& dro_result, double threshold) {
    int count = 0;
    for (const auto& [mode, w] : dro_result.worst_case_weights) {
        if (w > threshold) ++count;
    }
    return count;
}

std::vector<Scenario> WassersteinDRO::generate_topk_worst_case_scenarios(
    const DROResult& dro_result,
    int obstacle_id,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    int horizon,
    int base_scenario_id,
    int K
) {
    // Sort modes by Q* weight descending
    std::vector<std::pair<std::string, double>> sorted_modes;
    for (const auto& [mode_id, w] : dro_result.worst_case_weights) {
        if (w > 1e-12 && mode_models.find(mode_id) != mode_models.end()) {
            sorted_modes.push_back({mode_id, w});
        }
    }
    std::sort(sorted_modes.begin(), sorted_modes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    int n_inject = (K < 0) ? static_cast<int>(sorted_modes.size())
                           : std::min(K, static_cast<int>(sorted_modes.size()));

    std::vector<Scenario> results;
    results.reserve(n_inject);

    for (int i = 0; i < n_inject; ++i) {
        const auto& [mode_id, weight] = sorted_modes[i];
        const ModeModel& mode = mode_models.at(mode_id);

        std::vector<PredictionStep> steps;
        steps.reserve(horizon + 1);

        Eigen::Vector4d x = obs_state.to_array();
        Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
        steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

        for (int k = 0; k < horizon; ++k) {
            x = mode.A * x + mode.b;
            cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();
            steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));
        }

        ObstacleTrajectory traj(obstacle_id, mode_id, steps, weight);
        std::map<int, ObstacleTrajectory> trajs;
        trajs[obstacle_id] = traj;
        results.emplace_back(base_scenario_id + i, trajs, weight);
    }
    return results;
}

std::vector<Scenario> WassersteinDRO::generate_topk_adversarial_scenarios(
    const DROResult& dro_result,
    int obstacle_id,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<EgoState>& ego_ref,
    int horizon,
    int base_scenario_id,
    int K,
    double sigma_scale
) {
    // Sort modes by Q* weight descending
    std::vector<std::pair<std::string, double>> sorted_modes;
    for (const auto& [mode_id, w] : dro_result.worst_case_weights) {
        if (w > 1e-12 && mode_models.find(mode_id) != mode_models.end()) {
            sorted_modes.push_back({mode_id, w});
        }
    }
    std::sort(sorted_modes.begin(), sorted_modes.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    int n_inject = (K < 0) ? static_cast<int>(sorted_modes.size())
                           : std::min(K, static_cast<int>(sorted_modes.size()));

    std::vector<Scenario> results;
    results.reserve(n_inject);

    for (int i = 0; i < n_inject; ++i) {
        const auto& [mode_id, weight] = sorted_modes[i];
        const ModeModel& mode = mode_models.at(mode_id);

        auto means = propagate_mode_mean(obs_state, mode, horizon);
        auto covs = propagate_mode_covariance(mode, horizon);

        std::vector<PredictionStep> steps;
        steps.reserve(horizon + 1);
        steps.emplace_back(0, means[0], covs[0]);

        for (int k = 1; k <= horizon; ++k) {
            Eigen::Vector2d ego_pos;
            if (k < static_cast<int>(ego_ref.size())) {
                ego_pos = ego_ref[k].position();
            } else if (!ego_ref.empty()) {
                ego_pos = ego_ref.back().position();
            } else {
                steps.emplace_back(k, means[k], covs[k]);
                continue;
            }

            Eigen::Vector2d diff = ego_pos - means[k];
            double dist = diff.norm();
            if (dist < 1e-6) {
                steps.emplace_back(k, means[k], covs[k]);
                continue;
            }

            Eigen::Vector2d approach_dir = diff / dist;
            double var_along = (approach_dir.transpose() * covs[k] * approach_dir)(0, 0);
            double sigma_along = std::sqrt(std::max(0.0, var_along));
            Eigen::Vector2d adv_pos = means[k] + sigma_scale * sigma_along * approach_dir;
            steps.emplace_back(k, adv_pos, covs[k]);
        }

        ObstacleTrajectory traj(obstacle_id, mode_id, steps, weight);
        std::map<int, ObstacleTrajectory> trajs;
        trajs[obstacle_id] = traj;
        results.emplace_back(base_scenario_id + i, trajs, weight);
    }
    return results;
}

std::vector<Scenario> WassersteinDRO::sample_scenarios_from_qstar(
    const DROResult& dro_result,
    int obstacle_id,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    int horizon,
    int num_scenarios,
    std::mt19937& rng,
    int base_scenario_id
) {
    std::vector<Scenario> scenarios;
    scenarios.reserve(num_scenarios);

    // Build CDF from q* for categorical sampling
    std::vector<std::string> mode_ids;
    std::vector<double> weights;
    for (const auto& [id, w] : dro_result.worst_case_weights) {
        mode_ids.push_back(id);
        weights.push_back(w);
    }

    if (mode_ids.empty() || num_scenarios <= 0) {
        return scenarios;
    }

    std::discrete_distribution<int> dist(weights.begin(), weights.end());

    for (int s = 0; s < num_scenarios; ++s) {
        int mode_idx = dist(rng);
        const std::string& sampled_mode = mode_ids[mode_idx];
        double q_weight = weights[mode_idx];

        auto it = mode_models.find(sampled_mode);
        if (it == mode_models.end()) {
            scenarios.emplace_back(base_scenario_id + s, std::map<int, ObstacleTrajectory>{}, 0.0);
            continue;
        }

        const ModeModel& mode = it->second;

        // Forward-propagate mean trajectory with covariance
        std::vector<PredictionStep> steps;
        steps.reserve(horizon + 1);

        Eigen::Vector4d x = obs_state.to_array();
        Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();

        steps.emplace_back(0, x.head<2>(), cov.block<2, 2>(0, 0));

        for (int k = 0; k < horizon; ++k) {
            x = mode.A * x + mode.b;
            cov = mode.A * cov * mode.A.transpose() + mode.G * mode.G.transpose();
            steps.emplace_back(k + 1, x.head<2>(), cov.block<2, 2>(0, 0));
        }

        ObstacleTrajectory traj(obstacle_id, sampled_mode, steps, q_weight);
        std::map<int, ObstacleTrajectory> trajs;
        trajs[obstacle_id] = traj;

        scenarios.emplace_back(base_scenario_id + s, trajs, q_weight);
    }

    return scenarios;
}

void WassersteinDRO::set_rho_override(double rho) {
    rho_override_ = rho;
}
void WassersteinDRO::clear_rho_override() {
    rho_override_.reset();
}

double WassersteinDRO::get_adaptive_rho() const {
    if (rho_override_.has_value()) {
        return std::clamp(*rho_override_, config_.rho_min, config_.rho_max);
    }
    double rho = config_.rho_base;

    if (config_.use_calibrated_radius) {
        // Confidence-calibrated simplex-concentration radius.
        //
        // The nominal belief p_hat is an empirical categorical distribution over
        // N behaviour modes estimated from n observed interactions. It concentrates
        // in L1 as  P(||p_hat - p*||_1 >= eps) <= 2^N exp(-n eps^2 / 2)  (Devroye),
        // so at target miscoverage beta the L1 half-width is
        //   eps_n(beta) = sqrt( 2 (N ln2 + ln(1/beta)) / n ).
        // With a metric ground cost, W1(p_hat, p*) <= (ground-metric diameter) * eps,
        // the diameter folded into rho_base as the calibration scale. Then p* lies
        // in the ball of radius rho_n(beta) with probability >= 1 - beta, so the
        // reweighted worst-case risk upper-bounds the true risk at confidence 1-beta.
        // Unlike the heuristic rho_base*(1+alpha/sqrt(n))*h_term (which plateaus at
        // ~rho_base), this SHRINKS to rho_min as n -> inf (statistical consistency)
        // and GROWS with the mode count N and the confidence level.
        double N = (max_entropy_ > 1e-12) ? std::exp(max_entropy_) : 1.0;  // modes = exp(log N)
        double n = std::max(1.0, static_cast<double>(observation_count_));
        double beta = std::clamp(config_.confidence_beta, 1e-6, 0.5);
        double eps = std::sqrt(2.0 * (N * std::log(2.0) + std::log(1.0 / beta)) / n);
        rho = config_.rho_min + config_.rho_base * eps;
    } else if (config_.adaptive_rho) {
        // Legacy heuristic: scale up with fewer observations (more uncertainty)
        double n_term = 1.0;
        if (observation_count_ > 0) {
            n_term = 1.0 + config_.confidence_alpha / std::sqrt(
                static_cast<double>(observation_count_));
        } else {
            n_term = 1.0 + config_.confidence_alpha;  // max uncertainty
        }

        // Scale up with higher entropy (more uniform = more uncertain)
        double h_ratio = (max_entropy_ > 1e-12) ? entropy_ / max_entropy_ : 0.0;
        double h_term = 1.0 + config_.entropy_gamma * h_ratio;

        rho = config_.rho_base * n_term * h_term;
    }

    return std::clamp(rho, config_.rho_min, config_.rho_max);
}

void WassersteinDRO::update_prediction_error(double error) {
    (void)error;  // Reserved for future adaptive epsilon refinement
}

void WassersteinDRO::set_observation_count(int n) {
    observation_count_ = n;
}

// ============================================================================
// Private methods
// ============================================================================

std::vector<std::vector<double>> WassersteinDRO::compute_transport_cost_matrix(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    int horizon
) {
    int M = static_cast<int>(mode_ids.size());
    std::vector<std::vector<double>> D(M, std::vector<double>(M, 0.0));

    // Precompute per-mode mean trajectories and covariances
    struct ModeTrajectoryData {
        std::vector<Eigen::Vector2d> means;
        std::vector<Eigen::Matrix2d> covs;
    };
    std::vector<ModeTrajectoryData> mode_data(M);

    for (int i = 0; i < M; ++i) {
        auto it = mode_models.find(mode_ids[i]);
        if (it == mode_models.end()) continue;
        mode_data[i].means = propagate_mode_mean(obs_state, it->second, horizon);
        mode_data[i].covs = propagate_mode_covariance(it->second, horizon);
    }

    // Branch on ground cost type
    if (config_.ground_cost_type == DROGroundCostType::ZERO_ONE) {
        // 0/1 cost: D[i][j] = (i != j) ? 1 : 0
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                D[i][j] = (i != j) ? 1.0 : 0.0;
            }
        }
    } else if (config_.ground_cost_type == DROGroundCostType::EUCLIDEAN_MEAN) {
        // Euclidean mean: ||mu_i - mu_j|| averaged over horizon
        for (int i = 0; i < M; ++i) {
            for (int j = i + 1; j < M; ++j) {
                double sum_dist = 0.0;
                int count = 0;
                int n_steps = std::min(
                    static_cast<int>(mode_data[i].means.size()),
                    static_cast<int>(mode_data[j].means.size()));

                for (int k = 1; k < n_steps; ++k) {
                    sum_dist += (mode_data[i].means[k] - mode_data[j].means[k]).norm();
                    count++;
                }

                double avg_dist = (count > 0) ? sum_dist / count : 0.0;
                D[i][j] = avg_dist;
                D[j][i] = avg_dist;
            }
        }
    } else {
        // Default: W2 Bures metric
        for (int i = 0; i < M; ++i) {
            for (int j = i + 1; j < M; ++j) {
                double sum_w2 = 0.0;
                int count = 0;
                int n_steps = std::min(
                    static_cast<int>(mode_data[i].means.size()),
                    static_cast<int>(mode_data[j].means.size()));

                for (int k = 1; k < n_steps; ++k) {
                    double w2 = gaussian_w2_2d(
                        mode_data[i].means[k], mode_data[i].covs[k],
                        mode_data[j].means[k], mode_data[j].covs[k]
                    );
                    sum_w2 += w2;
                    count++;
                }

                double avg_w2 = (count > 0) ? sum_w2 / count : 0.0;
                D[i][j] = avg_w2;
                D[j][i] = avg_w2;
            }
        }
    }

    return D;
}

std::map<std::string, double> WassersteinDRO::compute_risk_vector(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    const std::vector<EgoState>& ego_ref_traj,
    int horizon,
    double safety_radius,
    int num_discs,
    double vehicle_length
) {
    std::map<std::string, double> risk;

    const double alpha = config_.alpha_one_sided;

    // JOINT_VAR / JOINT_CVAR: true risk measure of the joint-horizon Euclidean
    // violation. Dispatch out to the Monte Carlo estimator; the surrogate path
    // below is not used at all.
    if (config_.risk_measure == DRORiskMeasure::JOINT_VAR ||
        config_.risk_measure == DRORiskMeasure::JOINT_CVAR) {
        return compute_risk_vector_joint(obs_state, mode_models, mode_ids,
                                         ego_ref_traj, horizon, safety_radius,
                                         num_discs, vehicle_length);
    }

    // SURROGATE_VAR (default, bit-for-bit master/CDC'26) and SURROGATE_CVAR.
    // z_alpha is only the VaR coefficient; the CVaR branch cannot be expressed as
    // a coefficient swap (see cvar_clamped_gaussian) and is handled at the use site.
    const double z_alpha = normal_quantile(alpha);
    const double sigma_floor = config_.sigma_floor;

    for (const auto& mode_id : mode_ids) {
        auto it = mode_models.find(mode_id);
        if (it == mode_models.end()) {
            risk[mode_id] = 0.0;
            continue;
        }

        auto means = propagate_mode_mean(obs_state, it->second, horizon);
        auto covs = propagate_mode_covariance(it->second, horizon);

        const int n_steps = std::min({static_cast<int>(means.size()),
                                      static_cast<int>(covs.size()),
                                      horizon + 1});

        double max_risk = 0.0;

        for (int k = 1; k < n_steps; ++k) {
            // Ego reference at step k (clamp if ref traj shorter)
            const EgoState& ego_state =
                (k < static_cast<int>(ego_ref_traj.size()))
                    ? ego_ref_traj[k]
                    : ego_ref_traj.back();

            // Disc centers at this step
            std::vector<Eigen::Vector2d> disc_positions;
            if (num_discs > 1) {
                disc_positions = compute_ego_disc_positions(
                    ego_state, num_discs, vehicle_length);
            } else {
                disc_positions = { ego_state.position() };
            }

            // Take worst (most dangerous) disc
            double step_risk = 0.0;

            for (const auto& c_d : disc_positions) {
                const Eigen::Vector2d mu = means[k];
                const Eigen::Matrix2d Sigma = covs[k];

                // Mean distance to disc center
                const Eigen::Vector2d diff = mu - c_d;
                const double dist = diff.norm();

                // Direction from ego disc to obstacle mean
                const Eigen::Vector2d n = safe_unit(diff);

                // Compute directional sigma based on risk mode
                double sigma_dir = 0.0;
                if (config_.risk_mode == DRORiskMode::FULL) {
                    // Directional std dev: sqrt(n^T Sigma n)
                    double var_dir = n.transpose() * Sigma * n;
                    if (!std::isfinite(var_dir) || var_dir < 0.0) var_dir = 0.0;
                    sigma_dir = std::max(std::sqrt(var_dir), sigma_floor);
                }
                // NO_COV and DISTANCE_ONLY: sigma_dir stays 0

                // Linearised violation Vtil ~ N(mu_V, sigma_dir^2), mu_V = R - dist.
                const double mu_V = safety_radius - dist;

                double r_kd;
                if (config_.risk_measure == DRORiskMeasure::SURROGATE_CVAR) {
                    // Correct clamp order: CVaR_a([Vtil]_+), NOT [CVaR_a(Vtil)]_+.
                    r_kd = cvar_clamped_gaussian(mu_V, sigma_dir, alpha);
                } else {
                    // VaR is a quantile and commutes with [.]_+, so clamping last is exact.
                    r_kd = std::max(0.0, mu_V + z_alpha * sigma_dir);
                }

                step_risk = std::max(step_risk, r_kd);
            }

            max_risk = std::max(max_risk, step_risk);
        }

        risk[mode_id] = max_risk;
    }

    return risk;
}

std::map<std::string, double> WassersteinDRO::compute_risk_vector_joint(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    const std::vector<EgoState>& ego_ref_traj,
    int horizon,
    double safety_radius,
    int num_discs,
    double vehicle_length
) {
    std::map<std::string, double> risk;

    const double alpha = config_.alpha_one_sided;
    const int n_samples = std::max(1, config_.joint_risk_samples);
    const bool want_cvar = (config_.risk_measure == DRORiskMeasure::JOINT_CVAR);

    // Precompute ego disc centres per step once -- they are deterministic (the ego
    // reference), so they are shared across modes and samples.
    std::vector<std::vector<Eigen::Vector2d>> disc_centres(horizon + 1);
    for (int k = 1; k <= horizon; ++k) {
        const EgoState& ego_state = (k < static_cast<int>(ego_ref_traj.size()))
                                        ? ego_ref_traj[k]
                                        : ego_ref_traj.back();
        disc_centres[k] = (num_discs > 1)
                              ? compute_ego_disc_positions(ego_state, num_discs, vehicle_length)
                              : std::vector<Eigen::Vector2d>{ ego_state.position() };
    }

    const Eigen::Vector4d x0 = obs_state.to_array();

    for (const auto& mode_id : mode_ids) {
        auto it = mode_models.find(mode_id);
        if (it == mode_models.end()) {
            risk[mode_id] = 0.0;
            continue;
        }
        const ModeModel& mode = it->second;
        const int n_noise = static_cast<int>(mode.G.cols());

        // Common random numbers: reseed per mode from the SAME fixed seed, so every
        // mode sees an identical noise stream. This makes r[m] deterministic across
        // calls (no jitter in the reweighting) and removes MC noise from BETWEEN-mode
        // comparisons, which is all the W1 LP actually consumes.
        std::mt19937_64 rng(config_.joint_risk_seed);
        std::normal_distribution<double> gauss(0.0, 1.0);

        std::vector<double> samples;
        samples.reserve(n_samples);

        for (int s = 0; s < n_samples; ++s) {
            // Roll out the mode's own dynamics: x_{k+1} = A x_k + b + G w_k.
            // Sampling the ROLLOUT (rather than each step's marginal independently)
            // is what makes this JOINT: the correlation A induces across steps is
            // carried exactly, which is precisely what max_k VaR(V_k) throws away.
            Eigen::Vector4d x = x0;
            double worst = 0.0;

            for (int k = 1; k <= horizon; ++k) {
                Eigen::VectorXd w(n_noise);
                for (int i = 0; i < n_noise; ++i) w(i) = gauss(rng);
                x = mode.A * x + mode.b + mode.G * w;

                const Eigen::Vector2d p = x.head<2>();
                for (const auto& c_d : disc_centres[k]) {
                    // TRUE Euclidean violation -- no projection onto the mean
                    // direction, no Gaussian surrogate for the distance.
                    const double viol = safety_radius - (p - c_d).norm();
                    worst = std::max(worst, viol);
                }
            }
            samples.push_back(std::max(worst, 0.0));  // [.]_+ per sample
        }

        risk[mode_id] = want_cvar ? empirical_cvar(samples, alpha)
                                  : empirical_var(samples, alpha);
    }

    return risk;
}

std::vector<Eigen::Vector2d> WassersteinDRO::propagate_mode_mean(
    const ObstacleState& obs_state,
    const ModeModel& mode,
    int horizon
) {
    std::vector<Eigen::Vector2d> means;
    means.reserve(horizon + 1);

    Eigen::Vector4d x = obs_state.to_array();
    means.push_back(x.head<2>());

    for (int k = 0; k < horizon; ++k) {
        x = mode.A * x + mode.b;
        means.push_back(x.head<2>());
    }

    return means;
}

std::vector<Eigen::Matrix2d> WassersteinDRO::propagate_mode_covariance(
    const ModeModel& mode,
    int horizon
) {
    std::vector<Eigen::Matrix2d> covs;
    covs.reserve(horizon + 1);

    Eigen::Matrix4d cov4 = Eigen::Matrix4d::Zero();
    covs.push_back(cov4.block<2, 2>(0, 0));

    for (int k = 0; k < horizon; ++k) {
        cov4 = mode.A * cov4 * mode.A.transpose() + mode.G * mode.G.transpose();
        covs.push_back(cov4.block<2, 2>(0, 0));
    }

    return covs;
}

double WassersteinDRO::gaussian_w2_2d(
    const Eigen::Vector2d& mu1, const Eigen::Matrix2d& cov1,
    const Eigen::Vector2d& mu2, const Eigen::Matrix2d& cov2
) {
    // W2^2 = ||mu1 - mu2||^2 + Bures^2(cov1, cov2)
    // Bures^2 = tr(cov1) + tr(cov2) - 2*tr(sqrt(sqrt(cov1)*cov2*sqrt(cov1)))

    double mean_dist_sq = (mu1 - mu2).squaredNorm();

    // For 2D, use closed-form matrix square root
    Eigen::Matrix2d sqrt_cov1 = matrix_sqrt_2x2(cov1);

    // M = sqrt(cov1) * cov2 * sqrt(cov1)
    Eigen::Matrix2d M = sqrt_cov1 * cov2 * sqrt_cov1;

    // sqrt(M) using closed-form 2x2
    Eigen::Matrix2d sqrt_M = matrix_sqrt_2x2(M);

    double bures_sq = cov1.trace() + cov2.trace() - 2.0 * sqrt_M.trace();
    // Clamp to avoid numerical negatives
    bures_sq = std::max(0.0, bures_sq);

    return std::sqrt(mean_dist_sq + bures_sq);
}

Eigen::Matrix2d WassersteinDRO::matrix_sqrt_2x2(const Eigen::Matrix2d& M) {
    // Closed-form 2x2 matrix square root:
    // sqrt(M) = (M + sqrt(det(M)) * I) / sqrt(tr(M) + 2*sqrt(det(M)))

    double det_M = M.determinant();
    double tr_M = M.trace();

    // Handle near-zero or negative-definite cases
    double sqrt_det = (det_M > 0) ? std::sqrt(det_M) : 0.0;
    double denom_sq = tr_M + 2.0 * sqrt_det;

    if (denom_sq < 1e-15) {
        // M is approximately zero
        return Eigen::Matrix2d::Zero();
    }

    double denom = std::sqrt(denom_sq);
    Eigen::Matrix2d result = (M + sqrt_det * Eigen::Matrix2d::Identity()) / denom;

    return result;
}

std::pair<double, double> WassersteinDRO::solve_kantorovich_dual(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids,
    double rho
) {
    // Binary search on lambda >= 0
    // Dual: g(lambda) = lambda*rho + sum_i w_i * max_j(r_j - lambda*D[i][j])
    // g(lambda) is convex in lambda, we minimize it

    // Find reasonable bounds for lambda
    double r_max = 0.0;
    for (const auto& [_, r] : risk_vector) {
        r_max = std::max(r_max, r);
    }

    if (r_max < 1e-12) {
        // All modes have zero risk — Q* = P_hat
        double val = evaluate_dual(0.0, nominal_weights, risk_vector, D, mode_ids, rho);
        return {0.0, val};
    }

    // Find max transport cost for upper bound on lambda
    double d_min_nonzero = std::numeric_limits<double>::max();
    int M = static_cast<int>(mode_ids.size());
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            if (i != j && D[i][j] > 1e-12) {
                d_min_nonzero = std::min(d_min_nonzero, D[i][j]);
            }
        }
    }

    // lambda_max: above this, the transport cost penalty dominates,
    // so no mass moves. Upper bound: r_max / d_min
    double lambda_max = (d_min_nonzero < std::numeric_limits<double>::max() && d_min_nonzero > 0)
        ? r_max / d_min_nonzero * 2.0
        : r_max * 10.0;
    double lambda_min = 0.0;

    // Binary search: find lambda that minimizes the dual
    // The dual g(lambda) is convex, so we search for the minimum
    constexpr int MAX_ITER = 50;
    double best_lambda = 0.0;
    double best_val = evaluate_dual(0.0, nominal_weights, risk_vector, D, mode_ids, rho);

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double lambda_mid = (lambda_min + lambda_max) / 2.0;
        double val_mid = evaluate_dual(lambda_mid, nominal_weights, risk_vector, D, mode_ids, rho);

        // Check gradient direction using finite difference
        double delta = std::max(1e-8, lambda_mid * 1e-6);
        double val_plus = evaluate_dual(lambda_mid + delta, nominal_weights, risk_vector, D, mode_ids, rho);
        double grad = (val_plus - val_mid) / delta;

        if (val_mid < best_val) {
            best_val = val_mid;
            best_lambda = lambda_mid;
        }

        if (grad > 0) {
            // Increasing: minimum is to the left
            lambda_max = lambda_mid;
        } else {
            // Decreasing: minimum is to the right
            lambda_min = lambda_mid;
        }

        if (lambda_max - lambda_min < 1e-10) break;
    }

    return {best_lambda, best_val};
}

double WassersteinDRO::evaluate_dual(
    double lambda,
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids,
    double rho
) {
    // g(lambda) = lambda*rho + sum_i w_i * max_j(r_j - lambda*D[i][j])
    int M = static_cast<int>(mode_ids.size());
    double sum = lambda * rho;

    for (int i = 0; i < M; ++i) {
        double w_i = nominal_weights.at(mode_ids[i]);
        double max_val = -std::numeric_limits<double>::infinity();

        for (int j = 0; j < M; ++j) {
            double r_j = risk_vector.at(mode_ids[j]);
            double cost = r_j - lambda * D[i][j];
            max_val = std::max(max_val, cost);
        }

        sum += w_i * max_val;
    }

    return sum;
}

// ============================================================================
// Feasible-by-construction Q* recovery via dual-guided bracketing + plan mixing
// ============================================================================

TransportPlan WassersteinDRO::build_plan(
    double lambda,
    TiePolicy tie_policy,
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids
) {
    const int M = static_cast<int>(mode_ids.size());

    // Normalize nominal weights
    std::map<std::string, double> p = nominal_weights;
    double p_total = 0.0;
    for (const auto& [_, w] : p) p_total += w;
    if (p_total > 0.0) {
        for (auto& [_, w] : p) w /= p_total;
    }

    TransportPlan plan;
    plan.lambda = lambda;
    for (const auto& id : mode_ids) plan.q[id] = 0.0;

    for (int i = 0; i < M; ++i) {
        double p_i = p.at(mode_ids[i]);
        if (p_i < 1e-15) continue;

        // Compute scores s_j = r_j - lambda * D[i][j]
        double best_score = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < M; ++j) {
            double s = risk_vector.at(mode_ids[j]) - lambda * D[i][j];
            best_score = std::max(best_score, s);
        }

        // Collect all tied maximizers
        std::vector<int> ties;
        for (int j = 0; j < M; ++j) {
            double s = risk_vector.at(mode_ids[j]) - lambda * D[i][j];
            if (s >= best_score - 1e-12) {
                ties.push_back(j);
            }
        }

        // Break ties according to policy
        int j_star = ties[0];
        if (ties.size() > 1) {
            if (tie_policy == TiePolicy::MIN_COST) {
                double min_d = std::numeric_limits<double>::infinity();
                for (int j : ties) {
                    if (D[i][j] < min_d) {
                        min_d = D[i][j];
                        j_star = j;
                    }
                }
            } else {  // MAX_COST
                double max_d = -1.0;
                for (int j : ties) {
                    if (D[i][j] > max_d) {
                        max_d = D[i][j];
                        j_star = j;
                    }
                }
            }
        }

        plan.q[mode_ids[j_star]] += p_i;
        plan.transport_cost += p_i * D[i][j_star];
    }

    // Compute expected risk
    for (const auto& id : mode_ids) {
        plan.expected_risk += plan.q[id] * risk_vector.at(id);
    }

    return plan;
}

std::pair<TransportPlan, TransportPlan> WassersteinDRO::bracket_plans(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids,
    double rho
) {
    // High-cost plan: lambda=0 with MAX_COST tie-breaking (encourages movement)
    auto plan_hi = build_plan(0.0, TiePolicy::MAX_COST, nominal_weights, risk_vector, D, mode_ids);

    // If already feasible at lambda=0, no bracketing needed
    if (plan_hi.transport_cost <= rho) {
        auto plan_lo = plan_hi;  // Both are the same
        return {plan_lo, plan_hi};
    }

    // Find lambda_large where cost drops below rho
    double lambda_lo = 1.0;
    TransportPlan plan_lo;

    // Compute a reasonable upper bound for lambda
    double r_max = 0.0;
    for (const auto& [_, r] : risk_vector) r_max = std::max(r_max, r);
    double d_min = std::numeric_limits<double>::max();
    int M = static_cast<int>(mode_ids.size());
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
            if (i != j && D[i][j] > 1e-12)
                d_min = std::min(d_min, D[i][j]);
    double lambda_cap = (d_min < std::numeric_limits<double>::max() && d_min > 0)
        ? r_max / d_min * 10.0 : r_max * 100.0;

    for (int attempt = 0; attempt < 30; ++attempt) {
        plan_lo = build_plan(lambda_lo, TiePolicy::MIN_COST, nominal_weights, risk_vector, D, mode_ids);
        if (plan_lo.transport_cost <= rho) break;
        lambda_lo *= 2.0;
        if (lambda_lo > lambda_cap) {
            // Fallback: stay-put plan (q = p, cost = 0)
            plan_lo.lambda = lambda_cap;
            plan_lo.q = nominal_weights;
            double p_total = 0.0;
            for (const auto& [_, w] : plan_lo.q) p_total += w;
            if (p_total > 0.0) for (auto& [_, w] : plan_lo.q) w /= p_total;
            plan_lo.transport_cost = 0.0;
            plan_lo.expected_risk = 0.0;
            for (const auto& id : mode_ids)
                plan_lo.expected_risk += plan_lo.q[id] * risk_vector.at(id);
            break;
        }
    }

    return {plan_lo, plan_hi};
}

void WassersteinDRO::refine_bracket(
    TransportPlan& plan_lo,
    TransportPlan& plan_hi,
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids,
    double rho,
    int max_iter
) {
    // Invariant: plan_lo.transport_cost <= rho <= plan_hi.transport_cost
    for (int iter = 0; iter < max_iter; ++iter) {
        if (std::abs(plan_hi.transport_cost - plan_lo.transport_cost) < 1e-10) break;
        if (std::abs(plan_hi.lambda - plan_lo.lambda) < 1e-12) break;

        double lambda_mid = 0.5 * (plan_lo.lambda + plan_hi.lambda);

        // Build both tie policies at midpoint
        auto plan_min = build_plan(lambda_mid, TiePolicy::MIN_COST,
                                   nominal_weights, risk_vector, D, mode_ids);
        auto plan_max = build_plan(lambda_mid, TiePolicy::MAX_COST,
                                   nominal_weights, risk_vector, D, mode_ids);

        if (plan_min.transport_cost > rho) {
            // Even min-cost plan overshoots: need higher lambda
            plan_hi = plan_min;
        } else if (plan_max.transport_cost < rho) {
            // Even max-cost plan undershoots: need lower lambda
            plan_lo = plan_max;
        } else {
            // rho is between min and max at this lambda: tighten both sides
            plan_lo = plan_min;
            plan_hi = plan_max;
        }
    }
}

WorstCaseRecoveryResult WassersteinDRO::mix_plans_to_radius(
    const TransportPlan& plan_lo,
    const TransportPlan& plan_hi,
    const std::vector<std::string>& mode_ids,
    double rho
) {
    WorstCaseRecoveryResult result;

    if (std::abs(plan_hi.transport_cost - plan_lo.transport_cost) < 1e-15) {
        // Cannot mix; use the low-cost (feasible) plan
        result.q_star = plan_lo.q;
        result.implied_transport_cost = plan_lo.transport_cost;
        result.feasible = true;
        result.mix_alpha = 0.0;
        return result;
    }

    double alpha = (rho - plan_lo.transport_cost) /
                   (plan_hi.transport_cost - plan_lo.transport_cost);
    alpha = std::max(0.0, std::min(1.0, alpha));

    // Mix: q* = alpha * q_hi + (1 - alpha) * q_lo
    for (const auto& id : mode_ids) {
        double q_lo = plan_lo.q.count(id) ? plan_lo.q.at(id) : 0.0;
        double q_hi = plan_hi.q.count(id) ? plan_hi.q.at(id) : 0.0;
        result.q_star[id] = alpha * q_hi + (1.0 - alpha) * q_lo;
    }

    // Normalize for numerical stability
    double total = 0.0;
    for (const auto& [_, w] : result.q_star) total += w;
    if (total > 0.0) {
        for (auto& [_, w] : result.q_star) w /= total;
    }

    result.implied_transport_cost = alpha * plan_hi.transport_cost +
                                    (1.0 - alpha) * plan_lo.transport_cost;
    result.feasible = true;  // Feasible by construction
    result.mix_alpha = alpha;

    return result;
}

WorstCaseRecoveryResult WassersteinDRO::recover_feasible_qstar(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_vector,
    const std::vector<std::vector<double>>& D,
    const std::vector<std::string>& mode_ids,
    double rho
) {
    // Edge case: no modes or all zero risk -> return nominal
    double r_max = 0.0;
    for (const auto& [_, r] : risk_vector) r_max = std::max(r_max, r);
    if (mode_ids.empty() || r_max < 1e-12) {
        WorstCaseRecoveryResult result;
        result.q_star = nominal_weights;
        result.implied_transport_cost = 0.0;
        result.feasible = true;
        result.mix_alpha = 0.0;
        return result;
    }

    // Step 1: Bracket
    auto [plan_lo, plan_hi] = bracket_plans(
        nominal_weights, risk_vector, D, mode_ids, rho);

    // If high-cost plan already feasible, just use it (max risk within budget)
    if (plan_hi.transport_cost <= rho + 1e-12) {
        WorstCaseRecoveryResult result;
        result.q_star = plan_hi.q;
        result.implied_transport_cost = plan_hi.transport_cost;
        result.feasible = true;
        result.mix_alpha = 1.0;
        return result;
    }

    // Step 2: Refine bracket
    refine_bracket(plan_lo, plan_hi, nominal_weights, risk_vector, D, mode_ids, rho);

    // Step 3: Mix to hit budget exactly
    return mix_plans_to_radius(plan_lo, plan_hi, mode_ids, rho);
}

}  // namespace scenario_mpc
