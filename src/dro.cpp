/**
 * @filedro.cpp
 * @brief Implementation of DRO for scenario MPC.
 */

#include "dro.hpp"
#include "collision_constraints.hpp"
#include "primal_ot.hpp"
#include "mode_weights.hpp"
#include "schuurmans_ambiguity.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

// General one-sided normal quantile z_alpha = Phi^{-1}(alpha), by bisection on the
// erfc-based cdf. ~1e-15 after 200 halvings of [-10, 10].
double normal_quantile(double alpha) {
    const double a = std::clamp(alpha, 1e-12, 1.0 - 1e-12);
    double lo = -10.0, hi = 10.0;
    for (int i = 0; i < 200; ++i) {
        const double m = 0.5 * (lo + hi);
        if (0.5 * std::erfc(-m / std::sqrt(2.0)) < a) lo = m; else hi = m;
    }
    return 0.5 * (lo + hi);
}

// CVaR (expected-shortfall) coefficient k_alpha = phi(z_alpha) / (1 - alpha) for a
// standard normal, z_alpha = Phi^{-1}(alpha). It is the tail-mean beyond the VaR
// quantile, so k_alpha > z_alpha (e.g. 2.063 vs 1.645 at alpha=0.95). Using it in
// place of the one-sided quantile makes the directional safety margin
// R + k_alpha * sigma_dir the coherent, tail-aware CVaR analogue of the VaR margin.
double cvar_coefficient(double alpha) {
    const double z = normal_quantile(alpha);
    const double two_pi = 6.283185307179586;
    const double phi = std::exp(-0.5 * z * z) / std::sqrt(two_pi);
    return phi / (1.0 - alpha);
}

double pdf(double t) {
    const double two_pi = 6.283185307179586;
    return std::exp(-0.5 * t * t) / std::sqrt(two_pi);
}
double cdf(double t) {
    return 0.5 * std::erfc(-t / std::sqrt(2.0));
}

// CVaR_alpha of the CLAMPED Gaussian violation [V]_+ , V ~ N(mu, sigma^2).

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

// ---------------------------------------------------------------------------
// Gaussian-mixture risk (MIXTURE_VAR / MIXTURE_CVAR). See the MIXTURE_* note in
// types.hpp for why an equally-weighted mixture is the right object here.
// ---------------------------------------------------------------------------

// E[(X - q)_+] for X ~ N(mu, sigma^2) = (mu-q)Phi((mu-q)/sigma) + sigma*phi((mu-q)/sigma).
double gaussian_upper_partial_mean(double mu, double sigma, double q) {
    if (sigma <= 0.0) return std::max(mu - q, 0.0);
    const double t = (mu - q) / sigma;
    return (mu - q) * normal_cdf(t) + sigma * normal_pdf(t);
}

// VaR_alpha of an equally-weighted Gaussian mixture: the unique q solving
//     (1/K) sum_s Phi((q - mu_s)/sigma_s) = alpha.
// The mixture CDF is continuous and strictly increasing, so bisection converges
// unconditionally. The +-12 sigma bracket covers any alpha representable in double.
// NOTE: returns the UNCLAMPED quantile; callers apply [.]_+ (VaR commutes with it).
double mixture_var(const std::vector<double>& mu, const std::vector<double>& sigma,
                   double alpha) {
    const size_t K = mu.size();
    if (K == 0) return 0.0;
    double lo = std::numeric_limits<double>::infinity();
    double hi = -std::numeric_limits<double>::infinity();
    for (size_t s = 0; s < K; ++s) {
        lo = std::min(lo, mu[s] - 12.0 * std::max(sigma[s], 0.0));
        hi = std::max(hi, mu[s] + 12.0 * std::max(sigma[s], 0.0));
    }
    if (!(lo < hi)) return mu[0];
    const double a = std::clamp(alpha, 1e-12, 1.0 - 1e-12);
    for (int it = 0; it < 200; ++it) {
        const double m = 0.5 * (lo + hi);
        double F = 0.0;
        for (size_t s = 0; s < K; ++s) {
            F += (sigma[s] > 0.0) ? normal_cdf((m - mu[s]) / sigma[s])
                                  : (m >= mu[s] ? 1.0 : 0.0);
        }
        F /= static_cast<double>(K);
        if (F < a) lo = m; else hi = m;
    }
    return 0.5 * (lo + hi);
}

// CVaR_alpha of [V]_+ where V is the equally-weighted Gaussian mixture above.
//
// Rockafellar-Uryasev: CVaR_a(Z) = min_q { q + E[(Z-q)_+]/(1-a) }, attained at
// q* = VaR_a(Z). For Z = [V]_+ we have q* = [VaR_a(V)]_+ >= 0, and for any q >= 0
// ([V]_+ - q)_+ == (V - q)_+ -- so the clamp is handled EXACTLY rather than by the
// invalid [CVaR(V)]_+ shortcut. At K = 1 this reproduces cvar_clamped_gaussian()
// identically (both of its branches fall out of the single formula).
double mixture_cvar_clamped(const std::vector<double>& mu,
                            const std::vector<double>& sigma, double alpha) {
    const size_t K = mu.size();
    if (K == 0) return 0.0;
    const double q = std::max(0.0, mixture_var(mu, sigma, alpha));
    double tail = 0.0;
    for (size_t s = 0; s < K; ++s) tail += gaussian_upper_partial_mean(mu[s], sigma[s], q);
    tail /= static_cast<double>(K);
    return std::max(0.0, q + tail / (1.0 - std::clamp(alpha, 1e-12, 1.0 - 1e-12)));
}

// Safe unit vector: returns (1,0) if input is near-zero.
Eigen::Vector2d safe_unit(const Eigen::Vector2d& v, double eps = 1e-12) {
    const double n = v.norm();
    if (n < eps) return Eigen::Vector2d(1.0, 0.0);
    return v / n;
}

}  // anonymous namespace

namespace dro_mpc {

namespace {

// One row of the entropic plan: Pi_i: = p_i * softmax_j((r_j - lambda D_ij)/tau).
// Computed in log-space (subtract the row max) so large (r - lambda D)/tau does not
// overflow -- at small tau the exponent is O(1/tau) and naive exp() overflows fast.
void entropic_row(
    double p_i, const std::vector<double>& r,
    const std::vector<double>& D_i, double lambda, double tau,
    std::vector<double>& row_out
) {
    const size_t M = r.size();
    row_out.assign(M, 0.0);
    double best = -std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < M; ++j) {
        best = std::max(best, (r[j] - lambda * D_i[j]) / tau);
    }
    double Z = 0.0;
    for (size_t j = 0; j < M; ++j) {
        row_out[j] = std::exp((r[j] - lambda * D_i[j]) / tau - best);
        Z += row_out[j];
    }
    if (!(Z > 0.0) || !std::isfinite(Z)) {          // degenerate: fall back to uniform
        for (size_t j = 0; j < M; ++j) row_out[j] = p_i / static_cast<double>(M);
        return;
    }
    for (size_t j = 0; j < M; ++j) row_out[j] *= p_i / Z;
}

}  // anonymous namespace

EntropicOTResult solve_entropic_ot(
    const std::map<std::string, double>& nominal_weights,
    const std::map<std::string, double>& risk_per_mode,
    const std::vector<std::vector<double>>& transport_cost_matrix,
    const std::vector<std::string>& mode_ids,
    double rho,
    double tau
) {
    EntropicOTResult out;
    const size_t M = mode_ids.size();
    if (M == 0 || transport_cost_matrix.size() != M || !(tau > 0.0)) return out;

    std::vector<double> p(M), r(M);
    for (size_t i = 0; i < M; ++i) {
        auto pit = nominal_weights.find(mode_ids[i]);
        auto rit = risk_per_mode.find(mode_ids[i]);
        p[i] = (pit != nominal_weights.end()) ? pit->second : 0.0;
        r[i] = (rit != risk_per_mode.end()) ? rit->second : 0.0;
    }

    // Transport cost and target marginal at a given lambda.
    std::vector<double> row(M);
    auto evaluate = [&](double lambda, std::vector<double>& q_out) {
        q_out.assign(M, 0.0);
        double cost = 0.0;
        for (size_t i = 0; i < M; ++i) {
            entropic_row(p[i], r, transport_cost_matrix[i], lambda, tau, row);
            for (size_t j = 0; j < M; ++j) {
                q_out[j] += row[j];
                cost += row[j] * transport_cost_matrix[i][j];
            }
        }
        return cost;
    };

    // The transport cost is non-increasing in lambda (raising the price of movement
    // shifts each softmax back toward its own source). Bisect for the smallest
    // lambda >= 0 meeting the budget; lambda = 0 if the budget is already slack.
    std::vector<double> q(M);
    double cost0 = evaluate(0.0, q);
    double lambda = 0.0;
    if (cost0 > rho) {
        double lo = 0.0, hi = 1.0;
        for (int k = 0; k < 60 && evaluate(hi, q) > rho; ++k) hi *= 2.0;  // bracket
        for (int k = 0; k < 200; ++k) {                                   // bisect
            const double mid = 0.5 * (lo + hi);
            (evaluate(mid, q) > rho) ? lo = mid : hi = mid;
        }
        lambda = hi;
    }
    out.transport_cost = evaluate(lambda, q);
    out.lambda = lambda;

    double q_min = std::numeric_limits<double>::infinity(), risk = 0.0, total = 0.0;
    for (size_t j = 0; j < M; ++j) total += q[j];
    if (!(total > 0.0)) return out;
    for (size_t j = 0; j < M; ++j) {
        q[j] /= total;                       // guard against drift
        out.q[mode_ids[j]] = q[j];
        q_min = std::min(q_min, q[j]);
        risk += q[j] * r[j];
    }
    out.q_min = q_min;
    out.expected_risk = risk;
    out.solved = (out.transport_cost <= rho + 1e-6) && (q_min > 0.0);
    return out;
}

WassersteinDRO::WassersteinDRO(const DROConfig& config)
    : config_(config) {}

DROResult WassersteinDRO::compute_worst_case_weights(
    const std::map<std::string, double>& nominal_weights,
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<EgoState>& ego_linearization_traj,
    int horizon,
    double ego_r,
    double obs_r,
    double margin,
    int risk_horizon,
    int num_discs,
    double vehicle_length,
    const Eigen::MatrixXd* transition
) {
    if (ego_linearization_traj.empty()) {
        throw std::invalid_argument(
            "compute_worst_case_weights requires a nonempty "
            "ego linearization trajectory."
        );
    }

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

    // Compute transport cost matrix D[i][j]
    result.transport_cost_matrix = compute_transport_cost_matrix(
        obs_state, mode_models, mode_ids, horizon
    );

    // Compute risk vector r[m]
    double safety_threshold = ego_r + obs_r + margin;
    int effective_risk_horizon = (risk_horizon > 0) ? risk_horizon : horizon;

    const bool have_transition =
        (transition != nullptr &&
         transition->rows() == static_cast<Eigen::Index>(mode_ids.size()) &&
         transition->cols() == static_cast<Eigen::Index>(mode_ids.size()));
    const Eigen::MatrixXd* chain = have_transition ? transition : nullptr;
    const auto risk_measure = config_.radius_calibration.risk_measure;

    if (risk_measure == DRORiskMeasure::MIXTURE_VAR ||
        risk_measure == DRORiskMeasure::MIXTURE_CVAR) {
        result.risk_per_mode = compute_risk_vector_mixture(
            obs_state, mode_models, mode_ids, ego_linearization_traj,
            effective_risk_horizon, safety_threshold, num_discs, vehicle_length,
            chain
        );
    } else if (risk_measure == DRORiskMeasure::JOINT_VAR ||
               risk_measure == DRORiskMeasure::JOINT_CVAR) {
        result.risk_per_mode = compute_risk_vector_joint(
            obs_state, mode_models, mode_ids, ego_linearization_traj,
            effective_risk_horizon, safety_threshold, num_discs, vehicle_length,
            chain
        );
    } else if (have_transition) {
        result.risk_per_mode = compute_risk_vector_switching(
            obs_state, mode_models, mode_ids, ego_linearization_traj,
            effective_risk_horizon, safety_threshold, num_discs, vehicle_length,
            *transition
        );
    } else {
        result.risk_per_mode = compute_risk_vector(
            obs_state, mode_models, mode_ids, ego_linearization_traj,
            effective_risk_horizon, safety_threshold, num_discs, vehicle_length
        );
    }

    // Set ambiguity radius 
    if (config_.radius_calibration.divergence != AmbiguityDivergence::WASSERSTEIN) {
        const int M = static_cast<int>(mode_ids.size());
        std::vector<double> phat(M), xi(M);
        for (int i = 0; i < M; ++i) {
            auto it = nominal_weights.find(mode_ids[i]);
            phat[i] = (it != nominal_weights.end()) ? it->second : 0.0;
            xi[i]   = result.risk_per_mode.count(mode_ids[i]) ? result.risk_per_mode[mode_ids[i]] : 0.0;
        }
        const int m = std::max(1, observation_count_);
        const double beta = std::clamp(config_.radius_calibration.confidence_beta, 1e-6, 0.5);
        double kmax = 0.0;
        for (const auto& row : result.transport_cost_matrix)
            for (double v : row) kmax = std::max(kmax, v);
        const double r = schuurmans::ambiguity_radius(
            config_.radius_calibration.divergence, M, m, beta, kmax);
        schuurmans::WorstCase wc = schuurmans::worst_case_expectation(
            config_.radius_calibration.divergence, phat, xi, r, &result.transport_cost_matrix);
        for (int i = 0; i < M; ++i) result.worst_case_weights[mode_ids[i]] = wc.p[i];
        result.rho_used = r;
        result.worst_case_risk = wc.value;
        result.implied_transport_cost = wc.divergence;
        result.recovery_feasible = wc.feasible;
        double floor_val = std::numeric_limits<double>::infinity(); int support = 0;
        for (const auto& [_, w] : result.worst_case_weights) {
            floor_val = std::min(floor_val, w); if (w > 0.0) ++support;
        }
        result.qstar_support_floor = std::isfinite(floor_val) ? floor_val : 0.0;
        result.qstar_support_size = support;
        result.satisfies_full_support = (support == M) && (result.qstar_support_floor > 0.0);
        return result;
    }

    // Solve Kantorovich dual
    auto [opt_lambda, dual_val] = solve_kantorovich_dual(
        nominal_weights, result.risk_per_mode,
        result.transport_cost_matrix, mode_ids, rho
    );
    result.optimal_lambda = opt_lambda;
    result.worst_case_risk = dual_val;

    // Recover q*.
    
    if (config_.radius_calibration.use_entropic_allocator) {
        EntropicOTResult ent = solve_entropic_ot(
            nominal_weights, result.risk_per_mode,
            result.transport_cost_matrix, mode_ids, rho, config_.radius_calibration.entropic_tau
        );
        if (ent.solved) {
            result.worst_case_weights = ent.q;
            result.implied_transport_cost = ent.transport_cost;
            result.recovery_feasible = true;
            result.optimal_lambda = ent.lambda;
            double floor_val = std::numeric_limits<double>::infinity();
            int support = 0;
            for (const auto& [_, w] : result.worst_case_weights) {
                floor_val = std::min(floor_val, w);
                if (w > 0.0) ++support;
            }
            result.qstar_support_floor = std::isfinite(floor_val) ? floor_val : 0.0;
            result.qstar_support_size = support;
            result.satisfies_full_support =
                (support == static_cast<int>(result.worst_case_weights.size()))
                && (result.qstar_support_floor > 0.0);
            return result;
        }
    
    }
    //
    // Solve the primal OT LP 
    bool want_primal_ot = config_.radius_calibration.use_primal_ot;
    if (const char* env_ot = std::getenv("USE_PRIMAL_OT")) {
        want_primal_ot = (env_ot[0] == '1');
    }
    if (want_primal_ot) {
        PrimalOTResult ot = solve_primal_ot(
            nominal_weights, result.risk_per_mode,
            result.transport_cost_matrix, mode_ids, rho);
        if (ot.solved) {
            result.worst_case_weights = std::move(ot.q);
            result.implied_transport_cost = ot.transport_cost;
            result.recovery_feasible = (ot.transport_cost <= rho + 1e-6);
            double floor_val = std::numeric_limits<double>::infinity();
            int support = 0;
            for (const auto& [_, w] : result.worst_case_weights) {
                floor_val = std::min(floor_val, w);
                if (w > 0.0) ++support;
            }
            result.qstar_support_floor = std::isfinite(floor_val) ? floor_val : 0.0;
            result.qstar_support_size = support;
            result.satisfies_full_support =
                (support == static_cast<int>(result.worst_case_weights.size()))
                && (result.qstar_support_floor > 0.0);
            return result;
        }
        
    }

    auto recovery = recover_feasible_qstar(
        nominal_weights, result.risk_per_mode,
        result.transport_cost_matrix, mode_ids, rho
    );
    result.worst_case_weights = std::move(recovery.q_star);
    result.implied_transport_cost = recovery.implied_transport_cost;
    result.recovery_feasible = recovery.feasible;

    if (!result.worst_case_weights.empty()) {
        double floor_val = std::numeric_limits<double>::infinity();
        int support = 0;
        for (const auto& [_, w] : result.worst_case_weights) {
            floor_val = std::min(floor_val, w);
            if (w > 0.0) ++support;
        }
        result.qstar_support_floor = std::isfinite(floor_val) ? floor_val : 0.0;
        result.qstar_support_size = support;
        result.satisfies_full_support =
            (support == static_cast<int>(result.worst_case_weights.size()))
            && (result.qstar_support_floor > 0.0);
    }

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

    // A mode absent from mode_models has no trajectory to compare against. Track
    // that here rather than letting it silently produce a zero row: D[i][j] = 0
    // for i != j turns D into a pseudometric and makes transport into and out of
    // that mode FREE, so the ball would admit reweightings rho never paid for.
    std::vector<bool> has_traj(M, false);
    for (int i = 0; i < M; ++i) {
        auto it = mode_models.find(mode_ids[i]);
        if (it == mode_models.end()) continue;
        mode_data[i].means = propagate_mode_mean(obs_state, it->second, horizon);
        mode_data[i].covs = propagate_mode_covariance(it->second, horizon);
        has_traj[i] = mode_data[i].means.size() > 1;
    }

    // 0/1 is combinatorial -- it needs no trajectories, so it is exempt from all
    // of the trajectory bookkeeping below.
    if (config_.ground_cost_type == DROGroundCostType::ZERO_ONE) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                D[i][j] = (i != j) ? 1.0 : 0.0;
        return D;
    }

    // All trajectory-based ground costs share one aggregator: the mean over the
    // horizon of a per-step distance between the two modes' position Gaussians.
    //   D_ij = (1/N) sum_{k=1..N} d(P^i_k, P^j_k)
    // k starts at 1 because every mode shares x_0, so step 0 contributes nothing.
    // A mean of metrics is itself a metric, so this aggregation preserves
    // symmetry, D_ii = 0 and the triangle inequality for any metric d.
    auto horizon_mean = [&](int i, int j, auto&& step_distance) {
        const int n_steps = std::min(static_cast<int>(mode_data[i].means.size()),
                                     static_cast<int>(mode_data[j].means.size()));
        double sum = 0.0;
        int count = 0;
        for (int k = 1; k < n_steps; ++k) {
            sum += step_distance(k);
            ++count;
        }
        return (count > 0) ? sum / count : 0.0;
    };

    for (int i = 0; i < M; ++i) {
        for (int j = i + 1; j < M; ++j) {
            if (!has_traj[i] || !has_traj[j]) continue;  // backfilled below

            double d_ij = 0.0;
            switch (config_.ground_cost_type) {
                case DROGroundCostType::EUCLIDEAN_MEAN:
                    d_ij = horizon_mean(i, j, [&](int k) {
                        return (mode_data[i].means[k] - mode_data[j].means[k]).norm();
                    });
                    break;

                case DROGroundCostType::W1_METRIC:
                    d_ij = horizon_mean(i, j, [&](int k) {
                        return sliced_w1_gaussian_2d(
                            mode_data[i].means[k], mode_data[i].covs[k],
                            mode_data[j].means[k], mode_data[j].covs[k]);
                    });
                    break;

                case DROGroundCostType::W2_BURES:
                    d_ij = horizon_mean(i, j, [&](int k) {
                        return gaussian_w2_2d(
                            mode_data[i].means[k], mode_data[i].covs[k],
                            mode_data[j].means[k], mode_data[j].covs[k]);
                    });
                    break;

                case DROGroundCostType::ZERO_ONE:
                    break;  // returned above
            }
            D[i][j] = d_ij;
            D[j][i] = d_ij;
        }
    }

    // Backfill modes with no trajectory at the diameter of the valid submatrix.
    // That is the largest value that still satisfies the triangle inequality
    // everywhere, so it keeps D a metric while making an unknown mode maximally
    // expensive to reach instead of free.
    bool any_missing = false;
    for (int i = 0; i < M; ++i) any_missing |= !has_traj[i];
    if (any_missing) {
        double diameter = 0.0;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < M; ++j)
                if (has_traj[i] && has_traj[j]) diameter = std::max(diameter, D[i][j]);

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                if (i == j) continue;
                if (!has_traj[i] || !has_traj[j]) {
                    D[i][j] = diameter;
                    D[j][i] = diameter;
                }
            }
        }
    }
    return D;
}

std::map<std::string, double> WassersteinDRO::compute_risk_vector(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    const std::vector<EgoState>& ego_linearization_traj,
    int horizon,
    double safety_radius,
    int num_discs,
    double vehicle_length
) {
    std::map<std::string, double> risk;

    const double alpha = config_.radius_calibration.alpha_one_sided;

    // JOINT_VAR / JOINT_CVAR: true risk measure of the joint-horizon Euclidean
    // violation. Dispatch out to the Monte Carlo estimator; the surrogate path
    // below is not used at all.
    if (config_.radius_calibration.risk_measure == DRORiskMeasure::JOINT_VAR ||
        config_.radius_calibration.risk_measure == DRORiskMeasure::JOINT_CVAR) {
        return compute_risk_vector_joint(obs_state, mode_models, mode_ids,
                                         ego_linearization_traj, horizon, safety_radius,
                                         num_discs, vehicle_length);
    }

    // BONFERRONI_VAR: PROPER Bonferroni VaR on the TRUE per-step Euclidean violation
    // (Monte Carlo on each step's marginal Gaussian). Dispatch out; the linearised
    // surrogate path below is not used.
    if (config_.radius_calibration.risk_measure == DRORiskMeasure::BONFERRONI_VAR) {
        return compute_risk_vector_bonferroni(obs_state, mode_models, mode_ids,
                                              ego_linearization_traj, horizon, safety_radius,
                                              num_discs, vehicle_length);
    }


    // Bonferroni: inflate the per-step level to alpha' = 1 - (1-alpha)/(N_s*D) so
    // the union over the N_s*D (step, disc) violation events is controlled at alpha.
    // The number of union terms must match the loops below exactly -- k runs 1..N_s
    // and d runs over the discs -- or the guarantee is void.
    const bool bonferroni = (config_.radius_calibration.risk_measure == DRORiskMeasure::SURROGATE_VAR_BONFERRONI);
    double alpha_eff = alpha;
    if (bonferroni) {
        const double n_events = std::max(1.0, static_cast<double>(horizon) *
                                              static_cast<double>(std::max(1, num_discs)));
        alpha_eff = 1.0 - (1.0 - alpha) / n_events;
    }

    // z_alpha is only the VaR coefficient; the CVaR branch cannot be expressed as
    // a coefficient swap (see cvar_clamped_gaussian) and is handled at the use site.
    const double z_alpha = normal_quantile(alpha_eff);
    const double sigma_floor = config_.radius_calibration.sigma_floor;

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
                (k < static_cast<int>(ego_linearization_traj.size()))
                    ? ego_linearization_traj[k]
                    : ego_linearization_traj.back();

            // Disc centers at this step
            std::vector<Eigen::Vector2d> disc_positions;
            if (num_discs > 1) {
                disc_positions = compute_ego_disc_positions(
                    ego_state, num_discs, vehicle_length);
            } else {
                disc_positions = { ego_state.position() };
            }

            double step_risk = 0.0;

            for (const auto& c_d : disc_positions) {
                const Eigen::Vector2d mu = means[k];
                const Eigen::Matrix2d Sigma = covs[k];

                // Mean distance to disc center
                const Eigen::Vector2d diff = mu - c_d;
                const double dist = diff.norm();

                // Direction from ego disc to obstacle mean
                const Eigen::Vector2d n = safe_unit(diff);

                // Directional std: sqrt(n^T Sigma n). Surrogate risk is always
                // computed from the chosen DRORiskMeasure with this sigma.
                double sigma_dir = 0.0;
                {
                    double var_dir = n.transpose() * Sigma * n;
                    if (!std::isfinite(var_dir) || var_dir < 0.0) var_dir = 0.0;
                    sigma_dir = std::max(std::sqrt(var_dir), sigma_floor);
                }

                // Linearised violation Vtil ~ N(mu_V, sigma_dir^2), mu_V = R - dist.
                const double mu_V = safety_radius - dist;

                double r_kd;
                if (config_.radius_calibration.risk_measure == DRORiskMeasure::SURROGATE_CVAR) {
                    // Correct clamp order: CVaR_a([Vtil]_+), NOT [CVaR_a(Vtil)]_+.
                    r_kd = cvar_clamped_gaussian(mu_V, sigma_dir, alpha);
                } else {
                    // VaR is a quantile and commutes with [.]_+, so clamping last is exact.
                    r_kd = std::max(0.0, mu_V + z_alpha * sigma_dir);
                }

                step_risk = std::max(step_risk, r_kd);
            }

            max_risk = std::max(max_risk, step_risk); // Most dangerous disc's risk 
        }

        risk[mode_id] = max_risk;
    }

    return risk;
}

double WassersteinDRO::surrogate_traj_violation(
    const std::vector<Eigen::Vector2d>& means,
    const std::vector<Eigen::Matrix2d>& covs,
    const std::vector<EgoState>& ego_traj,
    int horizon, double safety_radius, int num_discs, double vehicle_length,
    double z_alpha, double alpha) const
{
    const double sigma_floor = config_.radius_calibration.sigma_floor;
    const int n_steps = std::min({static_cast<int>(means.size()),
                                  static_cast<int>(covs.size()),
                                  horizon + 1});
    double max_risk = 0.0;
    for (int k = 1; k < n_steps; ++k) {
        const EgoState& ego_state =
            (k < static_cast<int>(ego_traj.size())) ? ego_traj[k] : ego_traj.back();
        std::vector<Eigen::Vector2d> disc_positions;
        if (num_discs > 1)
            disc_positions = compute_ego_disc_positions(ego_state, num_discs, vehicle_length);
        else
            disc_positions = { ego_state.position() };

        double step_risk = 0.0;
        for (const auto& c_d : disc_positions) {
            const Eigen::Vector2d mu = means[k];
            const Eigen::Matrix2d Sigma = covs[k];
            const Eigen::Vector2d diff = mu - c_d;
            const double dist = diff.norm();
            const Eigen::Vector2d n = safe_unit(diff);
            double var_dir = n.transpose() * Sigma * n;
            if (!std::isfinite(var_dir) || var_dir < 0.0) var_dir = 0.0;
            const double sigma_dir = std::max(std::sqrt(var_dir), sigma_floor);
            const double mu_V = safety_radius - dist;
            double r_kd;
            if (config_.radius_calibration.risk_measure == DRORiskMeasure::SURROGATE_CVAR)
                r_kd = cvar_clamped_gaussian(mu_V, sigma_dir, alpha);
            else
                r_kd = std::max(0.0, mu_V + z_alpha * sigma_dir);
            step_risk = std::max(step_risk, r_kd);
        }
        max_risk = std::max(max_risk, step_risk);
    }
    return max_risk;
}

std::pair<double, double> WassersteinDRO::surrogate_traj_gaussian(
    const std::vector<Eigen::Vector2d>& means,
    const std::vector<Eigen::Matrix2d>& covs,
    const std::vector<EgoState>& ego_traj,
    int horizon, double safety_radius, int num_discs, double vehicle_length,
    double z_alpha) const
{
    const double sigma_floor = config_.radius_calibration.sigma_floor;
    const int n_steps = std::min({static_cast<int>(means.size()),
                                  static_cast<int>(covs.size()),
                                  horizon + 1});

    // Select the dominant (k,d) by the UNCLAMPED VaR score, and return its Gaussian
    // parameters. Selecting pre-clamp matters: a component with mu << 0 is genuinely
    // safe and must keep its negative mean so the mixture tail is not inflated.
    double best_score = -std::numeric_limits<double>::infinity();
    double best_mu = 0.0;
    double best_sigma = sigma_floor;

    for (int k = 1; k < n_steps; ++k) {
        const EgoState& ego_state =
            (k < static_cast<int>(ego_traj.size())) ? ego_traj[k] : ego_traj.back();
        std::vector<Eigen::Vector2d> disc_positions;
        if (num_discs > 1)
            disc_positions = compute_ego_disc_positions(ego_state, num_discs, vehicle_length);
        else
            disc_positions = { ego_state.position() };

        for (const auto& c_d : disc_positions) {
            const Eigen::Vector2d diff = means[k] - c_d;
            const Eigen::Vector2d n = safe_unit(diff);
            double var_dir = n.transpose() * covs[k] * n;
            if (!std::isfinite(var_dir) || var_dir < 0.0) var_dir = 0.0;
            const double sd = std::max(std::sqrt(var_dir), sigma_floor);
            const double mv = safety_radius - diff.norm();
            const double score = mv + z_alpha * sd;
            if (score > best_score) {
                best_score = score;
                best_mu = mv;
                best_sigma = sd;
            }
        }
    }
    return { best_mu, best_sigma };
}

std::map<std::string, double> WassersteinDRO::compute_risk_vector_mixture(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    const std::vector<EgoState>& ego_linearization_traj,
    int horizon,
    double safety_radius,
    int num_discs,
    double vehicle_length,
    const Eigen::MatrixXd* transition
) {
    std::map<std::string, double> risk;
    const int M = static_cast<int>(mode_ids.size());
    if (M == 0) return risk;

    const double alpha = config_.radius_calibration.alpha_one_sided;
    const double z_alpha = normal_quantile(alpha);
    const bool want_cvar =
        (config_.radius_calibration.risk_measure == DRORiskMeasure::MIXTURE_CVAR);

    const bool switching =
        (transition != nullptr &&
         transition->rows() == static_cast<Eigen::Index>(M) &&
         transition->cols() == static_cast<Eigen::Index>(M));

    // Without a chain the mixture has ONE component (the held mode), which makes
    // MIXTURE_VAR/CVAR collapse exactly onto SURROGATE_VAR/CVAR.
    const int K = switching
        ? std::max(1, config_.radius_calibration.mixture_sequence_samples)
        : 1;

    for (int i = 0; i < M; ++i) {
        const std::string& mode_id = mode_ids[i];
        if (mode_models.find(mode_id) == mode_models.end()) {
            risk[mode_id] = 0.0;
            continue;
        }

        // Point-mass initial belief: r[m] is the danger of STARTING in mode m.
        ModeDistribution e_m;
        for (const auto& mid : mode_ids) e_m[mid] = 0.0;
        e_m[mode_id] = 1.0;

        // Common random numbers across modes: identical sequence stream, so the
        // BETWEEN-mode differences the W1 LP consumes carry no sampling noise.
        std::mt19937 rng(static_cast<uint32_t>(
            config_.radius_calibration.joint_risk_seed ^ 0x9E3779B9ULL));

        std::vector<double> mu(K), sigma(K);

        for (int s = 0; s < K; ++s) {
            // Propagate mean/covariance along one sampled mode sequence (held mode
            // when there is no chain). predict_before_first_sample=false: e_m means
            // "in mode m NOW", so the first step must be governed by m itself.
            std::vector<std::string> seq;
            if (switching) {
                seq = sample_mode_sequence(e_m, *transition, mode_ids, horizon, rng,
                                           /*predict_before_first_sample=*/false);
            }

            std::vector<Eigen::Vector2d> means; means.reserve(horizon + 1);
            std::vector<Eigen::Matrix2d> covs;  covs.reserve(horizon + 1);
            Eigen::Vector4d x = obs_state.to_array();
            Eigen::Matrix4d cov = Eigen::Matrix4d::Zero();
            means.emplace_back(x.head<2>());
            covs.emplace_back(cov.block<2, 2>(0, 0));

            for (int k = 0; k < horizon; ++k) {
                const std::string& step_mode =
                    switching && k < static_cast<int>(seq.size()) ? seq[k] : mode_id;
                auto it = mode_models.find(step_mode);
                if (it == mode_models.end()) {
                    means.emplace_back(x.head<2>());
                    covs.emplace_back(cov.block<2, 2>(0, 0));
                    continue;
                }
                const ModeModel& m = it->second;
                x = m.A * x + m.b;
                cov = m.A * cov * m.A.transpose() + m.G * m.G.transpose();
                means.emplace_back(x.head<2>());
                covs.emplace_back(cov.block<2, 2>(0, 0));
            }

            auto [mu_s, sigma_s] = surrogate_traj_gaussian(
                means, covs, ego_linearization_traj, horizon, safety_radius,
                num_discs, vehicle_length, z_alpha);
            mu[s] = mu_s;
            sigma[s] = sigma_s;
        }

        risk[mode_id] = want_cvar ? mixture_cvar_clamped(mu, sigma, alpha)
                                  : std::max(0.0, mixture_var(mu, sigma, alpha));
    }

    return risk;
}

std::map<std::string, double> WassersteinDRO::compute_risk_vector_bonferroni(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    const std::vector<EgoState>& ego_linearization_traj,
    int horizon,
    double safety_radius,
    int num_discs,
    double vehicle_length
) {
    std::map<std::string, double> risk;

    const double alpha = config_.radius_calibration.alpha_one_sided;
    const int n_samples = std::max(256, config_.radius_calibration.joint_risk_samples);

    // Bonferroni union correction over the N_s * D per-step, per-disc events, so the
    // max over (k,d) still controls the JOINT violation at level alpha. The union
    // term count must match the (k,d) loops below exactly (k = 1..horizon, d over discs).
    const double n_events = std::max(1.0, static_cast<double>(horizon) *
                                          static_cast<double>(std::max(1, num_discs)));
    const double alpha_eff = 1.0 - (1.0 - alpha) / n_events;
    // VaR_{alpha'}([R - dist]_+) = [R - Q_{1-alpha'}(dist)]_+: the alpha'-quantile of
    // the (decreasing) violation is the (1-alpha')-quantile of the distance.
    const double tail = std::clamp(1.0 - alpha_eff, 0.0, 1.0);

    // Deterministic ego disc centres per step (shared across modes and samples).
    std::vector<std::vector<Eigen::Vector2d>> disc_centres(horizon + 1);
    for (int k = 1; k <= horizon; ++k) {
        const EgoState& e = (k < static_cast<int>(ego_linearization_traj.size()))
                                ? ego_linearization_traj[k]
                                : ego_linearization_traj.back();
        disc_centres[k] = (num_discs > 1)
                              ? compute_ego_disc_positions(e, num_discs, vehicle_length)
                              : std::vector<Eigen::Vector2d>{ e.position() };
    }

    for (const auto& mode_id : mode_ids) {
        auto it = mode_models.find(mode_id);
        if (it == mode_models.end()) { risk[mode_id] = 0.0; continue; }

        // Per-step MARGINAL mean/covariance (Bonferroni uses per-step marginals, not
        // the correlated joint rollout).
        auto means = propagate_mode_mean(obs_state, it->second, horizon);
        auto covs  = propagate_mode_covariance(it->second, horizon);
        const int n_steps = std::min({static_cast<int>(means.size()),
                                      static_cast<int>(covs.size()), horizon + 1});

        // Common random numbers across modes (deterministic reweighting, no jitter).
        std::mt19937_64 rng(config_.radius_calibration.joint_risk_seed);
        std::normal_distribution<double> gauss(0.0, 1.0);

        double max_var = 0.0;
        for (int k = 1; k < n_steps; ++k) {
            // 2x2 Cholesky of Sigma_k (robust).
            const Eigen::Matrix2d S = covs[k];
            const double l11 = std::sqrt(std::max(S(0, 0), 1e-18));
            const double l21 = S(1, 0) / l11;
            const double l22 = std::sqrt(std::max(S(1, 1) - l21 * l21, 0.0));

            const size_t nd = disc_centres[k].size();
            std::vector<std::vector<double>> dist(nd);
            for (auto& v : dist) v.reserve(n_samples);

            // Sample the step-k marginal position once, score every disc from it.
            for (int s = 0; s < n_samples; ++s) {
                const double z0 = gauss(rng), z1 = gauss(rng);
                const Eigen::Vector2d x =
                    means[k] + Eigen::Vector2d(l11 * z0, l21 * z0 + l22 * z1);
                for (size_t d = 0; d < nd; ++d)
                    dist[d].push_back((x - disc_centres[k][d]).norm());
            }
            for (size_t d = 0; d < nd; ++d) {
                std::sort(dist[d].begin(), dist[d].end());
                const int idx = std::clamp(
                    static_cast<int>(std::floor(tail * (dist[d].size() - 1))),
                    0, static_cast<int>(dist[d].size()) - 1);
                const double var_kd = std::max(0.0, safety_radius - dist[d][idx]);
                max_var = std::max(max_var, var_kd);
            }
        }
        risk[mode_id] = max_var;
    }
    return risk;
}

std::map<std::string, double> WassersteinDRO::compute_risk_vector_joint(
    const ObstacleState& obs_state,
    const std::map<std::string, ModeModel>& mode_models,
    const std::vector<std::string>& mode_ids,
    const std::vector<EgoState>& ego_linearization_traj,
    int horizon,
    double safety_radius,
    int num_discs,
    double vehicle_length,
    const Eigen::MatrixXd* transition
) {
    std::map<std::string, double> risk;

    const double alpha = config_.radius_calibration.alpha_one_sided;
    const int n_samples = std::max(1, config_.radius_calibration.joint_risk_samples);
    const bool want_cvar = (config_.radius_calibration.risk_measure == DRORiskMeasure::JOINT_CVAR);

    // Markov-jump: each sample draws a mode SEQUENCE together with the noise path, so
    // the estimator is a risk measure of the JOINT (sequence, noise) law rather than a
    // mean over sequences. transition = I reproduces the held-mode estimator exactly.
    const bool switching =
        (transition != nullptr &&
         transition->rows() == static_cast<Eigen::Index>(mode_ids.size()) &&
         transition->cols() == static_cast<Eigen::Index>(mode_ids.size()));

    // Precompute ego disc centres per step once -- they are deterministic (the ego
    // linearization trajectory), so they are shared across modes and samples.
    std::vector<std::vector<Eigen::Vector2d>> disc_centres(horizon + 1);
    for (int k = 1; k <= horizon; ++k) {
        const EgoState& ego_state =
            (k < static_cast<int>(ego_linearization_traj.size()))
                ? ego_linearization_traj[k]
                : ego_linearization_traj.back();
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

        // Common random numbers: reseed per mode from the SAME fixed seed, so every
        // mode sees an identical noise stream. This makes r[m] deterministic across
        // calls (no jitter in the reweighting) and removes MC noise from BETWEEN-mode
        // comparisons, which is all the W1 LP actually consumes.
        std::mt19937_64 rng(config_.radius_calibration.joint_risk_seed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        // Separate stream for the chain so the noise stream (and therefore every
        // published held-mode JOINT_* number) is untouched when switching is off.
        std::mt19937 seq_rng(static_cast<uint32_t>(
            config_.radius_calibration.joint_risk_seed ^ 0x9E3779B9ULL));

        ModeDistribution e_m;
        if (switching) {
            for (const auto& mid : mode_ids) e_m[mid] = 0.0;
            e_m[mode_id] = 1.0;
        }

        std::vector<double> samples;
        samples.reserve(n_samples);

        for (int s = 0; s < n_samples; ++s) {
            // Roll out the mode's own dynamics: x_{k+1} = A x_k + b + G w_k.
            // Sampling the ROLLOUT (rather than each step's marginal independently)
            // is what makes this JOINT: the correlation A induces across steps is
            // carried exactly, which is precisely what max_k VaR(V_k) throws away.
            // Under `switching` the mode sequence is drawn per sample as well, so the
            // same statement holds for the chain: no averaging over sequences.
            std::vector<std::string> seq;
            if (switching) {
                seq = sample_mode_sequence(e_m, *transition, mode_ids, horizon, seq_rng,
                                           /*predict_before_first_sample=*/false);
            }

            Eigen::Vector4d x = x0;
            double worst = 0.0;

            for (int k = 1; k <= horizon; ++k) {
                const ModeModel* mk = &mode;
                if (switching && k - 1 < static_cast<int>(seq.size())) {
                    auto mit = mode_models.find(seq[k - 1]);
                    if (mit != mode_models.end()) mk = &mit->second;
                }
                const int nk = static_cast<int>(mk->G.cols());
                Eigen::VectorXd w(nk);
                for (int i = 0; i < nk; ++i) w(i) = gauss(rng);
                x = mk->A * x + mk->b + mk->G * w;

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

namespace {

/**
 * @brief Exact W1 between two 1D Gaussians.
 *
 * In 1D the optimal transport map is the monotone quantile map
 * T(x) = mu2 + (s2/s1)(x - mu1), so the displacement is
 *   X - T(X) = (s1 - s2) Z + (mu1 - mu2),  Z ~ N(0,1),
 * i.e. itself Gaussian with mean d = mu1 - mu2 and std |s| = |s1 - s2|. Hence W1
 * is the mean of a FOLDED normal:
 *   W1 = |s| sqrt(2/pi) exp(-d^2 / (2 s^2)) + d erf( d / (|s| sqrt(2)) ).
 * Equal stds collapse this to |d|; equal means collapse it to |s| sqrt(2/pi).
 */
double gaussian_w1_1d(double mu1, double sd1, double mu2, double sd2) {
    const double d = mu1 - mu2;
    const double s = std::abs(sd1 - sd2);

    // s -> 0 sends the exponential to 0 and the erf to sign(d); taking the limit
    // explicitly also avoids the 0/0 when the means coincide as well. The same
    // limit is reached once the mean gap dominates the std gap, which is the
    // common case -- at |d/s| > 8 the exponential is ~1e-14 and the erf is 1 to
    // within 1e-15, so skipping both transcendentals costs nothing measurable.
    if (s < 1e-12 || std::abs(d) > 8.0 * s) return std::abs(d);

    static const double kSqrt2OverPi = std::sqrt(2.0 / M_PI);
    return s * kSqrt2OverPi * std::exp(-(d * d) / (2.0 * s * s))
         + d * std::erf(d / (s * M_SQRT2));
}

/**
 * @brief Number of projection directions in the sliced-W1 quadrature.
 *
 * Two convergence regimes, both measured in tests/test_ground_cost.cpp:
 *
 *  - Distinct covariances (the case the metric exists to resolve): the slice
 *    integrand is smooth and pi-periodic, so the midpoint rule converges
 *    SPECTRALLY -- ~1e-8 by n=128, machine precision by n=512.
 *  - Equal covariances: the integrand degenerates to |<theta, dmu>|, whose kink
 *    caps convergence at O(n^-2). No finite positive-weight rule fixes this: a
 *    finite sum of |<theta_l, v>| is the gauge of a zonotope (a polytope), which
 *    can never equal the Euclidean norm exactly.
 *
 * n=128 puts the degenerate-case error at ~3e-5 absolute, i.e. the reduction to
 * EUCLIDEAN_MEAN holds up to a direction-dependent factor of 1 +/- 3e-5. That is
 * far below anything rho is sensitive to, and it is the right trade: splitting
 * the mean term off analytically WOULD make that reduction exact, but the
 * leftover covariance excess is not subadditive, so D would lose its guaranteed
 * triangle inequality -- and metricity is what makes rho a radius at all.
 *
 * A quadrature detail rather than a modelling knob, so not exposed in DROConfig.
 */
constexpr int kSlicedW1Directions = 128;

/// Midpoint-rule directions on [0, pi), built once. theta and -theta give the same
/// 1D W1 because negation is an isometry of R, so half the circle is the whole
/// integral. Fixed across all pairs, which is what keeps the quadrature a positive
/// combination of pullback metrics and therefore itself a metric.
const std::vector<Eigen::Vector2d>& slice_directions() {
    static const std::vector<Eigen::Vector2d> table = [] {
        std::vector<Eigen::Vector2d> t;
        t.reserve(kSlicedW1Directions);
        for (int l = 0; l < kSlicedW1Directions; ++l) {
            const double a = M_PI * (l + 0.5) / kSlicedW1Directions;
            t.emplace_back(std::cos(a), std::sin(a));
        }
        return t;
    }();
    return table;
}

}  // namespace

double WassersteinDRO::sliced_w1_gaussian_2d(
    const Eigen::Vector2d& mu1, const Eigen::Matrix2d& cov1,
    const Eigen::Vector2d& mu2, const Eigen::Matrix2d& cov2
) {
    double acc = 0.0;
    for (const Eigen::Vector2d& theta : slice_directions()) {
        const double v1 = theta.dot(cov1 * theta);
        const double v2 = theta.dot(cov2 * theta);
        acc += gaussian_w1_1d(theta.dot(mu1), std::sqrt(std::max(0.0, v1)),
                              theta.dot(mu2), std::sqrt(std::max(0.0, v2)));
    }

    // Mean over directions, then the pi/2 normalisation that sends equal
    // covariances to ||mu1 - mu2||:
    //   (1/pi) \int_0^pi |<theta, dmu>| d theta = (2/pi) ||dmu||.
    return (M_PI / 2.0) * acc / kSlicedW1Directions;
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

}  // namespace dro_mpc
