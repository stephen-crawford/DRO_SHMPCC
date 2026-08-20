/**
 * @file mpc_controller.cpp
 * @brief Implementation of Adaptive Scenario-Based MPC Controller.
 */

#include "mpc_controller.hpp"
#include "reference_path.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace dro_mpc {

AdaptiveScenarioMPC::AdaptiveScenarioMPC(const RuntimeConfig& config)
    : config_(config), ego_dynamics_(config.mpc.ego.dynamics, config.mpc.dt) {
    config_.validate();
    default_modes_ = create_obstacle_mode_models(config_.mpc.dt);

    // Initialize DRO module from nested DROConfig.
    if (config_.dro.enabled) {
        dro_ = WassersteinDRO(config_.dro.solver);
    }

    // Initialize random number generator
    std::random_device rd;
    rng_ = std::mt19937(rd());
}

void AdaptiveScenarioMPC::initialize_obstacle(
    int obstacle_id,
    int obstacle_class,
    const std::map<std::string, ModeModel>& available_modes
) {
    const auto& modes = available_modes.empty() ? default_modes_ : available_modes;

    ModeHistory history(obstacle_id, modes, obstacle_class);
    history.max_history = (config_.mpc.sampling.max_history_length > 0) ? config_.mpc.sampling.max_history_length : config_.mpc.horizon * 10;

    // Copy observations from any existing sibling of the same class
    for (const auto& [other_id, other_cls] : obstacle_classes_) {
        if (other_cls == obstacle_class && other_id != obstacle_id) {
            auto it = mode_histories_.find(other_id);
            if (it != mode_histories_.end()) {
                history.observed_modes = it->second.observed_modes;
                break;
            }
        }
    }

    obstacle_classes_[obstacle_id] = obstacle_class;
    mode_histories_[obstacle_id] = history;
}

void AdaptiveScenarioMPC::update_mode_observation(
    int obstacle_id,
    int obstacle_class,
    const std::string& observed_mode,
    int timestep
) {
    if (mode_histories_.find(obstacle_id) == mode_histories_.end()) {
        initialize_obstacle(obstacle_id, obstacle_class);
    }

    if (timestep < 0) {
        timestep = iteration_count_;
    }

    // Record observation for all obstacles sharing this class
    int obs_class = obstacle_classes_.count(obstacle_id)
        ? obstacle_classes_[obstacle_id] : obstacle_class;
    for (auto& [other_id, hist] : mode_histories_) {
        if (hist.obstacle_class == obs_class) {
            hist.record_observation(timestep, observed_mode);
        }
    }
}

MPCResult AdaptiveScenarioMPC::solve(
    const EgoState& ego_state,
    const std::map<int, ObstacleState>& obstacles,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    double path_progress,
    double path_length
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    iteration_count_++;

    // Auto-create straight-line reference path if none set (Paper Eq. 5-6: MPCC)
    if (!reference_path_.has_value()) {
        reference_path_ = ReferencePath::create_straight(ego_state.position(), goal);
    }

    // Initialize spline parameter on ego state if not yet tracking
    // This projects the current position onto the reference path to get initial s
    EgoState ego_with_spline = ego_state;
    if (!ego_with_spline.has_spline() && reference_path_.has_value()) {
        ego_with_spline.s = reference_path_->find_closest_point(ego_with_spline.position());
    }

    // Ensure all obstacles have mode histories
    for (const auto& [obs_id, _] : obstacles) {
        if (mode_histories_.find(obs_id) == mode_histories_.end()) {
            initialize_obstacle(obs_id, 0);
        }
    }

    // Step 1: Initialize reference trajectory (warmstart from previous)
    initialize_reference_trajectory(ego_with_spline, goal, reference_velocity);

    // Step 2: Sample scenarios (DRO reshaped weights q* when DRO on, or nominal otherwise)
    int dro_injected = 0;

    // When DRO is enabled: compute worst-case distribution q* and either
    //   (a) QSTAR_SAMPLE: resample ALL S scenarios from q*, or
    //   (b) DRO/ADVERSARIAL: sample nominally, then inject worst-case scenario(s).
    if (config_.dro.enabled && !reference_trajectory_.empty()) {
        const int S = config_.mpc.sampling.num_scenarios;
        int pre_dro_safe_horizon = config_.mpc.safe_horizon_enabled
            ? config_.compute_safe_horizon(S)
            : config_.mpc.horizon;

        // Compute DRO q* and nominal weights for each obstacle.
        // per_obs_nominal: the single P_hat from belief/weights (or custom OT).
        std::map<int, DROResult> dro_results;
        std::map<int, std::map<std::string, double>> per_obs_nominal;
        for (const auto& [obs_id, obs_state] : obstacles) {
            auto hist_it = mode_histories_.find(obs_id);
            if (hist_it == mode_histories_.end()) continue;

            // Frequency / belief-based weights (the single nominal P_hat).
            auto freq_weights = compute_mode_weights(
                hist_it->second, config_.mpc.sampling.mode_belief, iteration_count_
            );

            // Optional custom weights override both sampling and DRO center.
            auto ot_it = custom_per_obstacle_weights_.find(obs_id);
            bool have_custom = ot_it != custom_per_obstacle_weights_.end()
                               && !ot_it->second.empty();
            std::map<std::string, double> nominal =
                have_custom ? ot_it->second : freq_weights;

            if (nominal.empty()) continue;
            per_obs_nominal[obs_id] = nominal;
            dro_.set_observation_count(
                static_cast<int>(hist_it->second.observed_modes.size()));

            // Markov-jump obstacle: hand the DRO risk model the transition chain so
            // per-mode risk is computed over within-horizon switching rather than a
            // held mode. Ordering matches `nominal` (== the solver's mode_ids order).
            Eigen::MatrixXd obs_transition;
            const Eigen::MatrixXd* transition_ptr = nullptr;
            if (config_.mpc.sampling.use_markov_mode_sampling) {
                std::vector<std::string> modes_order;
                modes_order.reserve(nominal.size());
                for (const auto& [mid, _] : nominal) modes_order.push_back(mid);
                const int M = static_cast<int>(modes_order.size());
                obs_transition = compute_mode_transition_matrix(
                    hist_it->second, modes_order,
                    config_.mpc.sampling.mode_belief.alpha(M),
                    config_.mpc.sampling.mode_belief.kappa(M));
                transition_ptr = &obs_transition;
            }

            dro_results[obs_id] = dro_.compute_worst_case_weights(
                nominal, obs_state, hist_it->second.available_modes,
                reference_trajectory_, config_.mpc.horizon,
                config_.mpc.ego.radius, config_.obstacle_radius,
                config_.mpc.constraints.safety_margin,
                pre_dro_safe_horizon,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length,
                transition_ptr
            );
        }

        if (config_.dro.resolved_injection_mode() == InjectionMode::UNIFORM_COVERAGE) {
            // BASELINE B1: Force each observed mode to appear at least once.
            // Use nominal weights but with ensure_mode_coverage=true.
            if (!per_obs_nominal.empty()) {
                scenarios_ = sample_scenarios_with_weights(
                    obstacles, mode_histories_, per_obs_nominal,
                    config_.mpc.horizon, S, true /*ensure_mode_coverage*/, &rng_
                );
            }
        } else if (config_.dro.resolved_injection_mode() == InjectionMode::SOFTMAX_RISK) {
            // BASELINE B2: p(m) ∝ exp(tau * r_m), no Wasserstein geometry.
            std::map<int, std::map<std::string, double>> softmax_weights;
            for (auto& [obs_id, dro_result] : dro_results) {
                std::map<std::string, double> w;
                double max_r = 0.0;
                for (const auto& [m, r] : dro_result.risk_per_mode)
                    max_r = std::max(max_r, r);
                double sum = 0.0;
                for (const auto& [m, r] : dro_result.risk_per_mode) {
                    w[m] = std::exp(config_.dro.softmax_tau * (r - max_r));  // subtract max for stability
                    sum += w[m];
                }
                if (sum > 0.0) for (auto& [_, v] : w) v /= sum;
                softmax_weights[obs_id] = w;
            }
            if (!softmax_weights.empty()) {
                scenarios_ = sample_scenarios_with_weights(
                    obstacles, mode_histories_, softmax_weights,
                    config_.mpc.horizon, S, config_.mpc.sampling.ensure_mode_coverage, &rng_
                );
            }
        } else if (config_.dro.resolved_injection_mode() == InjectionMode::EPSILON_GREEDY_INJ) {
            // BASELINE B3: (1-eps)*nominal + eps*uniform over modes.
            std::map<int, std::map<std::string, double>> eg_weights;
            double eps = config_.dro.eps_greedy_epsilon;
            for (auto& [obs_id, nom_w] : per_obs_nominal) {
                std::map<std::string, double> w;
                int M_modes = static_cast<int>(nom_w.size());
                double unif = (M_modes > 0) ? 1.0 / M_modes : 0.0;
                for (const auto& [m, p] : nom_w) {
                    w[m] = (1.0 - eps) * p + eps * unif;
                }
                eg_weights[obs_id] = w;
            }
            if (!eg_weights.empty()) {
                scenarios_ = sample_scenarios_with_weights(
                    obstacles, mode_histories_, eg_weights,
                    config_.mpc.horizon, S, config_.mpc.sampling.ensure_mode_coverage, &rng_
                );
            }
        } else if (config_.dro.resolved_injection_mode() == InjectionMode::TOP_RISK_INJECT) {
            // BASELINE B4: Sample nominally, then inject top-K modes by r_m (no WDRO).
            if (!per_obs_nominal.empty()) {
                scenarios_ = sample_scenarios_with_weights(
                    obstacles, mode_histories_, per_obs_nominal,
                    config_.mpc.horizon, S, config_.mpc.sampling.ensure_mode_coverage, &rng_
                );
            }
            int next_id = S;
            int K = config_.dro.injection_count;
            for (const auto& [obs_id, obs_state] : obstacles) {
                auto dr_it = dro_results.find(obs_id);
                if (dr_it == dro_results.end()) continue;
                auto hist_it = mode_histories_.find(obs_id);
                if (hist_it == mode_histories_.end()) continue;

                // Sort modes by risk r_m descending
                std::vector<std::pair<double, std::string>> risk_sorted;
                for (const auto& [m, r] : dr_it->second.risk_per_mode) {
                    if (r > 1e-12) risk_sorted.push_back({r, m});
                }
                std::sort(risk_sorted.begin(), risk_sorted.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

                int n_inject = (K < 0) ? static_cast<int>(risk_sorted.size())
                                       : std::min(K, static_cast<int>(risk_sorted.size()));
                for (int ki = 0; ki < n_inject; ++ki) {
                    const std::string& mode = risk_sorted[ki].second;
                    DROResult fake_result = dr_it->second;
                    for (auto& [m, w] : fake_result.worst_case_weights) w = 0.0;
                    fake_result.worst_case_weights[mode] = 1.0;

                    auto s = dro_.generate_worst_case_scenario(
                        fake_result, obs_id, obs_state,
                        hist_it->second.available_modes,
                        config_.mpc.horizon, next_id);
                    if (!s.trajectories.empty()) {
                        scenarios_.push_back(std::move(s));
                        dro_injected++;
                        next_id++;
                    }
                }
            }
        } else if (config_.dro.resolved_injection_mode() == InjectionMode::DIVERSE_RISK_INJECT) {
            // BASELINE B5: Greedy diverse-risk selection (facility-location style).
            // Select K modes maximizing risk * min-distance-to-already-selected.
            if (!per_obs_nominal.empty()) {
                scenarios_ = sample_scenarios_with_weights(
                    obstacles, mode_histories_, per_obs_nominal,
                    config_.mpc.horizon, S, config_.mpc.sampling.ensure_mode_coverage, &rng_
                );
            }
            int next_id = S;
            int K = config_.dro.injection_count;
            for (const auto& [obs_id, obs_state] : obstacles) {
                auto dr_it = dro_results.find(obs_id);
                if (dr_it == dro_results.end()) continue;
                auto hist_it = mode_histories_.find(obs_id);
                if (hist_it == mode_histories_.end()) continue;

                const auto& rpm = dr_it->second.risk_per_mode;
                const auto& D = dr_it->second.transport_cost_matrix;

                // Build mode_id -> index mapping (matches DRO's mode_ids order)
                std::vector<std::string> mode_ids;
                for (const auto& [m, _] : rpm) mode_ids.push_back(m);
                int M = static_cast<int>(mode_ids.size());
                if (M == 0) continue;

                // Greedy diverse selection
                std::vector<int> selected;
                std::vector<bool> used(M, false);

                // First: argmax r_m
                int best = 0;
                for (int i = 1; i < M; ++i) {
                    if (rpm.at(mode_ids[i]) > rpm.at(mode_ids[best])) best = i;
                }
                selected.push_back(best);
                used[best] = true;

                int n_inject = (K < 0) ? M : std::min(K, M);
                // Subsequent: argmax r_m * min_dist(m, selected)
                for (int ki = 1; ki < n_inject; ++ki) {
                    int best_next = -1;
                    double best_score = -1.0;
                    for (int i = 0; i < M; ++i) {
                        if (used[i]) continue;
                        double r_i = rpm.at(mode_ids[i]);
                        if (r_i < 1e-12) continue;
                        // min distance to any already-selected mode
                        double min_d = 1e9;
                        for (int si : selected) {
                            if (i < static_cast<int>(D.size()) && si < static_cast<int>(D[i].size()))
                                min_d = std::min(min_d, D[i][si]);
                        }
                        double score = r_i * min_d;
                        if (score > best_score) { best_score = score; best_next = i; }
                    }
                    if (best_next < 0) break;
                    selected.push_back(best_next);
                    used[best_next] = true;
                }

                // Inject selected modes
                for (int si : selected) {
                    const std::string& mode = mode_ids[si];
                    DROResult fake_result = dr_it->second;
                    for (auto& [m, w] : fake_result.worst_case_weights) w = 0.0;
                    fake_result.worst_case_weights[mode] = 1.0;

                    auto s = dro_.generate_worst_case_scenario(
                        fake_result, obs_id, obs_state,
                        hist_it->second.available_modes,
                        config_.mpc.horizon, next_id);
                    if (!s.trajectories.empty()) {
                        scenarios_.push_back(std::move(s));
                        dro_injected++;
                        next_id++;
                    }
                }
            }
        } else {
            // Q* SAMPLING PATH (QSTAR_SAMPLE or default):
            // Resample ALL S scenarios from q* distribution (Eq. 3-4).
            std::map<int, std::map<std::string, double>> per_obs_weights_dro;
            for (auto& [obs_id, dro_result] : dro_results) {
                per_obs_weights_dro[obs_id] = dro_result.worst_case_weights;
            }

            if (!per_obs_weights_dro.empty()) {
                if (config_.mpc.sampling.use_markov_mode_sampling) {
                    // Seed the Markov chain from Q*: the reweighted belief sets the
                    // initial mode distribution, then the estimated transition
                    // matrix propagates it over the horizon.
                    scenarios_ = sample_scenarios_markov(
                        obstacles, mode_histories_, &per_obs_weights_dro,
                        config_.mpc.horizon, S, config_.mpc.sampling.mode_belief, &rng_
                    );
                } else {
                    scenarios_ = sample_scenarios_with_weights(
                        obstacles, mode_histories_, per_obs_weights_dro,
                        config_.mpc.horizon, S,
                        config_.mpc.sampling.ensure_mode_coverage, &rng_
                    );
                }
            }
        }
    }

    // When custom per-obstacle weights are set (e.g. from OT predictor),
    // use them for scenario sampling (only fires when DRO is off).
    if (scenarios_.empty() && !custom_per_obstacle_weights_.empty()) {
        scenarios_ = sample_scenarios_with_weights(
            obstacles, mode_histories_, custom_per_obstacle_weights_,
            config_.mpc.horizon, config_.mpc.sampling.num_scenarios,
            config_.mpc.sampling.ensure_mode_coverage, &rng_
        );
    }

    if (scenarios_.empty()) {
        if (config_.mpc.sampling.use_markov_mode_sampling) {
            // Base (non-DRO) Markov path: belief derived from weight_type +
            // the Dirichlet prior, no Q* override.
            scenarios_ = sample_scenarios_markov(
                obstacles, mode_histories_, nullptr,
                config_.mpc.horizon, config_.mpc.sampling.num_scenarios, config_.mpc.sampling.mode_belief, &rng_
            );
        } else if (config_.mpc.sampling.ensure_mode_coverage) {
            // Coverage-forcing baseline (deliberately breaks i.i.d.): route through the
            // SINGLE i.i.d. sampler with computed nominal weights + ensure_mode_coverage,
            // instead of a separate coverage function.
            std::map<int, std::map<std::string, double>> nominal_w;
            for (const auto& [obs_id, hist] : mode_histories_) {
                auto w = compute_mode_weights(hist, config_.mpc.sampling.mode_belief, iteration_count_);
                if (!w.empty()) nominal_w[obs_id] = w;
            }
            scenarios_ = sample_scenarios_with_weights(
                obstacles, mode_histories_, nominal_w, config_.mpc.horizon,
                config_.mpc.sampling.num_scenarios, /*ensure_mode_coverage=*/true, &rng_
            );
        } else {
            scenarios_ = sample_scenarios(
                obstacles, mode_histories_, config_.mpc.horizon,
                config_.mpc.sampling.num_scenarios, config_.mpc.sampling.mode_belief, iteration_count_, &rng_
            );
        }
    }

    // Clear custom weights after use (they're set per-solve by external code)
    custom_per_obstacle_weights_.clear();

    // Add any pre-injected scenarios (e.g. DRO worst-case)
    for (auto& sc : pre_injected_scenarios_) {
        sc.is_injected = true;
        scenarios_.push_back(sc);
    }
    pre_injected_scenarios_.clear();

    // Verify scenario sufficiency for epsilon guarantee (Part 4).
    // de Groot sizes S from the SUPPORT LIMIT n̄ (horizon-independent) via the exact
    // NSO bound (Eq. 8), not from the decision-variable dimension.
    {
        int S_actual = static_cast<int>(scenarios_.size());
        int S_required = config_.compute_required_scenarios(
            config_.mpc.constraints.support_cap_nbar);
        (void)config_.compute_effective_epsilon(
            S_actual, config_.mpc.constraints.support_cap_nbar);

        if (S_actual < S_required && config_.mpc.sampling.enforce_scenario_count) {
            // Auto-increase: sample additional scenarios
            int additional_count = S_required - S_actual;
            auto additional = sample_scenarios(
                obstacles,
                mode_histories_,
                config_.mpc.horizon,
                additional_count,
                config_.mpc.sampling.mode_belief,
                iteration_count_,
                &rng_
            );
            scenarios_.insert(scenarios_.end(), additional.begin(), additional.end());
        } else if (S_actual < 3) {
            // Ensure minimum scenario count even without enforcement
            int additional_count = std::max(
                5, config_.mpc.sampling.num_scenarios - S_actual
            );
            auto additional = sample_scenarios(
                obstacles,
                mode_histories_,
                config_.mpc.horizon,
                additional_count,
                config_.mpc.sampling.mode_belief,
                iteration_count_,
                &rng_
            );
            scenarios_.insert(scenarios_.end(), additional.begin(), additional.end());
        }
    }

    // Step 3b: Verify scenario sufficiency for epsilon guarantee (Part 4)
    if (config_.mpc.sampling.enforce_scenario_count) {
        int S_required = config_.compute_required_scenarios(
            config_.mpc.constraints.support_cap_nbar);
        int S_actual = static_cast<int>(scenarios_.size());
        if (S_actual < S_required) {
            // Auto-increase: sample additional scenarios
            int additional_count = S_required - S_actual;
            auto additional = sample_scenarios(
                obstacles, mode_histories_, config_.mpc.horizon,
                additional_count, config_.mpc.sampling.mode_belief, iteration_count_, &rng_
            );
            scenarios_.insert(scenarios_.end(), additional.begin(), additional.end());
        }
    }

    // Step 4: Fixed collision normals from the numerical linearization trajectory.
    // Normals are held constant for the subsequent QP (Case B also linearizes
    // heading-dependent disc centers about the same numerical reference).
    auto constraint_start = std::chrono::high_resolution_clock::now();

    // de Groot 2023 Definition-2 geometric dominance pruning: drop scenarios whose
    // collision half-spaces are IMPLIED (on the reachable ball) by a more-restrictive
    // scenario's, evaluated at the reference trajectory. The implication is certified
    // exactly on the actual (a, b) per ego disc, so it is sound (never drops an active
    // scenario) rather than an a-priori heuristic. Injected worst-case scenarios are
    // never pruned; enforce_all_scenarios disables pruning entirely to keep the full S
    // constraints (restores the Theorem-1 premise).
    if (!config_.mpc.sampling.enforce_all_scenarios) {
        const double combined_radius = config_.mpc.ego.radius +
                                       config_.obstacle_radius +
                                       config_.mpc.constraints.safety_margin;
        scenarios_ = prune_dominated_scenarios(
            scenarios_, reference_trajectory_, combined_radius,
            config_.mpc.ego.num_discs, config_.mpc.ego.length);
    }

    auto constraints = compute_linearized_constraints(
        reference_trajectory_,
        scenarios_,
        config_.mpc.ego.radius,
        config_.obstacle_radius,
        config_.mpc.constraints.safety_margin,
        config_.mpc.ego.num_discs,
        config_.mpc.ego.length
    );
    // Step 4b: Safe horizon truncation (SH-MPC)
    // Reduce constraint horizon to N_safe based on configured mode:
    // - PRACTICAL:          N_safe = min(N, floor(S / (2*n_u)))
    // - THEORETICAL_SIMPLE: Eq. 23, S >= (2/eps)*(ln(1/beta) + d)
    // - THEORETICAL_TIGHT:  Eq. 25 (very conservative)
    int effective_horizon = config_.mpc.horizon;
    if (config_.mpc.safe_horizon_enabled) {
        // Use only i.i.d. sampled scenario count for safe horizon computation.
        // DRO-injected scenarios are deterministic additional constraints that
        // tighten the feasible set but do NOT satisfy the i.i.d. assumption
        // required by Calafiore-Campi scenario theory (Theorem 1).
        // Counting them would inflate N_safe beyond what the theory certifies.
        int S_for_sh = config_.mpc.sampling.num_scenarios;
        effective_horizon = config_.compute_safe_horizon(S_for_sh);

        if (effective_horizon < config_.mpc.horizon) {
            // Filter out constraints beyond the safe horizon
            constraints.erase(
                std::remove_if(constraints.begin(), constraints.end(),
                    [effective_horizon](const CollisionConstraint& c) {
                        return c.k >= effective_horizon;
                    }),
                constraints.end()
            );
        }
    }

    // Step 4c: Drop far-away (trivially non-binding) constraints. The support
    // reduction itself was already done at the scenario level via dominance
    // pruning above (de Groot Algorithm 3), which is non-distorting.
    constraints = filter_constraints_by_clearance(
        constraints, reference_trajectory_, 20.0);

    auto constraint_end = std::chrono::high_resolution_clock::now();

    // Step 5: Solve optimization problem
    auto qp_start = std::chrono::high_resolution_clock::now();
    MPCResult result = solve_optimization(
        ego_with_spline, goal, reference_velocity, constraints,
        path_progress, path_length, effective_horizon
    );

    auto qp_end = std::chrono::high_resolution_clock::now();

    // Record safe horizon and DRO injection count
    result.safe_horizon = effective_horizon;
    result.num_dro_injected = dro_injected;
    result.constraint_construction_time =
        std::chrono::duration<double>(constraint_end - constraint_start).count();
    result.qp_solve_time =
        std::chrono::duration<double>(qp_end - qp_start).count();

    if (result.success) {
        // Update reference trajectory for next iteration
        reference_trajectory_ = result.ego_trajectory;
    }

    // Record timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    result.solve_time = elapsed.count();
    solve_times_.push_back(result.solve_time);

    return result;
}

void AdaptiveScenarioMPC::initialize_reference_trajectory(
    const EgoState& ego_state,
    const Eigen::Vector2d& goal,
    double reference_velocity
) {
    if (!reference_trajectory_.empty() && reference_trajectory_.size() > 1) {
        // Shift previous trajectory forward
        reference_trajectory_.erase(reference_trajectory_.begin());

        // Extend to full horizon
        while (static_cast<int>(reference_trajectory_.size()) <= config_.mpc.horizon) {
            const EgoState& last = reference_trajectory_.back();
            // Simple constant velocity extension
            EgoState new_state(
                last.x + last.v * std::cos(last.theta) * config_.mpc.dt,
                last.y + last.v * std::sin(last.theta) * config_.mpc.dt,
                last.theta,
                last.v
            );
            // Propagate spline parameter algebraically
            if (last.has_spline() && reference_path_.has_value()) {
                new_state.s = EgoDynamics::compute_spline_update(
                    last, new_state, *reference_path_, config_.mpc.dt);
            }
            reference_trajectory_.push_back(new_state);
        }

        // Update first state to current
        reference_trajectory_[0] = ego_state;
    } else {
        // Initialize with straight-line trajectory to goal
        reference_trajectory_ = generate_straight_line_trajectory(ego_state, goal, reference_velocity);
    }
}

std::vector<EgoState> AdaptiveScenarioMPC::generate_straight_line_trajectory(
    const EgoState& start,
    const Eigen::Vector2d& goal,
    double reference_velocity
) {
    std::vector<EgoState> trajectory;
    trajectory.reserve(config_.mpc.horizon + 1);

    // Initialize spline parameter for starting state if path available
    EgoState start_with_s = start;
    if (!start_with_s.has_spline() && reference_path_.has_value()) {
        start_with_s.s = reference_path_->find_closest_point(start_with_s.position());
    }
    trajectory.push_back(start_with_s);

    EgoState current = start_with_s;

    for (int k = 0; k < config_.mpc.horizon; ++k) {
        // Direction to goal
        Eigen::Vector2d to_goal = goal - current.position();
        double dist = to_goal.norm();

        double desired_theta;
        if (dist > 0.1) {
            Eigen::Vector2d direction = to_goal / dist;
            desired_theta = std::atan2(direction(1), direction(0));
        } else {
            desired_theta = current.theta;
        }

        // Simple propagation - use reference_velocity as cap instead of hardcoded 2.0
        double v = std::min(current.v + 0.5 * config_.mpc.dt, reference_velocity);

        EgoState next_state(
            current.x + v * std::cos(desired_theta) * config_.mpc.dt,
            current.y + v * std::sin(desired_theta) * config_.mpc.dt,
            desired_theta,
            v
        );
        // Propagate spline parameter
        if (current.has_spline() && reference_path_.has_value()) {
            next_state.s = EgoDynamics::compute_spline_update(
                current, next_state, *reference_path_, config_.mpc.dt);
        }
        trajectory.push_back(next_state);
        current = next_state;
    }

    return trajectory;
}

MPCResult AdaptiveScenarioMPC::solve_optimization(
    const EgoState& ego_state,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    const std::vector<CollisionConstraint>& constraints,
    double path_progress,
    double path_length,
    int cost_horizon
) {
    if (config_.solver.use_sqp_solver) {
        return solve_optimization_sqp(
            ego_state, goal, reference_velocity, constraints,
            path_progress, path_length, cost_horizon
        );
    }

    // Heuristic fallback: simple optimization without CasADi
    int N = config_.mpc.horizon;

    auto trajectory = generate_straight_line_trajectory(ego_state, goal, reference_velocity);
    std::vector<EgoInput> inputs;
    inputs.reserve(N);

    for (int k = 0; k < N; ++k) {
        if (k + 1 < static_cast<int>(trajectory.size())) {
            const EgoState& current = trajectory[k];
            const EgoState& next_state = trajectory[k + 1];

            double a = (next_state.v - current.v) / config_.mpc.dt;
            double w = (next_state.theta - current.theta) / config_.mpc.dt;

            a = std::clamp(a, config_.mpc.ego.dynamics.min_acceleration, config_.mpc.ego.dynamics.max_acceleration);
            w = std::clamp(w, -config_.mpc.ego.dynamics.max_steering_rate, config_.mpc.ego.dynamics.max_steering_rate);

            inputs.emplace_back(a, w);
        } else {
            inputs.emplace_back(0, 0);
        }
    }

    trajectory = ego_dynamics_.rollout(ego_state, inputs);

    auto [max_violation, violated] = evaluate_constraint_violation(constraints, trajectory);

    if (max_violation > 0) {
        auto [new_trajectory, new_inputs] = apply_simple_avoidance(
            ego_state, trajectory, inputs, constraints
        );
        trajectory = new_trajectory;
        inputs = new_inputs;
    }

    const double effective_goal_weight = config_.mpc.objective.goal_weight;
    (void)path_progress; (void)path_length;

    double cost = 0.0;
    for (int k = 0; k <= N; ++k) {
        Eigen::Vector2d pos_diff = trajectory[k].position() - goal;
        double weight = effective_goal_weight;
        if (k == N) weight *= 2.0;
        cost += weight * pos_diff.squaredNorm();

        double w_vel = config_.mpc.objective.velocity_weight;
        double v_diff = trajectory[k].v - reference_velocity;
        cost += w_vel * v_diff * v_diff;

    }
    for (int k = 0; k < N; ++k) {
        cost += config_.mpc.objective.acceleration_weight * inputs[k].a * inputs[k].a;
        cost += config_.mpc.objective.steering_weight * inputs[k].delta * inputs[k].delta;
    }

    // MPCC cost terms for heuristic solver (all steps 1..N)
    // Uses integrated spline parameter when available
    if (reference_path_.has_value()) {
        const auto& path = *reference_path_;
        for (int k = 1; k <= N; ++k) {
            double s_k;
            if (trajectory[k].has_spline()) {
                s_k = std::clamp(trajectory[k].s, 0.0, path.total_length());
            } else {
                s_k = path.find_closest_point(trajectory[k].position());
            }
            PathPoint pp = path.get_point_at(s_k);
            double ph = pp.heading;
            Eigen::Vector2d n_ref(-std::sin(ph), std::cos(ph));
            Eigen::Vector2d t_ref(std::cos(ph), std::sin(ph));
            Eigen::Vector2d diff = trajectory[k].position() - pp.position;
            double e_c = n_ref.dot(diff);
            double e_l = -t_ref.dot(diff);
            cost += config_.mpc.objective.contour_weight * e_c * e_c;
            cost += config_.mpc.objective.lag_weight * e_l * e_l;
        }
        double s_term;
        if (trajectory[N].has_spline()) {
            s_term = std::clamp(trajectory[N].s, 0.0, path.total_length());
        } else {
            s_term = path.find_closest_point(trajectory[N].position());
        }
        double desired_heading = path.get_heading_at(s_term);
        double heading_err = trajectory[N].theta - desired_heading;
        while (heading_err > M_PI) heading_err -= 2 * M_PI;
        while (heading_err < -M_PI) heading_err += 2 * M_PI;
        cost += config_.mpc.objective.terminal_heading_weight * heading_err * heading_err;
    }

    MPCResult result;
    result.success = true;
    result.ego_trajectory = trajectory;
    result.control_inputs = inputs;
    result.cost = cost;

    return result;
}

// ============================================================================
// SQP Solver
// ============================================================================

MPCResult AdaptiveScenarioMPC::solve_optimization_sqp(
    const EgoState& ego_state,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    const std::vector<CollisionConstraint>& constraints,
    double path_progress,
    double path_length,
    int cost_horizon
) {
    const int N = config_.mpc.horizon;

    // 1. Build initial reference trajectory and extract inputs
    auto x_ref = generate_straight_line_trajectory(ego_state, goal, reference_velocity);
    std::vector<EgoInput> u_ref;
    u_ref.reserve(N);

    for (int k = 0; k < N; ++k) {
        if (k + 1 < static_cast<int>(x_ref.size())) {
            double a = (x_ref[k + 1].v - x_ref[k].v) / config_.mpc.dt;
            double w = (x_ref[k + 1].theta - x_ref[k].theta) / config_.mpc.dt;
            a = std::clamp(a, config_.mpc.ego.dynamics.min_acceleration, config_.mpc.ego.dynamics.max_acceleration);
            w = std::clamp(w, -config_.mpc.ego.dynamics.max_steering_rate, config_.mpc.ego.dynamics.max_steering_rate);
            u_ref.emplace_back(a, w);
        } else {
            u_ref.emplace_back(0, 0);
        }
    }

    // Re-propagate to get consistent reference (with spline if path available)
    if (reference_path_.has_value()) {
        x_ref = ego_dynamics_.rollout_with_spline(ego_state, u_ref, *reference_path_);
    } else {
        x_ref = ego_dynamics_.rollout(ego_state, u_ref);
    }

    // Use warmstarted reference if available
    if (!reference_trajectory_.empty() &&
        static_cast<int>(reference_trajectory_.size()) > N) {
        // Extract inputs from warmstarted reference
        std::vector<EgoInput> warm_inputs;
        warm_inputs.reserve(N);
        for (int k = 0; k < N; ++k) {
            const auto& curr = reference_trajectory_[k];
            const auto& next = reference_trajectory_[k + 1];
            double a = (next.v - curr.v) / config_.mpc.dt;
            double w = (next.theta - curr.theta) / config_.mpc.dt;
            a = std::clamp(a, config_.mpc.ego.dynamics.min_acceleration, config_.mpc.ego.dynamics.max_acceleration);
            w = std::clamp(w, -config_.mpc.ego.dynamics.max_steering_rate, config_.mpc.ego.dynamics.max_steering_rate);
            warm_inputs.emplace_back(a, w);
        }
        std::vector<EgoState> warm_traj;
        if (reference_path_.has_value()) {
            warm_traj = ego_dynamics_.rollout_with_spline(ego_state, warm_inputs, *reference_path_);
        } else {
            warm_traj = ego_dynamics_.rollout(ego_state, warm_inputs);
        }

        // Use warmstart if it's reasonable (not too far from ego)
        double warm_dist = (warm_traj[1].position() - ego_state.position()).norm();
        if (warm_dist < 5.0) {
            u_ref = warm_inputs;
            x_ref = warm_traj;
        }
    }

    // Douglas-Rachford projection: ensure warmstart satisfies collision constraints
    if (!constraints.empty()) {
        project_warmstart_to_safety(
            x_ref, constraints, /*max_projection_sweeps=*/10, /*tolerance=*/1e-3);
    }

    // 2. SQP loop
    for (int sqp_iter = 0; sqp_iter < config_.solver.sqp_max_iterations; ++sqp_iter) {
        // Build and solve QP subproblem
        QPProblem qp = build_condensed_qp(
            x_ref, u_ref, goal, reference_velocity, constraints,
            path_progress, path_length, cost_horizon
        );

        QPSettings qp_settings;
        qp_settings.max_iterations = config_.solver.qp_max_iterations;
        qp_settings.abs_tol = config_.solver.qp_tolerance;
        qp_settings.adaptive_rho = true;

        QPResult qp_result = qp_solver_.solve(qp, qp_settings);

        Eigen::VectorXd delta_u = qp_result.x;

        // Check SQP convergence
        if (delta_u.norm() < config_.solver.sqp_convergence_tol) {
            break;
        }

        // Line search: try full step, then half, then quarter
        double alpha = 1.0;
        std::vector<EgoInput> best_inputs = u_ref;
        std::vector<EgoState> best_traj = x_ref;
        double best_violation = std::numeric_limits<double>::max();

        for (int ls = 0; ls < 3; ++ls) {
            std::vector<EgoInput> trial_inputs;
            trial_inputs.reserve(N);
            for (int k = 0; k < N; ++k) {
                double a_new = u_ref[k].a + alpha * delta_u(2 * k);
                double w_new = u_ref[k].delta + alpha * delta_u(2 * k + 1);
                a_new = std::clamp(a_new, config_.mpc.ego.dynamics.min_acceleration, config_.mpc.ego.dynamics.max_acceleration);
                w_new = std::clamp(w_new, -config_.mpc.ego.dynamics.max_steering_rate, config_.mpc.ego.dynamics.max_steering_rate);
                trial_inputs.emplace_back(a_new, w_new);
            }

            std::vector<EgoState> trial_traj;
            if (reference_path_.has_value()) {
                trial_traj = ego_dynamics_.rollout_with_spline(ego_state, trial_inputs, *reference_path_);
            } else {
                trial_traj = ego_dynamics_.rollout(ego_state, trial_inputs);
            }
            auto [max_viol, _] = evaluate_constraint_violation(constraints, trial_traj);

            if (max_viol < best_violation || ls == 0) {
                best_violation = max_viol;
                best_inputs = trial_inputs;
                best_traj = trial_traj;
                if (max_viol <= 0) break;  // Feasible — accept
            }
            alpha *= 0.5;
        }

        // Update reference for next SQP iteration
        u_ref = best_inputs;
        x_ref = best_traj;

        // Warm-start QP solver for next iteration
        qp_solver_.warm_start(delta_u * 0.0);  // zero since we re-linearize
    }

    // Hard velocity-bound enforcement on the returned plan.
    // The velocity update is exact (v_{k+1} = v_k + a_k dt), so clamping each
    // acceleration to the interval that keeps v_{k+1} in [min_velocity, max_velocity]
    // (intersected with the accel box) guarantees the bound in every downstream
    // rollout -- including the harness, which re-applies these inputs -- regardless
    // of whether the ADMM QP fully converged the soft velocity rows above. When the
    // reference speed exceeds the cap the per-step QP row is momentarily infeasible
    // (cannot brake far enough in one dt); this saturation is the hard backstop.
    if (config_.mpc.constraints.enable_velocity_bounds) {
        double v_cur = ego_state.v;
        for (int k = 0; k < N; ++k) {
            const double v_lo_a = (config_.mpc.ego.dynamics.min_velocity - v_cur) / config_.mpc.dt;
            const double v_hi_a = (config_.mpc.ego.dynamics.max_velocity - v_cur) / config_.mpc.dt;
            const double lo = std::max(config_.mpc.ego.dynamics.min_acceleration, v_lo_a);
            const double hi = std::min(config_.mpc.ego.dynamics.max_acceleration, v_hi_a);
            double a;
            if (lo <= hi) {
                a = std::clamp(u_ref[k].a, lo, hi);       // velocity-feasible accel window
            } else if (v_hi_a < config_.mpc.ego.dynamics.min_acceleration) {
                a = config_.mpc.ego.dynamics.min_acceleration;             // above v_max: brake as hard as allowed
            } else {
                a = config_.mpc.ego.dynamics.max_acceleration;             // below v_min: accelerate as hard as allowed
            }
            u_ref[k] = EgoInput(a, u_ref[k].delta);
            v_cur += a * config_.mpc.dt;                      // exact velocity propagation
        }
        // Re-roll the trajectory so x_ref matches the clamped inputs.
        x_ref = reference_path_.has_value()
                    ? ego_dynamics_.rollout_with_spline(ego_state, u_ref, *reference_path_)
                    : ego_dynamics_.rollout(ego_state, u_ref);
    }

    const double effective_goal_weight = config_.mpc.objective.goal_weight;
    (void)path_progress; (void)path_length;

    // Compute final cost
    double cost = 0.0;

    for (int k = 0; k <= N; ++k) {
        Eigen::Vector2d pos_diff = x_ref[k].position() - goal;
        double weight = effective_goal_weight;
        if (k == N) weight *= 2.0;
        cost += weight * pos_diff.squaredNorm();

        double w_vel = config_.mpc.objective.velocity_weight;
        double v_diff = x_ref[k].v - reference_velocity;
        cost += w_vel * v_diff * v_diff;

    }
    for (int k = 0; k < N; ++k) {
        cost += config_.mpc.objective.acceleration_weight * u_ref[k].a * u_ref[k].a;
        cost += config_.mpc.objective.steering_weight * u_ref[k].delta * u_ref[k].delta;
    }

    // MPCC cost terms for final cost report (all steps 1..N)
    // Uses integrated spline parameter when available
    if (reference_path_.has_value()) {
        const auto& path = *reference_path_;
        for (int k = 1; k <= N; ++k) {
            double s_k;
            if (x_ref[k].has_spline()) {
                s_k = std::clamp(x_ref[k].s, 0.0, path.total_length());
            } else {
                s_k = path.find_closest_point(x_ref[k].position());
            }
            PathPoint pp = path.get_point_at(s_k);
            double ph = pp.heading;
            Eigen::Vector2d n_ref(-std::sin(ph), std::cos(ph));
            Eigen::Vector2d t_ref(std::cos(ph), std::sin(ph));
            Eigen::Vector2d diff = x_ref[k].position() - pp.position;
            double e_c = n_ref.dot(diff);
            double e_l = -t_ref.dot(diff);
            cost += config_.mpc.objective.contour_weight * e_c * e_c;
            cost += config_.mpc.objective.lag_weight * e_l * e_l;
        }
        double s_term;
        if (x_ref[N].has_spline()) {
            s_term = std::clamp(x_ref[N].s, 0.0, path.total_length());
        } else {
            s_term = path.find_closest_point(x_ref[N].position());
        }
        double desired_heading = path.get_heading_at(s_term);
        double heading_err = x_ref[N].theta - desired_heading;
        while (heading_err > M_PI) heading_err -= 2 * M_PI;
        while (heading_err < -M_PI) heading_err += 2 * M_PI;
        cost += config_.mpc.objective.terminal_heading_weight * heading_err * heading_err;
    }

    // Check feasibility
    auto [final_violation, _] = evaluate_constraint_violation(constraints, x_ref);
    bool feasible = (final_violation <= 0.01);  // small tolerance

    MPCResult result;
    result.success = feasible;
    result.ego_trajectory = x_ref;
    result.control_inputs = u_ref;
    result.cost = cost;

    // If infeasible, try safe fallback
    if (!feasible) {
        auto fallback = generate_safe_fallback(ego_state);
        auto [fb_viol, __] = evaluate_constraint_violation(constraints, fallback.ego_trajectory);
        // Use SQP result if it's better than fallback, even if not perfectly feasible
        if (final_violation < fb_viol || final_violation < 0.1) {
            result.success = true;  // Approximately feasible
        } else {
            return fallback;
        }
    }

    return result;
}

QPProblem AdaptiveScenarioMPC::build_condensed_qp(
    const std::vector<EgoState>& x_ref,
    const std::vector<EgoInput>& u_ref,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    const std::vector<CollisionConstraint>& constraints,
    double path_progress,
    double path_length,
    int cost_horizon
) {
    const int N = config_.mpc.horizon;
    const int n_u = 2;  // [a, w]
    const int n_dec = n_u * N;  // total decision variables (delta_u)

    // Position extraction matrix: E selects [x, y] from [x, y, theta, v]
    Eigen::Matrix<double, 2, 4> E = Eigen::Matrix<double, 2, 4>::Zero();
    E(0, 0) = 1.0;  // x
    E(1, 1) = 1.0;  // y

    // Velocity extraction row: selects v from [x, y, theta, v]
    Eigen::RowVector4d V_row = Eigen::RowVector4d::Zero();
    V_row(3) = 1.0;  // v

    // Step 1: Linearize dynamics at each timestep
    std::vector<Eigen::Matrix4d> A_k(N);
    std::vector<Eigen::Matrix<double, 4, 2>> B_k(N);

    for (int k = 0; k < N; ++k) {
        auto [Ak, Bk] = ego_dynamics_.get_jacobians(
            x_ref[k].to_array(), u_ref[k].to_array()
        );
        A_k[k] = Ak;
        B_k[k] = Bk;
    }

    // Step 2: Build condensed sensitivity matrices M[k][j]
    // delta_x[k+1] = sum_{j=0}^{k} M[k+1][j] * delta_u[j]
    // where M[k][j] = Phi(k, j+1) * B[j], Phi(k,j) = A[k-1]*...*A[j]
    //
    // We store P[k][j] = E * M[k][j] (2x2 position sensitivity)
    // and    Vk[j]    = V_row * M[k][j] (1x2 velocity sensitivity)
    //
    // For efficiency, build incrementally:
    //   M[k+1][j] = A[k] * M[k][j]  for j < k
    //   M[k+1][k] = B[k]

    // M_prev[j] stores M[k][j] for the current k
    // We iterate k from 1 to N, building M[k][j] from M[k-1][j]

    // Position sensitivities P[k][j] for k=1..N, j=0..k-1 (2x2 each)
    // Stored as P_all[k] = 2 x (2*N) matrix, columns 2j..2j+1 = P[k][j]
    std::vector<Eigen::MatrixXd> P_all(N + 1, Eigen::MatrixXd::Zero(2, n_dec));
    // Velocity sensitivities V_all[k][j] for k=1..N (1x2 each)
    std::vector<Eigen::RowVectorXd> V_all(N + 1, Eigen::RowVectorXd::Zero(n_dec));

    // Heading sensitivities THETA_all[k] for MPCC terminal heading cost
    Eigen::RowVector4d THETA_row = Eigen::RowVector4d::Zero();
    THETA_row(2) = 1.0;  // selects theta from [x, y, theta, v]
    std::vector<Eigen::RowVectorXd> THETA_all(N + 1, Eigen::RowVectorXd::Zero(n_dec));

    // M_current[j] = M[k][j] (4x2 matrices), we only need the current k's
    std::vector<Eigen::Matrix<double, 4, 2>> M_current(N, Eigen::Matrix<double, 4, 2>::Zero());

    for (int k = 1; k <= N; ++k) {
        // M[k][j] = A[k-1] * M[k-1][j] for j < k-1
        // M[k][k-1] = B[k-1]
        std::vector<Eigen::Matrix<double, 4, 2>> M_new(N, Eigen::Matrix<double, 4, 2>::Zero());

        for (int j = 0; j < k - 1; ++j) {
            M_new[j] = A_k[k - 1] * M_current[j];
        }
        M_new[k - 1] = B_k[k - 1];

        // Extract position, velocity, and heading sensitivities
        for (int j = 0; j < k; ++j) {
            Eigen::Matrix<double, 2, 2> Pkj = E * M_new[j];
            P_all[k].block<2, 2>(0, 2 * j) = Pkj;

            Eigen::RowVector2d Vkj = V_row * M_new[j];
            V_all[k].segment<2>(2 * j) = Vkj;

            Eigen::RowVector2d THkj = THETA_row * M_new[j];
            THETA_all[k].segment<2>(2 * j) = THkj;
        }

        M_current = M_new;
    }

    // Step 3: Build Hessian H (n_dec x n_dec)
    // H_ij = sum_k w_goal * P[k,i]^T * P[k,j]
    //       + sum_k w_vel * V[k,i]^T * V[k,j]
    //       + diag(w_accel, w_steer, w_accel, w_steer, ...)

    const double effective_goal_weight = config_.mpc.objective.goal_weight;
    (void)path_progress; (void)path_length;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_dec, n_dec);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n_dec);

    for (int k = 1; k <= N; ++k) {
        double w_goal = effective_goal_weight;
        if (k == N) w_goal *= 2.0;  // Terminal cost boost

        // Goal tracking: w_goal * P[k]^T * P[k]
        H += w_goal * P_all[k].transpose() * P_all[k];

        // Velocity tracking: w_vel * V[k]^T * V[k]
        double w_vel = config_.mpc.objective.velocity_weight;
        H += w_vel * V_all[k].transpose() * V_all[k];
    }

    // Control effort: diagonal terms
    for (int k = 0; k < N; ++k) {
        H(2 * k, 2 * k) += config_.mpc.objective.acceleration_weight;
        H(2 * k + 1, 2 * k + 1) += config_.mpc.objective.steering_weight;
    }

    // Regularize for positive definiteness
    H.diagonal().array() += 1e-6;

    // MPCC cost terms (Paper Eq. 6): contouring + lag + terminal heading
    // Applied to all steps 1..N. Safe horizon only truncates collision
    // constraints (already filtered upstream in solve()), not objectives.
    //
    // Uses integrated spline parameter s from EgoState when available,
    // falling back to find_closest_point() projection otherwise.
    // Integrated s provides smoother, more consistent path tracking
    // since it evolves algebraically rather than jumping between
    // closest-point projections.
    if (reference_path_.has_value()) {
        const double w_c = config_.mpc.objective.contour_weight;
        const double w_l = config_.mpc.objective.lag_weight;
        const double w_theta = config_.mpc.objective.terminal_heading_weight;
        const auto& path = *reference_path_;

        for (int k = 1; k <= N; ++k) {
            // Use integrated spline parameter if available, else project
            double s_ref;
            if (x_ref[k].has_spline()) {
                s_ref = std::clamp(x_ref[k].s, 0.0, path.total_length());
            } else {
                s_ref = path.find_closest_point(x_ref[k].position());
            }
            PathPoint pp = path.get_point_at(s_ref);

            // Normal and tangent vectors at path point
            double ph = pp.heading;
            Eigen::Vector2d t_ref(std::cos(ph), std::sin(ph));   // tangent
            Eigen::Vector2d n_ref(-std::sin(ph), std::cos(ph));  // normal

            // Contouring error sensitivity: e_c = n^T * (pos - path_pos)
            Eigen::RowVectorXd N_c = n_ref.transpose() * P_all[k];  // 1 x n_dec
            double e_c_ref = n_ref.dot(x_ref[k].position() - pp.position);

            // Lag error sensitivity: e_l = -t^T * (pos - path_pos)
            Eigen::RowVectorXd T_l = -t_ref.transpose() * P_all[k];  // 1 x n_dec
            double e_l_ref = -t_ref.dot(x_ref[k].position() - pp.position);

            // Add to Hessian
            H.noalias() += w_c * N_c.transpose() * N_c;
            H.noalias() += w_l * T_l.transpose() * T_l;

            // Add to gradient (computed below alongside goal gradient)
            g.noalias() += w_c * N_c.transpose() * e_c_ref;
            g.noalias() += w_l * T_l.transpose() * e_l_ref;
        }

        // Terminal heading alignment at end of horizon
        if (w_theta > 0) {
            int k_terminal = N;
            double s_terminal;
            if (x_ref[k_terminal].has_spline()) {
                s_terminal = std::clamp(x_ref[k_terminal].s, 0.0, path.total_length());
            } else {
                s_terminal = path.find_closest_point(x_ref[k_terminal].position());
            }
            double desired_heading = path.get_heading_at(s_terminal);
            double heading_err = x_ref[k_terminal].theta - desired_heading;
            // Wrap to [-pi, pi]
            while (heading_err > M_PI) heading_err -= 2 * M_PI;
            while (heading_err < -M_PI) heading_err += 2 * M_PI;

            H.noalias() += w_theta * THETA_all[k_terminal].transpose() * THETA_all[k_terminal];
            g.noalias() += w_theta * THETA_all[k_terminal].transpose() * heading_err;
        }
    }

    // Step 4: Build gradient g
    for (int k = 1; k <= N; ++k) {
        double w_goal = effective_goal_weight;
        if (k == N) w_goal *= 2.0;

        // Position error at reference: p_ref[k] - goal
        Eigen::Vector2d pos_err = x_ref[k].position() - goal;
        g += w_goal * P_all[k].transpose() * pos_err;

        // Velocity error at reference: v_ref[k] - v_target
        double w_vel = config_.mpc.objective.velocity_weight;
        double vel_err = x_ref[k].v - reference_velocity;
        g += w_vel * V_all[k].transpose() * vel_err;
    }

    // Step 5: Build constraint matrix C and RHS d
    // Fixed-normal collision half-spaces (normals frozen from numerical x_ref):
    //   a^T c_d(x) >= b
    // with disc center c_d = [p_x, p_y] + ℓ [cos θ, sin θ].
    // Linearize c_d about x_ref[k]:
    //   c_d ≈ c_bar + J [Δp_x, Δp_y, Δθ]^T
    //   J = [[1, 0, -ℓ sin θ̄], [0, 1, ℓ cos θ̄]]
    // Condensed: a^T J [P; Θ] δu >= b - a^T c_bar

    int n_constraints = static_cast<int>(constraints.size());
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_constraints, n_dec);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(n_constraints);

    for (int i = 0; i < n_constraints; ++i) {
        const auto& con = constraints[i];
        int k = con.k;  // Timestep of this constraint
        if (k < 1 || k > N) continue;

        const Eigen::Vector2d lin_pt =
            (con.linearization_point.squaredNorm() > 1e-20)
                ? con.linearization_point
                : x_ref[k].position();

        const double offset = con.disc_offset;
        const double theta = x_ref[k].theta;
        const Eigen::Vector2d j_theta(
            -offset * std::sin(theta),
             offset * std::cos(theta)
        );

        // a^T J maps (Δp_x, Δp_y, Δθ) → scalar
        const double cx = con.a(0);
        const double cy = con.a(1);
        const double ctheta = con.a.dot(j_theta);

        C.row(i) = cx * P_all[k].row(0)
                 + cy * P_all[k].row(1)
                 + ctheta * THETA_all[k];

        d(i) = con.b - con.a.dot(lin_pt);
    }

    // Step 5b: Contouring constraints (road boundary halfplanes)
    // For each timestep, enforce ego stays within road_width/2 of path centerline.
    // Left boundary:  -n^T * p_ego >= -(n^T * p_path + road_width/2)
    // Right boundary:  n^T * p_ego >=  (n^T * p_path - road_width/2)
    if (config_.mpc.enable_contouring_constraints && reference_path_.has_value()) {
        const auto& path = *reference_path_;
        const double half_width = config_.mpc.constraints.road_width / 2.0;
        const int n_road = 2 * N;  // 2 constraints per timestep (left + right)

        // Expand C and d to include road boundary constraints
        Eigen::MatrixXd C_new(n_constraints + n_road, n_dec);
        Eigen::VectorXd d_new(n_constraints + n_road);
        C_new.topRows(n_constraints) = C;
        d_new.head(n_constraints) = d;

        for (int k = 1; k <= N; ++k) {
            double s_ref;
            if (x_ref[k].has_spline()) {
                s_ref = std::clamp(x_ref[k].s, 0.0, path.total_length());
            } else {
                s_ref = path.find_closest_point(x_ref[k].position());
            }
            PathPoint pp = path.get_point_at(s_ref);
            double ph = pp.heading;
            Eigen::Vector2d n_ref(-std::sin(ph), std::cos(ph));  // normal (left)

            int row_right = n_constraints + 2 * (k - 1);
            int row_left  = n_constraints + 2 * (k - 1) + 1;

            // Right boundary: n^T * p_ego >= n^T * p_path - half_width
            Eigen::RowVector2d nT = n_ref.transpose();
            C_new.row(row_right) = nT * P_all[k];
            d_new(row_right) = (n_ref.dot(pp.position) - half_width) - n_ref.dot(x_ref[k].position());

            // Left boundary: -n^T * p_ego >= -(n^T * p_path + half_width)
            C_new.row(row_left) = -nT * P_all[k];
            d_new(row_left) = -(n_ref.dot(pp.position) + half_width) + n_ref.dot(x_ref[k].position());
        }

        C = C_new;
        d = d_new;
    }

    // Step 5c: Hard velocity bounds  v_k in [min_velocity, max_velocity].
    // Velocity is a state, not a decision variable, so it is enforced through the
    // condensed velocity sensitivity V_all[k] (1 x n_dec, dv_k/d(delta_u)):
    //   v_ref[k] + V_all[k] du <= v_max   =>  -V_all[k] du >= v_ref[k] - v_max
    //   v_ref[k] + V_all[k] du >= v_min   =>   V_all[k] du >= v_min - v_ref[k]
    // Two rows per step k=1..N. Kept in the same >= convention as the other rows.
    if (config_.mpc.constraints.enable_velocity_bounds) {
        const int n_cur = static_cast<int>(C.rows());
        const int n_vel = 2 * N;
        Eigen::MatrixXd C_new(n_cur + n_vel, n_dec);
        Eigen::VectorXd d_new(n_cur + n_vel);
        C_new.topRows(n_cur) = C;
        d_new.head(n_cur) = d;

        for (int k = 1; k <= N; ++k) {
            const double v_ref_k = x_ref[k].v;
            int row_lo = n_cur + 2 * (k - 1);      // lower bound: v_k >= v_min
            int row_hi = n_cur + 2 * (k - 1) + 1;  // upper bound: v_k <= v_max

            C_new.row(row_lo) = V_all[k];
            d_new(row_lo) = config_.mpc.ego.dynamics.min_velocity - v_ref_k;

            C_new.row(row_hi) = -V_all[k];
            d_new(row_hi) = v_ref_k - config_.mpc.ego.dynamics.max_velocity;
        }

        C = C_new;
        d = d_new;
    }

    // Step 6: Box constraints on delta_u
    Eigen::VectorXd lb(n_dec), ub(n_dec);
    for (int k = 0; k < N; ++k) {
        lb(2 * k) = config_.mpc.ego.dynamics.min_acceleration - u_ref[k].a;
        ub(2 * k) = config_.mpc.ego.dynamics.max_acceleration - u_ref[k].a;
        lb(2 * k + 1) = -config_.mpc.ego.dynamics.max_steering_rate - u_ref[k].delta;
        ub(2 * k + 1) = config_.mpc.ego.dynamics.max_steering_rate - u_ref[k].delta;
    }

    QPProblem qp;
    qp.H = H;
    qp.g = g;
    qp.C = C;
    qp.d = d;
    qp.lb = lb;
    qp.ub = ub;

    return qp;
}

std::pair<std::vector<EgoState>, std::vector<EgoInput>>
AdaptiveScenarioMPC::apply_simple_avoidance(
    const EgoState& ego_state,
    std::vector<EgoState> trajectory,
    std::vector<EgoInput> inputs,
    const std::vector<CollisionConstraint>& constraints
) {
    // Group constraints by timestep
    std::map<int, std::vector<CollisionConstraint>> by_k;
    for (const auto& c : constraints) {
        by_k[c.k].push_back(c);
    }

    // Adjust inputs to avoid violations
    std::vector<EgoInput> new_inputs = inputs;

    for (int k = 0; k < static_cast<int>(inputs.size()); ++k) {
        if (by_k.find(k) == by_k.end()) {
            continue;
        }

        // Check violations at this timestep
        if (k + 1 < static_cast<int>(trajectory.size())) {
            Eigen::Vector2d ego_pos = trajectory[k + 1].position();

            auto it = by_k.find(k + 1);
            if (it != by_k.end()) {
                for (const auto& constraint : it->second) {
                    double value = constraint.evaluate(ego_pos);

                    if (value < 0) {
                        // Constraint violated - adjust steering to avoid
                        Eigen::Vector2d avoidance_direction = constraint.a;
                        double current_heading = trajectory[k].theta;

                        // Compute steering adjustment
                        double desired_heading = std::atan2(
                            avoidance_direction(1), avoidance_direction(0)
                        );
                        double heading_diff = desired_heading - current_heading;

                        // Wrap to [-pi, pi]
                        while (heading_diff > M_PI) heading_diff -= 2 * M_PI;
                        while (heading_diff < -M_PI) heading_diff += 2 * M_PI;

                        // Apply steering adjustment
                        double new_w = new_inputs[k].delta + 0.3 * heading_diff;
                        new_w = std::clamp(new_w, -config_.mpc.ego.dynamics.max_steering_rate,
                                          config_.mpc.ego.dynamics.max_steering_rate);
                        new_inputs[k] = EgoInput(new_inputs[k].a, new_w);
                    }
                }
            }
        }
    }

    // Re-propagate with new inputs
    std::vector<EgoState> new_trajectory = ego_dynamics_.rollout(ego_state, new_inputs);

    return {new_trajectory, new_inputs};
}

MPCResult AdaptiveScenarioMPC::generate_safe_fallback(const EgoState& ego_state) {
    std::vector<EgoState> trajectory;
    std::vector<EgoInput> inputs;
    trajectory.reserve(config_.mpc.horizon + 1);
    inputs.reserve(config_.mpc.horizon);

    trajectory.push_back(ego_state);
    EgoState current = ego_state;

    for (int k = 0; k < config_.mpc.horizon; ++k) {
        // Brake gently
        EgoInput input(-1.0, 0.0);
        inputs.push_back(input);

        EgoState next_state = ego_dynamics_.propagate(current, input);
        trajectory.push_back(next_state);
        current = next_state;
    }

    MPCResult result;
    result.success = false;
    result.ego_trajectory = trajectory;
    result.control_inputs = inputs;
    result.cost = std::numeric_limits<double>::infinity();

    return result;
}

void AdaptiveScenarioMPC::set_reference_path(const ReferencePath& path) {
    reference_path_ = path;
}
void AdaptiveScenarioMPC::clear_reference_path() {
    reference_path_.reset();
}

MPCStatistics AdaptiveScenarioMPC::get_statistics() const {
    MPCStatistics stats;
    stats.iteration_count = iteration_count_;
    stats.num_obstacles = static_cast<int>(mode_histories_.size());
    stats.num_scenarios = static_cast<int>(scenarios_.size());

    if (!solve_times_.empty()) {
        double sum = 0.0;
        double max_time = 0.0;
        for (double t : solve_times_) {
            sum += t;
            max_time = std::max(max_time, t);
        }
        stats.avg_solve_time = sum / solve_times_.size();
        stats.max_solve_time = max_time;
    }

    return stats;
}

void AdaptiveScenarioMPC::reset() {
    mode_histories_.clear();
    obstacle_classes_.clear();
    scenarios_.clear();
    reference_trajectory_.clear();
    solve_times_.clear();
    custom_per_obstacle_weights_.clear();
    iteration_count_ = 0;
}

void AdaptiveScenarioMPC::set_custom_mode_weights(
    int obstacle_id,
    const std::map<std::string, double>& weights
) {
    custom_per_obstacle_weights_[obstacle_id] = weights;
}

void AdaptiveScenarioMPC::clear_custom_mode_weights() {
    custom_per_obstacle_weights_.clear();
}

void AdaptiveScenarioMPC::inject_scenario(const Scenario& scenario) {
    pre_injected_scenarios_.push_back(scenario);
}

void AdaptiveScenarioMPC::clear_injected_scenarios() {
    pre_injected_scenarios_.clear();
}

void AdaptiveScenarioMPC::update_mode_model(
    const std::string& mode_id,
    const Eigen::Vector4d& b_new,
    const Eigen::Matrix4d& G_new
) {
    // Update default modes
    auto it = default_modes_.find(mode_id);
    if (it != default_modes_.end()) {
        it->second.b = b_new;
        it->second.G = G_new;
    }

    // Update mode histories for all obstacles
    for (auto& [obs_id, history] : mode_histories_) {
        auto mode_it = history.available_modes.find(mode_id);
        if (mode_it != history.available_modes.end()) {
            mode_it->second.b = b_new;
            mode_it->second.G = G_new;
        }
    }
}

}  // namespace dro_mpc

