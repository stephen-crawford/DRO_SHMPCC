/**
 * @file mpc_controller.cpp
 * @brief Implementation of Adaptive Scenario-Based MPC Controller.
 */

#include "mpc_controller.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>

namespace scenario_mpc {

AdaptiveScenarioMPC::AdaptiveScenarioMPC(const ScenarioMPCConfig& config)
    : config_(config), ego_dynamics_(config.dt) {
    config_.validate();
    default_modes_ = create_obstacle_mode_models(config_.dt);

    // Initialize DRO module from config
    if (config_.enable_dro) {
        DROConfig dro_cfg;
        dro_cfg.epsilon_base = config_.dro_epsilon_base;
        dro_cfg.epsilon_min = config_.dro_epsilon_min;
        dro_cfg.epsilon_max = config_.dro_epsilon_max;
        dro_cfg.adaptive_epsilon = config_.dro_adaptive_epsilon;
        dro_cfg.risk_sigma_scale = config_.dro_risk_sigma_scale;
        dro_ = WassersteinDRO(dro_cfg);
    }

    // Initialize random number generator
    std::random_device rd;
    rng_ = std::mt19937(rd());
}

void AdaptiveScenarioMPC::initialize_obstacle(
    int obstacle_id,
    const std::map<std::string, ModeModel>& available_modes
) {
    const auto& modes = available_modes.empty() ? default_modes_ : available_modes;

    ModeHistory history(obstacle_id, modes);
    history.max_history = config_.horizon * 10;
    mode_histories_[obstacle_id] = history;
}

void AdaptiveScenarioMPC::update_mode_observation(
    int obstacle_id,
    const std::string& observed_mode,
    int timestep
) {
    if (mode_histories_.find(obstacle_id) == mode_histories_.end()) {
        initialize_obstacle(obstacle_id);
    }

    if (timestep < 0) {
        timestep = iteration_count_;
    }

    mode_histories_[obstacle_id].record_observation(timestep, observed_mode);
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

    // Ensure all obstacles have mode histories
    for (const auto& [obs_id, _] : obstacles) {
        if (mode_histories_.find(obs_id) == mode_histories_.end()) {
            initialize_obstacle(obs_id);
        }
    }

    // Step 1: Initialize reference trajectory (warmstart from previous)
    initialize_reference_trajectory(ego_state, goal, reference_velocity);

    // Step 2: Sample scenarios (or use current for compiler; or custom weights for Conformal/Hazard/Bandit)
    bool use_current = use_current_scenarios_next_;
    use_current_scenarios_next_ = false;

    if (!use_current && !custom_per_obstacle_weights_.empty()) {
        std::map<int, std::map<std::string, double>> per_obs_weights;
        for (const auto& [obs_id, _] : obstacles) {
            auto custom_it = custom_per_obstacle_weights_.find(obs_id);
            if (custom_it != custom_per_obstacle_weights_.end()) {
                per_obs_weights[obs_id] = custom_it->second;
            } else {
                auto hist_it = mode_histories_.find(obs_id);
                if (hist_it != mode_histories_.end()) {
                    per_obs_weights[obs_id] = compute_mode_weights(
                        hist_it->second, config_.weight_type,
                        config_.recency_decay, iteration_count_
                    );
                }
            }
        }
        if (!per_obs_weights.empty()) {
            scenarios_ = sample_scenarios_with_weights(
                obstacles, mode_histories_, per_obs_weights,
                config_.horizon, config_.num_scenarios,
                config_.ensure_mode_coverage, &rng_
            );
        }
    }
    if (!use_current && (custom_per_obstacle_weights_.empty() || scenarios_.empty())) {
        if (config_.ensure_mode_coverage) {
            scenarios_ = sample_scenarios_with_mode_coverage(
                obstacles, mode_histories_, config_.horizon,
                config_.num_scenarios, config_.weight_type,
                config_.recency_decay, iteration_count_, &rng_
            );
        } else {
            scenarios_ = sample_scenarios(
                obstacles, mode_histories_, config_.horizon,
                config_.num_scenarios, config_.weight_type,
                config_.recency_decay, iteration_count_, &rng_
            );
        }
    }

    // Pre-compute safe horizon N_s for DRO risk truncation
    int pre_dro_safe_horizon = config_.horizon;
    if (config_.safe_horizon_enabled) {
        int S_pre = static_cast<int>(scenarios_.size());
        pre_dro_safe_horizon = config_.compute_safe_horizon(S_pre);
    }

    // Step 2b: DRO worst-case scenario injection
    // For each obstacle, solve the Wasserstein DRO to find the single most
    // dangerous reachable mode (highest CVaR within W1 ball of P_hat), then
    // deterministically inject its mean trajectory as an additional hard
    // constraint.  This guarantees rare-mode coverage without scaling S.
    // OT (W2 Bures metric) provides the ground cost D[i][j] for the
    // Wasserstein ball, keeping the OT role clean and reviewer-friendly.
    int dro_injected = 0;
    if (config_.enable_dro && !reference_trajectory_.empty()) {
        int next_id = static_cast<int>(scenarios_.size());

        for (const auto& [obs_id, obs_state] : obstacles) {
            auto hist_it = mode_histories_.find(obs_id);
            if (hist_it == mode_histories_.end()) continue;

            // Compute nominal weights from mode history
            auto nominal = compute_mode_weights(
                hist_it->second, config_.weight_type,
                config_.recency_decay, iteration_count_
            );

            // Set observation count for adaptive epsilon
            dro_.set_observation_count(
                static_cast<int>(hist_it->second.observed_modes.size()));

            // Solve DRO: find Q* within W1 ball around P_hat
            // Risk horizon truncated to N_s; multi-disc uses worst disc
            auto dro_result = dro_.compute_worst_case_weights(
                nominal, obs_state, hist_it->second.available_modes,
                reference_trajectory_, config_.horizon,
                config_.ego_radius, config_.obstacle_radius,
                config_.safety_margin,
                pre_dro_safe_horizon,
                config_.num_discs,
                config_.vehicle_length
            );

            // Generate and inject scenario based on injection mode
            Scenario wc(next_id, {}, 0.0);
            if (config_.injection_mode == InjectionMode::ADVERSARIAL) {
                wc = dro_.generate_adversarial_scenario(
                    dro_result, obs_id, obs_state,
                    hist_it->second.available_modes,
                    reference_trajectory_,
                    config_.horizon, next_id,
                    config_.adversarial_sigma_scale
                );
            } else {
                wc = dro_.generate_worst_case_scenario(
                    dro_result, obs_id, obs_state,
                    hist_it->second.available_modes,
                    config_.horizon, next_id
                );
            }
            if (!wc.trajectories.empty()) {
                wc.is_injected = true;  // Protect from dominance pruning
                scenarios_.push_back(std::move(wc));
                next_id++;
                dro_injected++;
            }
        }
    }

    // Step 3: Prune dominated scenarios (Algorithm 3)
    scenarios_ = prune_dominated_scenarios(
        scenarios_,
        reference_trajectory_,
        config_.ego_radius,
        config_.obstacle_radius
    );

    // Verify scenario sufficiency for epsilon guarantee (Part 4)
    {
        int n_x = 4, n_u = 2;
        int d = config_.horizon * n_x + config_.horizon * n_u;
        int S_actual = static_cast<int>(scenarios_.size());
        int S_required = config_.compute_required_scenarios(d);
        double eps_effective = config_.compute_effective_epsilon(S_actual, d);

        if (S_actual < S_required && config_.enforce_scenario_count) {
            // Auto-increase: sample additional scenarios
            int additional_count = S_required - S_actual;
            auto additional = sample_scenarios(
                obstacles,
                mode_histories_,
                config_.horizon,
                additional_count,
                config_.weight_type,
                config_.recency_decay,
                iteration_count_,
                &rng_
            );
            scenarios_.insert(scenarios_.end(), additional.begin(), additional.end());
        } else if (S_actual < 3) {
            // Ensure minimum scenario count even without enforcement
            int additional_count = std::max(
                5, config_.num_scenarios - S_actual
            );
            auto additional = sample_scenarios(
                obstacles,
                mode_histories_,
                config_.horizon,
                additional_count,
                config_.weight_type,
                config_.recency_decay,
                iteration_count_,
                &rng_
            );
            scenarios_.insert(scenarios_.end(), additional.begin(), additional.end());
        }
    }

    // Step 3b: Verify scenario sufficiency for epsilon guarantee (Part 4)
    if (config_.enforce_scenario_count) {
        int n_x = 4, n_u = 2;
        int d = config_.horizon * n_x + config_.horizon * n_u;
        int S_required = config_.compute_required_scenarios(d);
        int S_actual = static_cast<int>(scenarios_.size());
        if (S_actual < S_required) {
            // Auto-increase: sample additional scenarios
            int additional_count = S_required - S_actual;
            auto additional = sample_scenarios(
                obstacles, mode_histories_, config_.horizon,
                additional_count, config_.weight_type,
                config_.recency_decay, iteration_count_, &rng_
            );
            scenarios_.insert(scenarios_.end(), additional.begin(), additional.end());
        }
    }

    // Step 4: Compute linearized constraints (multi-disc D=num_discs)
    auto constraint_start = std::chrono::high_resolution_clock::now();
    auto constraints = compute_linearized_constraints(
        reference_trajectory_,
        scenarios_,
        config_.ego_radius,
        config_.obstacle_radius,
        config_.safety_margin,
        config_.num_discs
    );
    if (!certificate_radii_.empty()) {
        constraints = tighten_constraints_by_certificate(constraints, certificate_radii_);
    }

    // Step 4b: Safe horizon truncation (SH-MPC)
    // Reduce constraint horizon to N_safe based on configured mode:
    // - PRACTICAL:          N_safe = min(N, floor(S / (2*n_u)))
    // - THEORETICAL_SIMPLE: Eq. 23, S >= (2/eps)*(ln(1/beta) + d)
    // - THEORETICAL_TIGHT:  Eq. 25 (very conservative)
    int effective_horizon = config_.horizon;
    if (config_.safe_horizon_enabled) {
        // Use pre-pruning scenario count: SH theory (Theorem 1) applies to
        // the i.i.d. sampled scenarios, not the post-pruning support set.
        int S_for_sh = config_.num_scenarios + dro_injected;
        effective_horizon = config_.compute_safe_horizon(S_for_sh);

        if (effective_horizon < config_.horizon) {
            // Filter out constraints beyond the safe horizon
            constraints.erase(
                std::remove_if(constraints.begin(), constraints.end(),
                    [effective_horizon](const CollisionConstraint& c) {
                        return c.k > effective_horizon;
                    }),
                constraints.end()
            );
        }
    }

    auto constraint_end = std::chrono::high_resolution_clock::now();

    // Step 5: Solve optimization problem
    auto qp_start = std::chrono::high_resolution_clock::now();
    MPCResult result = solve_optimization(
        ego_state, goal, reference_velocity, constraints,
        path_progress, path_length
    );

    auto qp_end = std::chrono::high_resolution_clock::now();

    // Record safe horizon and DRO injection count
    result.safe_horizon = effective_horizon;
    result.num_dro_injected = dro_injected;
    result.constraint_construction_time =
        std::chrono::duration<double>(constraint_end - constraint_start).count();
    result.qp_solve_time =
        std::chrono::duration<double>(qp_end - qp_start).count();

    // Step 6: Remove inactive scenarios (Algorithm 4)
    if (result.success) {
        auto [remaining, active] = remove_inactive_scenarios(
            scenarios_, constraints, result.ego_trajectory
        );
        scenarios_ = remaining;
        result.active_scenarios = std::vector<int>(active.begin(), active.end());

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
        while (static_cast<int>(reference_trajectory_.size()) <= config_.horizon) {
            const EgoState& last = reference_trajectory_.back();
            // Simple constant velocity extension
            EgoState new_state(
                last.x + last.v * std::cos(last.theta) * config_.dt,
                last.y + last.v * std::sin(last.theta) * config_.dt,
                last.theta,
                last.v
            );
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
    trajectory.reserve(config_.horizon + 1);
    trajectory.push_back(start);

    EgoState current = start;

    for (int k = 0; k < config_.horizon; ++k) {
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
        double v = std::min(current.v + 0.5 * config_.dt, reference_velocity);

        EgoState next_state(
            current.x + v * std::cos(desired_theta) * config_.dt,
            current.y + v * std::sin(desired_theta) * config_.dt,
            desired_theta,
            v
        );
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
    double path_length
) {
    if (config_.use_sqp_solver) {
        return solve_optimization_sqp(
            ego_state, goal, reference_velocity, constraints,
            path_progress, path_length
        );
    }

    // Heuristic fallback: simple optimization without CasADi
    int N = config_.horizon;

    auto trajectory = generate_straight_line_trajectory(ego_state, goal, reference_velocity);
    std::vector<EgoInput> inputs;
    inputs.reserve(N);

    for (int k = 0; k < N; ++k) {
        if (k + 1 < static_cast<int>(trajectory.size())) {
            const EgoState& current = trajectory[k];
            const EgoState& next_state = trajectory[k + 1];

            double a = (next_state.v - current.v) / config_.dt;
            double w = (next_state.theta - current.theta) / config_.dt;

            a = std::clamp(a, config_.min_acceleration, config_.max_acceleration);
            w = std::clamp(w, -config_.max_steering_rate, config_.max_steering_rate);

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

    double effective_goal_weight = config_.goal_weight;
    bool progress_aware = (path_progress >= 0 && path_length > 0);
    double progress_fraction = 0.0;
    if (progress_aware) {
        progress_fraction = path_progress / path_length;
        if (progress_fraction > config_.goal_scale_start_fraction) {
            double t = (progress_fraction - config_.goal_scale_start_fraction)
                     / (1.0 - config_.goal_scale_start_fraction);
            double scale = 1.0 + t * (config_.goal_weight_scale_max - 1.0);
            effective_goal_weight = config_.goal_weight * scale;
        }
    }

    double cost = 0.0;
    for (int k = 0; k <= N; ++k) {
        Eigen::Vector2d pos_diff = trajectory[k].position() - goal;
        double weight = effective_goal_weight;
        if (k == N) weight *= 2.0;
        cost += weight * pos_diff.squaredNorm();

        double v_diff = trajectory[k].v - reference_velocity;
        cost += config_.velocity_weight * v_diff * v_diff;

        if (progress_aware) {
            double v_deficit = config_.min_velocity_threshold - trajectory[k].v;
            if (v_deficit > 0) {
                cost += config_.min_velocity_penalty * v_deficit * v_deficit;
            }
        }
    }
    for (int k = 0; k < N; ++k) {
        cost += config_.acceleration_weight * inputs[k].a * inputs[k].a;
        cost += config_.steering_weight * inputs[k].delta * inputs[k].delta;
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
    double path_length
) {
    const int N = config_.horizon;

    // 1. Build initial reference trajectory and extract inputs
    auto x_ref = generate_straight_line_trajectory(ego_state, goal, reference_velocity);
    std::vector<EgoInput> u_ref;
    u_ref.reserve(N);

    for (int k = 0; k < N; ++k) {
        if (k + 1 < static_cast<int>(x_ref.size())) {
            double a = (x_ref[k + 1].v - x_ref[k].v) / config_.dt;
            double w = (x_ref[k + 1].theta - x_ref[k].theta) / config_.dt;
            a = std::clamp(a, config_.min_acceleration, config_.max_acceleration);
            w = std::clamp(w, -config_.max_steering_rate, config_.max_steering_rate);
            u_ref.emplace_back(a, w);
        } else {
            u_ref.emplace_back(0, 0);
        }
    }

    // Re-propagate to get consistent reference
    x_ref = ego_dynamics_.rollout(ego_state, u_ref);

    // Use warmstarted reference if available
    if (!reference_trajectory_.empty() &&
        static_cast<int>(reference_trajectory_.size()) > N) {
        // Extract inputs from warmstarted reference
        std::vector<EgoInput> warm_inputs;
        warm_inputs.reserve(N);
        for (int k = 0; k < N; ++k) {
            const auto& curr = reference_trajectory_[k];
            const auto& next = reference_trajectory_[k + 1];
            double a = (next.v - curr.v) / config_.dt;
            double w = (next.theta - curr.theta) / config_.dt;
            a = std::clamp(a, config_.min_acceleration, config_.max_acceleration);
            w = std::clamp(w, -config_.max_steering_rate, config_.max_steering_rate);
            warm_inputs.emplace_back(a, w);
        }
        auto warm_traj = ego_dynamics_.rollout(ego_state, warm_inputs);

        // Use warmstart if it's reasonable (not too far from ego)
        double warm_dist = (warm_traj[1].position() - ego_state.position()).norm();
        if (warm_dist < 5.0) {
            u_ref = warm_inputs;
            x_ref = warm_traj;
        }
    }

    // 2. SQP loop
    for (int sqp_iter = 0; sqp_iter < config_.sqp_max_iterations; ++sqp_iter) {
        // Build and solve QP subproblem
        QPProblem qp = build_condensed_qp(
            x_ref, u_ref, goal, reference_velocity, constraints,
            path_progress, path_length
        );

        QPSettings qp_settings;
        qp_settings.max_iterations = config_.qp_max_iterations;
        qp_settings.abs_tol = config_.qp_tolerance;
        qp_settings.adaptive_rho = true;

        QPResult qp_result = qp_solver_.solve(qp, qp_settings);

        Eigen::VectorXd delta_u = qp_result.x;

        // Check SQP convergence
        if (delta_u.norm() < config_.sqp_convergence_tol) {
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
                a_new = std::clamp(a_new, config_.min_acceleration, config_.max_acceleration);
                w_new = std::clamp(w_new, -config_.max_steering_rate, config_.max_steering_rate);
                trial_inputs.emplace_back(a_new, w_new);
            }

            auto trial_traj = ego_dynamics_.rollout(ego_state, trial_inputs);
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

    // Compute progress-aware goal weight
    double effective_goal_weight = config_.goal_weight;
    bool progress_aware = (path_progress >= 0 && path_length > 0);
    if (progress_aware) {
        double progress_fraction = path_progress / path_length;
        if (progress_fraction > config_.goal_scale_start_fraction) {
            double t = (progress_fraction - config_.goal_scale_start_fraction)
                     / (1.0 - config_.goal_scale_start_fraction);
            double scale = 1.0 + t * (config_.goal_weight_scale_max - 1.0);
            effective_goal_weight = config_.goal_weight * scale;
        }
    }

    // Compute final cost
    double cost = 0.0;
    for (int k = 0; k <= N; ++k) {
        Eigen::Vector2d pos_diff = x_ref[k].position() - goal;
        double weight = effective_goal_weight;
        if (k == N) weight *= 2.0;
        cost += weight * pos_diff.squaredNorm();

        double v_diff = x_ref[k].v - reference_velocity;
        cost += config_.velocity_weight * v_diff * v_diff;

        if (progress_aware) {
            double v_deficit = config_.min_velocity_threshold - x_ref[k].v;
            if (v_deficit > 0) {
                cost += config_.min_velocity_penalty * v_deficit * v_deficit;
            }
        }
    }
    for (int k = 0; k < N; ++k) {
        cost += config_.acceleration_weight * u_ref[k].a * u_ref[k].a;
        cost += config_.steering_weight * u_ref[k].delta * u_ref[k].delta;
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
    double path_length
) {
    const int N = config_.horizon;
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

        // Extract position and velocity sensitivities
        for (int j = 0; j < k; ++j) {
            Eigen::Matrix<double, 2, 2> Pkj = E * M_new[j];
            P_all[k].block<2, 2>(0, 2 * j) = Pkj;

            Eigen::RowVector2d Vkj = V_row * M_new[j];
            V_all[k].segment<2>(2 * j) = Vkj;
        }

        M_current = M_new;
    }

    // Step 3: Build Hessian H (n_dec x n_dec)
    // H_ij = sum_k w_goal * P[k,i]^T * P[k,j]
    //       + sum_k w_vel * V[k,i]^T * V[k,j]
    //       + diag(w_accel, w_steer, w_accel, w_steer, ...)

    // Compute progress-aware goal weight
    double effective_goal_weight = config_.goal_weight;
    bool progress_aware = (path_progress >= 0 && path_length > 0);
    if (progress_aware) {
        double progress_fraction = path_progress / path_length;
        if (progress_fraction > config_.goal_scale_start_fraction) {
            double t = (progress_fraction - config_.goal_scale_start_fraction)
                     / (1.0 - config_.goal_scale_start_fraction);
            double scale = 1.0 + t * (config_.goal_weight_scale_max - 1.0);
            effective_goal_weight = config_.goal_weight * scale;
        }
    }

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_dec, n_dec);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n_dec);

    for (int k = 1; k <= N; ++k) {
        double w_goal = effective_goal_weight;
        if (k == N) w_goal *= 2.0;  // Terminal cost boost

        // Goal tracking: w_goal * P[k]^T * P[k]
        H += w_goal * P_all[k].transpose() * P_all[k];

        // Velocity tracking: w_vel * V[k]^T * V[k]
        H += config_.velocity_weight * V_all[k].transpose() * V_all[k];
    }

    // Control effort: diagonal terms
    for (int k = 0; k < N; ++k) {
        H(2 * k, 2 * k) += config_.acceleration_weight;
        H(2 * k + 1, 2 * k + 1) += config_.steering_weight;
    }

    // Regularize for positive definiteness
    H.diagonal().array() += 1e-6;

    // Step 4: Build gradient g
    for (int k = 1; k <= N; ++k) {
        double w_goal = effective_goal_weight;
        if (k == N) w_goal *= 2.0;

        // Position error at reference: p_ref[k] - goal
        Eigen::Vector2d pos_err = x_ref[k].position() - goal;
        g += w_goal * P_all[k].transpose() * pos_err;

        // Velocity error at reference: v_ref[k] - v_target
        double vel_err = x_ref[k].v - reference_velocity;
        g += config_.velocity_weight * V_all[k].transpose() * vel_err;
    }

    // Step 5: Build constraint matrix C and RHS d
    // For constraint i at timestep k: a_i^T * p_ego[k] >= b_i
    // Linearized: a_i^T * (p_ref[k] + P[k] * delta_u) >= b_i
    // => a_i^T * P[k] * delta_u >= b_i - a_i^T * p_ref[k]

    int n_constraints = static_cast<int>(constraints.size());
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_constraints, n_dec);
    Eigen::VectorXd d = Eigen::VectorXd::Zero(n_constraints);

    for (int i = 0; i < n_constraints; ++i) {
        const auto& con = constraints[i];
        int k = con.k;  // Timestep of this constraint
        if (k < 1 || k > N) continue;

        // C[i,:] = a^T * P[k]
        Eigen::RowVector2d aT = con.a.transpose();
        C.row(i) = aT * P_all[k];

        // d[i] = b - a^T * p_ref[k] (uses disc linearization point for multi-disc)
        Eigen::Vector2d lin_pt = (con.linearization_point.squaredNorm() > 1e-20) ?
            con.linearization_point : x_ref[k].position();
        d(i) = con.b - con.a.dot(lin_pt);
    }

    // Step 6: Box constraints on delta_u
    Eigen::VectorXd lb(n_dec), ub(n_dec);
    for (int k = 0; k < N; ++k) {
        lb(2 * k) = config_.min_acceleration - u_ref[k].a;
        ub(2 * k) = config_.max_acceleration - u_ref[k].a;
        lb(2 * k + 1) = -config_.max_steering_rate - u_ref[k].delta;
        ub(2 * k + 1) = config_.max_steering_rate - u_ref[k].delta;
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
                        new_w = std::clamp(new_w, -config_.max_steering_rate,
                                          config_.max_steering_rate);
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
    trajectory.reserve(config_.horizon + 1);
    inputs.reserve(config_.horizon);

    trajectory.push_back(ego_state);
    EgoState current = ego_state;

    for (int k = 0; k < config_.horizon; ++k) {
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

void AdaptiveScenarioMPC::set_dro_epsilon_override(std::optional<double> rho) {
    if (rho.has_value()) {
        dro_.set_epsilon_override(*rho);
    } else {
        dro_.clear_epsilon_override();
    }
}
void AdaptiveScenarioMPC::clear_dro_epsilon_override() {
    dro_.clear_epsilon_override();
}

void AdaptiveScenarioMPC::set_custom_mode_weights(int obstacle_id, const std::map<std::string, double>& weights) {
    custom_per_obstacle_weights_[obstacle_id] = weights;
}
void AdaptiveScenarioMPC::clear_custom_mode_weights() {
    custom_per_obstacle_weights_.clear();
}

void AdaptiveScenarioMPC::set_certificate_radii(const std::vector<double>& radii) {
    certificate_radii_ = radii;
}
void AdaptiveScenarioMPC::clear_certificate_radii() {
    certificate_radii_.clear();
}

void AdaptiveScenarioMPC::set_scenarios(const std::vector<Scenario>& scenarios) {
    scenarios_ = scenarios;
}
void AdaptiveScenarioMPC::set_use_current_scenarios_next(bool use) {
    use_current_scenarios_next_ = use;
}

void AdaptiveScenarioMPC::sample_and_set_scenarios(const std::map<int, ObstacleState>& obstacles, int N) {
    if (N <= 0) return;
    if (config_.ensure_mode_coverage) {
        scenarios_ = sample_scenarios_with_mode_coverage(
            obstacles, mode_histories_, config_.horizon, N,
            config_.weight_type, config_.recency_decay, iteration_count_, &rng_
        );
    } else {
        scenarios_ = sample_scenarios(
            obstacles, mode_histories_, config_.horizon, N,
            config_.weight_type, config_.recency_decay, iteration_count_, &rng_
        );
    }
    use_current_scenarios_next_ = true;
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
    scenarios_.clear();
    reference_trajectory_.clear();
    solve_times_.clear();
    iteration_count_ = 0;
}

}  // namespace scenario_mpc
