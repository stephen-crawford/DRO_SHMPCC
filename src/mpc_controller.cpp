/**
 * @file mpc_controller.cpp
 * @brief Implementation of Scenario-Based MPC Controller.
 */

#include "mpc_controller.hpp"
#include "reference_path.hpp"
#include "path_progress_linearization.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <set>
#include <iostream>
#include <limits>
#include <string>

namespace dro_mpc {

namespace {

// A negative signed clearance is a
// violation; a small positive clearance is treated as binding support.
constexpr double kSupportBindingTolerance = 1e-3;

/// Add the distinct scenario IDs whose collision constraints are either
/// binding or violated by `trajectory`.  Scenario support is a property of a
/// joint scenario, so multiple obstacle/stage/disc constraints with the same
/// scenario_id deliberately contribute only once.
void collect_active_or_violated_scenarios(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& trajectory,
    std::set<int>& scenario_ids
) {
    for (const auto& constraint : constraints) {
        if (constraint.scenario_id < 0 || constraint.k < 0 ||
            constraint.k >= static_cast<int>(trajectory.size())) {
            continue;
        }

        const Eigen::Vector2d disc_center = compute_collision_disc_center(
            trajectory[constraint.k], constraint);
        const double signed_clearance = constraint.evaluate(disc_center);
        if (std::isfinite(signed_clearance) &&
            signed_clearance <= kSupportBindingTolerance) {
            scenario_ids.insert(constraint.scenario_id);
        }
    }
}

std::vector<int> sorted_scenario_ids(
    const std::vector<CollisionConstraint>& constraints,
    const std::vector<EgoState>& trajectory
) {
    std::set<int> scenario_ids;
    collect_active_or_violated_scenarios(constraints, trajectory, scenario_ids);
    return {scenario_ids.begin(), scenario_ids.end()};
}

}  // namespace

MPCController::MPCController(const RuntimeConfig& config)
    : config_(config), ego_dynamics_(config.mpc.ego.dynamics, config.mpc.dt) {
    // Resolve the certified
    // sample size from n_bar + R before any scenarios are drawn.  Direct
    // RuntimeConfig users must receive the same behavior as YAML/harness
    // callers, which already normalize during their conversion lifecycle.
    config_.normalize();
    config_.validate();
    default_modes_ = create_obstacle_mode_models(config_.mpc.dt);

    // Initialize DRO module from nested DROConfig.
    if (config_.dro.enabled) {
        dro_ = DRO(config_.dro.solver);
    }

    // Initialize random number generator
    if (config_.random_seed != 0) {
        rng_ = std::mt19937(config_.random_seed);
    } else {
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }
}

void MPCController::initialize_obstacle(
    int obstacle_id,
    int obstacle_class_id,
    const std::map<std::string, ModeModel>& available_modes
) {
    const auto& modes = available_modes.empty() ? default_modes_ : available_modes;

    ModeHistory history(obstacle_id, modes, obstacle_class_id);
    history.max_history_length = config_.mpc.sampling.max_history_length;

    // Reconstruct the class aggregate from every existing sibling. Histories are
    // normally synchronized, so deduplicate observations that were broadcast.
    std::vector<std::pair<int, std::string>> class_observations;
    for (const auto& [other_id, other_class_id] : obstacle_class_ids_) {
        if (other_class_id == obstacle_class_id && other_id != obstacle_id) {
            auto it = mode_histories_.find(other_id);
            if (it != mode_histories_.end()) {
                class_observations.insert(
                    class_observations.end(),
                    it->second.observed_modes.begin(),
                    it->second.observed_modes.end());
            }
        }
    }
    std::sort(class_observations.begin(), class_observations.end());
    class_observations.erase(
        std::unique(class_observations.begin(), class_observations.end()),
        class_observations.end());
    if (history.max_history_length > 0 &&
        static_cast<int>(class_observations.size()) > history.max_history_length) {
        class_observations.erase(
            class_observations.begin(),
            class_observations.begin() +
                (class_observations.size() - history.max_history_length));
    }
    history.observed_modes = std::move(class_observations);

    obstacle_class_ids_[obstacle_id] = obstacle_class_id;
    mode_histories_[obstacle_id] = history;
}

void MPCController::update_mode_observation(
    int obstacle_id,
    int obstacle_class_id,
    const std::string& observed_mode,
    int timestep
) {
    auto class_it = obstacle_class_ids_.find(obstacle_id);
    if (mode_histories_.find(obstacle_id) == mode_histories_.end()
        || class_it == obstacle_class_ids_.end()
        || class_it->second != obstacle_class_id) {
        initialize_obstacle(obstacle_id, obstacle_class_id);
    }

    if (timestep < 0) {
        timestep = iteration_count_;
    }

    // Record observation for all obstacles sharing this class
    const int current_obstacle_class_id = obstacle_class_ids_.at(obstacle_id);
    for (auto& [other_id, hist] : mode_histories_) {
        if (hist.obstacle_class_id == current_obstacle_class_id) {
            hist.record_observation(timestep, observed_mode);
        }
    }
}

MPCResult MPCController::solve(
    const EgoState& ego_state,
    const std::map<int, ObstacleState>& obstacles,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    double path_progress,
    double path_length
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    iteration_count_++;
    last_dro_results_.clear();

    // A scenario certificate is attached to one decision problem.  Keep the
    // controller RNG for reproducibility, but regenerate trajectories from the
    // current obstacle states at every receding-horizon solve.  Retaining old
    // trajectories would leave their k=0 state anchored to a past measurement.
    scenarios_.clear();
    last_linearized_constraints_.clear();

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

    // Step 2: Sample scenarios (DRO reshapes the categorical to q*, then i.i.d. sample)
    std::map<int, std::map<std::string, double>> sampling_weights;
    bool used_external_sampling_weights = false;
    std::map<int, Eigen::MatrixXd> sampling_transitions;
    if (config_.mpc.sampling.markov_jump_system) {
        for (const auto& [obs_id, history] : mode_histories_) {
            std::vector<std::string> modes;
            for (const auto& [mode_id, _] : history.available_modes)
                modes.push_back(mode_id);
            const int mode_count = static_cast<int>(modes.size());
            if (mode_count > 0) {
                sampling_transitions[obs_id] = compute_mode_transition_matrix(
                    history, modes,
                    config_.mpc.sampling.mode_belief.alpha(mode_count),
                    config_.mpc.sampling.mode_belief.kappa(mode_count));
            }
        }
    }

    // When DRO is enabled: compute worst-case distribution q* and resample all S
    // scenarios from it.  Safe-Horizon certification is joint over the full
    // prediction horizon, so DRO's risk assessment must cover that same N.
    if (config_.dro.enabled && !reference_trajectory_.empty()) {
        const int S = config_.mpc.sampling.num_scenarios;
        const int risk_horizon = config_.mpc.horizon;

        // Compute DRO q* from the nominal p hat
        for (const auto& [obs_id, obs_state] : obstacles) {
            auto hist_it = mode_histories_.find(obs_id);
            if (hist_it == mode_histories_.end()) continue;

            // p hat
            auto nominal_weights = compute_mode_weights(
                hist_it->second, config_.mpc.sampling.mode_belief
            );

            if (nominal_weights.empty()) continue;
            // The calibrated ambiguity radius requires an IID-like sample
            // count.  A held mode observed on successive MPC ticks is one
            // evidence episode, not one independent categorical draw.
            dro_.set_observation_count(
                hist_it->second.ambiguity_radius_sample_count());

            // Markov-jump obstacle: hand the DRO risk model the transition chain so
            // per-mode risk is computed over within-horizon switching rather than a
            // held mode. Ordering matches `nominal` (== the solver's mode_ids order).
            Eigen::MatrixXd obs_transition;
            const Eigen::MatrixXd* transition_ptr = nullptr;
            if (config_.mpc.sampling.markov_jump_system) {
                std::vector<std::string> modes_order;
                modes_order.reserve(nominal_weights.size());
                for (const auto& [mid, _] : nominal_weights) modes_order.push_back(mid);
                const int M = static_cast<int>(modes_order.size());
                obs_transition = compute_mode_transition_matrix(
                    hist_it->second, modes_order,
                    config_.mpc.sampling.mode_belief.alpha(M),
                    config_.mpc.sampling.mode_belief.kappa(M));
                transition_ptr = &obs_transition;
            }

            last_dro_results_[obs_id] = dro_.compute_worst_case_weights(
                nominal_weights, obs_state, hist_it->second.available_modes,
                reference_trajectory_, config_.mpc.horizon,
                config_.mpc.ego.radius, config_.obstacle_radius,
                config_.mpc.constraints.safety_margin,
                risk_horizon,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length,
                transition_ptr
            );
        }

        // Resample ALL S scenarios from q*.
        std::map<int, std::map<std::string, double>> per_obs_weights_dro;
        for (const auto& [obs_id, dro_result] : last_dro_results_) {
            per_obs_weights_dro[obs_id] = dro_result.worst_case_weights;
        }

        if (!per_obs_weights_dro.empty()) {
            sampling_weights = per_obs_weights_dro;
            used_external_sampling_weights = true;
            if (config_.mpc.sampling.markov_jump_system) {
                // Seed the Markov chain from Q*: the reweighted belief sets the
                // initial mode distribution, then the estimated transition
                // matrix propagates it over the horizon.
                scenarios_ = sample_scenarios(
                    obstacles, mode_histories_, &per_obs_weights_dro,
                    config_.mpc.horizon, S, config_.mpc.sampling.mode_belief,
                    &sampling_transitions, &rng_);
            } else {
                scenarios_ = sample_scenarios(
                    obstacles, mode_histories_, &per_obs_weights_dro,
                    config_.mpc.horizon, S, config_.mpc.sampling.mode_belief,
                    nullptr, &rng_);
            }
        }
    }

    // When custom per-obstacle weights are set (e.g. from OT predictor),
    // use them for scenario sampling (only fires when DRO is off).
    if (scenarios_.empty() && !custom_per_obstacle_weights_.empty()) {
        sampling_weights = custom_per_obstacle_weights_;
        used_external_sampling_weights = true;
        scenarios_ = sample_scenarios(
            obstacles, mode_histories_, &sampling_weights,
            config_.mpc.horizon, config_.mpc.sampling.num_scenarios,
            config_.mpc.sampling.mode_belief,
            config_.mpc.sampling.markov_jump_system ? &sampling_transitions : nullptr,
            &rng_);
    }

    if (scenarios_.empty()) {
        if (config_.mpc.sampling.markov_jump_system) {
            // Base (non-DRO) Markov path: belief derived from weight_type +
            // the Dirichlet prior, no Q* override.
            scenarios_ = sample_scenarios(
                obstacles, mode_histories_, nullptr,
                config_.mpc.horizon, config_.mpc.sampling.num_scenarios,
                config_.mpc.sampling.mode_belief, &sampling_transitions, &rng_);
        } else {
            scenarios_ = sample_scenarios(
                obstacles, mode_histories_, nullptr, config_.mpc.horizon,
                config_.mpc.sampling.num_scenarios,
                config_.mpc.sampling.mode_belief, nullptr, &rng_
            );
        }
    }

    // Clear custom weights after use (they're set per-solve by external code)
    custom_per_obstacle_weights_.clear();

    // Safe-Horizon sizes S from the total support cap n̄ + R, not
    // from horizon length or decision dimension.  Normalized configurations
    // sample this target immediately; this fallback keeps the invariant true
    // if a caller provides an existing undersized scenario set.
    int S_actual = static_cast<int>(scenarios_.size());
    const bool support_certification_enabled = config_.mpc.uses_safe_horizon();
    const int S_required = support_certification_enabled
        ? config_.compute_required_scenarios()
        : S_actual;

    auto next_scenario_id = [&]() {
        int next_id = 0;
        for (const auto& scenario : scenarios_) {
            next_id = std::max(next_id, scenario.scenario_id + 1);
        }
        return next_id;
    };

    auto sample_additional_scenarios = [&](int count, int scenario_id_offset) {
        if (config_.mpc.sampling.markov_jump_system) {
            const auto* initial_belief =
                (used_external_sampling_weights && !sampling_weights.empty())
                    ? &sampling_weights : nullptr;
            return sample_scenarios(
                obstacles, mode_histories_, initial_belief,
                config_.mpc.horizon, count,
                config_.mpc.sampling.mode_belief, &sampling_transitions, &rng_,
                scenario_id_offset);
        }
        return sample_scenarios(
            obstacles, mode_histories_,
            (used_external_sampling_weights && !sampling_weights.empty())
                ? &sampling_weights : nullptr,
            config_.mpc.horizon, count, config_.mpc.sampling.mode_belief,
            nullptr, &rng_, scenario_id_offset
        );
    };

    // scenario_module uses the explicit automatic-sizing setting; with it off
    // an intentionally static sample count is left untouched.
    const bool enforce_support_sample_count = support_certification_enabled &&
        config_.mpc.sampling.automatically_compute_sample_size;
    if (S_actual < S_required && enforce_support_sample_count) {
        // Append independent scenarios with fresh joint IDs.  Reusing IDs
        // would collapse distinct draws during support-union accounting.
        int additional_count = S_required - S_actual;
        auto additional = sample_additional_scenarios(
            additional_count, next_scenario_id());
        scenarios_.insert(scenarios_.end(), additional.begin(), additional.end());
    } else if (S_actual < 3) {
        // Ensure minimum scenario count even without enforcement
        int additional_count = std::max(
            5, config_.mpc.sampling.num_scenarios - S_actual
        );
        auto additional = sample_additional_scenarios(
            additional_count, next_scenario_id());
        scenarios_.insert(scenarios_.end(), additional.begin(), additional.end());
    }

    S_actual = static_cast<int>(scenarios_.size());
    const bool sample_count_sufficient = support_certification_enabled &&
        S_actual >= S_required;


    // Step 4: Fixed collision normals from the numerical linearization trajectory.
    // Normals are held constant for the subsequent QP (Case B also linearizes
    // heading-dependent disc centers about the same numerical reference).
    auto constraint_start = std::chrono::high_resolution_clock::now();

    // de Groot 2023 Definition-2 geometric dominance pruning: drop scenarios whose
    // collision half-spaces are IMPLIED (on the reachable ball) by a more-restrictive
    // scenario's, evaluated at the reference trajectory.

    const double combined_radius = config_.combined_radius();

    // Preserve the complete i.i.d. draw set in `scenarios_` for this decision
    // problem: S and scenario identity define its certificate.  Dominance
    // reduction is only a local constraint-construction optimization for this
    // numerical linearization; the next receding-horizon solve draws a fresh,
    // state-conditioned set above.
   // This remains dynamically consistent.
    const auto dynamic_reference = reference_trajectory_;

    // Separate copy used only for collision geometry.
    auto constraint_reference = dynamic_reference;

    if (config_.mpc.uses_safe_horizon()) {
        const bool anchor_ok =
            prepare_safe_horizon_anchors(
                constraint_reference,
                scenarios_,
                combined_radius,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length);

        if (!anchor_ok) {
            throw std::runtime_error(
                "Safe-Horizon anchor projection failed.");
        }
    }

    auto all_constraints =
        compute_linearized_constraints(
            constraint_reference,
            scenarios_,
            config_.mpc.ego.radius,
            config_.obstacle_radius,
            config_.mpc.constraints.safety_margin,
            config_.mpc.ego.num_discs,
            config_.mpc.ego.length);

    const double max_abs_velocity =
        std::max(
            std::abs(config_.mpc.ego.dynamics.min_velocity),
            std::abs(config_.mpc.ego.dynamics.max_velocity));

    auto constraints =
        reduce_to_free_space_polytopes(
            all_constraints,
            constraint_reference,
            dynamic_reference,
            config_.mpc.ego.num_discs,
            config_.mpc.ego.length,
            max_abs_velocity,
            config_.mpc.dt,
            20);

    // Safe Horizon refers to support-bounded joint-risk
    // certification across the *complete* MPC horizon.  Do not erase late
    // collision stages and do not apply the optional distance filter here:
    // either operation would silently change the sampled constraint problem
    // whose support union is being certified.  Non-SH controllers retain the
    // engineering clearance filter for their ordinary performance path.
    if (!config_.mpc.uses_safe_horizon()) {
        constraints = filter_constraints_by_clearance(
            constraints, reference_trajectory_,
            config_.mpc.constraints.clearance_filter_distance);
    }

    if (capture_linearized_constraints_) last_linearized_constraints_ = constraints;

    auto constraint_end = std::chrono::high_resolution_clock::now();

    // Step 5: Solve optimization problem
    auto qp_start = std::chrono::high_resolution_clock::now();
    MPCResult result = solve_optimization(
        ego_with_spline, goal, reference_velocity, constraints,
        path_progress, path_length
    );

    auto qp_end = std::chrono::high_resolution_clock::now();

    result.sampled_scenarios = S_actual;
    result.required_scenarios = support_certification_enabled ? S_required : -1;
    result.sample_count_sufficient = sample_count_sufficient;
    result.certified_horizon = -1;
    if (!support_certification_enabled) {
        result.certificate_status = SafeHorizonCertificateStatus::NOT_REQUESTED;
    } else if (!sample_count_sufficient) {
        result.certificate_status =
            SafeHorizonCertificateStatus::INSUFFICIENT_SCENARIOS;
    } else if (!result.sampled_constraints_satisfied) {
        result.certificate_status = SafeHorizonCertificateStatus::PLAN_INFEASIBLE;
    } else if (result.used_fallback) {
        result.certificate_status = SafeHorizonCertificateStatus::FALLBACK_NOT_CERTIFIED;
    } else if (result.support_cap_status == SupportCapStatus::NOT_EVALUATED) {
        result.certificate_status =
            SafeHorizonCertificateStatus::SUPPORT_NOT_EVALUATED;
    } else if (result.support_cap_status == SupportCapStatus::SUPPORT_EXCEEDED) {
        result.certificate_status = SafeHorizonCertificateStatus::SUPPORT_EXCEEDED;
    } else if (result.support_cap_status == SupportCapStatus::WITHIN_LIMIT) {
        // Only this conjunction represents the full-horizon certificate.
        result.certificate_status = SafeHorizonCertificateStatus::CERTIFIED;
        result.certified_horizon = config_.mpc.horizon;
    }
    for (const auto& [_, dro_result] : last_dro_results_) {
        result.ambiguity_radius_used = std::max(
            result.ambiguity_radius_used, dro_result.rho_used);
        result.dro_risk_evaluation_time +=
            dro_result.risk_diagnostics.evaluation_seconds;
    }
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

void MPCController::initialize_reference_trajectory(
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

std::vector<EgoState> MPCController::generate_straight_line_trajectory(
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

        // MPCC seeds follow the reference geometry, independently of an endpoint
        // or a requested tracking speed; its objective rewards spline progress.
        const bool is_mpcc = config_.mpc.type == MPCType::MPCC ||
                             config_.mpc.type == MPCType::SH_MPCC;
        if (is_mpcc && reference_path_.has_value()) {
            desired_theta = reference_path_->get_heading_at(current.s);
        }
        const double seed_speed = is_mpcc ? config_.mpc.ego.dynamics.max_velocity
                                         : reference_velocity;
        double v = std::min(current.v + 0.5 * config_.mpc.dt, seed_speed);

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

MPCResult MPCController::solve_optimization(
    const EgoState& ego_state,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    const std::vector<CollisionConstraint>& constraints,
    double path_progress,
    double path_length
) {
    return solve_optimization_sqp(
        ego_state, goal, reference_velocity, constraints,
        path_progress, path_length
    );
}

// ============================================================================
// SQP Solver
// ============================================================================

MPCResult MPCController::solve_optimization_sqp(
    const EgoState& ego_state,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    const std::vector<CollisionConstraint>& constraints,
    double path_progress,
    double path_length
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
            w = std::clamp(w, -config_.mpc.ego.dynamics.max_omega, config_.mpc.ego.dynamics.max_omega);
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
            w = std::clamp(w, -config_.mpc.ego.dynamics.max_omega, config_.mpc.ego.dynamics.max_omega);
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

    // Keep the condensed-QP nominal dynamically consistent: its sensitivity-only
    // recursion assumes x_ref[k+1] == f(x_ref[k], u_ref[k]). Geometric preparation
    // belongs to the separate normal anchors, never to this control rollout.

    // de Groot's online support estimate is the union of distinct scenarios
    // that are active OR violated at each SQP iterate.  It is intentionally
    // not just the support set of the final trajectory: a scenario that held
    // any intermediate convex approximation in place must remain counted.
    // The configuration resolves the reference-compatible total cap n̄ + R.
    const bool support_certification_enabled = config_.mpc.uses_safe_horizon();
    const int support_limit = support_certification_enabled
        ? config_.support_limit()
        : -1;
    std::set<int> support_scenarios;
    int support_iterations_evaluated = 0;
    auto account_support_evaluation = [&](const std::vector<EgoState>& trajectory) {
        if (!support_certification_enabled) return true;
        collect_active_or_violated_scenarios(
            constraints, trajectory, support_scenarios);
        ++support_iterations_evaluated;
        return static_cast<int>(support_scenarios.size()) <= support_limit;
    };

    // 2. SQP loop
    for (int sqp_iter = 0; sqp_iter < config_.solver.sqp_max_iterations; ++sqp_iter) {
        // Build and solve QP subproblem
        QPProblem qp = build_condensed_qp(
            x_ref, u_ref, goal, reference_velocity, constraints,
            path_progress, path_length
        );

        QPSettings qp_settings;
        qp_settings.max_iterations = config_.solver.qp_max_iterations;
        qp_settings.tolerance = config_.solver.qp_tolerance;
        
        std::cerr
        << "[SQP] before_solve"
        << " iter=" << sqp_iter
        << " vars=" << qp.H.rows()
        << " constraints=" << qp.C.rows()
        << " max_d=" << (qp.d.size() ? qp.d.maxCoeff() : 0.0)
        << std::endl;

        double max_d = -std::numeric_limits<double>::infinity();
        int worst_row = -1;

        for (int i = 0; i < qp.d.size(); ++i) {
            if (qp.d(i) > max_d) {
                max_d = qp.d(i);
                worst_row = i;
            }
        }

        const int n_collision =
            static_cast<int>(constraints.size());

        const int n_road =
            (config_.mpc.enable_contouring_constraints &&
            reference_path_.has_value())
                ? 2 * N
                : 0;

        std::string worst_type = "unknown";

        if (worst_row >= 0) {
            if (worst_row < n_collision) {
                worst_type = "collision";
            } else if (worst_row < n_collision + n_road) {
                worst_type = "road";
            } else {
                worst_type = "velocity";
            }
        }

        std::cerr
            << "[QP WORST]"
            << " iter=" << sqp_iter
            << " row=" << worst_row
            << " type=" << worst_type
            << " d=" << max_d
            << std::endl;

        if (worst_row >= 0 && worst_row < n_collision) {
            const auto& con = constraints[worst_row];

            std::cerr
                << "[QP WORST COLLISION]"
                << " scenario=" << con.scenario_id
                << " k=" << con.k
                << " obstacle=" << con.obstacle_id
                << " disc=" << con.disc_index
                << std::endl;
        }

        QPResult qp_result = qp_solver_.solve(qp, qp_settings);

        std::cerr
        << "[SQP] after_solve"
        << " iter=" << sqp_iter
        << " converged=" << qp_result.converged
        << " status=" << qp_result.status
        << " iterations=" << qp_result.iterations
        << " primal_residual=" << qp_result.primal_residual
        << " dual_residual=" << qp_result.dual_residual
        << " x_size=" << qp_result.x.size()
        << " x_finite=" << qp_result.x.allFinite()
        << std::endl;

        Eigen::VectorXd delta_u = qp_result.x;

        // An invalid primal vector is not an SQP iterate and must not be used
        // to fabricate a support certificate (or indexed below).
        if (delta_u.size() != 2 * N || !delta_u.allFinite()) {
            break;
        }

        // Check SQP convergence
        if (delta_u.norm() < config_.solver.sqp_convergence_tol) {
            account_support_evaluation(x_ref);
            break;
        }

        // Line search: try full step, then half, then quarter.// Line search: try full step, then half, then quarter.
        //
        // If the current iterate violates collision constraints, require the
        // line search to reduce that violation.
        //
        // If the current iterate is already collision-feasible, accept the
        // largest trial step that remains collision-feasible. This allows the
        // QP objective (MPCC contouring/progress/etc.) to actually move the
        // trajectory.
        auto [current_violation, _current_violated] =
            evaluate_constraint_violation(constraints, x_ref);

        double alpha = 1.0;

        std::vector<EgoInput> best_inputs = u_ref;
        std::vector<EgoState> best_traj = x_ref;
        double best_violation = current_violation;

        bool accepted = false;
        double accepted_alpha = 0.0;

        for (int ls = 0; ls < 3; ++ls) {
            std::vector<EgoInput> trial_inputs;
            trial_inputs.reserve(N);

            for (int k = 0; k < N; ++k) {
                double a_new =
                    u_ref[k].a + alpha * delta_u(2 * k);

                double w_new =
                    u_ref[k].omega + alpha * delta_u(2 * k + 1);

                a_new = std::clamp(
                    a_new,
                    config_.mpc.ego.dynamics.min_acceleration,
                    config_.mpc.ego.dynamics.max_acceleration);

                w_new = std::clamp(
                    w_new,
                    -config_.mpc.ego.dynamics.max_omega,
                    config_.mpc.ego.dynamics.max_omega);

                trial_inputs.emplace_back(a_new, w_new);
            }

            std::vector<EgoState> trial_traj;

            if (reference_path_.has_value()) {
                trial_traj = ego_dynamics_.rollout_with_spline(
                    ego_state,
                    trial_inputs,
                    *reference_path_);
            } else {
                trial_traj = ego_dynamics_.rollout(
                    ego_state,
                    trial_inputs);
            }

            auto [trial_violation, _trial_violated] =
                evaluate_constraint_violation(
                    constraints,
                    trial_traj);

            if (current_violation <= 0.0) {
                // Current trajectory is collision-feasible.
                //
                // Take the largest QP step that preserves feasibility.
                // Since alpha decreases each iteration, the first feasible
                // candidate is the preferred one.
                if (trial_violation <= 0.0) {
                    best_violation = trial_violation;
                    best_inputs = trial_inputs;
                    best_traj = trial_traj;
                    accepted = true;
                    accepted_alpha = alpha;
                    break;
                }
            } else {
                // Current trajectory is infeasible.
                // Only accept a candidate that actually improves feasibility.
                if (trial_violation < best_violation) {
                    best_violation = trial_violation;
                    best_inputs = trial_inputs;
                    best_traj = trial_traj;
                    accepted = true;
                    accepted_alpha = alpha;
                }

                if (best_violation <= 0.0) {
                    break;
                }
            }

            alpha *= 0.5;
        }

        std::cerr
            << "[SQP LINESEARCH]"
            << " iter=" << sqp_iter
            << " current_violation=" << current_violation
            << " accepted_violation=" << best_violation
            << " accepted=" << accepted
            << " alpha=" << accepted_alpha
            << " delta_u_norm=" << delta_u.norm()
            << std::endl;

        // If the current trajectory is infeasible and no candidate improves it,
        // stop rather than deliberately making feasibility worse.
        if (!accepted && current_violation > 0.0) {
            std::cerr
                << "[SQP] line search stalled"
                << " iter=" << sqp_iter
                << " violation=" << current_violation
                << std::endl;

            break;
        }

        // If current_violation == 0 and no candidate is feasible, retaining the
        // existing feasible iterate is preferable to stepping into collision.
        if (!accepted && current_violation <= 0.0) {
            std::cerr
            << "[SQP] no additional feasible SQP step; "
            << "keeping current collision-feasible iterate"
            << " iter=" << sqp_iter
            << std::endl;
        }

        // Apply accepted SQP step.
        u_ref = best_inputs;
        x_ref = best_traj;

        std::set<int> before = support_scenarios;

        const bool support_ok =
            account_support_evaluation(x_ref);

        for (int id : support_scenarios) {
            if (before.find(id) == before.end()) {
                std::cerr
                    << "[SQP SUPPORT ADD]"
                    << " iter=" << sqp_iter
                    << " scenario=" << id
                    << std::endl;
            }
        }
        std::cerr
        << "[SQP SUPPORT]"
        << " iter=" << sqp_iter
        << " support_size=" << support_scenarios.size()
        << " support_limit=" << support_limit
        << " support_ok=" << support_ok
        << std::endl;

        if (!support_ok) {
            std::cerr
                << "[SQP] support cap exceeded; "
                << "continuing optimization without certificate"
                << std::endl;
        }
    }

    // Hard velocity-bound enforcement on the returned plan.
    // The velocity update is exact (v_{k+1} = v_k + a_k dt), so clamping each
    // acceleration to the interval that keeps v_{k+1} in [min_velocity, max_velocity]
    // (intersected with the accel box) guarantees the bound in every downstream
    // rollout -- including the harness, which re-applies these inputs -- regardless
    // of whether the QP solver fully converged the soft velocity rows above. When the
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
            u_ref[k] = EgoInput(a, u_ref[k].omega);
            v_cur += a * config_.mpc.dt;                      // exact velocity propagation
        }
        // Re-roll the trajectory so x_ref matches the clamped inputs.
        x_ref = reference_path_.has_value()
                    ? ego_dynamics_.rollout_with_spline(ego_state, u_ref, *reference_path_)
                    : ego_dynamics_.rollout(ego_state, u_ref);
    }

    // The hard velocity backstop may change the returned trajectory after the
    // last SQP solve.  Include that returned plan in the monotone support
    // union as a final verification pass, so result support never describes a
    // different trajectory than the one handed to the caller.
    account_support_evaluation(x_ref);

    const bool is_mpcc = config_.mpc.type == MPCType::MPCC ||
                         config_.mpc.type == MPCType::SH_MPCC;
    const double effective_goal_weight = is_mpcc ? 0.0 : config_.mpc.objective.goal_weight;
    const double effective_velocity_weight = config_.mpc.objective.velocity_weight;
    (void)path_progress; (void)path_length;

    // Compute final cost
    double cost = 0.0;

    for (int k = 0; k <= N; ++k) {
        Eigen::Vector2d pos_diff = x_ref[k].position() - goal;
        double weight = effective_goal_weight;
        if (k == N) weight *= 2.0;
        cost += weight * pos_diff.squaredNorm();

        double w_vel = effective_velocity_weight;
        double v_diff = x_ref[k].v - reference_velocity;
        cost += w_vel * v_diff * v_diff;

    }
    for (int k = 0; k < N; ++k) {
        cost += config_.mpc.objective.acceleration_weight * u_ref[k].a * u_ref[k].a;
        cost += config_.mpc.objective.steering_weight * u_ref[k].omega * u_ref[k].omega;
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

    if (is_mpcc) {
        cost -= config_.mpc.objective.progress_weight *
            (x_ref[N].s - x_ref[0].s) / (N * config_.mpc.dt);
    }

    // Check feasibility
    auto [final_violation, violated_constraints] =
    evaluate_constraint_violation(constraints, x_ref);

    std::cerr
    << "[SQP FINAL]"
    << " final_violation=" << final_violation
    << " violated_constraints=" << violated_constraints.size()
    << std::endl;

    const auto final_active_scenarios =
    sorted_scenario_ids(
        constraints,
        x_ref);

    std::cerr
    << "[SUPPORT FINAL]"
    << " sqp_union=" << support_scenarios.size()
    << " final_active=" << final_active_scenarios.size()
    << " support_limit=" << support_limit
    << std::endl;

    std::cerr << "[SUPPORT FINAL IDS]";

    for (int id : final_active_scenarios) {
        std::cerr << " " << id;
    }

    std::cerr << std::endl;

    std::cerr << "[SUPPORT UNION IDS]";

    for (int id : support_scenarios) {
        std::cerr << " " << id;
    }

    std::cerr << std::endl;

    for (const auto& con : violated_constraints) {
        const Eigen::Vector2d disc_center =
            compute_collision_disc_center(x_ref.at(con.k), con);

    const double signed_clearance =
        con.evaluate(disc_center);

    std::cerr
    << "[FINAL VIOL]"
    << " scenario=" << con.scenario_id
    << " k=" << con.k
    << " obstacle=" << con.obstacle_id
    << " disc=" << con.disc_index
    << " signed_clearance=" << signed_clearance
    << std::endl;
}
    bool feasible = (final_violation <= 0.01);

    MPCResult result;
    result.success = feasible;
    result.sampled_constraints_satisfied = feasible;
    result.ego_trajectory = x_ref;
    result.control_inputs = u_ref;
    result.cost = cost;

    auto populate_active_scenarios = [&constraints](
        const std::vector<EgoState>& trajectory,
        MPCResult& mpc_result
    ) {
        mpc_result.active_scenarios = sorted_scenario_ids(constraints, trajectory);
    };

    auto populate_support_accounting = [&](MPCResult& mpc_result) {
        if (!support_certification_enabled) {
            mpc_result.support_scenarios.clear();
            mpc_result.support_size = 0;
            mpc_result.support_limit = -1;
            mpc_result.support_cap_satisfied = false;
            mpc_result.support_cap_status = SupportCapStatus::NOT_EVALUATED;
            mpc_result.support_iterations_evaluated = 0;
            return;
        }

        mpc_result.support_scenarios.assign(
            support_scenarios.begin(), support_scenarios.end());
        mpc_result.support_size = static_cast<int>(support_scenarios.size());
        mpc_result.support_limit = support_limit;
        mpc_result.support_iterations_evaluated = support_iterations_evaluated;

        if (support_iterations_evaluated == 0) {
            // A cap cannot certify a solve for which no finite SQP iterate was
            // inspected.  Keep this distinct from a valid zero-support solve.
            mpc_result.support_cap_satisfied = false;
            mpc_result.support_cap_status = SupportCapStatus::NOT_EVALUATED;
            return;
        }

        mpc_result.support_cap_satisfied =
            mpc_result.support_size <= mpc_result.support_limit;
        mpc_result.support_cap_status = mpc_result.support_cap_satisfied
            ? SupportCapStatus::WITHIN_LIMIT
            : SupportCapStatus::SUPPORT_EXCEEDED;
    };
    populate_active_scenarios(x_ref, result);
    populate_support_accounting(result);

    // If infeasible, try safe fallback
    if (!feasible) {
        auto fallback = generate_safe_fallback(ego_state);
        auto [fb_viol, fb_violated] = evaluate_constraint_violation(constraints, fallback.ego_trajectory);
        // An infeasible SQP plan is never executable. Validate the existing
        // braking fallback against the sampled collision rows and hard limits.

        std::cerr
        << "[SQP FALLBACK]"
        << " final_violation=" << final_violation
        << " fallback_violation=" << fb_viol
        << " fallback_violated_constraints="
        << fb_violated.size()
        << std::endl;
        fallback.used_fallback = true;
        fallback.sampled_constraints_satisfied = std::isfinite(fb_viol) && fb_viol <= 0.01;
        bool admissible = fallback.sampled_constraints_satisfied;
        for (const auto& input : fallback.control_inputs) {
            admissible = admissible && input.to_array().allFinite() &&
                input.a >= config_.mpc.ego.dynamics.min_acceleration &&
                input.a <= config_.mpc.ego.dynamics.max_acceleration &&
                std::abs(input.omega) <= config_.mpc.ego.dynamics.max_omega;
        }
        double fallback_s = ego_state.s;
        if (reference_path_.has_value() && fallback_s < 0.0)
            fallback_s = reference_path_->find_closest_point(ego_state.position());
        for (std::size_t k = 0; k < fallback.ego_trajectory.size(); ++k) {
            const auto& state = fallback.ego_trajectory[k];
            admissible = admissible && state.to_array().allFinite();
            if (k > 0 && config_.mpc.constraints.enable_velocity_bounds) {
                admissible = admissible && state.v >= config_.mpc.ego.dynamics.min_velocity &&
                    state.v <= config_.mpc.ego.dynamics.max_velocity;
            }
            if (config_.mpc.enable_contouring_constraints && reference_path_.has_value()) {
                fallback_s = reference_path_->find_closest_point(state.position(), fallback_s);
                const auto point = reference_path_->get_point_at(fallback_s);
                const Eigen::Vector2d normal(-std::sin(point.heading), std::cos(point.heading));
                if (k > 0)
                    admissible = admissible && std::abs(normal.dot(state.position() - point.position))
                        <= config_.mpc.constraints.road_width / 2.0;
            }
        }
        fallback.success = admissible;
        populate_active_scenarios(fallback.ego_trajectory, fallback);
        // Preserve the original SQP support union; a fallback does not inherit
        // a certificate for a different optimization result.
        populate_support_accounting(fallback);
        return fallback;
    }

    return result;
}

QPProblem MPCController::build_condensed_qp(
    const std::vector<EgoState>& x_ref,
    const std::vector<EgoInput>& u_ref,
    const Eigen::Vector2d& goal,
    double reference_velocity,
    const std::vector<CollisionConstraint>& constraints,
    double path_progress,
    double path_length
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

    const bool is_mpcc = config_.mpc.type == MPCType::MPCC ||
                         config_.mpc.type == MPCType::SH_MPCC;
    const double effective_goal_weight = is_mpcc ? 0.0 : config_.mpc.objective.goal_weight;
    const double effective_velocity_weight = config_.mpc.objective.velocity_weight;
    (void)path_progress; (void)path_length;

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_dec, n_dec);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n_dec);

    for (int k = 1; k <= N; ++k) {
        double w_goal = effective_goal_weight;
        if (k == N) w_goal *= 2.0;  // Terminal cost boost

        // Goal tracking: w_goal * P[k]^T * P[k]
        H += w_goal * P_all[k].transpose() * P_all[k];

        // Velocity tracking: w_vel * V[k]^T * V[k]
        double w_vel = effective_velocity_weight;
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
        double w_vel = effective_velocity_weight;
        double vel_err = x_ref[k].v - reference_velocity;
        g += w_vel * V_all[k].transpose() * vel_err;
    }

    if (is_mpcc && reference_path_.has_value()) {
        // s_k = projection(p_k, s_{k-1}); chain through the active projection
        // branch and the existing condensed dynamics. The initial s is fixed.
        Eigen::RowVectorXd progress_sensitivity = Eigen::RowVectorXd::Zero(n_dec);
        for (int k = 1; k <= N; ++k) {
            const auto derivative = linearize_path_progress(
                *reference_path_, x_ref[k].position(), x_ref[k - 1].s);
            progress_sensitivity = derivative.position * P_all[k] +
                derivative.previous_progress * progress_sensitivity;
        }
        // Existing quadratic terms represent half the reported squared costs.
        // Apply the same 1/2 scaling to the linear progress reward.
        g -= (0.5 * config_.mpc.objective.progress_weight /
              (N * config_.mpc.dt)) * progress_sensitivity.transpose();
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

        // Anchor the row at x_ref[k], the SAME point P_all/THETA_all differentiate
        // about. con.linearization_point records where the NORMAL was frozen, which
        // is a different (earlier) trajectory: x_ref is re-rolled from the current
        // ego state with clamped warm inputs, then moved
        // each SQP iteration. Anchoring on it would make the row mis-state clearance
        // by a^T(c_bar_frozen - c_d(x_ref[k])) in an uncontrolled direction.
        const auto row = linearize_constraint_at_state(con, x_ref[k]);

        C.row(i) = row.gradient(0) * P_all[k].row(0)
                 + row.gradient(1) * P_all[k].row(1)
                 + row.gradient(2) * THETA_all[k];

        // a^T c_d(x_ref[k]) + (a^T J) δu >= b  =>  C δu >= -clearance(x_ref[k]).
        d(i) = -row.value;
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
        lb(2 * k + 1) = -config_.mpc.ego.dynamics.max_omega - u_ref[k].omega;
        ub(2 * k + 1) = config_.mpc.ego.dynamics.max_omega - u_ref[k].omega;
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

MPCResult MPCController::generate_safe_fallback(const EgoState& ego_state) {
    std::vector<EgoState> trajectory;
    std::vector<EgoInput> inputs;
    trajectory.reserve(config_.mpc.horizon + 1);
    inputs.reserve(config_.mpc.horizon);

    trajectory.push_back(ego_state);
    EgoState current = ego_state;

    for (int k = 0; k < config_.mpc.horizon; ++k) {
        // Brake gently to zero, then hold instead of predicting reverse motion.
        double acceleration = std::max(-1.0, -current.v / config_.mpc.dt);
        // Avoid a negative terminal speed caused solely by division rounding.
        if (current.v >= 0.0 && current.v + config_.mpc.dt * acceleration < 0.0)
            acceleration = std::nextafter(acceleration, 0.0);
        EgoInput input(acceleration, 0.0);
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

void MPCController::set_reference_path(const ReferencePath& path) {
    reference_path_ = path;
}
void MPCController::clear_reference_path() {
    reference_path_.reset();
}

MPCStatistics MPCController::get_statistics() const {
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

void MPCController::reset() {
    mode_histories_.clear();
    obstacle_class_ids_.clear();
    scenarios_.clear();
    last_linearized_constraints_.clear();
    reference_trajectory_.clear();
    solve_times_.clear();
    custom_per_obstacle_weights_.clear();
    qp_solver_.clear();
    iteration_count_ = 0;
}

void MPCController::set_custom_mode_weights(
    int obstacle_id,
    const std::map<std::string, double>& weights
) {
    custom_per_obstacle_weights_[obstacle_id] = weights;
}

void MPCController::clear_custom_mode_weights() {
    custom_per_obstacle_weights_.clear();
}

    void MPCController::update_mode_model(
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
