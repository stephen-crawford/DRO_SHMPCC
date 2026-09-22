/**

@file mpc_controller.cpp

@brief Implementation of Scenario-Based MPC Controller.
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
#include <array>

namespace dro_mpc {

namespace {

// A negative signed clearance is a
// violation; a small positive clearance is treated as binding support.
constexpr double kSupportBindingTolerance = 1e-3;

/// Add the distinct scenario IDs whose collision constraints are either
/// binding or violated by trajectory.  Scenario support is a property of a
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

std::set<int> backup_incompatible_scenarios(
const std::vector<EgoState>& backup,
const std::vector<Scenario>& scenarios,
double combined_radius,
int num_discs,
double vehicle_length,
double tolerance = 1e-9)
{
std::set<int> incompatible;

for (const auto& scenario : scenarios) {
    bool scenario_bad = false;

    // k = 0 is the measured current state.
    for (std::size_t k = 1;
         k < backup.size() && !scenario_bad;
         ++k) {

        const auto discs =
            compute_ego_disc_positions(
                backup[k],
                num_discs,
                vehicle_length);

        for (const auto& [obstacle_id, prediction] :
             scenario.trajectories) {

            if (k >= prediction.steps.size()) {
                continue;
            }

            const Eigen::Vector2d obstacle_position =
                prediction.steps[k].mean;

            for (const auto& disc : discs) {
                const double clearance =
                    (disc - obstacle_position).norm()
                    - combined_radius;

                if (clearance < -tolerance) {
                    scenario_bad = true;
                    break;
                }
            }

            if (scenario_bad) {
                break;
            }
        }
    }

    if (scenario_bad) {
        incompatible.insert(scenario.scenario_id);
    }
}

return incompatible;

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
std::vector<ModeObservation> class_observations;
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
        hist.record_observation(
            timestep,
            obstacle_id,       // obstacle that ACTUALLY produced the observation
            observed_mode
        );
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
const auto start_time = std::chrono::high_resolution_clock::now();
++iteration_count_;
const bool allow_nominal = config_.mpc.type == MPCType::SH_MPCC_DRO_FALLBACK;
// Retry from the previous executed plan, not the failed DRO linearization.
const auto previous_reference = allow_nominal ? reference_trajectory_ : std::vector<EgoState>{};
const auto previous_controls = allow_nominal ? last_feasible_controls_ : std::vector<EgoInput>{};
const bool previous_backup = has_feasible_backup_;
auto result = solve_attempt(ego_state, obstacles, goal, reference_velocity,
                            path_progress, path_length, config_.dro.enabled);
if (allow_nominal && !result.success) {
    reference_trajectory_ = previous_reference;
    last_feasible_controls_ = previous_controls;
    has_feasible_backup_ = previous_backup;
    // A fresh draw from the nominal belief; external/Q* weights must not leak in.
    custom_per_obstacle_weights_.clear();
    result = solve_attempt(ego_state, obstacles, goal, reference_velocity,
                           path_progress, path_length, false);
    result.nominal_fallback_attempted = true;
    result.used_nominal_fallback = result.success;
    std::cerr << "[NOMINAL FALLBACK] step=" << iteration_count_
              << " success=" << result.success << std::endl;
}
result.solve_time = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now() - start_time).count();
solve_times_.push_back(result.solve_time);
return result;
}

MPCResult MPCController::solve_attempt(
const EgoState& ego_state,
const std::map<int, ObstacleState>& obstacles,
const Eigen::Vector2d& goal,
double reference_velocity,
double path_progress,
double path_length,
bool use_dro
) {
last_dro_results_.clear();
last_removed_scenario_ids_.clear();

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
if (use_dro && !reference_trajectory_.empty()) {
    const int S = config_.mpc.sampling.num_scenarios;
    const int risk_horizon = config_.mpc.horizon;

    // Compute DRO q* from the nominal p hat
    for (const auto& [obs_id, obs_state] : obstacles) {
        auto hist_it = mode_histories_.find(obs_id);
        if (hist_it == mode_histories_.end()) continue;
        // p_hat: Dirichlet posterior-predictive center of the ambiguity ball.
        auto nominal_weights = compute_mode_weights(
            hist_it->second,
            config_.mpc.sampling.mode_belief
        );

        if (nominal_weights.empty()) {
            continue;
        }

        // Raw realized categorical counts.
        //
        // These are deliberately kept separate from nominal_weights:
        //   - observed_counts construct the finite-sample confidence region;
        //   - nominal_weights are the center of the Wasserstein ball.
        const auto observed_counts =
            hist_it->second.get_mode_counts();

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

        last_dro_results_[obs_id] =
            dro_.compute_worst_case_weights(
                nominal_weights,
                observed_counts, 
                obs_state,
                hist_it->second.available_modes,
                reference_trajectory_,
                config_.mpc.horizon,
                config_.mpc.ego.radius,
                config_.obstacle_radius,
                config_.mpc.constraints.safety_margin,
                risk_horizon,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length,
                transition_ptr
            );

// ============================================================
// DRO DIAGNOSTICS
// ============================================================
const auto& dro_result =
    last_dro_results_.at(obs_id);

double nominal_risk = 0.0;
double qstar_risk = 0.0;
double l1_distance = 0.0;

for (const auto& [mode_id, p] : nominal_weights) {

    const auto risk_it =
        dro_result.risk_per_mode.find(mode_id);

    const auto q_it =
        dro_result.worst_case_weights.find(mode_id);

    const double risk =
        risk_it != dro_result.risk_per_mode.end()
            ? risk_it->second
            : 0.0;

    const double q =
        q_it != dro_result.worst_case_weights.end()
            ? q_it->second
            : 0.0;

    nominal_risk += p * risk;
    qstar_risk += q * risk;
    l1_distance += std::abs(q - p);
}

const double tv_distance =
    0.5 * l1_distance;

const double budget_usage =
    dro_result.rho_used > 1e-12
        ? dro_result.implied_transport_cost /
              dro_result.rho_used
        : 0.0;

std::cerr
    << "[DRO SUMMARY]"
    << " step=" << iteration_count_
    << " obstacle=" << obs_id
    << " S=" << S
    << " rho=" << dro_result.rho_used
    << " rho_raw=" << dro_result.rho_before_clamp
    << " D_max=" << dro_result.transport_diameter
    << " D_min=" << dro_result.transport_min_offdiag
    << " tv_bound=" << dro_result.max_tv_from_radius
    << " eta=" << dro_result.normalized_rho_used
    << " eta_raw=" << dro_result.normalized_rho_before_clamp
    << " radius_m=" << dro_result.radius_observation_count
    << " radius_d=" << dro_result.radius_mode_count
    << " transport_cost="
    << dro_result.implied_transport_cost
    << " budget_usage=" << budget_usage
    << " lambda=" << dro_result.optimal_lambda
    << " q_support=" << dro_result.qstar_support_size
    << " nominal_risk=" << nominal_risk
    << " qstar_risk=" << qstar_risk
    << " risk_lift="
    << qstar_risk - nominal_risk
    << " tv=" << tv_distance
    << std::endl;

for (const auto& [mode_id, p] : nominal_weights) {

    const auto risk_it =
        dro_result.risk_per_mode.find(mode_id);

    const auto q_it =
        dro_result.worst_case_weights.find(mode_id);

    const double risk =
        risk_it != dro_result.risk_per_mode.end()
            ? risk_it->second
            : 0.0;

    const double q =
        q_it != dro_result.worst_case_weights.end()
            ? q_it->second
            : 0.0;

    std::cerr
        << "[DRO MODE]"
        << " step=" << iteration_count_
        << " obstacle=" << obs_id
        << " mode=" << mode_id
        << " p=" << p
        << " risk=" << risk
        << " q=" << q
        << " delta_q=" << q - p
        << " q_times_S="
        << q * static_cast<double>(S)
        << std::endl;
}
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

// ============================================================
// ACTUAL SCENARIO-SAMPLING DIAGNOSTICS
// ============================================================
if (use_dro) {

    for (const auto& [obs_id, dro_result] :
         last_dro_results_) {

        std::map<std::string, int>
            first_mode_counts;

        std::map<std::string, int>
            occupancy_counts;

        int trajectory_count = 0;
        int occupancy_total = 0;

        for (const auto& scenario : scenarios_) {

            const auto traj_it =
                scenario.trajectories.find(obs_id);

            if (traj_it ==
                scenario.trajectories.end()) {
                continue;
            }

            const auto& trajectory =
                traj_it->second;

            /*
             * Markov rollout:
             * sampled_mode_sequence contains one mode per
             * prediction stage.
             */
            if (!trajectory.sampled_mode_sequence.empty()) {

                ++trajectory_count;

                ++first_mode_counts[
                    trajectory.sampled_mode_sequence.front()
                ];

                for (const auto& mode_id :
                     trajectory.sampled_mode_sequence) {

                    ++occupancy_counts[mode_id];
                    ++occupancy_total;
                }
            }

            /*
             * Non-Markov / held-mode rollout:
             *
             * sampled_mode_sequence is empty, so use the
             * trajectory's representative mode directly.
             *
             * This is the part your previous logger was missing.
             */
            else {

                ++trajectory_count;

                ++first_mode_counts[
                    trajectory.mode_id
                ];

                occupancy_counts[
                    trajectory.mode_id
                ] += config_.mpc.horizon;

                occupancy_total +=
                    config_.mpc.horizon;
            }
        }

        for (const auto& [mode_id, q] :
             dro_result.worst_case_weights) {

            const int first_count =
                first_mode_counts.count(mode_id)
                    ? first_mode_counts.at(mode_id)
                    : 0;

            const int occupancy =
                occupancy_counts.count(mode_id)
                    ? occupancy_counts.at(mode_id)
                    : 0;

            const double first_fraction =
                trajectory_count > 0
                    ? static_cast<double>(
                          first_count) /
                          static_cast<double>(
                              trajectory_count)
                    : 0.0;

            const double occupancy_fraction =
                occupancy_total > 0
                    ? static_cast<double>(
                          occupancy) /
                          static_cast<double>(
                              occupancy_total)
                    : 0.0;

            std::cerr
                << "[DRO SAMPLE]"
                << " step=" << iteration_count_
                << " obstacle=" << obs_id
                << " mode=" << mode_id
                << " q=" << q
                << " q_times_S="
                << q *
                   static_cast<double>(S_actual)
                << " first_count="
                << first_count
                << " first_fraction="
                << first_fraction
                << " occupancy="
                << occupancy
                << " occupancy_fraction="
                << occupancy_fraction
                << std::endl;
        }
    }
}

// ------------------------------------------------------------------------
// Construct shifted dynamically feasible candidate from previous solution.
// ------------------------------------------------------------------------

std::vector<EgoInput> backup_inputs;
std::vector<EgoState> backup_trajectory;
bool backup_available = false;
FailureDiagnostics failure_diagnostics;

const int N = config_.mpc.horizon;

if (has_feasible_backup_ &&
    static_cast<int>(last_feasible_controls_.size()) == N) {

    backup_inputs.reserve(N);

    // Shift u_1,...,u_{N-1}.
    for (int k = 1; k < N; ++k) {
        backup_inputs.push_back(
            last_feasible_controls_[k]);
    }

    /*
    * One-step terminal extension.
    *
    * This is the part that eventually corresponds to kappa_f in
    * the recursive-feasibility proof.
    */
    EgoInput terminal_input =
        last_feasible_controls_.back();

    backup_inputs.push_back(terminal_input);

    // Re-roll from the ACTUAL measured current state.
    if (reference_path_.has_value()) {
        backup_trajectory =
            ego_dynamics_.rollout_with_spline(
                ego_with_spline,
                backup_inputs,
                *reference_path_);
    } else {
        backup_trajectory =
            ego_dynamics_.rollout(
                ego_with_spline,
                backup_inputs);
    }

    backup_available =
        static_cast<int>(backup_trajectory.size()) == N + 1 &&
        trajectory_is_deterministically_admissible(
            backup_trajectory,
            backup_inputs);

    std::cerr
        << "[RF BACKUP]"
        << " available=" << backup_available
        << " controls=" << backup_inputs.size()
        << " states=" << backup_trajectory.size()
        << std::endl;
}

// ------------------------------------------------------------------------
// Recovery seed.
//
// Later solves use the shifted previously-feasible MPC solution.
// The first solve has no such backup, so construct a dynamically-consistent
// cold-start control sequence that can be used by the SAME homotopy recovery
// machinery.
// ------------------------------------------------------------------------

std::vector<EgoInput> recovery_seed_inputs;
std::vector<EgoState> recovery_seed_trajectory;
bool recovery_seed_available = false;

if (backup_available) {

    recovery_seed_inputs = backup_inputs;
    recovery_seed_trajectory = backup_trajectory;
    recovery_seed_available = true;

} else if (config_.mpc.uses_safe_horizon()) {

    /*
     * Build the same kind of dynamically-consistent initial trajectory
     * that solve_optimization_sqp() constructs internally.
     *
     * Do NOT use reference_trajectory_ directly as a dynamics seed:
     * generate_straight_line_trajectory() is geometric. Infer controls
     * from it and then re-roll the true dynamics.
     */
    auto cold_geometry =
        generate_straight_line_trajectory(
            ego_with_spline,
            goal,
            reference_velocity);

    recovery_seed_inputs.reserve(N);

    for (int k = 0; k < N; ++k) {

        double a =
            (cold_geometry[k + 1].v -
             cold_geometry[k].v) /
            config_.mpc.dt;

        double dtheta =
            cold_geometry[k + 1].theta -
            cold_geometry[k].theta;

        while (dtheta > M_PI)
            dtheta -= 2.0 * M_PI;

        while (dtheta < -M_PI)
            dtheta += 2.0 * M_PI;

        double omega =
            dtheta /
            config_.mpc.dt;

        a =
            std::clamp(
                a,
                config_.mpc.ego.dynamics.min_acceleration,
                config_.mpc.ego.dynamics.max_acceleration);

        omega =
            std::clamp(
                omega,
                -config_.mpc.ego.dynamics.max_omega,
                 config_.mpc.ego.dynamics.max_omega);

        recovery_seed_inputs.emplace_back(
            a,
            omega);
    }

    if (reference_path_.has_value()) {

        recovery_seed_trajectory =
            ego_dynamics_.rollout_with_spline(
                ego_with_spline,
                recovery_seed_inputs,
                *reference_path_);

    } else {

        recovery_seed_trajectory =
            ego_dynamics_.rollout(
                ego_with_spline,
                recovery_seed_inputs);
    }

    recovery_seed_available =
        static_cast<int>(
            recovery_seed_trajectory.size()) == N + 1 &&
        trajectory_is_deterministically_admissible(
            recovery_seed_trajectory,
            recovery_seed_inputs);

    /*
     * If the straight/path-following cold start is not even
     * deterministically admissible, try the braking rollout as a seed.
     */
    if (!recovery_seed_available) {

        auto braking_seed =
            generate_safe_fallback(
                ego_with_spline);

        if (static_cast<int>(
                braking_seed.control_inputs.size()) == N &&
            static_cast<int>(
                braking_seed.ego_trajectory.size()) == N + 1 &&
            trajectory_is_deterministically_admissible(
                braking_seed.ego_trajectory,
                braking_seed.control_inputs)) {

            recovery_seed_inputs =
                braking_seed.control_inputs;

            recovery_seed_trajectory =
                braking_seed.ego_trajectory;

            recovery_seed_available = true;
        }
    }

    std::cerr
        << "[COLD START SEED]"
        << " available="
        << recovery_seed_available
        << " controls="
        << recovery_seed_inputs.size()
        << " states="
        << recovery_seed_trajectory.size()
        << std::endl;
}


// Step 4: Fixed collision normals from the numerical linearization trajectory.
// Normals are held constant for the subsequent QP (Case B also linearizes
// heading-dependent disc centers about the same numerical reference).
auto constraint_start = std::chrono::high_resolution_clock::now();

// de Groot 2023 Definition-2 geometric dominance pruning: drop scenarios whose
// collision half-spaces are IMPLIED (on the reachable ball) by a more-restrictive
// scenario's, evaluated at the reference trajectory.

const double combined_radius =
    config_.combined_radius();

std::set<int> removed_scenarios;

std::vector<Scenario> constraint_scenarios =
    scenarios_;

failure_diagnostics.backup_available = backup_available;

// Default behavior for first solve / non-SH controller.
auto dynamic_reference =
    reference_trajectory_;

auto constraint_reference =
    dynamic_reference;

std::vector<EgoInput> feasible_warmstart_inputs;

bool use_multi_homotopy_recovery = false;

if (config_.mpc.uses_safe_horizon() &&
backup_available) {

removed_scenarios =
    backup_incompatible_scenarios(
        backup_trajectory,
        scenarios_,
        combined_radius,
        config_.mpc.ego.num_discs,
        config_.mpc.ego.length);

const int removal_count =
    static_cast<int>(
        removed_scenarios.size());

const int removal_limit =
    config_.scenario_removal_budget();

failure_diagnostics.backup_removal_budget_exceeded = removal_count > removal_limit;

if (removal_count <= removal_limit) {

    // ============================================================
    // Case 1:
    // Shifted previous solution can be preserved within the
    // permitted scenario-removal budget.
    // ============================================================

    constraint_scenarios.erase(
        std::remove_if(
            constraint_scenarios.begin(),
            constraint_scenarios.end(),
            [&](const Scenario& scenario) {
                return removed_scenarios.find(
                           scenario.scenario_id)
                       != removed_scenarios.end();
            }),
        constraint_scenarios.end());

    last_removed_scenario_ids_ =
        removed_scenarios;

    /*
     * The shifted trajectory itself is dynamically feasible,
     * and after removal it is collision-feasible.
     */
    dynamic_reference =
        backup_trajectory;

    constraint_reference =
        backup_trajectory;

    feasible_warmstart_inputs =
        backup_inputs;

    const double removal_fraction =
        scenarios_.empty()
            ? 0.0
            : static_cast<double>(
                  removal_count) /
              static_cast<double>(
                  scenarios_.size());

    std::cerr
        << "[RF REMOVAL ACCEPTED]"
        << " sampled=" << scenarios_.size()
        << " removed=" << removal_count
        << " retained="
        << constraint_scenarios.size()
        << " fraction="
        << removal_fraction
        << " limit="
        << removal_limit
        << std::endl;
}
else {

    std::cerr
        << "[RF REMOVAL REJECTED]"
        << " sampled=" << scenarios_.size()
        << " required=" << removal_count
        << " limit=" << removal_limit
        << std::endl;

    /*
    * Removal budget exceeded.
    *
    * Keep every sampled scenario. We will construct several
    * candidate collision homotopies after this branch and try
    * dynamics-aware DR restoration for each.
    */
    removed_scenarios.clear();
    last_removed_scenario_ids_.clear();

    constraint_scenarios =
        scenarios_;

    feasible_warmstart_inputs.clear();

    /*
    * The shifted previous solution remains the dynamically
    * admissible seed from which every homotopy is generated.
    */
    dynamic_reference =
        backup_trajectory;

    constraint_reference =
        backup_trajectory;

    use_multi_homotopy_recovery = true;

} } // End SH with shifted backup 

else if (config_.mpc.uses_safe_horizon()) {

    // ================================================================
    // First MPC solve:
    //
    // Always let the ordinary Safe-Horizon SQP try first.
    // The dynamically-feasible cold-start seed is reserved for
    // second-stage homotopy recovery if this ordinary solve fails.
    // ================================================================

    const bool anchor_ok =
        prepare_safe_horizon_anchors(
            constraint_reference,
            scenarios_,
            combined_radius,
            config_.mpc.ego.num_discs,
            config_.mpc.ego.length);

    if (!anchor_ok) {
        if (config_.mpc.type == MPCType::SH_MPCC_DRO_FALLBACK) {
            use_multi_homotopy_recovery = true;
        } else {
            throw std::runtime_error(
                "Safe-Horizon first-solve anchor projection failed.");
        }
    }

    std::cerr
        << "[COLD START ORDINARY SQP]"
        << " recovery_seed_available="
        << recovery_seed_available
        << " homotopy_deferred=1"
        << std::endl;
}

const double max_abs_velocity =
    std::max(
        std::abs(config_.mpc.ego.dynamics.min_velocity),
        std::abs(config_.mpc.ego.dynamics.max_velocity));

std::vector<CollisionConstraint> constraints;

auto build_homotopy_recovery_problem = [&]() -> bool {

    if (!recovery_seed_available) {
        if (config_.mpc.type == MPCType::SH_MPCC_DRO_FALLBACK) return false;
        throw std::runtime_error(
            "Homotopy recovery requested without "
            "a dynamically feasible recovery seed.");
    }

    const ConflictInfo conflict =
        find_backup_conflict_window(
            recovery_seed_trajectory,
            constraint_scenarios,
            combined_radius);

    const int first_bad =
        conflict.first_k;

    const int last_bad =
        conflict.last_k;

    const int conflict_obstacle_id =
        conflict.obstacle_id;

    std::cerr
        << "[HOMOTOPY CONFLICT]"
        << " obstacle=" << conflict_obstacle_id
        << " first=" << first_bad
        << " last=" << last_bad
        << std::endl;

    /*
    * AUTO is evaluated too, but it is only a fallback.
    *
    * PATH / LEFT / RIGHT are the explicit homotopies that are
    * allowed to compete for normal selection.
    */
    const std::array<HomotopySide, 4> homotopies = {
        HomotopySide::Auto,
        HomotopySide::Path,
        HomotopySide::Left,
        HomotopySide::Right
    };

    HomotopyRecoveryCandidate best;
    HomotopyRecoveryCandidate auto_recovery;

    /*
    * Keep the AUTO polytope as a last-resort ordinary-SQP
    * fallback if every DR attempt fails.
    */
    std::vector<CollisionConstraint>
        auto_constraints;

    std::vector<EgoState>
        auto_anchor;

    bool auto_polytope_available = false;

    /*

Evaluate braking immediately when the removal budget is exceeded.



Do this BEFORE committing to a passing homotopy.  The braking

candidate remains available when LEFT / RIGHT become expensive.
*/
std::vector<EgoInput> braking_inputs =
make_braking_seed_inputs(
ego_with_spline,
recovery_seed_inputs,
recovery_seed_trajectory);

std::vector<EgoState> braking_trajectory;

if (reference_path_.has_value()) {

braking_trajectory =
    ego_dynamics_.rollout_with_spline(
        ego_with_spline,
        braking_inputs,
        *reference_path_);

}
else {

braking_trajectory =
    ego_dynamics_.rollout(
        ego_with_spline,
        braking_inputs);

}

const bool braking_deterministic_ok =
trajectory_is_deterministically_admissible(
braking_trajectory,
braking_inputs);

const auto braking_incompatible =
backup_incompatible_scenarios(
braking_trajectory,
constraint_scenarios,
combined_radius,
config_.mpc.ego.num_discs,
config_.mpc.ego.length);

failure_diagnostics.braking_collision_feasible = braking_incompatible.empty();
failure_diagnostics.any_homotopy_geometrically_feasible = 0;

const bool direct_braking_available =
braking_deterministic_ok &&
braking_incompatible.empty();

std::cerr
<< "[BRAKE CANDIDATE]"
<< " feasible="
<< direct_braking_available
<< " incompatible="
<< braking_incompatible.size()
<< " v0="
<< ego_with_spline.v
<< " terminal_v="
<< braking_trajectory.back().v
<< std::endl;

    for (const HomotopySide side : homotopies) {

        const char* label = "unknown";

        switch (side) {
            case HomotopySide::Auto:
                label = "auto";
                break;

            case HomotopySide::Path:
                label = "path";
                break;

            case HomotopySide::Left:
                label = "left";
                break;

            case HomotopySide::Right:
                label = "right";
                break;
        }

        std::cerr
            << "[HOMOTOPY TRY]"
            << " side=" << label
            << " first_conflict=" << first_bad
            << " last_conflict=" << last_bad
            << std::endl;

        std::vector<EgoState> candidate_anchor =
        make_homotopy_seed(
            recovery_seed_trajectory,
            conflict_obstacle_id,
            side,
            first_bad,
            last_bad,
            combined_radius);

        /*
        * PATH has a strict meaning: follow the reference-path
        * homotopy. Do not let the generic geometric projection
        * move it onto another side of an obstacle.
        */
        if (side == HomotopySide::Path) {

            const auto path_incompatible =
                backup_incompatible_scenarios(
                    candidate_anchor,
                    constraint_scenarios,
                    combined_radius,
                    config_.mpc.ego.num_discs,
                    config_.mpc.ego.length);

            if (!path_incompatible.empty()) {

                std::cerr
                    << "[HOMOTOPY RESULT]"
                    << " side=path"
                    << " path_blocked=1"
                    << " incompatible="
                    << path_incompatible.size()
                    << std::endl;

                continue;
            }
        }

    /*
    * AUTO / LEFT / RIGHT may use geometric repair.
    * PATH does not.
    */
    bool anchor_ok = true;

    if (side != HomotopySide::Path) {

        anchor_ok =
            prepare_safe_horizon_anchors(
                candidate_anchor,
                constraint_scenarios,
                combined_radius,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length);
    }

    if (!anchor_ok) {

        std::cerr
            << "[HOMOTOPY RESULT]"
            << " side=" << label
            << " anchor_ok=0"
            << std::endl;

        continue;
    }

        failure_diagnostics.any_homotopy_geometrically_feasible = 1;

        auto candidate_raw =
            compute_linearized_constraints(
                candidate_anchor,
                constraint_scenarios,
                config_.mpc.ego.radius,
                config_.obstacle_radius,
                config_.mpc.constraints.safety_margin,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length);

        auto candidate_constraints =
            reduce_to_free_space_polytopes(
                candidate_raw,
                candidate_anchor,
                recovery_seed_trajectory,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length,
                max_abs_velocity,
                config_.mpc.dt,
                20);

        /*
        * Preserve AUTO so ordinary SQP can still be attempted if
        * none of the homotopies can be DR-restored.
        */
        if (side == HomotopySide::Auto) {
            auto_anchor =
                candidate_anchor;

            auto_constraints =
                candidate_constraints;

            auto_polytope_available = true;
        }

        std::vector<EgoInput>
            candidate_inputs;

        std::vector<EgoState>
            candidate_trajectory;

        std::cerr
            << "[HOMOTOPY DR START]"
            << " side=" << label
            << " constraints="
            << candidate_constraints.size()
            << std::endl;

        const bool restored =
            restore_dynamically_feasible_plan(
                ego_with_spline,
                goal,
                reference_velocity,
                candidate_constraints,
                recovery_seed_inputs,
                candidate_inputs,
                candidate_trajectory,
                path_progress,
                path_length);

        if (!restored) {
            std::cerr
                << "[HOMOTOPY RESULT]"
                << " side=" << label
                << " restored=0"
                << std::endl;

            continue;
        }

        const auto [violation, violated] =
    evaluate_constraint_violation(
        candidate_constraints,
        candidate_trajectory);

    const bool deterministic_ok =
        trajectory_is_deterministically_admissible(
            candidate_trajectory,
            candidate_inputs);

    /*
    * The reduced fixed-normal polytope is only the optimization
    * representation. Verify the restored nonlinear trajectory
    * against the ORIGINAL sampled obstacle geometry as well.
    */
    const auto exact_incompatible =
        backup_incompatible_scenarios(
            candidate_trajectory,
            constraint_scenarios,
            combined_radius,
            config_.mpc.ego.num_discs,
            config_.mpc.ego.length);

    if (violation > 1e-4 ||
        !deterministic_ok ||
        !exact_incompatible.empty()) {

        std::cerr
            << "[HOMOTOPY RESULT]"
            << " side=" << label
            << " restored=1"
            << " verified=0"
            << " violation="
            << violation
            << " exact_incompatible="
            << exact_incompatible.size()
            << std::endl;

        continue;
    }

        const double score =
            score_homotopy_recovery(
                candidate_trajectory,
                candidate_inputs,
                recovery_seed_inputs);

        std::cerr
            << "[HOMOTOPY RESULT]"
            << " side=" << label
            << " restored=1"
            << " verified=1"
            << " score=" << score
            << std::endl;

        /*
        * AUTO is useful as a recovery fallback, but do not let it
        * compete against the explicit PATH / LEFT / RIGHT choices.
        */
        if (side == HomotopySide::Auto) {

            auto_recovery.success = true;
            auto_recovery.side = side;
            auto_recovery.anchor =
                std::move(candidate_anchor);
            auto_recovery.constraints =
                std::move(candidate_constraints);
            auto_recovery.inputs =
                std::move(candidate_inputs);
            auto_recovery.trajectory =
                std::move(candidate_trajectory);
            auto_recovery.score = score;

            continue;
        }

        if (!best.success ||
            score < best.score) {

            best.success = true;
            best.side = side;
            best.anchor =
                std::move(candidate_anchor);
            best.constraints =
                std::move(candidate_constraints);
            best.inputs =
                std::move(candidate_inputs);
            best.trajectory =
                std::move(candidate_trajectory);
            best.score = score;
        }
    }

    /*

Temporary diagnostic threshold.



A large homotopy score means the passing maneuver is becoming

increasingly expensive / far from the nominal path.  If stopping

is still feasible, yield before becoming committed to that pass.
*/
constexpr double kYieldScoreThreshold = 50.0;

const bool prefer_braking =
direct_braking_available &&
(!best.success ||
best.score > kYieldScoreThreshold);

std::cerr
<< "[RECOVERY DECISION]"
<< " brake_available="
<< direct_braking_available
<< " homotopy_available="
<< best.success
<< " homotopy_score="
<< (best.success
? best.score
: std::numeric_limits<double>::infinity())
<< " prefer_braking="
<< prefer_braking
<< std::endl;

    if (prefer_braking) {

/*
 * The exact braking rollout is already dynamically feasible and
 * satisfies every fresh sampled scenario.  Build the fixed
 * collision geometry around this witness.
 */
auto braking_raw =
    compute_linearized_constraints(
        braking_trajectory,
        constraint_scenarios,
        config_.mpc.ego.radius,
        config_.obstacle_radius,
        config_.mpc.constraints.safety_margin,
        config_.mpc.ego.num_discs,
        config_.mpc.ego.length);

auto braking_constraints =
    reduce_to_free_space_polytopes(
        braking_raw,
        braking_trajectory,
        braking_trajectory,
        config_.mpc.ego.num_discs,
        config_.mpc.ego.length,
        max_abs_velocity,
        config_.mpc.dt,
        20);

constraint_reference =
    braking_trajectory;

dynamic_reference =
    braking_trajectory;

constraints =
    std::move(braking_constraints);

feasible_warmstart_inputs =
    braking_inputs;

std::cerr
    << "[RECOVERY SELECT]"
    << " mode=brake"
    << " homotopy_score="
    << (best.success
            ? best.score
            : std::numeric_limits<double>::infinity())
    << " terminal_v="
    << dynamic_reference.back().v
    << std::endl;

}
else if (best.success) {

        constraint_reference =
            std::move(best.anchor);

        constraints =
            std::move(best.constraints);

        feasible_warmstart_inputs =
            std::move(best.inputs);

        dynamic_reference =
            std::move(best.trajectory);

        const char* label = "unknown";

        switch (best.side) {
            case HomotopySide::Auto:
                label = "auto";
                break;

            case HomotopySide::Path:
                label = "path";
                break;

            case HomotopySide::Left:
                label = "left";
                break;

            case HomotopySide::Right:
                label = "right";
                break;
        }

        std::cerr
            << "[HOMOTOPY SELECT]"
            << " side=" << label
            << " score=" << best.score
            << std::endl;
    }

    else if (auto_recovery.success) {

        /*
        * None of PATH / LEFT / RIGHT could be dynamically restored.
        * Use the successfully restored AUTO solution only now.
        */
        constraint_reference =
            std::move(auto_recovery.anchor);

        constraints =
            std::move(auto_recovery.constraints);

        feasible_warmstart_inputs =
            std::move(auto_recovery.inputs);

        dynamic_reference =
            std::move(auto_recovery.trajectory);

        std::cerr
            << "[HOMOTOPY SELECT]"
            << " side=auto_fallback"
            << " score=" << auto_recovery.score
            << std::endl;
    }
    else {

std::cerr
    << "[HOMOTOPY SELECT]"
    << " no_dynamically_feasible_candidate=1"
    << std::endl;

bool braking_selected = false;

std::cerr
    << "[BRAKE DR RECOVERY START]"
    << " deterministic_ok="
    << braking_deterministic_ok
    << " direct_incompatible="
    << braking_incompatible.size()
    << " v0="
    << ego_with_spline.v
    << " terminal_v="
    << braking_trajectory.back().v
    << std::endl;

/*
 * Direct braking was already evaluated before the homotopy
 * selection. If we are here, pure braking was not selected.
 *
 * Try to combine braking with a small geometric avoidance
 * maneuver and restore dynamic feasibility with DR.
 */
if (braking_deterministic_ok) {

    std::vector<EgoState> braking_anchor =
        braking_trajectory;

    const bool braking_anchor_ok =
        prepare_safe_horizon_anchors(
            braking_anchor,
            constraint_scenarios,
            combined_radius,
            config_.mpc.ego.num_discs,
            config_.mpc.ego.length);

    std::cerr
        << "[BRAKE ANCHOR]"
        << " ok="
        << braking_anchor_ok
        << std::endl;

    if (braking_anchor_ok) {

        auto braking_raw =
            compute_linearized_constraints(
                braking_anchor,
                constraint_scenarios,
                config_.mpc.ego.radius,
                config_.obstacle_radius,
                config_.mpc.constraints.safety_margin,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length);

        auto braking_constraints =
            reduce_to_free_space_polytopes(
                braking_raw,
                braking_anchor,
                braking_trajectory,
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length,
                max_abs_velocity,
                config_.mpc.dt,
                20);

        std::vector<EgoInput>
            restored_braking_inputs;

        std::vector<EgoState>
            restored_braking_trajectory;

        std::cerr
            << "[BRAKE DR START]"
            << " constraints="
            << braking_constraints.size()
            << std::endl;

        const bool braking_restored =
            restore_dynamically_feasible_plan(
                ego_with_spline,
                goal,
                reference_velocity,
                braking_constraints,
                braking_inputs,
                restored_braking_inputs,
                restored_braking_trajectory,
                path_progress,
                path_length);

        std::cerr
            << "[BRAKE DR RESULT]"
            << " restored="
            << braking_restored
            << std::endl;

        if (braking_restored) {

            /*
             * Verify the restored trajectory against the ORIGINAL
             * sampled scenarios, not only the reduced halfspaces.
             */
            const auto exact_incompatible =
                backup_incompatible_scenarios(
                    restored_braking_trajectory,
                    constraint_scenarios,
                    combined_radius,
                    config_.mpc.ego.num_discs,
                    config_.mpc.ego.length);

            const bool exact_deterministic_ok =
                trajectory_is_deterministically_admissible(
                    restored_braking_trajectory,
                    restored_braking_inputs);

            std::cerr
                << "[BRAKE DR VERIFY]"
                << " deterministic_ok="
                << exact_deterministic_ok
                << " incompatible="
                << exact_incompatible.size()
                << std::endl;

            if (exact_deterministic_ok &&
                exact_incompatible.empty()) {

                constraint_reference =
                    std::move(braking_anchor);

                constraints =
                    std::move(braking_constraints);

                feasible_warmstart_inputs =
                    std::move(restored_braking_inputs);

                dynamic_reference =
                    std::move(restored_braking_trajectory);

                braking_selected = true;

                std::cerr
                    << "[BRAKE RECOVERY SELECT]"
                    << " mode=dr"
                    << " terminal_v="
                    << dynamic_reference.back().v
                    << std::endl;
            }
        }
    }
}

/*
 * Brake+DR also failed. Only now use the old
 * last-chance AUTO-polytope SQP.
 */
if (!braking_selected) {

    std::cerr
        << "[BRAKE RECOVERY RESULT]"
        << " success=0"
        << std::endl;

    if (auto_polytope_available) {

        constraint_reference =
            std::move(auto_anchor);

        constraints =
            std::move(auto_constraints);

        feasible_warmstart_inputs.clear();
    }
        else {

        std::cerr
            << "[HOMOTOPY RECOVERY BUILD]"
            << " success=0"
            << " reason=no_valid_auto_polytope"
            << std::endl;

        return false;
    }
    }
}  // end: no best/auto recovery branch

/*
 * Reaching this point means a recovery optimization problem
 * has successfully been constructed in `constraints`.
 */
return true;

};  // build_homotopy_recovery_problem


auto build_ordinary_constraint_problem = [&]() {

 /*
 * Existing behavior for:
 *
 *   - first solve
 *   - accepted shifted-backup removal
 *   - non-SH controller
 */
 auto all_constraints =
     compute_linearized_constraints(
         constraint_reference,
         constraint_scenarios,
         config_.mpc.ego.radius,
         config_.obstacle_radius,
         config_.mpc.constraints.safety_margin,
         config_.mpc.ego.num_discs,
         config_.mpc.ego.length);

 constraints =
     reduce_to_free_space_polytopes(
         all_constraints,
         constraint_reference,
         dynamic_reference,
         config_.mpc.ego.num_discs,
         config_.mpc.ego.length,
         max_abs_velocity,
         config_.mpc.dt,
         20);

};  // build_ordinary_constraint_problem

if (use_multi_homotopy_recovery) {

    /*
     * Existing recursive-feasibility recovery path:
     * the shifted backup exceeded the removal budget.
     */
    const bool recovery_problem_built =
        build_homotopy_recovery_problem();

    if (!recovery_problem_built) {
        if (config_.mpc.type == MPCType::SH_MPCC_DRO_FALLBACK) {
            // No admissible recovery problem: let solve() advance to nominal,
            // or retain terminal rejection if this was already nominal.
            auto failed = generate_safe_fallback(ego_with_spline);
            failed.failure_diagnostics = failure_diagnostics;
            failed.sampled_scenarios = S_actual;
            failed.required_scenarios = support_certification_enabled ? S_required : -1;
            failed.sample_count_sufficient = sample_count_sufficient;
            failed.certificate_status = support_certification_enabled
                ? SafeHorizonCertificateStatus::PLAN_INFEASIBLE
                : SafeHorizonCertificateStatus::NOT_REQUESTED;
            return failed;
        }
        throw std::runtime_error(
            "Recursive-feasibility homotopy recovery failed "
            "to construct a valid recovery problem.");
    }

} else {

    /*
     * Normal path:
     * first solve, accepted backup removal, or non-SH controller.
     */
    build_ordinary_constraint_problem();
}

// Safe Horizon refers to support-bounded joint-risk
// certification across the complete MPC horizon.  Do not erase late
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
MPCResult result =
solve_optimization_sqp(
ego_with_spline,
goal,
reference_velocity,
constraints,
removed_scenarios,
feasible_warmstart_inputs,
path_progress,
path_length);
// ------------------------------------------------------------------------
// Cold-start second-stage homotopy recovery.
//
// Important:
//   1. The ordinary Safe-Horizon SQP has already been attempted.
//   2. Homotopy recovery is only entered if that ordinary solve did
//      not produce a normal feasible SQP solution.
//   3. If homotopy recovery itself fails, preserve the original result.
// ------------------------------------------------------------------------

const bool ordinary_sqp_failed =
    !result.success ||
    !result.sampled_constraints_satisfied ||
    result.used_fallback;

/*
 * Cold-start homotopy recovery is meaningful only if the
 * dynamically-consistent recovery seed actually conflicts
 * with at least one fresh sampled obstacle scenario.
 *
 * This prevents unrelated SQP failures (road constraints,
 * velocity bounds, numerical failure, etc.) from incorrectly
 * invoking LEFT / RIGHT obstacle homotopies.
 */
std::set<int> recovery_seed_incompatible;

if (config_.mpc.uses_safe_horizon() &&
    !backup_available &&
    recovery_seed_available) {

    recovery_seed_incompatible =
        backup_incompatible_scenarios(
            recovery_seed_trajectory,
            scenarios_,
            combined_radius,
            config_.mpc.ego.num_discs,
            config_.mpc.ego.length);
}

const bool recovery_seed_has_conflict =
    !recovery_seed_incompatible.empty();

const bool try_cold_start_homotopy =
    config_.mpc.uses_safe_horizon() &&
    !backup_available &&
    recovery_seed_available &&
    recovery_seed_has_conflict &&
    !use_multi_homotopy_recovery &&
    ordinary_sqp_failed;

std::cerr
    << "[COLD START RECOVERY CHECK]"
    << " ordinary_sqp_failed="
    << ordinary_sqp_failed
    << " seed_available="
    << recovery_seed_available
    << " incompatible="
    << recovery_seed_incompatible.size()
    << " try_homotopy="
    << try_cold_start_homotopy
    << std::endl;

if (try_cold_start_homotopy) {

    std::cerr
        << "[COLD START SQP FAILED]"
        << " success="
        << result.success
        << " sampled_constraints_satisfied="
        << result.sampled_constraints_satisfied
        << " used_fallback="
        << result.used_fallback
        << " trying_homotopy=1"
        << std::endl;

    /*
     * Preserve the complete result from the ordinary first solve.
     *
     * If homotopy construction or the second SQP fails, this result
     * remains the executable result.
     */
    MPCResult ordinary_result =
        result;

    const auto ordinary_constraints =
        constraints;

    /*
     * First-solve recovery uses the complete fresh sampled set.
     * There is no recursive-feasibility scenario removal here.
     */
    removed_scenarios.clear();
    last_removed_scenario_ids_.clear();

    constraint_scenarios =
        scenarios_;

    /*
     * Homotopy generation starts from the dynamically-consistent
     * cold-start seed.
     */
    dynamic_reference =
        recovery_seed_trajectory;

    constraint_reference =
        recovery_seed_trajectory;

    feasible_warmstart_inputs.clear();
    constraints.clear();

    /*
     * Discrete recovery selection is now actually being attempted.
     */
    use_multi_homotopy_recovery = true;

    const bool recovery_problem_built =
        build_homotopy_recovery_problem();

    if (!recovery_problem_built) {

        std::cerr
            << "[COLD START HOMOTOPY RETRY]"
            << " recovery_problem_built=0"
            << " keeping_ordinary_result=1"
            << std::endl;

        /*
         * Recovery construction failed before there was even a
         * second SQP problem worth solving.
         */
        result =
            std::move(ordinary_result);

        constraints =
            ordinary_constraints;

        use_multi_homotopy_recovery =
            false;

        if (capture_linearized_constraints_) {
            last_linearized_constraints_ =
                constraints;
        }
    }
    else {

        if (capture_linearized_constraints_) {
            last_linearized_constraints_ =
                constraints;
        }

        std::cerr
            << "[COLD START HOMOTOPY RETRY]"
            << " constraints="
            << constraints.size()
            << " warmstart_controls="
            << feasible_warmstart_inputs.size()
            << std::endl;

        MPCResult recovery_result =
            solve_optimization_sqp(
                ego_with_spline,
                goal,
                reference_velocity,
                constraints,
                removed_scenarios,
                feasible_warmstart_inputs,
                path_progress,
                path_length);

        /*
         * Only accept the homotopy retry if the SQP itself produced
         * a feasible result.
         *
         * A fallback returned by the second solve does not replace
         * the preserved ordinary result.
         */
        const bool homotopy_sqp_succeeded =
            recovery_result.success &&
            recovery_result.sampled_constraints_satisfied &&
            !recovery_result.used_fallback;

        if (homotopy_sqp_succeeded) {

            std::cerr
                << "[COLD START HOMOTOPY RETRY]"
                << " success=1"
                << " selected=1"
                << std::endl;

            result =
                std::move(recovery_result);
        }
        else {

            std::cerr
                << "[COLD START HOMOTOPY RETRY]"
                << " success=0"
                << " keeping_ordinary_result=1"
                << std::endl;

            result =
                std::move(ordinary_result);

            constraints =
                ordinary_constraints;

            /*
             * No homotopy-derived trajectory is being returned,
             * so certificate processing below must describe the
             * original ordinary result.
             */
            use_multi_homotopy_recovery =
                false;

            if (capture_linearized_constraints_) {
                last_linearized_constraints_ =
                    constraints;
            }
        }
    }
}

result.failure_diagnostics.backup_available = failure_diagnostics.backup_available;
result.failure_diagnostics.backup_removal_budget_exceeded = failure_diagnostics.backup_removal_budget_exceeded;
result.failure_diagnostics.braking_collision_feasible = failure_diagnostics.braking_collision_feasible;
result.failure_diagnostics.any_homotopy_geometrically_feasible = failure_diagnostics.any_homotopy_geometrically_feasible;

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
} else if (use_multi_homotopy_recovery) {

 /*
 * The sampled realization was used to select between several
 * candidate convex subproblems. Until support accounting for
 * this discrete selection is formalized, do not report the
 * standard Safe-Horizon certificate.
 */
 result.certificate_status =
     SafeHorizonCertificateStatus::SUPPORT_NOT_EVALUATED;

 result.certified_horizon = -1;

} else if (result.support_cap_status ==
SupportCapStatus::NOT_EVALUATED) {
result.certificate_status =
SafeHorizonCertificateStatus::SUPPORT_NOT_EVALUATED;
} else if (result.support_cap_status == SupportCapStatus::SUPPORT_EXCEEDED) {
result.certificate_status = SafeHorizonCertificateStatus::SUPPORT_EXCEEDED;
} else if (result.support_cap_status == SupportCapStatus::WITHIN_LIMIT) {
// Only this conjunction represents the full-horizon certificate.
result.certificate_status = SafeHorizonCertificateStatus::CERTIFIED;
result.certified_horizon = config_.mpc.horizon;
}
for (const auto& [obs_id, dro_result] : last_dro_results_) {
    result.ambiguity_radius_used = std::max(
        result.ambiguity_radius_used,
        dro_result.rho_used);

    result.dro_risk_evaluation_time +=
        dro_result.risk_diagnostics.evaluation_seconds;
}
result.constraint_construction_time =
std::chrono::duration<double>(constraint_end - constraint_start).count();
result.qp_solve_time =
std::chrono::duration<double>(qp_end - qp_start).count();

if (result.success) {
reference_trajectory_ =
result.ego_trajectory;

if (result.sampled_constraints_satisfied &&
static_cast<int>(
result.control_inputs.size()) ==
config_.mpc.horizon) {

 last_feasible_controls_ =
     result.control_inputs;

 has_feasible_backup_ = true;

}
}

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
                         (config_.mpc.type == MPCType::SH_MPCC ||
                          config_.mpc.type == MPCType::SH_MPCC_DRO_FALLBACK);
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

bool MPCController::trajectory_is_deterministically_admissible(
const std::vector<EgoState>& trajectory,
const std::vector<EgoInput>& inputs
) const
{
if (static_cast<int>(inputs.size()) !=
config_.mpc.horizon) {
return false;
}

if (static_cast<int>(trajectory.size()) !=
    config_.mpc.horizon + 1) {
    return false;
}

// Check all inputs.
for (const auto& input : inputs) {
    if (!input.to_array().allFinite()) {
        return false;
    }

    if (input.a <
            config_.mpc.ego.dynamics.min_acceleration ||
        input.a >
            config_.mpc.ego.dynamics.max_acceleration) {
        return false;
    }

    if (std::abs(input.omega) >
        config_.mpc.ego.dynamics.max_omega) {
        return false;
    }
}

double path_s = trajectory.front().s;

if (reference_path_.has_value() &&
    path_s < 0.0) {

    path_s =
        reference_path_->find_closest_point(
            trajectory.front().position());
}

// Check all states.
for (std::size_t k = 0;
     k < trajectory.size();
     ++k) {

    const auto& state = trajectory[k];

    if (!state.to_array().allFinite()) {
        return false;
    }

    // Velocity bounds.
    if (k > 0 &&
        config_.mpc.constraints.enable_velocity_bounds) {

        if (state.v <
                config_.mpc.ego.dynamics.min_velocity ||
            state.v >
                config_.mpc.ego.dynamics.max_velocity) {

            return false;
        }
    }

    // Road / contouring bounds.
    if (k > 0 &&
        config_.mpc.enable_contouring_constraints &&
        reference_path_.has_value()) {

        path_s =
            reference_path_->find_closest_point(
                state.position(),
                path_s);

        const auto point =
            reference_path_->get_point_at(path_s);

        const Eigen::Vector2d normal(
            -std::sin(point.heading),
             std::cos(point.heading));

        const double lateral_error =
            std::abs(
                normal.dot(
                    state.position() -
                    point.position));

        if (lateral_error >
            config_.mpc.constraints.road_width / 2.0) {

            return false;
        }
    }
}

return true;

}

bool MPCController::project_onto_linearized_set(
const Eigen::VectorXd& target,
const Eigen::MatrixXd& C,
const Eigen::VectorXd& d,
const Eigen::VectorXd& lb,
const Eigen::VectorXd& ub,
Eigen::VectorXd& projection)
{
const int n = static_cast<int>(target.size());

if (n <= 0 ||
    lb.size() != n ||
    ub.size() != n ||
    C.cols() != n ||
    d.size() != C.rows()) {
    return false;
}

// If the set only consists of the control box,
// projection is just component-wise clipping.
if (C.rows() == 0) {
    projection = target;

    for (int i = 0; i < n; ++i) {
        projection(i) =
            std::clamp(
                projection(i),
                lb(i),
                ub(i));
    }

    return projection.allFinite();
}

/*
 * Euclidean projection:
 *
 *     min_x  1/2 ||x - target||^2
 *
 * subject to
 *
 *     C x >= d
 *     lb <= x <= ub.
 *
 * Since your QP convention is
 *
 *     min 1/2 x' H x + g' x,
 *
 * H = I and g = -target give the desired projection.
 */
QPProblem qp;

qp.H =
    Eigen::MatrixXd::Identity(n, n);

qp.g =
    -target;

qp.C = C;
qp.d = d;
qp.lb = lb;
qp.ub = ub;

QPSettings settings;
settings.max_iterations =
    config_.solver.qp_max_iterations;
settings.tolerance =
    config_.solver.qp_tolerance;

const QPResult result =
    qp_solver_.solve(qp, settings);


std::cerr
<< "[DR PROJECTION QP]"
<< " converged=" << result.converged
<< " status=" << result.status
<< " iterations=" << result.iterations
<< " primal_residual=" << result.primal_residual
<< " dual_residual=" << result.dual_residual
<< " x_size=" << result.x.size()
<< " x_finite="
<< (result.x.size() == n &&
    result.x.allFinite())
<< std::endl;


if (result.x.size() == n &&
    result.x.allFinite()) {

    double constraint_violation = 0.0;

    if (C.rows() > 0) {
        constraint_violation =
            std::max(
                0.0,
                (d - C * result.x).maxCoeff());
    }

    double bound_violation = 0.0;

    for (int i = 0; i < n; ++i) {
        bound_violation =
            std::max(
                bound_violation,
                lb(i) - result.x(i));

        bound_violation =
            std::max(
                bound_violation,
                result.x(i) - ub(i));
    }

    std::cerr
        << "[DR PROJECTION CHECK]"
        << " linear_violation="
        << constraint_violation
        << " bound_violation="
        << std::max(0.0, bound_violation)
        << std::endl;
}

if (result.x.size() != n ||
    !result.x.allFinite()) {

    return false;
}

projection =
    result.x;

/*
 * Don't trust only the solver status.
 * Explicitly verify the returned projection.
 */
const double tolerance =
    std::max(
        1e-5,
        10.0 * config_.solver.qp_tolerance);

if (C.rows() > 0) {
    const Eigen::VectorXd residual =
        C * projection - d;

    if (residual.minCoeff() < -tolerance) {
        return false;
    }
}

for (int i = 0; i < n; ++i) {
    if (projection(i) <
            lb(i) - tolerance ||
        projection(i) >
            ub(i) + tolerance) {

        return false;
    }
}

return true;

}

bool MPCController::douglas_rachford_control_projection(
const QPProblem& full_qp,
int num_collision_constraints,
Eigen::VectorXd& projected_delta_u)
{
const int n =
static_cast<int>(full_qp.H.rows());

const int m =
    static_cast<int>(full_qp.C.rows());

if (n <= 0 ||
    full_qp.lb.size() != n ||
    full_qp.ub.size() != n ||
    num_collision_constraints < 0 ||
    num_collision_constraints > m) {

    return false;
}

/*
 * build_condensed_qp() constructs:
 *
 *   [ collision rows ]
 *   [ road rows      ]
 *   [ velocity rows  ]
 *
 * Therefore:
 *
 * B = sampled collision constraints + control box
 * A = deterministic road/velocity constraints + control box
 */
const int n_collision =
    num_collision_constraints;

const int n_deterministic =
    m - n_collision;

const Eigen::MatrixXd C_collision =
    full_qp.C.topRows(n_collision);

const Eigen::VectorXd d_collision =
    full_qp.d.head(n_collision);

const Eigen::MatrixXd C_deterministic =
    full_qp.C.bottomRows(n_deterministic);

const Eigen::VectorXd d_deterministic =
    full_qp.d.tail(n_deterministic);

auto full_violation =
    [&](const Eigen::VectorXd& x) {

        double violation = 0.0;

        if (full_qp.C.rows() > 0) {
            const Eigen::VectorXd residual =
                full_qp.d -
                full_qp.C * x;

            violation =
                std::max(
                    violation,
                    residual.maxCoeff());
        }

        for (int i = 0; i < x.size(); ++i) {
            violation =
                std::max(
                    violation,
                    full_qp.lb(i) - x(i));

            violation =
                std::max(
                    violation,
                    x(i) - full_qp.ub(i));
        }

        return std::max(0.0, violation);
    };

Eigen::VectorXd z =
    Eigen::VectorXd::Zero(n);

Eigen::VectorXd last_collision;
Eigen::VectorXd last_deterministic;

constexpr int kMaxDRIterations = 50;
constexpr double kRelaxation = 1.0;
constexpr double kFeasibilityTolerance = 1e-5;

for (int iter = 0;
     iter < kMaxDRIterations;
     ++iter) {

    // p_B = P_B(z)
    Eigen::VectorXd p_collision;

    if (!project_onto_linearized_set(
            z,
            C_collision,
            d_collision,
            full_qp.lb,
            full_qp.ub,
            p_collision)) {

        std::cerr
            << "[DR RESTORE]"
            << " iter=" << iter
            << " collision_projection_failed=1"
            << std::endl;

        return false;
    }

    // Reflection through B.
    const Eigen::VectorXd reflected =
        2.0 * p_collision - z;

    // p_A = P_A(2 P_B(z) - z)
    Eigen::VectorXd p_deterministic;

    if (!project_onto_linearized_set(
            reflected,
            C_deterministic,
            d_deterministic,
            full_qp.lb,
            full_qp.ub,
            p_deterministic)) {

        std::cerr
            << "[DR RESTORE]"
            << " iter=" << iter
            << " deterministic_projection_failed=1"
            << std::endl;

        return false;
    }

    last_collision =
        p_collision;

    last_deterministic =
        p_deterministic;

    const double collision_violation =
        full_violation(p_collision);

    const double deterministic_violation =
        full_violation(p_deterministic);

    const double dr_residual =
        (p_deterministic -
         p_collision).norm();

    std::cerr
        << "[DR RESTORE]"
        << " iter=" << iter
        << " collision_shadow="
        << collision_violation
        << " deterministic_shadow="
        << deterministic_violation
        << " residual="
        << dr_residual
        << std::endl;

    /*
     * Either shadow iterate lying in the complete intersection
     * is a valid linearized feasible delta-u.
     */
    if (collision_violation <=
        kFeasibilityTolerance) {

        projected_delta_u =
            p_collision;

        return true;
    }

    if (deterministic_violation <=
        kFeasibilityTolerance) {

        projected_delta_u =
            p_deterministic;

        return true;
    }

    /*
     * Douglas-Rachford:
     *
     * z+ = z + lambda(P_A(2 P_B(z) - z) - P_B(z)).
     */
    z +=
        kRelaxation *
        (p_deterministic -
         p_collision);
}

std::cerr
    << "[DR RESTORE]"
    << " max_iterations_reached=1"
    << std::endl;

return false;

}

bool MPCController::restore_dynamically_feasible_plan(
const EgoState& ego_state,
const Eigen::Vector2d& goal,
double reference_velocity,
const std::vector<CollisionConstraint>& constraints,
const std::vector<EgoInput>& seed_inputs,
std::vector<EgoInput>& restored_inputs,
std::vector<EgoState>& restored_trajectory,
double path_progress,
double path_length)
{
const int N =
config_.mpc.horizon;

if (static_cast<int>(
        seed_inputs.size()) != N) {

    return false;
}

auto rollout =
    [&](const std::vector<EgoInput>& inputs) {

        if (reference_path_.has_value()) {
            return ego_dynamics_.rollout_with_spline(
                ego_state,
                inputs,
                *reference_path_);
        }

        return ego_dynamics_.rollout(
            ego_state,
            inputs);
    };

std::vector<EgoInput> u =
    seed_inputs;

std::vector<EgoState> x =
    rollout(u);

constexpr int kMaxOuterIterations = 15;
constexpr double kCollisionTolerance = 1e-4;

for (int outer = 0;
     outer < kMaxOuterIterations;
     ++outer) {

    const auto [current_violation,
                current_violated] =
        evaluate_constraint_violation(
            constraints,
            x);

    const bool deterministic_ok =
        trajectory_is_deterministically_admissible(
            x,
            u);

    std::cerr
        << "[DR RESTORE OUTER]"
        << " iter=" << outer
        << " collision_violation="
        << current_violation
        << " violated_rows="
        << current_violated.size()
        << " deterministic_ok="
        << deterministic_ok
        << std::endl;

    /*
     * We have an actual nonlinear dynamics rollout satisfying
     * both deterministic constraints and fixed collision
     * halfspaces.
     */
    if (std::isfinite(current_violation) &&
        current_violation <=
            kCollisionTolerance &&
        deterministic_ok) {

        restored_inputs =
            u;

        restored_trajectory =
            x;

        return true;
    }

    /*
     * Build the same condensed linearization used by SQP.
     *
     * DR ignores H/g and operates only on C,d,lb,ub.
     */
    QPProblem linearized =
        build_condensed_qp(
            x,
            u,
            goal,
            reference_velocity,
            constraints,
            path_progress,
            path_length);

    Eigen::VectorXd delta_u;

    const bool linearized_restored =
        douglas_rachford_control_projection(
            linearized,
            static_cast<int>(
                constraints.size()),
            delta_u);

    if (!linearized_restored ||
        delta_u.size() != 2 * N ||
        !delta_u.allFinite()) {

        std::cerr
            << "[DR RESTORE OUTER]"
            << " iter=" << outer
            << " linearized_restore_failed=1"
            << std::endl;

        return false;
    }

    /*
     * The DR point solves the current affine approximation.
     * Now reroll the REAL nonlinear dynamics and use a line
     * search before accepting it.
     */
    bool accepted = false;

    double alpha = 1.0;

    std::vector<EgoInput> best_inputs =
        u;

    std::vector<EgoState> best_trajectory =
        x;

    double best_violation =
        current_violation;

    for (int ls = 0;
         ls < 8;
         ++ls) {

        std::vector<EgoInput> trial_inputs =
            u;

        for (int k = 0;
             k < N;
             ++k) {

            const double a =
                std::clamp(
                    u[k].a +
                        alpha *
                        delta_u(2 * k),
                    config_.mpc.ego.dynamics
                        .min_acceleration,
                    config_.mpc.ego.dynamics
                        .max_acceleration);

            const double omega =
                std::clamp(
                    u[k].omega +
                        alpha *
                        delta_u(2 * k + 1),
                    -config_.mpc.ego.dynamics
                         .max_omega,
                    config_.mpc.ego.dynamics
                         .max_omega);

            trial_inputs[k] =
                EgoInput(a, omega);
        }

        std::vector<EgoState>
            trial_trajectory =
                rollout(trial_inputs);

        const auto [trial_violation,
                    trial_violated] =
            evaluate_constraint_violation(
                constraints,
                trial_trajectory);

        const bool trial_deterministic_ok =
            trajectory_is_deterministically_admissible(
                trial_trajectory,
                trial_inputs);

        std::cerr
            << "[DR RESTORE LINESEARCH]"
            << " outer=" << outer
            << " ls=" << ls
            << " alpha=" << alpha
            << " violation="
            << trial_violation
            << " violated_rows="
            << trial_violated.size()
            << " deterministic_ok="
            << trial_deterministic_ok
            << std::endl;

        /*
         * Since the seed is deterministically admissible,
         * don't leave that deterministic feasible set.
         *
         * Collision violation must strictly improve, or reach
         * numerical feasibility.
         */
        if (trial_deterministic_ok &&
            std::isfinite(trial_violation) &&
            (trial_violation <=
                 kCollisionTolerance ||
             trial_violation <
                 best_violation - 1e-7)) {

            best_inputs =
                std::move(trial_inputs);

            best_trajectory =
                std::move(trial_trajectory);

            best_violation =
                trial_violation;

            accepted = true;

            break;
        }

        alpha *= 0.5;
    }

    if (!accepted) {

        std::cerr
            << "[DR RESTORE OUTER]"
            << " iter=" << outer
            << " nonlinear_line_search_failed=1"
            << " current_violation="
            << current_violation
            << std::endl;

        return false;
    }

    u =
        std::move(best_inputs);

    x =
        std::move(best_trajectory);
}

const auto [final_violation,
            final_violated] =
    evaluate_constraint_violation(
        constraints,
        x);

const bool deterministic_ok =
    trajectory_is_deterministically_admissible(
        x,
        u);

std::cerr
    << "[DR RESTORE FINAL]"
    << " violation="
    << final_violation
    << " violated_rows="
    << final_violated.size()
    << " deterministic_ok="
    << deterministic_ok
    << std::endl;

if (std::isfinite(final_violation) &&
    final_violation <=
        kCollisionTolerance &&
    deterministic_ok) {

    restored_inputs =
        std::move(u);

    restored_trajectory =
        std::move(x);

    return true;
}

return false;

}
MPCController::ConflictInfo
MPCController::find_backup_conflict_window(
const std::vector<EgoState>& backup,
const std::vector<Scenario>& scenarios,
double combined_radius
) const
{
const int N =
std::min(
config_.mpc.horizon,
static_cast<int>(backup.size()) - 1);

struct ObstacleConflictStats {
    int count = 0;
    int first_k =
        std::numeric_limits<int>::max();
    int last_k = -1;
};
std::map<int, ObstacleConflictStats> stats;

for (const auto& scenario : scenarios) {

    for (int k = 1; k <= N; ++k) {

        const auto discs =
            compute_ego_disc_positions(
                backup[k],
                config_.mpc.ego.num_discs,
                config_.mpc.ego.length);

        for (const auto& [obstacle_id, prediction] :
             scenario.trajectories) {

            if (k >=
                static_cast<int>(
                    prediction.steps.size())) {
                continue;
            }

            const Eigen::Vector2d obstacle_position =
                prediction.steps[k].mean;

            bool obstacle_bad = false;

            for (const auto& disc : discs) {

                const double clearance =
                    (disc - obstacle_position).norm()
                    - combined_radius;

                if (clearance < 0.0) {
                    obstacle_bad = true;
                    break;
                }
            }

            if (obstacle_bad) {

                auto& s =
                    stats[obstacle_id];

                ++s.count;

                s.first_k =
                    std::min(
                        s.first_k,
                        k);

                s.last_k =
                    std::max(
                        s.last_k,
                        k);
            }
        }
    }
}

ConflictInfo result;

int best_count = 0;

for (const auto& [obstacle_id, s] : stats) {

    if (s.count > best_count) {

        best_count =
            s.count;

        result.obstacle_id =
            obstacle_id;

        result.first_k =
            s.first_k;

        result.last_k =
            s.last_k;
    }
}

return result;

}

    std::vector<EgoState>

MPCController::make_homotopy_seed(
const std::vector<EgoState>& backup,
int conflict_obstacle_id,
HomotopySide side,
int first_conflict_k,
int last_conflict_k,
double combined_radius
) const
{
std::vector<EgoState> seed =
backup;

/*
 * AUTO means "do not impose a discrete homotopy".
 * The existing geometric anchor repair will handle it.
 */
if (side == HomotopySide::Auto ||
    !reference_path_.has_value() ||
    first_conflict_k < 1 ||
    last_conflict_k < first_conflict_k) {

    return seed;
}

const auto& path =
    *reference_path_;

const int N =
    std::min(
        config_.mpc.horizon,
        static_cast<int>(seed.size()) - 1);

/*
* Use the entire available prediction interval to establish
* the homotopy. A late conflict should not require a sudden
* lateral maneuver six steps before impact.
*/
constexpr int kRampAfter = 3;

const int start_k = 1;

const int end_k =
    std::min(
        N,
        last_conflict_k + kRampAfter);

/*
 * Keep the seed inside the nominal road corridor.
 */
const double available_half_width =
    0.5 *
    config_.mpc.constraints.road_width;

const double max_lateral =
    std::max(
        0.0,
        available_half_width - 0.25);

/*
 * Extra seed clearance around the representative obstacle.
 * This is only for choosing the homotopy; the actual SH
 * constraints still use the exact combined radius.
 */
constexpr double kSeedMargin = 0.15;

const double clearance_offset =
    combined_radius +
    kSeedMargin;

auto smoothstep =
    [](double x) {

        x =
            std::clamp(
                x,
                0.0,
                1.0);

        return
            x * x *
            (3.0 - 2.0 * x);
    };

/*
 * Component-wise median. We use medians rather than the most
 * extreme sampled obstacle state so one tail sample cannot send
 * the homotopy seed several metres across the road.
 */
auto median =
    [](std::vector<double> values) {

        if (values.empty()) {
            return 0.0;
        }

        const std::size_t middle =
            values.size() / 2;

        std::nth_element(
            values.begin(),
            values.begin() + middle,
            values.end());

        return values[middle];
    };

for (int k = start_k;
     k <= end_k;
     ++k) {

    double weight = 1.0;

    if (k < first_conflict_k) {

        const double progress =
            static_cast<double>(k) /
            static_cast<double>(
                std::max(
                    1,
                    first_conflict_k));

        weight =
            smoothstep(progress);
    }
    else if (k > last_conflict_k) {

        const double denom =
            std::max(
                1,
                end_k -
                last_conflict_k);

        weight =
            1.0 -
            smoothstep(
                static_cast<double>(
                    k -
                    last_conflict_k) /
                static_cast<double>(
                    denom));
    }

    /*
     * Reference-path geometry.
     */
    double s_ref;

    if (backup[k].has_spline()) {

        s_ref =
            std::clamp(
                backup[k].s,
                0.0,
                path.total_length());
    }
    else {

        s_ref =
            path.find_closest_point(
                backup[k].position());
    }

    const PathPoint pp =
        path.get_point_at(s_ref);

    const Eigen::Vector2d normal(
        -std::sin(pp.heading),
         std::cos(pp.heading));

    const Eigen::Vector2d backup_position =
        backup[k].position();

    const double backup_lateral =
        normal.dot(
            backup_position -
            pp.position);

    /*
     * PATH candidate:
     *
     * Simply return toward the reference centerline.
     */
    double target_lateral = 0.0;

    if (side == HomotopySide::Left ||
        side == HomotopySide::Right) {

        std::vector<Eigen::Vector2d>
obstacle_positions;

for (const auto& scenario : scenarios_) {

const auto prediction_it =
    scenario.trajectories.find(
        conflict_obstacle_id);

if (prediction_it ==
    scenario.trajectories.end()) {
    continue;
}

const auto& prediction =
    prediction_it->second;

if (k >=
    static_cast<int>(
        prediction.steps.size())) {
    continue;
}

obstacle_positions.push_back(
    prediction.steps[k].mean);

}

if (obstacle_positions.empty()) {
continue;
}

std::vector<double> xs;
std::vector<double> ys;

xs.reserve(
obstacle_positions.size());

ys.reserve(
obstacle_positions.size());

for (const auto& p :
obstacle_positions) {

xs.push_back(p.x());
ys.push_back(p.y());

}

const Eigen::Vector2d
representative_obstacle(
median(xs),
median(ys));



        /*
         * Obstacle-relative lateral coordinate.
         *
         * Positive = left of path tangent.
         * Negative = right of path tangent.
         */
        const double obstacle_lateral =
            normal.dot(
                representative_obstacle -
                pp.position);

        if (side ==
            HomotopySide::Left) {

            target_lateral =
                obstacle_lateral +
                clearance_offset;
        }
        else {

            target_lateral =
                obstacle_lateral -
                clearance_offset;
        }

        /*
         * Respect the road corridor.
         */
        target_lateral =
            std::clamp(
                target_lateral,
                -max_lateral,
                max_lateral);
    }

    /*
     * Smoothly interpolate from the dynamically feasible backup
     * to the selected path-/obstacle-relative homotopy.
     */
    const double desired_lateral =
        (1.0 - weight) *
            backup_lateral +
        weight *
            target_lateral;

    const Eigen::Vector2d shifted_position =
        backup_position +
        (desired_lateral -
         backup_lateral) *
            normal;

    seed[k].x =
        shifted_position.x();

    seed[k].y =
        shifted_position.y();

    std::cerr
        << "[HOMOTOPY SEED]"
        << " side="
        << (side == HomotopySide::Path
                ? "path"
                : side ==
                  HomotopySide::Left
                    ? "left"
                    : "right")
        << " k=" << k
        << " backup_lat="
        << backup_lateral
        << " target_lat="
        << target_lateral
        << " desired_lat="
        << desired_lateral
        << std::endl;
}

return seed;

}

double MPCController::score_homotopy_recovery(
    const std::vector<EgoState>& trajectory,
    const std::vector<EgoInput>& inputs,
    const std::vector<EgoInput>& backup_inputs
) const
{
    if (!reference_path_.has_value() ||
        trajectory.size() < 2) {
        return 0.0;
    }

    const auto& path = *reference_path_;

    double score = 0.0;

    constexpr double w_path = 10.0;
    constexpr double w_control = 0.05;

    // NEW
    constexpr double w_heading = 20.0;
    constexpr double w_backward = 200.0;
    constexpr double w_progress = 20.0;

    double forward_progress = 0.0;

    for (std::size_t k = 1;
         k < trajectory.size();
         ++k) {

        double s_ref;

        if (trajectory[k].has_spline()) {
            s_ref = std::clamp(
                trajectory[k].s,
                0.0,
                path.total_length()
            );
        } else {
            s_ref = path.find_closest_point(
                trajectory[k].position()
            );
        }

        const PathPoint pp =
            path.get_point_at(s_ref);

        const Eigen::Vector2d tangent(
            std::cos(pp.heading),
            std::sin(pp.heading)
        );

        const Eigen::Vector2d normal(
            -std::sin(pp.heading),
             std::cos(pp.heading)
        );

        /*
         * Existing contour-error cost.
         */
        const double contour_error =
            normal.dot(
                trajectory[k].position() -
                pp.position
            );

        score +=
            w_path *
            contour_error *
            contour_error;

        /*
         * ------------------------------------------------------
         * NEW: heading alignment
         * ------------------------------------------------------
         */
        double heading_error =
            trajectory[k].theta -
            pp.heading;

        while (heading_error > M_PI)
            heading_error -= 2.0 * M_PI;

        while (heading_error < -M_PI)
            heading_error += 2.0 * M_PI;

        score +=
            w_heading *
            heading_error *
            heading_error;

        /*
         * ------------------------------------------------------
         * NEW: physical forward displacement
         *
         * Do NOT rely only on stored spline s here.
         *
         * This directly measures whether the physical vehicle
         * moved in the forward path-tangent direction.
         * ------------------------------------------------------
         */
        const Eigen::Vector2d dp =
            trajectory[k].position() -
            trajectory[k - 1].position();

        const double ds_forward =
            tangent.dot(dp);

        forward_progress +=
            ds_forward;

        if (ds_forward < 0.0) {
            score +=
                w_backward *
                ds_forward *
                ds_forward;
        }
    }

    /*
     * Reward net physical forward progress.
     */
    score -=
        w_progress *
        forward_progress;

    /*
     * Existing control-deviation cost.
     */
    const std::size_t M =
        std::min(
            inputs.size(),
            backup_inputs.size()
        );

    for (std::size_t k = 0;
         k < M;
         ++k) {

        const double da =
            inputs[k].a -
            backup_inputs[k].a;

        const double dw =
            inputs[k].omega -
            backup_inputs[k].omega;

        score +=
            w_control *
            (da * da + dw * dw);
    }

    return score;
}

std::vector<EgoInput>
MPCController::make_braking_seed_inputs(
const EgoState& ego_state,
const std::vector<EgoInput>& backup_inputs,
const std::vector<EgoState>& backup_trajectory
) const
{
const int N =
config_.mpc.horizon;

std::vector<EgoInput> inputs;
inputs.reserve(N);

double v =
    std::max(0.0, ego_state.v);

for (int k = 0; k < N; ++k) {

    /*
     * Maximum admissible braking until zero speed.
     *
     * -v/dt is the acceleration that would stop exactly
     * during this stage.
     */
    double a = 0.0;

    if (v > 1e-6) {

        a =
            std::max(
                config_.mpc.ego.dynamics.min_acceleration,
                -v / config_.mpc.dt);

        /*
         * A braking recovery must never deliberately
         * accelerate.
         */
        a =
            std::min(
                a,
                0.0);

        a =
            std::clamp(
                a,
                config_.mpc.ego.dynamics.min_acceleration,
                config_.mpc.ego.dynamics.max_acceleration);
    }

    /*
     * Preserve the curvature of the previous feasible plan
     * while slowing down.
     *
     * Since approximately omega = curvature * v,
     * scale the backup yaw rate with speed.
     */
    double omega = 0.0;

    if (k < static_cast<int>(backup_inputs.size())) {

        double backup_v =
            v;

        if (k <
            static_cast<int>(
                backup_trajectory.size())) {

            backup_v =
                std::abs(
                    backup_trajectory[k].v);
        }

        const double speed_ratio =
            std::clamp(
                v /
                std::max(
                    backup_v,
                    0.25),
                0.0,
                1.0);

        omega =
            backup_inputs[k].omega *
            speed_ratio;

        omega =
            std::clamp(
                omega,
                -config_.mpc.ego.dynamics.max_omega,
                 config_.mpc.ego.dynamics.max_omega);
    }

    /*
     * Once stopped, don't rotate in place.
     */
    if (v <= 1e-3) {
        omega = 0.0;
    }

    inputs.emplace_back(
        a,
        omega);

    v =
        std::max(
            0.0,
            v +
            config_.mpc.dt * a);
}

return inputs;

}

// ============================================================================
// SQP Solver
// ============================================================================

MPCResult MPCController::solve_optimization_sqp(
const EgoState& ego_state,
const Eigen::Vector2d& goal,
double reference_velocity,
const std::vector<CollisionConstraint>& constraints,
const std::set<int>& pre_support_scenarios,
const std::vector<EgoInput>& feasible_warmstart_inputs,
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


if (static_cast<int>(feasible_warmstart_inputs.size()) == N) {
    // Use the shifted feasible controls directly as the SQP nominal.
    u_ref = feasible_warmstart_inputs;

    if (reference_path_.has_value()) {
        x_ref =
            ego_dynamics_.rollout_with_spline(
                ego_state,
                u_ref,
                *reference_path_);
    } else {
        x_ref =
            ego_dynamics_.rollout(
                ego_state,
                u_ref);
    }

    std::cerr
        << "[RF SQP INIT]"
        << " using_shifted_feasible_candidate=1"
        << std::endl;
} else if (!reference_trajectory_.empty() &&
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
std::set<int> support_scenarios =
pre_support_scenarios;

std::cerr
<< "[RF SUPPORT SEED]"
<< " removed=" << pre_support_scenarios.size()
<< " support_limit=" << support_limit
<< std::endl;
int support_iterations_evaluated = 0;
auto account_support_evaluation = [&](const std::vector<EgoState>& trajectory) {
    if (!support_certification_enabled) return true;
    collect_active_or_violated_scenarios(
        constraints, trajectory, support_scenarios);
    ++support_iterations_evaluated;
    return static_cast<int>(support_scenarios.size()) <= support_limit;
};

int last_qp_converged = -1;

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
    last_qp_converged = qp_result.converged;

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
                     (config_.mpc.type == MPCType::SH_MPCC ||
                          config_.mpc.type == MPCType::SH_MPCC_DRO_FALLBACK);
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

// Check feasibilityf
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
result.failure_diagnostics.last_qp_converged = last_qp_converged;
// Check actual sampled disc geometry separately from conservative linearized rows.
// This additional read-only check never participates in plan acceptance.
if (!feasible && std::all_of(x_ref.begin(), x_ref.end(),
        [](const EgoState& state) { return state.to_array().allFinite(); })) {
    result.failure_diagnostics.sqp_sampled_collision_feasible =
        backup_incompatible_scenarios(x_ref, scenarios_, config_.combined_radius(),
            config_.mpc.ego.num_discs, config_.mpc.ego.length).empty();
}
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
    fallback.failure_diagnostics = result.failure_diagnostics;
    if (std::all_of(fallback.ego_trajectory.begin(), fallback.ego_trajectory.end(),
            [](const EgoState& state) { return state.to_array().allFinite(); })) {
        fallback.failure_diagnostics.fallback_sampled_collision_feasible =
            backup_incompatible_scenarios(fallback.ego_trajectory, scenarios_, config_.combined_radius(),
                config_.mpc.ego.num_discs, config_.mpc.ego.length).empty();
    }
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
                     (config_.mpc.type == MPCType::SH_MPCC ||
                          config_.mpc.type == MPCType::SH_MPCC_DRO_FALLBACK);
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
last_feasible_controls_.clear();
has_feasible_backup_ = false;
last_removed_scenario_ids_.clear();
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