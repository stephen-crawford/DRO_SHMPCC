/**
 * @file contouring_mpc.cpp
 * @brief Implementation of Contouring MPC with Safe Horizon.
 */

#include "contouring_mpc.hpp"
#include <algorithm>
#include <cmath>

namespace scenario_mpc {

ContouringMPC::ContouringMPC(const ContouringMPCConfig& config)
    : config_(config), base_controller_(config) {
    std::random_device rd;
    rng_ = std::mt19937(rd());

    // Create default S-curve path
    reference_path_ = ReferencePath::create_s_curve(
        config_.path_length,
        config_.s_curve_amplitude
    );
}

void ContouringMPC::set_reference_path(const ReferencePath& path) {
    reference_path_ = path;
}

void ContouringMPC::add_obstacle(const ObstacleConfiguration& config) {
    obstacle_configs_[config.obstacle_id] = config;
    current_modes_[config.obstacle_id] = config.initial_mode;

    // Create obstacle state
    ObstacleState obs;
    obs.x = config.initial_position.x();
    obs.y = config.initial_position.y();
    obs.vx = config.initial_velocity.x();
    obs.vy = config.initial_velocity.y();
    obstacles_[config.obstacle_id] = obs;

    // Initialize in base controller
    base_controller_.initialize_obstacle(config.obstacle_id);
    base_controller_.update_mode_observation(
        config.obstacle_id,
        behavior_to_mode_string(config.initial_mode)
    );
}

void ContouringMPC::place_obstacles_along_path(
    std::vector<ObstacleConfiguration>& configs,
    const std::vector<double>& path_fractions,
    const std::vector<double>& lateral_offsets,
    const std::vector<double>& velocities_toward_path
) {
    for (size_t i = 0; i < configs.size(); ++i) {
        double fraction = path_fractions[i];
        double offset = lateral_offsets[i];
        double vel = velocities_toward_path[i];

        // Get position on path
        double s = fraction * reference_path_.total_length();
        PathPoint pp = reference_path_.get_point_at(s);

        // Compute offset position (perpendicular to path)
        Eigen::Vector2d normal(-std::sin(pp.heading), std::cos(pp.heading));
        Eigen::Vector2d position = pp.position + offset * normal;

        // Velocity toward centerline
        Eigen::Vector2d velocity = -std::copysign(vel, offset) * normal;

        configs[i].initial_position = position;
        configs[i].initial_velocity = velocity;
        configs[i].obstacle_id = static_cast<int>(i);

        add_obstacle(configs[i]);
    }
}

void ContouringMPC::update_obstacle(
    int obstacle_id,
    const Eigen::Vector2d& position,
    const Eigen::Vector2d& velocity,
    const std::string& observed_mode
) {
    if (obstacles_.find(obstacle_id) == obstacles_.end()) {
        return;
    }

    obstacles_[obstacle_id].x = position.x();
    obstacles_[obstacle_id].y = position.y();
    obstacles_[obstacle_id].vx = velocity.x();
    obstacles_[obstacle_id].vy = velocity.y();

    base_controller_.update_mode_observation(obstacle_id, observed_mode);
}

ContouringMPCResult ContouringMPC::solve(
    const EgoState& ego_state,
    double current_path_progress,
    double reference_velocity
) {
    iteration_count_++;

    // Update obstacle modes stochastically
    for (auto& [obs_id, _] : obstacles_) {
        maybe_switch_obstacle_mode(obs_id);
    }

    // Compute goal based on path progress
    double lookahead = reference_velocity * config_.horizon * config_.dt;
    Eigen::Vector2d goal = get_goal_from_progress(current_path_progress, lookahead);

    // Blend goal toward path endpoint when near end (>85% progress)
    double progress_fraction = current_path_progress / reference_path_.total_length();
    if (progress_fraction > 0.85) {
        double alpha = (progress_fraction - 0.85) / 0.15;
        alpha = std::min(alpha, 1.0);
        Eigen::Vector2d endpoint = reference_path_.get_position_at(reference_path_.total_length());
        goal = (1.0 - alpha) * goal + alpha * endpoint;
    }

    // Solve base MPC with path progress for progress-aware cost
    MPCResult base_result = base_controller_.solve(
        ego_state, obstacles_, goal, reference_velocity,
        current_path_progress, reference_path_.total_length()
    );

    // Build contouring result
    ContouringMPCResult result;
    result.success = base_result.success;
    result.ego_trajectory = base_result.ego_trajectory;
    result.control_inputs = base_result.control_inputs;
    result.active_scenarios = base_result.active_scenarios;
    result.solve_time = base_result.solve_time;
    result.cost = base_result.cost;

    // Compute contouring metrics
    if (!result.ego_trajectory.empty()) {
        result.path_progress = current_path_progress;
        result.lateral_error = reference_path_.compute_lateral_offset(
            ego_state.position(), current_path_progress
        );

        double path_heading = reference_path_.get_heading_at(current_path_progress);
        result.heading_error = ego_state.theta - path_heading;
        while (result.heading_error > M_PI) result.heading_error -= 2 * M_PI;
        while (result.heading_error < -M_PI) result.heading_error += 2 * M_PI;
    }

    // Compute obstacle distances
    for (const auto& [obs_id, obs] : obstacles_) {
        double dist = (ego_state.position() - obs.position()).norm();
        result.obstacle_distances.push_back(dist);
    }

    result.num_scenarios_used = static_cast<int>(base_controller_.scenarios().size());
    result.num_active_constraints = static_cast<int>(result.active_scenarios.size());

    return result;
}

ObstacleState ContouringMPC::simulate_obstacle_step(int obstacle_id, double dt) {
    if (obstacles_.find(obstacle_id) == obstacles_.end()) {
        return ObstacleState();
    }

    ObstacleState& obs = obstacles_[obstacle_id];
    const ObstacleConfiguration& config = obstacle_configs_[obstacle_id];
    ObstacleBehavior mode = current_modes_[obstacle_id];

    // Apply mode-specific dynamics
    double ax = 0, ay = 0;

    switch (mode) {
        case ObstacleBehavior::CONSTANT_VELOCITY:
            // No acceleration
            break;

        case ObstacleBehavior::TURN_LEFT: {
            double omega = 0.3;
            double speed = std::sqrt(obs.vx * obs.vx + obs.vy * obs.vy);
            double heading = std::atan2(obs.vy, obs.vx);
            heading += omega * dt;
            obs.vx = speed * std::cos(heading);
            obs.vy = speed * std::sin(heading);
            break;
        }

        case ObstacleBehavior::TURN_RIGHT: {
            double omega = -0.3;
            double speed = std::sqrt(obs.vx * obs.vx + obs.vy * obs.vy);
            double heading = std::atan2(obs.vy, obs.vx);
            heading += omega * dt;
            obs.vx = speed * std::cos(heading);
            obs.vy = speed * std::sin(heading);
            break;
        }

        case ObstacleBehavior::DECELERATING: {
            double decel = 0.5;
            double speed = std::sqrt(obs.vx * obs.vx + obs.vy * obs.vy);
            if (speed > 0.1) {
                double factor = std::max(0.0, 1.0 - decel * dt / speed);
                obs.vx *= factor;
                obs.vy *= factor;
            }
            break;
        }

        case ObstacleBehavior::ACCELERATING: {
            double accel = 0.3;
            double speed = std::sqrt(obs.vx * obs.vx + obs.vy * obs.vy);
            if (speed > 0.01) {
                double factor = 1.0 + accel * dt / speed;
                obs.vx *= factor;
                obs.vy *= factor;
            }
            break;
        }

        case ObstacleBehavior::LANE_CHANGE_LEFT:
            obs.vy += 0.3 * dt;
            break;

        case ObstacleBehavior::LANE_CHANGE_RIGHT:
            obs.vy -= 0.3 * dt;
            break;

        case ObstacleBehavior::PATH_INTERSECT: {
            // Oscillate across path
            double s = reference_path_.find_closest_point(obs.position());
            double offset = reference_path_.compute_lateral_offset(obs.position(), s);

            // Change direction when too far from centerline
            double max_offset = config_.road_width / 2 - 0.5;
            if (std::abs(offset) > max_offset) {
                // Reverse perpendicular velocity component
                double heading = reference_path_.get_heading_at(s);
                Eigen::Vector2d normal(-std::sin(heading), std::cos(heading));
                double perp_vel = obs.vx * normal.x() + obs.vy * normal.y();
                if ((offset > 0 && perp_vel > 0) || (offset < 0 && perp_vel < 0)) {
                    // Reverse direction
                    obs.vx -= 2 * perp_vel * normal.x();
                    obs.vy -= 2 * perp_vel * normal.y();
                }
            }
            break;
        }
    }

    // Update position
    obs.x += obs.vx * dt;
    obs.y += obs.vy * dt;

    // Update mode observation
    base_controller_.update_mode_observation(
        obstacle_id, behavior_to_mode_string(mode)
    );

    return obs;
}

void ContouringMPC::reset() {
    base_controller_.reset();
    obstacles_.clear();
    obstacle_configs_.clear();
    current_modes_.clear();
    iteration_count_ = 0;
}

double ContouringMPC::compute_contouring_cost(
    const std::vector<EgoState>& trajectory,
    double start_progress
) const {
    double cost = 0;
    double progress = start_progress;

    for (const auto& state : trajectory) {
        // Find closest point on path
        double s = reference_path_.find_closest_point(state.position());

        // Contouring error (lateral)
        double lateral = reference_path_.compute_lateral_offset(state.position(), s);
        cost += config_.contour_weight * lateral * lateral;

        // Lag error (progress)
        double lag = s - progress;
        cost += config_.lag_weight * lag * lag;

        progress = s;
    }

    // Terminal cost
    if (!trajectory.empty()) {
        const auto& final_state = trajectory.back();
        double s = reference_path_.find_closest_point(final_state.position());
        double lateral = reference_path_.compute_lateral_offset(final_state.position(), s);
        cost += config_.terminal_weight * lateral * lateral;

        double path_heading = reference_path_.get_heading_at(s);
        double heading_error = final_state.theta - path_heading;
        cost += config_.terminal_angle_weight * heading_error * heading_error;
    }

    return cost;
}

bool ContouringMPC::check_road_constraints(const EgoState& state, double path_progress) const {
    double lateral = reference_path_.compute_lateral_offset(state.position(), path_progress);
    double half_width = config_.road_width / 2 - config_.ego_radius;
    return std::abs(lateral) <= half_width;
}

Eigen::Vector2d ContouringMPC::get_goal_from_progress(double current_progress, double lookahead) const {
    double goal_s = std::min(current_progress + lookahead, reference_path_.total_length());
    return reference_path_.get_position_at(goal_s);
}

std::string ContouringMPC::behavior_to_mode_string(ObstacleBehavior behavior) {
    switch (behavior) {
        case ObstacleBehavior::CONSTANT_VELOCITY: return "constant_velocity";
        case ObstacleBehavior::TURN_LEFT: return "turn_left";
        case ObstacleBehavior::TURN_RIGHT: return "turn_right";
        case ObstacleBehavior::DECELERATING: return "decelerating";
        case ObstacleBehavior::ACCELERATING: return "constant_velocity";  // Map to CV
        case ObstacleBehavior::LANE_CHANGE_LEFT: return "lane_change_left";
        case ObstacleBehavior::LANE_CHANGE_RIGHT: return "lane_change_right";
        case ObstacleBehavior::PATH_INTERSECT: return "constant_velocity";
        default: return "constant_velocity";
    }
}

void ContouringMPC::maybe_switch_obstacle_mode(int obstacle_id) {
    const ObstacleConfiguration& config = obstacle_configs_[obstacle_id];

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (dist(rng_) < config.mode_switch_probability) {
        // Switch to a random available mode
        if (!config.available_modes.empty()) {
            std::uniform_int_distribution<int> mode_dist(0, config.available_modes.size() - 1);
            int idx = mode_dist(rng_);
            const std::string& new_mode = config.available_modes[idx];

            // Map string to behavior
            if (new_mode == "constant_velocity") {
                current_modes_[obstacle_id] = ObstacleBehavior::CONSTANT_VELOCITY;
            } else if (new_mode == "turn_left") {
                current_modes_[obstacle_id] = ObstacleBehavior::TURN_LEFT;
            } else if (new_mode == "turn_right") {
                current_modes_[obstacle_id] = ObstacleBehavior::TURN_RIGHT;
            } else if (new_mode == "decelerating") {
                current_modes_[obstacle_id] = ObstacleBehavior::DECELERATING;
            } else if (new_mode == "lane_change_left") {
                current_modes_[obstacle_id] = ObstacleBehavior::LANE_CHANGE_LEFT;
            } else if (new_mode == "lane_change_right") {
                current_modes_[obstacle_id] = ObstacleBehavior::LANE_CHANGE_RIGHT;
            }
        }
    }
}

std::vector<ObstacleConfiguration> create_default_obstacle_configs() {
    std::vector<ObstacleConfiguration> configs(4);

    // Obstacle 0: Constant velocity
    configs[0].name = "Constant Velocity Obstacle";
    configs[0].initial_mode = ObstacleBehavior::CONSTANT_VELOCITY;
    configs[0].available_modes = {"constant_velocity", "decelerating"};
    configs[0].mode_switch_probability = 0.05;
    configs[0].radius = 0.35;
    configs[0].position_std = 0.15;
    configs[0].uncertainty_growth = 0.08;

    // Obstacle 1: Turning
    configs[1].name = "Turning Obstacle";
    configs[1].initial_mode = ObstacleBehavior::TURN_LEFT;
    configs[1].available_modes = {"turn_left", "turn_right", "constant_velocity"};
    configs[1].mode_switch_probability = 0.1;
    configs[1].radius = 0.35;
    configs[1].position_std = 0.15;
    configs[1].uncertainty_growth = 0.08;

    // Obstacle 2: Decelerating
    configs[2].name = "Decelerating Obstacle";
    configs[2].initial_mode = ObstacleBehavior::DECELERATING;
    configs[2].available_modes = {"decelerating", "constant_velocity"};
    configs[2].mode_switch_probability = 0.08;
    configs[2].radius = 0.35;
    configs[2].position_std = 0.15;
    configs[2].uncertainty_growth = 0.08;

    // Obstacle 3: Lane changing
    configs[3].name = "Lane Change Obstacle";
    configs[3].initial_mode = ObstacleBehavior::LANE_CHANGE_LEFT;
    configs[3].available_modes = {"lane_change_left", "lane_change_right", "constant_velocity"};
    configs[3].mode_switch_probability = 0.15;
    configs[3].radius = 0.35;
    configs[3].position_std = 0.15;
    configs[3].uncertainty_growth = 0.08;

    return configs;
}

}  // namespace scenario_mpc
