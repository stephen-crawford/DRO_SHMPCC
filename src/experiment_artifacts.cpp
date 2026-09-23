/**
 * @file experiment_artifacts.cpp
 * @brief Reproducibility and visualization artifacts for canonical rollouts.
 */

#include "experiment_artifacts_internal.hpp"
#include "collision_constraints.hpp"

#include "schuurmans_ambiguity.hpp"
#include "simple_gif_encoder.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
#include <map>
#include <tuple>

namespace dro_mpc {
namespace detail {
namespace {

namespace fs = std::filesystem;

std::string yaml_quote(const std::string& value) {
    std::string quoted = "\"";
    for (const char ch : value) {
        if (ch == '\\' || ch == '\"') quoted += '\\';
        quoted += ch;
    }
    return quoted + "\"";
}

std::string xml_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '\"': escaped += "&quot;"; break;
            case '\'': escaped += "&apos;"; break;
            default: escaped += ch; break;
        }
    }
    return escaped;
}

std::string safe_path_component(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        out += std::isalnum(ch) || ch == '-' || ch == '_' ?
            static_cast<char>(ch) : '_';
    }
    return out.empty() ? "rollout" : out;
}

std::string environment_yaml_name(EnvironmentType environment) {
    switch (environment) {
        case EnvironmentType::T_INTERSECTION: return "t_intersection";
        case EnvironmentType::FOUR_WAY_INTERSECTION: return "four_way_intersection";
        case EnvironmentType::S_CURVE: return "s_curve";
        case EnvironmentType::TWO_LANE_ROUNDABOUT: return "two_lane_roundabout";
        case EnvironmentType::FOUR_LANE_ROUNDABOUT: return "four_lane_roundabout";
        case EnvironmentType::TWO_LANE_HIGHWAY: return "two_lane_highway";
        case EnvironmentType::FOUR_LANE_HIGHWAY: return "four_lane_highway";
        case EnvironmentType::ENTER_RAMP: return "enter_ramp";
        case EnvironmentType::EXIT_RAMP: return "exit_ramp";
        case EnvironmentType::OVERTAKE_SLOW_LEAD: return "overtake_slow_lead";
        case EnvironmentType::NARROW_CORRIDOR: return "narrow_corridor";
        case EnvironmentType::INTERSECTION: return "intersection";
        case EnvironmentType::ONCOMING: return "oncoming";
    }
    return "s_curve";
}

std::string switch_regime_yaml_name(ModeSwitchConfiguration regime) {
    return regime == ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM
        ? "markov_jump_system" : "hold_over_horizon";
}

std::string obstacle_history_yaml_name(ObstacleHistoryConfiguration history) {
    return history == ObstacleHistoryConfiguration::SHARED
        ? "shared_history_classes" : "independent_history";
}

std::string belief_kind_yaml_name(NominalBeliefKind kind) {
    return kind == NominalBeliefKind::STICKY ? "sticky" : "dirichlet";
}

void write_bool(std::ofstream& out, const char* key, bool value) {
    out << key << ": " << (value ? "true" : "false") << '\n';
}

template <typename T>
void write_scalar(std::ofstream& out, const char* key, const T& value) {
    out << key << ": " << value << '\n';
}

void write_string_list(std::ofstream& out, const char* key,
                       const std::vector<std::string>& values) {
    out << key << ": [";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ", ";
        out << values[i];
    }
    out << "]\n";
}

void write_double_list(std::ofstream& out, const char* key,
                       const std::vector<double>& values) {
    out << key << ": [";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ", ";
        out << values[i];
    }
    out << "]\n";
}

void write_obstacle_states(std::ofstream& out,
                           const std::vector<ObstacleState>& states) {
    out << "obstacle_initial_states: [";
    for (std::size_t i = 0; i < states.size(); ++i) {
        if (i != 0) out << "; ";
        out << states[i].x << ',' << states[i].y << ','
            << states[i].vx << ',' << states[i].vy;
    }
    out << "]\n";
}

void write_waypoints(std::ofstream& out,
                     const std::vector<Eigen::Vector2d>& waypoints) {
    out << "path_waypoints: [";
    for (std::size_t i = 0; i < waypoints.size(); ++i) {
        if (i != 0) out << "; ";
        out << waypoints[i].x() << ',' << waypoints[i].y();
    }
    out << "]\n";
}

void write_resolved_config(std::ofstream& out, const ExperimentConfig& config) {
    const auto& mpc = config.mpc;
    const auto& sampling = mpc.sampling;
    const auto& constraints = mpc.constraints;
    const auto& ego = mpc.ego;
    const auto& dynamics = ego.dynamics;
    const auto& dro = config.dro;
    const auto& dro_solver = dro.solver;
    const auto& radius = dro_solver.radius_calibration;
    const auto& obstacle = config.obstacles;
    const auto& environment = config.environment;

    out << "# Resolved, replayable ExperimentConfig.\n";
    write_scalar(out, "mpc_type", mpc_type_name(mpc.type));
    write_bool(out, "nominal_resampling_baseline", mpc.nominal_resampling_baseline);
    write_scalar(out, "progress_weight", mpc.objective.progress_weight);
    write_scalar(out, "horizon", mpc.horizon);
    write_scalar(out, "dt", mpc.dt);
    write_scalar(out, "num_scenarios", sampling.num_scenarios);
    write_bool(out, "automatically_compute_sample_size",
               sampling.automatically_compute_sample_size);
    write_bool(out, "safe_horizon_enabled", mpc.safe_horizon_enabled);
    write_bool(out, "enable_contouring_constraints", mpc.enable_contouring_constraints);
    write_scalar(out, "road_width", constraints.road_width);
    write_scalar(out, "safety_margin", constraints.safety_margin);
    write_bool(out, "enable_velocity_bounds", constraints.enable_velocity_bounds);
    write_scalar(out, "clearance_filter_distance", constraints.clearance_filter_distance);
    write_scalar(out, "support_cap_n_bar", constraints.support_cap_n_bar);
    write_scalar(out, "scenario_removal_budget", constraints.scenario_removal_budget);

    write_scalar(out, "ego_radius", ego.radius);
    write_scalar(out, "ego_length", ego.length);
    write_scalar(out, "num_discs", ego.num_discs);
    write_scalar(out, "max_velocity", dynamics.max_velocity);
    write_scalar(out, "min_velocity", dynamics.min_velocity);
    write_scalar(out, "max_acceleration", dynamics.max_acceleration);
    write_scalar(out, "min_acceleration", dynamics.min_acceleration);
    write_scalar(out, "max_omega", dynamics.max_omega);

    write_scalar(out, "belief_kind", belief_kind_yaml_name(sampling.belief_kind));
    write_scalar(out, "self_persistence_prior",
                 sampling.mode_belief.self_persistence_prior);
    write_bool(out, "markov_jump_system", sampling.markov_jump_system);
    write_scalar(out, "max_history_length", sampling.max_history_length);
    write_scalar(out, "one_minus_chance_constraint_violation_probability",
                 sampling.one_minus_chance_constraint_violation_probability);
    write_scalar(out, "chance_of_certificate_violation",
                 sampling.chance_of_certificate_violation);

    write_bool(out, "dro_enabled", dro.enabled);
    write_scalar(out, "fixed_rho", dro.fixed_rho);
    write_scalar(out, "base_radius", dro_solver.base_radius);
    write_scalar(out, "min_radius", dro_solver.min_radius);
    write_scalar(out, "max_radius", dro_solver.max_radius);
    write_bool(out, "use_calibrated_radius", radius.use_calibrated_radius);
    write_scalar(out, "confidence_beta", radius.confidence_beta);
    write_scalar(out, "alpha_one_sided", radius.alpha_one_sided);
    write_scalar(out, "calibration_scale", radius.calibration_scale);
    write_bool(out, "use_primal_ot", radius.use_primal_ot);
    write_scalar(out, "risk_measure", risk_measure_name(radius.risk_measure));
    write_scalar(out, "risk_scoring_model",
                 risk_scoring_model_name(radius.risk_scoring_model));
    write_scalar(out, "risk_horizon", radius.risk_horizon);
    write_scalar(out, "divergence", schuurmans::divergence_name(radius.divergence));
    write_scalar(out, "ground_cost", ground_cost_name(dro_solver.ground_cost_type));
    write_scalar(out, "joint_risk_samples", radius.joint_risk_samples);
    write_scalar(out, "joint_risk_seed", radius.joint_risk_seed);
    write_scalar(out, "mixture_sequence_samples", radius.mixture_sequence_samples);
    write_scalar(out, "sigma_floor", radius.sigma_floor);
    write_bool(out, "use_entropic_allocator", radius.use_entropic_allocator);
    write_scalar(out, "entropic_tau", radius.entropic_tau);

    write_scalar(out, "sqp_max_iterations", config.solver.sqp_max_iterations);
    write_scalar(out, "sqp_convergence_tol", config.solver.sqp_convergence_tol);
    write_scalar(out, "qp_max_iterations", config.solver.qp_max_iterations);
    write_scalar(out, "qp_tolerance", config.solver.qp_tolerance);

    write_scalar(out, "obstacle_radius", config.obstacle_radius);
    write_scalar(out, "num_obstacles", obstacle.num_obstacles);
    write_scalar(out, "num_modes", obstacle.num_modes);
    write_scalar(out, "num_classes", obstacle.num_classes);
    write_bool(out, "obstacle_place_on_path", obstacle.place_on_path);
    write_scalar(out, "obstacles_per_class", obstacle.obstacles_per_class);
    write_scalar(out, "obstacle_history", obstacle_history_yaml_name(obstacle.history));
    write_scalar(out, "switch_regime", switch_regime_yaml_name(obstacle.switch_regime));
    write_scalar(out, "switch_prob", obstacle.switch_prob);
    write_bool(out, "randomize_available_modes", obstacle.randomize_available_modes);
    write_bool(out, "randomize_modes_per_obstacle", obstacle.randomize_modes_per_obstacle);
    write_string_list(out, "obs_modes", obstacle.obs_modes);
    write_scalar(out, "rare_mode", yaml_quote(obstacle.rare_mode));
    write_scalar(out, "rare_mode_probability", obstacle.rare_mode_probability);
    write_double_list(out, "obs_arc_fractions", obstacle.obs_arc_fractions);
    write_obstacle_states(out, obstacle.initial_obstacle_states);
    write_scalar(out, "obs_path_fraction", obstacle.default_arc_fraction);
    write_scalar(out, "obstacle_process_noise", obstacle.process_noise);
    write_bool(out, "obstacle_prediction_noise", obstacle.prediction_noise);
    write_scalar(out, "obstacle_speed_cap", obstacle.speed_cap);
    write_scalar(out, "obstacle_behavior", yaml_quote(obstacle.behavior));
    write_scalar(out, "obstacle_behavior_initial_speed", obstacle.behavior_initial_speed);
    write_scalar(out, "obstacle_behavior_path_offset", obstacle.behavior_path_offset);
    write_scalar(out, "shift_psi", obstacle.shift.psi);
    write_scalar(out, "shift_boost", obstacle.shift.dangerous_boost);
    write_scalar(out, "boosted_mode", obstacle.shift.boosted_mode);

    write_scalar(out, "environment", environment_yaml_name(environment.type));
    write_bool(out, "path_completion_termination", environment.path_completion_termination);
    write_scalar(out, "path_completion_fraction", environment.path_completion_fraction);
    write_waypoints(out, environment.path_waypoints);
    write_bool(out, "path_closed_loop", environment.path_closed_loop);
    write_scalar(out, "path_sample_spacing", environment.path_sample_spacing);
    write_scalar(out, "path_control_point_spacing", environment.path_control_point_spacing);
    write_scalar(out, "path_max_lateral_acceleration",
                 environment.path_max_lateral_acceleration);
    write_scalar(out, "s_curve_length", environment.s_curve_length);
    write_scalar(out, "s_curve_amplitude", environment.s_curve_amplitude);
    write_scalar(out, "s_curve_points", environment.s_curve_points);
    write_scalar(out, "ego_initial_v", environment.ego_initial_v);
    write_scalar(out, "lane_count", environment.lane_count);
    write_scalar(out, "lane_width", environment.lane_width);
    write_scalar(out, "shoulder_width", environment.shoulder_width);
    write_scalar(out, "median_width", environment.median_width);
    write_scalar(out, "road_length", environment.road_length);
    write_scalar(out, "intersection_box_size", environment.intersection_box_size);
    write_scalar(out, "corner_radius", environment.corner_radius);
    write_scalar(out, "ramp_length", environment.ramp_length);
    write_scalar(out, "merge_length", environment.merge_length);
    write_scalar(out, "roundabout_radius", environment.roundabout_radius);
    write_scalar(out, "corridor_width", environment.corridor_width);
    write_scalar(out, "curve_radius", environment.curve_radius);
    write_scalar(out, "transition_length", environment.transition_length);

    write_scalar(out, "rollout_steps", config.rollout.rollout_steps);
    write_scalar(out, "scenario_tag", yaml_quote(config.rollout.scenario_tag));
    write_scalar(out, "method_name", yaml_quote(config.rollout.method_name));
    write_scalar(out, "metrics_v_ref", config.rollout.metrics_v_ref);
    write_scalar(out, "artifact_output_directory",
                 yaml_quote(config.artifacts.output_directory));
    write_scalar(out, "artifact_run_name", yaml_quote(config.artifacts.run_name));
    write_bool(out, "artifact_write_manifest",
               config.artifacts.write_reproducibility_manifest);
    write_bool(out, "artifact_write_trace_csv", config.artifacts.write_trace_csv);
    write_bool(out, "artifact_write_analysis_csv", config.artifacts.write_analysis_csv);
    write_bool(out, "artifact_capture_attempt_diagnostics", config.artifacts.capture_attempt_diagnostics);
    write_bool(out, "artifact_show_support_scenarios", config.artifacts.show_support_scenarios);
    write_scalar(out, "artifact_support_preview_count", config.artifacts.support_preview_count);
    write_bool(out, "artifact_show_linearized_constraints",
               config.artifacts.show_linearized_constraints);
    write_bool(out, "artifact_write_visualization_svg",
               config.artifacts.write_visualization_svg);
    write_bool(out, "artifact_write_visualization_gif",
               config.artifacts.write_visualization_gif);
    write_bool(out, "artifact_show_sampled_scenarios", config.artifacts.show_sampled_scenarios);
    write_scalar(out, "artifact_scenario_preview_count", config.artifacts.scenario_preview_count);
    write_scalar(out, "artifact_gif_frame_stride",
                 config.artifacts.gif_frame_stride);
    write_scalar(out, "artifact_gif_playback_rate",
                 config.artifacts.gif_playback_rate);
    write_bool(out, "artifact_write_rviz_replay",
               config.artifacts.write_rviz_replay_bundle);
}

void require_open(const std::ofstream& stream, const fs::path& path) {
    if (!stream) {
        throw std::runtime_error("could not write experiment artifact '" +
                                 path.string() + "'");
    }
}

void write_manifest(const fs::path& path, const ExperimentConfig& config,
                    const RolloutRecord& record, const SeedBundle& seeds) {
    std::ofstream out(path);
    require_open(out, path);
    out << std::setprecision(17);
    out << "artifact_schema_version: 3\n";
    out << "source_revision: " << yaml_quote(
#ifdef DRO_MPC_GIT_REVISION
        DRO_MPC_GIT_REVISION
#else
        "unknown"
#endif
    ) << "\n";
    out << "source_tree_state_at_build: " << yaml_quote(
#ifdef DRO_MPC_SOURCE_TREE_STATE
        DRO_MPC_SOURCE_TREE_STATE
#else
        "unknown"
#endif
    ) << "\n";
    out << "config_source: " << yaml_quote(config.config_source) << "\n";
    out << "master_seed: " << seeds.master << "\n";
    out << "plant_seed: " << seeds.env << "\n";
    out << "predictor_seed: " << seeds.predictor << "\n";
    out << "controller_seed: " << seeds.scenario << "\n";
    out << "method: " << yaml_quote(record.method) << "\n";
    out << "scenario: " << yaml_quote(record.scenario) << "\n";
    out << "qp_backend: " << yaml_quote(record.qp_backend) << "\n";
    out << "qp_solver_identity: " << yaml_quote(record.qp_solver_identity) << "\n";
    out << "collision: " << (record.collision ? "true" : "false") << "\n";
    out << "completed_path: " << (record.completed_path ? "true" : "false") << "\n";
    out << "termination_reason: " << yaml_quote(record.termination_reason) << "\n";
    out << "failed_decision_step: " << record.failed_decision_step << "\n";
    out << "resolved_config: resolved_config.yaml\n";
}

void write_trace_csv(const fs::path& path, const RolloutTrace& trace) {
    std::ofstream out(path);
    require_open(out, path);
    out << std::setprecision(17);
    out << "step,time_s,actor,obstacle_id,mode,x,y,theta,v,vx,vy,"
           "path_progress,minimum_clearance,ambiguity_radius,solve_time_ms,collision\n";
    for (const auto& frame : trace.frames) {
        out << frame.step << ',' << frame.time_seconds << ",ego,-1,,"
            << frame.ego.x << ',' << frame.ego.y << ',' << frame.ego.theta << ','
            << frame.ego.v << ",,," << frame.path_progress << ','
            << frame.minimum_clearance << ',' << frame.ambiguity_radius << ','
            << frame.solve_time_ms << ',' << (frame.collision ? 1 : 0) << '\n';
        for (std::size_t i = 0; i < frame.obstacles.size(); ++i) {
            const auto& obstacle = frame.obstacles[i];
            const std::string mode = i < frame.obstacle_modes.size()
                ? frame.obstacle_modes[i] : "";
            out << frame.step << ',' << frame.time_seconds << ",obstacle," << i << ','
                << mode << ',' << obstacle.x << ',' << obstacle.y << ",,,"
                << obstacle.vx << ',' << obstacle.vy << ',' << frame.path_progress << ','
                << frame.minimum_clearance << ',' << frame.ambiguity_radius << ','
                << frame.solve_time_ms << ',' << (frame.collision ? 1 : 0) << '\n';
        }
    }
}

struct WorldBounds {
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    void include(const Eigen::Vector2d& point) {
        min_x = std::min(min_x, point.x());
        max_x = std::max(max_x, point.x());
        min_y = std::min(min_y, point.y());
        max_y = std::max(max_y, point.y());
    }

    void include(const ReferencePath& path) {
        for (const auto& point : path.points()) include(point.position);
    }

    void include_trace(const RolloutTrace& trace) {
        for (const auto& frame : trace.frames) {
            include(frame.ego.position());
            for (const auto& obstacle : frame.obstacles) include(obstacle.position());
            if (!frame.show_support_scenarios) {
                for (const auto& row : frame.linearized_constraints) {
                    if (row.k < 0 || static_cast<size_t>(row.k) >= frame.predicted_ego.size()) continue;
                    const Eigen::Vector2d disc = compute_collision_disc_center(frame.predicted_ego[row.k], row);
                    if (disc.allFinite()) include(disc);
                }
            }
        }
    }

    void pad() {
        if (!std::isfinite(min_x) || !std::isfinite(min_y)) {
            min_x = min_y = -1.0;
            max_x = max_y = 1.0;
        }
        const double span_x = std::max(1.0, max_x - min_x);
        const double span_y = std::max(1.0, max_y - min_y);
        const double margin = 0.08 * std::max(span_x, span_y) + 0.5;
        min_x -= margin;
        max_x += margin;
        min_y -= margin;
        max_y += margin;
    }
};

struct SvgTransform {
    double scale = 1.0;
    double offset_x = 0.0;
    double offset_y = 0.0;

    std::pair<double, double> map(const Eigen::Vector2d& point) const {
        return {offset_x + scale * point.x(), offset_y - scale * point.y()};
    }
};

SvgTransform make_svg_transform(const WorldBounds& bounds, double width, double height) {
    constexpr double padding = 60.0;
    const double scale = std::min(
        (width - 2.0 * padding) / std::max(1.0, bounds.max_x - bounds.min_x),
        (height - 2.0 * padding) / std::max(1.0, bounds.max_y - bounds.min_y));
    const double drawn_width = scale * (bounds.max_x - bounds.min_x);
    const double drawn_height = scale * (bounds.max_y - bounds.min_y);
    return {scale,
            0.5 * (width - drawn_width) - scale * bounds.min_x,
            0.5 * (height - drawn_height) + scale * bounds.max_y};
}

// The GIF uses an intentionally small fixed palette.  It keeps artifacts
// compact, makes the animation portable, and mirrors the visual vocabulary of
// rollout.svg without adding an image-processing dependency.
enum GifPaletteIndex : std::uint8_t {
    GIF_BACKGROUND = 0,
    GIF_ROAD = 1,
    GIF_ROUTE = 2,
    GIF_EGO = 3,
    GIF_OBSTACLE_0 = 4,
    GIF_OBSTACLE_1 = 5,
    GIF_OBSTACLE_2 = 6,
    GIF_OBSTACLE_3 = 7,
    GIF_START = 8,
    GIF_COLLISION = 9,
    GIF_WHITE = 10,
    GIF_SUPPORT = 11,
    GIF_SUPPORT_HALFSPACE = 12,
    GIF_UNUSED_13 = 13,
    GIF_UNUSED_14 = 14,
    GIF_UNUSED_15 = 15,
};

constexpr std::array<gif::Color, 16> GIF_PALETTE = {{
    {16, 24, 32},    // background
    {99, 115, 129},  // roads
    {87, 217, 163},  // route
    {88, 166, 255},  // ego
    {255, 140, 66},  // obstacle 0
    {247, 120, 186}, // obstacle 1
    {210, 153, 34},  // obstacle 2
    {163, 113, 247}, // obstacle 3
    {87, 217, 163},  // start/final no-collision
    {248, 81, 73},   // collision
    {230, 237, 243}, // white
    {0, 229, 255}, {190, 195, 200}, {150, 155, 160},
    {110, 115, 120}, {70, 75, 80},
}};

constexpr int GIF_WIDTH = 800;
constexpr int GIF_HEIGHT = 520;

class IndexedCanvas {
public:
    IndexedCanvas(int width, int height, std::uint8_t background)
        : width_(width), height_(height), pixels_(
            static_cast<std::size_t>(width) * height, background) {}

    void clear(std::uint8_t color) {
        std::fill(pixels_.begin(), pixels_.end(), color);
    }

    void draw_disk(int cx, int cy, int radius, std::uint8_t color) {
        const int squared_radius = radius * radius;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy <= squared_radius) put(cx + dx, cy + dy, color);
            }
        }
    }

    void draw_line(double start_x, double start_y, double end_x, double end_y,
                   int thickness, std::uint8_t color) {
        int x0 = static_cast<int>(std::lround(start_x));
        int y0 = static_cast<int>(std::lround(start_y));
        const int x1 = static_cast<int>(std::lround(end_x));
        const int y1 = static_cast<int>(std::lround(end_y));
        const int dx = std::abs(x1 - x0);
        const int step_x = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0);
        const int step_y = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        const int radius = std::max(0, thickness / 2);
        while (true) {
            draw_disk(x0, y0, radius, color);
            if (x0 == x1 && y0 == y1) break;
            const int twice_error = 2 * error;
            if (twice_error >= dy) {
                error += dy;
                x0 += step_x;
            }
            if (twice_error <= dx) {
                error += dx;
                y0 += step_y;
            }
        }
    }

    const std::vector<std::uint8_t>& pixels() const noexcept { return pixels_; }

private:
    void put(int x, int y, std::uint8_t color) {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return;
        pixels_[static_cast<std::size_t>(y) * width_ + x] = color;
    }

    int width_ = 0;
    int height_ = 0;
    std::vector<std::uint8_t> pixels_;
};

std::uint8_t gif_obstacle_color(std::size_t index) {
    constexpr std::array<std::uint8_t, 4> colors = {
        GIF_OBSTACLE_0, GIF_OBSTACLE_1, GIF_OBSTACLE_2, GIF_OBSTACLE_3};
    return colors[index % colors.size()];
}

void draw_path_on_canvas(IndexedCanvas& canvas, const ReferencePath& path,
                         const SvgTransform& transform, int thickness,
                         std::uint8_t color) {
    if (path.points().size() < 2) return;
    for (std::size_t index = 1; index < path.points().size(); ++index) {
        const auto [x0, y0] = transform.map(path.points()[index - 1].position);
        const auto [x1, y1] = transform.map(path.points()[index].position);
        canvas.draw_line(x0, y0, x1, y1, thickness, color);
    }
}

void draw_actor_history(IndexedCanvas& canvas, const RolloutTrace& trace,
                        std::size_t last_frame, int obstacle_id,
                        const SvgTransform& transform, int thickness,
                        std::uint8_t color) {
    if (trace.frames.size() < 2 || last_frame == 0) return;
    const std::size_t capped_last = std::min(last_frame, trace.frames.size() - 1);
    bool has_previous = false;
    Eigen::Vector2d previous = Eigen::Vector2d::Zero();
    for (std::size_t index = 0; index <= capped_last; ++index) {
        const auto& frame = trace.frames[index];
        if (obstacle_id >= 0 && obstacle_id >= static_cast<int>(frame.obstacles.size())) {
            has_previous = false;
            continue;
        }
        const Eigen::Vector2d current = obstacle_id < 0
            ? frame.ego.position() : frame.obstacles[obstacle_id].position();
        if (has_previous) {
            const auto [x0, y0] = transform.map(previous);
            const auto [x1, y1] = transform.map(current);
            canvas.draw_line(x0, y0, x1, y1, thickness, color);
        }
        previous = current;
        has_previous = true;
    }
}

void draw_current_actor(IndexedCanvas& canvas, const Eigen::Vector2d& position,
                        double heading, const SvgTransform& transform,
                        int radius, std::uint8_t color) {
    const auto [x, y] = transform.map(position);
    canvas.draw_disk(static_cast<int>(std::lround(x)), static_cast<int>(std::lround(y)),
                     radius, color);
    canvas.draw_line(x, y, x + 2.2 * radius * std::cos(heading),
                     y - 2.2 * radius * std::sin(heading), 2, GIF_WHITE);
}

// Disc-space boundary segments (2 m), with a 0.35 m tick into a.dot(c) >= b.
// These are retained collision rows, not a projection of the entire QP feasible set.
bool constraint_glyph(const CollisionConstraint& row, Eigen::Vector2d& center,
                      Eigen::Vector2d& tangent, Eigen::Vector2d& inward,
                      const RolloutTraceFrame* frame = nullptr) {
    const double norm = row.a.norm();
    if (!std::isfinite(norm) || norm <= 0.0 || !std::isfinite(row.b) ||
        !row.linearization_point.allFinite()) return false;
    inward = row.a / norm;
    Eigen::Vector2d anchor = row.linearization_point;
    if (frame && row.obstacle_id >= 0 && static_cast<size_t>(row.obstacle_id) < frame->obstacles.size())
        anchor = frame->obstacles[row.obstacle_id].position();
    center = anchor + inward * ((row.b - row.a.dot(anchor)) / norm);
    tangent = Eigen::Vector2d(-inward.y(), inward.x());
    return center.allFinite();
}

// White is the first predicted stage; four gray bands represent later stages.
std::uint8_t constraint_stage_color(int k, int horizon) {
    if (k <= 1) return GIF_WHITE;
    return static_cast<std::uint8_t>(12 + std::min(3, 4 * (k - 2) / std::max(1, horizon - 1)));
}

std::optional<Eigen::Vector2d> predicted_constraint_disc(
    const RolloutTraceFrame& frame, const CollisionConstraint& row) {
    if (row.k < 0 || static_cast<size_t>(row.k) >= frame.predicted_ego.size()) return std::nullopt;
    const Eigen::Vector2d disc = compute_collision_disc_center(frame.predicted_ego[row.k], row);
    if (!disc.allFinite()) return std::nullopt;
    return disc;
}

bool is_support_constraint(
    const RolloutTraceFrame& frame,
    const CollisionConstraint& row)
{
    return std::find(
               frame.support_scenario_ids.begin(),
               frame.support_scenario_ids.end(),
               row.scenario_id)
           != frame.support_scenario_ids.end();
}

bool frame_decision_failed(
    const RolloutTrace& trace,
    const RolloutTraceFrame& frame)
{
    const auto it = std::find_if(
        trace.decisions.begin(),
        trace.decisions.end(),
        [&](const auto& decision) {
            return decision.step == frame.step;
        });

    return it != trace.decisions.end() &&
           !it->success;
}

int constraint_horizon(const RolloutTraceFrame& frame) {
    int horizon = 1;
    for (const auto& row : frame.linearized_constraints) horizon = std::max(horizon, row.k);
    return horizon;
}


// Display only: keep actual joint scenarios, suppress paths within 10 cm at
// every matching obstacle/stage. All support forecasts remain in the CSV.
std::vector<const Scenario*> preview_scenarios(const RolloutTraceFrame& frame, int limit) {
    std::vector<const Scenario*> preview;
    for (const auto& candidate : frame.sampled_scenarios) {
        if (!frame.show_support_scenarios) { preview.push_back(&candidate); continue; }
        bool duplicate = false;
        for (const auto* selected : preview) {
            bool same = candidate.trajectories.size() == selected->trajectories.size();
            for (const auto& [id, path] : candidate.trajectories) {
                const auto it = selected->trajectories.find(id);
                if (it == selected->trajectories.end() || it->second.steps.size() != path.steps.size()) {
                    same = false; break;
                }
                for (size_t k = 0; k < path.steps.size(); ++k) {
                    if ((path.steps[k].mean - it->second.steps[k].mean).norm() > .1) {
                        same = false; break;
                    }
                }
                if (!same) break;
            }
            if (same) { duplicate = true; break; }
        }
        if (!duplicate) preview.push_back(&candidate);
        if (preview.size() >= static_cast<size_t>(limit)) break;
    }
    return preview;
}

void render_gif_frame(IndexedCanvas& canvas, const RolloutTrace& trace,
                      std::size_t frame_index, const SvgTransform& transform, int support_preview_count) {
    canvas.clear(GIF_BACKGROUND);
    for (const auto& road : trace.road_centerlines) {
        draw_path_on_canvas(canvas, road, transform, 2, GIF_ROAD);
    }
    draw_path_on_canvas(canvas, trace.route, transform, 3, GIF_ROUTE);
    draw_actor_history(canvas, trace, frame_index, -1, transform, 2, GIF_EGO);

    const auto& frame = trace.frames[frame_index];

const bool failed_decision =
    frame.has_decision &&
    frame_decision_failed(trace, frame);

constexpr double kViolationDisplayTolerance = 1e-6;

    // A disappearing group fades over two recorded frames. These gray lines
    // retain the previous decision's exact geometry; they are not current rows.
    std::set<std::pair<int, int>> visible_groups;
    for (const auto& row : frame.linearized_constraints) {
        if (row.k == 1 && (!frame.show_support_scenarios || is_support_constraint(frame, row)))
            visible_groups.emplace(row.obstacle_id, row.disc_index);
    }
    if (!failed_decision) {
        for (size_t age = 1; age <= 2 && age <= frame_index; ++age) {
            const auto& prior = trace.frames[frame_index - age];
            std::set<std::pair<int, int>> fading_groups;
            for (const auto& row : prior.linearized_constraints) {
                const auto key = std::make_pair(row.obstacle_id, row.disc_index);
                if (row.k != 1 || visible_groups.count(key) ||
                    (frame.show_support_scenarios && !is_support_constraint(prior, row))) continue;
                Eigen::Vector2d center, tangent, inward;
                if (!constraint_glyph(row, center, tangent, inward, &prior)) continue;
                const auto [x0, y0] = transform.map(center - 1.5 * tangent);
                const auto [x1, y1] = transform.map(center + 1.5 * tangent);
                const auto [cx, cy] = transform.map(center);
                const auto [nx, ny] = transform.map(center + .35 * inward);
                const std::uint8_t color = age == 1 ? 14 : 15;
                canvas.draw_line(x0, y0, x1, y1, 1, color);
                canvas.draw_line(cx, cy, nx, ny, 1, color);
                fading_groups.insert(key);
            }
            visible_groups.insert(fading_groups.begin(), fading_groups.end());
        }
    }


        for (const auto& row : frame.linearized_constraints) {

            /*
            * Evaluate this exact retained halfspace against its matching
            * predicted ego disc.
            */
            const auto predicted_disc =
                predicted_constraint_disc(frame, row);

            double signed_clearance =
                std::numeric_limits<double>::infinity();

            if (predicted_disc) {
                signed_clearance =
                    row.evaluate(*predicted_disc);
            }

            const bool violated =
                predicted_disc.has_value() &&
                std::isfinite(signed_clearance) &&
                signed_clearance < -kViolationDisplayTolerance;

            /*
            * Normal animation:
            *     only show k=1.
            *
            * Failed MPC decision:
            *     additionally show violated future rows, regardless of k.
            */
            const bool show_normal_row =
                row.k == 1;

            const bool show_failed_row =
                failed_decision && violated;

            if (!show_normal_row &&
                !show_failed_row) {
                continue;
            }

            /*
            * In support-scenario mode, ordinary k=1 rows are restricted
            * to support scenarios. A violated row on a failed decision is
            * ALWAYS shown so that the actual failure cannot be hidden.
            */
            if (frame.show_support_scenarios &&
                !show_failed_row &&
                !is_support_constraint(frame, row)) {
                continue;
            }

            Eigen::Vector2d center;
            Eigen::Vector2d tangent;
            Eigen::Vector2d inward;

            if (!constraint_glyph(
                    row,
                    center,
                    tangent,
                    inward, &frame)) {
                continue;
            }

            const auto [x0, y0] =
                transform.map(center - 1.5 * tangent);

            const auto [x1, y1] =
                transform.map(center + 1.5 * tangent);

            const auto [cx, cy] =
                transform.map(center);

            const auto [nx, ny] =
                transform.map(
                    center + 0.35 * inward);

            /*
            * Red = the halfspace that the returned trajectory
            *       actually violates on a failed solve.
            *
            * Gray/white = ordinary immediately relevant constraint.
            */
            const std::uint8_t color =
                show_failed_row
                    ? static_cast<std::uint8_t>(GIF_COLLISION)
                    : frame.show_support_scenarios
                        ? static_cast<std::uint8_t>(
                            GIF_SUPPORT_HALFSPACE)
                        : static_cast<std::uint8_t>(
                            GIF_WHITE);

            const int thickness =
                show_failed_row ? 3 : 2;

            canvas.draw_line(
                x0, y0,
                x1, y1,
                thickness,
                color);

            canvas.draw_line(
                cx, cy,
                nx, ny,
                thickness,
                color);
        }
    const auto draw_forecasts = [&]() {
        for (const auto* scenario : preview_scenarios(frame, support_preview_count)) {
            for (const auto& [id, prediction] : scenario->trajectories) {
                const std::uint8_t color =
                frame.show_support_scenarios
                    ? static_cast<std::uint8_t>(GIF_SUPPORT)
                    : gif_obstacle_color(id);
                for (size_t k = 1; k < prediction.steps.size(); ++k) {
                    const auto [x0, y0] = transform.map(prediction.steps[k-1].mean);
                    const auto [x1, y1] = transform.map(prediction.steps[k].mean);
                    canvas.draw_line(x0, y0, x1, y1, 1, color);
                }

            }
        }
    };
    if (!frame.show_support_scenarios) draw_forecasts();
    for (std::size_t obstacle = 0; obstacle < frame.obstacles.size(); ++obstacle) {
        const std::uint8_t color = gif_obstacle_color(obstacle);
        draw_actor_history(canvas, trace, frame_index, static_cast<int>(obstacle),
                           transform, 1, color);
        const auto& state = frame.obstacles[obstacle];
        const double heading = std::atan2(state.vy, state.vx);
        draw_current_actor(canvas, state.position(), heading, transform, 5, color);
    }
    draw_current_actor(canvas, frame.ego.position(), frame.ego.theta, transform, 7, GIF_EGO);
    // Draw support forecasts last: short predictions must not disappear under actor markers.
        if (frame.show_support_scenarios) draw_forecasts();
    }

std::uint64_t scaled_trace_elapsed_centiseconds(
    const ExperimentArtifactConfig& artifacts, const RolloutTrace& trace,
    std::size_t frame_index
) {
    if (frame_index >= trace.frames.size()) {
        throw std::logic_error("GIF timing requested an invalid trace frame");
    }
    const double elapsed_seconds = trace.frames[frame_index].time_seconds -
        trace.frames.front().time_seconds;
    const double centiseconds = 100.0 * elapsed_seconds / artifacts.gif_playback_rate;
    if (!std::isfinite(centiseconds) || centiseconds < 0.0 ||
        centiseconds > static_cast<double>(std::numeric_limits<long long>::max())) {
        throw std::runtime_error("cannot derive a valid GIF timeline from rollout trace");
    }
    return static_cast<std::uint64_t>(std::llround(centiseconds));
}

void write_gif_frame_with_delay(
    gif::IndexedAnimationWriter<GIF_PALETTE.size()>& writer,
    const std::vector<std::uint8_t>& pixels, std::uint64_t delay_centiseconds
) {
    // GIF delays are 16-bit centiseconds. Repeating an identical image preserves
    // an arbitrarily long recorded pause instead of silently shortening it.
    do {
        const std::uint16_t chunk = static_cast<std::uint16_t>(
            std::min<std::uint64_t>(delay_centiseconds, 65535U));
        writer.write_frame(pixels, chunk);
        delay_centiseconds -= chunk;
    } while (delay_centiseconds != 0U);
}

void write_visualization_gif(const fs::path& path, const ExperimentConfig& config,
                             const RolloutTrace& trace) {
    std::ofstream out(path, std::ios::binary);
    require_open(out, path);
    gif::IndexedAnimationWriter<GIF_PALETTE.size()> writer(
        out, static_cast<std::uint16_t>(GIF_WIDTH), static_cast<std::uint16_t>(GIF_HEIGHT),
        GIF_PALETTE);
    IndexedCanvas canvas(GIF_WIDTH, GIF_HEIGHT, GIF_BACKGROUND);

    WorldBounds bounds;
    bounds.include(trace.route);
    for (const auto& road : trace.road_centerlines) bounds.include(road);
    bounds.include_trace(trace);
    bounds.pad();
    const SvgTransform transform = make_svg_transform(bounds, GIF_WIDTH, GIF_HEIGHT);

    if (trace.frames.empty()) {
        writer.write_frame(canvas.pixels(), 0);
        writer.finish();
        return;
    }

    std::vector<std::size_t> frame_indices;
    for (std::size_t index = 0; index < trace.frames.size();
         index += static_cast<std::size_t>(config.artifacts.gif_frame_stride)) {
        frame_indices.push_back(index);
    }
    if (frame_indices.back() != trace.frames.size() - 1) {
        frame_indices.push_back(trace.frames.size() - 1);
    }

    std::vector<std::uint64_t> frame_delays_centiseconds;
    frame_delays_centiseconds.reserve(frame_indices.size());
    for (std::size_t output_index = 0; output_index < frame_indices.size(); ++output_index) {
        const std::size_t trace_index = frame_indices[output_index];
        const std::uint64_t current_time = scaled_trace_elapsed_centiseconds(
            config.artifacts, trace, trace_index);
        const std::uint64_t next_time = output_index + 1 < frame_indices.size()
            ? scaled_trace_elapsed_centiseconds(
                config.artifacts, trace, frame_indices[output_index + 1])
            : current_time;
        if (next_time < current_time) {
            throw std::runtime_error("cannot create GIF from non-monotonic rollout trace timestamps");
        }
        frame_delays_centiseconds.push_back(next_time - current_time);
    }

    // A zero terminal GIF delay is commonly interpreted as a decoder-specific
    // default (often 100 ms). Reserve one centisecond for the final recorded
    // state and take it from the latest nonzero prior interval, so standard
    // decoders display the terminal state while the total loop duration remains
    // the recorded execution duration.
    bool terminal_tick_reserved = false;
    for (std::size_t index = frame_delays_centiseconds.size(); index > 1; --index) {
        std::uint64_t& prior_delay = frame_delays_centiseconds[index - 2];
        if (prior_delay != 0U) {
            --prior_delay;
            frame_delays_centiseconds.back() = 1U;
            terminal_tick_reserved = true;
            break;
        }
    }
    if (!terminal_tick_reserved) frame_delays_centiseconds.back() = 1U;
    // Hold failed terminal decisions long enough to inspect the
    // violated future halfspaces.
    if (!trace.frames.empty() &&
        frame_decision_failed(
            trace,
            trace.frames.back())) {

        constexpr std::uint64_t
            kFailureHoldCentiseconds = 200U;

        frame_delays_centiseconds.back() =
            kFailureHoldCentiseconds;
    }

    for (std::size_t output_index = 0; output_index < frame_indices.size(); ++output_index) {
        render_gif_frame(canvas, trace, frame_indices[output_index], transform,
                         config.artifacts.support_preview_count);
        write_gif_frame_with_delay(
            writer, canvas.pixels(), frame_delays_centiseconds[output_index]);
    }
    writer.finish();
}

void write_rviz_scene_csv(const fs::path& path, const RolloutTrace& trace) {
    std::ofstream out(path);
    require_open(out, path);
    out << std::setprecision(17);
    out << "kind,id,point_index,x,y,z\n";
    const auto write_path = [&](const char* kind, int id, const ReferencePath& reference) {
        for (std::size_t index = 0; index < reference.points().size(); ++index) {
            const auto& point = reference.points()[index].position;
            out << kind << ',' << id << ',' << index << ',' << point.x() << ','
                << point.y() << ",0\n";
        }
    };
    for (std::size_t road = 0; road < trace.road_centerlines.size(); ++road) {
        write_path("road", static_cast<int>(road), trace.road_centerlines[road]);
    }
    write_path("route", 0, trace.route);
}

void write_rviz_geometry_csv(const fs::path& path, const ExperimentConfig& config) {
    std::ofstream out(path);
    require_open(out, path);
    out << std::setprecision(17);
    out << "actor,radius,length,num_discs,safety_margin\n";
    out << "ego," << config.mpc.ego.radius << ',' << config.mpc.ego.length << ','
        << config.mpc.ego.num_discs << ',' << config.mpc.constraints.safety_margin << '\n';
    out << "obstacle," << config.obstacle_radius << ",0,1,"
        << config.mpc.constraints.safety_margin << '\n';
}

void write_rviz_config(const fs::path& path) {
    std::ofstream out(path);
    require_open(out, path);
    out << R"RVIZ(Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 1
      Class: rviz_default_plugins/Path
      Color: 87; 217; 163
      Enabled: true
      Line Style: Lines
      Line Width: 0.08
      Name: Reference route
      Topic:
        Depth: 5
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: dro_mpc/reference_path
    - Alpha: 1
      Class: rviz_default_plugins/Path
      Color: 88; 166; 255
      Enabled: true
      Line Style: Lines
      Line Width: 0.08
      Name: Ego trajectory
      Topic:
        Depth: 5
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: dro_mpc/ego_path
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Marker Topic:
        Depth: 10
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: dro_mpc/markers
      Name: Road and actors
  Enabled: true
  Global Options:
    Background Color: 16; 24; 32
    Fixed Frame: map
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
    - Class: rviz_default_plugins/SetGoal
Window Geometry:
  Height: 900
  Width: 1400
)RVIZ";
    if (!out) {
        throw std::runtime_error("failed while writing RViz configuration");
    }
}

void write_path_polyline(std::ofstream& out, const ReferencePath& path,
                         const SvgTransform& transform, const char* css_class) {
    if (path.points().empty()) return;
    out << "<polyline class=\"" << css_class << "\" points=\"";
    for (const auto& point : path.points()) {
        const auto [x, y] = transform.map(point.position);
        out << x << ',' << y << ' ';
    }
    out << "\"/>\n";
}

void write_trace_polyline(std::ofstream& out, const RolloutTrace& trace,
                          int obstacle_id, const SvgTransform& transform,
                          const std::string& css_class) {
    if (trace.frames.empty()) return;
    out << "<polyline class=\"" << css_class << "\" points=\"";
    for (const auto& frame : trace.frames) {
        const Eigen::Vector2d point = obstacle_id < 0
            ? frame.ego.position()
            : (obstacle_id < static_cast<int>(frame.obstacles.size())
                   ? frame.obstacles[obstacle_id].position()
                   : Eigen::Vector2d::Zero());
        const auto [x, y] = transform.map(point);
        out << x << ',' << y << ' ';
    }
    out << "\"/>\n";
}

void write_visualization_svg(const fs::path& path, const ExperimentConfig& config,
                             const RolloutRecord& record, const RolloutTrace& trace) {
    std::ofstream out(path);
    require_open(out, path);
    constexpr double width = 1200.0;
    constexpr double height = 780.0;
    WorldBounds bounds;
    bounds.include(trace.route);
    for (const auto& road : trace.road_centerlines) bounds.include(road);
    bounds.include_trace(trace);
    bounds.pad();
    const SvgTransform transform = make_svg_transform(bounds, width, height);

    out << std::setprecision(10);
    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << ' '
        << height << "\">\n"
           "<rect width=\"100%\" height=\"100%\" fill=\"#101820\"/>\n"
           "<style>text{font-family:system-ui,sans-serif;fill:#e6edf3;font-size:16px}"
           ".road{fill:none;stroke:#637381;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}"
           ".route{fill:none;stroke:#57d9a3;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}"
           ".ego{fill:none;stroke:#58a6ff;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}"
           ".obs0{fill:none;stroke:#ff8c42;stroke-width:3}.obs1{fill:none;stroke:#f778ba;stroke-width:3}"
           ".obs2{fill:none;stroke:#d29922;stroke-width:3}.obs3{fill:none;stroke:#a371f7;stroke-width:3}"
           ".marker{stroke:#101820;stroke-width:2}.legend{font-size:14px}</style>\n";
    out << "<text x=\"30\" y=\"32\">"
        << xml_escape(record.method) << " / " << xml_escape(record.scenario)
        << " / seed " << record.seed << "</text>\n";
    for (const auto& road : trace.road_centerlines) write_path_polyline(out, road, transform, "road");
    write_path_polyline(out, trace.route, transform, "route");
    if (config.artifacts.show_linearized_constraints && !config.artifacts.show_support_scenarios) {
        for (auto frame = trace.frames.rbegin(); frame != trace.frames.rend(); ++frame) {
            if (!frame->has_decision) continue;
            out << "<g id=\"linearized-constraints\" stroke=\"#e6edf3\" stroke-width=\"2\">\n";
            const int stage_horizon = constraint_horizon(*frame);
            for (const auto& row : frame->linearized_constraints) {
                Eigen::Vector2d center, tangent, inward;
                if (!constraint_glyph(row, center, tangent, inward, &*frame)) continue;
                const auto [x0, y0] = transform.map(center - 1.5 * tangent);
                const auto [x1, y1] = transform.map(center + 1.5 * tangent);
                const auto [cx, cy] = transform.map(center);
                const auto [nx, ny] = transform.map(center + 0.35 * inward);
                const auto color = GIF_PALETTE[constraint_stage_color(row.k, stage_horizon)];
                const std::string shade = "rgb(" + std::to_string(color.red) + "," +
                    std::to_string(color.green) + "," + std::to_string(color.blue) + ")";
                out << "<path stroke=\"" << shade << "\" d=\"M " << x0 << ' ' << y0 << " L " << x1 << ' ' << y1
                    << " M " << cx << ' ' << cy << " L " << nx << ' ' << ny << "\"><title>decision "
                    << frame->step << ", horizon " << row.k << ", disc " << row.disc_index
                    << ", obstacle " << row.obstacle_id << ", scenario " << row.scenario_id << "</title></path>\n";

            }
            out << "</g><text x=\"30\" y=\"55\" class=\"legend\">White: k=0/1; darker gray: later stages. Disc coordinates/residuals: CSV. Decision "
                << frame->step << "</text><text x=\"30\" y=\"75\" class=\"legend\">Blue: executed history; compare each boundary only to its same-stage predicted disc. Ticks: feasible side.</text>\n";
            if (frame->predicted_ego.empty())
                out << "<text x=\"30\" y=\"95\" class=\"legend\">No returned predicted trajectory; disc evaluation unavailable.</text>\n";
            break;
        }
    }
    if (config.artifacts.show_support_scenarios) {
        for (auto frame = trace.frames.rbegin(); frame != trace.frames.rend(); ++frame) {
            if (!frame->has_decision) continue;
            out << "<g id=\"support-scenarios\" fill=\"none\" stroke=\"#00e5ff\" stroke-width=\"1\">\n";
            const auto preview = preview_scenarios(*frame, config.artifacts.support_preview_count);
            for (const auto* scenario : preview) {
                for (const auto& [id, prediction] : scenario->trajectories) {
                    out << "<polyline points=\"";
                    for (const auto& step : prediction.steps) {
                        const auto [x, y] = transform.map(step.mean);
                        out << x << ',' << y << ' ';
                    }
                    out << "\"><title>support scenario " << scenario->scenario_id
                        << ", obstacle " << id << "</title></polyline>\n";
                }
            }
            out << "</g><text x=\"30\" y=\"55\" class=\"legend\">Cyan: support forecasts: "
                << preview.size() << " displayed / " << frame->support_scenario_ids.size() << " total; decision " << frame->step
                << (frame->support_evaluated ? " (SQP support estimate)" : " (support not evaluated)")
                << "</text>\n";
            break;
        }
    }
    write_trace_polyline(out, trace, -1, transform, "ego");
    const std::array<const char*, 4> obstacle_classes = {"obs0", "obs1", "obs2", "obs3"};
    const std::size_t obstacle_count = trace.frames.empty() ? 0 : trace.frames.front().obstacles.size();
    for (std::size_t obstacle = 0; obstacle < obstacle_count; ++obstacle) {
        write_trace_polyline(out, trace, static_cast<int>(obstacle), transform,
                             obstacle_classes[obstacle % obstacle_classes.size()]);
    }
    if (!trace.frames.empty()) {
        const auto& first = trace.frames.front();
        const auto& last = trace.frames.back();
        const auto [first_x, first_y] = transform.map(first.ego.position());
        const auto [last_x, last_y] = transform.map(last.ego.position());
        out << "<circle class=\"marker\" cx=\"" << first_x << "\" cy=\"" << first_y
            << "\" r=\"6\" fill=\"#58a6ff\"/>\n"
            << "<circle class=\"marker\" cx=\"" << last_x << "\" cy=\"" << last_y
            << "\" r=\"7\" fill=\"" << (record.collision ? "#f85149" : "#57d9a3")
            << "\"/>\n";
    }
    out << "<text class=\"legend\" x=\"30\" y=\"" << height - 45
        << "\">green: route   blue: ego   orange/pink/gold/purple: obstacles"
           "   result: " << (record.collision ? "collision" : "no collision")
        << "</text>\n</svg>\n";
}

}  // namespace

std::string write_rollout_artifacts(
    const ExperimentConfig& config,
    const RolloutRecord& record,
    const SeedBundle& seeds,
    const RolloutTrace& trace
) {
    if (!config.artifacts.enabled()) return {};

    const std::string default_name = safe_path_component(record.method) + "_" +
        safe_path_component(record.scenario) + "_seed_" + std::to_string(record.seed);
    const std::string run_name = safe_path_component(
        config.artifacts.run_name.empty() ? default_name : config.artifacts.run_name);
    const fs::path run_directory = fs::path(config.artifacts.output_directory) / run_name;
    std::error_code error;
    fs::create_directories(run_directory, error);
    if (error) {
        throw std::runtime_error("could not create experiment artifact directory '" +
                                 run_directory.string() + "' (" + error.message() + ")");
    }
    const std::string artifact_directory = run_directory.lexically_normal().string();
    RolloutRecord artifact_record = record;
    artifact_record.artifact_directory = artifact_directory;

    if (config.artifacts.write_reproducibility_manifest) {
        write_manifest(run_directory / "reproducibility.yaml", config, artifact_record, seeds);
        std::ofstream resolved_config(run_directory / "resolved_config.yaml");
        require_open(resolved_config, run_directory / "resolved_config.yaml");
        resolved_config << std::setprecision(17);
        write_resolved_config(resolved_config, config);
        if (config.config_source != "in_memory" && fs::exists(config.config_source)) {
            fs::copy_file(config.config_source, run_directory / "source_config.yaml",
                          fs::copy_options::overwrite_existing, error);
            if (error) {
                throw std::runtime_error("could not copy source config into artifact bundle: " +
                                         error.message());
            }
        }
    }
    {
        // Always replace this file, including with an empty preview when disabled.
        // Reusing a bundle must not expose forecasts from an earlier rollout.
        std::ofstream samples(run_directory / "sampled_scenarios.csv");
        require_open(samples, run_directory / "sampled_scenarios.csv");
        std::ofstream summary(run_directory / "sampled_scenario_summary.csv");
        require_open(summary, run_directory / "sampled_scenario_summary.csv");
        summary << "step,scenario_count,max_sample_deviation\n" << std::setprecision(17);
        for (const auto& frame : trace.frames)
            if (frame.scenario_count > 0)
                summary << frame.step << ',' << frame.scenario_count << ','
                        << frame.max_sample_deviation << '\n';
        samples << "step,obstacle_id,scenario_id,horizon_step,x,y\n" << std::setprecision(17);
        for (const auto& frame : trace.frames)
            for (const auto& scenario : frame.sampled_scenarios)
                for (const auto& [id, prediction] : scenario.trajectories)
                    for (size_t k = 0; k < prediction.steps.size(); ++k)
                        samples << frame.step << ',' << id << ',' << scenario.scenario_id << ','
                                << k << ',' << prediction.steps[k].mean.x() << ','
                                << prediction.steps[k].mean.y() << '\n';
    }
    {
        const auto path = run_directory / "support_scenarios.csv";
        std::ofstream out(path);
        require_open(out, path);
        out << "step,support_evaluated,support_count,scenario_ids\n";
        for (const auto& frame : trace.frames) {
            if (!frame.has_decision) continue;
            out << frame.step << ',' << frame.support_evaluated << ',' << frame.support_scenario_ids.size() << ',';
            for (size_t i = 0; i < frame.support_scenario_ids.size(); ++i) {
                if (i) out << ';';
                out << frame.support_scenario_ids[i];
            }
            out << '\n';
        }
    }
    if (config.artifacts.write_analysis_csv) {
        if (config.artifacts.capture_attempt_diagnostics) {
            std::ofstream attempts(run_directory / "attempts.csv");
            std::ofstream mechanism(run_directory / "mode_mechanism.csv");
            std::ofstream transport(run_directory / "transport_costs.csv");
            require_open(attempts, run_directory / "attempts.csv");
            require_open(mechanism, run_directory / "mode_mechanism.csv");
            require_open(transport, run_directory / "transport_costs.csv");
            transport << std::setprecision(17)
                << "step,attempt,obstacle_id,source_mode,target_mode,cost,radius_observation_count\n";
            attempts << std::setprecision(17)
                << "step,attempt,success,dro_enabled,solve_ms,scenario_count,qp_calls\n";
            mechanism << std::setprecision(17)
                << "step,attempt,obstacle_id,mode,nominal_probability,sampling_probability,risk_score,rho,sampled_count,scenario_count,true_mode\n";
            for (const auto& decision : trace.decisions) {
                for (size_t i = 0; i < decision.attempts.size(); ++i) {
                    const auto& a = decision.attempts[i];
                    attempts << decision.step << ',' << i << ',' << a.success << ',' << a.dro_enabled
                        << ',' << 1000*a.elapsed_seconds << ',' << a.sampled_scenarios << ',' << a.qp_calls << '\n';
                    for (const auto& [id, weights] : a.nominal_weights) {
                        if (a.transport_costs.count(id)) {
                            size_t source = 0;
                            for (const auto& source_weight : weights) {
                                size_t target = 0;
                                for (const auto& target_weight : weights) {
                                    transport << decision.step << ',' << i << ',' << id << ','
                                        << source_weight.first << ',' << target_weight.first << ','
                                        << a.transport_costs.at(id).at(source).at(target) << ','
                                        << a.radius_observation_counts.at(id) << '\n';
                                    ++target;
                                }
                                ++source;
                            }
                        }
                        for (const auto& [mode, probability] : weights) {
                            mechanism << decision.step << ',' << i << ',' << id << ',' << mode << ','
                                << probability << ',' << a.sampling_weights.at(id).at(mode) << ',';
                            if (a.risk_scores.count(id)) mechanism << a.risk_scores.at(id).at(mode);
                            mechanism << ',';
                            if (a.radii.count(id)) mechanism << a.radii.at(id);
                            int count = 0;
                            if (a.initial_mode_counts.count(id) && a.initial_mode_counts.at(id).count(mode))
                                count = a.initial_mode_counts.at(id).at(mode);
                            mechanism << ',' << count << ',' << a.sampled_scenarios << ',';
                            for (const auto& coverage : decision.mode_coverage)
                                if (coverage.obstacle_id == id) mechanism << coverage.true_mode;
                            mechanism << '\n';
                        }
                    }
                }
            }
        }
        const auto decisions_path = run_directory / "decisions.csv";
        const auto coverage_path = run_directory / "mode_coverage.csv";
        std::ofstream decisions(decisions_path), coverage(coverage_path);
        require_open(decisions, decisions_path);
        require_open(coverage, coverage_path);
        decisions << std::setprecision(17)
            << "step,solve_ms,success,certificate_requested,certified,applied_control_effort,scenario_count,backup_available,backup_removal_budget_exceeded,backup_dro_failed,braking_collision_feasible,any_homotopy_geometrically_feasible,last_qp_converged,sqp_sampled_collision_feasible,fallback_sampled_collision_feasible,nominal_fallback_attempted,used_nominal_fallback\n";
        coverage << "step,obstacle_id,class_id,true_mode,sampled_modes,represented,scenario_count\n";
        for (const auto& decision : trace.decisions) {
            decisions << decision.step << ',' << decision.solve_ms << ',' << decision.success << ','
                << decision.certificate_requested << ',' << decision.certified << ','
                << decision.applied_control_effort << ',' << decision.scenario_count
                << ',' << decision.failure_diagnostics.backup_available
                << ',' << decision.failure_diagnostics.backup_removal_budget_exceeded
                << ',' << decision.failure_diagnostics.backup_dro_failed
                << ',' << decision.failure_diagnostics.braking_collision_feasible
                << ',' << decision.failure_diagnostics.any_homotopy_geometrically_feasible
                << ',' << decision.failure_diagnostics.last_qp_converged
                << ',' << decision.failure_diagnostics.sqp_sampled_collision_feasible
                << ',' << decision.failure_diagnostics.fallback_sampled_collision_feasible
                << ',' << decision.nominal_fallback_attempted
                << ',' << decision.used_nominal_fallback
                << '\n';
            for (const auto& item : decision.mode_coverage) {
                coverage << decision.step << ',' << item.obstacle_id << ',' << item.class_id << ','
                    << item.true_mode << ',';
                for (size_t i = 0; i < item.sampled_modes.size(); ++i) {
                    if (i) coverage << ';';
                    coverage << item.sampled_modes[i];
                }
                coverage << ',' << item.represented << ',' << decision.scenario_count << '\n';
            }
        }
        write_rviz_geometry_csv(run_directory / "geometry.csv", config);
    }
    if (config.artifacts.write_trace_csv) {
        write_trace_csv(run_directory / "trace.csv", trace);
    }
    if (config.artifacts.show_linearized_constraints) {
        const auto path = run_directory / "linearized_constraints.csv";
        std::ofstream out(path);
        require_open(out, path);
        out << std::setprecision(17)
            << "step,time_s,horizon_step,obstacle_id,scenario_id,disc_index,disc_offset,a_x,a_y,b,anchor_x,anchor_y,predicted_disc_x,predicted_disc_y,geometric_residual\n";
        for (const auto& frame : trace.frames) {
            for (const auto& row : frame.linearized_constraints) {
                out << frame.step << ',' << frame.time_seconds << ',' << row.k << ','
                    << row.obstacle_id << ',' << row.scenario_id << ',' << row.disc_index << ','
                    << row.disc_offset << ',' << row.a.x() << ',' << row.a.y() << ',' << row.b << ','
                    << row.linearization_point.x() << ',' << row.linearization_point.y() << ',';
                const auto disc = predicted_constraint_disc(frame, row);
                if (disc) out << disc->x() << ',' << disc->y() << ',' << row.evaluate(*disc);
                else out << ",,";
                out << '\n';
            }
        }
    }
    if (config.artifacts.write_visualization_svg) {
        write_visualization_svg(run_directory / "rollout.svg", config, artifact_record, trace);
    }
    if (config.artifacts.write_visualization_gif) {
        write_visualization_gif(run_directory / "rollout.gif", config, trace);
    }
    if (config.artifacts.write_rviz_replay_bundle) {
        write_rviz_scene_csv(run_directory / "scene.csv", trace);
        write_rviz_geometry_csv(run_directory / "geometry.csv", config);
        write_rviz_config(run_directory / "rollout.rviz");
    }
    CSVWriter record_writer((run_directory / "rollout.csv").string());
    record_writer.write_header();
    record_writer.write_record(artifact_record);
    record_writer.flush();
    return artifact_directory;
}

}  // namespace detail
}  // namespace dro_mpc
