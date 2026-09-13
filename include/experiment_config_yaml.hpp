/**
 * @file experiment_config_yaml.hpp
 * @brief Load an ExperimentConfig from a flat `key: value` YAML test-config file.
 *
 * Self-contained (no yaml-cpp dependency): parses a flat key/value list with
 * `#` comments, quoted strings, and comma-separated lists.
 *
 * load_experiment_config(path) starts from configs/default.yaml, then overlays
 * `path`. Unknown keys are ignored unless `strict`.
 *
 *   ExperimentConfig cfg = yaml_config::load_experiment_config("my_test.yaml");
 *   RolloutRecord rec = run_experiment_rollout(cfg, seed);
 */
#ifndef DRO_MPC_EXPERIMENT_CONFIG_YAML_HPP
#define DRO_MPC_EXPERIMENT_CONFIG_YAML_HPP

#include "experiment_harness.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dro_mpc {
namespace yaml_config {

inline std::string trim(std::string s) {
    auto notspace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
    return s;
}
inline std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
inline bool to_bool(const std::string& v) {
    const std::string l = lower(trim(v));
    if (l == "true" || l == "1" || l == "yes" || l == "on") return true;
    if (l == "false" || l == "0" || l == "no" || l == "off") return false;
    throw std::invalid_argument("expected a boolean value");
}
inline std::vector<std::string> split_csv(const std::string& v) {
    std::vector<std::string> out;
    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (!tok.empty() && (tok.front() == '"' || tok.front() == '\'')) tok = tok.substr(1, tok.size() - 2);
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}
inline std::vector<double> split_csv_d(const std::string& v) {
    std::vector<double> out;
    for (const auto& s : split_csv(v)) out.push_back(std::stod(s));
    return out;
}
inline std::vector<ObstacleState> parse_obstacle_states(const std::string& v) {
    std::vector<ObstacleState> states;
    std::stringstream groups(v);
    std::string group;
    while (std::getline(groups, group, ';')) {
        group = trim(group);
        while (group.size() >= 2 && group.front() == '[' && group.back() == ']') {
            group = trim(group.substr(1, group.size() - 2));
        }
        if (group.empty()) continue;
        const auto values = split_csv_d(group);
        if (values.size() != 2 && values.size() != 4) {
            throw std::invalid_argument(
                "each obstacle start must be x,y or x,y,vx,vy");
        }
        const double vx = values.size() == 4 ? values[2] : 0.0;
        const double vy = values.size() == 4 ? values[3] : 0.0;
        states.emplace_back(values[0], values[1], vx, vy);
    }
    return states;
}

inline std::vector<Eigen::Vector2d> parse_path_waypoints(const std::string& v) {
    std::vector<Eigen::Vector2d> waypoints;
    std::stringstream groups(v);
    std::string group;
    while (std::getline(groups, group, ';')) {
        group = trim(group);
        while (group.size() >= 2 && group.front() == '[' && group.back() == ']') {
            group = trim(group.substr(1, group.size() - 2));
        }
        if (group.empty()) continue;
        const auto values = split_csv_d(group);
        if (values.size() != 2) {
            throw std::invalid_argument(
                "each path waypoint must be x,y");
        }
        waypoints.emplace_back(values[0], values[1]);
    }
    return waypoints;
}

inline std::string strip_comment(const std::string& line) {
    bool in_single_quote = false;
    bool in_double_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\'' && !in_double_quote) in_single_quote = !in_single_quote;
        if (line[i] == '"' && !in_single_quote) in_double_quote = !in_double_quote;
        if (line[i] == '#' && !in_single_quote && !in_double_quote)
            return line.substr(0, i);
    }
    return line;
}

inline size_t find_mapping_colon(const std::string& line) {
    bool in_single_quote = false;
    bool in_double_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\'' && !in_double_quote) in_single_quote = !in_single_quote;
        if (line[i] == '"' && !in_single_quote) in_double_quote = !in_double_quote;
        if (line[i] == ':' && !in_single_quote && !in_double_quote)
            return i;
    }
    return std::string::npos;
}

inline MPCType parse_mpc(const std::string& v) {
    std::string l = lower(v);
    if (l == "mpc")                          return MPCType::MPC;
    if (l == "mpcc")                         return MPCType::MPCC;
    if (l == "sh_mpc" || l == "sh-mpc")     return MPCType::SH_MPC;
    if (l == "sh_mpcc" || l == "sh-mpcc")   return MPCType::SH_MPCC;
    throw std::invalid_argument("unknown mpc_type '" + v + "'");
}
inline EnvironmentType parse_env(const std::string& v) {
    std::string l = lower(v);
    if (l == "t_intersection" || l == "t-intersection") return EnvironmentType::T_INTERSECTION;
    if (l == "four_way_intersection" || l == "fourway") return EnvironmentType::FOUR_WAY_INTERSECTION;
    if (l == "s_curve" || l == "scurve") return EnvironmentType::S_CURVE;
    if (l == "two_lane_roundabout") return EnvironmentType::TWO_LANE_ROUNDABOUT;
    if (l == "four_lane_roundabout") return EnvironmentType::FOUR_LANE_ROUNDABOUT;
    if (l == "two_lane_highway") return EnvironmentType::TWO_LANE_HIGHWAY;
    if (l == "four_lane_highway") return EnvironmentType::FOUR_LANE_HIGHWAY;
    if (l == "enter_ramp") return EnvironmentType::ENTER_RAMP;
    if (l == "exit_ramp") return EnvironmentType::EXIT_RAMP;
    if (l == "overtake" || l == "overtake_slow_lead") return EnvironmentType::OVERTAKE_SLOW_LEAD;
    if (l == "narrow"   || l == "narrow_corridor")    return EnvironmentType::NARROW_CORRIDOR;
    if (l == "intersection")                          return EnvironmentType::INTERSECTION;
    if (l == "oncoming")                              return EnvironmentType::ONCOMING;
    throw std::invalid_argument("unknown environment '" + v + "'");
}
inline ModeSwitchConfiguration parse_switch(const std::string& v) {
    std::string l = lower(v);
    if (l == "hold" || l == "hold_over_horizon") return ModeSwitchConfiguration::HOLD_OVER_HORIZON;
    if (l == "markov" || l == "markov_jump_system")
        return ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM;
    throw std::invalid_argument("unknown switch_regime '" + v + "'");
}
inline ObstacleHistoryConfiguration parse_obstacle_history(const std::string& v) {
    const std::string l = lower(v);
    if (l == "shared" || l == "shared_history" ||
        l == "shared_history_classes") {
        return ObstacleHistoryConfiguration::SHARED;
    }
    if (l == "independent" || l == "independent_history") {
        return ObstacleHistoryConfiguration::INDEPENDENT;
    }
    throw std::invalid_argument("unknown obstacle_history '" + v + "'");
}
inline NominalBeliefKind parse_belief(const std::string& v) {
    const std::string l = lower(v);
    if (l == "sticky") return NominalBeliefKind::STICKY;
    if (l == "dirichlet") return NominalBeliefKind::DIRICHLET;
    throw std::invalid_argument("unknown belief_kind '" + v + "'");
}
inline DROGroundCostType parse_ground(const std::string& v) {
    std::string l = lower(v);
    if (l == "w2_bures" || l == "w2-bures" || l == "w2")
                                      return DROGroundCostType::W2_BURES;
    if (l == "w1_metric" || l == "w1-metric")
                                      return DROGroundCostType::W1_METRIC;
    if (l == "zero_one" || l == "zero-one")
                                      return DROGroundCostType::ZERO_ONE;
    if (l == "euclidean_mean" || l == "euclidean-mean")
                                      return DROGroundCostType::EUCLIDEAN_MEAN;
    throw std::invalid_argument("unknown ground_cost '" + v + "'");
}
inline DRORiskMeasure parse_risk(const std::string& v) {
    std::string l = lower(v);
    if (l == "surrogate_var")            return DRORiskMeasure::SURROGATE_VAR;
    if (l == "surrogate_cvar")           return DRORiskMeasure::SURROGATE_CVAR;
    if (l == "surrogate_var_bonferroni") return DRORiskMeasure::SURROGATE_VAR_BONFERRONI;
    if (l == "bonferroni_var" || l == "proper_bonferroni_var")
                                         return DRORiskMeasure::BONFERRONI_VAR;
    if (l == "mixture_var")              return DRORiskMeasure::MIXTURE_VAR;
    if (l == "mixture_cvar")             return DRORiskMeasure::MIXTURE_CVAR;
    if (l == "joint_var")                return DRORiskMeasure::JOINT_VAR;
    if (l == "joint_cvar")               return DRORiskMeasure::JOINT_CVAR;
    throw std::invalid_argument("unknown risk_measure '" + v + "'");
}
inline DRORiskScoringModel parse_risk_scoring_model(const std::string& v) {
    const std::string l = lower(v);
    if (l == "inherit" || l == "inherit_risk_measure") {
        return DRORiskScoringModel::INHERIT_RISK_MEASURE;
    }
    if (l == "certified_surrogate" || l == "surrogate_bonferroni") {
        return DRORiskScoringModel::CERTIFIED_SURROGATE;
    }
    if (l == "euclidean_bonferroni" || l == "euclidean_bonferroni_var") {
        return DRORiskScoringModel::EUCLIDEAN_BONFERRONI_VAR;
    }
    if (l == "euclidean_joint_var" || l == "joint_var") {
        return DRORiskScoringModel::EUCLIDEAN_JOINT_VAR;
    }
    if (l == "euclidean_joint_cvar" || l == "joint_cvar" ||
        l == "euclidean_joint") {
        return DRORiskScoringModel::EUCLIDEAN_JOINT_CVAR;
    }
    throw std::invalid_argument("unknown risk_scoring_model '" + v + "'");
}
inline AmbiguityDivergence parse_divergence(const std::string& v) {
    std::string l = lower(v);
    if (l == "wasserstein" || l == "wass")             return AmbiguityDivergence::WASSERSTEIN;
    if (l == "total_variation" || l == "tv")             return AmbiguityDivergence::TOTAL_VARIATION;
    if (l == "kullback_leibler" || l == "kl")            return AmbiguityDivergence::KULLBACK_LEIBLER;
    if (l == "jensen_shannon" || l == "js")              return AmbiguityDivergence::JENSEN_SHANNON;
    if (l == "hellinger")                                return AmbiguityDivergence::HELLINGER;
    throw std::invalid_argument("unknown divergence '" + v + "'");
}

/// Search order: compile-time DRO_MPC_DEFAULT_CONFIG, env var, cwd-relative paths.
inline std::string default_config_path() {
#ifdef DRO_MPC_DEFAULT_CONFIG
    {
        std::ifstream in(DRO_MPC_DEFAULT_CONFIG);
        if (in) return DRO_MPC_DEFAULT_CONFIG;
    }
#endif
    if (const char* env = std::getenv("DRO_MPC_DEFAULT_CONFIG")) {
        std::ifstream in(env);
        if (in) return env;
    }
    const char* candidates[] = {
        "configs/default.yaml",
        "../configs/default.yaml",
        "../../configs/default.yaml",
        "default.yaml"
    };
    for (const char* p : candidates) {
        std::ifstream in(p);
        if (in) return p;
    }
    return "configs/default.yaml";
}

inline void apply_yaml_file(ExperimentConfig& cfg, const std::string& path, bool strict) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("load_experiment_config: cannot open " + path);

    std::string line;
    int lineno = 0;
    bool saw_num_scenarios = false;
    bool saw_automatic_sample_sizing = false;
    while (std::getline(in, line)) {
        ++lineno;
        line = strip_comment(line);
        line = trim(line);
        if (line.empty()) continue;
        auto colon = find_mapping_colon(line);
        if (colon == std::string::npos) {
            if (strict) throw std::runtime_error("malformed YAML entry");
            continue;
        }

        const std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));
        if (val.size() >= 2 && (val.front() == '"' || val.front() == '\'') && val.back() == val.front())
            val = val.substr(1, val.size() - 2);
        if (val.size() >= 2 && val.front() == '[' && val.back() == ']')
            val = val.substr(1, val.size() - 2);
        const std::string k = lower(key);

        try {
            if      (k == "mpc_type")                      { cfg.mpc.type = parse_mpc(val); cfg.mpc.sync_from_type(); }
            else if (k == "progress_weight")               cfg.mpc.objective.progress_weight = std::stod(val);
            else if (k == "horizon")                       cfg.mpc.horizon = std::stoi(val);
            else if (k == "dt")                            cfg.mpc.dt = std::stod(val);
            else if (k == "num_scenarios") {
                cfg.mpc.sampling.num_scenarios = std::stoi(val);
                saw_num_scenarios = true;
            }
            else if (k == "road_width")                    cfg.mpc.constraints.road_width = std::stod(val);
            else if (k == "safety_margin")                 cfg.mpc.constraints.safety_margin = std::stod(val);
            else if (k == "enable_contouring_constraints") cfg.mpc.enable_contouring_constraints = to_bool(val);
            else if (k == "safe_horizon_enabled" || k == "enable_safe_horizon")
                                                           cfg.mpc.safe_horizon_enabled = to_bool(val);
            else if (k == "enable_velocity_bounds")        cfg.mpc.constraints.enable_velocity_bounds = to_bool(val);
            else if (k == "support_cap_n_bar" || k == "n_bar")
                                                           cfg.mpc.constraints.support_cap_n_bar = std::stoi(val);
            else if (k == "scenario_removal_budget")       cfg.mpc.constraints.scenario_removal_budget = std::stoi(val);
            else if (k == "safe_horizon_min" || k == "safe_horizon_mode" ||
                     k == "forced_safe_horizon")
                throw std::invalid_argument(
                    "temporal safe-horizon truncation was removed; Safe Horizon certifies the full MPC horizon");
            else if (k == "support_cap_nbar")
                throw std::invalid_argument("renamed to support_cap_n_bar");
            else if (k == "enable_scenario_removal" || k == "scenario_removal_enabled")
                throw std::invalid_argument(
                    "removed because online scenario removal is not implemented; use scenario_removal_budget for conservative certification");
            else if (k == "scenario_removal_count" || k == "removal_count")
                throw std::invalid_argument("renamed to scenario_removal_budget");
            else if (k == "clearance_filter_distance")     cfg.mpc.constraints.clearance_filter_distance = std::stod(val);
            else if (k == "ego_radius")                    cfg.mpc.ego.radius = std::stod(val);
            else if (k == "ego_length")                    cfg.mpc.ego.length = std::stod(val);
            else if (k == "num_discs")                     cfg.mpc.ego.num_discs = std::stoi(val);
            else if (k == "max_velocity")                  cfg.mpc.ego.dynamics.max_velocity = std::stod(val);
            else if (k == "min_velocity")                  cfg.mpc.ego.dynamics.min_velocity = std::stod(val);
            else if (k == "max_acceleration")              cfg.mpc.ego.dynamics.max_acceleration = std::stod(val);
            else if (k == "min_acceleration")              cfg.mpc.ego.dynamics.min_acceleration = std::stod(val);
            else if (k == "max_omega" || k == "max_steering_rate")
                                                           cfg.mpc.ego.dynamics.max_omega = std::stod(val);
            else if (k == "belief_kind")                   cfg.mpc.sampling.belief_kind = parse_belief(val);
            else if (k == "self_persistence_prior")        cfg.mpc.sampling.mode_belief.self_persistence_prior = std::stod(val);
            else if (k == "markov_jump_system") {
                const bool enabled = to_bool(val);
                cfg.mpc.sampling.markov_jump_system = enabled;
                cfg.obstacles.switch_regime = enabled
                    ? ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM
                    : ModeSwitchConfiguration::HOLD_OVER_HORIZON;
            }
            else if (k == "compute_automatically" ||
                     k == "automatically_compute_sample_size" ||
                     k == "auto_compute_sample_size")
            {
                cfg.mpc.sampling.automatically_compute_sample_size = to_bool(val);
                saw_automatic_sample_sizing = true;
            }
            // Compatibility for old overlays: its prior effect was to make
            // the controller fill an undersized sample set, which is exactly
            // the reference automatic-sizing behavior.
            else if (k == "enforce_certified_scenario_count" || k == "enforce_scenario_count")
            {
                cfg.mpc.sampling.automatically_compute_sample_size = to_bool(val);
                saw_automatic_sample_sizing = true;
            }
            else if (k == "max_history_length")            cfg.mpc.sampling.max_history_length = std::stoi(val);
            else if (k == "one_minus_chance_constraint_violation_probability" || k == "confidence_level")
                                                           cfg.mpc.sampling.one_minus_chance_constraint_violation_probability = std::stod(val);
            else if (k == "chance_of_certificate_violation" || k == "beta")
                                                           cfg.mpc.sampling.chance_of_certificate_violation = std::stod(val);
            else if (k == "dro_enabled")                   cfg.dro.enabled = to_bool(val);
            else if (k == "fixed_rho")                     cfg.dro.fixed_rho = std::stod(val);
            else if (k == "risk_measure")                  cfg.dro.solver.radius_calibration.risk_measure = parse_risk(val);
            else if (k == "risk_scoring_model" || k == "risk_scoring")
                                                           cfg.dro.solver.radius_calibration.risk_scoring_model = parse_risk_scoring_model(val);
            else if (k == "risk_horizon")                  cfg.dro.solver.radius_calibration.risk_horizon = std::stoi(val);
            else if (k == "divergence")                    cfg.dro.solver.radius_calibration.divergence = parse_divergence(val);
            else if (k == "ground_cost")                   cfg.dro.solver.ground_cost_type = parse_ground(val);
            else if (k == "confidence_beta")               cfg.dro.solver.radius_calibration.confidence_beta = std::stod(val);
            else if (k == "calibration_scale")             cfg.dro.solver.radius_calibration.calibration_scale = std::stod(val);
            else if (k == "alpha_one_sided")               cfg.dro.solver.radius_calibration.alpha_one_sided = std::stod(val);
            else if (k == "use_calibrated_radius")         cfg.dro.solver.radius_calibration.use_calibrated_radius = to_bool(val);
            else if (k == "use_primal_ot")                 cfg.dro.solver.radius_calibration.use_primal_ot = to_bool(val);
            else if (k == "min_radius")                    cfg.dro.solver.min_radius = std::stod(val);
            else if (k == "max_radius")                    cfg.dro.solver.max_radius = std::stod(val);
            else if (k == "base_radius")                   cfg.dro.solver.base_radius = std::stod(val);
            else if (k == "use_entropic_allocator")        cfg.dro.solver.radius_calibration.use_entropic_allocator = to_bool(val);
            else if (k == "entropic_tau")                  cfg.dro.solver.radius_calibration.entropic_tau = std::stod(val);
            else if (k == "sigma_floor")                   cfg.dro.solver.radius_calibration.sigma_floor = std::stod(val);
            else if (k == "joint_risk_samples")            cfg.dro.solver.radius_calibration.joint_risk_samples = std::stoi(val);
            else if (k == "joint_risk_seed")               cfg.dro.solver.radius_calibration.joint_risk_seed = std::stoull(val);
            else if (k == "mixture_sequence_samples")      cfg.dro.solver.radius_calibration.mixture_sequence_samples = std::stoi(val);
            else if (k == "sqp_max_iterations")            cfg.solver.sqp_max_iterations = std::stoi(val);
            else if (k == "sqp_convergence_tol")           cfg.solver.sqp_convergence_tol = std::stod(val);
            else if (k == "qp_max_iterations")             cfg.solver.qp_max_iterations = std::stoi(val);
            else if (k == "qp_tolerance")                  cfg.solver.qp_tolerance = std::stod(val);
            else if (k == "obstacle_radius")               cfg.obstacle_radius = std::stod(val);
            else if (k == "switch_prob")                   cfg.obstacles.switch_prob = std::stod(val);
            else if (k == "switch_regime") {
                cfg.obstacles.switch_regime = parse_switch(val);
                cfg.mpc.sampling.markov_jump_system =
                    (cfg.obstacles.switch_regime == ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM);
            }
            else if (k == "obstacle_history" || k == "history_configuration")
                                                           cfg.obstacles.history = parse_obstacle_history(val);
            else if (k == "num_obstacles")                 cfg.obstacles.num_obstacles = std::stoi(val);
            else if (k == "num_classes") cfg.obstacles.num_classes = std::stoi(val);
            else if (k == "obstacle_place_on_path") cfg.obstacles.place_on_path = to_bool(val);
            else if (k == "num_modes")                     cfg.obstacles.num_modes = std::stoi(val);
            else if (k == "obstacles_per_class")           cfg.obstacles.obstacles_per_class = std::stoi(val);
            else if (k == "obs_modes")                     cfg.obstacles.obs_modes = split_csv(val);
            else if (k == "randomize_available_modes")     cfg.obstacles.randomize_available_modes = to_bool(val);
            else if (k == "randomize_modes_per_obstacle")  cfg.obstacles.randomize_modes_per_obstacle = to_bool(val);
            else if (k == "rare_mode")                     cfg.obstacles.rare_mode = val;
            else if (k == "rare_switch_prob")              cfg.obstacles.rare_switch_prob = std::stod(val);
            else if (k == "obs_arc_fractions")             cfg.obstacles.obs_arc_fractions = split_csv_d(val);
            else if (k == "obstacle_initial_states" || k == "obstacle_starts")
                                                           cfg.obstacles.initial_obstacle_states = parse_obstacle_states(val);
            else if (k == "obs_path_fraction")             cfg.obstacles.default_arc_fraction = std::stod(val);
            else if (k == "obstacle_behavior") cfg.obstacles.behavior = val;
            else if (k == "obstacle_behavior_initial_speed") cfg.obstacles.behavior_initial_speed = std::stod(val);
            else if (k == "obstacle_behavior_path_offset") cfg.obstacles.behavior_path_offset = std::stod(val);
            else if (k == "obstacle_prediction_noise") cfg.obstacles.prediction_noise = to_bool(val);
            else if (k == "obstacle_process_noise")        cfg.obstacles.process_noise = std::stod(val);
            else if (k == "obstacle_speed_cap")            cfg.obstacles.speed_cap = std::stod(val);
            else if (k == "shift_psi" || k == "shift_rho") cfg.obstacles.shift.psi = std::stod(val);
            else if (k == "shift_boost")                  cfg.obstacles.shift.dangerous_boost = std::stod(val);
            else if (k == "boosted_mode")                  cfg.obstacles.shift.boosted_mode = std::stoi(val);
            else if (k == "environment")                   cfg.environment.type = parse_env(val);
            else if (k == "path_completion_fraction")      cfg.environment.path_completion_fraction = std::stod(val);
            else if (k == "path_completion_termination")   cfg.environment.path_completion_termination = to_bool(val);
            else if (k == "path_waypoints")                cfg.environment.path_waypoints = parse_path_waypoints(val);
            else if (k == "path_closed_loop" || k == "path_closed")
                                                           cfg.environment.path_closed_loop = to_bool(val);
            else if (k == "path_sample_spacing")            cfg.environment.path_sample_spacing = std::stod(val);
            else if (k == "path_control_point_spacing")     cfg.environment.path_control_point_spacing = std::stod(val);
            else if (k == "path_max_lateral_acceleration")  cfg.environment.path_max_lateral_acceleration = std::stod(val);
            else if (k == "s_curve_length")                cfg.environment.s_curve_length = std::stod(val);
            else if (k == "s_curve_amplitude")             cfg.environment.s_curve_amplitude = std::stod(val);
            else if (k == "s_curve_points")                cfg.environment.s_curve_points = std::stoi(val);
            else if (k == "ego_initial_v")                 cfg.environment.ego_initial_v = std::stod(val);
            else if (k == "lane_count")                    cfg.environment.lane_count = std::stoi(val);
            else if (k == "lane_width")                    cfg.environment.lane_width = std::stod(val);
            else if (k == "shoulder_width")                cfg.environment.shoulder_width = std::stod(val);
            else if (k == "median_width")                  cfg.environment.median_width = std::stod(val);
            else if (k == "road_length")                   cfg.environment.road_length = std::stod(val);
            else if (k == "intersection_box_size")         cfg.environment.intersection_box_size = std::stod(val);
            else if (k == "corner_radius")                 cfg.environment.corner_radius = std::stod(val);
            else if (k == "ramp_length")                   cfg.environment.ramp_length = std::stod(val);
            else if (k == "merge_length")                  cfg.environment.merge_length = std::stod(val);
            else if (k == "roundabout_radius")             cfg.environment.roundabout_radius = std::stod(val);
            else if (k == "corridor_width")                cfg.environment.corridor_width = std::stod(val);
            else if (k == "curve_radius")                  cfg.environment.curve_radius = std::stod(val);
            else if (k == "transition_length")             cfg.environment.transition_length = std::stod(val);
            else if (k == "rollout_steps")                 cfg.rollout.rollout_steps = std::stoi(val);
            else if (k == "scenario_tag")                  cfg.rollout.scenario_tag = val;
            else if (k == "method_name")                   cfg.rollout.method_name = val;
            else if (k == "metrics_v_ref")                 cfg.rollout.metrics_v_ref = std::stod(val);
            else if (k == "artifact_output_directory" || k == "artifact_output_dir")
                                                           cfg.artifacts.output_directory = val;
            else if (k == "artifact_run_name" || k == "artifact_label")
                                                           cfg.artifacts.run_name = val;
            else if (k == "artifact_write_manifest")       cfg.artifacts.write_reproducibility_manifest = to_bool(val);
            else if (k == "artifact_write_analysis_csv") cfg.artifacts.write_analysis_csv = to_bool(val);
            else if (k == "artifact_write_trace_csv")      cfg.artifacts.write_trace_csv = to_bool(val);
            else if (k == "artifact_write_visualization_svg")
                                                           cfg.artifacts.write_visualization_svg = to_bool(val);
            else if (k == "artifact_write_visualization_gif" ||
                     k == "artifact_write_gif")
                                                           cfg.artifacts.write_visualization_gif = to_bool(val);
            else if (k == "artifact_show_support_scenarios") cfg.artifacts.show_support_scenarios = to_bool(val);
            else if (k == "artifact_show_linearized_constraints") cfg.artifacts.show_linearized_constraints = to_bool(val);
            else if (k == "artifact_show_sampled_scenarios") cfg.artifacts.show_sampled_scenarios = to_bool(val);
            else if (k == "artifact_scenario_preview_count") cfg.artifacts.scenario_preview_count = std::stoi(val);
            else if (k == "artifact_gif_frame_stride")    cfg.artifacts.gif_frame_stride = std::stoi(val);
            else if (k == "artifact_gif_playback_rate")  cfg.artifacts.gif_playback_rate = std::stod(val);
            else if (k == "artifact_gif_frame_delay_ms")
                throw std::invalid_argument(
                    "artifact_gif_frame_delay_ms was removed; use artifact_gif_playback_rate");
            else if (k == "artifact_write_rviz_replay" ||
                     k == "artifact_write_rviz")
                                                           cfg.artifacts.write_rviz_replay_bundle = to_bool(val);
            else if (k == "use_sqp_solver")
                throw std::invalid_argument("removed because acados SQP is the only solver path");
            else if (k == "injection_mode" || k == "injection_count" || k == "reweighting")
                throw std::invalid_argument(
                    "removed because DRO reweights the scenario distribution instead of injecting scenarios");
            else if (strict)
                throw std::runtime_error("unknown key '" + key + "'");
        } catch (const std::exception& e) {
            throw std::runtime_error("load_experiment_config: " + path + ":" + std::to_string(lineno) +
                                     " bad entry '" + key + ": " + val + "' (" + e.what() + ")");
        }
    }

    // An overlay that provides S but does not explicitly request automatic
    // sizing is a manual-S experiment. This prevents inherited defaults from
    // silently changing a labeled scenario-budget sweep. A file that sets both
    // options makes its policy explicit, regardless of key order.
    if (saw_num_scenarios && !saw_automatic_sample_sizing) {
        cfg.mpc.sampling.automatically_compute_sample_size = false;
    }
}

/**
 * @brief Load an ExperimentConfig from a flat `key: value` YAML file.
 *
 * Always starts from configs/default.yaml (or C++ fallbacks if that file is
 * missing). If `path` is non-empty and different from the default file, it is
 * applied as an overlay.
 */
inline ExperimentConfig load_experiment_config(const std::string& path = "", bool strict = false) {
    ExperimentConfig cfg;
    const std::string def = default_config_path();
    std::ifstream default_file(def);
    if (default_file) {
        apply_yaml_file(cfg, def, strict);
        cfg.config_source = def;
    }
    if (!path.empty() && path != def) {
        apply_yaml_file(cfg, path, strict);
        cfg.config_source = path;
    }
    cfg.normalize();
    try {
        // Validate after normalization so YAML-loaded automatic sizing and
        // fixed-radius settings are checked in the same form used at runtime.
        cfg.to_scenario_mpc_config().validate();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "load_experiment_config: " + cfg.config_source +
            " invalid configuration (" + e.what() + ")");
    }
    return cfg;
}

}  // namespace yaml_config
}  // namespace dro_mpc

#endif  // DRO_MPC_EXPERIMENT_CONFIG_YAML_HPP
