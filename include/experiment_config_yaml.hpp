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
    std::string l = lower(v);
    return l == "true" || l == "1" || l == "yes" || l == "on";
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

inline MPCType parse_mpc(const std::string& v) {
    std::string l = lower(v);
    if (l == "mpc")    return MPCType::MPC;
    if (l == "mpcc")   return MPCType::MPCC;
    if (l == "sh_mpc") return MPCType::SH_MPC;
    return MPCType::SH_MPCC;
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
    return EnvironmentType::S_CURVE;
}
inline ModeSwitchConfiguration parse_switch(const std::string& v) {
    std::string l = lower(v);
    if (l == "hold" || l == "hold_over_horizon") return ModeSwitchConfiguration::HOLD_OVER_HORIZON;
    return ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM;
}
inline SafeHorizonTruncationRule parse_sh(const std::string& v) {
    std::string l = lower(v);
    if (l == "uncertified_practical" || l == "practical") return SafeHorizonTruncationRule::UNCERTIFIED_PRACTICAL;
    if (l == "theoretical_tight")                         return SafeHorizonTruncationRule::THEORETICAL_TIGHT;
    if (l == "theoretical_simple")                        return SafeHorizonTruncationRule::THEORETICAL_SIMPLE;
    return SafeHorizonTruncationRule::FIXED_NBAR;
}
inline NominalBeliefKind parse_belief(const std::string& v) {
    return lower(v) == "sticky" ? NominalBeliefKind::STICKY : NominalBeliefKind::DIRICHLET;
}
inline DROGroundCostType parse_ground(const std::string& v) {
    std::string l = lower(v);
    if (l == "w1_metric")      return DROGroundCostType::W1_METRIC;
    if (l == "zero_one")       return DROGroundCostType::ZERO_ONE;
    if (l == "euclidean_mean") return DROGroundCostType::EUCLIDEAN_MEAN;
    return DROGroundCostType::W2_BURES;
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
    return DRORiskMeasure::SURROGATE_VAR_BONFERRONI;
}
inline AmbiguityDivergence parse_divergence(const std::string& v) {
    std::string l = lower(v);
    if (l == "total_variation" || l == "tv")             return AmbiguityDivergence::TOTAL_VARIATION;
    if (l == "kullback_leibler" || l == "kl")            return AmbiguityDivergence::KULLBACK_LEIBLER;
    if (l == "jensen_shannon" || l == "js")              return AmbiguityDivergence::JENSEN_SHANNON;
    if (l == "hellinger")                                return AmbiguityDivergence::HELLINGER;
    return AmbiguityDivergence::WASSERSTEIN;
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
    while (std::getline(in, line)) {
        ++lineno;
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;

        const std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));
        if (val.size() >= 2 && (val.front() == '"' || val.front() == '\'') && val.back() == val.front())
            val = val.substr(1, val.size() - 2);
        if (val.size() >= 2 && val.front() == '[' && val.back() == ']')
            val = val.substr(1, val.size() - 2);
        const std::string k = lower(key);

        try {
            if      (k == "mpc_type")                      { cfg.mpc.type = parse_mpc(val); cfg.mpc.sync_from_type(); }
            else if (k == "horizon")                       cfg.mpc.horizon = std::stoi(val);
            else if (k == "dt")                            cfg.mpc.dt = std::stod(val);
            else if (k == "num_scenarios")                 cfg.mpc.sampling.num_scenarios = std::stoi(val);
            else if (k == "road_width")                    cfg.mpc.constraints.road_width = std::stod(val);
            else if (k == "safety_margin")                 cfg.mpc.constraints.safety_margin = std::stod(val);
            else if (k == "enable_contouring_constraints") cfg.mpc.enable_contouring_constraints = to_bool(val);
            else if (k == "safe_horizon_enabled")          cfg.mpc.safe_horizon_enabled = to_bool(val);
            else if (k == "enable_velocity_bounds")        cfg.mpc.constraints.enable_velocity_bounds = to_bool(val);
            else if (k == "safe_horizon_min")              cfg.mpc.constraints.safe_horizon_min = std::stoi(val);
            else if (k == "safe_horizon_mode")             cfg.mpc.constraints.safe_horizon_mode = parse_sh(val);
            else if (k == "forced_safe_horizon")           cfg.mpc.constraints.forced_safe_horizon = std::stoi(val);
            else if (k == "support_cap_nbar")              cfg.mpc.constraints.support_cap_nbar = std::stoi(val);
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
            else if (k == "markov_jump_system")            cfg.mpc.sampling.markov_jump_system = to_bool(val);
            else if (k == "enforce_certified_scenario_count" || k == "enforce_scenario_count")
                                                           cfg.mpc.sampling.enforce_certified_scenario_count = to_bool(val);
            else if (k == "ensure_mode_coverage")          cfg.mpc.sampling.ensure_mode_coverage = to_bool(val);
            else if (k == "max_history_length")            cfg.mpc.sampling.max_history_length = std::stoi(val);
            else if (k == "one_minus_chance_constraint_violation_probability" || k == "confidence_level")
                                                           cfg.mpc.sampling.one_minus_chance_constraint_violation_probability = std::stod(val);
            else if (k == "chance_of_certificate_violation" || k == "beta")
                                                           cfg.mpc.sampling.chance_of_certificate_violation = std::stod(val);
            else if (k == "dro_enabled")                   cfg.dro.enabled = to_bool(val);
            else if (k == "fixed_rho")                     cfg.dro.fixed_rho = std::stod(val);
            else if (k == "risk_measure")                  cfg.dro.solver.radius_calibration.risk_measure = parse_risk(val);
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
            else if (k == "mixture_sequence_samples")      cfg.dro.solver.radius_calibration.mixture_sequence_samples = std::stoi(val);
            else if (k == "use_sqp_solver")                cfg.solver.use_sqp_solver = to_bool(val);
            else if (k == "sqp_max_iterations")            cfg.solver.sqp_max_iterations = std::stoi(val);
            else if (k == "sqp_convergence_tol")           cfg.solver.sqp_convergence_tol = std::stod(val);
            else if (k == "qp_max_iterations")             cfg.solver.qp_max_iterations = std::stoi(val);
            else if (k == "qp_tolerance")                  cfg.solver.qp_tolerance = std::stod(val);
            else if (k == "obstacle_radius")               cfg.obstacle_radius = std::stod(val);
            else if (k == "switch_prob")                   cfg.obstacles.switch_prob = std::stod(val);
            else if (k == "switch_regime")                 cfg.obstacles.switch_regime = parse_switch(val);
            else if (k == "num_obstacles")                 cfg.obstacles.num_obstacles = std::stoi(val);
            else if (k == "num_modes")                     cfg.obstacles.num_modes = std::stoi(val);
            else if (k == "obstacles_per_class")           cfg.obstacles.obstacles_per_class = std::stoi(val);
            else if (k == "obs_modes")                     cfg.obstacles.obs_modes = split_csv(val);
            else if (k == "rare_mode")                     cfg.obstacles.rare_mode = val;
            else if (k == "rare_switch_prob")              cfg.obstacles.rare_switch_prob = std::stod(val);
            else if (k == "obs_arc_fractions")             cfg.obstacles.obs_arc_fractions = split_csv_d(val);
            else if (k == "obs_path_fraction")             cfg.obstacles.default_arc_fraction = std::stod(val);
            else if (k == "obstacle_process_noise")        cfg.obstacles.process_noise = std::stod(val);
            else if (k == "obstacle_speed_cap")            cfg.obstacles.speed_cap = std::stod(val);
            else if (k == "shift_psi" || k == "shift_rho") cfg.obstacles.shift.psi = std::stod(val);
            else if (k == "shift_boost")                  cfg.obstacles.shift.dangerous_boost = std::stod(val);
            else if (k == "boosted_mode")                  cfg.obstacles.shift.boosted_mode = std::stoi(val);
            else if (k == "environment")                   cfg.environment.type = parse_env(val);
            else if (k == "path_completion_fraction")      cfg.environment.path_completion_fraction = std::stod(val);
            else if (k == "path_completion_termination")   cfg.environment.path_completion_termination = to_bool(val);
            else if (k == "s_curve_length")                cfg.environment.s_curve_length = std::stod(val);
            else if (k == "s_curve_amplitude")             cfg.environment.s_curve_amplitude = std::stod(val);
            else if (k == "s_curve_points")                cfg.environment.s_curve_points = std::stoi(val);
            else if (k == "ego_initial_v")                 cfg.environment.ego_initial_v = std::stod(val);
            else if (k == "rollout_steps")                 cfg.rollout.rollout_steps = std::stoi(val);
            else if (k == "scenario_tag")                  cfg.rollout.scenario_tag = val;
            else if (k == "method_name")                   cfg.rollout.method_name = val;
            else if (k == "metrics_v_ref")                 cfg.rollout.metrics_v_ref = std::stod(val);
            else if (k == "injection_mode" || k == "injection_count" || k == "reweighting")
                continue;  // removed; ignore leftover keys in old overlays
            else if (strict)
                throw std::runtime_error("unknown key '" + key + "'");
        } catch (const std::exception& e) {
            throw std::runtime_error("load_experiment_config: " + path + ":" + std::to_string(lineno) +
                                     " bad entry '" + key + ": " + val + "' (" + e.what() + ")");
        }
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
    try {
        apply_yaml_file(cfg, def, strict);
    } catch (const std::exception&) {
        // C++ in-class initializers already match the intended defaults.
    }
    if (!path.empty() && path != def) {
        apply_yaml_file(cfg, path, strict);
    }
    cfg.normalize();
    return cfg;
}

}  // namespace yaml_config
}  // namespace dro_mpc

#endif  // DRO_MPC_EXPERIMENT_CONFIG_YAML_HPP
