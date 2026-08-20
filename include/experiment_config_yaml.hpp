/**
 * @file experiment_config_yaml.hpp
 * @brief Load an ExperimentConfig from a flat `key: value` YAML test-config file.
 *
 * Self-contained (no yaml-cpp dependency): parses a flat key/value list with
 * `#` comments, quoted strings, and comma-separated lists. Every knob a test
 * cares about — including road_width and safety_margin — is exposed as a key.
 * Unknown keys are ignored unless `strict`. See configs/example_test.yaml.
 *
 *   ExperimentConfig cfg = yaml_config::load_experiment_config("my_test.yaml");
 *   RolloutRecord rec = run_experiment_rollout(cfg, seed);
 */
#ifndef DRO_MPC_EXPERIMENT_CONFIG_YAML_HPP
#define DRO_MPC_EXPERIMENT_CONFIG_YAML_HPP

#include "experiment_harness.hpp"

#include <algorithm>
#include <cctype>
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

// ---- enum parsers ----------------------------------------------------------
inline MPCConfiguration parse_mpc(const std::string& v) {
    std::string l = lower(v);
    if (l == "mpc")    return MPCConfiguration::MPC;
    if (l == "mpcc")   return MPCConfiguration::MPCC;
    if (l == "sh_mpc") return MPCConfiguration::SH_MPC;
    return MPCConfiguration::SH_MPCC;
}
inline EnvironmentType parse_env(const std::string& v) {
    std::string l = lower(v);
    if (l == "overtake" || l == "overtake_slow_lead") return EnvironmentType::OVERTAKE_SLOW_LEAD;
    if (l == "narrow"   || l == "narrow_corridor")    return EnvironmentType::NARROW_CORRIDOR;
    if (l == "intersection")                          return EnvironmentType::INTERSECTION;
    return EnvironmentType::ONCOMING;
}
inline ModeSwitchConfiguration parse_switch(const std::string& v) {
    std::string l = lower(v);
    if (l == "hold" || l == "hold_over_horizon") return ModeSwitchConfiguration::HOLD_OVER_HORIZON;
    return ModeSwitchConfiguration::MARKOV_JUMP_SYSTEM;
}
inline ReweightedDistributionUse parse_reweight(const std::string& v) {
    std::string l = lower(v);
    if (l == "injection_only")                          return ReweightedDistributionUse::INJECTION_ONLY;
    if (l == "sampling_and_injection" || l == "both")   return ReweightedDistributionUse::SAMPLING_AND_INJECTION;
    return ReweightedDistributionUse::SAMPLING_ONLY;
}
inline InjectionMode parse_injection(const std::string& v) {
    std::string l = lower(v);
    if (l == "none")                return InjectionMode::NONE;
    if (l == "dro" || l == "top_risk_inject") return InjectionMode::TOP_RISK_INJECT;
    if (l == "qstar_sample")        return InjectionMode::QSTAR_SAMPLE;
    if (l == "top_risk_inject")     return InjectionMode::TOP_RISK_INJECT;
    if (l == "diverse_risk_inject") return InjectionMode::DIVERSE_RISK_INJECT;
    return InjectionMode::QSTAR_SAMPLE;
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
    if (l == "joint_var")                return DRORiskMeasure::JOINT_VAR;
    if (l == "joint_cvar")               return DRORiskMeasure::JOINT_CVAR;
    return DRORiskMeasure::SURROGATE_VAR_BONFERRONI;  // default: efficient closed-form
}
inline AmbiguityDivergence parse_divergence(const std::string& v) {  // Schuurmans Table I
    std::string l = lower(v);
    if (l == "total_variation" || l == "tv")             return AmbiguityDivergence::TOTAL_VARIATION;
    if (l == "kullback_leibler" || l == "kl")            return AmbiguityDivergence::KULLBACK_LEIBLER;
    if (l == "jensen_shannon" || l == "js")              return AmbiguityDivergence::JENSEN_SHANNON;
    if (l == "hellinger")                                return AmbiguityDivergence::HELLINGER;
    return AmbiguityDivergence::WASSERSTEIN;
}

/**
 * @brief Load an ExperimentConfig from a flat `key: value` YAML file.
 *
 * @param path   Path to the YAML file.
 * @param strict If true, throw on an unrecognized key (else ignore it).
 * @return A normalized ExperimentConfig.
 */
inline ExperimentConfig load_experiment_config(const std::string& path, bool strict = false) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("load_experiment_config: cannot open " + path);

    ExperimentConfig cfg;
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
            // ---- MPC / constraints (road_width & safety_margin live here) ----
            if      (k == "mpc_type")                      { cfg.mpc.type = parse_mpc(val); cfg.mpc.sync_from_type(); }
            else if (k == "horizon")                       cfg.mpc.horizon = std::stoi(val);
            else if (k == "num_scenarios")                 cfg.mpc.sampling.num_scenarios = std::stoi(val);
            else if (k == "road_width")                    cfg.mpc.constraints.road_width = std::stod(val);
            else if (k == "safety_margin")                 cfg.mpc.constraints.safety_margin = std::stod(val);
            else if (k == "enable_contouring_constraints") cfg.mpc.enable_contouring_constraints = to_bool(val);
            else if (k == "enable_velocity_bounds")        cfg.mpc.constraints.enable_velocity_bounds = to_bool(val);
            else if (k == "safe_horizon_min")              cfg.mpc.constraints.safe_horizon_min = std::stoi(val);
            else if (k == "safe_horizon_mode")             cfg.mpc.constraints.safe_horizon_mode = parse_sh(val);
            else if (k == "forced_safe_horizon")           cfg.mpc.constraints.forced_safe_horizon = std::stoi(val);
            else if (k == "support_cap_nbar")              cfg.mpc.constraints.support_cap_nbar = std::stoi(val);
            // ---- ego geometry + dynamics ----
            else if (k == "ego_radius")                    cfg.mpc.ego.radius = std::stod(val);
            else if (k == "ego_length")                    cfg.mpc.ego.length = std::stod(val);
            else if (k == "num_discs")                     cfg.mpc.ego.num_discs = std::stoi(val);
            else if (k == "max_velocity")                  cfg.mpc.ego.dynamics.max_velocity = std::stod(val);
            else if (k == "min_velocity")                  cfg.mpc.ego.dynamics.min_velocity = std::stod(val);
            else if (k == "max_acceleration")              cfg.mpc.ego.dynamics.max_acceleration = std::stod(val);
            else if (k == "min_acceleration")              cfg.mpc.ego.dynamics.min_acceleration = std::stod(val);
            else if (k == "max_steering_rate")             cfg.mpc.ego.dynamics.max_steering_rate = std::stod(val);
            // ---- nominal belief ----
            else if (k == "belief_kind")                   cfg.mpc.sampling.belief_kind = parse_belief(val);
            // ---- DRO ----
            else if (k == "dro_enabled")                   cfg.dro.enabled = to_bool(val);
            else if (k == "reweighting")                   cfg.dro.reweighting = parse_reweight(val);
            else if (k == "injection_mode")                cfg.dro.injection_mode = parse_injection(val);
            else if (k == "injection_count")               cfg.dro.injection_count = std::stoi(val);
            else if (k == "fixed_rho")                     cfg.dro.fixed_rho = std::stod(val);
            else if (k == "risk_measure")                  cfg.dro.solver.radius_calibration.risk_measure = parse_risk(val);
            else if (k == "divergence")                    cfg.dro.solver.radius_calibration.divergence = parse_divergence(val);
            else if (k == "ground_cost")                   cfg.dro.solver.ground_cost_type = parse_ground(val);
            else if (k == "confidence_beta")               cfg.dro.solver.radius_calibration.confidence_beta = std::stod(val);
            else if (k == "calibration_scale")             cfg.dro.solver.radius_calibration.calibration_scale = std::stod(val);
            else if (k == "min_radius")                    cfg.dro.solver.min_radius = std::stod(val);
            else if (k == "max_radius")                    cfg.dro.solver.max_radius = std::stod(val);
            else if (k == "use_entropic_allocator")        cfg.dro.solver.radius_calibration.use_entropic_allocator = to_bool(val);
            else if (k == "entropic_tau")                  cfg.dro.solver.radius_calibration.entropic_tau = std::stod(val);
            // ---- obstacles / mode process ----
            else if (k == "obstacle_radius")               cfg.obstacle_radius = std::stod(val);
            else if (k == "switch_prob")                   cfg.obstacles.switch_prob = std::stod(val);
            else if (k == "switch_regime")                 cfg.obstacles.switch_regime = parse_switch(val);
            else if (k == "num_obstacles")                 cfg.obstacles.num_obstacles = std::stoi(val);
            else if (k == "num_modes")                     cfg.obstacles.num_modes = std::stoi(val);
            else if (k == "obs_modes")                     cfg.obstacles.obs_modes = split_csv(val);
            else if (k == "rare_mode")                     cfg.obstacles.rare_mode = val;
            else if (k == "rare_switch_prob")              cfg.obstacles.rare_switch_prob = std::stod(val);
            // ---- environment ----
            else if (k == "environment")                   cfg.environment.type = parse_env(val);
            else if (k == "path_completion_fraction")      cfg.environment.path_completion_fraction = std::stod(val);
            // ---- rollout / logging ----
            else if (k == "rollout_steps")                 cfg.rollout.rollout_steps = std::stoi(val);
            else if (k == "scenario_tag")                  cfg.rollout.scenario_tag = val;
            else if (k == "method_name")                   cfg.rollout.method_name = val;
            else if (strict)
                throw std::runtime_error("unknown key '" + key + "'");
        } catch (const std::exception& e) {
            throw std::runtime_error("load_experiment_config: " + path + ":" + std::to_string(lineno) +
                                     " bad entry '" + key + ": " + val + "' (" + e.what() + ")");
        }
    }

    cfg.normalize();
    return cfg;
}

}  // namespace yaml_config
}  // namespace dro_mpc

#endif  // DRO_MPC_EXPERIMENT_CONFIG_YAML_HPP
