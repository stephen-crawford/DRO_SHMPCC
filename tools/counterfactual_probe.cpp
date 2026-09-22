/** Isolated frozen-state diagnostics; never feeds a control to the live rollout. */
#include "experiment_config_yaml.hpp"
#include "experiment_harness.hpp"
#include "mpc_controller.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

namespace fs = std::filesystem;
namespace dro_mpc {

class CounterfactualProbe {
    using Weights = std::map<int, std::map<std::string, double>>;

    static std::unique_ptr<MPCController> fork(const MPCController& source, bool dro_enabled) {
        auto config = source.config_;
        config.dro.enabled = dro_enabled;
        auto copy = std::make_unique<MPCController>(config);
        // QP workspaces are intentionally fresh: AcadosQPSolver starts every
        // subproblem deterministically and uses no RNG/warm-start state.
        copy->ego_dynamics_ = source.ego_dynamics_;
        copy->default_modes_ = source.default_modes_;
        copy->mode_histories_ = source.mode_histories_;
        copy->obstacle_class_ids_ = source.obstacle_class_ids_;
        copy->scenarios_ = source.scenarios_;
        copy->dro_ = source.dro_;
        copy->last_dro_results_ = source.last_dro_results_;
        copy->reference_trajectory_ = source.reference_trajectory_;
        copy->rng_ = source.rng_;
        copy->solve_times_ = source.solve_times_;
        copy->iteration_count_ = source.iteration_count_;
        copy->reference_path_ = source.reference_path_;
        copy->custom_per_obstacle_weights_ = source.custom_per_obstacle_weights_;
        copy->capture_linearized_constraints_ = source.capture_linearized_constraints_;
        copy->last_linearized_constraints_ = source.last_linearized_constraints_;
        copy->last_feasible_controls_ = source.last_feasible_controls_;
        copy->has_feasible_backup_ = source.has_feasible_backup_;
        copy->last_removed_scenario_ids_ = source.last_removed_scenario_ids_;
        return copy;
    }

    static MPCResult solve(MPCController& controller, const DecisionContext& c) {
        return controller.solve(c.ego, c.obstacles, c.goal, c.reference_velocity,
                                c.path_progress, c.path_length);
    }

    static std::ofstream output(const fs::path& path) {
        std::ofstream stream(path);
        if (!stream) throw std::runtime_error("cannot write " + path.string());
        stream << std::setprecision(17);
        return stream;
    }

    static void plan(const fs::path& root, const std::string& label, const MPCResult& result) {
        auto states = output(root/(label+"_plan.csv"));
        states << "k,x,y,theta,v,s\n";
        for (size_t k = 0; k < result.ego_trajectory.size(); ++k) {
            const auto& x = result.ego_trajectory[k];
            states << k << ',' << x.x << ',' << x.y << ',' << x.theta << ',' << x.v << ',' << x.s << '\n';
        }
        auto controls = output(root/(label+"_controls.csv"));
        controls << "k,a,omega\n";
        for (size_t k = 0; k < result.control_inputs.size(); ++k)
            controls << k << ',' << result.control_inputs[k].a << ',' << result.control_inputs[k].omega << '\n';
    }

public:
    static void run(const MPCController& source, const DecisionContext& c,
                    const fs::path& root, int samples, unsigned mc_seed,
                    const std::vector<double>& scales) {
        if (!source.config_.dro.enabled)
            throw std::runtime_error("probe requires a DRO source run");
        if (source.config_.mpc.sampling.markov_jump_system)
            throw std::runtime_error("probe currently supports held-mode prediction only; Markov replay is not implemented");
        if (!source.custom_per_obstacle_weights_.empty())
            throw std::runtime_error("external sampling weights are not supported by this probe");
        auto nominal = fork(source, false);
        auto robust = fork(source, true);
        const auto nominal_plan = solve(*nominal, c);
        const auto robust_plan = solve(*robust, c);
        plan(root, "nominal", nominal_plan);
        plan(root, "dro", robust_plan);
        auto snapshot = output(root/"snapshot.csv");
        snapshot << "step,x,y,theta,v,s,path_progress,path_length,reference_velocity,nominal_success,dro_success,samples,mc_seed\n";
        snapshot << c.step << ',' << c.ego.x << ',' << c.ego.y << ',' << c.ego.theta << ','
                 << c.ego.v << ',' << c.ego.s << ',' << c.path_progress << ',' << c.path_length << ','
                 << c.reference_velocity << ',' << nominal_plan.success << ',' << robust_plan.success << ','
                 << samples << ',' << mc_seed << '\n';
        auto obstacles = output(root/"obstacles.csv");
        obstacles << "obstacle_id,x,y,vx,vy\n";
        for (const auto& [id, state] : c.obstacles)
            obstacles << id << ',' << state.x << ',' << state.y << ',' << state.vx << ',' << state.vy << '\n';
        auto history_file = output(root/"mode_history.csv");
        history_file << "obstacle_id,class_id,timestep,source_obstacle_id,mode\n";
        for (const auto& [id, history] : source.mode_histories_)
            for (const auto& observation : history.observed_modes)
                history_file << id << ',' << history.obstacle_class_id << ',' << observation.timestep << ','
                             << observation.source_obstacle_id << ',' << observation.mode_id << '\n';
        auto rng_file = output(root/"controller_rng.txt");
        rng_file << source.rng_ << '\n';
        MPCResult warmstart;
        warmstart.ego_trajectory = source.reference_trajectory_;
        warmstart.control_inputs = source.last_feasible_controls_;
        plan(root, "source_warmstart", warmstart);
        Weights p, q;
        auto weights = output(root/"weights.csv");
        weights << "obstacle_id,mode,p,q,delta,risk_score,rho\n";
        for (const auto& [id, state] : c.obstacles) {
            p[id] = compute_mode_weights(source.mode_histories_.at(id), source.config_.mpc.sampling.mode_belief);
            q[id] = robust->last_dro_results_.at(id).worst_case_weights;
            const auto& dro = robust->last_dro_results_.at(id);
            for (const auto& [mode, probability] : p.at(id))
                weights << id << ',' << mode << ',' << probability << ',' << q.at(id).at(mode) << ','
                        << q.at(id).at(mode)-probability << ',' << dro.risk_per_mode.at(mode) << ',' << dro.rho_used << '\n';
        }
        auto trials = output(root/"counterfactual_trials.csv");
        trials << "law,trial,collision,first_step_collision,first_collision_k,min_margin\n";
        auto summary = output(root/"counterfactual_summary.csv");
        summary << "law,status,samples,collisions,first_step_collisions,collision_rate,min_margin\n";
        auto replay = [&](const std::string& label, const Weights& distribution) {
            if (!nominal_plan.success) {
                summary << label << ",nominal_plan_inadmissible,0,,,,\n";
                return;
            }
            std::mt19937 rng(mc_seed);  // independent of both controller and plant RNGs
            int collisions = 0, first_step = 0;
            double minimum = std::numeric_limits<double>::infinity();
            for (int i = 0; i < samples; ++i) {
                const auto scenarios = sample_scenarios(c.obstacles, source.mode_histories_,
                    &distribution, source.config_.mpc.horizon, 1,
                    source.config_.mpc.sampling.mode_belief, nullptr, &rng);
                double margin = std::numeric_limits<double>::infinity();
                int first = -1;
                for (int k = 1; k <= source.config_.mpc.horizon; ++k) {
                    const auto discs = compute_ego_disc_positions(nominal_plan.ego_trajectory.at(k),
                        source.config_.mpc.ego.num_discs, source.config_.mpc.ego.length);
                    for (const auto& [id, trajectory] : scenarios.at(0).trajectories) {
                        for (const auto& disc : discs) {
                            const double value = (disc-trajectory.steps.at(k).mean).norm()-source.config_.combined_radius();
                            if (!std::isfinite(value)) throw std::runtime_error("nonfinite counterfactual margin");
                            margin = std::min(margin, value);
                            if (value < 0.0 && first < 0) first = k;
                        }
                    }
                }
                collisions += first >= 0;
                first_step += first == 1;
                minimum = std::min(minimum, margin);
                trials << label << ',' << i << ',' << (first >= 0) << ',' << (first == 1) << ',' << first << ',' << margin << '\n';
            }
            summary << label << ",evaluated," << samples << ',' << collisions << ',' << first_step << ','
                    << static_cast<double>(collisions)/samples << ',' << minimum << '\n';
        };
        replay("p", p);
        replay("qstar", q);
        for (const auto& [id, modes] : q) {
            for (const auto& [mode, probability] : modes) {
                if (probability <= p.at(id).at(mode)) continue;
                auto conditional = q;
                for (auto& [other_mode, weight] : conditional.at(id)) weight = other_mode == mode ? 1.0 : 0.0;
                replay("boosted_o"+std::to_string(id)+"_"+mode, conditional);
            }
        }
        // Optional fixed-state radius sensitivity. Recompute q with the existing
        // fixed-radius DRO API, then solve a copied controller via its existing
        // custom categorical interface. This is NOT a calibrated certificate or
        // a full closed-loop alpha-scaled experiment.
        auto sweep = output(root/"radius_sweep.csv");
        sweep << "alpha,success,sampled_constraints_satisfied,used_fallback\n";
        auto radii = output(root/"radius_sweep_weights.csv");
        radii << "alpha,obstacle_id,rho_requested,rho_used,mode,q\n";
        for (double alpha : scales) {
            auto variant = fork(source, false);
            auto reference = fork(source, true);
            EgoState ego = c.ego;
            if (!ego.has_spline()) ego.s = reference->reference_path_->find_closest_point(ego.position());
            reference->initialize_reference_trajectory(ego, c.goal, c.reference_velocity);
            for (const auto& [id, state] : c.obstacles) {
                auto config = source.config_.dro.solver;
                const double radius = alpha*robust->last_dro_results_.at(id).rho_used;
                config.radius_calibration.use_calibrated_radius = false;
                config.base_radius = radius;
                config.min_radius = 0.0;
                config.max_radius = std::max(config.max_radius, radius);
                DRO dro(config);
                const auto& history = source.mode_histories_.at(id);
                // TODO: Update this to the new framework dro.set_observation_count(history.ambiguity_radius_sample_count());
                const auto result = dro.compute_worst_case_weights(p.at(id), history.get_mode_counts(),
                    state, history.available_modes, reference->reference_trajectory_, source.config_.mpc.horizon,
                    source.config_.mpc.ego.radius, source.config_.obstacle_radius,
                    source.config_.mpc.constraints.safety_margin, source.config_.mpc.horizon,
                    source.config_.mpc.ego.num_discs, source.config_.mpc.ego.length);
                variant->set_custom_mode_weights(id, result.worst_case_weights);
                for (const auto& [mode, probability] : result.worst_case_weights)
                    radii << alpha << ',' << id << ',' << radius << ',' << result.rho_used << ',' << mode << ',' << probability << '\n';
            }
            const auto result = solve(*variant, c);
            sweep << alpha << ',' << result.success << ',' << result.sampled_constraints_satisfied << ',' << result.used_fallback << '\n';
            plan(root, "alpha_"+std::to_string(alpha), result);
        }
    }
};
}  // namespace dro_mpc

int main(int argc, char** argv) {
    using namespace dro_mpc;
    try {
        std::string config_path;
        fs::path root;
        unsigned seed = 0, mc_seed = 12345;
        int step = -1, samples = 1000;
        std::vector<double> scales;
        auto integer = [](const std::string& value, unsigned long long maximum) {
            size_t consumed = 0;
            const auto parsed = std::stoull(value, &consumed);
            if (value.empty() || value.front() == '-' || consumed != value.size() || parsed > maximum)
                throw std::runtime_error("invalid integer argument: " + value);
            return parsed;
        };
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "counterfactual_probe --config YAML --seed N --step ZERO_BASED --output DIR [--samples N] [--mc-seed N] [--radius-scales 0,0.25,0.5,0.75,1]\n";
                return 0;
            }
            if (++i >= argc) throw std::runtime_error("missing argument for " + key);
            const std::string value = argv[i];
            if (key == "--config") config_path = value;
            else if (key == "--output") root = value;
            else if (key == "--seed") seed = integer(value, std::numeric_limits<unsigned>::max());
            else if (key == "--mc-seed") mc_seed = integer(value, std::numeric_limits<unsigned>::max());
            else if (key == "--step") step = integer(value, std::numeric_limits<int>::max()-1);
            else if (key == "--samples") samples = integer(value, std::numeric_limits<int>::max());
            else if (key == "--radius-scales") {
                std::istringstream stream(value);
                std::string token;
                while (std::getline(stream, token, ',')) {
                    size_t consumed = 0;
                    double scale = std::stod(token, &consumed);
                    if (consumed != token.size()) throw std::runtime_error("invalid radius scale");
                    for (double previous : scales)
                        if (std::to_string(previous) == std::to_string(scale))
                            throw std::runtime_error("radius scale labels must be distinct to six decimals");
                    if (!std::isfinite(scale) || scale < 0.0) throw std::runtime_error("invalid radius scale");
                    scales.push_back(scale);
                }
            } else throw std::runtime_error("unknown argument: " + key);
        }
        if (step < 0 || samples < 1 || root.empty() || config_path.empty()) throw std::runtime_error("config/output/step and positive samples required");
        if (fs::exists(root)) throw std::runtime_error("output already exists; use a new directory");
        auto config = yaml_config::load_experiment_config(config_path, true);
        config.rollout.rollout_steps = step+1;
        config.artifacts.output_directory = (root/"replay").string();
        config.artifacts.run_name = "source";
        config.artifacts.write_analysis_csv = true;
        config.artifacts.write_trace_csv = true;
        config.artifacts.write_reproducibility_manifest = true;
        config.artifacts.write_visualization_svg = false;
        config.artifacts.write_visualization_gif = false;
        config.artifacts.write_rviz_replay_bundle = false;
        fs::create_directories(root);
        bool probed = false;
        config.rollout.decision_callback = [&](const DecisionContext& c, const MPCController& controller) {
            if (c.step != step) return;
            CounterfactualProbe::run(controller, c, root, samples, mc_seed, scales);
            probed = true;
        };
        run_experiment_rollout(config, seed);
        if (!probed) throw std::runtime_error("source rollout ended before requested decision");
        std::cout << "Probed zero-based decision " << step << " at " << root << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "counterfactual_probe: " << error.what() << '\n';
        return 1;
    }
}
