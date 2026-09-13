/**
 * @file experiment_runner.cpp
 * @brief Small supported CLI for canonical harness rollouts and artifacts.
 */

#include "experiment_config_yaml.hpp"
#include "experiment_harness.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace dro_mpc;

namespace {

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable
        << " [--config FILE] [--seed N] [--rollouts N] [--output DIRECTORY]"
           " [--label NAME] [--manifest|--no-manifest] [--trace|--no-trace]"
           " [--svg|--no-svg] [--gif|--no-gif] [--rviz|--no-rviz]"
           " [--linearized-constraints|--no-linearized-constraints]"
           " [--support-scenarios|--no-support-scenarios]"
           " [--gif-frame-stride N] [--gif-playback-rate R]\n\n"
           "Runs the canonical experiment harness. Each rollout writes a deterministic\n"
           "artifact bundle containing resolved_config.yaml, reproducibility.yaml,\n"
           "rollout.csv, trace.csv, and selected SVG/GIF/RViz outputs; summary.csv is\n"
           "written at the output root. YAML artifact settings are preserved unless a\n"
           "corresponding CLI flag is supplied.\n";
}

unsigned parse_unsigned(const std::string& value, const char* option) {
    try {
        const unsigned long parsed = std::stoul(value);
        if (parsed > std::numeric_limits<unsigned>::max()) {
            throw std::out_of_range("out of range");
        }
        return static_cast<unsigned>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " expects a non-negative integer");
    }
}

int parse_positive_int(const std::string& value, const char* option) {
    try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) throw std::out_of_range("not positive");
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " expects a positive integer");
    }
}

double parse_positive_double(const std::string& value, const char* option) {
    try {
        std::size_t parsed_characters = 0;
        const double parsed = std::stod(value, &parsed_characters);
        if (parsed_characters != value.size() || !std::isfinite(parsed) || parsed <= 0.0) {
            throw std::out_of_range("not positive");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " expects a finite positive number");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path;
        std::optional<std::string> output_override;
        std::string label;
        unsigned seed = 1u;
        int rollout_count = 1;
        std::optional<bool> write_manifest;
        std::optional<bool> write_trace;
        std::optional<bool> write_svg;
        std::optional<bool> write_gif;
        std::optional<bool> write_rviz;
        std::optional<bool> show_linearized_constraints;
        std::optional<bool> show_support_scenarios;
        std::optional<int> gif_frame_stride;
        std::optional<double> gif_playback_rate;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            const auto need_value = [&](const char* option) -> std::string {
                if (++i >= argc) throw std::invalid_argument(std::string(option) + " requires a value");
                return argv[i];
            };
            if (argument == "--help" || argument == "-h") {
                print_usage(argv[0]);
                return 0;
            }
            if (argument == "--config") config_path = need_value("--config");
            else if (argument == "--seed") seed = parse_unsigned(need_value("--seed"), "--seed");
            else if (argument == "--rollouts") {
                rollout_count = parse_positive_int(need_value("--rollouts"), "--rollouts");
            } else if (argument == "--output") output_override = need_value("--output");
            else if (argument == "--label") label = need_value("--label");
            else if (argument == "--manifest") write_manifest = true;
            else if (argument == "--no-manifest") write_manifest = false;
            else if (argument == "--trace") write_trace = true;
            else if (argument == "--no-trace") write_trace = false;
            else if (argument == "--svg") write_svg = true;
            else if (argument == "--no-svg") write_svg = false;
            else if (argument == "--gif") write_gif = true;
            else if (argument == "--no-gif") write_gif = false;
            else if (argument == "--support-scenarios") show_support_scenarios = true;
            else if (argument == "--no-support-scenarios") show_support_scenarios = false;
            else if (argument == "--linearized-constraints") show_linearized_constraints = true;
            else if (argument == "--no-linearized-constraints") show_linearized_constraints = false;
            else if (argument == "--rviz") write_rviz = true;
            else if (argument == "--no-rviz") write_rviz = false;
            else if (argument == "--gif-frame-stride") {
                gif_frame_stride = parse_positive_int(
                    need_value("--gif-frame-stride"), "--gif-frame-stride");
            } else if (argument == "--gif-playback-rate") {
                gif_playback_rate = parse_positive_double(
                    need_value("--gif-playback-rate"), "--gif-playback-rate");
            }
            else throw std::invalid_argument("unknown option '" + argument + "'");
        }

        ExperimentConfig config = yaml_config::load_experiment_config(config_path, true);
        if (output_override.has_value()) {
            config.artifacts.output_directory = *output_override;
        }
        if (config.artifacts.output_directory.empty()) {
            config.artifacts.output_directory = "artifacts";
        }
        if (write_manifest.has_value()) {
            config.artifacts.write_reproducibility_manifest = *write_manifest;
        }
        if (write_trace.has_value()) config.artifacts.write_trace_csv = *write_trace;
        if (write_svg.has_value()) config.artifacts.write_visualization_svg = *write_svg;
        if (write_gif.has_value()) config.artifacts.write_visualization_gif = *write_gif;
        if (show_support_scenarios.has_value())
            config.artifacts.show_support_scenarios = *show_support_scenarios;
        if (show_linearized_constraints.has_value())
            config.artifacts.show_linearized_constraints = *show_linearized_constraints;
        if (write_rviz.has_value()) config.artifacts.write_rviz_replay_bundle = *write_rviz;
        if (gif_frame_stride.has_value()) config.artifacts.gif_frame_stride = *gif_frame_stride;
        if (gif_playback_rate.has_value()) config.artifacts.gif_playback_rate = *gif_playback_rate;

        fs::create_directories(config.artifacts.output_directory);
        CSVWriter summary((fs::path(config.artifacts.output_directory) / "summary.csv").string());
        summary.write_header();

        for (int rollout = 0; rollout < rollout_count; ++rollout) {
            ExperimentConfig run_config = config;
            const unsigned run_seed = seed + static_cast<unsigned>(rollout);
            if (!label.empty()) {
                run_config.artifacts.run_name = rollout_count == 1
                    ? label
                    : label + "_" + std::to_string(rollout);
            }
            const RolloutRecord record = run_experiment_rollout(run_config, run_seed);
            summary.write_record(record);
            std::cout << "seed=" << run_seed
                      << " collision=" << (record.collision ? "yes" : "no")
                      << " termination=" << record.termination_reason
                      << " artifact=" << record.artifact_directory << '\n';
        }
        summary.flush();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "experiment_runner: " << error.what() << '\n';
        return 1;
    }
}
