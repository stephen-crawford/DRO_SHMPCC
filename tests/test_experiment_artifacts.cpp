/**
 * @file test_experiment_artifacts.cpp
 * @brief End-to-end contract for canonical harness artifact bundles.
 */

#include "experiment_harness.hpp"
#include "experiment_config_yaml.hpp"
#include "rviz_replay_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace dro_mpc;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

int line_count(const std::string& contents) {
    int count = 0;
    for (const char ch : contents) {
        if (ch == '\n') ++count;
    }
    return count;
}

std::uint16_t gif_u16(const std::string& contents, std::size_t offset) {
    return static_cast<std::uint16_t>(static_cast<unsigned char>(contents[offset])) |
           (static_cast<std::uint16_t>(static_cast<unsigned char>(contents[offset + 1])) << 8U);
}

struct GifTiming {
    int frame_count = -1;
    std::uint64_t total_delay_centiseconds = 0;
    std::vector<std::uint16_t> frame_delays_centiseconds;
};

GifTiming gif_timing(const std::string& contents) {
    if (contents.size() < 13 || contents.substr(0, 6) != "GIF89a") return {};
    std::size_t offset = 13;
    const auto require = [&](std::size_t count) { return offset + count <= contents.size(); };
    const auto skip_sub_blocks = [&]() -> bool {
        while (true) {
            if (!require(1)) return false;
            const std::size_t count = static_cast<unsigned char>(contents[offset++]);
            if (count == 0) return true;
            if (!require(count)) return false;
            offset += count;
        }
    };
    const unsigned char screen_packed = static_cast<unsigned char>(contents[10]);
    if ((screen_packed & 0x80U) != 0U) {
        const std::size_t palette_size = 3U *
            (1U << ((screen_packed & 0x07U) + 1U));
        if (!require(palette_size)) return {};
        offset += palette_size;
    }

    GifTiming timing;
    timing.frame_count = 0;
    std::uint16_t next_delay_centiseconds = 0;
    while (require(1)) {
        const unsigned char introducer = static_cast<unsigned char>(contents[offset++]);
        if (introducer == 0x3bU) return timing;
        if (introducer == 0x21U) {
            if (!require(1)) return {};
            const unsigned char label = static_cast<unsigned char>(contents[offset++]);
            if (label == 0xf9U) {
                // Graphic-control extension: size, packed, delay-low/high,
                // transparent-index, terminator.
                if (!require(6) || static_cast<unsigned char>(contents[offset]) != 0x04U ||
                    static_cast<unsigned char>(contents[offset + 5]) != 0U) {
                    return {};
                }
                next_delay_centiseconds = gif_u16(contents, offset + 2);
                offset += 6;
                continue;
            }
            if (!skip_sub_blocks()) return {};
            continue;
        }
        if (introducer != 0x2cU || !require(9)) return {};
        const unsigned char image_packed = static_cast<unsigned char>(contents[offset + 8]);
        offset += 9;
        if ((image_packed & 0x80U) != 0U) {
            const std::size_t palette_size = 3U *
                (1U << ((image_packed & 0x07U) + 1U));
            if (!require(palette_size)) return {};
            offset += palette_size;
        }
        if (!require(1)) return {};
        ++offset;  // LZW minimum code size
        if (!skip_sub_blocks()) return {};
        ++timing.frame_count;
        timing.total_delay_centiseconds += next_delay_centiseconds;
        timing.frame_delays_centiseconds.push_back(next_delay_centiseconds);
        next_delay_centiseconds = 0;
    }
    return {};
}

}  // namespace

int main() {
    const fs::path output = fs::temp_directory_path() /
        "dro_shmpcc_experiment_artifact_regression";
    std::error_code error;
    fs::remove_all(output, error);

    ExperimentConfig config = default_experiment_config();
    config.mpc.type = MPCType::MPCC;
    config.mpc.sync_from_type();
    config.mpc.horizon = 4;
    config.mpc.sampling.set_manual_sample_count(4);
    config.dro.enabled = false;
    config.rollout.rollout_steps = 4;
    config.rollout.scenario_tag = "artifact_regression";
    config.artifacts.output_directory = output.string();
    config.artifacts.run_name = "single_run";
    config.artifacts.write_visualization_gif = true;
    config.artifacts.gif_frame_stride = 1;
    config.artifacts.gif_playback_rate = 1.0;
    config.artifacts.write_rviz_replay_bundle = true;

    constexpr unsigned seed = 88221u;
    const RolloutRecord record = run_experiment_rollout(config, seed);
    const fs::path run_directory = output / "single_run";
    check(record.artifact_directory == run_directory.string(),
          "rollout record reports the canonical artifact directory");
    check(fs::exists(run_directory / "reproducibility.yaml") &&
              fs::exists(run_directory / "resolved_config.yaml") &&
              fs::exists(run_directory / "source_config.yaml") &&
              fs::exists(run_directory / "rollout.csv") &&
              fs::exists(run_directory / "trace.csv") &&
              fs::exists(run_directory / "rollout.svg") &&
              fs::exists(run_directory / "rollout.gif") &&
              fs::exists(run_directory / "scene.csv") &&
              fs::exists(run_directory / "geometry.csv") &&
              fs::exists(run_directory / "rollout.rviz"),
          "harness writes a complete reproducibility and visualization bundle");

    const std::string manifest = read_file(run_directory / "reproducibility.yaml");
    const std::string resolved = read_file(run_directory / "resolved_config.yaml");
    const std::string trace = read_file(run_directory / "trace.csv");
    const std::string svg = read_file(run_directory / "rollout.svg");
    const std::string constraints = read_file(run_directory / "linearized_constraints.csv");
    check(constraints.find("step,time_s,horizon_step,obstacle_id,scenario_id,disc_index") == 0 &&
              std::count(constraints.begin(), constraints.end(), '\n') > 1 &&
              svg.find("id=\"linearized-constraints\"") != std::string::npos,
          "default visualization exports retained rows and draws SVG boundaries");
    const std::string gif = read_file(run_directory / "rollout.gif");
    const std::string scene = read_file(run_directory / "scene.csv");
    const std::string geometry = read_file(run_directory / "geometry.csv");
    const std::string rviz = read_file(run_directory / "rollout.rviz");
    const std::string record_csv = read_file(run_directory / "rollout.csv");
    const GifTiming gif_replay_timing = gif_timing(gif);
    check(manifest.find("master_seed: 88221") != std::string::npos &&
              manifest.find("plant_seed:") != std::string::npos &&
              manifest.find("qp_solver_identity:") != std::string::npos,
          "manifest records master/derived seeds and solver identity");
    check(resolved.find("horizon: 4") != std::string::npos &&
              resolved.find("rollout_steps: 4") != std::string::npos &&
              resolved.find("scenario_tag: \"artifact_regression\"") != std::string::npos,
          "resolved configuration captures effective harness settings");
    check(line_count(trace) == 1 + 2 * (record.total_steps + 1) &&
              trace.find("ego") != std::string::npos && trace.find("obstacle") != std::string::npos,
          "trace contains initial and post-step ego/obstacle states");
    check(svg.find("<svg") != std::string::npos &&
              svg.find("green: route") != std::string::npos &&
              svg.find("blue: ego") != std::string::npos,
          "SVG contains route and realized actor trajectories");
    check(gif.substr(0, 6) == "GIF89a" && gif_u16(gif, 6) == 800 &&
              gif_u16(gif, 8) == 520 && gif.find("NETSCAPE2.0") != std::string::npos &&
              gif_replay_timing.frame_count == record.total_steps + 1,
          "GIF is a looping 800x520 replay with every initial and post-step state");
    const auto expected_gif_duration_centiseconds = static_cast<std::uint64_t>(std::llround(
        100.0 * static_cast<double>(record.total_steps) * config.mpc.dt /
        config.artifacts.gif_playback_rate));
    check(gif_replay_timing.total_delay_centiseconds == expected_gif_duration_centiseconds,
          "GIF timeline equals the complete recorded execution duration");
    check(!gif_replay_timing.frame_delays_centiseconds.empty() &&
              gif_replay_timing.frame_delays_centiseconds.back() > 0,
          "GIF gives the final recorded state an explicit display interval");
    check(scene.find("kind,id,point_index,x,y,z") == 0 &&
              scene.find("route,") != std::string::npos &&
              geometry.find("actor,radius,length,num_discs,safety_margin") == 0 &&
              rviz.find("Value: dro_mpc/reference_path") != std::string::npos &&
              rviz.find("Value: dro_mpc/markers") != std::string::npos,
          "RViz bundle contains recorded scene geometry and configured topics");
    const auto replay_scene = dro_mpc::rviz_replay::load_scene(run_directory / "scene.csv");
    const auto replay_geometry = dro_mpc::rviz_replay::load_geometry(
        run_directory / "geometry.csv");
    const auto replay_trace = dro_mpc::rviz_replay::load_trace(run_directory / "trace.csv");
    check(!replay_scene.route.empty() && !replay_scene.roads.empty() &&
              replay_trace.size() == static_cast<std::size_t>(record.total_steps + 1) &&
              replay_trace.front().has_ego && replay_trace.front().obstacles.size() == 1,
          "the dependency-free RViz reader accepts the generated scene and trace data");
    check(std::abs(replay_geometry.ego_radius - config.mpc.ego.radius) < 1e-12 &&
              std::abs(replay_geometry.ego_length - config.mpc.ego.length) < 1e-12 &&
              replay_geometry.ego_num_discs == config.mpc.ego.num_discs &&
              std::abs(replay_geometry.obstacle_radius - config.obstacle_radius) < 1e-12 &&
              std::abs(replay_geometry.safety_margin - config.mpc.constraints.safety_margin) < 1e-12,
          "RViz replay geometry matches the collision model used by the rollout");
    check(record_csv.find("artifact_directory") != std::string::npos &&
              record_csv.find(run_directory.string()) != std::string::npos,
          "rollout CSV links back to its artifact directory");

    ExperimentConfig repeat_config = config;
    repeat_config.artifacts.run_name = "repeat_run";
    const RolloutRecord repeated = run_experiment_rollout(repeat_config, seed);
    check(repeated.artifact_directory == (output / "repeat_run").string() &&
              read_file(output / "repeat_run" / "rollout.gif") == gif,
          "same configuration and seed produce byte-identical GIF replay artifacts");

    ExperimentConfig replay_config = yaml_config::load_experiment_config(
        (run_directory / "resolved_config.yaml").string(), /*strict=*/true);
    check(replay_config.artifacts.output_directory == output.string() &&
              replay_config.artifacts.run_name == "single_run" &&
              replay_config.artifacts.write_reproducibility_manifest &&
              replay_config.artifacts.write_trace_csv &&
              replay_config.artifacts.show_linearized_constraints &&
              replay_config.artifacts.write_visualization_svg &&
              replay_config.artifacts.write_visualization_gif &&
              replay_config.artifacts.gif_frame_stride == 1 &&
              std::abs(replay_config.artifacts.gif_playback_rate - 1.0) < 1e-12 &&
              replay_config.artifacts.write_rviz_replay_bundle,
          "artifact settings survive resolved-YAML loading");
    replay_config.artifacts.show_linearized_constraints = false;
    replay_config.artifacts.run_name = "constraints_disabled";
    const RolloutRecord replay = run_experiment_rollout(replay_config, seed);
    check(replay.collision == record.collision &&
              replay.collision_step == record.collision_step &&
              replay.total_steps == record.total_steps &&
              std::abs(replay.total_progress - record.total_progress) < 1e-12 &&
              std::abs(replay.min_clearance - record.min_clearance) < 1e-12 &&
              std::abs(replay.control_effort - record.control_effort) < 1e-12,
          "resolved configuration replays the model-driven rollout outcome");

    check(read_file(output / "constraints_disabled" / "rollout.gif") != gif,
          "constraint option changes rendered GIF frames");
    check(!fs::exists(output / "constraints_disabled" / "linearized_constraints.csv") &&
              read_file(output / "constraints_disabled" / "rollout.svg").find(
                  "id=\"linearized-constraints\"") == std::string::npos,
          "disabled constraint visualization omits CSV and overlay");
    fs::remove_all(output, error);
    check(!error && !fs::exists(output),
          "artifact regression cleanup removes its isolated temporary directory");
    return failures == 0 ? 0 : 1;
}
