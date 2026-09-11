/**
 * @file test_experiment_runner_cli.cpp
 * @brief Integration contract for experiment_runner artifact-option inheritance.
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

#ifndef EXPERIMENT_RUNNER_PATH
#error "EXPERIMENT_RUNNER_PATH must be supplied by CMake"
#endif

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

std::string shell_quote(const std::string& value) {
    std::string result = "'";
    for (const char character : value) {
        if (character == '\'') result += "'\\''";
        else result += character;
    }
    return result + "'";
}

int run(const std::string& arguments) {
    return std::system((shell_quote(EXPERIMENT_RUNNER_PATH) + " " + arguments).c_str());
}

}  // namespace

int main() {
    const fs::path output = fs::temp_directory_path() /
        "dro_shmpcc_experiment_runner_cli_regression";
    std::error_code error;
    fs::remove_all(output, error);

    const std::string output_arg = shell_quote(output.string());
    const int yaml_result = run(
        "--config configs/visualization_demo.yaml --seed 913 --output " + output_arg +
        " --label yaml_inheritance");
    const fs::path inherited = output / "yaml_inheritance";
    check(yaml_result == 0, "experiment_runner accepts the visualization demo YAML");
    check(fs::exists(inherited / "rollout.gif") && fs::exists(inherited / "scene.csv") &&
              fs::exists(inherited / "geometry.csv") && fs::exists(inherited / "rollout.rviz") &&
              !fs::exists(inherited / "rollout.svg"),
          "runner preserves YAML-selected GIF/RViz outputs without CLI visualization flags");

    check(fs::exists(inherited / "linearized_constraints.csv"),
          "runner enables linearized constraint diagnostics by default");

    const int override_result = run(
        "--config configs/visualization_demo.yaml --seed 914 --output " + output_arg +
        " --label cli_override --svg --no-gif --no-rviz --no-linearized-constraints");
    const fs::path overridden = output / "cli_override";
    check(override_result == 0, "experiment_runner accepts explicit visualization overrides");
    check(fs::exists(overridden / "rollout.svg") && !fs::exists(overridden / "rollout.gif") &&
              !fs::exists(overridden / "scene.csv") && !fs::exists(overridden / "geometry.csv") &&
              !fs::exists(overridden / "rollout.rviz"),
          "explicit CLI visualization flags override YAML settings precisely");

    check(!fs::exists(overridden / "linearized_constraints.csv"),
          "runner accepts constraint visualization opt-out");
    fs::remove_all(output, error);
    check(!error && !fs::exists(output),
          "runner CLI regression cleanup removes its isolated temporary output");
    return failures == 0 ? 0 : 1;
}
