/**
 * @file gan_scenario_loader.cpp
 * @brief Load GAN-generated scenarios from CSV.
 */

#include "gan_scenario_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

namespace scenario_mpc {

namespace {

struct CsvRow {
    int scenario_id = 0;
    int obstacle_id = 0;
    int k = 0;
    double dx = 0;
    double dy = 0;
};

bool parse_header(const std::string& line) {
    return line.find("scenario_id") != std::string::npos;
}

bool parse_row(const std::string& line, CsvRow& row) {
    std::istringstream iss(line);
    std::string cell;
    if (!std::getline(iss, cell, ',')) return false;
    row.scenario_id = std::stoi(cell);
    if (!std::getline(iss, cell, ',')) return false;
    row.obstacle_id = std::stoi(cell);
    if (!std::getline(iss, cell, ',')) return false;
    row.k = std::stoi(cell);
    if (!std::getline(iss, cell, ',')) return false;
    row.dx = std::stod(cell);
    if (!std::getline(iss, cell, ',')) return false;
    row.dy = std::stod(cell);
    return true;
}

}  // namespace

std::vector<Scenario> load_scenarios_from_gan_csv(
    const std::string& csv_path,
    const std::map<int, ObstacleState>& obstacles,
    int horizon
) {
    std::vector<Scenario> out;
    if (obstacles.empty()) return out;

    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) return out;

    std::string line;
    if (!std::getline(ifs, line) || !parse_header(line)) return out;

    // Group rows by scenario_id, then by obstacle_id: (scenario_id, obstacle_id) -> list of (k, dx, dy)
    std::map<int, std::map<int, std::vector<std::tuple<int, double, double>>>> by_scenario;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        CsvRow row;
        if (!parse_row(line, row)) continue;
        if (obstacles.find(row.obstacle_id) == obstacles.end()) continue;
        if (row.k < 0 || row.k > horizon) continue;
        by_scenario[row.scenario_id][row.obstacle_id].emplace_back(row.k, row.dx, row.dy);
    }

    for (const auto& [scenario_id, obs_map] : by_scenario) {
        std::map<int, ObstacleTrajectory> trajectories;
        for (const auto& [obstacle_id, steps_raw] : obs_map) {
            auto it = obstacles.find(obstacle_id);
            if (it == obstacles.end()) continue;
            const ObstacleState& obs = it->second;
            // Sort by k and build PredictionSteps with mean = (obs.x+dx, obs.y+dy)
            std::vector<std::tuple<int, double, double>> steps_sorted(steps_raw.begin(), steps_raw.end());
            std::sort(steps_sorted.begin(), steps_sorted.end(),
                     [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
            std::vector<PredictionStep> steps;
            steps.reserve(steps_sorted.size());
            for (const auto& [k, dx, dy] : steps_sorted) {
                Eigen::Vector2d mean(obs.x + dx, obs.y + dy);
                steps.emplace_back(k, mean, Eigen::Matrix2d::Zero());
            }
            if (static_cast<int>(steps.size()) < horizon + 1) continue;
            ObstacleTrajectory traj(obstacle_id, "gan", steps, 1.0);
            trajectories[obstacle_id] = traj;
        }
        if (trajectories.empty()) continue;
        out.emplace_back(scenario_id, trajectories, 1.0);
    }

    return out;
}

// -----------------------------------------------------------------------------
// GANScenarioCache: load once, materialize many (avoids file I/O per step)
// -----------------------------------------------------------------------------

bool GANScenarioCache::load(const std::string& csv_path, int horizon) {
    waypoints.clear();
    horizon_cached = horizon;
    std::ifstream ifs(csv_path);
    if (!ifs.is_open()) return false;
    std::string line;
    if (!std::getline(ifs, line) || !parse_header(line)) return false;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        CsvRow row;
        if (!parse_row(line, row)) continue;
        if (row.k < 0 || row.k > horizon) continue;
        waypoints.push_back({row.scenario_id, row.obstacle_id, row.k, row.dx, row.dy});
    }
    return !waypoints.empty();
}

std::vector<Scenario> GANScenarioCache::materialize(
    const std::map<int, ObstacleState>& obstacles,
    int horizon,
    int max_scenarios
) const {
    std::vector<Scenario> out;
    if (obstacles.empty() || waypoints.empty()) return out;
    std::map<int, std::map<int, std::vector<std::tuple<int, double, double>>>> by_scenario;
    for (const auto& w : waypoints) {
        if (obstacles.find(w.obstacle_id) == obstacles.end()) continue;
        by_scenario[w.scenario_id][w.obstacle_id].emplace_back(w.k, w.dx, w.dy);
    }
    int count = 0;
    for (const auto& [scenario_id, obs_map] : by_scenario) {
        if (max_scenarios > 0 && count >= max_scenarios) break;
        std::map<int, ObstacleTrajectory> trajectories;
        for (const auto& [obstacle_id, steps_raw] : obs_map) {
            auto it = obstacles.find(obstacle_id);
            if (it == obstacles.end()) continue;
            const ObstacleState& obs = it->second;
            std::vector<std::tuple<int, double, double>> steps_sorted(steps_raw.begin(), steps_raw.end());
            std::sort(steps_sorted.begin(), steps_sorted.end(),
                     [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
            std::vector<PredictionStep> steps;
            steps.reserve(steps_sorted.size());
            for (const auto& [k, dx, dy] : steps_sorted) {
                Eigen::Vector2d mean(obs.x + dx, obs.y + dy);
                steps.emplace_back(k, mean, Eigen::Matrix2d::Zero());
            }
            if (static_cast<int>(steps.size()) < horizon + 1) continue;
            ObstacleTrajectory traj(obstacle_id, "gan", steps, 1.0);
            trajectories[obstacle_id] = traj;
        }
        if (trajectories.empty()) continue;
        out.emplace_back(scenario_id, trajectories, 1.0);
        ++count;
    }
    return out;
}

}  // namespace scenario_mpc
