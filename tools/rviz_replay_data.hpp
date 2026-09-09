/**
 * @file rviz_replay_data.hpp
 * @brief Dependency-free reader for RViz replay scene and trace artifacts.
 */

#ifndef DRO_MPC_RVIZ_REPLAY_DATA_HPP
#define DRO_MPC_RVIZ_REPLAY_DATA_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dro_mpc {
namespace rviz_replay {

struct Point2D {
    double x = 0.0;
    double y = 0.0;
};

struct ActorState {
    int id = -1;
    Point2D position;
    double theta = 0.0;
    double vx = 0.0;
    double vy = 0.0;
};

struct ReplayFrame {
    int step = 0;
    double time_seconds = 0.0;
    ActorState ego;
    bool has_ego = false;
    std::vector<ActorState> obstacles;
};

struct ReplayScene {
    std::vector<std::vector<Point2D>> roads;
    std::vector<Point2D> route;
};

struct ReplayGeometry {
    double ego_radius = 0.0;
    double ego_length = 0.0;
    int ego_num_discs = 0;
    double obstacle_radius = 0.0;
    double safety_margin = 0.0;
};

namespace internal {

inline std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

inline double parse_double(const std::string& value, const std::filesystem::path& path,
                           int line) {
    try {
        std::size_t parsed_characters = 0;
        const double parsed = std::stod(value, &parsed_characters);
        if (parsed_characters != value.size() || !std::isfinite(parsed)) {
            throw std::out_of_range("not finite or fully parsed");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error(path.string() + ":" + std::to_string(line) +
                                 " has an invalid floating-point field");
    }
}

inline int parse_int(const std::string& value, const std::filesystem::path& path, int line) {
    try {
        std::size_t parsed_characters = 0;
        const int parsed = std::stoi(value, &parsed_characters);
        if (parsed_characters != value.size()) throw std::out_of_range("not fully parsed");
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error(path.string() + ":" + std::to_string(line) +
                                 " has an invalid integer field");
    }
}

}  // namespace internal

inline ReplayScene load_scene(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open RViz scene '" + path.string() + "'");
    std::string line;
    std::getline(input, line);
    if (line != "kind,id,point_index,x,y,z") {
        throw std::runtime_error("unexpected scene.csv header in '" + path.string() + "'");
    }

    std::map<int, std::vector<std::pair<int, Point2D>>> roads;
    std::vector<std::pair<int, Point2D>> route;
    int line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto fields = internal::split_csv(line);
        if (fields.size() != 6) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     " has an unexpected number of columns");
        }
        const int id = internal::parse_int(fields[1], path, line_number);
        const int point_index = internal::parse_int(fields[2], path, line_number);
        const Point2D point{internal::parse_double(fields[3], path, line_number),
                            internal::parse_double(fields[4], path, line_number)};
        (void)internal::parse_double(fields[5], path, line_number);
        if (fields[0] == "road") roads[id].push_back({point_index, point});
        else if (fields[0] == "route") route.push_back({point_index, point});
        else throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                      " has unknown scene kind '" + fields[0] + "'");
    }

    const auto ordered_points = [&](std::vector<std::pair<int, Point2D>> points) {
        std::sort(points.begin(), points.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<Point2D> result;
        result.reserve(points.size());
        for (std::size_t index = 0; index < points.size(); ++index) {
            if (points[index].first != static_cast<int>(index)) {
                throw std::runtime_error("RViz scene has missing or duplicate point indices: '" +
                                         path.string() + "'");
            }
            result.push_back(points[index].second);
        }
        return result;
    };

    ReplayScene scene;
    for (auto& [id, points] : roads) {
        (void)id;
        scene.roads.push_back(ordered_points(std::move(points)));
    }
    scene.route = ordered_points(std::move(route));
    if (scene.route.empty()) {
        throw std::runtime_error("RViz scene has no reference route: '" + path.string() + "'");
    }
    return scene;
}

inline ReplayGeometry load_geometry(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open RViz geometry '" + path.string() + "'");
    std::string line;
    std::getline(input, line);
    if (line != "actor,radius,length,num_discs,safety_margin") {
        throw std::runtime_error("unexpected geometry.csv header in '" + path.string() + "'");
    }

    ReplayGeometry geometry;
    bool saw_ego = false;
    bool saw_obstacle = false;
    double obstacle_safety_margin = 0.0;
    int line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto fields = internal::split_csv(line);
        if (fields.size() != 5) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     " has an unexpected number of columns");
        }
        const double radius = internal::parse_double(fields[1], path, line_number);
        const double length = internal::parse_double(fields[2], path, line_number);
        const int num_discs = internal::parse_int(fields[3], path, line_number);
        const double safety_margin = internal::parse_double(fields[4], path, line_number);
        if (fields[0] == "ego") {
            if (saw_ego) throw std::runtime_error("duplicate ego geometry in '" + path.string() + "'");
            geometry.ego_radius = radius;
            geometry.ego_length = length;
            geometry.ego_num_discs = num_discs;
            geometry.safety_margin = safety_margin;
            saw_ego = true;
        } else if (fields[0] == "obstacle") {
            if (saw_obstacle) {
                throw std::runtime_error("duplicate obstacle geometry in '" + path.string() + "'");
            }
            geometry.obstacle_radius = radius;
            if (length != 0.0 || num_discs != 1) {
                throw std::runtime_error("invalid obstacle geometry in '" + path.string() + "'");
            }
            obstacle_safety_margin = safety_margin;
            saw_obstacle = true;
        } else {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     " has unknown geometry actor '" + fields[0] + "'");
        }
    }
    if (!saw_ego || !saw_obstacle || !std::isfinite(geometry.ego_radius) ||
        !std::isfinite(geometry.ego_length) || !std::isfinite(geometry.obstacle_radius) ||
        !std::isfinite(geometry.safety_margin) || geometry.ego_radius <= 0.0 ||
        geometry.ego_length < 0.0 || geometry.ego_num_discs <= 0 ||
        geometry.obstacle_radius <= 0.0 || geometry.safety_margin < 0.0) {
        throw std::runtime_error("invalid replay geometry in '" + path.string() + "'");
    }
    if (geometry.safety_margin != obstacle_safety_margin) {
        throw std::runtime_error("inconsistent safety margins in '" + path.string() + "'");
    }
    return geometry;
}

inline std::vector<ReplayFrame> load_trace(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open RViz trace '" + path.string() + "'");
    std::string line;
    std::getline(input, line);
    const std::string expected_header =
        "step,time_s,actor,obstacle_id,mode,x,y,theta,v,vx,vy,path_progress,"
        "minimum_clearance,ambiguity_radius,solve_time_ms,collision";
    if (line != expected_header) {
        throw std::runtime_error("unexpected trace.csv header in '" + path.string() + "'");
    }

    std::map<int, ReplayFrame> by_step;
    int line_number = 1;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        const auto fields = internal::split_csv(line);
        if (fields.size() != 16) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     " has an unexpected number of columns");
        }
        const int step = internal::parse_int(fields[0], path, line_number);
        if (step < 0) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     " has a negative step index");
        }
        const double time_seconds = internal::parse_double(fields[1], path, line_number);
        const auto [frame_iterator, inserted] = by_step.try_emplace(step);
        ReplayFrame& frame = frame_iterator->second;
        frame.step = step;
        if (inserted) frame.time_seconds = time_seconds;
        else if (frame.time_seconds != time_seconds) {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     " has inconsistent timestamps for one frame");
        }
        ActorState actor;
        actor.id = internal::parse_int(fields[3], path, line_number);
        actor.position = {internal::parse_double(fields[5], path, line_number),
                          internal::parse_double(fields[6], path, line_number)};
        if (fields[2] == "ego") {
            if (actor.id != -1) {
                throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                         " has an invalid ego actor id");
            }
            if (frame.has_ego) {
                throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                         " duplicates an ego state for one frame");
            }
            actor.theta = internal::parse_double(fields[7], path, line_number);
            frame.ego = actor;
            frame.has_ego = true;
        } else if (fields[2] == "obstacle") {
            if (actor.id < 0) {
                throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                         " has a negative obstacle id");
            }
            if (std::any_of(frame.obstacles.begin(), frame.obstacles.end(),
                            [&](const ActorState& existing) { return existing.id == actor.id; })) {
                throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                         " duplicates an obstacle state for one frame");
            }
            actor.vx = internal::parse_double(fields[9], path, line_number);
            actor.vy = internal::parse_double(fields[10], path, line_number);
            frame.obstacles.push_back(actor);
        } else {
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     " has unknown actor '" + fields[2] + "'");
        }
    }

    std::vector<ReplayFrame> frames;
    frames.reserve(by_step.size());
    int previous_step = -1;
    double previous_time_seconds = -std::numeric_limits<double>::infinity();
    for (auto& [step, frame] : by_step) {
        if (!frame.has_ego) {
            throw std::runtime_error("trace frame missing ego state in '" + path.string() + "'");
        }
        if (previous_step >= 0 && step != previous_step + 1) {
            throw std::runtime_error("trace has non-consecutive step indices in '" + path.string() + "'");
        }
        if (!(frame.time_seconds > previous_time_seconds)) {
            throw std::runtime_error("trace has non-monotonic timestamps in '" + path.string() + "'");
        }
        std::sort(frame.obstacles.begin(), frame.obstacles.end(),
                  [](const ActorState& a, const ActorState& b) { return a.id < b.id; });
        frames.push_back(std::move(frame));
        previous_step = step;
        previous_time_seconds = frames.back().time_seconds;
    }
    if (frames.empty()) throw std::runtime_error("RViz trace is empty: '" + path.string() + "'");
    if (frames.front().step != 0) {
        throw std::runtime_error("RViz trace does not start at step zero: '" + path.string() + "'");
    }
    return frames;
}

}  // namespace rviz_replay
}  // namespace dro_mpc

#endif  // DRO_MPC_RVIZ_REPLAY_DATA_HPP
