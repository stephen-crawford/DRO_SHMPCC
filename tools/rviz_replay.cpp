/**
 * @file rviz_replay.cpp
 * @brief ROS 2/RViz replay node for a canonical experiment artifact bundle.
 *
 * This target is intentionally optional.  It consumes scene.csv, geometry.csv,
 * and trace.csv produced by the experiment harness, so RViz playback remains
 * a visualization of the recorded rollout rather than a second simulation.
 */

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "rviz_replay_data.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* FRAME_ID = "map";

using dro_mpc::rviz_replay::ActorState;
using dro_mpc::rviz_replay::Point2D;
using dro_mpc::rviz_replay::ReplayFrame;
using dro_mpc::rviz_replay::ReplayGeometry;
using dro_mpc::rviz_replay::ReplayScene;
using dro_mpc::rviz_replay::load_geometry;
using dro_mpc::rviz_replay::load_scene;
using dro_mpc::rviz_replay::load_trace;

struct Options {
    fs::path artifact_directory;
    double playback_rate = 1.0;
    bool loop = false;
};

geometry_msgs::msg::Point to_ros_point(const Point2D& point, double z = 0.0) {
    geometry_msgs::msg::Point result;
    result.x = point.x;
    result.y = point.y;
    result.z = z;
    return result;
}

void set_color(visualization_msgs::msg::Marker& marker,
               float red, float green, float blue, float alpha = 1.0F) {
    marker.color.r = red;
    marker.color.g = green;
    marker.color.b = blue;
    marker.color.a = alpha;
}

Options parse_options(const std::vector<std::string>& arguments) {
    Options options;
    const std::string executable = arguments.empty()
        ? "dro_mpc_rviz_replay" : arguments.front();
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string& argument = arguments[index];
        const auto need_value = [&](const char* option) -> std::string {
            if (++index >= arguments.size()) {
                throw std::invalid_argument(std::string(option) + " requires a value");
            }
            return arguments[index];
        };
        if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: " << executable
                      << " --artifact DIRECTORY [--rate MULTIPLIER] [--loop]\n";
            std::exit(0);
        }
        if (argument == "--artifact") options.artifact_directory = need_value("--artifact");
        else if (argument == "--rate") {
            try {
                options.playback_rate = std::stod(need_value("--rate"));
            } catch (const std::exception&) {
                throw std::invalid_argument("--rate expects a positive number");
            }
            if (!std::isfinite(options.playback_rate) ||
                !(options.playback_rate > 0.0) || options.playback_rate > 1000.0) {
                throw std::invalid_argument("--rate expects a finite number in (0, 1000]");
            }
        } else if (argument == "--loop") options.loop = true;
        else throw std::invalid_argument("unknown option '" + argument + "'");
    }
    if (options.artifact_directory.empty()) {
        throw std::invalid_argument("--artifact DIRECTORY is required");
    }
    return options;
}

class RvizReplay final : public rclcpp::Node {
public:
    RvizReplay(ReplayScene scene, ReplayGeometry geometry, std::vector<ReplayFrame> frames,
               double playback_rate, bool loop)
        : Node("dro_mpc_rviz_replay"), scene_(std::move(scene)), geometry_(std::move(geometry)),
          frames_(std::move(frames)), playback_rate_(playback_rate), loop_(loop),
          playback_started_(std::chrono::steady_clock::now()) {
        rclcpp::QoS qos(1);
        qos.reliable().transient_local();
        reference_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
            "dro_mpc/reference_path", qos);
        ego_path_publisher_ = create_publisher<nav_msgs::msg::Path>(
            "dro_mpc/ego_path", qos);
        marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "dro_mpc/markers", qos);

        publish_current_frame();
        timer_ = create_wall_timer(std::chrono::milliseconds(10),
                                   std::bind(&RvizReplay::advance, this));
    }

private:
    visualization_msgs::msg::Marker line_marker(
        int id, const std::string& name, const std::vector<Point2D>& points,
        float red, float green, float blue, double width
    ) const {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = FRAME_ID;
        marker.header.stamp = get_clock()->now();
        marker.ns = name;
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = width;
        set_color(marker, red, green, blue);
        marker.points.reserve(points.size());
        for (const auto& point : points) marker.points.push_back(to_ros_point(point, 0.02));
        return marker;
    }

    nav_msgs::msg::Path make_path(const std::vector<Point2D>& points) const {
        nav_msgs::msg::Path path;
        path.header.frame_id = FRAME_ID;
        path.header.stamp = get_clock()->now();
        path.poses.reserve(points.size());
        for (const auto& point : points) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path.header;
            pose.pose.position = to_ros_point(point, 0.05);
            pose.pose.orientation.w = 1.0;
            path.poses.push_back(std::move(pose));
        }
        return path;
    }

    visualization_msgs::msg::Marker disc_marker(
        int id, const std::string& name, const Point2D& position, double radius,
        float red, float green, float blue, float alpha = 1.0F
    ) const {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = FRAME_ID;
        marker.header.stamp = get_clock()->now();
        marker.ns = name;
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.position = to_ros_point(position, 0.15);
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2.0 * radius;
        marker.scale.y = 2.0 * radius;
        marker.scale.z = 0.30;
        set_color(marker, red, green, blue, alpha);
        return marker;
    }

    std::vector<Point2D> ego_disc_positions(const ActorState& ego) const {
        std::vector<Point2D> positions;
        positions.reserve(static_cast<std::size_t>(geometry_.ego_num_discs));
        if (geometry_.ego_num_discs == 1) {
            positions.push_back(ego.position);
            return positions;
        }

        const double spacing = geometry_.ego_length /
            static_cast<double>(geometry_.ego_num_discs - 1);
        const double cosine = std::cos(ego.theta);
        const double sine = std::sin(ego.theta);
        for (int disc = 0; disc < geometry_.ego_num_discs; ++disc) {
            const double offset = -0.5 * geometry_.ego_length +
                static_cast<double>(disc) * spacing;
            positions.push_back({ego.position.x + offset * cosine,
                                 ego.position.y + offset * sine});
        }
        return positions;
    }

    void publish_current_frame() {
        const auto& frame = frames_[frame_index_];
        reference_path_publisher_->publish(make_path(scene_.route));

        std::vector<Point2D> ego_history;
        ego_history.reserve(frame_index_ + 1);
        for (std::size_t index = 0; index <= frame_index_; ++index) {
            ego_history.push_back(frames_[index].ego.position);
        }
        ego_path_publisher_->publish(make_path(ego_history));

        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker clear;
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        markers.markers.push_back(clear);
        int marker_id = 0;
        for (const auto& road : scene_.roads) {
            markers.markers.push_back(line_marker(
                marker_id++, "road", road, 0.39F, 0.45F, 0.51F, 0.15));
        }
        markers.markers.push_back(line_marker(
            marker_id++, "route", scene_.route, 0.34F, 0.85F, 0.64F, 0.10));
        for (const auto& disc : ego_disc_positions(frame.ego)) {
            markers.markers.push_back(disc_marker(
                marker_id++, "ego_collision_disc", disc, geometry_.ego_radius,
                0.35F, 0.65F, 1.0F));
        }
        const std::vector<std::array<float, 3>> obstacle_colors = {
            {1.0F, 0.55F, 0.26F}, {0.97F, 0.47F, 0.73F},
            {0.82F, 0.60F, 0.13F}, {0.64F, 0.44F, 0.97F}};
        for (const auto& [key, points] : frame.sampled_scenarios) {
            const auto& color = obstacle_colors[static_cast<size_t>(std::max(0, key.first)) % obstacle_colors.size()];
            markers.markers.push_back(line_marker(marker_id++, "sampled_scenarios", points,
                color[0], color[1], color[2], 0.025));
        }
        for (const auto& obstacle : frame.obstacles) {
            const auto& color = obstacle_colors[
                static_cast<std::size_t>(std::max(0, obstacle.id)) % obstacle_colors.size()];
            // This radius is the exact forbidden center region for an ego collision disc:
            // obstacle radius + ego-disc radius + configured safety margin.
            markers.markers.push_back(disc_marker(
                marker_id++, "obstacle_clearance_boundary", obstacle.position,
                geometry_.obstacle_radius + geometry_.ego_radius + geometry_.safety_margin,
                color[0], color[1], color[2], 0.16F));
            markers.markers.push_back(disc_marker(
                marker_id++, "obstacle", obstacle.position, geometry_.obstacle_radius,
                color[0], color[1], color[2]));
        }
        marker_publisher_->publish(markers);
    }

    void advance() {
        const double elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - playback_started_).count() * playback_rate_;
        const double target_time = frames_.front().time_seconds + elapsed_seconds;
        bool changed = false;
        while (frame_index_ + 1 < frames_.size() &&
               frames_[frame_index_ + 1].time_seconds <= target_time) {
            ++frame_index_;
            changed = true;
        }
        if (changed) publish_current_frame();

        if (frame_index_ + 1 != frames_.size()) {
            return;
        }
        const double final_frame_hold_seconds = frames_.size() > 1
            ? frames_.back().time_seconds - frames_[frames_.size() - 2].time_seconds
            : 0.0;
        if (target_time < frames_.back().time_seconds + final_frame_hold_seconds) return;
        if (loop_ && frames_.size() > 1) {
            frame_index_ = 0;
            playback_started_ = std::chrono::steady_clock::now();
            publish_current_frame();
        } else {
            timer_->cancel();
        }
    }

    ReplayScene scene_;
    ReplayGeometry geometry_;
    std::vector<ReplayFrame> frames_;
    double playback_rate_ = 1.0;
    bool loop_ = false;
    std::size_t frame_index_ = 0;
    std::chrono::steady_clock::time_point playback_started_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr reference_path_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ego_path_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        // Keep standard ROS arguments (namespace, remapping, parameters) out of
        // the application parser while still passing them to rclcpp::init below.
        const std::vector<std::string> application_arguments =
            rclcpp::remove_ros_arguments(argc, argv);
        const Options options = parse_options(application_arguments);
        const ReplayScene scene = load_scene(options.artifact_directory / "scene.csv");
        const ReplayGeometry geometry = load_geometry(options.artifact_directory / "geometry.csv");
        std::vector<ReplayFrame> trace = load_trace(options.artifact_directory / "trace.csv");
        dro_mpc::rviz_replay::load_sampled_scenarios(options.artifact_directory / "sampled_scenarios.csv", trace);
        rclcpp::init(argc, argv);
        const auto node = std::make_shared<RvizReplay>(
            scene, geometry, trace, options.playback_rate, options.loop);
        RCLCPP_INFO(node->get_logger(), "replaying %zu frames from %s",
                    trace.size(), options.artifact_directory.c_str());
        rclcpp::spin(node);
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dro_mpc_rviz_replay: " << error.what() << '\n';
        if (rclcpp::ok()) rclcpp::shutdown();
        return 1;
    }
}
