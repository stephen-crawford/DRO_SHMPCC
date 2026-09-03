#include "experiment_harness.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace dro_mpc;

namespace {
struct Case {
    EnvironmentType type;
    const char* name;
    bool expects_lateral_variation;
    bool expects_closed_route;
    ReferencePath::PathType expected_path_type;
};

int failures = 0;

void check(bool condition, const std::string& message) {
    std::cout << (condition ? "PASS: " : "FAIL: ") << message << '\n';
    if (!condition) ++failures;
}

void write_svg(const std::vector<std::pair<Case, ReferencePath>>& paths,
               const std::vector<std::vector<ReferencePath>>& roads,
               const std::string& filename) {
    constexpr double cell_width = 300.0;
    constexpr double cell_height = 220.0;
    constexpr int columns = 3;
    std::ofstream svg(filename);
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"880\" "
           "viewBox=\"0 0 900 880\">\n"
           "<rect width=\"100%\" height=\"100%\" fill=\"#101820\"/>\n"
           "<style>text{font-family:monospace;fill:#e6edf3;font-size:14px}"
           ".grid{stroke:#31404e;stroke-width:1}.road{fill:none;stroke:#6c7a89;"
           "stroke-width:2;stroke-linejoin:round;stroke-linecap:round}.route{fill:none;stroke:#57d9a3;"
           "stroke-width:3;stroke-linejoin:round;stroke-linecap:round}</style>\n";
    for (size_t i = 0; i < paths.size(); ++i) {
        const int col = static_cast<int>(i % columns);
        const int row = static_cast<int>(i / columns);
        const double x0 = col * cell_width;
        const double y0 = row * cell_height;
        const auto& [test_case, path] = paths[i];
        const auto& points = path.points();
        double min_x = points.front().position.x(), max_x = min_x;
        double min_y = points.front().position.y(), max_y = min_y;
        for (const auto& road : roads[i]) for (const auto& point : road.points()) {
            min_x = std::min(min_x, point.position.x()); max_x = std::max(max_x, point.position.x());
            min_y = std::min(min_y, point.position.y()); max_y = std::max(max_y, point.position.y());
        }
        const double span_x = std::max(1.0, max_x - min_x);
        const double span_y = std::max(1.0, max_y - min_y);
        const double scale = std::min(240.0 / span_x, 140.0 / span_y);
        const double offset_x = x0 + 30.0 + 0.5 * (240.0 - scale * span_x);
        const double offset_y = y0 + 175.0 - 0.5 * (140.0 - scale * span_y);
        svg << "<rect class=\"grid\" x=\"" << x0 + 5 << "\" y=\"" << y0 + 5
            << "\" width=\"290\" height=\"210\" fill=\"none\"/>\n";
        svg << "<text x=\"" << x0 + 15 << "\" y=\"" << y0 + 28 << "\">"
            << test_case.name << "</text>\n";
        for (size_t r = 1; r < roads[i].size(); ++r) {
            svg << "<polyline class=\"road\" points=\"";
            for (const auto& point : roads[i][r].points()) {
                const double x = offset_x + scale * (point.position.x() - min_x);
                const double y = offset_y - scale * (point.position.y() - min_y);
                svg << x << ',' << y << ' ';
            }
            svg << "\"/>\n";
        }
        svg << "<polyline class=\"route\" points=\"";
        for (const auto& point : points) {
            const double x = offset_x + scale * (point.position.x() - min_x);
            const double y = offset_y - scale * (point.position.y() - min_y);
            svg << x << ',' << y << ' ';
        }
        svg << "\"/>\n";
    }
    svg << "</svg>\n";
}

void write_bmp(const std::vector<std::pair<Case, ReferencePath>>& paths,
               const std::vector<std::vector<ReferencePath>>& roads,
               const std::string& filename) {
    constexpr int width = 900, height = 880, cell_width = 300, cell_height = 220;
    std::vector<std::array<unsigned char, 3>> pixels(width * height, {16, 24, 32});
    const auto pixel = [&](int x, int y, std::array<unsigned char, 3> color) {
        if (x >= 0 && x < width && y >= 0 && y < height) pixels[y * width + x] = color;
    };
    const auto line = [&](int x0, int y0, int x1, int y1, std::array<unsigned char, 3> color) {
        const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int error = dx + dy;
        while (true) {
            for (int ox = -1; ox <= 1; ++ox)
                for (int oy = -1; oy <= 1; ++oy) pixel(x0 + ox, y0 + oy, color);
            if (x0 == x1 && y0 == y1) break;
            const int twice_error = 2 * error;
            if (twice_error >= dy) { error += dy; x0 += sx; }
            if (twice_error <= dx) { error += dx; y0 += sy; }
        }
    };
    for (size_t i = 0; i < paths.size(); ++i) {
        const auto& points = paths[i].second.points();
        double min_x = points.front().position.x(), max_x = min_x;
        double min_y = points.front().position.y(), max_y = min_y;
        for (const auto& road : roads[i]) for (const auto& point : road.points()) {
            min_x = std::min(min_x, point.position.x()); max_x = std::max(max_x, point.position.x());
            min_y = std::min(min_y, point.position.y()); max_y = std::max(max_y, point.position.y());
        }
        const double scale = std::min(240.0 / std::max(1.0, max_x - min_x),
                                      140.0 / std::max(1.0, max_y - min_y));
        const int x0 = static_cast<int>((i % 3) * cell_width + 30.0 + 0.5 * (240.0 - scale * (max_x - min_x)));
        const int y0 = static_cast<int>((i / 3) * cell_height + 175.0 - 0.5 * (140.0 - scale * (max_y - min_y)));
        const auto draw_path = [&](const ReferencePath& path, std::array<unsigned char, 3> color) {
            const auto& draw_points = path.points();
            for (size_t p = 1; p < draw_points.size(); ++p) {
                const auto to_pixel = [&](const Eigen::Vector2d& point) {
                    return std::pair<int, int>(
                        x0 + static_cast<int>(scale * (point.x() - min_x)),
                        y0 - static_cast<int>(scale * (point.y() - min_y)));
                };
                const auto a = to_pixel(draw_points[p - 1].position);
                const auto b = to_pixel(draw_points[p].position);
                line(a.first, a.second, b.first, b.second, color);
            }
        };
        for (size_t r = 1; r < roads[i].size(); ++r) draw_path(roads[i][r], {108, 122, 137});
        draw_path(paths[i].second, {87, 217, 163});
    }
    const uint32_t row_size = (width * 3 + 3) & ~3U;
    const uint32_t pixel_bytes = row_size * height;
    const uint32_t file_size = 54 + pixel_bytes;
    std::ofstream bmp(filename, std::ios::binary);
    const auto write_u16 = [&](uint16_t value) {
        bmp.put(static_cast<char>(value & 0xff)); bmp.put(static_cast<char>((value >> 8) & 0xff));
    };
    const auto write_u32 = [&](uint32_t value) {
        for (int byte = 0; byte < 4; ++byte) bmp.put(static_cast<char>((value >> (8 * byte)) & 0xff));
    };
    bmp.put('B'); bmp.put('M'); write_u32(file_size); write_u32(0); write_u32(54);
    write_u32(40); write_u32(width); write_u32(height); write_u16(1); write_u16(24);
    write_u32(0); write_u32(pixel_bytes); write_u32(0); write_u32(0); write_u32(0); write_u32(0);
    const std::array<unsigned char, 3> padding = {0, 0, 0};
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            const auto& color = pixels[y * width + x];
            bmp.put(static_cast<char>(color[2])); bmp.put(static_cast<char>(color[1]));
            bmp.put(static_cast<char>(color[0]));
        }
        bmp.write(reinterpret_cast<const char*>(padding.data()), row_size - width * 3);
    }
}
}  // namespace

int main(int argc, char** argv) {
    const std::string output = argc > 1 ? argv[1] : "/tmp/environment_topologies.svg";
    const std::string bitmap_output = argc > 2 ? argv[2] : "/tmp/environment_topologies.bmp";
    EnvironmentExperimentConfig config;
    config.road_length = 90.0;
    config.lane_width = 3.6;
    config.intersection_box_size = 24.0;
    config.corner_radius = 9.0;
    config.ramp_length = 55.0;
    config.merge_length = 35.0;
    config.roundabout_radius = 20.0;
    config.s_curve_length = 50.0;
    config.s_curve_amplitude = 5.0;

    const std::vector<Case> cases = {
        {EnvironmentType::T_INTERSECTION, "T intersection", true, false, ReferencePath::PathType::CUSTOM},
        {EnvironmentType::FOUR_WAY_INTERSECTION, "Four-way intersection", true, false, ReferencePath::PathType::STRAIGHT},
        {EnvironmentType::S_CURVE, "S curve", true, false, ReferencePath::PathType::S_CURVE},
        {EnvironmentType::TWO_LANE_ROUNDABOUT, "Two-lane roundabout", true, true, ReferencePath::PathType::CIRCLE},
        {EnvironmentType::FOUR_LANE_ROUNDABOUT, "Four-lane roundabout", true, true, ReferencePath::PathType::CIRCLE},
        {EnvironmentType::TWO_LANE_HIGHWAY, "Two-lane highway", false, false, ReferencePath::PathType::STRAIGHT},
        {EnvironmentType::FOUR_LANE_HIGHWAY, "Four-lane highway", false, false, ReferencePath::PathType::STRAIGHT},
        {EnvironmentType::ENTER_RAMP, "Enter ramp", true, false, ReferencePath::PathType::CUSTOM},
        {EnvironmentType::EXIT_RAMP, "Exit ramp", true, false, ReferencePath::PathType::CUSTOM},
        {EnvironmentType::OVERTAKE_SLOW_LEAD, "Overtake", false, false, ReferencePath::PathType::STRAIGHT},
        {EnvironmentType::NARROW_CORRIDOR, "Narrow corridor", false, false, ReferencePath::PathType::STRAIGHT},
        {EnvironmentType::ONCOMING, "Oncoming", false, false, ReferencePath::PathType::STRAIGHT},
        {EnvironmentType::INTERSECTION, "Intersection", true, false, ReferencePath::PathType::STRAIGHT},
    };

    std::vector<std::pair<Case, ReferencePath>> paths;
    std::vector<std::vector<ReferencePath>> roads;
    for (const Case& test_case : cases) {
        auto path = build_environment_reference_path(test_case.type, config);
        auto road_centerlines = build_environment_road_centerlines(test_case.type, config);
        const auto& points = path.points();
        double min_y = points.front().position.y(), max_y = min_y;
        for (const auto& point : points) {
            min_y = std::min(min_y, point.position.y());
            max_y = std::max(max_y, point.position.y());
        }
        check(path.num_points() >= 2 && path.total_length() > 1.0,
              std::string(test_case.name) + " has a usable route");
        check(path.path_type() == test_case.expected_path_type,
              std::string(test_case.name) + " has the expected topology class");
        check(test_case.expects_lateral_variation ? (max_y - min_y > 0.5) :
                                                   (max_y - min_y < 1e-8),
              std::string(test_case.name) + " has the expected lateral shape");
        const double endpoint_gap = (points.front().position - points.back().position).norm();
        check(test_case.expects_closed_route ? endpoint_gap < 1e-8 : endpoint_gap >= 1e-8,
              std::string(test_case.name) + (test_case.expects_closed_route ?
                  " closes at its starting point" : " remains an open route"));
        const bool is_intersection = test_case.type == EnvironmentType::T_INTERSECTION ||
            test_case.type == EnvironmentType::FOUR_WAY_INTERSECTION ||
            test_case.type == EnvironmentType::INTERSECTION;
        check(!is_intersection || road_centerlines.size() >= 2,
              std::string(test_case.name) + " includes perpendicular road centerlines");
        paths.emplace_back(test_case, std::move(path));
        roads.push_back(std::move(road_centerlines));
    }
    write_svg(paths, roads, output);
    write_bmp(paths, roads, bitmap_output);
    std::cout << "Wrote topology visualization: " << output << '\n';
    std::cout << "Wrote topology raster: " << bitmap_output << '\n';
    return failures == 0 ? 0 : 1;
}
