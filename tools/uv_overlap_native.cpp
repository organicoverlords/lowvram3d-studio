#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
constexpr int GRID_SIZE = 256;
constexpr double AREA_EPSILON_UV = 0.25 / (2048.0 * 2048.0);
constexpr double COORD_EPSILON = 1e-10;
constexpr double UV_LOWER_BOUND = -1e-6;
constexpr double UV_UPPER_BOUND = 1.000001;

struct Point { double x; double y; };
struct Triangle { Point p[3]; };
struct Report {
    std::size_t candidate_pairs = 0;
    std::size_t tested_pairs = 0;
    std::size_t positive_pairs = 0;
    double total_area = 0.0;
    double max_area = 0.0;
    std::size_t degenerate = 0;
    std::size_t out_of_bounds = 0;
    std::size_t ignored = 0;
    bool timed_out = false;
    bool failed = false;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> positive_pair_list;
};

double signed_area(const Triangle& t) {
    return 0.5 * ((t.p[1].x - t.p[0].x) * (t.p[2].y - t.p[0].y)
        - (t.p[2].x - t.p[0].x) * (t.p[1].y - t.p[0].y));
}

double polygon_area(const std::vector<Point>& polygon) {
    if (polygon.size() < 3) return 0.0;
    double first = 0.0, second = 0.0;
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const Point& a = polygon[i];
        const Point& b = polygon[(i + 1) % polygon.size()];
        first += a.x * b.y;
        second += a.y * b.x;
    }
    return std::abs(first - second) * 0.5;
}

std::vector<Point> clip_convex(const Triangle& subject, const Triangle& raw_clipper) {
    Triangle clipper = raw_clipper;
    const double area = signed_area(clipper);
    if (area < 0.0) std::swap(clipper.p[0], clipper.p[2]);
    std::vector<Point> output = {subject.p[0], subject.p[1], subject.p[2]};
    for (int edge_index = 0; edge_index < 3; ++edge_index) {
        if (output.empty()) return {};
        const Point start = clipper.p[edge_index];
        const Point end = clipper.p[(edge_index + 1) % 3];
        const Point edge{end.x - start.x, end.y - start.y};
        std::vector<Point> clipped;
        const std::size_t count = output.size();
        auto distance = [&](const Point& point) {
            return edge.x * (point.y - start.y) - edge.y * (point.x - start.x);
        };
        for (std::size_t current = 0; current < count; ++current) {
            const std::size_t following = (current + 1) % count;
            const double current_distance = distance(output[current]);
            const double following_distance = distance(output[following]);
            const bool current_inside = current_distance >= -COORD_EPSILON;
            const bool following_inside = following_distance >= -COORD_EPSILON;
            if (current_inside) clipped.push_back(output[current]);
            if (current_inside != following_inside) {
                const double denominator = current_distance - following_distance;
                if (std::abs(denominator) > COORD_EPSILON) {
                    const double t = current_distance / denominator;
                    clipped.push_back({
                        output[current].x + t * (output[following].x - output[current].x),
                        output[current].y + t * (output[following].y - output[current].y),
                    });
                }
            }
        }
        output.swap(clipped);
    }
    return output;
}

bool separated(const Triangle& a, const Triangle& b) {
    double aminx = a.p[0].x, amaxx = a.p[0].x, aminy = a.p[0].y, amaxy = a.p[0].y;
    double bminx = b.p[0].x, bmaxx = b.p[0].x, bminy = b.p[0].y, bmaxy = b.p[0].y;
    for (int i = 1; i < 3; ++i) {
        aminx = std::min(aminx, a.p[i].x); amaxx = std::max(amaxx, a.p[i].x);
        aminy = std::min(aminy, a.p[i].y); amaxy = std::max(amaxy, a.p[i].y);
        bminx = std::min(bminx, b.p[i].x); bmaxx = std::max(bmaxx, b.p[i].x);
        bminy = std::min(bminy, b.p[i].y); bmaxy = std::max(bmaxy, b.p[i].y);
    }
    return amaxx < bminx - COORD_EPSILON || bmaxx < aminx - COORD_EPSILON
        || amaxy < bminy - COORD_EPSILON || bmaxy < aminy - COORD_EPSILON;
}

bool timed_out(const std::chrono::steady_clock::time_point& started, double seconds) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() > seconds;
}

Report detect(const std::vector<Triangle>& triangles, int resolution, double timeout,
              std::size_t max_candidate_pairs, bool collect_pairs) {
    Report report;
    std::vector<std::uint8_t> testable(triangles.size(), 1);
    std::vector<std::array<int, 4>> cells(triangles.size());
    std::vector<std::vector<std::uint32_t>> buckets(GRID_SIZE * GRID_SIZE);
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        bool out = false;
        double lowx = triangles[i].p[0].x, highx = lowx;
        double lowy = triangles[i].p[0].y, highy = lowy;
        for (int j = 0; j < 3; ++j) {
            const Point& p = triangles[i].p[j];
            out = out || p.x < UV_LOWER_BOUND || p.x > UV_UPPER_BOUND
                || p.y < UV_LOWER_BOUND || p.y > UV_UPPER_BOUND;
            lowx = std::min(lowx, p.x); highx = std::max(highx, p.x);
            lowy = std::min(lowy, p.y); highy = std::max(highy, p.y);
        }
        if (out) { ++report.out_of_bounds; testable[i] = 0; continue; }
        if (std::abs(signed_area(triangles[i])) <= AREA_EPSILON_UV) {
            ++report.degenerate; testable[i] = 0; continue;
        }
        auto cell = [](double value) {
            long long result = static_cast<long long>(value * GRID_SIZE);
            return static_cast<int>(std::max<long long>(0, std::min<long long>(GRID_SIZE - 1, result)));
        };
        cells[i] = {cell(lowx), cell(highx), cell(lowy), cell(highy)};
        for (int x = cells[i][0]; x <= cells[i][1]; ++x)
            for (int y = cells[i][2]; y <= cells[i][3]; ++y)
                buckets[x * GRID_SIZE + y].push_back(static_cast<std::uint32_t>(i));
    }

    const auto started = std::chrono::steady_clock::now();
    std::unordered_set<std::uint64_t> candidate_set;
    for (const auto& members : buckets) {
        if (members.size() < 2) continue;
        for (std::size_t i = 0; i + 1 < members.size(); ++i) {
            for (std::size_t j = i + 1; j < members.size(); ++j) {
                std::uint32_t a = members[i], b = members[j];
                if (a > b) std::swap(a, b);
                candidate_set.insert((static_cast<std::uint64_t>(a) << 32) | b);
                if (candidate_set.size() > max_candidate_pairs) {
                    report.candidate_pairs = candidate_set.size(); report.failed = true; return report;
                }
            }
        }
        if (timed_out(started, timeout)) { report.timed_out = true; report.failed = true; return report; }
    }
    report.candidate_pairs = candidate_set.size();
    std::vector<std::uint64_t> ordered(candidate_set.begin(), candidate_set.end());
    std::sort(ordered.begin(), ordered.end());
    for (const std::uint64_t key : ordered) {
        if (timed_out(started, timeout)) { report.timed_out = true; report.failed = true; return report; }
        const std::size_t first = static_cast<std::size_t>(key >> 32);
        const std::size_t second = static_cast<std::size_t>(key & 0xffffffffu);
        if (separated(triangles[first], triangles[second])) continue;
        ++report.tested_pairs;
        const double area = polygon_area(clip_convex(triangles[first], triangles[second]));
        if (area > AREA_EPSILON_UV) {
            ++report.positive_pairs; report.total_area += area; report.max_area = std::max(report.max_area, area);
            if (collect_pairs) report.positive_pair_list.emplace_back(static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(second));
        } else if (area > 0.0) ++report.ignored;
    }
    return report;
}

void emit(const Report& report, int resolution) {
    std::cout << std::setprecision(17);
    std::cout << "candidate_pair_count=" << report.candidate_pairs << '\n'
        << "tested_pair_count=" << report.tested_pairs << '\n'
        << "positive_overlap_pair_count=" << report.positive_pairs << '\n'
        << "positive_overlap_total_area_uv=" << report.total_area << '\n'
        << "positive_overlap_max_area_uv=" << report.max_area << '\n'
        << "positive_overlap_total_texels_equivalent=" << report.total_area * resolution * resolution << '\n'
        << "degenerate_uv_triangle_count=" << report.degenerate << '\n'
        << "out_of_bounds_triangle_count=" << report.out_of_bounds << '\n'
        << "ignored_noise_intersection_count=" << report.ignored << '\n'
        << "timed_out=" << (report.timed_out ? 1 : 0) << '\n'
        << "success=" << ((!report.failed && !report.timed_out) ? 1 : 0) << '\n';
    for (const auto& pair : report.positive_pair_list)
        std::cout << "positive_pair=" << pair.first << ',' << pair.second << '\n';
}
}

int main(int argc, char** argv) {
    if (argc != 7) return 2;
    const char* input = argv[1];
    const int resolution = std::stoi(argv[2]);
    const double timeout = std::stod(argv[3]);
    const std::size_t max_pairs = static_cast<std::size_t>(std::stoull(argv[4]));
    const bool collect_pairs = std::stoi(argv[5]) != 0;
    const char* output = argv[6];
    std::ifstream stream(input, std::ios::binary);
    std::uint64_t count = 0;
    if (!stream.read(reinterpret_cast<char*>(&count), sizeof(count))) return 3;
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(Triangle))) return 4;
    std::vector<Triangle> triangles(static_cast<std::size_t>(count));
    if (!stream.read(reinterpret_cast<char*>(triangles.data()), static_cast<std::streamsize>(count * sizeof(Triangle)))) return 5;
    const Report report = detect(triangles, resolution, timeout, max_pairs, collect_pairs);
    std::ofstream out(output, std::ios::trunc);
    if (!out) return 6;
    auto old = std::cout.rdbuf(out.rdbuf());
    emit(report, resolution);
    std::cout.rdbuf(old);
    return 0;
}
