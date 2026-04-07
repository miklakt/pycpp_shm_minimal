#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "../src/shared_memory_access.hpp"

using SharedMemoryAccess::Fields::dt;
using SharedMemoryAccess::Fields::mass;
using SharedMemoryAccess::Fields::oscillator_frequency;
using SharedMemoryAccess::Fields::spring_k;
using SharedMemoryAccess::Fields::timestep;
using SharedMemoryAccess::Fields::z;
using SharedMemoryAccess::Fields::z_prev;

using ArrayType = std::remove_reference_t<decltype(z)>;
constexpr std::size_t Rows = std::extent<ArrayType, 0>::value;
constexpr std::size_t Cols = std::extent<ArrayType, 1>::value;
constexpr int SourceCol = 2;
constexpr int SourceWidth = 2;
constexpr float TwoPi = 6.28318530717958647692f;

ArrayType next{};

inline void apply_absorbing_boundaries() {
    const int last_row = static_cast<int>(Rows) - 1;
    const int last_col = static_cast<int>(Cols) - 1;
    const float wave_speed = std::sqrt(std::max(0.0f, spring_k));
    const float denom = wave_speed * dt + 1.0f;
    const float r = denom > 0.0f ? (wave_speed * dt - 1.0f) / denom : 0.0f;

    for (int j = 1; j < last_col; ++j) {
        next[0][j] = z[1][j] + r * (next[1][j] - z[0][j]);
        next[last_row][j] = z[last_row - 1][j] + r * (next[last_row - 1][j] - z[last_row][j]);
    }

    for (int i = 1; i < last_row; ++i) {
        next[i][0] = z[i][1] + r * (next[i][1] - z[i][0]);
        next[i][last_col] = z[i][last_col - 1] + r * (next[i][last_col - 1] - z[i][last_col]);
    }

    next[0][0] = 0.5f * (next[0][1] + next[1][0]);
    next[0][last_col] = 0.5f * (next[0][last_col - 1] + next[1][last_col]);
    next[last_row][0] = 0.5f * (next[last_row - 1][0] + next[last_row][1]);
    next[last_row][last_col] = 0.5f * (next[last_row - 1][last_col] + next[last_row][last_col - 1]);
}

inline void apply_source(ArrayType& field, float value) {
    const int half_width = SourceWidth / 2;
    for (int i = 0; i < static_cast<int>(Rows); ++i) {
        for (int dj = -half_width; dj < half_width; ++dj) {
            const int j = SourceCol + dj;
            if (j < 0 || j >= static_cast<int>(Cols)) {
                continue;
            }
            field[i][j] = value;
        }
    }
}

inline void step_wave() {
    const float omega = TwoPi * oscillator_frequency;
    const float next_source = std::sin(omega * (timestep + dt));

#pragma omp parallel for num_threads(4)
    for (int i = 1; i < static_cast<int>(Rows) - 1; ++i) {
        for (int j = 1; j < static_cast<int>(Cols) - 1; ++j) {
            const float zc = z[i][j];
            const float m = mass[i][j];

            // Infinite mass means a pinned node.
            if (!std::isfinite(m)) {
                next[i][j] = zc;
                continue;
            }

            const float lap =
                z[i - 1][j] +
                z[i + 1][j] +
                z[i][j - 1] +
                z[i][j + 1] -
                4.0f * zc;

            next[i][j] = 2.0f * zc - z_prev[i][j] + spring_k * dt * dt * lap / m;
        }
    }

    apply_source(next, next_source);
    apply_absorbing_boundaries();
    std::memcpy(z_prev, z, sizeof(z));
    std::memcpy(z, next, sizeof(z));
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number of iterations>" << std::endl;
        return 1;
    }

    const int iterations = std::atoi(argv[1]);
    if (iterations <= 0) {
        std::cerr << "Number of iterations must be a positive integer." << std::endl;
        return 1;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            step_wave();
            timestep = timestep + dt;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        std::cout << "Done, " << iterations << " iterations in "
                  << elapsed_seconds.count() << " seconds." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
