//g++ -std=c++20 -O3 diffusion/diffusion.cpp -o diffusion/bin/diffusion
#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono> // For benchmarking
#include <algorithm>
#include "../src/shared_memory_access.hpp"


using SharedMemoryAccess::Fields::c; //concentration
using SharedMemoryAccess::Fields::dt; //time step
using SharedMemoryAccess::Fields::timestep; //simulation time

using ArrayType = std::remove_reference_t<decltype(c)>; //type of array, like float[800][600]
constexpr std::size_t Rows = std::extent<ArrayType, 0>::value;
constexpr std::size_t Cols = std::extent<ArrayType, 1>::value;

ArrayType temp{0}; //local temporary array



// Apply boundary conditions
inline void apply_boundary_conditions(){
    constexpr float source_value = 1.0f; 
    constexpr float sink_value = 0.0f;
    // Source left, sink right
    for (size_t j = 0; j < Cols; ++j) {
        c[0][j] = source_value;
        c[Rows-1][j] = sink_value;
    }

    // Top boundary (mirror condition)
    for (size_t i = 0; i < Rows; ++i) {
        c[i][0] = c[i][1];
        c[i][Cols-1] = c[i][Cols-2];
    }
}

inline void perform_diffusion(const int iterations){
    #pragma omp parallel for num_threads(4)
    for (int i = 1; i < Rows - 1; ++i) {
        for (int j = 1; j < Cols - 1; ++j) {
            temp[i][j] = c[i][j] + dt * (
                c[i-1][j] + c[i+1][j] +
                c[i][j - 1] + c[i][j + 1] 
                - 4 * c[i][j]
            );
        }
    }
    std::memcpy(c, temp, sizeof(c));
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number of iterations>" << std::endl;
        return 1;
    }

    int iterations = std::atoi(argv[1]);
    if (iterations <= 0) {
        std::cerr << "Number of iterations must be a positive integer." << std::endl;
        return 1;
    }

    try {
        
        // Benchmark the code
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter){
            perform_diffusion(iterations);
            apply_boundary_conditions();
            timestep = timestep + dt;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        // Output the benchmark results
        std::cout << "Done, " << iterations << " iterations in "
                  << elapsed_seconds.count() << " seconds." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
