//g++ -std=c++20 -O3 diffusion/diffusion.cpp -o diffusion/bin/diffusion
#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono> // For benchmarking
#include "../src/shared_memory_access.hpp"


// Apply boundary conditions
template <typename ArrayType>
void apply_boundary_conditions(ArrayType& c, float source_value = 1.0f, float sink_value = 0.0f) {
    // Left boundary (source)
    using T = typename std::remove_all_extents<ArrayType>::type; // Deduce scalar type (e.g., float)
    constexpr std::size_t Rows = std::extent<ArrayType, 0>::value; // Deduce rows
    constexpr std::size_t Cols = std::extent<ArrayType, 1>::value; // Deduce columns
    // Source left, sink right

    for (size_t j = 0; j < Cols; ++j) {
        c[0][j] = static_cast<T>(source_value);
        c[Rows-1][j] = static_cast<T>(sink_value);
    }

    // Top boundary (mirror condition)
    for (size_t i = 0; i < Rows; ++i) {
        c[i][0] = c[i][1];
        c[i][Cols-1] = c[i][Cols-2];
    }
}

//Function to perform diffusion
template <typename ArrayType>
void performDiffusion(ArrayType& matrix, const float dt, const int iterations) {
    ArrayType temp{0};

    constexpr std::size_t Rows = std::extent<ArrayType, 0>::value;
    constexpr std::size_t Cols = std::extent<ArrayType, 1>::value;
    
    for (int iter = 0; iter < iterations; ++iter){
        #pragma omp parallel for num_threads(12)
        for (int i = 1; i < Rows - 1; ++i) {
            for (int j = 1; j < Cols - 1; ++j) {
                temp[i][j] = matrix[i][j] + dt * (
                    matrix[i-1][j] + matrix[i+1][j] +
                    matrix[i][j - 1] + matrix[i][j + 1] 
                    - 4 * matrix[i][j]
                );
            }
        }

        std::memcpy(matrix, temp, sizeof(matrix));
        apply_boundary_conditions(matrix);
    }

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
        
        using SharedMemoryAccess::Fields::c;
        using SharedMemoryAccess::Fields::dt;
        using SharedMemoryAccess::Fields::timestep;

        // Benchmark the diffusion process
        auto start_time = std::chrono::high_resolution_clock::now();

        performDiffusion(c, dt, iterations);
        timestep = timestep + dt*iterations;

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
