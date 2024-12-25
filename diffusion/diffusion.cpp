#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono> // For benchmarking
#include "../src/shared_memory_access.hpp"
#include "../src/eigen_map.hpp"

// Function to perform diffusion
// void performDiffusion(auto& matrix, float dt) {
//     auto temp = SharedMemoryAccess::CopyEigen(matrix); 
//     // Diffusion kernel: simple average of neighbors
//     for (int i = 1; i < matrix.rows() - 1; ++i) {
//         for (int j = 1; j < matrix.cols() - 1; ++j) {
//             temp(i, j) = matrix(i, j) + dt * (
//                 matrix(i - 1, j) + matrix(i + 1, j) +
//                 matrix(i, j - 1) + matrix(i, j + 1) - 4 * matrix(i, j)
//             );
//         }
//     }
//     matrix = temp;
// }

// Vectorized diffusion using Eigen block operations
void performDiffusion(auto& matrix, float dt) {
    auto temp = SharedMemoryAccess::CopyEigen(matrix);

    // Define the block size (excluding the boundary)
    const int rows = matrix.rows() - 2;
    const int cols = matrix.cols() - 2;

    // Apply the diffusion kernel using matrix shifts and block operations
    temp.block(1, 1, rows, cols) =
        matrix.block(1, 1, rows, cols) + dt * (
            matrix.block(0, 1, rows, cols) +  // Top neighbor
            matrix.block(2, 1, rows, cols) +  // Bottom neighbor
            matrix.block(1, 0, rows, cols) +  // Left neighbor
            matrix.block(1, 2, rows, cols) -  // Right neighbor
            4 * matrix.block(1, 1, rows, cols) // Center
        );

    // Assign the result back to the original matrix
    matrix = temp;
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
        // Map shared memory variables
        MAP_SHM(c, c_raw);
        MAP_SHM(dt, dt);

        // Wrap shared memory array to Eigen matrix
        auto c = SharedMemoryAccess::WrapToEigen(c_raw);

        // Benchmark the diffusion process
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            performDiffusion(c, dt);
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
