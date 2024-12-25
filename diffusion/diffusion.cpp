#include <iostream>
#include <cstdlib> // For std::atoi
#include "../src/shared_memory_access.hpp"
#include "../src/eigen_map.hpp"

void performDiffusion(auto& matrix, float dt) {
    // Copy can not be don by simple assignment operator since 'matrix' is a Map object not a PlainObject as 'temp'
    auto temp = SharedMemoryAccess::CopyEigen(matrix); 
    // Diffusion kernel: simple average of neighbors
    for (int i = 1; i < matrix.rows() - 1; ++i) {
        for (int j = 1; j < matrix.cols() - 1; ++j) {
            temp(i, j) = matrix(i, j) + dt * (
                matrix(i - 1, j) + matrix(i + 1, j) +
                matrix(i, j - 1) + matrix(i, j + 1) - 4 * matrix(i, j)
            );
        }
    }
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
        // Create constant pointers to mutable fixed-length multidimensional arrays in the shared memory
        // In the shared memory there are an array c[1000][1000] and a float dt
        MAP_SHM(c,c_raw);
        MAP_SHM(dt,dt);


        // To make manipulation easier the data is mapped to Eigen::Matrix<float, rows=1000, cols=1000>
        auto c = SharedMemoryAccess::WrapToEigen(c_raw);

        // Perform the diffusion process
        for (int iter = 0; iter < iterations; ++iter, performDiffusion(c, dt));

        // Output the final state of the matrix
        std::cout << "Done, " << iterations << " iterations:" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
