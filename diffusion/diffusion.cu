#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono> // For benchmarking
#include <cuda_runtime.h>
#include <memory> // For std::unique_ptr
#include "../src/shared_memory_access.hpp"

using SharedMemoryAccess::Fields::c; // concentration
using SharedMemoryAccess::Fields::dt; // time step
using SharedMemoryAccess::Fields::timestep; // simulation time

using ArrayType = std::remove_reference_t<decltype(c)>; // type of array, like float[800][600]
constexpr std::size_t Rows = std::extent<ArrayType, 0>::value;
constexpr std::size_t Cols = std::extent<ArrayType, 1>::value;

ArrayType temp{0}; // local temporary array

__device__ __constant__ float d_dt;
__device__ __constant__ float d_timestep;

// Pitch size width in bytes of the allocated memory for a single row, for memory alignment
// Must be the same for all arrays
size_t pitch;

// Define CUDA block and grid sizes
// For 2D stencil operations
dim3 blockSize(16, 16);
dim3 gridSize((Cols + blockSize.x - 1) / blockSize.x, (Rows + blockSize.y - 1) / blockSize.y);

// Define CUDA block and grid sizes
// For 1D linear operations
const int blockSize1D(256);
const int gridSize1D((std::max(Rows, Cols) + blockSize1D-1) / blockSize1D);

// Allocate device memory with automatic cleanup using std::unique_ptr
auto make_unique_ptr_cuda(){
    return std::unique_ptr<float, decltype(&cudaFree)>(
        [&]{
            float* ptr; 
            cudaMallocPitch((void**)&ptr, &pitch, Cols * sizeof(float), Rows); 
            return ptr;}(), 
            cudaFree
        );
}

// Apply boundary conditions to grid edges
__global__ void apply_boundary_conditions(float* d_c, size_t pitch) {
    constexpr float source_value = 1.0f;
    constexpr float sink_value = 0.0f;

    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < Cols) {
        float* row_start = (float*)((char*)d_c + 0 * pitch);
        row_start[j] = source_value;

        row_start = (float*)((char*)d_c + (Rows - 1) * pitch);
        row_start[j] = sink_value;
    }

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < Rows) {
        float* row_start = (float*)((char*)d_c + i * pitch);
        row_start[0] = row_start[1];
        row_start[Cols - 1] = row_start[Cols - 2];
    }
}

// Perform diffusion step on the grid
__global__ void perform_diffusion(const float* d_c, float* d_temp, size_t pitch) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > 0 && i < Rows - 1 && j > 0 && j < Cols - 1) {
        float* current_row = (float*)((char*)d_c + i * pitch);
        float* upper_row = (float*)((char*)d_c + (i - 1) * pitch);
        float* lower_row = (float*)((char*)d_c + (i + 1) * pitch);

        float* temp_row = (float*)((char*)d_temp + i * pitch);
        temp_row[j] = current_row[j] + d_dt * (
            upper_row[j] + lower_row[j] +
            current_row[j - 1] + current_row[j + 1] -
            4 * current_row[j]
        );
    }
}


int main(int argc, char* argv[]) {
    // Validate and parse command-line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number of iterations> <update_every>" << std::endl;
        return 1;
    }
    int iterations = std::atoi(argv[1]);
    int update_every = std::atoi(argv[2]);
    if (iterations <= 0 || update_every <= 0) {
        std::cerr << "Number of iterations and update_every must be positive integers." << std::endl;
        return 1;
    }

    // Allocate device memory
    auto d_c = make_unique_ptr_cuda();
    auto d_temp = make_unique_ptr_cuda();

    // Copy constants to device
    cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));
    cudaMemcpyToSymbol(d_timestep, &timestep, sizeof(float));

    // Copy initial data to device memory
    cudaMemcpy2D(d_c.get(), pitch, c, Cols * sizeof(float), Cols * sizeof(float), Rows, cudaMemcpyHostToDevice);

    // Start benchmarking
    auto start_time = std::chrono::high_resolution_clock::now();

    // Main computation loop
    for (int iter = 0; iter < iterations; ++iter) {
        perform_diffusion<<<gridSize, blockSize>>>(d_c.get(), d_temp.get(), pitch);
        apply_boundary_conditions<<<gridSize1D, blockSize1D>>>(d_temp.get(), pitch);

        std::swap(d_c, d_temp);
        timestep += dt;

        // Periodically update results on host
        if ((iter + 1) % update_every == 0) {
            cudaDeviceSynchronize();
            cudaMemcpyToSymbol(d_timestep, &timestep, sizeof(float));
            cudaMemcpy2D(c, Cols * sizeof(float), d_c.get(), pitch, Cols * sizeof(float), Rows, cudaMemcpyDeviceToHost);
        }
    }

    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(d_timestep, &timestep, sizeof(float));
    cudaMemcpy2D(c, Cols * sizeof(float), d_c.get(), pitch, Cols * sizeof(float), Rows, cudaMemcpyDeviceToHost);

     // End benchmarking
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Done, " << iterations << " iterations in "
              << elapsed_seconds.count() << " seconds." << std::endl;

    return 0;
}
