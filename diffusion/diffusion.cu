#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono> // For benchmarking
#include <cuda_runtime.h>
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

// Apply boundary conditions
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

// Perform diffusion
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

    dim3 blockSize(16, 16);
    dim3 gridSize((Cols + blockSize.x - 1) / blockSize.x, (Rows + blockSize.y - 1) / blockSize.y);

    float* d_c;
    float* d_temp;
    size_t pitch; //width of a row in bytes when allocating 2D memory on a CUDA device

    {
    if (cudaMallocPitch((void**)&d_c, &pitch, Cols * sizeof(float), Rows) != cudaSuccess) {
        std::cerr << "Error: Failed to allocate device memory for d_c." << std::endl;
        return 1;
    }

    if (cudaMallocPitch((void**)&d_temp, &pitch, Cols * sizeof(float), Rows) != cudaSuccess) {
        std::cerr << "Error: Failed to allocate device memory for d_temp." << std::endl;
        cudaFree(d_c);
        return 1;
    }

    if (cudaMemcpyToSymbol(d_dt, &dt, sizeof(float)) != cudaSuccess) {
        std::cerr << "Error: Failed to copy dt to device constant memory." << std::endl;
        cudaFree(d_c);
        cudaFree(d_temp);
        return 1;
    }

    if (cudaMemcpyToSymbol(d_timestep, &timestep, sizeof(float)) != cudaSuccess) {
        std::cerr << "Error: Failed to copy timestep to device constant memory." << std::endl;
        cudaFree(d_c);
        cudaFree(d_temp);
        return 1;
    }

    if (cudaMemcpy2D(d_c, pitch, c, Cols * sizeof(float), Cols * sizeof(float), Rows, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error: Failed to copy c to device memory." << std::endl;
        cudaFree(d_c);
        cudaFree(d_temp);
        return 1;
    }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        
        perform_diffusion<<<gridSize, blockSize>>>(d_c, d_temp, pitch);
        // if (cudaDeviceSynchronize() != cudaSuccess) {
        //     std::cerr << "Error: Failed to synchronize after perform_diffusion kernel." << std::endl;
        //     cudaFree(d_c);
        //     cudaFree(d_temp);
        //     return 1;
        // }

        std::swap(d_c, d_temp);

        apply_boundary_conditions<<<(std::max(Rows, Cols) + 255) / 256, 256>>>(d_c, pitch);
        // if (cudaDeviceSynchronize() != cudaSuccess) {
        //     std::cerr << "Error: Failed to synchronize after apply_boundary_conditions kernel." << std::endl;
        //     cudaFree(d_c);
        //     cudaFree(d_temp);
        //     return 1;
        // }

        timestep += dt;
        // if (cudaMemcpyToSymbol(d_timestep, &timestep, sizeof(float)) != cudaSuccess) {
        //     std::cerr << "Error: Failed to update timestep in device constant memory." << std::endl;
        //     cudaFree(d_c);
        //     cudaFree(d_temp);
        //     return 1;
        // }

        if ((iter + 1) % update_every == 0) {
            if (cudaMemcpy2D(c, Cols * sizeof(float), d_c, pitch, Cols * sizeof(float), Rows, cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::cerr << "Error: Failed to copy c from device to host." << std::endl;
                cudaFree(d_c);
                cudaFree(d_temp);
                return 1;
            }
        }
    }

    if (cudaMemcpy2D(c, Cols * sizeof(float), d_c, pitch, Cols * sizeof(float), Rows, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Error: Failed to copy c from device to host." << std::endl;
        cudaFree(d_c);
        cudaFree(d_temp);
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    std::cout << "Done, " << iterations << " iterations in "
              << elapsed_seconds.count() << " seconds." << std::endl;

    cudaFree(d_c);
    cudaFree(d_temp);

    return 0;
}
