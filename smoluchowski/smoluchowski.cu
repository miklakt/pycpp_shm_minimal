#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono>  // For benchmarking
#include <cuda_runtime.h>
#include <memory>  // For std::unique_ptr
#include "../src/shared_memory_access.hpp"

// Exposing Shared Memory fields
using SharedMemoryAccess::Fields::c;
using SharedMemoryAccess::Fields::c_next;
using SharedMemoryAccess::Fields::D_x;
using SharedMemoryAccess::Fields::D_y;
using SharedMemoryAccess::Fields::dU_x;
using SharedMemoryAccess::Fields::dU_y;
using SharedMemoryAccess::Fields::alpha_x;
using SharedMemoryAccess::Fields::alpha_y;
using SharedMemoryAccess::Fields::lambda_n;
using SharedMemoryAccess::Fields::lambda_s;
using SharedMemoryAccess::Fields::div_J;
using SharedMemoryAccess::Fields::dt;
using SharedMemoryAccess::Fields::timestep;

using ArrayType = std::remove_reference_t<decltype(c)>; // Type of array, like float[800][600]
constexpr std::size_t Rows = std::extent<ArrayType, 0>::value;
constexpr std::size_t Cols = std::extent<ArrayType, 1>::value;

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

auto make_unique_ptr_1D_column_vector_cuda(){
        return std::unique_ptr<float, decltype(&cudaFree)>(
        [&]{
            float* ptr; 
            cudaMalloc((void**)&ptr, Rows * sizeof(float)); 
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

#define ACCESS_2D(ptr, i, j, pitch) ((float*)((char*)ptr + (i) * pitch))[j]
// Drift-diffusion kernel
__global__ void drift_diffusion(
    float* d_c, float* d_c_next, 
    float* d_D_x, float* d_D_y, 
    float* d_alpha_x, float* d_alpha_y, 
    float* d_dU_x, float* d_dU_y, 
    float* d_lambda_n, float* d_lambda_s, 
    float* d_div_J, 
    size_t pitch) {
        
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > 0 && i < Rows - 1 && j > 0 && j < Cols - 1) {
        // Extract neighboring concentrations
        float c_P = ACCESS_2D(d_c, i, j, pitch);
        float c_E = ACCESS_2D(d_c, i + 1, j, pitch);
        float c_W = ACCESS_2D(d_c, i - 1, j, pitch);
        float c_N = ACCESS_2D(d_c, i, j + 1, pitch);
        float c_S = ACCESS_2D(d_c, i, j - 1, pitch);

        // Concentration gradients
        float grad_c_e = c_E - c_P;
        float grad_c_w = c_P - c_W;
        float grad_c_n = c_N - c_P;
        float grad_c_s = c_P - c_S;

        // Diffusion fluxes
        float J_dif_e = -ACCESS_2D(d_D_x, i, j, pitch) * grad_c_e;
        float J_dif_w = -ACCESS_2D(d_D_x, i - 1, j, pitch) * grad_c_w;
        float J_dif_n = -ACCESS_2D(d_D_y, i, j, pitch) * grad_c_n;
        float J_dif_s = -ACCESS_2D(d_D_y, i, j - 1, pitch) * grad_c_s;

        // Alpha coefficients
        float alpha_e = ACCESS_2D(d_alpha_x, i, j, pitch);
        float alpha_w = 1.0f - ACCESS_2D(d_alpha_x, i - 1, j, pitch);
        float alpha_n = ACCESS_2D(d_alpha_y, i, j, pitch);
        float alpha_s = 1.0f - ACCESS_2D(d_alpha_y, i, j - 1, pitch);

        // Concentrations at faces
        float c_e = c_E * alpha_e + c_P * (1.0f - alpha_e);
        float c_w = c_W * alpha_w + c_P * (1.0f - alpha_w);
        float c_n = c_N * alpha_n + c_P * (1.0f - alpha_n);
        float c_s = c_S * alpha_s + c_P * (1.0f - alpha_s);

        // Advection fluxes
        float J_adv_e = -ACCESS_2D(d_D_x, i, j, pitch) * ACCESS_2D(d_dU_x, i, j, pitch) * c_e;
        float J_adv_w = -ACCESS_2D(d_D_x, i - 1, j, pitch) * ACCESS_2D(d_dU_x, i - 1, j, pitch) * c_w;
        float J_adv_n = -ACCESS_2D(d_D_y, i, j, pitch) * ACCESS_2D(d_dU_y, i, j, pitch) * c_n;
        float J_adv_s = -ACCESS_2D(d_D_y, i, j - 1, pitch) * ACCESS_2D(d_dU_y, i, j - 1, pitch) * c_s;

        // Total fluxes
        float J_E = J_dif_e + J_adv_e;
        float J_W = J_dif_w + J_adv_w;
        float J_N = J_dif_n + J_adv_n;
        float J_S = J_dif_s + J_adv_s;

        float J_tot = -J_E + J_W - d_lambda_n[j] * J_N + d_lambda_s[j] * J_S;

        // Update divergence of flux and concentration
        ACCESS_2D(d_div_J, i, j, pitch) = -J_tot;
        ACCESS_2D(d_c_next, i, j, pitch) = c_P + J_tot * d_dt;
    }
}

// Macro for allocating and copying data to device memory
// Creates pointer to an array on the device with a prefix d_
#define ALLOC2D_AND_COPY_TO_DEVICE(X)                                           \
    auto d_##X = make_unique_ptr_cuda();                         \
    cudaMemcpy2D(d_##X.get(), pitch, X, Cols * sizeof(float), Cols * sizeof(float), Rows, cudaMemcpyHostToDevice);

#define ALLOC1D_AND_COPY_TO_DEVICE(X)                                           \
    auto d_##X = make_unique_ptr_1D_column_vector_cuda();                         \
    cudaMemcpy(d_##X.get(), X, Rows * sizeof(float), cudaMemcpyHostToDevice);


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

    // Copy constants to device
    cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));
    cudaMemcpyToSymbol(d_timestep, &timestep, sizeof(float));

    // Allocate and copy to device memory
    ALLOC2D_AND_COPY_TO_DEVICE(c);
    ALLOC2D_AND_COPY_TO_DEVICE(c_next);
    ALLOC2D_AND_COPY_TO_DEVICE(D_x);
    ALLOC2D_AND_COPY_TO_DEVICE(D_y);
    ALLOC2D_AND_COPY_TO_DEVICE(dU_x);
    ALLOC2D_AND_COPY_TO_DEVICE(dU_y);
    ALLOC2D_AND_COPY_TO_DEVICE(alpha_x);
    ALLOC2D_AND_COPY_TO_DEVICE(alpha_y);
    ALLOC2D_AND_COPY_TO_DEVICE(div_J);

    ALLOC1D_AND_COPY_TO_DEVICE(lambda_n);
    ALLOC1D_AND_COPY_TO_DEVICE(lambda_s);

    // Start benchmarking
    auto start_time = std::chrono::high_resolution_clock::now();

        // Main computation loop
    for (int iter = 0; iter < iterations; ++iter) {

        drift_diffusion<<<gridSize, blockSize>>>(
                                        d_c.get(), d_c_next.get(), 
                                        d_D_x.get(), d_D_y.get(), 
                                        d_alpha_x.get(), d_alpha_y.get(), 
                                        d_dU_x.get(), d_dU_y.get(), 
                                        d_lambda_n.get(), d_lambda_s.get(), 
                                        d_div_J.get(), 
                                        pitch);
        apply_boundary_conditions<<<gridSize1D, blockSize1D>>>(d_c_next.get(), pitch);

        std::swap(d_c, d_c_next);
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
