#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono>  // For benchmarking
#include "../src/shared_memory_access.hpp"

//Exposing Shared Memory fields
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

using ArrayType = std::remove_reference_t<decltype(c)>; //type of array, like float[800][600]
constexpr std::size_t Rows = std::extent<ArrayType, 0>::value;
constexpr std::size_t Cols = std::extent<ArrayType, 1>::value;


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

void drift_diffusion() {
    //using T = typename std::remove_all_extents<ArrayType>::type; // Deduce scalar type (e.g., float);
    #pragma omp parallel for num_threads(4)
    for (int i = 1; i < Rows - 1; ++i) {
        for (int j = 1; j < Cols - 1; ++j) {
            // Extract neighboring concentrations
            auto c_P = c[i][j];     // Current cell
            auto c_E = c[i+1][j]; // East neighbor
            auto c_W = c[i-1][j]; // West neighbor
            auto c_N = c[i][j+1]; // North neighbor
            auto c_S = c[i][j-1]; // South neighbor

            // Concentration gradients
            auto grad_c_e = c_E - c_P;
            auto grad_c_w = c_P - c_W;
            auto grad_c_n = c_N - c_P;
            auto grad_c_s = c_P - c_S;

            // Diffusion fluxes due to potential gradient
            auto J_dif_e = -D_x[i][j] * grad_c_e;
            auto J_dif_w = -D_x[i-1][j] * grad_c_w;
            auto J_dif_n = -D_y[i][j] * grad_c_n;
            auto J_dif_s = -D_y[i][j-1] * grad_c_s;

            // Alpha coefficients for upwind scheme
            auto alpha_e = alpha_x[i][j];
            auto alpha_w = 1.0f - alpha_x[i-1][j];
            auto alpha_n = alpha_y[i][j];
            auto alpha_s = 1.0f - alpha_y[i][j-1];

            // Concentrations at faces with upwind correction
            auto c_e = c_E * alpha_e + c_P * (1.0f - alpha_e);
            auto c_w = c_W * alpha_w + c_P * (1.0f - alpha_w);
            auto c_n = c_N * alpha_n + c_P * (1.0f - alpha_n);
            auto c_s = c_S * alpha_s + c_P * (1.0f - alpha_s);

            // Advection fluxes due to potential gradient
            auto J_adv_e = -D_x[i][j] * dU_x[i][j] * c_e;
            auto J_adv_w = -D_x[i-1][j] * dU_x[i-1][j] * c_w;
            auto J_adv_n = -D_y[i][j] * dU_y[i][j] * c_n;
            auto J_adv_s = -D_y[i][j-1] * dU_y[i][j-1] * c_s;

            // Total fluxes at cell faces
            auto J_E = J_dif_e + J_adv_e;
            auto J_W = J_dif_w + J_adv_w;
            auto J_N = J_dif_n + J_adv_n;
            auto J_S = J_dif_s + J_adv_s;

            auto J_tot = -J_E + J_W - lambda_n[i][j] * J_N + lambda_s[i][j] * J_S;

            div_J[i][j] = -J_tot;       // Update divergence of flux
            c_next[i][j] = c_P + J_tot * dt; // Update concentration
        }
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
        // Benchmark the diffusion process
        auto start_time = std::chrono::high_resolution_clock::now();

         for (int iter = 0; iter < iterations; ++iter){
            drift_diffusion();
            apply_boundary_conditions();
            std::swap(c, c_next);
            timestep = timestep+dt*iterations;
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
