#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono>  // For benchmarking
#include "../src/shared_memory_access.hpp"
#include "../src/eigen_map.hpp"

// Apply boundary conditions
template <typename ArrayType>
void apply_boundary_conditions(ArrayType& c, float source_value = 1.0f, float sink_value = 0.0f) {
    // Left boundary (source)
    using T = typename std::remove_all_extents<ArrayType>::type; // Deduce scalar type (e.g., float)
    constexpr std::size_t Rows = std::extent<ArrayType, 0>::value; // Deduce rows
    constexpr std::size_t Cols = std::extent<ArrayType, 1>::value; // Deduce columns
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

// Drift-diffusion step template function
template <typename ArrayType, typename TimeStep_>
void drift_diffusion_step(
    const ArrayType& c,
    const ArrayType& D_x, const ArrayType& D_y,
    const ArrayType& dU_x, const ArrayType& dU_y,
    const ArrayType& alpha_x, const ArrayType& alpha_y,
    const ArrayType& lambda_n, const ArrayType& lambda_s,
    ArrayType& c_next, 
    //MapType& grad_c_e, MapType& grad_c_n,
    //MapType& J_dif_e, MapType& J_dif_n,
    //MapType& J_adv_e, MapType& J_adv_n,
    //MapType& J_E, MapType& J_N,
    //MapType& div_J,
    TimeStep_ dt) {

    using T = typename std::remove_all_extents<ArrayType>::type; // Deduce scalar type (e.g., float)
    constexpr std::size_t Rows = std::extent<ArrayType, 0>::value; // Deduce rows
    constexpr std::size_t Cols = std::extent<ArrayType, 1>::value; // Deduce columns

    #pragma omp parallel for
    for (int i = 1; i < Rows - 1; ++i) {
        for (int j = 1; j < Cols - 1; ++j) {
            // Extract neighboring concentrations
            T c_P = c[i][j];     // Current cell
            T c_E = c[i+1][j]; // East neighbor
            T c_W = c[i-1][j]; // West neighbor
            T c_N = c[i][j+1]; // North neighbor
            T c_S = c[i][j-1]; // South neighbor

            // Concentration gradients
            T grad_c_e = c_E - c_P;
            T grad_c_w = c_P - c_W;
            T grad_c_n = c_N - c_P;
            T grad_c_s = c_P - c_S;

            // Diffusion fluxes due to potential gradient
            T J_dif_e = -D_x[i][j] * grad_c_e;
            T J_dif_w = -D_x[i-1][j] * grad_c_w;
            T J_dif_n = -D_y[i][j] * grad_c_n;
            T J_dif_s = -D_y[i][j-1] * grad_c_s;

            // Alpha coefficients for upwind scheme
            T alpha_e = alpha_x[i][j];
            T alpha_w = 1.0f - alpha_x[i-1][j];
            T alpha_n = alpha_y[i][j];
            T alpha_s = 1.0f - alpha_y[i][j-1];

            // Concentrations at faces with upwind correction
            T c_e = c_E * alpha_e + c_P * (1.0f - alpha_e);
            T c_w = c_W * alpha_w + c_P * (1.0f - alpha_w);
            T c_n = c_N * alpha_n + c_P * (1.0f - alpha_n);
            T c_s = c_S * alpha_s + c_P * (1.0f - alpha_s);

            // Advection fluxes due to potential gradient
            T J_adv_e = -D_x[i][j] * dU_x[i][j] * c_e;
            T J_adv_w = -D_x[i-1][j] * dU_x[i-1][j] * c_w;
            T J_adv_n = -D_y[i][j] * dU_y[i][j] * c_n;
            T J_adv_s = -D_y[i][j-1] * dU_y[i][j-1] * c_s;

            // Total fluxes at cell faces
            T J_E = J_dif_e + J_adv_e;
            T J_W = J_dif_w + J_adv_w;
            T J_N = J_dif_n + J_adv_n;
            T J_S = J_dif_s + J_adv_s;
            // T J_E = J_dif_e;
            // T J_W = J_dif_w;
            // T J_N = J_dif_n;
            // T J_S = J_dif_s;

            // Total divergence of flux (Adams-Bashforth method)
            //T J_tot_current = -J_E + J_W - lambda_n[i][j] * J_N + lambda_s[i][j] * J_S;
            //T J_tot_prev = div_J[i][j]; // Previous value of divergence
            //T J_tot = (3.0f / 2.0f) * J_tot_current - (1.0f / 2.0f) * J_tot_prev;

            T J_tot = -J_E + J_W - lambda_n[i][j] * J_N + lambda_s[i][j] * J_S;

            //div_J[i][j] = -J_tot;       // Update divergence of flux
            c_next[i][j] = c_P + J_tot * dt; // Update concentration
        }
    }
    //c = c_next;
    //for (int i = 0; i < rows; ++i) {c_next(i,0) = 1.0f; c_next(i, cols-1) = 0.0f;};
    //apply_boundary_conditions(c_next);
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
        MAP_SHM(c, c);
        MAP_SHM(c_next, c_next);
        MAP_SHM(D_x, D_x);
        MAP_SHM(D_y, D_y);
        MAP_SHM(dU_x, dU_x);
        MAP_SHM(dU_y, dU_y);
        MAP_SHM(alpha_x, alpha_x);
        MAP_SHM(alpha_y, alpha_y);
        MAP_SHM(lambda_n, lambda_n);
        MAP_SHM(lambda_s, lambda_s);
        // MAP_SHM(grad_c_e, grad_c_e);
        // MAP_SHM(grad_c_n, grad_c_n);
        // MAP_SHM(J_dif_e, J_dif_e);
        // MAP_SHM(J_dif_n, J_dif_n);
        // MAP_SHM(J_adv_e, J_adv_e);
        // MAP_SHM(J_adv_n, J_adv_n);
        // MAP_SHM(J_E, J_E);
        // MAP_SHM(J_N, J_N);
        // MAP_SHM(div_J, div_J);
        MAP_SHM(dt, dt);
        MAP_SHM(timestep, timestep);

        // Benchmark the diffusion process
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            apply_boundary_conditions(c);
            drift_diffusion_step(
                c, 
                D_x, D_y, 
                dU_x, dU_y, 
                alpha_x, alpha_y, 
                lambda_n, lambda_s,
                c_next, 
                //grad_c_e, grad_c_n, J_dif_e, J_dif_n, J_adv_e, J_adv_n, J_E, J_N, div_J, 
                dt);
            std::swap(c, c_next);
            timestep = timestep+dt;
        }
        apply_boundary_conditions(c);

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
