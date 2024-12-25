DOES NOT WORK YET

#include <iostream>
#include <cstdlib> // For std::atoi
#include <chrono>  // For benchmarking
#include "../src/shared_memory_access.hpp"
#include "../src/eigen_map.hpp"

// Apply boundary conditions
template <typename MapType>
void apply_boundary_conditions(MapType& c, float source_value =1.0f, float sink_value =0.0f) {
    int rows = c.rows();
    int cols = c.cols();

    c.eval();
    // Top and bottom boundaries (mirror condition)
    c.row(0) = c.row(1);
    c.row(rows - 1) = c.row(rows - 2);

    // Left boundary (source)
    c.col(0).setConstant(source_value);

    // Right boundary (sink)
    c.col(cols - 1).setConstant(sink_value);
    c.eval();
}

// Apply boundary conditions
// template <typename ArrayType>
// void apply_boundary_conditions(ArrayType& c, float source_value = 1.0f, float sink_value = 0.0f) {
//     // Left boundary (source)
//     using T = typename std::remove_all_extents<ArrayType>::type; // Deduce scalar type (e.g., float)
//     constexpr std::size_t Rows = std::extent<ArrayType, 0>::value; // Deduce rows
//     constexpr std::size_t Cols = std::extent<ArrayType, 1>::value; // Deduce columns
//     for (size_t j = 0; j < Cols; ++j) {
//         c[0][j] = static_cast<T>(source_value);
//         c[Rows-1][j] = static_cast<T>(sink_value);
//     }

//     // Top boundary (mirror condition)
//     for (size_t i = 0; i < Rows; ++i) {
//         c[i][0] = c[i][1];
//         c[i][Cols-1] = c[i][Cols-2];
//     }
// }

// // Drift-diffusion step template function
// template <typename MapType>
// void drift_diffusion_step(
//     const MapType& c,
//     const MapType& D_x, const MapType& D_y,
//     const MapType& dU_x, const MapType& dU_y,
//     const MapType& alpha_x, const MapType& alpha_y,
//     const MapType& lambda_n, const MapType& lambda_s,
//     MapType& c_next, 
//     //MapType& grad_c_e, MapType& grad_c_n,
//     //MapType& J_dif_e, MapType& J_dif_n,
//     //MapType& J_adv_e, MapType& J_adv_n,
//     //MapType& J_E, MapType& J_N,
//     //MapType& div_J,
//     float dt) {

//     const int rows = c.rows();
//     const int cols = c.cols();
//     for (int i = 1; i < rows - 1; ++i) {
//         for (int j = 1; j < cols - 1; ++j) {
//             // Extract neighboring concentrations
//             float c_P = c(i, j);     // Current cell
//             float c_E = c(i + 1, j); // East neighbor
//             float c_W = c(i - 1, j); // West neighbor
//             float c_N = c(i, j + 1); // North neighbor
//             float c_S = c(i, j - 1); // South neighbor

//             // Concentration gradients
//             float grad_c_e = c_E - c_P;
//             float grad_c_w = c_P - c_W;
//             float grad_c_n = c_N - c_P;
//             float grad_c_s = c_P - c_S;

//             // Diffusion fluxes due to potential gradient
//             float J_dif_e = -D_x(i, j) * grad_c_e;
//             float J_dif_w = -D_x(i - 1, j) * grad_c_w;
//             float J_dif_n = -D_y(i, j) * grad_c_n;
//             float J_dif_s = -D_y(i, j - 1) * grad_c_s;

//             // Alpha coefficients for upwind scheme
//             float alpha_e = alpha_x(i, j);
//             float alpha_w = 1.0f - alpha_x(i - 1, j);
//             float alpha_n = alpha_y(i, j);
//             float alpha_s = 1.0f - alpha_y(i, j - 1);

//             // Concentrations at faces with upwind correction
//             float c_e = c_E * alpha_e + c_P * (1.0f - alpha_e);
//             float c_w = c_W * alpha_w + c_P * (1.0f - alpha_w);
//             float c_n = c_N * alpha_n + c_P * (1.0f - alpha_n);
//             float c_s = c_S * alpha_s + c_P * (1.0f - alpha_s);

//             // Advection fluxes due to potential gradient
//             float J_adv_e = -D_x(i, j) * dU_x(i, j) * c_e;
//             float J_adv_w = -D_x(i - 1, j) * dU_x(i - 1, j) * c_w;
//             float J_adv_n = -D_y(i, j) * dU_y(i, j) * c_n;
//             float J_adv_s = -D_y(i, j - 1) * dU_y(i, j - 1) * c_s;

//             // Total fluxes at cell faces
//             // float J_E = J_dif_e + J_adv_e;
//             // float J_W = J_dif_w + J_adv_w;
//             // float J_N = J_dif_n + J_adv_n;
//             // float J_S = J_dif_s + J_adv_s;
//             float J_E = J_dif_e;
//             float J_W = J_dif_w;
//             float J_N = J_dif_n;
//             float J_S = J_dif_s;

//             // Total divergence of flux (Adams-Bashforth method)
//             //float J_tot_current = -J_E + J_W - lambda_n(i, j) * J_N + lambda_s(i, j) * J_S;
//             //float J_tot_prev = div_J(i, j); // Previous value of divergence
//             //float J_tot = (3.0f / 2.0f) * J_tot_current - (1.0f / 2.0f) * J_tot_prev;

//             float J_tot = -J_E + J_W - lambda_n(i, j) * J_N + lambda_s(i, j) * J_S;

//             //div_J(i, j) = -J_tot;       // Update divergence of flux
//             c_next(i, j) = c_P + J_tot * dt; // Update concentration
//             c_next.eval();
//         }
//     }
    
//     //c = c_next;
//     //for (int i = 0; i < rows; ++i) {c_next(i,0) = 1.0f; c_next(i, cols-1) = 0.0f;};
//     //apply_boundary_conditions(c_next);
// }

template <typename MapType>
void drift_diffusion_step(
    const MapType& c,
    const MapType& D_x, const MapType& D_y,
    const MapType& dU_x, const MapType& dU_y,
    const MapType& alpha_x, const MapType& alpha_y,
    const MapType& lambda_n, const MapType& lambda_s,
    MapType& c_next,
    float dt) {

    const int rows = c.rows();
    const int cols = c.cols();

    // Initialize face concentrations and fluxes
    Eigen::MatrixXf c_e = c.block(1, 0, rows - 1, cols);
    Eigen::MatrixXf c_w = c.block(0, 0, rows - 1, cols);
    Eigen::MatrixXf c_n = c.block(0, 1, rows, cols - 1);
    Eigen::MatrixXf c_s = c.block(0, 0, rows, cols - 1);

    // Include boundary contributions
    c_e.row(rows - 1) = c.row(rows - 1);  // East boundary
    c_w.row(0) = c.row(0);                // West boundary
    c_n.col(cols - 1) = c.col(cols - 1);  // North boundary
    c_s.col(0) = c.col(0);                // South boundary

    // Gradients
    Eigen::MatrixXf grad_c_e = c_e - c.block(0, 0, rows - 1, cols);
    Eigen::MatrixXf grad_c_w = c.block(0, 0, rows - 1, cols) - c_w;
    Eigen::MatrixXf grad_c_n = c_n - c.block(0, 0, rows, cols - 1);
    Eigen::MatrixXf grad_c_s = c.block(0, 0, rows, cols - 1) - c_s;

    // Diffusion fluxes
    Eigen::MatrixXf J_dif_e = -D_x.block(0, 0, rows - 1, cols).array() * grad_c_e.array();
    Eigen::MatrixXf J_dif_w = -D_x.block(0, 0, rows - 1, cols).array() * grad_c_w.array();
    Eigen::MatrixXf J_dif_n = -D_y.block(0, 0, rows, cols - 1).array() * grad_c_n.array();
    Eigen::MatrixXf J_dif_s = -D_y.block(0, 0, rows, cols - 1).array() * grad_c_s.array();

    // Alpha coefficients for upwind scheme
    Eigen::MatrixXf alpha_e = alpha_x.block(0, 0, rows - 1, cols);
    Eigen::MatrixXf alpha_w = 1.0f - alpha_x.block(0, 0, rows - 1, cols);
    Eigen::MatrixXf alpha_n = alpha_y.block(0, 0, rows, cols - 1);
    Eigen::MatrixXf alpha_s = 1.0f - alpha_y.block(0, 0, rows, cols - 1);

    // Upwind face concentrations
    c_e = c_e.array() * alpha_e.array() + c.block(0, 0, rows - 1, cols).array() * (1.0f - alpha_e.array());
    c_w = c_w.array() * alpha_w.array() + c.block(0, 0, rows - 1, cols).array() * (1.0f - alpha_w.array());
    c_n = c_n.array() * alpha_n.array() + c.block(0, 0, rows, cols - 1).array() * (1.0f - alpha_n.array());
    c_s = c_s.array() * alpha_s.array() + c.block(0, 0, rows, cols - 1).array() * (1.0f - alpha_s.array());

    // Advection fluxes
    Eigen::MatrixXf J_adv_e = -D_x.block(0, 0, rows - 1, cols).array() * dU_x.block(0, 0, rows - 1, cols).array() * c_e.array();
    Eigen::MatrixXf J_adv_w = -D_x.block(0, 0, rows - 1, cols).array() * dU_x.block(0, 0, rows - 1, cols).array() * c_w.array();
    Eigen::MatrixXf J_adv_n = -D_y.block(0, 0, rows, cols - 1).array() * dU_y.block(0, 0, rows, cols - 1).array() * c_n.array();
    Eigen::MatrixXf J_adv_s = -D_y.block(0, 0, rows, cols - 1).array() * dU_y.block(0, 0, rows, cols - 1).array() * c_s.array();

    // Total fluxes
    Eigen::MatrixXf J_E = J_dif_e + J_adv_e;
    Eigen::MatrixXf J_W = J_dif_w + J_adv_w;
    Eigen::MatrixXf J_N = J_dif_n + J_adv_n;
    Eigen::MatrixXf J_S = J_dif_s + J_adv_s;

    // Divergence of fluxes
    Eigen::MatrixXf J_tot = -J_E + J_W - lambda_n.array() * J_N.array() + lambda_s.array() * J_S.array();

    // Update concentrations
    c_next.block(1, 1, rows - 2, cols - 2) =
        c.block(1, 1, rows - 2, cols - 2).array() + dt * J_tot.array();

    apply_boundary_conditions(c_next);
    c_next.eval();
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
        MAP_SHM(c_next, c_next_raw);
        MAP_SHM(D_x, D_x_raw);
        MAP_SHM(D_y, D_y_raw);
        MAP_SHM(dU_x, dU_x_raw);
        MAP_SHM(dU_y, dU_y_raw);
        MAP_SHM(alpha_x, alpha_x_raw);
        MAP_SHM(alpha_y, alpha_y_raw);
        MAP_SHM(lambda_n, lambda_n_raw);
        MAP_SHM(lambda_s, lambda_s_raw);
        // MAP_SHM(grad_c_e, grad_c_e_raw);
        // MAP_SHM(grad_c_n, grad_c_n_raw);
        // MAP_SHM(J_dif_e, J_dif_e_raw);
        // MAP_SHM(J_dif_n, J_dif_n_raw);
        // MAP_SHM(J_adv_e, J_adv_e_raw);
        // MAP_SHM(J_adv_n, J_adv_n_raw);
        // MAP_SHM(J_E, J_E_raw);
        // MAP_SHM(J_N, J_N_raw);
        // MAP_SHM(div_J, div_J_raw);
        MAP_SHM(dt, dt);
        MAP_SHM(timestep, timestep);

        // Wrap shared memory arrays to Eigen matrices
        auto c = SharedMemoryAccess::WrapToEigen(c_raw);
        auto c_next = SharedMemoryAccess::WrapToEigen(c_next_raw);
        auto D_x = SharedMemoryAccess::WrapToEigen(D_x_raw);
        auto D_y = SharedMemoryAccess::WrapToEigen(D_y_raw);
        auto dU_x = SharedMemoryAccess::WrapToEigen(dU_x_raw);
        auto dU_y = SharedMemoryAccess::WrapToEigen(dU_y_raw);
        auto alpha_x = SharedMemoryAccess::WrapToEigen(alpha_x_raw);
        auto alpha_y = SharedMemoryAccess::WrapToEigen(alpha_y_raw);
        auto lambda_n = SharedMemoryAccess::WrapToEigen(lambda_n_raw);
        auto lambda_s = SharedMemoryAccess::WrapToEigen(lambda_s_raw);
        // auto grad_c_e = SharedMemoryAccess::WrapToEigen(grad_c_e_raw);
        // auto grad_c_n = SharedMemoryAccess::WrapToEigen(grad_c_n_raw);
        // auto J_dif_e = SharedMemoryAccess::WrapToEigen(J_dif_e_raw);
        // auto J_dif_n = SharedMemoryAccess::WrapToEigen(J_dif_n_raw);
        // auto J_adv_e = SharedMemoryAccess::WrapToEigen(J_adv_e_raw);
        // auto J_adv_n = SharedMemoryAccess::WrapToEigen(J_adv_n_raw);
        // auto J_E = SharedMemoryAccess::WrapToEigen(J_E_raw);
        // auto J_N = SharedMemoryAccess::WrapToEigen(J_N_raw);
        // auto div_J = SharedMemoryAccess::WrapToEigen(div_J_raw);

        // Benchmark the diffusion process
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < iterations; ++iter) {
            //apply_boundary_conditions(c);
            drift_diffusion_step(
                c, 
                D_x, D_y, 
                dU_x, dU_y, 
                alpha_x, alpha_y, 
                lambda_n, lambda_s,
                c_next, 
                //grad_c_e, grad_c_n, J_dif_e, J_dif_n, J_adv_e, J_adv_n, J_E, J_N, div_J, 
                dt);
            c = c_next;
            timestep = timestep+dt;
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
