#pragma once

#define EIGEN_STACK_ALLOCATION_LIMIT 10000000 // Increase to 10 MB
#include <Eigen/Dense>
#include <type_traits>
#include <cstddef>

#include "shared_memory_access.hpp"

namespace SharedMemoryAccess {
// Generalized wrapper function for 1D and 2D arrays
template <typename ArrayType>
auto WrapToEigen(ArrayType& arr) {
    using T = typename std::remove_all_extents<ArrayType>::type; // Deduce scalar type (e.g., float)

    constexpr std::size_t Dimensions = std::rank<ArrayType>::value; // Number of dimensions
    static_assert(std::is_arithmetic<T>::value, "Eigen requires arithmetic types (e.g., float, double, int)");

    if constexpr (Dimensions == 1) {
        // Handle 1D arrays (map to Eigen::Vector)
        constexpr std::size_t Size = std::extent<ArrayType, 0>::value; // Deduce size
        return Eigen::Map<Eigen::Matrix<T, Size, 1>>(&arr[0]); // Map as column vector
    } else if constexpr (Dimensions == 2) {
        // Handle 2D arrays (map to Eigen::Matrix)
        constexpr std::size_t Rows = std::extent<ArrayType, 0>::value; // Deduce rows
        constexpr std::size_t Cols = std::extent<ArrayType, 1>::value; // Deduce columns
        return Eigen::Map<Eigen::Matrix<T, Rows, Cols>>(&arr[0][0]); // Map as matrix
    } else {
        throw std::runtime_error("Unexpected array dimensions. Only 1D or 2D arrays are supported.");
        return 1;
    }
}
} // namespace SharedMemoryAccess