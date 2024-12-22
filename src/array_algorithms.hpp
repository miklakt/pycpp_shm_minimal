#include <iostream>
#include <cmath>   // For boundary conditions
#include <tuple>
#include <type_traits>

// Recursive dimension extractor
template <typename T, typename = void>
struct ExtractDimensions;

template <typename T>
struct ExtractDimensions<T, std::enable_if_t<std::is_array_v<T>>> {
    static constexpr std::size_t size = std::extent_v<T>;
    using Subarray = std::remove_extent_t<T>;
    using Next = ExtractDimensions<Subarray>;

    static constexpr auto as_tuple() {
        if constexpr (std::is_array_v<Subarray>) {
            return std::tuple_cat(std::make_tuple(size), Next::as_tuple());
        } else {
            return std::make_tuple(size);
        }
    }
};

// Helper to get dimensions as a tuple
template <typename T>
constexpr auto GetDimensions() {
    return ExtractDimensions<T>::as_tuple();
}

// Recursive traversal function to apply diffusion
template <typename ArrayType, typename CoordType, typename Func>
void traverse_and_apply(ArrayType& array, CoordType& indices, Func&& func, std::size_t dim_index = 0) {
    constexpr auto dims = GetDimensions<ArrayType>();
    if constexpr (dim_index == std::tuple_size_v<decltype(dims)>) {
        // Base case: all dimensions specified, apply function
        func(indices);
    } else {
        // Recursive case: iterate over current dimension
        for (std::size_t i = 0; i < std::get<dim_index>(dims); ++i) {
            indices[dim_index] = i;
            traverse_and_apply(array, indices, func, dim_index + 1);
        }
    }
}