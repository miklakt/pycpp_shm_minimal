#include <tuple>
#include <type_traits>
#include <cstddef>
#include <iostream>
#include <utility> // for std::forward

#include "shared_memory_layout.hxx"
#include "shared_memory_access.hpp"

//------------------------------------------------------------------------------
// Recursive helper to iterate through multi-dimensional arrays and apply a function
//------------------------------------------------------------------------------
template <typename T, typename Func, std::size_t N>
void ApplyFunctionElementwiseImpl(T (&array)[N], Func&& func) {
    for (std::size_t i = 0; i < N; ++i) {
        if constexpr (std::is_array_v<T>) {
            // Recursive case: apply function to sub-array
            ApplyFunctionElementwiseImpl(array[i], std::forward<Func>(func));
        } else {
            // Base case: apply function to scalar element
            func(array[i]);
        }
    }
}

//------------------------------------------------------------------------------
// ApplyFunction: Apply an element-wise function to an array given its tag
//------------------------------------------------------------------------------
template <typename Tag, typename Func>
void ApplyFunctionElementwise(Func&& func) {
    // 1. Get the array type and pointer using the tag
    using ArrayType = typename SharedMemoryLayout::field_info<Tag>::type;

    // Ensure it is an array type
    static_assert(std::is_array_v<ArrayType>, "Tag must correspond to an array type.");

    // 2. Map the shared memory field using the tag
    auto arrayPtr = getSharedMemoryFieldPtr<Tag>();

    // 3. Apply the function to the array
    ApplyFunctionElementwiseImpl(*arrayPtr, std::forward<Func>(func));
}


