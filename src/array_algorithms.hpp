#include <tuple>
#include <type_traits>
#include <functional>
#include <utility> // for std::forward

#include "shared_memory_layout.hxx"
#include "shared_memory_access.hpp"

//------------------------------------------------------------------------------
// Recursive helper to iterate through multi-dimensional arrays and apply a function
//------------------------------------------------------------------------------
template <typename TIn, typename TOut, typename Func, std::size_t N>
void ApplyFunctionElementwiseImpl(const TIn (&inputArray)[N], TOut (&outputArray)[N], Func&& func) {
    for (std::size_t i = 0; i < N; ++i) {
        if constexpr (std::is_array_v<TIn> && std::is_array_v<TOut>) {
            // Recursive case: apply function to sub-arrays
            ApplyFunctionElementwiseImpl(inputArray[i], outputArray[i], std::forward<Func>(func));
        } else {
            // Base case: apply function to scalar elements
            outputArray[i] = func(inputArray[i]); // Assign function result to output
        }
    }
}

//------------------------------------------------------------------------------
// ApplyFunction: Apply an element-wise function to input and output arrays given their tags
//------------------------------------------------------------------------------
template <typename InputTag, typename OutputTag, typename Func>
void ApplyFunctionElementwise(Func&& func) {
    // 1. Get the array types and pointers using the tags
    using InputArrayType = typename SharedMemoryLayout::field_info<InputTag>::type;
    using OutputArrayType = typename SharedMemoryLayout::field_info<OutputTag>::type;

    // Ensure both are array types
    static_assert(std::is_array_v<InputArrayType>, "InputTag must correspond to an array type.");
    static_assert(std::is_array_v<OutputArrayType>, "OutputTag must correspond to an array type.");

    // 2. Map the shared memory fields using the tags
    const auto inputArrayPtr = getSharedMemoryFieldPtr<InputTag>();
    auto outputArrayPtr = getSharedMemoryFieldPtr<OutputTag>();

    // 3. Apply the function to the input and output arrays
    ApplyFunctionElementwiseImpl(*inputArrayPtr, *outputArrayPtr, std::forward<Func>(func));
}

struct ApplyElementwise_{
    template<typename InputTag>
};