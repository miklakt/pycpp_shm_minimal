#include <iostream>
#include "shared_memory_access.hpp"     // Includes getSharedMemoryFieldPtr
#include "array_algorithms.hpp"         // Includes ApplyFunctionElementwise

int main() {
    try {
        // Apply a function to increment elements within specific boundaries
        ApplyFunctionElementwise<SharedMemoryLayout::myarr_tag>(
            [](auto& x) { x += 1.0f; }//, // Function to apply
            //1, 3, 2, 5                 // Boundaries: [1, 3) for dim 0, [2, 5) for dim 1
        );

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}