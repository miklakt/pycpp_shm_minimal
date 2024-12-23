#include <iostream>
#include "shared_memory_access.hpp"     // Includes getSharedMemoryFieldPtr
#include "array_algorithms.hpp"         // Includes ApplyFunctionElementwise

int main() {
    try {
        // Apply a function to increment elements of myarr and store to myarr2
        ApplyFunctionElementwise<SharedMemoryLayout::myarr_tag, SharedMemoryLayout::myarr2_tag>(
            [](const auto& x) { return  x+1.0f; }//, // Function to apply
        );
        // Apply a function to increment elements of myarr2 and store to myarr
        ApplyFunctionElementwise<SharedMemoryLayout::myarr2_tag, SharedMemoryLayout::myarr_tag>(
            [](const auto& x) { return  x+1.0f; }//, // Function to apply
        );

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}