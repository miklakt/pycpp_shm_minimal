#include <iostream>
#include <cstdint>

#include "shared_memory_access.hpp"

// Include the header or code from above

int main()
{
    try {
        // A) Map a scalar int32 at offset 0 (Dims... empty => T*)
        //std::int32_t* scalar_ptr = mapSharedMemoryVariadic<std::int32_t, 0>();

        auto scalar_ptr = getSharedMemoryFieldPtr<SharedMemoryLayout::myint_tag>();

        std::cout << "Scalar initially: " << *scalar_ptr << "\n";
        *scalar_ptr += 10;
        std::cout << "Scalar after modification: " << *scalar_ptr << "\n";



        // map array of floats
        auto arr2d = getSharedMemoryFieldPtr<SharedMemoryLayout::myarr_tag>();

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                std::cout << i << "\t" << j << "\n";
                (*arr2d)[i][j] = i+j;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}
