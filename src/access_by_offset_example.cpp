#include <iostream>
#include <cstdint>

#include "shared_memory_access.hpp"

// Include the header or code from above

int main()
{
    try {
        // A) Map a scalar int32 at offset 0 (Dims... empty => T*)
        std::int32_t* scalar_ptr = mapSharedMemory<std::int32_t, 0>();
        std::cout << "Scalar initially: " << *scalar_ptr << "\n";
        *scalar_ptr += 10;
        std::cout << "Scalar after modification: " << *scalar_ptr << "\n";

        const uint8_t nx=8, ny=12, offset=8;


        // map array of floats
        float (*arr2d)[nx][ny] = mapSharedMemory<float, offset, nx, ny>();

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                std::cout << i << "\t" << j << "\n";
                (*arr2d)[i][j] = i+j;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}
