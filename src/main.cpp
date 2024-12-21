#include <iostream>
#include <cstdint>

#include "shared_memory_access.hpp"

// Include the header or code from above

int main()
{
    try {
        // A) Map a scalar int32 at offset 0 (Dims... empty => T*)
        std::int32_t* scalar_ptr = mapSharedMemoryVariadic<std::int32_t>("my_shm_name", 0);
        std::cout << "Scalar initially: " << *scalar_ptr << "\n";
        *scalar_ptr += 10;
        std::cout << "Scalar after modification: " << *scalar_ptr << "\n";

        const uint8_t nx=100,ny=200,offset=8;
        // nx=100;
        // ny=200;
        // offset = 8;

        // B) Map float[2][3] at offset 4
        float (*arr2d)[nx][ny] = mapSharedMemoryVariadic<float, nx, ny>("my_shm_name", offset);
        // arr2d is a pointer to float[2][3]. Access with (*arr2d)[row][col].

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
