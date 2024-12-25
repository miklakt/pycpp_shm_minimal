//run shm_allocator.py first to generate shared_memory_layout.hxx and allocate shared memory
// to compile
//g++ -std=c++23 -I/usr/include/eigen3 src/eigen_map_example.cpp -o bin/eigen_map_example -O3
#include <iostream>

#include "shared_memory_access.hpp"
#include "eigen_map.hpp"


int main() {
    // Access fields by tag (using example `myint`, `myarr`, etc.)
    MAP_SHM(myint, myint);
    MAP_SHM(myarr, myarr);

    // Create local variable and copy the content from the shared memory array
    // You do not need to know the type and dimensions
    CREATE_COPY(myarr, local_copy);
    
    // Copy the content from the shared memory array to existing variable
    // It is up to you to define size and dimensions, will raise en error if wrong
    float local_copy2[10][10] = {};
    COPY(myarr, local_copy2);

    // Map 2D array to Eigen::Matrix it does not allocate new memory, but just mwp the existing one
    auto eigen_matrix = SharedMemoryAccess::WrapToEigen(myarr);
    // Copy can not be don by simple assignment operator since 'eigen_matrix' is a Map object not a PlainObject as 'matrix'
    auto eigen_matrix2 = SharedMemoryAccess::CopyEigen(eigen_matrix);
    std::cout << "Original Matrix:\n" << eigen_matrix << "\n";

    // Modify the matrix (modifies the underlying shared memory)
    eigen_matrix(0, 0) = 3.14f;
    eigen_matrix(1, 1) = 6.28f;
    std::cout << "Modified Matrix:\n" << eigen_matrix << "\n";

    // Original matrix is modified by Eigen operations
    std::cout << "Original Matrix:\n";
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << myarr[i][j] << " "; // Print each element in the row
        }
        std::cout << '\n'; // Move to the next row
    }

    // Local copy is not modified
    std::cout << "Local Copy Matrix:\n";
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << local_copy[i][j] << " "; // Print each element in the row
        }
        std::cout << '\n'; // Move to the next row
    }

    // Local copy is not
    std::cout << "Local Copy 2 Matrix:\n";
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            std::cout << local_copy2[i][j] << " "; // Print each element in the row
        }
        std::cout << '\n'; // Move to the next row
    }


    // Eigen_matrix copy
    std::cout << "Copied Matrix:\n" << eigen_matrix2 << "\n";


    return 0;
}
