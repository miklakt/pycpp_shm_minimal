#include <iostream>

#include "shared_memory_access.hpp"
#include "eigen_map.hpp"

int main() {
    // Access fields by tag (using example `myint`, `myarr`, etc.)
    MAP_SHM(myint, myint);
    MAP_SHM(myarr, myarr);

    // Copy the content from the shared memory array to the local variable
    CREATE_COPY(myarr, local_copy);

    float local_copy2[10][10] = {};
    COPY(myarr, local_copy2);

    // Map 2D array to Eigen::Matrix
    auto eigen_matrix = SharedMemoryAccess::WrapToEigen(myarr);
    std::cout << eigen_matrix.cols() <<"\n";
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


    // Local copy is not
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

    return 0;
}
