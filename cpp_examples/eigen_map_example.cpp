// First create shared memory in Python (keep that script running):
//   python3 cpp_examples/create_shared_memory.py
// Then compile.
// From project root:
// g++ -std=c++23 -O3 -I/usr/include/eigen3 -DSHM_DISABLE_FIELD_ALIASES -DSHM_LAYOUT_HEADER=\"../cpp_examples/shared_memory_layout.hxx\" cpp_examples/eigen_map_example.cpp -o cpp_examples/bin/eigen_map_example
// From cpp_examples/:
// g++ -std=c++23 -O3 -I/usr/include/eigen3 -DSHM_DISABLE_FIELD_ALIASES -DSHM_LAYOUT_HEADER=\"../cpp_examples/shared_memory_layout.hxx\" eigen_map_example.cpp -o bin/eigen_map_example
#include <iostream>
#include <cstring>
#include <type_traits>
#include <string>

#include "../src/shared_memory_access.hpp"
#include "eigen_map.hpp"


int main() {
    try {
        // Access fields by tag (using example `myint`, `myarr`, etc.)
        auto& myint = SharedMemoryAccess::get<SharedMemoryLayout::myint_tag>();
        auto& myarr = SharedMemoryAccess::get<SharedMemoryLayout::myarr_tag>();

        // Create local variable and copy the content from the shared memory array
        // You do not need to know the type and dimensions
        using ArrayType = std::remove_reference_t<decltype(myarr)>;
        ArrayType local_copy{};
        std::memcpy(local_copy, myarr, sizeof(local_copy));
        
        // Copy the content from the shared memory array to existing variable
        // It is up to you to define size and dimensions, will raise en error if wrong
        float local_copy2[10][10] = {};
        static_assert(sizeof(local_copy2) == sizeof(myarr), "Mismatched array size for copy.");
        std::memcpy(local_copy2, myarr, sizeof(local_copy2));

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
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        const std::string message = e.what();
        if (message.find("Failed to open shared memory") != std::string::npos) {
            std::cerr << "Shared memory not found. Create it first in Python:\n"
                      << "  python3 cpp_examples/create_shared_memory.py\n"
                      << "Keep that Python process running while you execute this binary.\n"
                      << "Or from cpp_examples/: \n"
                      << "  python3 create_shared_memory.py\n";
        }
        return 1;
    }
}
