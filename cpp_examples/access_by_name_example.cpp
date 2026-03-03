// First create shared memory in Python (keep that script running):
//   python3 cpp_examples/create_shared_memory.py
// Then compile.
// From project root:
// g++ -std=c++20 -O3 -DSHM_DISABLE_FIELD_ALIASES -DSHM_LAYOUT_HEADER=\"../cpp_examples/shared_memory_layout.hxx\" cpp_examples/access_by_name_example.cpp -o cpp_examples/bin/access_by_name_example
// From cpp_examples/:
// g++ -std=c++20 -O3 -DSHM_DISABLE_FIELD_ALIASES -DSHM_LAYOUT_HEADER=\"../cpp_examples/shared_memory_layout.hxx\" access_by_name_example.cpp -o bin/access_by_name_example
#include <iostream>
#include <algorithm>
#include <iterator> 
#include <string>

#include "../src/shared_memory_access.hpp"

int main() {
    try {
        auto& myint = SharedMemoryAccess::get<SharedMemoryLayout::myint_tag>();
        auto& myarr = SharedMemoryAccess::get<SharedMemoryLayout::myarr_tag>();
        auto& myarr_flat = SharedMemoryAccess::flatten(myarr);


        std::cout << "Type of myint: " << typeid(myint).name() << std::endl;
        std::cout << "Type of myarr: " << typeid(myarr).name() << std::endl;
        std::cout << "Size of myarr: " << SharedMemoryAccess::get_size(myarr) << std::endl;

        std::cout << "Value of myint: " << myint << std::endl;
        std::cout << "Value of myarr[0][0]: " << myarr[0][0] << std::endl;

        // Example to set a new value in shared memory
        myint = 42;
        myarr[0][0] += 3.14f;
        std::cout << "Updated myint: " << myint << std::endl;
        std::cout << "Updated myarr[0][0]: " << myarr[0][0] << std::endl;
        std::cout << "Updated myarr_flat[0]: " << myarr_flat[0] << std::endl;

        // Apply some elementwise function to flatten memory view
        std::transform(
            std::begin(myarr_flat), std::end(myarr_flat), std::begin(myarr_flat), 
            [](float x){static int i = 0; i++; return i;}
            );

        // A reshaped view to an array, pointing to the same chunk of memory
        auto& myarr_reshaped = SharedMemoryAccess::reshape<25,2,2>(myarr);
        myarr_reshaped[1][1][1] = -1;
        std::cout << "Updated myarr_reshaped[1][1][1]: " << myarr_reshaped[1][1][1] << std::endl;

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
