//g++ -std=c++20 -O3 src/access_by_name_example.cpp -o bin/assess_by_name_example
#include <iostream>
#include <algorithm> //for std::transform
#include <iterator> // for std::begin, std::end

#include "shared_memory_access.hpp"

int main() {
    try {
        using SharedMemoryAccess::Fields::myint;
        using SharedMemoryAccess::Fields::myarr;
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
    }
}



