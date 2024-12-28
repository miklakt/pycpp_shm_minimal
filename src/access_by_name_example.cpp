//g++ -std=c++20 -O3 src/access_by_name_example.cpp -o bin/assess_by_name_example
#include <iostream>
#include <cstdint>
#include <algorithm> //for std::transform
#include <iterator> // for std::begin, std::end

#include "shared_memory_access.hpp"

// Include the header or code from above

// Example usage
int main() {
    try {
        using SharedMemoryAccess::Fields::myint;
        using SharedMemoryAccess::Fields::myarr;
        auto& myarr_flat = SharedMemoryAccess::flatten(myarr);


        std::cout << "Type of myint: " << typeid(myint).name() << std::endl;
        std::cout << "Type of myarr: " << typeid(myarr).name() << std::endl;

        std::cout << "Value of myint: " << myint << std::endl;
        std::cout << "Value of myarr[0][0]: " << myarr[0][0] << std::endl;

        // Example to set a new value in shared memory
        myint = 42;
        myarr[0][0] += 3.14f;
        std::cout << "Updated myint: " << myint << std::endl;
        std::cout << "Updated myarr[0][0]: " << myarr[0][0] << std::endl;
        std::cout << "Updated myarr_flat[0]: " << myarr_flat[0] << std::endl;

        std::transform(std::begin(myarr_flat), std::end(myarr_flat), std::begin(myarr_flat), [](float x){return x +1.0f;});
        //std::transform(std::begin(myarr_flat2), std::end(myarr_flat2), std::begin(myarr_flat2), [](float x){std::cout<<x<<" "; return x;});

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}



