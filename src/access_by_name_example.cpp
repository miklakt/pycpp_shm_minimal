#include <iostream>
#include <cstdint>

#include "shared_memory_access.hpp"

// Include the header or code from above

// Example usage
int main() {
    try {
        // Access fields by tag (using example `myint`, `myarr`, etc.)
        MAP_SHM(myint, myint);
        MAP_SHM(myarr, myarr);

        std::cout << "Type of myint: " << typeid(myint).name() << std::endl;
        std::cout << "Type of myarr: " << typeid(myarr).name() << std::endl;

        std::cout << "Value of myint: " << myint << std::endl;
        std::cout << "Value of myarr[0][0]: " << myarr[0][0] << std::endl;

        // Example to set a new value in shared memory
        myint = 42;
        myarr[0][0] += 3.14f;
        std::cout << "Updated myint: " << myint << std::endl;
        std::cout << "Updated myarr[0][0]: " << myarr[0][0] << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}



