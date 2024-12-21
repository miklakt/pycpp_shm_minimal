#pragma once

#include <cstddef>    // size_t
#include <string>
#include <stdexcept>
#include <sys/mman.h> // mmap, munmap
#include <fcntl.h>    // shm_open, O_* constants
#include <unistd.h>   // close, lseek
#include <cstring>    // strerror

//
// Meta-function that builds T[D1][D2]...[Dn] out of T and variadic Dims.
//
// Example: NDArray<float, 2,3>::type is float[2][3].
//          NDArray<int, 4,5,6>::type is int[4][5][6].
//
template <typename T, std::size_t... Dims>
struct NDArray {
    using type = T;  // no dimensions => scalar T
};

template <typename T, std::size_t First, std::size_t... Rest>
struct NDArray<T, First, Rest...> {
    // Recursively build sub-array
    using sub = typename NDArray<T, Rest...>::type;
    // Prepend dimension [First]
    using type = sub[First];
};

//
// 2. Helper alias: NDPtr<T, D1, D2, ...> => "pointer to T[D1][D2]..."
//
template <typename T, std::size_t... Dims>
using NDPtr = typename NDArray<T, Dims...>::type*;

//
// 3. The variadic template function that:
//    - Opens the named shared memory
//    - mmaps the entire segment
//    - Returns NDPtr<T, Dims...> (pointer to T[D1][D2]...[Dn])
//
template <typename T, std::size_t... Dims>
NDPtr<T, Dims...> mapSharedMemoryVariadic(const std::string& shmName, std::size_t offset = 0)
{
    // 1) Open existing shared memory (created by Python or another process).
    int fd = shm_open(shmName.c_str(), O_RDWR, 0666);
    if (fd < 0) {
        throw std::runtime_error("Failed to open shared memory '" + shmName + "': " + std::strerror(errno));
    }

    // 2) Determine the total size so we can map the entire region.
    //    In a real app, you might know the size from JSON or pass it in.
    off_t size = lseek(fd, 0, SEEK_END);
    if (size < 0) {
        close(fd);
        throw std::runtime_error("lseek() failed: " + std::string(std::strerror(errno)));
    }

    // 3) Map into our address space.
    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) {
        throw std::runtime_error("mmap() failed: " + std::string(std::strerror(errno)));
    }

    // 4) Return a pointer to the typed region at 'offset'.
    //    The type is T[D1][D2]...[Dn] if we have dims, or just T if none.
    return reinterpret_cast<NDPtr<T, Dims...>>(
        static_cast<char*>(addr) + offset
    );
}
