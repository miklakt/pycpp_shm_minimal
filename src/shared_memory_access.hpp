#pragma once

#include <cstddef>    // size_t
#include <string>
#include <stdexcept>
#include <sys/mman.h> // mmap, munmap
#include <fcntl.h>    // shm_open, O_* constants
#include <unistd.h>   // close, lseek
#include <cstring>    // strerror

#include "shared_memory_layout.hxx"

//------------------------------------------------------------------------------
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
// Helper alias: NDPtr<T, D1, D2, ...> => "pointer to T[D1][D2]..."
//
template <typename T, std::size_t... Dims>
using NDPtr = typename NDArray<T, Dims...>::type*;

//------------------------------------------------------------------------------
//
// The variadic template function that:
//    - Opens the named shared memory
//    - mmaps the entire segment
//    - Returns NDPtr<T, Dims...> (pointer to T[D1][D2]...[Dn])
//
template <typename T, std::size_t offset, std::size_t... Dims>
NDPtr<T, Dims...> mapSharedMemory()
{
    // 1) Open existing shared memory (created by Python or another process).
    int fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (fd < 0) {
        throw std::runtime_error("Failed to open shared memory '" + std::string(SHM_NAME) + "': " + std::strerror(errno));
    }

    // 2) Determine the total size so we can map the entire region.
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

//------------------------------------------------------------------------------
// The `getSharedMemoryFieldPtr` template function that uses a tag from `SharedMemoryLayout` to map shared memory.
//
// Example usage:
//    auto myfield = getSharedMemoryFieldPtr<SharedMemoryLayout::myfield_tag>();
//------------------------------------------------------------------------------
template <typename Tag>
auto getSharedMemoryFieldPtr()
{
    using FieldType = typename SharedMemoryLayout::field_info<Tag>::type;
    constexpr std::size_t offset = SharedMemoryLayout::field_info<Tag>::offset;

    // Handle array types via NDArray. Automatically deduces dimensions if present.
    return mapSharedMemory<FieldType, offset>();
}