#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include "shared_memory_layout.hxx"



namespace SharedMemoryAccess {

    // Pointer to the mapped shared memory region
    static void* addr_ = nullptr;
    // Total size of the shared memory
    static std::size_t total_size_ = 0;

    // Function to initialize the shared memory mapping
    inline void initialize() {
        // 1) Open the existing shared memory segment using constexpr SHM_NAME
        int fd = shm_open(SHM_NAME, O_RDWR, 0666);
        if (fd < 0) {
            throw std::runtime_error("Failed to open shared memory '" + std::string(SHM_NAME) + "': " + std::strerror(errno));
        }

        // 2) Map the shared memory into the process's address space
        addr_ = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (addr_ == MAP_FAILED) {
            throw std::runtime_error("mmap() failed: " + std::string(std::strerror(errno)));
        }
    }

    // Template to get a reference to a shared memory field based on a tag
    template <typename Tag>
    inline auto& get() {
        // Check if shared memory is already initialized (addr_ is not nullptr)
        if (addr_ == nullptr) {
            initialize();  // Initialize shared memory if not already initialized
        }
        using FieldType = typename SharedMemoryLayout::field_info<Tag>::type;
        constexpr std::size_t offset = SharedMemoryLayout::field_info<Tag>::offset;

        // Compute the pointer to the field based on the offset
        auto* ptr = reinterpret_cast<FieldType*>(static_cast<char*>(addr_) + offset);
        
        return *reinterpret_cast<FieldType* const>(ptr);

    }

} // namespace SharedMemoryAccess

/**
 * @brief Maps a shared memory field to a reference variable.
 * 
 * This macro retrieves a reference to a field in shared memory, identified by the
 * specified tag, and assigns it to the provided destination variable.
 * The field in shared memory is accessed via the `SharedMemoryAccess::get` function,
 * and the macro automatically appends `_tag` to the tag name.
 * 
 * The destination variable (`dest`) must be a reference type, and it will be assigned
 * a reference to the shared memory field. This macro simplifies the process of accessing
 * shared memory fields without needing to manually type out the tag suffix.
 * 
 * @param source The tag that identifies the shared memory field in `SharedMemoryLayout`.
 * @param dest The destination reference variable to which the shared memory field is assigned.
 * 
 * Example usage:
 * @code
 * MAP_SHM_TO(myint, myint);  // Maps the shared memory field myint to the variable myint.
 * MAP_SHM_TO(myarr, myarr);  // Maps the shared memory array myarr to the variable myarr.
 * @endcode
 */
#define MAP_SHM(source, dest) \
    auto& dest = SharedMemoryAccess::get<SharedMemoryLayout::source##_tag>()

/**
 * @brief Macro to copy the content of a shared memory array to a local variable.
 *
 * This macro deduces the type of the shared memory array, creates a local variable
 * of the same type and size, and copies the content from the shared memory array
 * to the local variable using std::memcpy.
 *
 * @param src The source array (shared memory array).
 * @param dest The destination variable name.
 */
#define CREATE_COPY(src, dest)                                  \
    using ArrType_##dest = std::remove_reference<decltype(src)>::type; \
    ArrType_##dest dest = {};                                  \
    std::memcpy(dest, src, sizeof(src))

#define COPY(src, dest)                                                     \
    static_assert(std::is_same_v<std::remove_reference_t<decltype(src)>,    \
                                 std::remove_reference_t<decltype(dest)>>,  \
                  "COPY error: src and dest must have the same type!");     \
    std::memcpy(dest, src, sizeof(src))
