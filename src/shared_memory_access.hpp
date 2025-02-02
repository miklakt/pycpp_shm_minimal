#pragma once

#include <stdexcept>
#include <sys/mman.h>
#include <fcntl.h>
#include <cstring>
#include <memory>
#include "shared_memory_layout.hxx"



namespace SharedMemoryAccess {

    // Pointer to the mapped shared memory region
    static void* addr_ = nullptr;

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

    // Concept to check if a type is a valid SharedMemoryLayout::field_info specialization
    template <typename Tag>
    concept ValidTag = requires {typename SharedMemoryLayout::field_info<Tag>::type;};

    // Template to get a reference to a shared memory field based on a tag
    template <typename Tag>
    requires ValidTag<Tag>
    inline auto& get() {
        // Check if shared memory is already initialized (addr_ is not nullptr)
        if (addr_ == nullptr) {
            initialize();  // Initialize shared memory if not already initialized
        }
        using FieldType = typename SharedMemoryLayout::field_info<Tag>::type;
        constexpr std::size_t offset = SharedMemoryLayout::field_info<Tag>::offset;

        // Compute the pointer to the field based on the offset
        auto* ptr = reinterpret_cast<FieldType*>(static_cast<char* >(addr_) + offset);
        
        return *ptr;
    }

    template<typename FieldType>
    constexpr std::size_t get_size(){
        return sizeof(FieldType) / sizeof(typename std::remove_all_extents<FieldType>::type);
    }

    template<typename FieldType>
    constexpr std::size_t get_size(FieldType &arr){
        return sizeof(FieldType) / sizeof(typename std::remove_all_extents<FieldType>::type);
    }

    template <typename Tag>
    requires ValidTag<Tag>
    inline auto& get_flat() {
        // Check if shared memory is already initialized (addr_ is not nullptr)
        if (addr_ == nullptr) {
            initialize();  // Initialize shared memory if not already initialized
        }

        using FieldType = typename SharedMemoryLayout::field_info<Tag>::type;
        constexpr std::size_t offset = SharedMemoryLayout::field_info<Tag>::offset;
        constexpr std::size_t size = get_size<FieldType>();
        
        // Compute the pointer to the field based on the offset
        auto* ptr = reinterpret_cast<FieldType*>(static_cast<char*>(addr_) + offset);

        // Return a flattened array pointer
        return *reinterpret_cast<typename std::remove_all_extents<FieldType>::type(*)[size]>(ptr);
    }

    template<typename T>
    constexpr auto& flatten(T& array) {
        static_assert(std::is_array_v<T>, "Input must be a fixed-length multidimensional array.");
        constexpr std::size_t size = get_size<T>();
        
        // Reinterpret the reference to a flattened array
        return *reinterpret_cast<typename std::remove_all_extents<T>::type(*)[size]>(&array);
    }

    template <typename T, std::size_t... Dims> struct __multi_array_type;
    template <typename T> struct __multi_array_type<T> {using type = std::remove_reference_t<T>;};
    template <typename T, std::size_t FirstDim, std::size_t... Rest>
    struct __multi_array_type<T, FirstDim, Rest...> {
        using base_type = typename __multi_array_type<std::remove_reference_t<T>, Rest...>::type;
        using type = base_type[FirstDim];
    };

    template <std::size_t... Dims, typename ArrayType>
    inline auto& reshape(ArrayType& array)
    {
        constexpr std::size_t new_size = (Dims * ...);
        constexpr std::size_t original_size = get_size(array);

        static_assert(new_size==original_size,"Mismatch between input array size and the provided shape.");

        auto& flatten_array = flatten(array);
        
        using ElementType = typename std::remove_all_extents<ArrayType>::type;
        using NewArrayType = typename __multi_array_type<ElementType, Dims...>::type;
        return *reinterpret_cast<NewArrayType* >(&flatten_array);

    }
    namespace Fields{
        MAP_ALL_SHARED_MEMORY_FIELDS

    }
} // namespace SharedMemoryAccess