#pragma once

#include <memory>
#include "core/cuda.hpp"


enum class MemoryKind {
    UNSET,
    PageableHost,
    PinnedHost,
    CudaMem,
    Unified
};

// template <typename T>
// using UniquePtrToArray = std::unique_ptr<T[]>;

// template <typename T>
// using UniquePtrToDeviceArray = std::unique_ptr<T[], DevicePtrDeleter<T>>;

// template <typename T>
// using UniquePtrToPinnedArray = std::unique_ptr<T[], PinnedPtrDeleter<T>>;

// template <typename T>
// using UniquePtrToManagedArray = std::unique_ptr<T[], ManagedPtrDeleter<T>>;


// template <template <typename> class PtrType>
// struct PtrFactory;

// template <>
// struct PtrFactory<UniquePtrToArray> {
//     template <typename T>
//     static UniquePtrToArray<T> make(size_t numElements) {
//         return std::make_unique<T[]>(numElements);
//     }
// };

// template <>
// struct PtrFactory<UniquePtrToDeviceArray> {
//     template <typename T>
//     static UniquePtrToDeviceArray<T> make(size_t numElements) {
//         void* ptr = nullptr;
//         CUDA_THROW(cudaMalloc(&ptr, sizeof(T) * numElements));
//         return UniquePtrToDeviceArray<T>(static_cast<T*>(ptr));
//     }
// };


// template <>
// struct PtrFactory<UniquePtrToPinnedArray> {
//     template <typename T>
//     static UniquePtrToPinnedArray<T> make(size_t numElements) {
//         void* ptr = nullptr;
//         CUDA_THROW(cudaMallocHost(&ptr, sizeof(T) * numElements));
//         return UniquePtrToPinnedArray<T>(static_cast<T*>(ptr));
//     }
// };


// template <>
// struct PtrFactory<UniquePtrToManagedArray> {
//     template <typename T>
//     static UniquePtrToManagedArray<T> make(size_t numElements) {
//         void* ptr = nullptr;
//         CUDA_THROW(cudaMallocManaged(&ptr, sizeof(T) * numElements));
//         return UniquePtrToManagedArray<T>(static_cast<T*>(ptr));
//     }
// };


// template <typename T, template <typename> class PtrType>
// PtrType<T> makeUniquePtr(size_t numElements) {
//     return PtrFactory<PtrType>::template make<T>(numElements);
// }

struct MallocHostPolicy {
    static constexpr DeviceType deviceType = DeviceType::Host;
    static constexpr MemoryKind memoryKind = MemoryKind::MallocHost;

    struct Deleter {
        void operator()(std::byte* p) const noexcept {
            delete[] p;
        }
    };

    using BytePointer = std::unique_ptr<std::byte, Deleter>;

    static BytePointer allocateBytes(size_t numBytes) {
        return BytePointer(new std::byte[numBytes]);
    }
};


struct PinnedHostPolicy {
    static constexpr DeviceType deviceType = DeviceType::Host;
    static constexpr MemoryKind memoryKind = MemoryKind::PinnedHost;

    struct Deleter {
        void operator()(std::byte* p) const noexcept {
            if (p) {
                cudaFreeHost(p);
            }
        }
    };

    using BytePointer = std::unique_ptr<std::byte, Deleter>;

    static BytePointer allocateBytes(size_t numBytes) {
        void* raw = nullptr;

        const cudaError_t err = cudaMallocHost(&raw, numBytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        return BytePointer(static_cast<std::byte*>(raw));
    }
};


struct CudaDevicePolicy {
    static constexpr DeviceType deviceType = DeviceType::Cuda;
    static constexpr MemoryKind memoryKind = MemoryKind::CudaDevice;

    struct Deleter {
        void operator()(std::byte* p) const noexcept {
            if (p) {
                cudaFree(p);
            }
        }
    };

    using BytePointer = std::unique_ptr<std::byte, Deleter>;

    static BytePointer allocateBytes(size_t numBytes) {
        void* raw = nullptr;

        const cudaError_t err = cudaMalloc(&raw, numBytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        return BytePointer(static_cast<std::byte*>(raw));
    }
};


struct UnifiedMemoryPolicy {
    static constexpr DeviceType deviceType = DeviceType::Cuda;
    static constexpr MemoryKind memoryKind = MemoryKind::Unified;

    struct Deleter {
        void operator()(std::byte* p) const noexcept {
            if (p) {
                cudaFree(p);
            }
        }
    };

    using BytePointer = std::unique_ptr<std::byte, Deleter>;

    static BytePointer allocateBytes(size_t numBytes) {
        void* raw = nullptr;

        const cudaError_t err = cudaMallocManaged(&raw, numBytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        return BytePointer(static_cast<std::byte*>(raw));
    }
};