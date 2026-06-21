#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

#include "core/cuda.hpp"
#include "core/enums.hpp"

struct MallocHostPolicy {
    static constexpr DeviceType deviceType = DeviceType::CPU;
    static constexpr MemoryType memoryType = MemoryType::PageableHost;

    struct Deleter {
        void operator()(std::byte* ptr) const noexcept {
            delete[] ptr;
        }
    };

    using BytePointer = std::unique_ptr<std::byte, Deleter>;

    static BytePointer allocateBytes(size_t numBytes) {
        return BytePointer(new std::byte[numBytes]);
    }
};

struct PinnedHostPolicy {
    static constexpr DeviceType deviceType = DeviceType::CPU;
    static constexpr MemoryType memoryType = MemoryType::PinnedHost;

    struct Deleter {
        void operator()(std::byte* ptr) const noexcept {
            if (ptr) {
                cudaFreeHost(ptr);
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
    static constexpr DeviceType deviceType = DeviceType::CUDA;
    static constexpr MemoryType memoryType = MemoryType::CudaMem;

    struct Deleter {
        void operator()(std::byte* ptr) const noexcept {
            if (ptr) {
                cudaFree(ptr);
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
    static constexpr DeviceType deviceType = DeviceType::CUDA;
    static constexpr MemoryType memoryType = MemoryType::Unified;

    struct Deleter {
        void operator()(std::byte* ptr) const noexcept {
            if (ptr) {
                cudaFree(ptr);
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
