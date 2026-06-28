#pragma once

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr,                                                   \
                    "[CUDA ERROR] %s:%d — %s\n",                              \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)


#define CUDA_WARN(call)                                                       \
do {                                                                      \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr,                                                   \
                "[CUDA WARNING] %s:%d — %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(err));             \
    }                                                                     \
} while (0)


#define CUDA_THROW(call)                                                      \
do {                                                                      \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        throw std::runtime_error(                                         \
            std::string("[CUDA EXCEPTION] ") + cudaGetErrorString(err));  \
    }                                                                     \
} while (0)

#ifndef NDEBUG
    #define NVTX_RANGE(name) do { nvtxRangePushA(name); } while (0)
    #define NVTX_POP()      do { nvtxRangePop(); } while (0)
#else
    #define NVTX_RANGE(name) do { } while(0)
    #define NVTX_POP()      do { } while(0)
#endif

/**
 * @brief Return the active CUDA device id.
 * @throws std::runtime_error on CUDA failure.
 */
inline int currentCudaDevice() {
    int device = 0;
    CUDA_THROW(cudaGetDevice(&device));
    return device;
}

/**
 * @brief unique_ptr deleter for cudaMalloc memory.
 */
template <typename T>
struct DevicePtrDeleter {
    void operator()(T *ptr) const noexcept {
        if (ptr) {
            CUDA_WARN(cudaFree(ptr));
        }
    }
};

/**
 * @brief unique_ptr deleter for cudaMallocHost memory.
 */
template <typename T>
struct PinnedPtrDeleter {
    void operator()(T *ptr) const noexcept {
        if (ptr) {
            CUDA_WARN(cudaFreeHost(ptr));
        }
    }
};

/**
 * @brief unique_ptr deleter for cudaMallocManaged memory.
 */
template <typename T>
struct ManagedPtrDeleter {
    void operator()(T *ptr) const noexcept {
        if (ptr) {
            CUDA_WARN(cudaFree(ptr));
        }
    }
};

/**
 * @brief RAII wrapper around cudaStream_t.
 *
 * Default construction leaves the stream as nullptr, which is suitable for CPU
 * paths. createStream() lazily creates a CUDA stream for GPU paths.
 */
class CudaStream {

    public:
        CudaStream() = default;

        /**
         * @brief Create and return a valid CUDA stream wrapper.
         */
        static CudaStream create() {
            CudaStream stream;
            stream.createStream();
            return stream;
        }

        /**
         * @brief Lazily create the CUDA stream if not already valid.
         */
        void createStream() {
            if (m_stream) {
                return;
            }

            cudaError_t err = cudaStreamCreate(&m_stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaStreamCreate failed: ") + cudaGetErrorString(err)
                );
            }
        }

        /**
         * @brief Destroy the owned stream if one exists.
         */
        ~CudaStream() {
            if (m_stream) {
                cudaStreamDestroy(m_stream);
            }
        }

        CudaStream(const CudaStream&) = delete;
        CudaStream& operator=(const CudaStream&) = delete;

        CudaStream(CudaStream&& other) noexcept : m_stream(other.m_stream) {
            other.m_stream = nullptr;
        }

        CudaStream& operator=(CudaStream&& other) noexcept {
            if (this != &other) {
                if (m_stream) {
                    cudaStreamDestroy(m_stream);
                }
                m_stream = other.m_stream;
                other.m_stream = nullptr;
            }
            return *this;
        }

        /**
         * @brief Return the wrapped cudaStream_t, possibly nullptr.
         */
        cudaStream_t get() const noexcept {
            return m_stream;
        }

        /**
         * @brief True when a CUDA stream has been created.
         */
        bool valid() const noexcept {
            return m_stream != nullptr;
        }

        /**
         * @brief Implicit conversion to cudaStream_t for CUDA APIs.
         */
        operator cudaStream_t() const noexcept {
            return m_stream;
        }

    private:
        cudaStream_t m_stream{nullptr};
};
