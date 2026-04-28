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


template <typename T>
struct DevicePtrDeleter {
    void operator()(T *ptr) const noexcept {
        if (ptr) {
            CUDA_WARN(cudaFree(ptr));
        }
    }
};

template <typename T>
struct HostPtrDeleter {
    void operator()(T *ptr) const noexcept {
        if (ptr) {
            CUDA_WARN(cudaFreeHost(ptr));
        }
    }
};


class CudaStream {

    public:
        CudaStream() {
            cudaError_t err = cudaStreamCreate(&m_stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaStreamCreate failed: ") + cudaGetErrorString(err)
                );
            }
        }

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

        cudaStream_t get() const noexcept {
            return m_stream;
        }

        operator cudaStream_t() const noexcept {
            return m_stream;
        }

    private:
        cudaStream_t m_stream{nullptr};
};
