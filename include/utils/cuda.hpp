#pragma once

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <nvtx3/nvToolsExt.h>


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
struct CudaDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            CUDA_WARN(cudaFree(ptr));
        }
    }
};

