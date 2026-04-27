#include <memory>
#include <cuda_fp16.h>
#include <NvInfer.h>
#include <cuda_bf16.h>
#include <utils/cuda.hpp>

template <typename T>
using UniquePtrToArray = std::unique_ptr<T[]>;

template <typename T>
using CudaUniquePtrToArray = std::unique_ptr<T[], CudaDeleter<T>>;

template <template <typename> class PtrType>
struct PtrFactory;

template <>
struct PtrFactory<UniquePtrToArray> {
    template <typename T>
    static UniquePtrToArray<T> make(std::size_t numElements) {
        return std::make_unique<T[]>(numElements);
    }
};

template <>
struct PtrFactory<CudaUniquePtrToArray> {
    template <typename T>
    static CudaUniquePtrToArray<T> make(std::size_t numElements) {
        void* ptr = nullptr;
        CUDA_THROW(cudaMallocManaged(&ptr, sizeof(T) * numElements));
        return CudaUniquePtrToArray<T>(static_cast<T*>(ptr));
    }
};

template <typename T, template <typename> class PtrType>
PtrType<T> makeUniquePtr(std::size_t numElements) {
    return PtrFactory<PtrType>::template make<T>(numElements);
}

/** Copy half-precision buffer to float32 (CPU-side buffers). */
inline void castHalfToFloat(float* dst, const __half* src, size_t numElements) {
    NVTX_RANGE("CASTING_DATA_TO_FLOAT");
    for (size_t i = 0; i < numElements; ++i) {
        dst[i] = __half2float(src[i]);
    }
    NVTX_POP();
}


inline size_t getElementSize(nvinfer1::DataType dtype){

    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: 
            return sizeof(float);
        case nvinfer1::DataType::kHALF: 
            return sizeof(__half);
        case nvinfer1::DataType::kBF16: 
            return sizeof(__nv_bfloat16);
        case nvinfer1::DataType::kINT8: 
            return sizeof(int8_t);
        case nvinfer1::DataType::kINT32: 
            return sizeof(int32_t);
        case nvinfer1::DataType::kBOOL:
            return 1;
        default: throw std::runtime_error("Unknown data type");
    }
}
