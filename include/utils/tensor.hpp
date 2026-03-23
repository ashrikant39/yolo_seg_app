#pragma once

#include <unordered_map>
#include <NvInfer.h>
#include <string>
#include <memory>
#include <numeric>
#include <type_traits>
#include "utils/cuda.hpp"
#include <opencv2/core.hpp>


template <typename T>
using UniquePtrToArray = std::unique_ptr<T[]>;

template <typename T>
using CudaUniquePtrToArray = std::unique_ptr<T[], CudaDeleter<T>>;


template <typename T, template <typename> class PtrType>
/**
 * @brief Allocate contiguous tensor storage using either host or CUDA managed memory.
 *
 * For `CudaUniquePtrToArray`, this allocates via `cudaMallocManaged`.
 * For host pointers, this allocates via `std::make_unique<T[]>`.
 */
PtrType<T[]> makeUniquePtr(size_t numElements){

    if constexpr (std::is_same_v<PtrType<T[]>, CudaUniquePtrToArray<T>>) {
        T* ptr = nullptr;
        CUDA_THROW(cudaMallocManaged(&ptr, sizeof(T) * numElements));
        return PtrType<T[]>(ptr);
    }

    else {
        return std::make_unique<T[]>(numElements);
    }
}


template <typename T, template <typename> class PtrType>
class Tensor{

    public:
        Tensor() = delete;

        /**
         * @brief Construct a tensor wrapper around contiguous storage.
         * @param trtDtype TensorRT dtype metadata.
         * @param dims TensorRT dimensions.
         * @param mode TensorRT IO mode (input/output).
         */
        Tensor(nvinfer1::DataType trtDtype, const nvinfer1::Dims& dims, nvinfer1::TensorIOMode mode):
            m_trtDtype(trtDtype),
            m_dims(dims),
            m_mode(mode),
            m_numElements(std::accumulate(
                m_dims.d, 
                m_dims.d + m_dims.nbDims, 
                static_cast<size_t>(1),
                std::multiplies<>()
            )),
            m_unqPtr(makeUniquePtr<T, PtrType>(m_numElements)){}

        /**
         * @brief Total flattened element count derived from `dims`.
         */
        size_t getNumElements() const {
            return m_numElements;
        }

        nvinfer1::DataType getDtype() const {
            return m_trtDtype;
        }

        nvinfer1::Dims getDims() const {
            return m_dims;
        }

        nvinfer1::TensorIOMode getIOMode() const {
            return m_mode;
        }


        /**
         * @brief Mutable raw pointer to underlying contiguous storage.
         */
        T* ptr(){
            return m_unqPtr.get();
        }

        /**
         * @brief Const raw pointer to underlying contiguous storage.
         */
        const T* ptr() const {
            return m_unqPtr.get();
        }

    private:
        nvinfer1::DataType m_trtDtype;
        size_t m_numElements;
        nvinfer1::Dims m_dims;
        nvinfer1::TensorIOMode m_mode;
        PtrType<T[]> m_unqPtr;
};


template <typename T>
using TensorMap = std::unordered_map<std::string, Tensor<T, UniquePtrToArray>>;

template <typename T>
using CudaTensorMap = std::unordered_map<std::string, Tensor<T, CudaUniquePtrToArray>>;


/** Copy half-precision buffer to float32 (CPU-side buffers). */
inline void copyDataToFloat32(const cv::float16_t* src, float* dst, size_t numElements) {
    for (size_t i = 0; i < numElements; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}


template <template <typename> class PtrType>
/**
 * @brief Convert an FP16 tensor to FP32 into a temporary output tensor.
 *
 * @note Current implementation writes to a local temporary and does not return it.
 */
void castHalfToSinglePrecision(const Tensor<cv::float16_t, PtrType>& input){

    Tensor<float, PtrType> output = {
        nvinfer1::DataType::kFLOAT,
        input.getDims(),
        input.getIOMode()
    };

    cv::float16_t *src = input.ptr();
    float *dst = output.ptr();
    const size_t totalElements = input.getNumElements();

    for(int i = 0; i < totalElements; i++){
        dst[i] = static_cast<float>(src[i]);
    }

}