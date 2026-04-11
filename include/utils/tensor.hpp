#pragma once

#include <unordered_map>
#include <NvInfer.h>
#include <string>
#include <memory>
#include <numeric>
#include <type_traits>
#include "utils/cuda.hpp"
#include <opencv2/core.hpp>
#include <logger.hpp>

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


template <typename T, template <typename> class PtrType>
class Tensor{

    public:

        // Default constructor
        Tensor(){}

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
                dims.d, 
                dims.d + dims.nbDims, 
                static_cast<size_t>(1),
                std::multiplies<>()
            )) {
                m_unqPtr = makeUniquePtr<T, PtrType>(m_numElements);
            }

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
        PtrType<T> m_unqPtr;
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

inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-std::max(-50.f, std::min(50.f, x))));
}

/** Linear index for [batch][i1][i2][i3] with dims [B, D1, D2, D3]. */
inline size_t idx4(int b, int i1, int i2, int i3, int D1, int D2, int D3){
    return static_cast<size_t>(b) * static_cast<size_t>(D1 * D2 * D3) +
           static_cast<size_t>(i1) * static_cast<size_t>(D2 * D3) +
           static_cast<size_t>(i2) * static_cast<size_t>(D3) +
           static_cast<size_t>(i3);
}

/** Linear index for [batch][i1][i2] with dims [B, D1, D2]. */
inline size_t idx3(int b, int i1, int i2, int D1, int D2){
    return static_cast<size_t>(b) * static_cast<size_t>(D1 * D2) +
           static_cast<size_t>(i1) * static_cast<size_t>(D2) +
           static_cast<size_t>(i2);
}