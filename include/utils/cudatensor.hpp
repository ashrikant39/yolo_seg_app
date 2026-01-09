#pragma once

#include <unordered_map>
#include <NvInfer.h>
#include <string>
#include <memory>
#include <cuda_runtime_api.h>
#include <numeric>
#include <type_traits>

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


template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            CUDA_WARN(cudaFree(ptr));
        }
    }
};


template <typename T>
using UniquePtrToArray = std::unique_ptr<T[], std::default_delete<T[]>>;

template <typename T>
using CudaUniquePtrToArray = std::unique_ptr<T[], CudaDeleter<T>>;


template <typename T, template <typename> class PtrType>
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

        size_t getNumElements() const {
            return m_numElements;
        }

        nvinfer1::DataType getDtype() const {
            return m_trtDtype;
        }

        nvinfer1::Dims getDims() const {
            return m_dims;
        }

        nvinfer::TensorIOMode getIOMode() const {
            return m_mode;
        }


        T* ptr(){
            return m_unqPtr.get();
        }

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


template <template <typename> class PtrType>
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