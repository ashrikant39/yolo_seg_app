#pragma once

#include <unordered_map>
#include <NvInfer.h>
#include <string>
#include <memory>
#include <cuda_runtime_api.h>
#include "utils/cuda.hpp"
#include <numeric>

template <typename T>
using HostUniquePtr = std::unique_ptr<T, std::default_delete<T>>;

template <typename T, class UniquePtrType>
class Tensor{

    public:
        Tensor() = delete;

        Tensor(nvinfer1::DataType trtDtype, const nvinfer1::Dims& dims, nvinfer1::TensorIOMode mode):
            m_trtDtype(trtDtype),
            m_dims(dims),
            m_mode(mode){

                m_numElements = std::accumulate(
                    m_dims.d, 
                    m_dims.d + m_dims.nbDims, 
                    static_cast<size_t>(1),
                    std::multiplies<>()
                );

            }

        size_t getNumElements();
        nvinfer1::DataType getDtype();

        T* data(){
            return 
        }

    private:
        nvinfer1::DataType m_trtDtype;
        size_t m_numElements;
        nvinfer1::Dims m_dims;
        nvinfer1::TensorIOMode m_mode;
        UniquePtrType m_unqPtr;
};


template <typename T>
using TensorMap = std::unordered_map<std::string, Tensor<T, std::default_delete<T>>>;

template <typename T>
using CudaTensorMap = std::unordered_map<std::string, Tensor<T, CudaDeleter<T>>>;

