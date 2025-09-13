#pragma once

#include <unordered_map>
#include <NvInfer.h>
#include <string>

// // Tensor types

template <typename T>
struct Tensor{
    nvinfer1::DataType trtDtype{nvinfer1::DataType::kFLOAT};
    size_t numElements{};
    nvinfer1::Dims dims{};
    nvinfer1::TensorIOMode mode{nvinfer1::TensorIOMode::kNONE};
    T* ptr{};
};


template <typename T>
using TensorMap = std::unordered_map<std::string, Tensor<T>>;
