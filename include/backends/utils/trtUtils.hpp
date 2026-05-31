#pragma once

#include <vector>
#include <filesystem>
#include <NvInfer.h>
#include <array>

#include "core/tensor.hpp"
#include "logging/BackendLoggers/TrtLoggerAdaptor.hpp"

namespace fs = std::filesystem;
std::vector<char> readEngineFileToArray(const fs::path& fileName);

std::vector<std::string> getTensorNames(
    const std::unique_ptr<nvinfer1::ICudaEngine> engine,
    nvinfer1::TensorIOMode mode
);


void logFullModelInfo(
    TrtLoggerAdaptor& logger,
    const std::unique_ptr<nvinfer1::ICudaEngine> engine
);


inline IOMode fromTrtIOMode(nvinfer1::TensorIOMode trtmode) {

    switch (trtmode) {

        case nvinfer1::TensorIOMode::kINPUT:
            return IOMode::Input;
        
        case nvinfer1::TensorIOMode::kOUTPUT:
            return IOMode::Output;
        
        case nvinfer1::TensorIOMode::kNONE:
            return IOMode::None;

        default:
            throw std::runtime_error("Unsupported type for tensor io mode.");
    }
}

inline DType getTensorTypefromTrtDType(nvinfer1::DataType trtType) {
    
    switch (trtType) {
        
        case nvinfer1::DataType::kFLOAT:
            return DType::Float32;

        case nvinfer1::DataType::kHALF:
            return DType::Float16;

        case nvinfer1::DataType::kBF16:
            return DType::BFloat16;

        case nvinfer1::DataType::kINT8:
            return DType::Int8;

        case nvinfer1::DataType::kINT32:
            return DType::Int32;

        case nvinfer1::DataType::kUINT8:
            return DType::UInt8;

        case nvinfer1::DataType::kBOOL:
            return DType::Bool;
        
        case nvinfer1::DataType::kFP8:
        case nvinfer1::DataType::kFP4:
        case nvinfer1::DataType::kINT4:
            throw std::runtime_error("Unsupported type for tensor data type.");
    }

}

inline nvinfer1::DataType getTrtDTypeFromTensorType(DType dtype) {

    switch (dtype) {

        case DType::Float32:
            return nvinfer1::DataType::kFLOAT;
        
        case DType::Float16:
            return nvinfer1::DataType::kHALF;

        case DType::Int8:
            return nvinfer1::DataType::kINT8;

        case DType::Int32:
            return nvinfer1::DataType::kINT32;

        case DType::Bool:
            return nvinfer1::DataType::kBOOL;

        case DType::UInt8:
            return nvinfer1::DataType::kUINT8;

        case DType::BFloat16:
            return nvinfer1::DataType::kBF16;

        case DType::Int64:
            return nvinfer1::DataType::kINT64;

        default:
            throw std::runtime_error("Unsupported type for tensor data type.");
    }
}

Shape fromTrtDims(const nvinfer1::Dims& dims);
nvinfer1::Dims toTrtDims(const Shape& shape);
