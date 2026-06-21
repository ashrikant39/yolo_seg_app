#pragma once

#include <vector>
#include <filesystem>
#include <NvInfer.h>
#include <array>
#include <fstream>

#include "core/tensor.hpp"
#include "logging/BackendLoggers/TrtLoggerAdaptor.hpp"

namespace fs = std::filesystem;
std::vector<char> readEngineFileToArray(const fs::path& fileName);

std::vector<std::string> getTensorNames(
    const std::unique_ptr<nvinfer1::ICudaEngine>& engine,
    nvinfer1::TensorIOMode mode
);


void logFullModelInfo(
    TrtLoggerAdaptor& logger,
    const std::unique_ptr<nvinfer1::ICudaEngine>& engine
);


inline IOMode TrtIOMode2IOMode(nvinfer1::TensorIOMode trtmode) {

    switch (trtmode) {

        case nvinfer1::TensorIOMode::kINPUT:
            return IOMode::Input;

        case nvinfer1::TensorIOMode::kOUTPUT:
            return IOMode::Output;

        case nvinfer1::TensorIOMode::kNONE:
            return IOMode::None;

        default:
            throw std::runtime_error("Unsupported type for nvinfer tensor io mode.");
    }
}


inline nvinfer1::TensorIOMode IOMode2TrtIOMode(IOMode iomode) {

    switch (iomode) {

        case IOMode::Input:
            return nvinfer1::TensorIOMode::kINPUT;

        case IOMode::Output:
            return nvinfer1::TensorIOMode::kOUTPUT;

        case IOMode::None:
            return nvinfer1::TensorIOMode::kNONE;

        default:
            throw std::runtime_error("Unsupported type for tensor io mode.");
    }
}

inline DataType TrtType2DataType(nvinfer1::DataType trtType) {

    switch (trtType) {

        case nvinfer1::DataType::kFLOAT:
            return DataType::Float32;

        case nvinfer1::DataType::kHALF:
            return DataType::Float16;

        case nvinfer1::DataType::kBF16:
            return DataType::BFloat16;

        case nvinfer1::DataType::kINT8:
            return DataType::Int8;

        case nvinfer1::DataType::kINT32:
            return DataType::Int32;

        case nvinfer1::DataType::kUINT8:
            return DataType::UInt8;

        case nvinfer1::DataType::kBOOL:
            return DataType::Bool;

        case nvinfer1::DataType::kFP8:
        case nvinfer1::DataType::kFP4:
        case nvinfer1::DataType::kINT4:
            throw std::runtime_error("Unsupported type for tensor data type.");
    }

    throw std::runtime_error("Unsupported type for tensor data type.");

}

inline nvinfer1::DataType DataType2TrtType(DataType dtype) {

    switch (dtype) {

        case DataType::Float32:
            return nvinfer1::DataType::kFLOAT;

        case DataType::Float16:
            return nvinfer1::DataType::kHALF;

        case DataType::Int8:
            return nvinfer1::DataType::kINT8;

        case DataType::Int32:
            return nvinfer1::DataType::kINT32;

        case DataType::Bool:
            return nvinfer1::DataType::kBOOL;

        case DataType::UInt8:
            return nvinfer1::DataType::kUINT8;

        case DataType::BFloat16:
            return nvinfer1::DataType::kBF16;

        case DataType::Int64:
            return nvinfer1::DataType::kINT64;

        default:
            throw std::runtime_error("Unsupported type for tensor data type.");
    }
}

Shape TrtDims2Shape(const nvinfer1::Dims& dims);
nvinfer1::Dims ShapetoTrtDims(const Shape& shape);
