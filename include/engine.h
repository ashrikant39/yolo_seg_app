#pragma once

#include <NvInfer.h>
#include "logger.h"
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>
#include <filesystem>
#include <memory>
#include <eigenOps.h>

namespace fs = std::filesystem; 

// Map types
using HalfPtrMap = std::unordered_map<std::string, cv::float16_t*>;
using FloatPtrMap = std::unordered_map<std::string, float*>;

// Size Info

struct TensorSizeInfo {
    size_t totalSize;
    size_t totalBytes;
};

//  Wrapper class for running the inference
//  Must be kept alive throughout the program
//  Allocates CPU and GPU memory for inference.

class InferenceEngine
{
    public:

        InferenceEngine(const fs::path& engineFilePath, Logger& useLogger, bool logModelInformation); 

        ~InferenceEngine();

        bool allocateIOMemory();
        bool runAsynchronousInference(const std::vector<cv::float16_t>& preprocessedImageBuffer);
        std::vector<std::string> getTensorNames(nvinfer1::TensorIOMode mode);
        std::vector<nvinfer1::Dims> getTensorShapes(nvinfer1::TensorIOMode mode);
        std::vector<nvinfer1::DataType> getTensorDataTypes(nvinfer1::TensorIOMode mode);
        TensorSizeInfo getTensorTotalElemsAndBytes(const char* tensorName);
        TensorViewSharedPtr<float, 4> getOutputTensorViewFromMemory4D(const std::string& outputName);
        TensorViewSharedPtrMap<float, 4> getOutputTensorViewMap();
        
        void postProcessOutputs(const fs::path& saveDir, const std::vector<fs::path>& imagePaths);
        nvinfer1::Dims getInputDims();
        void logModelInfo();


    private:

        // Inference Runtime : Deserialize engine file
        // Engine : Representation of the model
        // Engine has the lifecycle of an entire inference run.
        // Execution Context: contains all of the state associated with a particular invocation

        Logger& m_logger; 
        std::unique_ptr<nvinfer1::IRuntime> m_runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
        HalfPtrMap m_DevicePtrMap, m_outputPtrMap;
        FloatPtrMap m_postProcessPtrMap;
        nvinfer1::DataType m_computeTypeInference;
};


// Load serialized engine file to an std::vector
std::vector<char> readEngineFileToArray(const fs::path& fileName);
size_t getElementSize(nvinfer1::DataType dtype);

// Copy float16 or bfloat16 data to float32
// originalDataPtr should contain the tensor to be copied
// copiedDataPtr should be an initialized pointer to a memory with totalElements
void copyDataToFloat32(
    cv::float16_t* sourceDataPtr, 
    float* destDataPtr, 
    size_t totalElements, 
    Logger& logger);
