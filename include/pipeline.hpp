#pragma once

#include <NvInfer.h>
#include <vector>
#include <opencv2/core.hpp>
#include <filesystem>
#include "utils/cudatensor.hpp"
#include "logger.hpp"
#include "video.hpp"
#include "postprocess.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;
//  Wrapper class for running the inference
//  Must be kept alive throughout the program
//  Allocates CPU and GPU memory for inference.

class InferencePipeline{
    
    public:

        InferencePipeline(
            const fs::path& engineFilePath, 
            const fs::path& logFilePath,
            const fs::path& videoDirPath,
            const fs::path& saveDirPath,
            bool logModelInformation
        );

        bool createInferenceTensors();
        bool runInference();
        std::vector<std::string> getTensorNames(nvinfer1::TensorIOMode mode);
        size_t getNumElements(const char* tensorName);
        void logModelInfo();
        void runInferencePipeline();

    private:

        // Inference Runtime : Deserialize engine file
        // Engine : Representation of the model
        // Engine has the lifecycle of an entire inference run.
        // Execution Context: contains all of the state associated with a particular invocation

        Logger m_logger; 
        std::unique_ptr<nvinfer1::IRuntime> m_runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context;
        CudaTensorMap<cv::float16_t> m_DeviceTensorMap;
        nvinfer1::DataType _computeTypeInference;
        std::unique_ptr<ImageBatchLoader> m_batchLoader;
        std::unique_ptr<PostProcessor> m_postProcessor;
};


// Load serialized engine file to an std::vector
std::vector<char> readEngineFileToArray(const fs::path& fileName);
size_t getElementSize(nvinfer1::DataType dtype);
