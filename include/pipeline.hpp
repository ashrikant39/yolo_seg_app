#pragma once

#include <NvInfer.h>
#include <vector>
#include <opencv2/core.hpp>
#include <filesystem>
#include "types/tensor_types.h"
#include "logger.h"
#include "video.h"
#include "postprocess.h"

namespace fs = std::filesystem;
using namespace std::chrono_literals;
//  Wrapper class for running the inference
//  Must be kept alive throughout the program
//  Allocates CPU and GPU memory for inference.

class InferencePipeline
{
    public:

        InferencePipeline(
            const fs::path& engineFilePath, 
            const fs::path& logFilePath,
            const fs::path& videoDirPath,
            const fs::path& saveDirPath,
            bool logModelInformation
        );

        ~InferencePipeline();

        bool createInferenceTensors();
        bool runAsynchronousInference(const cv::Mat& preprocessedImages);
        std::vector<std::string> getTensorNames(nvinfer1::TensorIOMode mode);
        size_t getNumElements(const char* tensorName);
        void logModelInfo();
        void runInferencePipeline();

    private:

        // Inference Runtime : Deserialize engine file
        // Engine : Representation of the model
        // Engine has the lifecycle of an entire inference run.
        // Execution Context: contains all of the state associated with a particular invocation

        Logger _logger; 
        std::unique_ptr<nvinfer1::IRuntime> _runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> _engine;
        TensorMap<cv::float16_t> _DeviceTensorMap, _outputTensorMap;
        nvinfer1::DataType _computeTypeInference;
        std::unique_ptr<VideoFromDirectory> _videoReader;
        std::unique_ptr<PostProcessor> _postProcessor;
};


// Load serialized engine file to an std::vector
std::vector<char> readEngineFileToArray(const fs::path& fileName);
size_t getElementSize(nvinfer1::DataType dtype);
