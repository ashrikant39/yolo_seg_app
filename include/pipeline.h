#pragma once

#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include "logger.h"
#include <vector>
#include <video.h>

using namespace nvinfer1;
//  Wrapper class for running the inference
class InferencePipeline
{
    public:

        InferencePipeline(const std::string&, Logger&);

        ~InferencePipeline(){
        }

        void LoadImagesToDevice(const std::vector<cv::Mat>&);
        void allocateIODeviceMemory();
        void runAsynchronousInference(const cv::Mat&, std::unordered_map<std::string, void*>&);
        void runSynchronousInference();

    private:

        // Inference Runtime : Deserialize engine file
        // Engine : Representation of the model
        // Engine has the lifecycle of an entire inference run.
        // Execution Context: contains all of the state associated with a particular invocation

        ILogger& m_logger; 
        std::unique_ptr<IRuntime> m_runtime;
        std::unique_ptr<ICudaEngine> m_engine;
        std::unordered_map<std::string, void*> m_DevicePtrMap;
};


// Load serialized engine file to an std::vector
std::vector<char> readEngineFileToArray(const std::string&);
size_t getElementSize(DataType);