#pragma once

#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include "logger.h"
#include <vector>
#include <video.h>

using NVRuntime = nvinfer1::IRuntime;
using NVEngine = nvinfer1::ICudaEngine;
using NVExecContext = nvinfer1::IExecutionContext;

//  Wrapper class for running the inference
class InferenceEngine
{
    public:

        InferenceEngine(const std::string&, Logger&);

        ~InferenceEngine()
        {
            m_executionContext.reset();
            m_engine.reset();
            m_runtime.reset();
        }

        void createNewEngine(const std::string&);
        void createNewExecutionContext();

        void LoadInputBuffer(const std::vector<cv::Mat>&, void*, int, int, int, int);
        void runAsynchronousInference();
        void runSynchronousInference();

    private:

        // Inference Runtime : Deserialize engine file
        // Engine : Representation of the model
        // Engine has the lifecycle of an entire inference run.
        // Execution Context: contains all of the state associated with a particular invocation

        NVLogger& m_logger; 
        std::unique_ptr<NVRuntime> m_runtime;
        std::unique_ptr<NVEngine> m_engine;
        std::unique_ptr<NVExecContext> m_executionContext;
};

