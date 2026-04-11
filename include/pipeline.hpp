#pragma once

#include <NvInfer.h>
#include <vector>
#include <opencv2/core.hpp>
#include <filesystem>
#include "utils/tensor.hpp"
#include "logger.hpp"
#include "video.hpp"
#include "postprocess.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

/**
 * @brief High-level TensorRT inference pipeline.
 *
 * This class owns the TensorRT runtime/engine/context and orchestrates:
 * 1) input loading/preprocessing,
 * 2) GPU inference execution,
 * 3) output post-processing and result writing.
 *
 * The object is intended to live for the full duration of an inference session.
 */
class InferencePipeline{
    
    public:

        /**
         * @brief Construct a pipeline from a serialized TensorRT engine file.
         * @param engineFilePath Path to `.engine` file.
         * @param logFilePath Path to the log file used by Logger.
         * @param videoDirPath Directory containing input images.
         * @param saveDirPath Directory where predictions/masks are written.
         * @param logModelInformation If true, logs tensor/layer metadata.
         */
        InferencePipeline(
            const fs::path& engineFilePath, 
            const fs::path& logFilePath,
            const fs::path& videoDirPath,
            const fs::path& saveDirPath,
            bool logModelInformation
        );

        /**
         * @brief Allocate and register TensorRT IO tensors.
         * @return true on success.
         */
        bool createInferenceTensors();
        /**
         * @brief Run one inference iteration on already prepared input buffers.
         * @return true if enqueue + synchronization succeeded.
         */
        bool runInference();
        /**
         * @brief Get tensor names by IO mode (input/output).
         */
        std::vector<std::string> getTensorNames(nvinfer1::TensorIOMode mode);
        /**
         * @brief Compute total number of elements for an engine tensor.
         */
        size_t getNumElements(const char* tensorName);
        /**
         * @brief Log selected model tensor/layer metadata.
         */
        void logModelInfo();
        /**
         * @brief Run the full pipeline over all image batches.
         */
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
        CudaTensorMap m_DeviceTensorMap;
        nvinfer1::DataType _computeTypeInference;
        std::unique_ptr<ImageBatchLoader> m_batchLoader;
        std::unique_ptr<PostProcessor> m_postProcessor;
};


/**
 * @brief Load a serialized TensorRT engine file into a byte buffer.
 */
std::vector<char> readEngineFileToArray(const fs::path& fileName);