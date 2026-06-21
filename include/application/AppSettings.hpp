#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>

#include "backends/utils/enums.hpp"
#include "core/enums.hpp"
#include "core/tensor.hpp"
#include "logging/enums.hpp"
#include "memory_management/enums.hpp"
#include "pre_process/utils/enums.hpp"
#include "sinks/utils/enums.hpp"
#include "source/utils/enums.hpp"

namespace fs = std::filesystem;

struct StaticSettings {
    static constexpr const char* DEFAULT_LOG_FILE = "main.log";
    static constexpr LoggingSeverityType DEFAULT_LOG_SEVERITY = LoggingSeverityType::INFO;
    static constexpr size_t NUM_IMG_CHANNELS = 3;
};

struct AppSettings {

    fs::path logFilePath = StaticSettings::DEFAULT_LOG_FILE;
    LoggingSeverityType loggingSeverity = StaticSettings::DEFAULT_LOG_SEVERITY;

    FrameSourceType frameSourceType = FrameSourceType::FOLDER;
    fs::path frameSourcePath;
    size_t origImgHeight = 0;
    size_t origImgWidth = 0;
    size_t batchSize = 1;

    ChannelOrderType imgChannelOrdering = ChannelOrderType::BGR;
    size_t imgPreProcessedImgH = 0;
    size_t imgPreProcessedImgW = 0;
    DataType preprocessedDataType = DataType::Float32;
    float imgPreProcessScalingFactor = 1.0f;
    float imgPreProcessMeanFactor = 0.0f;
    PreferredProcessingDevice preferredDevicePreProc = PreferredProcessingDevice::PREFER_CPU;
    std::unordered_map<std::string, int> ndimsOfInputs;

    BackendType inferenceBackendType = BackendType::UNSET;
    ModelType modelType = ModelType::UNSET;
    PreferredProcessingDevice preferredInferenceDevice = PreferredProcessingDevice::PREFER_GPU;
    fs::path serializedModelPath;

    TensorGroupList preProcessingTensorGroups;
    TensorGroupList inferenceTensorGroups;
    TensorGroupList postProcessingTensorGroups;
    TensorSpecMap inputTensorSpecs;
    TensorSpecMap outputTensorSpecs;

    OutputType outputType = OutputType::UNSET;
    PreferredProcessingDevice preferredDevicePostProc = PreferredProcessingDevice::PREFER_CPU;
    std::unordered_map<std::string, size_t> outputTensorStartLocs;
    float confThreshold = 0.25f;
    float iouThreshold = 0.50f;
    float maskThreshold = 0.50f;
    int maxDetections = 300;

    ResultSinkType resultSinkType = ResultSinkType::SAVE_DETECTIONS;
    SaveDetectionMode saveDetMode = SaveDetectionMode::NORMALIZED;
    DrawDetectionMode drawDetMode = DrawDetectionMode::UNSET;
    int lineThickness = 1;
    fs::path resultsDir;
};