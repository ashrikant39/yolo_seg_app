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

/**
 * @brief Compile-time defaults shared by the application.
 */
struct StaticSettings {
    static constexpr const char* DEFAULT_LOG_FILE = "main.log";
    static constexpr LoggingSeverityType DEFAULT_LOG_SEVERITY = LoggingSeverityType::INFO;
    static constexpr size_t NUM_IMG_CHANNELS = 3;
};

/**
 * @brief Fully parsed runtime configuration for the application.
 *
 * This is populated from YAML by loadAppSettingsFromYaml() and then fanned out
 * into the individual source, preprocessing, backend, memory, postprocessing,
 * and sink configuration structs.
 */
struct AppSettings {

    /** @brief Log file path and minimum severity. */
    fs::path logFilePath = StaticSettings::DEFAULT_LOG_FILE;
    LoggingSeverityType loggingSeverity = StaticSettings::DEFAULT_LOG_SEVERITY;

    /** @brief Frame source selection and source image geometry. */
    FrameSourceType frameSourceType = FrameSourceType::FOLDER;
    fs::path frameSourcePath;
    size_t origImgHeight = 0;
    size_t origImgWidth = 0;
    size_t batchSize = 1;

    /** @brief Preprocessing options used before inference. */
    ChannelOrderType imgChannelOrdering = ChannelOrderType::BGR;
    size_t imgPreProcessedImgH = 0;
    size_t imgPreProcessedImgW = 0;
    DataType preprocessedDataType = DataType::Float32;
    float imgPreProcessScalingFactor = 1.0f;
    float imgPreProcessMeanFactor = 0.0f;
    PreferredProcessingDevice preferredDevicePreProc = PreferredProcessingDevice::PREFER_CPU;
    std::unordered_map<std::string, int> ndimsOfInputs;

    /** @brief Inference backend and serialized model settings. */
    BackendType inferenceBackendType = BackendType::UNSET;
    ModelType modelType = ModelType::UNSET;
    PreferredProcessingDevice preferredInferenceDevice = PreferredProcessingDevice::PREFER_GPU;
    fs::path serializedModelPath;

    /** @brief Tensor groups and model IO tensor specifications. */
    TensorGroupList preProcessingTensorGroups;
    TensorGroupList inferenceTensorGroups;
    TensorGroupList postProcessingTensorGroups;
    TensorSpecMap inputTensorSpecs;
    TensorSpecMap outputTensorSpecs;

    /** @brief Postprocessing mode and thresholds. */
    OutputType outputType = OutputType::UNSET;
    PreferredProcessingDevice preferredDevicePostProc = PreferredProcessingDevice::PREFER_CPU;
    std::unordered_map<std::string, size_t> outputTensorStartLocs;
    float confThreshold = 0.25f;
    float iouThreshold = 0.50f;
    float maskThreshold = 0.50f;
    int maxDetections = 300;

    /** @brief Result sink mode and output directory. */
    ResultSinkType resultSinkType = ResultSinkType::SAVE_DETECTIONS;
    SaveDetectionMode saveDetMode = SaveDetectionMode::NORMALIZED;
    DrawDetectionMode drawDetMode = DrawDetectionMode::UNSET;
    int lineThickness = 1;
    fs::path resultsDir;
};
