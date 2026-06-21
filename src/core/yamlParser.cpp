#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "core/yamlParser.hpp"

namespace fs = std::filesystem;

namespace {

std::string normalize(std::string str) {
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        if (c == '-' || c == ' ' || c == '.') {
            return '_';
        }
        return static_cast<char>(std::tolower(c));
    });
    return str;
}

YAML::Node section(const YAML::Node& root, const std::string& name) {
    YAML::Node node = root[name];

    if (!node || !node.IsDefined()) {
        throw std::runtime_error("Missing YAML section: " + name);
    }

    if (!node.IsMap()) {
        throw std::runtime_error("YAML section must be a map: " + name);
    }

    return node;
}

template <typename T>
T required(const YAML::Node& node, const std::string& path, const std::string& key) {
    YAML::Node value = node[key];

    if (!value || !value.IsDefined()) {
        throw std::runtime_error("Missing YAML key: " + path + "." + key);
    }

    return value.as<T>();
}

template <typename T>
T optional(const YAML::Node& node, const std::string& key, const T& defaultValue) {
    YAML::Node value = node[key];

    if (!value || !value.IsDefined()) {
        return defaultValue;
    }

    return value.as<T>();
}

DataType parseDataType(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "float32" || v == "fp32") return DataType::Float32;
    if (v == "float16" || v == "fp16" || v == "half") return DataType::Float16;
    if (v == "int8") return DataType::Int8;
    if (v == "int32") return DataType::Int32;
    if (v == "bool" || v == "boolean") return DataType::Bool;
    if (v == "uint8") return DataType::UInt8;
    if (v == "bfloat16" || v == "bf16") return DataType::BFloat16;
    if (v == "int64") return DataType::Int64;

    throw std::runtime_error("Unsupported DataType string: " + raw);
}

IOMode parseIOMode(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "input" || v == "in") return IOMode::Input;
    if (v == "output" || v == "out") return IOMode::Output;
    if (v == "none") return IOMode::None;

    throw std::runtime_error("Unsupported IOMode string: " + raw);
}

FrameSourceType parseFrameSourceType(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "folder" || v == "dir" || v == "directory") return FrameSourceType::FOLDER;
    if (v == "video") return FrameSourceType::VIDEO;
    if (v == "unset") return FrameSourceType::UNSET;

    throw std::runtime_error("Unsupported FrameSourceType string: " + raw);
}

ChannelOrderType parseChannelOrder(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "rgb") return ChannelOrderType::RGB;
    if (v == "bgr") return ChannelOrderType::BGR;

    throw std::runtime_error("Unsupported ChannelOrderType string: " + raw);
}

PreferredProcessingDevice parsePreferredDevice(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "prefer_cpu" || v == "cpu") return PreferredProcessingDevice::PREFER_CPU;
    if (v == "prefer_gpu" || v == "gpu" || v == "cuda") return PreferredProcessingDevice::PREFER_GPU;

    throw std::runtime_error("Unsupported PreferredProcessingDevice string: " + raw);
}

BackendType parseBackendType(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "yolosegtrt" || v == "yolo_seg_trt" || v == "trt") return BackendType::YoloSegTRT;
    if (v == "unset") return BackendType::UNSET;

    throw std::runtime_error("Unsupported BackendType string: " + raw);
}

ModelType parseModelType(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "yolo_segmentation" || v == "segmentation") return ModelType::YOLO_SEGMENTATION;
    if (v == "yolo_detection" || v == "detection") return ModelType::YOLO_DETECTION;
    if (v == "unset") return ModelType::UNSET;

    throw std::runtime_error("Unsupported ModelType string: " + raw);
}

OutputType parseOutputType(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "yolo_raw_detection" || v == "raw_detection") {
        return OutputType::YOLO_RAW_DETECTION;
    }

    if (v == "yolo_raw_segmentation" || v == "raw_segmentation") {
        return OutputType::YOLO_RAW_SEGMENTATION;
    }

    if (v == "yolo_modified_segmentation" || v == "modified_segmentation") {
        return OutputType::YOLO_MODIFIED_SEGMENTATION;
    }

    if (v == "unset") return OutputType::UNSET;

    throw std::runtime_error("Unsupported OutputType string: " + raw);
}

LoggingSeverityType parseLoggingSeverity(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "info") return LoggingSeverityType::INFO;
    if (v == "warning" || v == "warn") return LoggingSeverityType::WARNING;
    if (v == "error") return LoggingSeverityType::ERROR;
    if (v == "verbose" || v == "debug") return LoggingSeverityType::VERBOSE;
    if (v == "internal_error") return LoggingSeverityType::INTERNAL_ERROR;

    throw std::runtime_error("Unsupported LoggingSeverityType string: " + raw);
}

TensorGroup parseTensorGroup(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "hostinput") return TensorGroup::HostInput;
    if (v == "pinnedinput") return TensorGroup::PinnedInput;
    if (v == "deviceinput") return TensorGroup::DeviceInput;
    if (v == "unifiedinput") return TensorGroup::UnifiedInput;

    if (v == "hostoutput") return TensorGroup::HostOutput;
    if (v == "pinnedoutput") return TensorGroup::PinnedOutput;
    if (v == "deviceoutput") return TensorGroup::DeviceOutput;
    if (v == "unifiedoutput") return TensorGroup::UnifiedOutput;

    if (v == "hostpostprocessoutput") return TensorGroup::HostPostProcessOutput;
    if (v == "devicepostprocessoutput") return TensorGroup::DevicePostProcessOutput;

    throw std::runtime_error("Unsupported TensorGroup string: " + raw);
}

ResultSinkType parseResultSinkType(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "save_detections" || v == "save") return ResultSinkType::SAVE_DETECTIONS;

    if (v == "draw_detections" || v == "draw") {
        return ResultSinkType::DRAW_DETCTIONS;
    }

    if (v == "unset") return ResultSinkType::UNSET;

    throw std::runtime_error("Unsupported ResultSinkType string: " + raw);
}

SaveDetectionMode parseSaveDetectionMode(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "normalized") return SaveDetectionMode::NORMALIZED;
    if (v == "raw") return SaveDetectionMode::RAW;
    if (v == "unset") return SaveDetectionMode::UNSET;

    throw std::runtime_error("Unsupported SaveDetectionMode string: " + raw);
}

DrawDetectionMode parseDrawDetectionMode(const std::string& raw) {
    const std::string v = normalize(raw);

    if (v == "unset") return DrawDetectionMode::UNSET;
    if (v == "boxes_only") return DrawDetectionMode::BOXES_ONLY;
    if (v == "masks_only") return DrawDetectionMode::MASKS_ONLY;
    if (v == "masks_with_boxes") return DrawDetectionMode::MASKS_WITH_BOXES;
    if (v == "contours_only") return DrawDetectionMode::CONTOURS_ONLY;

    if (v == "contours_with_boxes") {
        return DrawDetectionMode::COUNTOURS_WITH_BOXES;
    }

    throw std::runtime_error("Unsupported DrawDetectionMode string: " + raw);
}

Shape parseShape(const YAML::Node& shapeNode) {
    if (!shapeNode || !shapeNode.IsDefined() || !shapeNode.IsSequence()) {
        throw std::runtime_error("Tensor shape must be a sequence.");
    }

    Shape shape;

    for (const auto& dim : shapeNode) {
        shape.dims.push_back(dim.as<size_t>());
    }

    if (shape.dims.empty()) {
        throw std::runtime_error("Tensor shape cannot be empty.");
    }

    return shape;
}

TensorGroupList parseTensorGroupList(const YAML::Node& memory, const std::string& key) {
    TensorGroupList groups;

    YAML::Node node = memory[key];

    if (!node || !node.IsDefined()) {
        return groups;
    }

    if (!node.IsSequence()) {
        throw std::runtime_error("Expected sequence for memory." + key);
    }

    for (const auto& item : node) {
        groups.push_back(parseTensorGroup(item.as<std::string>()));
    }

    return groups;
}

TensorSpecMap parseTensorSpecMap(const YAML::Node& root, const std::string& sectionName) {
    YAML::Node tensorSection = section(root, sectionName);

    TensorSpecMap specs;

    for (const auto& it : tensorSection) {
        const std::string tensorName = it.first.as<std::string>();
        const YAML::Node tensorNode = it.second;

        TensorSpec spec;
        spec.shape = parseShape(tensorNode["shape"]);
        spec.dtype = parseDataType(required<std::string>(tensorNode, sectionName + "." + tensorName, "dtype"));
        spec.mode = parseIOMode(required<std::string>(tensorNode, sectionName + "." + tensorName, "mode"));

        specs.emplace(tensorName, spec);
    }

    return specs;
}

}  // namespace

AppSettings loadAppSettingsFromYaml(const fs::path& yamlPath) {
    YAML::Node root = YAML::LoadFile(yamlPath.string());

    AppSettings settings;

    const YAML::Node logging = section(root, "logging");
    const YAML::Node frameSource = section(root, "frame_source");
    const YAML::Node preprocess = section(root, "preprocess");
    const YAML::Node backend = section(root, "backend");
    const YAML::Node memory = section(root, "memory");
    const YAML::Node postprocess = section(root, "postprocess");
    const YAML::Node resultSink = section(root, "result_sink");

    settings.logFilePath = optional<std::string>(
        logging,
        "logFilePath",
        StaticSettings::DEFAULT_LOG_FILE
    );

    settings.loggingSeverity = parseLoggingSeverity(
        optional<std::string>(logging, "loggingSeverity", "info")
    );

    settings.frameSourceType = parseFrameSourceType(
        required<std::string>(frameSource, "frame_source", "frameSourceType")
    );

    settings.frameSourcePath = required<std::string>(
        frameSource,
        "frame_source",
        "frameSourcePath"
    );

    settings.origImgHeight = required<size_t>(
        frameSource,
        "frame_source",
        "origImgHeight"
    );

    settings.origImgWidth = required<size_t>(
        frameSource,
        "frame_source",
        "origImgWidth"
    );

    settings.batchSize = required<size_t>(
        frameSource,
        "frame_source",
        "batchSize"
    );

    settings.imgChannelOrdering = parseChannelOrder(
        required<std::string>(preprocess, "preprocess", "imgChannelOrdering")
    );

    settings.imgPreProcessedImgH = required<size_t>(
        preprocess,
        "preprocess",
        "imgPreProcessedImgH"
    );

    settings.imgPreProcessedImgW = required<size_t>(
        preprocess,
        "preprocess",
        "imgPreProcessedImgW"
    );

    settings.preprocessedDataType = parseDataType(
        required<std::string>(preprocess, "preprocess", "preprocessedDataType")
    );

    settings.imgPreProcessScalingFactor = optional<float>(
        preprocess,
        "imgPreProcessScalingFactor",
        1.0f
    );

    settings.imgPreProcessMeanFactor = optional<float>(
        preprocess,
        "imgPreProcessingMeanFactor",
        0.0f
    );

    settings.preferredDevicePreProc = parsePreferredDevice(
        optional<std::string>(preprocess, "preferredDevicePreProc", "prefer_cpu")
    );

    YAML::Node ndimsNode = preprocess["ndimsOfInputs"];
    if (ndimsNode && ndimsNode.IsDefined()) {
        for (const auto& it : ndimsNode) {
            settings.ndimsOfInputs[it.first.as<std::string>()] = it.second.as<int>();
        }
    }

    settings.inferenceBackendType = parseBackendType(
        required<std::string>(backend, "backend", "inferenceBackendType")
    );

    settings.modelType = parseModelType(
        required<std::string>(backend, "backend", "modelType")
    );

    settings.outputType = parseOutputType(
        required<std::string>(backend, "backend", "outputType")
    );

    settings.preferredInferenceDevice = parsePreferredDevice(
        required<std::string>(backend, "backend", "preferredInferenceDevice")
    );

    settings.serializedModelPath = required<std::string>(
        backend,
        "backend",
        "serializedModelPath"
    );

    settings.preProcessingTensorGroups =
        parseTensorGroupList(memory, "preProcessingTensorGroups");

    settings.inferenceTensorGroups =
        parseTensorGroupList(memory, "inferenceTensorGroups");

    settings.postProcessingTensorGroups =
        parseTensorGroupList(memory, "postProcessingTensorGroups");

    settings.inputTensorSpecs = parseTensorSpecMap(root, "inputTensorSpecs");
    settings.outputTensorSpecs = parseTensorSpecMap(root, "outputTensorSpecs");

    settings.preferredDevicePostProc = parsePreferredDevice(
        optional<std::string>(postprocess, "preferredDevicePostProc", "prefer_cpu")
    );

    settings.confThreshold = optional<float>(
        postprocess,
        "confThreshold",
        0.25f
    );

    settings.iouThreshold = optional<float>(
        postprocess,
        "iouThreshold",
        0.50f
    );

    settings.maskThreshold = optional<float>(
        postprocess,
        "maskThreshold",
        0.50f
    );

    settings.maxDetections = optional<int>(
        postprocess,
        "maxDetections",
        300
    );

    YAML::Node outputStartsNode = postprocess["outputTensorStartLocs"];
    if (outputStartsNode && outputStartsNode.IsDefined()) {
        for (const auto& it : outputStartsNode) {
            settings.outputTensorStartLocs[it.first.as<std::string>()] =
                it.second.as<size_t>();
        }
    }

    settings.resultsDir = required<std::string>(
        resultSink,
        "result_sink",
        "resultsDir"
    );

    settings.resultSinkType = parseResultSinkType(
        optional<std::string>(resultSink, "resultSinkType", "save_detections")
    );

    settings.saveDetMode = parseSaveDetectionMode(
        optional<std::string>(resultSink, "saveDetMode", "normalized")
    );

    settings.drawDetMode = parseDrawDetectionMode(
        optional<std::string>(resultSink, "drawDetMode", "unset")
    );

    settings.lineThickness = optional<int>(
        resultSink,
        "lineThickness",
        1
    );

    return settings;
}