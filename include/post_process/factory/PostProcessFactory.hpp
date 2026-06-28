#pragma once


#include "post_process/config/PostProcessorConfig.hpp"
#include "post_process/interface/PostProcessor.hpp"

/**
 * @brief Extracts configured output tensor names from a tensor specification map.
 *
 * @param outputSpecs Tensor specifications keyed by model output name.
 * @return Output tensor names in the map iteration order.
 */
inline std::vector<std::string> getOutputTensorNames(const TensorSpecMap& outputSpecs) {
    std::vector<std::string> names;
    names.reserve(outputSpecs.size());
    for (const auto& [name, info] : outputSpecs) {
        names.push_back(name);
    }
    return names;
}

/**
 * @brief Infers the YOLO model family from configured output tensor names.
 *
 * A single output is treated as a detection model. Multiple outputs are treated
 * as a segmentation model.
 *
 * @param outputNames Configured model output tensor names.
 * @return Inferred model type.
 * @throws std::runtime_error when no output names are configured.
 */
inline ModelType getModelTypeFromNames(const std::vector<std::string>& outputNames) {

    if (outputNames.empty()) {
        throw std::runtime_error("Config has output names missing");
    }
    if ( outputNames.size() == 1 ) {
        return ModelType::YOLO_DETECTION;

    } else {
        return ModelType::YOLO_SEGMENTATION;
    }

}

/**
 * @brief Infers the output layout variant from configured output tensor names.
 *
 * Raw TensorRT export names such as `output0`/`output1` are classified as raw
 * segmentation outputs. Named tensors such as `boxes`, `masks`, `classlabel`,
 * and `objectness` are classified as modified segmentation outputs.
 *
 * @param outputNames Configured model output tensor names.
 * @return Inferred output tensor layout type.
 * @throws std::runtime_error when no output names are configured.
 */
inline OutputType getOutputTypeFromNames(const std::vector<std::string>& outputNames) {

    if (outputNames.empty()) {
        throw std::runtime_error("Config has output names missing");
    }

    if ( outputNames.size() == 1 ) {
        return OutputType::YOLO_RAW_DETECTION;
    }

    for (const std::string& name : outputNames) {

        std::string::size_type idx = name.find("output");

        if ( idx != std::string::npos && std::isdigit(name.at(idx + 6)) ) {
            return OutputType::YOLO_RAW_SEGMENTATION;
        }
    }

    return OutputType::YOLO_MODIFIED_SEGMENTATION;
}

/**
 * @brief Validates and completes postprocessor settings before construction.
 *
 * @param config Mutable postprocessor config loaded from YAML.
 */
void validateParameters(PostProcessorConfig& config);

/**
 * @brief Creates the configured postprocessor implementation.
 *
 * @param config Postprocessor construction settings.
 * @return Owning pointer to the selected postprocessor.
 */
std::unique_ptr<PostProcessor> createPostProcessor(PostProcessorConfig config);
