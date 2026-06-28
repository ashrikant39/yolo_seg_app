#pragma once

#include <string>
#include <unordered_map>

#include "core/enums.hpp"
#include "core/tensor.hpp"

/**
 * @brief Configuration for converting backend outputs into detections.
 */
struct PostProcessorConfig {

    ///< Output tensor specifications keyed by model tensor name.
    TensorSpecMap outputSpecs;

    ///< Model/output mode and desired postprocessing device.
    ModelType modelType;
    OutputType outputType;
    PreferredProcessingDevice preferedDevice;
    
    ///< Detection and mask filtering thresholds.
    float confThreshold = 0.25f;
    float iouThreshold = 0.50f;
    float maskThreshold = 0.5f;
    int maxDetections = 300;

    ///< Start offsets into output tensors by tensor name.
    std::unordered_map<std::string, size_t> outputTensorStartLocs;

};
