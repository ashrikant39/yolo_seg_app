#pragma once

#include <string>
#include <unordered_map>

#include "core/enums.hpp"
#include "core/tensor.hpp"

struct PostProcessorConfig {

    // std::vector<std::string> outputNames;
    // std::unordered_map<std::string, Shape> outputDims;
    // std::unordered_map<std::string, DataType> outputTypes;
    TensorSpecMap outputSpecs;

    ModelType modelType;
    OutputType outputType;
    PreferredProcessingDevice preferedDevice;
    
    float confThreshold = 0.25f;
    float iouThreshold = 0.50f;
    float maskThreshold = 0.5f;
    int maxDetections = 300;

    // size_t boxStart = -1;
    // size_t maskStart = -1;
    // size_t classStart = -1;
    // size_t coeffStart = -1;
    // size_t objectnessStart = -1;
    std::unordered_map<std::string, size_t> outputTensorStartLocs;

};
