#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <NvInfer.h>
#include <memory>

#include "core/utils.hpp"

struct OutputInfo {
    Shape shape;
    DType type;
};

struct PostProcessorConfig {

    // std::vector<std::string> outputNames;
    // std::unordered_map<std::string, Shape> outputDims;
    // std::unordered_map<std::string, DType> outputTypes;
    std::unordered_map<std::string, OutputInfo> outputInfos;

    ModelType modelType;
    OutputType outputType;
    ProcessDevice preferedDevice;
    
    float confThreshold = 0.25f;
    float iouThreshold = 0.50f;
    float maskThreshold = 0.5f;
    int maxDetections = 300;

    size_t boxStart = -1;
    size_t maskStart = -1;
    size_t classStart = -1;
    size_t coeffStart = -1;
    size_t objectnessStart = -1;

};