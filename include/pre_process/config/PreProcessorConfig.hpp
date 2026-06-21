#pragma once

#include <cstddef>
#include <unordered_map>
#include <string>

#include "core/enums.hpp"
#include "pre_process/utils/enums.hpp"


struct PreProcessorConfig {

    ModelType modelType;
    PreferredProcessingDevice preferredDevice;
    ChannelOrderType imgRgbOrdering;
    size_t imgResizeHeight, imgResizeWidth, numImgChannels;
    DataType outputDataType;
    
    double imgScalingFactor = 1.0f;
    double imgMean = 0.0f;
    
    std::unordered_map<std::string, int> ndimsOfInputs;

};