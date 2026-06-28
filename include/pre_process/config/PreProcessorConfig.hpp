#pragma once

#include <cstddef>
#include <unordered_map>
#include <string>

#include "core/enums.hpp"
#include "pre_process/utils/enums.hpp"


/**
 * @brief Configuration for preprocessing input frames into model tensors.
 */
struct PreProcessorConfig {

    ///< Model and device selection.
    ModelType modelType;
    PreferredProcessingDevice preferredDevice;
    ///< Input channel interpretation and output tensor geometry.
    ChannelOrderType imgRgbOrdering;
    size_t imgResizeHeight, imgResizeWidth, numImgChannels;
    ///< Output tensor scalar type.
    DataType outputDataType;
    
    ///< Normalization parameters passed to OpenCV blob creation.
    double imgScalingFactor = 1.0f;
    double imgMean = 0.0f;
    
    ///< Expected input tensor rank by tensor name.
    std::unordered_map<std::string, int> ndimsOfInputs;

};
