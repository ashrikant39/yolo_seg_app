#pragma once

#include <cstddef>

#include "core/utils.hpp"

enum class ChannelOrder {
    RGB,
    BGR,
};

struct PreProcessorConfig {

    ModelType modelType;
    ProcessDevice preferredDevice;
    ChannelOrder rgbOrdering;
    size_t resizeHeight, resizeWidth, n_channels;
    DType outputDtype;
    
    double scalingFactor = 1.0f;
    double mean = 0.0f;
    
    int ndimsInput = 4;

};
