#pragma once

#include "sinks/utils/enums.hpp"

struct ResultSinkConfig {
    
    ResultSinkType sinkType;
    SaveDetectionMode saveMode;
    DrawDetectionMode drawDetectionMode;
    int lineThickness = 1;
    // MaskDrawingMode drawMaskMode;
};