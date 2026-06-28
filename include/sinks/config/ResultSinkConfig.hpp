#pragma once

#include "sinks/utils/enums.hpp"

/**
 * @brief Configuration for final postprocess output consumption.
 */
struct ResultSinkConfig {
    
    ///< Sink implementation to create.
    ResultSinkType sinkType;
    ///< Binary detection save mode.
    SaveDetectionMode saveMode;
    ///< Drawing strategy for image outputs.
    DrawDetectionMode drawDetectionMode;
    ///< Line thickness for drawing boxes or contours.
    int lineThickness = 1;
};
