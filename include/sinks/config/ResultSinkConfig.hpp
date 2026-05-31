#pragma once

enum class ResultSinkMode {
    UNSET,
    SAVE_DETECTIONS,
    DRAW_DETCTIONS
};

enum class SaveDetectionMode {
    UNSET,
    NORMALIZED,
    RAW,
};

enum class DrawDetectionMode {
    UNSET,
    BOXES_ONLY,
    MASKS_ONLY,
    MASKS_WITH_BOXES,
    CONTOURS_ONLY,
    COUNTOURS_WITH_BOXES,
};

// enum class MaskDrawingMode {
//     UNSET,
//     FROM_CONTOUR,
//     RAW
// }

struct ResultSinkConfig {
    
    ResultSinkMode sinkMode;
    SaveDetectionMode saveMode;
    DrawDetectionMode drawDetectionMode;
    int lineThickness = 1;
    // MaskDrawingMode drawMaskMode;
};