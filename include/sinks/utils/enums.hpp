#pragma once

/**
 * @brief Result sink implementation selected by YAML.
 */
enum class ResultSinkType {
    UNSET,
    SAVE_DETECTIONS,
    DRAW_DETCTIONS
};

/**
 * @brief Coordinate mode used when writing binary detections.
 */
enum class SaveDetectionMode {
    UNSET,
    NORMALIZED,
    RAW,
};

/**
 * @brief Drawing overlay mode used by the image result sink.
 */
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
