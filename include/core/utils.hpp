#pragma once


enum class LoggingSeverity {
    INFO,
    WARNING,
    ERROR,
    VERBOSE,
    INTERNAL_ERROR,
};

enum class ModelType {
    UNSET,                          // Not Set, figured out from output names in the config
    YOLO_DETECTION,                  // Yolo Detection Model
    YOLO_SEGMENTATION                // Yolo Segmentation Model
};

enum class OutputType {
    UNSET,                      // Not Set, figured out from output names in the config
    YOLO_RAW_DETECTION,              // Raw Unmodified Outputs of Detection Model
    YOLO_RAW_SEGMENTATION,           // Raw Unmodified Outputs of Segmentation Model
    YOLO_MODIFIED_SEGMENTATION       // Modified Outputs of Segmentation Model
};


enum class ProcessDevice {
    PREFER_CPU,
    PREFER_GPU
};
