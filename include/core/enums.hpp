#pragma once

enum class DeviceType {
    UNSET,
    CPU,
    CUDA
};


enum class MemoryType {
    UNSET,
    PageableHost,
    PinnedHost,
    CudaMem,
    Unified
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


enum class PreferredProcessingDevice {
    PREFER_CPU,
    PREFER_GPU
};

enum class DataType {
    Float32,
    Float16,
    Int8,
    Int32,
    Bool,
    UInt8,
    BFloat16,
    Int64
};


enum class IOMode {
    Input,
    Output,
    None
};
