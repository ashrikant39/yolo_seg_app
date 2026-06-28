#pragma once

/**
 * @brief Compute device that owns or consumes tensor data.
 */
enum class DeviceType {
    UNSET,
    CPU,
    CUDA
};


/**
 * @brief Allocation strategy used for tensor buffers.
 */
enum class MemoryType {
    UNSET,
    PageableHost,
    PinnedHost,
    CudaMem,
    Unified
};

/**
 * @brief Supported model families.
 */
enum class ModelType {
    UNSET,                          // Not Set, figured out from output names in the config
    YOLO_DETECTION,                  // Yolo Detection Model
    YOLO_SEGMENTATION                // Yolo Segmentation Model
};

/**
 * @brief Supported model output tensor layouts.
 */
enum class OutputType {
    UNSET,                      // Not Set, figured out from output names in the config
    YOLO_RAW_DETECTION,              // Raw Unmodified Outputs of Detection Model
    YOLO_RAW_SEGMENTATION,           // Raw Unmodified Outputs of Segmentation Model
    YOLO_MODIFIED_SEGMENTATION       // Modified Outputs of Segmentation Model
};


/**
 * @brief Preferred device for processor selection when multiple options exist.
 */
enum class PreferredProcessingDevice {
    PREFER_CPU,
    PREFER_GPU
};

/**
 * @brief Element data types accepted by tensor specifications.
 */
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


/**
 * @brief Tensor role in a model IO contract.
 */
enum class IOMode {
    Input,
    Output,
    None
};
