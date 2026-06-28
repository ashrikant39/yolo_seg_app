#pragma once

#include <filesystem>

#include "core/enums.hpp"
#include "backends/utils/enums.hpp"

namespace fs = std::filesystem;

/**
 * @brief Configuration used to create an inference backend.
 */
struct InferenceBackendConfig {
    ///< Backend implementation to instantiate.
    BackendType inferBackend;
    ///< Model family used by downstream factories.
    ModelType modelType;
    ///< Preferred execution device.
    PreferredProcessingDevice processDevice;
    ///< Serialized model or engine path.
    fs::path modelFilePath;
};
