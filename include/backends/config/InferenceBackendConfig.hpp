#pragma once

#include <filesystem>

#include "core/enums.hpp"
#include "backends/utils/enums.hpp"

namespace fs = std::filesystem;

struct InferenceBackendConfig {
    BackendType inferBackend;
    ModelType modelType;
    PreferredProcessingDevice processDevice;
    fs::path modelFilePath;
};
