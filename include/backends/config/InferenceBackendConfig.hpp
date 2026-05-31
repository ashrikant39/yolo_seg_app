#pragma once

#include <filesystem>
#include "core/utils.hpp"

namespace fs = std::filesystem;

enum class BackendType {
    UNSET,
    YoloSegTRT
};

struct InferenceBackendConfig {
    BackendType inferBackend;
    ModelType modelType;
    ProcessDevice processDevice;
    fs::path modelFilePath;
};
