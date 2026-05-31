#pragma once

#include <filesystem>

namespace fs = std::filesystem;

enum class Source {
    UNSET,
    FOLDER,
    VIDEO,
};

struct FrameSourceConfig {
    Source frameSourceType;
    fs::path sourcePath;
    size_t imgHeight, imgWidth, batchSize;
};