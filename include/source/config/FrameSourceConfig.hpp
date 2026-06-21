#pragma once

#include <filesystem>

#include "source/utils/enums.hpp"

namespace fs = std::filesystem;
struct FrameSourceConfig {
    FrameSourceType frameSourceType;
    fs::path sourcePath;
    size_t imgHeight, imgWidth, batchSize;
};
