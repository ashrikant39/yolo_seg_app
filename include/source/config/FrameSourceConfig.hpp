#pragma once

#include <filesystem>

#include "source/utils/enums.hpp"

namespace fs = std::filesystem;

/**
 * @brief Configuration for folder or video frame sources.
 */
struct FrameSourceConfig {
    ///< Source implementation to create.
    FrameSourceType frameSourceType;
    ///< Folder path or video file path.
    fs::path sourcePath;
    ///< Original frame geometry and batch size.
    size_t imgHeight, imgWidth, batchSize;
};
