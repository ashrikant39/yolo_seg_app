#pragma once

#include "source/config/FrameSourceConfig.hpp"
#include "source/interface/FrameSource.hpp"
#include "source/modes/FolderFrameSource.hpp"
#include "source/modes/VideoFrameSource.hpp"


/**
 * @brief Create a frame source from configuration.
 * @param config Source configuration.
 * @return Owning pointer to the selected source.
 * @throws std::runtime_error for unsupported source types or invalid paths.
 */
std::unique_ptr<FrameSource> createFrameSource(const FrameSourceConfig& config);
