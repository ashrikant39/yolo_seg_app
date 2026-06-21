#pragma once

#include "source/config/FrameSourceConfig.hpp"
#include "source/interface/FrameSource.hpp"
#include "source/modes/FolderFrameSource.hpp"
#include "source/modes/VideoFrameSource.hpp"


std::unique_ptr<FrameSource> createFrameSource(const FrameSourceConfig& config);