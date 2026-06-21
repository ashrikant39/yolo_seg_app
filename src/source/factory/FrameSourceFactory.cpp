#include "source/factory/FrameSourceFactory.hpp"

std::unique_ptr<FrameSource> createFrameSource(const FrameSourceConfig& config) {

    switch (config.frameSourceType) {

        case FrameSourceType::FOLDER:
            return std::make_unique<FolderFrameSource>(config);

        case FrameSourceType::VIDEO:
            return std::make_unique<VideoFrameSource>(config);

        case FrameSourceType::UNSET:
            throw std::runtime_error("Unsupported source format");

        default:
            throw std::runtime_error("Unsupported source format");
    }
}
