#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

constexpr size_t FRAME_START = 0;
constexpr uint64_t INVALID_TIMESTAMP = 0;

struct FrameMetadata {

    uint64_t frameId = FRAME_START;
    uint64_t timestampNs = INVALID_TIMESTAMP;
    bool isPadding = false;

    // Important paths
    fs::path sourcePath;
    fs::path imagePath;
    fs::path resultsDir;
    fs::path saveDetPath;
    std::string saveMaskDirName;

    // Original image geometry
    size_t originalWidth = 0;
    size_t originalHeight = 0;
    size_t originalChannels = 0;

    // Network input geometry
    size_t inputWidth = 0;
    size_t inputHeight = 0;

    // Network output geometry
    size_t outputWidth = 0;
    size_t outputHeight = 0;
    
};

struct Frame {
    cv::Mat image;
    FrameMetadata metadata;
};

// For measurements
inline uint64_t nowNs() {
    return std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()
                .time_since_epoch()).count();
}

// For Logging
inline uint64_t wallClockNowNs() {
    return std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::system_clock::now()
                .time_since_epoch()).count();
}


struct BatchFrameData {
    std::vector<cv::Mat> images;
    std::vector<FrameMetadata> metas;
};


inline size_t countSourceFrames(const BatchFrameData& batch) {
    return static_cast<size_t>(std::count_if(
        batch.metas.begin(),
        batch.metas.end(),
        [](const FrameMetadata& metadata) {
            return !metadata.isPadding;
        }
    ));
}