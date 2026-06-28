#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

constexpr size_t FRAME_START = 0;
constexpr uint64_t INVALID_TIMESTAMP = 0;

/**
 * @brief Metadata carried with each source frame and enriched through the pipeline.
 */
struct FrameMetadata {

    /** @brief Monotonic source frame id. */
    uint64_t frameId = FRAME_START;
    /** @brief Source timestamp in nanoseconds when available. */
    uint64_t timestampNs = INVALID_TIMESTAMP;
    /** @brief True when this is a zero-padding frame. */
    bool isPadding = false;

    /** @brief Source path for the frame source. */
    fs::path sourcePath;
    /** @brief Path of the image represented by this frame. */
    fs::path imagePath;
    /** @brief Directory where results for this run are written. */
    fs::path resultsDir;
    /** @brief Output path for serialized detections. */
    fs::path saveDetPath;
    /** @brief Directory name used for saved mask artifacts. */
    std::string saveMaskDirName;

    /** @brief Original image width in pixels. */
    size_t originalWidth = 0;
    /** @brief Original image height in pixels. */
    size_t originalHeight = 0;
    /** @brief Original image channel count. */
    size_t originalChannels = 0;

    /** @brief Network input width in pixels. */
    size_t inputWidth = 0;
    /** @brief Network input height in pixels. */
    size_t inputHeight = 0;

    /** @brief Network output width in pixels. */
    size_t outputWidth = 0;
    /** @brief Network output height in pixels. */
    size_t outputHeight = 0;
    
};

/**
 * @brief Image plus metadata produced by a FrameSource.
 */
struct Frame {
    cv::Mat image;
    FrameMetadata metadata;
};

/**
 * @brief Returns a monotonic timestamp in nanoseconds for elapsed-time measurement.
 */
inline uint64_t nowNs() {
    return std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()
                .time_since_epoch()).count();
}

/**
 * @brief Returns wall-clock timestamp in nanoseconds for logs and metadata.
 */
inline uint64_t wallClockNowNs() {
    return std::chrono::duration_cast<
        std::chrono::nanoseconds>(
            std::chrono::system_clock::now()
                .time_since_epoch()).count();
}


/**
 * @brief A batch of images and their corresponding metadata.
 */
struct BatchFrameData {
    /** Batch images in source order. */
    std::vector<cv::Mat> images;
    /** Metadata entries corresponding to images. */
    std::vector<FrameMetadata> metas;
};

/**
 * @brief Count non-padding frames in a batch.
 * @param batch Batch metadata to inspect.
 * @return Number of real source frames.
 */
inline size_t countSourceFrames(const BatchFrameData& batch) {
    return static_cast<size_t>(std::count_if(
        batch.metas.begin(),
        batch.metas.end(),
        [](const FrameMetadata& metadata) {
            return !metadata.isPadding;
        }
    ));
}
