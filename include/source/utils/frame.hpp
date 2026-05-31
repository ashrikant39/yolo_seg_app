#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>

constexpr size_t FRAME_START = 0;
constexpr uint64_t INVALID_TIMESTAMP = 0;

struct FrameMetaData { 
    uint64_t frameId;
    uint64_t timeStampNs;
    fs::path sourcePath;
};


struct Frame {
    cv::Mat image;
    FrameMetaData metadata;
};


struct BatchFrameData {
    std::vector<cv::Mat> images;
    std::vector<FrameMetaData> metas;
};