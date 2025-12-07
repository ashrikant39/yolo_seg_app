#pragma once

#include <vector>
#include <array>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

struct ImageBatchData{
    std::array<int, 4> batchShape;
    cv::Mat images;
    std::vector<fs::path> filePaths;
    
    ImageBatchData(int batchSize, int height, int width, int channels):
    batchShape{batchSize, channels, height, width},
    images(batchShape.size(), batchShape.data(), CV_16F, cv::Scalar(0)){
        filePaths.reserve(batchSize);
    }
};
